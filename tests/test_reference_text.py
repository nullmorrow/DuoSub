"""Export-only reference reuse, three-way edits, provenance and ASS boundaries."""
from __future__ import annotations

import copy
import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_stage0 import CN_ASS, JP_SRT, _write
from subkit.ass.document import AssDocument
from subkit.ass.text import VisibleText, plain_text
from subkit.core.display import DisplayRules
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.importer import import_source
from subkit.core.merge import MergeSettings, merge_episode
from subkit.core.ops import Editor
from subkit.core.reference_text import merge_reference, replace_visible
from subkit.model.serialize import document_to_json
from subkit.proc import CancelledError
from subkit.styles_default import default_styles_ass


class ThreeWayReference(unittest.TestCase):
    def setUp(self):
        self.to_s = DisplayRules(zh_script="simplified").convert
        self.to_t = DisplayRules(zh_script="traditional").convert

    def test_unchanged_reuses_counterpart_glyphs(self):
        result, strategy, _ = merge_reference("這是髙傲的戰士", "這是髙傲的戰士", "这是高傲的战士", self.to_s)
        self.assertEqual((result, strategy), ("这是高傲的战士", "reference"))

    def test_local_edit_preserves_original_translation_choice(self):
        result, strategy, _ = merge_reference("這套軟體很好用", "這套軟體很難用", "这套软件很好用", self.to_s)
        self.assertEqual((result, strategy), ("这套软件很难用", "merged"))

    def test_insertion_and_deletion(self):
        for current, expected in (("這套軟體很不好用", "这套软件很不好用"),
                                  ("這套軟體很用", "这套软件很用"),
                                  ("啊這套軟體很好用！", "啊这套软件很好用！")):
            with self.subTest(current=current):
                value, strategy, _ = merge_reference("這套軟體很好用", current, "这套软件很好用", self.to_s)
                self.assertEqual((value, strategy), (expected, "merged"))

    def test_reverse_direction(self):
        value, strategy, _ = merge_reference("这套软件很好用", "这套软件很难用", "這套軟體很好用", self.to_t)
        self.assertEqual((value, strategy), ("這套軟體很難用", "merged"))

    def test_edit_over_different_reference_translation_falls_back(self):
        value, strategy, reason = merge_reference("這套軟體很好用", "這套韌體很好用", "这套软件很好用", self.to_s)
        self.assertEqual((value, strategy), ("这套韧体很好用", "opencc"))
        self.assertIn("无法可靠", reason)

    def test_whole_rewrite_is_current_text(self):
        value, strategy, _ = merge_reference("過去版本", "完全重譯", "过去版本", self.to_s)
        self.assertEqual((value, strategy), ("完全重译", "opencc"))

    def test_repeated_glyphs_with_different_reference_spellings_fall_back(self):
        value, strategy, _ = merge_reference("發發", "發", "发髮", self.to_s)
        self.assertEqual((value, strategy), ("发", "opencc"))

    def test_ass_tags_and_breaks_are_kept(self):
        original = r"{\fad(0,720)}這是{\b1}髙傲{\b0}\N戰士"
        value, bounds = replace_visible(original, "这是高傲战士")
        self.assertEqual(value, r"{\fad(0,720)}这是{\b1}高傲{\b0}\N战士")
        self.assertEqual(bounds, list(range(7)))

    def test_length_change_within_one_token_maps_boundaries(self):
        value, bounds = replace_visible(r"{\fad(0,720)}軟體很好", "软件非常好")
        self.assertEqual(VisibleText(value).text, "软件非常好")
        self.assertEqual(len(bounds), 5)
        self.assertEqual(bounds[-1], 5)
        self.assertEqual(bounds, sorted(bounds))

    def test_length_change_cannot_cross_karaoke_or_format_boundary(self):
        with self.assertRaises(ValueError):
            replace_visible(r"軟{\kf30}體", "软件包")

    def test_reference_cannot_change_raw_or_inject_ass(self):
        for text, target in (("原文", r"坏{\pos(1,2)}字"), ("原文", "字\\N字"),
                             (r"{@Raw=軟體}好", "软件好"), (r"{\p1}m 0 0 l 1 1{\p0}字", "词")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                replace_visible(text, target)


class ReferenceExports(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="duosub-reference-text-"))
        self.orig = self.tmp / "originals"
        self.orig.mkdir()
        self.cn_path = _write(self.orig, "test.cht.ass", CN_ASS.replace("你好啊", "這套軟體很好用"))
        self.jp_path = _write(self.orig, "test.ja.srt", JP_SRT)
        self.ep_dir = self.tmp / "episode"
        self.cn = import_source(self.cn_path, src_id="cn", lang="zh-Hant", lane="translation",
                                episode_dir=self.ep_dir, cn_cfg={"dialogue": ["Default"]})
        self.ja = import_source(self.jp_path, src_id="jp", lang="ja", lane="original", episode_dir=self.ep_dir)
        self.sources = {"cn": self.cn, "jp": self.ja}
        self.doc = merge_episode("E1", self.cn, self.ja, MergeSettings()).document
        self.m = next(m for m in self.doc.members.values() if "這套" in m.text)
        self.template = AssDocument.parse(default_styles_ass())
        self.rules = DisplayRules(zh_script="simplified")
        self.selection = SimpleNamespace(target_script="zh-Hans", baseline=None,
            target=SimpleNamespace(path=str(self.orig / "test.chs.ass"), sha256="reference-hash",
                events=[SimpleNamespace(i=0, text="这套软件很好用")]),
            target_matches={0: 0}, baseline_matches={}, issues=[], protected_input_paths=[])

    def tearDown(self):
        for p in self.tmp.rglob("*"):
            if p.is_file():
                p.chmod(0o666)
        shutil.rmtree(self.tmp)

    def export(self, *, enabled=True, **kw):
        opts = ExportOptions(ruby=False, display=self.rules, reference_enabled=enabled)
        with patch("subkit.core.reference_text.resolve_reference", return_value=self.selection):
            return build_ass(self.doc, self.sources, self.template, self.ep_dir, opts, **kw)

    def row(self, report):
        return next(row for row in report.reference["rows"] if row["member"] == self.m.id)

    def original_baseline(self, text="這套軟體很好用"):
        self.selection.baseline = SimpleNamespace(path=str(self.orig / "original.cht.ass"), sha256="base-hash",
                                                 events=[SimpleNamespace(i=0, text=text)])
        self.selection.baseline_matches = {0: 0}

    def test_default_path_has_no_reference_reads_or_payload(self):
        with patch("subkit.core.reference_text.resolve_reference", side_effect=AssertionError("must be opt-in")):
            out, report = self.export(enabled=False)
        self.assertIsNone(report.reference)
        self.assertIn("这套软体很好用", out.events[0].text)

    def test_current_timing_style_and_tags_survive_direct_reference(self):
        self.original_baseline()
        self.m.start, self.m.end = 9876, 11234
        self.m.overrides = {"raw": r"\fad(0,720)", "an": 8, "pos": [800, 55]}
        before = copy.deepcopy(document_to_json(self.doc))
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.orig.iterdir()}
        out, report = self.export()
        event = next(e for e in out.events if "这套软件" in e.text)
        self.assertEqual((event.start, event.end, event.style), (988, 1123, self.m.style))
        self.assertIn(r"\fad(0,720)", event.text)
        self.assertIn(r"\pos(800,55)", event.text)
        row = self.row(report)
        self.assertEqual(row["strategy"], "reference")
        self.assertFalse(row["changed"])
        self.assertEqual(row["final"], plain_text(event.text, breaks="\n"))
        self.assertEqual(document_to_json(self.doc), before)
        self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.orig.iterdir()})

    def test_manual_edit_migrates_against_true_baseline(self):
        self.original_baseline()
        Editor(self.doc, self.sources).replace_text(self.m.id, "這套軟體很難用")
        out, rep = self.export()
        self.assertIn("这套软件很难用", self.row(rep)["final"])
        self.assertEqual(self.row(rep)["strategy"], "merged")
        self.assertTrue(self.row(rep)["changed"])

    def test_imported_pre_edit_is_compared_to_true_original(self):
        # This source was already edited before DuoSub imported it.
        self.original_baseline("這套軟體很有用")
        self.selection.target.events[0].text = "这套软件很有用"
        _out, rep = self.export()
        row = self.row(rep)
        self.assertEqual((row["generated"], row["strategy"]), ("这套软件很好用", "merged"))
        self.assertFalse(row["changed"])
        self.assertTrue(row["preimport_changed"])

    def test_missing_original_does_not_undo_possible_pre_import_edit(self):
        self.selection.target.events[0].text = "这套软件好用"
        _out, rep = self.export()
        row = self.row(rep)
        self.assertEqual((row["generated"], row["strategy"]), ("这套软体很好用", "opencc"))
        self.assertIn("缺少原版", row["reason"])

    def test_no_reference_automatically_falls_back_and_reports(self):
        self.selection.target_matches = {}
        _out, rep = self.export()
        self.assertEqual(self.row(rep)["strategy"], "opencc")
        self.assertEqual(self.row(rep)["final"], "这套软体很好用")

    def test_existing_pinyin_cannot_be_attached_to_different_reference_words(self):
        from subkit.core.pinyin_notes import annotate_pinyin
        self.original_baseline()
        annotate_pinyin(self.doc, self.m)
        options = ExportOptions(ruby=False, pinyin_mode="PY", display=self.rules, reference_enabled=True)
        with patch("subkit.core.reference_text.resolve_reference", return_value=self.selection):
            _out, rep = build_ass(self.doc, self.sources, self.template, self.ep_dir, options,
                                  measure_cache=self.tmp / "pinyin.sqlite3")
        self.assertEqual(self.row(rep)["final"], "这套软体很好用")
        self.assertEqual(self.row(rep)["strategy"], "opencc")
        self.assertIn("注音", self.row(rep)["reason"])

    def test_full_manual_rewrite_uses_tombstones_and_keeps_new_translation(self):
        self.original_baseline()
        Editor(self.doc, self.sources).replace_text(self.m.id, "完全重譯")
        _out, rep = self.export()
        self.assertEqual(self.row(rep)["source_event"], 0)
        self.assertEqual(self.row(rep)["final"], "完全重译")
        self.assertEqual(self.row(rep)["strategy"], "opencc")

    def test_split_member_does_not_duplicate_reference_sentence(self):
        from subkit.model.document import Member
        self.original_baseline()
        first, last = self.m.runs[0].slice(0, 3), self.m.runs[0].slice(3, len(self.m.runs[0].text))
        self.m.runs = [first]
        m2 = copy.deepcopy(self.m)
        m2.id, m2.runs = "split-second", [last]
        self.doc.members[m2.id] = m2
        self.doc.group(self.m.group).members.append(m2.id)
        _out, rep = self.export()
        rows = [row for row in rep.reference["rows"] if row["member"] in (self.m.id, m2.id)]
        self.assertTrue(all(row["strategy"] == "opencc" for row in rows))
        self.assertEqual("".join(row["final"] for row in rows), "这套软体很好用")

    def test_reference_change_invalidates_layout_even_with_export_cache(self):
        self.original_baseline()
        cache = ExportCache()
        _out, rep = self.export(cache=cache)
        self.assertEqual(self.row(rep)["final"], "这套软件很好用")
        self.selection.target.events[0].text = "这套软体很好用"
        _out, rep = self.export(cache=cache)
        self.assertEqual(self.row(rep)["final"], "这套软体很好用")

    def test_conversion_scope_and_off_direction_are_respected(self):
        for rules in (DisplayRules(), DisplayRules(zh_script="simplified", zh_scope={"dialogue": False})):
            self.rules = rules
            _out, rep = self.export()
            self.assertEqual(self.row(rep)["strategy"], "unchanged")
            self.assertEqual(self.row(rep)["final"], "這套軟體很好用")

    def test_cancellation_propagates_before_any_output(self):
        runner = SimpleNamespace(check_cancelled=lambda: (_ for _ in ()).throw(CancelledError()))
        with self.assertRaises(CancelledError):
            self.export(runner=runner)
        self.assertFalse((self.ep_dir / "out.ass").exists())

    def test_new_project_import_merge_and_episode_export_api(self):
        from opencc import OpenCC
        from subkit.core.workspace import Project
        native = self.tmp / "native"
        native.mkdir()
        shutil.copyfile(self.cn_path, native / self.cn_path.name)
        reference = native / "test.chs.ass"
        reference.write_text(OpenCC("t2s").convert(self.cn_path.read_text(encoding="utf-8-sig"))
                             .replace("这套软体", "这套软件"), encoding="utf-8-sig")
        project = Project.create(self.tmp / "new-project", "默认新工程验收")
        ep = project.add_episode("S01E001")
        ep.add_source(self.cn_path, lang="zh-Hant", lane="translation", cn_cfg={"dialogue": ["Default"]})
        ep.add_source(self.jp_path, lang="ja", lane="original")
        ep.merge()
        saved = copy.deepcopy(document_to_json(ep.document))
        path, report = ep.export(self.tmp / "new-release.ass", zh_direction="simplified",
                                reference_enabled=True, reference_path=reference)
        self.assertTrue(path.is_file())
        self.assertTrue(path.with_suffix(".reference.html").is_file())
        self.assertIn("这套软件很好用", path.read_text(encoding="utf-8-sig"))
        self.assertIn("reference", [row["strategy"] for row in report.reference["rows"]])
        self.assertEqual(document_to_json(ep.document), saved)

        # Normal report failures retain the ASS; cancellation retains neither.
        with patch("subkit.core.reference_report.render_reference_report", side_effect=RuntimeError("HTML failed")):
            failed, failed_report = ep.export(self.tmp / "html-failed.ass", zh_direction="simplified",
                                             reference_enabled=True, reference_path=reference)
        self.assertTrue(failed.is_file())
        self.assertFalse(failed.with_suffix(".reference.html").exists())
        self.assertTrue(any("HTML failed" in warning for warning in failed_report.warnings))
        with patch("subkit.core.reference_report.render_reference_report", side_effect=CancelledError()):
            with self.assertRaises(CancelledError):
                ep.export(self.tmp / "cancelled.ass", zh_direction="simplified",
                          reference_enabled=True, reference_path=reference)
        self.assertFalse((self.tmp / "cancelled.ass").exists())
        self.assertEqual(document_to_json(ep.document), saved)


if __name__ == "__main__":
    unittest.main()
