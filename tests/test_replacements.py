"""Literal Project display replacements, source safety, ruby, karaoke and review."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.test_halfwidth_kana import fixture
from subkit.ass.text import VisibleText, plain_text
from subkit.core import karaoke as kara
from subkit.core.display import DisplayRules, PunctuationRules, member_display_text
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.layout_index import LayoutIndex
from subkit.core.ops import Editor
from subkit.core.readings import anchor_of, visible_map
from subkit.core.replacements import (ReplacementRule, apply_replacements, frozen_rules, load_rules,
                                      normalize_rules, replace_ass_text, save_rules)
from subkit.core.review import group_record
from subkit.core.review_import import Answer, apply, match
from subkit.core.workspace import Project
from subkit.karaoke_fx import get_effect
from subkit.model.document import Annotation


def rule(source="余計", target="", lane="original", enabled=True):
    return dict(source=source, target=target, lane=lane, enabled=enabled)


class LiteralReplacements(unittest.TestCase):
    def test_empty_b_deletes_all_occurrences(self):
        self.assertEqual(apply_replacements("余計国余計", [rule()], lane="original"), "国")

    def test_names_literal_case_and_spaces(self):
        rows = [rule("Alice", "艾丽丝", "translation"), rule(" .?[] ", "・", "both")]
        self.assertEqual(apply_replacements("Alice alice .?[] ", rows, lane="translation"), "艾丽丝 alice・")
        self.assertEqual(normalize_rules([rule(" A ", " B ")])[0]["source"], " A ")

    def test_non_cascading_and_priority(self):
        rows = [rule("A", "B"), rule("B", "C")]
        self.assertEqual(apply_replacements("AB", rows, lane="original"), "BC")
        self.assertEqual(apply_replacements("ABC", [rule("AB", "X"), rule("ABC", "Y")], lane="original"), "XC")
        self.assertEqual(apply_replacements("ABC", [rule("ABC", "Y"), rule("AB", "X")], lane="original"), "Y")

    def test_lanes_disabled_and_identical(self):
        rows = [rule("A", "B", "original"), rule("A", "C", "translation"), rule("Z", "", "both", False)]
        self.assertEqual(apply_replacements("AZ", rows, lane="original"), "BZ")
        self.assertEqual(apply_replacements("AZ", rows, lane="translation"), "CZ")
        self.assertEqual(replace_ass_text("A", [rule("A", "A")], lane="original").matches, 0)

    def test_invalid_source_scope_or_ass_structure(self):
        for row in (rule(""), rule("A", "{bad}"), rule("A", r"\fad(0,1)"), rule("A", "x\ny"),
                    rule("A", lane="ja"), dict(source="A", target=None), dict(source="A", enabled="yes")):
            with self.subTest(row=row), self.assertRaises(ValueError):
                normalize_rules([row])
            self.assertEqual(normalize_rules([row], strict=False), [])

    def test_cross_style_tags_preserved_and_source_not_reparsed(self):
        text = r"{\fad(0,720)}ベ{\b1}ート{\b0}国"
        result = replace_ass_text(text, [rule("ベート", "貝特")], lane="original")
        self.assertEqual(VisibleText(result.text).text, "貝特国")
        self.assertIn(r"\fad(0,720)", result.text)
        self.assertIn(r"\b1", result.text)
        self.assertEqual(result.bounds[-1], 3)
        self.assertEqual(result.ranges, [(0, 3)])

    def test_raw_drawing_and_breaks_are_boundaries(self):
        for text in (r"A\NB", r"A{@Raw=X}B", r"A{\p1}m 0 0 l 2 2{\p0}B"):
            self.assertEqual(replace_ass_text(text, [rule("AB", "X")], lane="original").text, text)
        self.assertEqual(replace_ass_text(r"{@Raw=余計}余計", [rule()], lane="original").text, r"{@Raw=余計}")

    def test_expansion_deletion_boundary_maps(self):
        result = replace_ass_text("余計A国", [rule(), rule("A", "ABCDE")], lane="original")
        self.assertEqual(result.text, "ABCDE国")
        self.assertEqual(result.bounds, [0, 0, 0, 5, 6])
        self.assertEqual(result.deleted, {0, 1})
        self.assertEqual(result.matches, 2)

    def test_escapes_and_unicode_are_visible(self):
        result = replace_ass_text(r"田\h中\n😀", [rule("田 中", "山田"), rule("😀", "龍")], lane="original")
        self.assertEqual(VisibleText(result.text).text, "山田 龍")


class ReplacementPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub-replace-store-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.project = Project.create(self.path / "one", "一")

    def test_project_wide_isolation_reopen_and_no_episode_override(self):
        save_rules(self.project, [rule("艾絲", "艾丝", "translation")])
        ep = self.project.add_episode("E1")
        ep.data["settings"] = {"display": {"replacements": [rule("wrong", "bad")]}}
        self.project.data["seasons"][0]["settings"] = {"display": {"replacements": []}}
        self.assertEqual(DisplayRules.from_settings(ep.settings()).replacements[0].source, "艾絲")
        reopened = Project.open(self.project.root)
        self.assertEqual(load_rules(reopened), load_rules(self.project))
        other = Project.create(self.path / "two", "二")
        self.assertEqual(load_rules(other), [])
        draft = load_rules(self.project)
        draft[0]["target"] = "mutated"
        self.assertEqual(load_rules(self.project)[0]["target"], "艾丝")

    def test_failed_save_restores_memory_and_disk(self):
        saved = copy.deepcopy(self.project.data)
        original = (self.project.root / "subkit-project.json").read_bytes()
        with patch.object(self.project, "save", side_effect=OSError("blocked")):
            with self.assertRaises(OSError):
                save_rules(self.project, [rule()])
        self.assertEqual(self.project.data, saved)
        self.assertEqual((self.project.root / "subkit-project.json").read_bytes(), original)

    def test_invalid_and_identical_save_do_not_write(self):
        with patch.object(self.project, "save") as save:
            self.assertFalse(save_rules(self.project, []))
            with self.assertRaises(ValueError):
                save_rules(self.project, [rule("")])
        save.assert_not_called()


class ReplacementExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub-replace-export-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def build(self, doc, template, rows=(), *, display=None, **kw):
        rules = display or DisplayRules(replacements=frozen_rules(rows))
        return build_ass(doc, {}, template, self.path, ExportOptions(display=rules, **kw),
                         measure_cache=self.path / "render.sqlite3")

    def annotate(self, doc, a, b, text, aid="ann"):
        member = doc.members["m_kana"]
        ids = visible_map(member)[1]
        doc.annotations[aid] = Annotation(aid, "ruby", member.id, anchor_of(ids[a:b]), member.text[a:b], text,
                                         status="confirmed", parts=[[0, b-a, text]])

    def test_preview_release_ruby_delete_and_identity_hits_match_reference(self):
        doc, template = fixture("余国")
        self.annotate(doc, 0, 1, "よ", "extra")
        self.annotate(doc, 1, 2, "くに", "country")
        before = copy.deepcopy(doc)
        ass, rep = self.build(doc, template, [rule("余")])
        self.assertEqual(VisibleText(ass.events[0].text).text, "国")
        self.assertEqual(rep.ruby_lines, 1)
        self.assertEqual([o["ann"] for o in rep.owners if o["kind"] == "ruby"], ["country"])
        self.assertEqual(rep.owners[0]["ids"], [visible_map(doc.members["m_kana"])[1][1]])
        preview, prep = self.build(doc, template, [rule("余")], mode="preview")
        self.assertEqual(preview.events, ass.events)
        self.assertEqual(prep.owners, rep.owners)
        self.assertEqual((rep.replacement_lines, rep.replacement_matches, rep.replacement_deleted), (1, 1, 1))
        index = LayoutIndex(ass, rep.owners, measure_cache=self.path / "render.sqlite3")
        try:
            base = next(g for g in index.geometry(index.events_of_group("g_kana")) if g.kind == "base")
            box = base.chars[0]
            self.assertEqual(index.hit([base], box.cx, box.cy, pad=0).identity, rep.owners[0]["ids"][0])
        finally:
            index.close()
        self.assertEqual(doc, before)

    def test_all_deleted_has_no_ruby_or_ghost_glyph(self):
        doc, template = fixture("国")
        self.annotate(doc, 0, 1, "くに")
        ass, rep = self.build(doc, template, [rule("国")])
        self.assertEqual(ass.events, [])
        self.assertEqual(rep.ruby_lines, 0)
        self.assertEqual(rep.owners, [])
        self.assertEqual(rep.empty_events_removed, 1)

    def test_part_deleted_annotation_is_hidden_and_preserved(self):
        doc, template = fixture("日本")
        self.annotate(doc, 0, 2, "にほん")
        saved = copy.deepcopy(doc.annotations)
        ass, rep = self.build(doc, template, [rule("日")])
        self.assertEqual(VisibleText(ass.events[0].text).text, "本")
        self.assertEqual(rep.ruby_lines, 0)
        self.assertEqual(doc.annotations, saved)

    def test_expanded_name_keeps_existing_word_reading_and_original_editor(self):
        doc, template = fixture("国")
        self.annotate(doc, 0, 1, "くに")
        original = doc.members["m_kana"].text
        ass, rep = self.build(doc, template, [rule("国", "國名")])
        self.assertEqual(VisibleText(ass.events[0].text).text, "國名")
        self.assertEqual(rep.ruby_lines, 1)
        self.assertEqual(doc.members["m_kana"].text, original)
        self.assertEqual(len(set(rep.owners[0]["ids"])), 1)

    def test_replacements_then_conversion_punctuation_and_merge(self):
        doc, template = fixture(r"余国！\N山")
        rules = DisplayRules(replacements=frozen_rules([rule("余"), rule("国", "國名")]),
            punctuation=(("original", PunctuationRules(exclaim="half")),), merge_lines=(("original", "dialogue"),))
        ass, _rep = self.build(doc, template, display=rules)
        self.assertEqual(VisibleText(ass.events[0].text).text, "國名!  山")
        member = doc.members["m_kana"]
        member.lang, member.lane = "zh-Hant", "translation"
        rules = DisplayRules(zh_script="simplified", replacements=frozen_rules([rule("余", lane="translation")]))
        ass, _rep = self.build(doc, template, display=rules)
        self.assertEqual(plain_text(ass.events[0].text, breaks="\n"), "国！\n山")

    def test_cache_invalidation_enable_disable_reorder(self):
        doc, template = fixture("国")
        cache = ExportCache()
        saved = copy.deepcopy(doc)
        def build(rows):
            return build_ass(doc, {}, template, self.path,
                ExportOptions(ruby=False, display=DisplayRules(replacements=frozen_rules(rows))), cache=cache)
        base, _ = build([])
        one, rep = build([rule("国", "A"), rule("国", "B")])
        self.assertEqual(VisibleText(one.events[0].text).text, "A")
        self.assertEqual(rep.laid_out, 1)
        _same, hit = build([rule("国", "A"), rule("国", "B")])
        self.assertEqual(hit.laid_out, 0)
        reordered, rep = build([rule("国", "B"), rule("国", "A")])
        self.assertEqual(VisibleText(reordered.events[0].text).text, "B")
        restored, _ = build([])
        self.assertEqual(restored.serialize(), base.serialize())
        self.assertEqual(doc, saved)

    def test_deleted_karaoke_beat_never_sings_next_glyph_early(self):
        doc, template = fixture("あ国", song=True)
        m = doc.members["m_kana"]
        beats = kara.derive(doc, m, "mora")
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1000+i*300, 1300+i*300, "manual"
        doc.karaoke["k"] = kara.beats_to_track(m, beats, unit="mora", effect={"preset": "kf"})
        before = copy.deepcopy(doc)
        ass, _rep = self.build(doc, template, [rule("あ")])
        cursor = m.start
        times = []
        for text, duration in kara.parse_k_runs(ass.events[0].text):
            if text:
                times.append((text, cursor))
            cursor += duration*10
        self.assertEqual(times[0], ("国", beats[1].t0))
        self.assertEqual(doc, before)

    def test_deleted_advanced_fx_never_creates_removed_glyph(self):
        doc, template = fixture("あ国", song=True)
        m = doc.members["m_kana"]
        beats = kara.derive(doc, m, "mora")
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1000+i*200, 1200+i*200, "manual"
        doc.karaoke["k"] = kara.beats_to_track(m, beats, unit="mora", effect={"preset": "tremble"})
        contexts = []
        effect = get_effect("tremble")
        generate = effect.generate
        with patch.object(effect, "generate", side_effect=lambda ctx: (contexts.append(ctx), generate(ctx))[1]):
            self.build(doc, template, [rule("あ")])
        self.assertEqual({g.text for g in contexts[0].glyphs}, {"国"})
        self.assertEqual(contexts[0].glyphs[0].beats[0][1], beats[1].t0)

    def test_review_deletion_has_no_ghost_reading_and_changed_spans_cannot_write_back(self):
        doc, _template = fixture("余国")
        self.annotate(doc, 0, 1, "よ", "extra")
        self.annotate(doc, 1, 2, "くに", "country")
        rules = DisplayRules(replacements=frozen_rules([rule("余")]))
        record = group_record(doc, doc.groups[0], 1, display=rules)["original"][0]
        self.assertEqual((record["text"], record["annotated"]), ("国", "国(くに)"))
        rules = DisplayRules(replacements=frozen_rules([rule("余", "山")]))
        result = match(doc, {}, rules, Answer(group="g_kana", lang="ja", kind="text", target="山", old="山", new="川", reason="check"))
        self.assertEqual(result.status, "ambiguous")
        self.assertFalse(apply(Editor(doc, {}), result))

    def test_member_display_does_not_replace_name_or_gloss_metadata(self):
        from subkit.core.display import display_name
        doc, _template = fixture(r"余{@note}国")
        m = doc.members["m_kana"]
        m.speaker = {"label": "余"}
        rules = DisplayRules(replacements=frozen_rules([rule("余", "新")]))
        self.assertEqual(VisibleText(member_display_text(m, rules)[0]).text, "新国")
        self.assertEqual(display_name(m, rules), "余")

    def test_review_partial_piece_deletion_lists_only_visible_reading(self):
        doc, _template = fixture("日本")
        self.annotate(doc, 0, 2, "にほん")
        doc.annotations["ann"].parts = [[0, 1, "に"], [1, 2, "ほん"]]
        rules = DisplayRules(replacements=frozen_rules([rule("日")]))
        item = group_record(doc, doc.groups[0], 1, display=rules)["original"][0]
        self.assertEqual(item["annotated"], "本(ほん)")
        self.assertEqual(item["readings"][0]["reading"], "ほん")

    def test_pinyin_deleted_base_cannot_leave_pinyin(self):
        from subkit.core.pinyin_notes import annotate_pinyin
        doc, template = fixture("余国")
        m = doc.members["m_kana"]
        m.lang, m.lane, m.style = "zh-Hans", "translation", "CN"
        annotate_pinyin(doc, m)
        before = copy.deepcopy(doc)
        ass, rep = self.build(doc, template, [rule("余", lane="translation")], pinyin_mode="PY")
        self.assertEqual(VisibleText(ass.events[0].text).text, "国")
        self.assertEqual(rep.pinyin_lines, 1)
        self.assertEqual(doc, before)

    def test_reference_cannot_restore_batch_deleted_or_renamed_text(self):
        from tests.test_reference_text import ReferenceExports
        fixture = ReferenceExports("test_default_path_has_no_reference_reads_or_payload")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.original_baseline()
        for target, expected in (("硬體", "这套硬体很好用"), ("", "这套很好用")):
            fixture.rules = DisplayRules(zh_script="simplified",
                replacements=frozen_rules([rule("軟體", target, "translation")]))
            _ass, report = fixture.export()
            row = fixture.row(report)
            self.assertEqual((row["final"], row["strategy"]), (expected, "replacement"))
            self.assertEqual(row["current"], "這套軟體很好用")
            self.assertFalse(row["changed"])

    def test_obsolete_replacement_settings_preview_is_discarded(self):
        from subkit.core.export import ExportReport
        from subkit_app.session import EpisodeSession
        rows = [rule("国", "國")]
        fake = SimpleNamespace(sender=lambda: None, _preview_job=None, _preview_running=True, _closed=False,
            busy=Mock(), episode=SimpleNamespace(settings=lambda: {"display": {"replacements": rows}}),
            _applied_revision=-1, previewReady=Mock(), _preview_wanted=False,
            layout_current=lambda _rev: True, _start_preview=Mock())
        EpisodeSession._preview_done(fake, 0, "unused.ass", None, ExportReport(replacement_key=()))
        fake.previewReady.emit.assert_not_called()
        fake._start_preview.assert_called_once()

    def test_new_ruby_selection_does_not_guess_replaced_source_span(self):
        from PySide6.QtCore import QPoint
        from subkit_app.watch import WatchController
        doc, template = fixture("国")
        ass, report = self.build(doc, template, [rule("国", "國名")])
        owner = report.owners[0]
        fake = SimpleNamespace(_doc=lambda: doc, owners=report.owners, readingRequested=Mock(),
                               newReadingRequested=Mock(), note=Mock())
        geom = SimpleNamespace(index=0, member="m_kana", text="國名", ids=owner["ids"])
        WatchController._reading_for_identity(fake, geom, 0, owner["ids"][0], QPoint())
        fake.newReadingRequested.emit.assert_not_called()
        fake.note.emit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
