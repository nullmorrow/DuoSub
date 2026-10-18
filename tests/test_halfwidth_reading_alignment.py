"""Half-width literal anchors and read-only repair of old automatic ruby parts."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from subkit.ass.document import AssDocument, Event
from subkit.ass.text import VisibleText, tokenize
from subkit.core import karaoke as kara
from subkit.core.display import DisplayRules
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.readings import anchor_of, display_annotations, pieces_of, ruby_parts, visible_map
from subkit.core.review import annotated, group_record
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.model.source import Source, SourceEvent
from subkit.reading.kana_align import align
from subkit.styles_default import default_styles_ass
from subkit_app.views import group_editor


def fixture(surface="ﾃﾛ行為", reading="てろこうい", *, song=False, status="auto", origin="analyzer"):
    member = Member("jp", "g", "ja", "original", "song" if song else "dialogue", 1000, 5000,
                    style="OP.JP" if song else "JP", runs=[Run(surface, ins="i_jp", b=len(surface))])
    doc = Document("halfwidth-reading", groups=[Group("g", "song" if song else "dialogue", [member.id])],
                   members={member.id: member})
    ids = visible_map(member)[1]
    annotation = Annotation("a", "ruby", member.id, anchor_of(ids), surface, reading, status=status,
                            origin=origin, parts=[[2, 4, reading]])
    doc.annotations[annotation.id] = annotation
    template = AssDocument.parse(default_styles_ass())
    for style in template.styles:
        style.set("Fontname", "MS Gothic")
    return doc, member, annotation, template


def width(enabled=True):
    return DisplayRules.from_settings({"display": {"punctuation": {"original": {
        "halfwidth_kana": enabled, "scope_chars": ["dialogue", "song", "other"]}}}})


def ruby_texts(ass, report):
    return [VisibleText(event.text).text for event, owner in zip(ass.events, report.owners)
            if owner.get("kind") == "ruby"]


class HalfwidthLiteralAlignment(unittest.TestCase):
    def test_halfwidth_prefix_suffix_and_both_sides_keep_kanji_offsets(self):
        for surface, reading, expected in (
                ("ﾃﾛ行為", "てろこうい", [(2, 4, "こうい")]),
                ("行為ﾃﾛ", "こういてろ", [(0, 2, "こうい")]),
                ("ﾃﾛ行為ｶﾞ", "てろこういが", [(2, 4, "こうい")]),
                ("取ﾘ扱い", "とりあつかい", [(0, 1, "と"), (2, 3, "あつか")])):
            with self.subTest(surface=surface):
                self.assertEqual(align(surface, reading), expected)
                self.assertEqual(ruby_parts(surface, reading), [list(part) for part in expected])

    def test_voiced_semi_voiced_and_mixed_width_literals_keep_original_offsets(self):
        for surface, reading, expected in (
                ("ｶﾞ国", "がくに", [(2, 3, "くに")]),
                ("ﾊﾟ国", "ぱくに", [(2, 3, "くに")]),
                ("カﾞ国", "がくに", [(2, 3, "くに")]),
                ("ｶﾞ国カﾞ", "がくにが", [(2, 3, "くに")]),
                ("ｻﾞﾌﾞｰﾝ国", "ざぶーんくに", [(6, 7, "くに")])):
            with self.subTest(surface=surface):
                self.assertEqual(align(surface, reading), expected)

    def test_halfwidth_long_vowel_anchor_accepts_long_mark_or_vowel(self):
        for reading in ("てろーこうい", "てろうこうい"):
            with self.subTest(reading=reading):
                self.assertEqual(align("ﾃﾛｰ行為", reading), [(3, 5, "こうい")])

    def test_reading_is_original_katakana_slice(self):
        self.assertEqual(align("ﾃﾛ行為", "テロコウイ"), [(2, 4, "コウイ")])
        self.assertEqual(align("ｶﾞ国カﾞ", "ガクニガ"), [(2, 3, "クニ")])
        self.assertEqual(ruby_parts("ﾃﾛ行為", "テロコウイ"), [[2, 4, "コウイ"]])

    def test_pure_kana_empty_and_literal_mismatch_are_conservative(self):
        self.assertEqual(align("ﾃﾛ", "てろ"), [])
        self.assertEqual(ruby_parts("ﾃﾛ", "てろ"), [])
        self.assertIsNone(align("", "てろ"))
        self.assertIsNone(align("ﾃﾛ行為", ""))
        self.assertIsNone(align("ﾃﾛ行為", "べつのよみ"))
        self.assertEqual(ruby_parts("ﾃﾛ行為", "べつのよみ"), [[2, 4, "べつのよみ"]])

    def test_jukujikun_fallback_and_fullwidth_existing_behavior_remain(self):
        self.assertIsNone(align("お祖母さん", "ばあさん"))
        self.assertEqual(ruby_parts("お祖母さん", "ばあさん"), [[1, 3, "ばあ"]])
        self.assertIsNone(align("ｵ祖母さん", "ばあさん"))
        self.assertEqual(ruby_parts("ｵ祖母さん", "ばあさん"), [[1, 3, "ばあ"]])
        self.assertEqual(align("行った", "いった"), [(0, 1, "い")])
        self.assertEqual(align("解錠薬", "ステイタスシーフ"), [(0, 3, "ステイタスシーフ")])

    def test_normalization_is_limited_to_halfwidth_kana_literal_anchors(self):
        self.assertEqual(align("①ﾃﾛ国", "①てろくに"), [(3, 4, "くに")])
        self.assertIsNone(align("①ﾃﾛ国", "1てろくに"))
        self.assertIsNone(align("ﾃﾛ国", "てろゃ"))


class OldAutomaticAnnotationViews(unittest.TestCase):
    def test_old_auto_and_uncertain_parts_are_derived_without_mutating_storage(self):
        for status in ("auto", "uncertain"):
            with self.subTest(status=status):
                doc, member, ann, _tpl = fixture(status=status)
                saved = copy.deepcopy(document_to_json(doc))
                original_parts = copy.deepcopy(pieces_of(ann))
                view = display_annotations(doc, member, [ann])[0]
                self.assertIsNot(view, ann)
                self.assertEqual(view.parts, [[2, 4, "こうい"]])
                self.assertEqual({key for key, value in vars(view).items() if value != vars(ann)[key]}, {"parts"})
                self.assertEqual(pieces_of(ann), original_parts)
                self.assertEqual(document_to_json(doc), saved)

    def test_derived_parts_are_independent_of_stored_parts(self):
        doc, member, ann, _tpl = fixture()
        view = display_annotations(doc, member, [ann])[0]
        view.parts[0][2] = "temporary display only"
        self.assertEqual(ann.parts, [[2, 4, "てろこうい"]])

    def test_manual_confirmed_hidden_and_custom_origins_are_preserved(self):
        cases = [("confirmed", "analyzer"), ("hidden", "analyzer")]
        cases += [("auto", origin) for origin in ("manual", "dict", "rule", "legacy", "exception", "cc_hint", "voice")]
        for status, origin in cases:
            with self.subTest(status=status, origin=origin):
                doc, member, ann, _tpl = fixture(status=status, origin=origin)
                before = copy.deepcopy(document_to_json(doc))
                self.assertIs(display_annotations(doc, member, [ann])[0], ann)
                self.assertEqual(document_to_json(doc), before)

    def test_custom_parts_mono_nonruby_and_empty_parts_are_preserved(self):
        for update in ({"parts": [[2, 3, "こう"], [3, 4, "い"]]},
                       {"parts": [[0, 4, "てろこうい"]]}, {"mono": True}, {"kind": "pinyin"}, {"parts": []}):
            with self.subTest(update=update):
                doc, member, ann, _tpl = fixture()
                for key, value in update.items():
                    setattr(ann, key, value)
                before = copy.deepcopy(ann)
                self.assertIs(display_annotations(doc, member, [ann])[0], ann)
                self.assertEqual(ann, before)

    def test_fullwidth_no_kanji_or_unmatched_reading_are_not_guessed(self):
        for surface, reading, parts in (("テロ行為", "てろこうい", [[2, 4, "てろこうい"]]),
                                        ("ﾃﾛ", "てろ", [[0, 2, "てろ"]]),
                                        ("ﾃﾛ行為", "べつのよみ", [[2, 4, "べつのよみ"]])):
            with self.subTest(surface=surface):
                doc, member, ann, _tpl = fixture(surface, reading)
                ann.parts = parts
                self.assertIs(display_annotations(doc, member, [ann])[0], ann)

    def test_current_source_must_match_base_and_anchor(self):
        for change in ("base", "anchor"):
            with self.subTest(change=change):
                doc, member, ann, _tpl = fixture()
                if change == "base":
                    member.runs[0].text = "ﾃﾛ犯罪"
                else:
                    ann.anchor = [{"ins": "missing", "a": 0, "b": 4}]
                before = copy.deepcopy(document_to_json(doc))
                self.assertIs(display_annotations(doc, member, [ann])[0], ann)
                self.assertEqual(document_to_json(doc), before)

    def test_only_eligible_annotations_change_in_a_mixed_list(self):
        doc, member, ann, _tpl = fixture()
        manual = copy.deepcopy(ann)
        manual.id, manual.origin = "manual", "manual"
        anns = [manual, ann]
        views = display_annotations(doc, member, anns)
        self.assertIs(views[0], manual)
        self.assertIsNot(views[1], ann)
        self.assertEqual(views[1].parts, [[2, 4, "こうい"]])
        self.assertEqual(anns[1].parts, [[2, 4, "てろこうい"]])


class HalfwidthRubyExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub_halfwidth_reading_")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def build(self, doc, template, *, enabled=True, mode="preview", cache=None, sources=None):
        return build_ass(doc, sources or {}, template, self.path,
                         ExportOptions(mode=mode, display=width(enabled), include_actor=False),
                         cache=cache, measure_cache=self.path / "render.sqlite3")

    def test_preview_release_and_width_switch_use_correct_ruby_without_saving(self):
        doc, member, _ann, tpl = fixture()
        before = copy.deepcopy(document_to_json(doc))
        for enabled in (False, True):
            results = []
            for mode in ("preview", "release"):
                with self.subTest(enabled=enabled, mode=mode):
                    ass, report = self.build(doc, tpl, enabled=enabled, mode=mode)
                    base = next(event for event, owner in zip(ass.events, report.owners) if owner["kind"] == "base")
                    self.assertEqual(VisibleText(base.text).text, "テロ行為" if enabled else member.text)
                    self.assertEqual(ruby_texts(ass, report), ["こうい"])
                    self.assertEqual(report.ruby_lines, 1)
                    self.assertEqual(document_to_json(doc), before)
                    results.append([(event.start, event.end, event.style, event.text) for event in ass.events])
            self.assertEqual(results[0], results[1])

    def test_original_katakana_reading_slice_is_kept_in_export(self):
        doc, _member, _ann, tpl = fixture(reading="テロコウイ")
        ass, report = self.build(doc, tpl)
        self.assertEqual(ruby_texts(ass, report), ["コウイ"])

    def test_protected_manual_confirmed_and_hidden_annotations_keep_existing_behavior(self):
        for status, origin in (("auto", "manual"), ("confirmed", "analyzer"), ("hidden", "analyzer")):
            with self.subTest(status=status, origin=origin):
                doc, _member, _ann, tpl = fixture(status=status, origin=origin)
                before = copy.deepcopy(document_to_json(doc))
                ass, report = self.build(doc, tpl)
                self.assertEqual(ruby_texts(ass, report), [] if status == "hidden" else ["てろこうい"])
                self.assertEqual(document_to_json(doc), before)

    def test_old_cached_layout_is_invalidated_by_display_annotation_parts(self):
        doc, _member, _ann, tpl = fixture()
        cache = ExportCache()
        # Simulate an in-memory entry produced before compatibility views were
        # added; no document field or revision changes between these exports.
        with patch("subkit.core.export.display_annotations", side_effect=lambda _doc, _member, anns: anns):
            old, _old_report = self.build(doc, tpl, cache=cache)
        before = copy.deepcopy(document_to_json(doc))
        fixed, report = self.build(doc, tpl, cache=cache)
        self.assertEqual(report.laid_out, 1)
        self.assertEqual(ruby_texts(fixed, report), ["こうい"])
        self.assertNotEqual(old.serialize(), fixed.serialize())
        again, reused = self.build(doc, tpl, cache=cache)
        self.assertEqual(reused.laid_out, 0)
        self.assertEqual(again.serialize(), fixed.serialize())
        self.assertEqual(document_to_json(doc), before)

    def test_width_and_protected_status_cache_switches_do_not_reuse_wrong_ruby(self):
        doc, _member, ann, tpl = fixture()
        cache = ExportCache()
        for enabled, status, expected in ((True, "auto", "こうい"), (False, "auto", "こうい"),
                                           (False, "confirmed", "てろこうい"), (True, "auto", "こうい")):
            with self.subTest(enabled=enabled, status=status):
                ann.status = status
                ass, report = self.build(doc, tpl, enabled=enabled, cache=cache)
                self.assertEqual(report.laid_out, 1)
                self.assertEqual(ruby_texts(ass, report), [expected])
                _again, reused = self.build(doc, tpl, enabled=enabled, cache=cache)
                self.assertEqual(reused.laid_out, 0)

    def test_runs_tags_raw_and_source_file_remain_unchanged(self):
        raw = r"ﾃ{\b1}ﾛ行{\i1}為{@Raw=ﾃﾛ}"
        doc, member, ann, tpl = fixture()
        source_ass = AssDocument.parse(default_styles_ass())
        source_ass.events = [Event(start=100, end=500, style="JP", text=raw)]
        data = source_ass.serialize().encode("utf-8-sig")
        path = self.path / "original.ass"
        path.write_bytes(data)
        source = Source("s_jp", "ja", "original", "ass", str(path), path.name,
                        hashlib.sha256(data).hexdigest(), len(data), "utf-8-sig", "lf",
                        events=[SourceEvent(0, "Dialogue", 1000, 5000, raw, style="JP")])
        member.runs = [Run(token.raw, "tag" if token.kind in ("tags", "comment") else "text",
                           src=source.id, e=0, a=token.start, b=token.end) for token in tokenize(raw)]
        ann.anchor = anchor_of(visible_map(member)[1][:4])
        before, source_before = copy.deepcopy(document_to_json(doc)), copy.deepcopy(source.events_json())
        for mode in ("preview", "release"):
            ass, report = self.build(doc, tpl, mode=mode, sources={source.id: source})
            base = next(event for event, owner in zip(ass.events, report.owners) if owner["kind"] == "base")
            self.assertEqual(VisibleText(base.text).text, "テロ行為ﾃﾛ")
            self.assertEqual(ruby_texts(ass, report), ["こうい"])
            self.assertIn(r"\b1", base.text)
            self.assertIn(r"\i1", base.text)
            self.assertEqual(document_to_json(doc), before)
            self.assertEqual(source.events_json(), source_before)
            self.assertEqual(path.read_bytes(), data)

    def test_real_timed_tracks_and_translation_links_keep_old_parts_and_all_beats(self):
        doc, member, ann, tpl = fixture(song=True)
        beats = kara.derive(doc, member, "mora")
        for i, beat in enumerate(beats):
            beat.t0, beat.t1, beat.by = 1000 + i * 200, 1200 + i * 200, "manual"
        original_track = kara.beats_to_track(member, beats, state="confirmed", unit="mora",
                                             kid="k_jp", effect={"preset": "kf"})
        doc.karaoke[original_track["id"]] = original_track
        cn = Member("cn", "g", "zh-Hant", "translation", "song", 1000, 5000, layer=1, style="OP.CN",
                    runs=[Run("恐怖", ins="i_cn", b=2)])
        doc.members[cn.id] = cn
        doc.groups[0].members.append(cn.id)
        translated = kara.derive(doc, cn, "mora")
        span = beats[-1].t1 - beats[0].t0
        for i, beat in enumerate(translated):
            beat.t0 = beats[0].t0 + span * i // len(translated)
            beat.t1 = beats[0].t0 + span * (i + 1) // len(translated)
            beat.by = "manual"
        translated[0].link = {"start": kara.beat_link(beats[0], "start")}
        translated[-1].link.setdefault("end", kara.beat_link(beats[-1], "end"))
        doc.karaoke["k_cn"] = kara.beats_to_track(cn, translated, state="confirmed", unit="mora", kid="k_cn")
        before = copy.deepcopy(document_to_json(doc))
        derived = copy.deepcopy(kara.derive(doc, member, "mora"))
        views = {mid: copy.deepcopy(kara.effective(doc, doc.members[mid])) for mid in (member.id, cn.id)}
        signatures = {mid: kara.view_signature(view) for mid, view in views.items()}
        self.assertTrue(kara.has_karaoke(doc, member))
        self.assertEqual(views[cn.id].beats[0].link["start"], kara.beat_link(beats[0], "start"))
        self.assertEqual(views[cn.id].beats[-1].link["end"], kara.beat_link(beats[-1], "end"))
        self.assertTrue(kara.linked_times(doc, "g"))
        for enabled in (False, True):
            self.assertIs(display_annotations(doc, member, [ann])[0], ann)
            self.build(doc, tpl, enabled=enabled)
            self.assertEqual(kara.derive(doc, member, "mora"), derived)
            for mid in views:
                current = kara.effective(doc, doc.members[mid])
                self.assertEqual(current, views[mid])
                self.assertEqual(kara.view_signature(current), signatures[mid])
                self.assertFalse(any("reading_changed" in beat.flags for beat in current.beats))
            self.assertEqual(document_to_json(doc), before)
            self.assertEqual(ann.parts, [[2, 4, "てろこうい"]])


class HalfwidthRubyEditorReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_real_rubyline_paint_and_review_share_corrected_piece_readings(self):
        doc, member, ann, _tpl = fixture()
        before = copy.deepcopy(document_to_json(doc))
        line = group_editor.RubyLine()
        self.addCleanup(line.deleteLater)
        line.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        line.resize(600, 80)
        line.rules = width()
        line.set_member(doc, member)
        drawn = []
        original = group_editor.pieces_of

        def pieces(view):
            result = original(view)
            drawn.extend(copy.deepcopy(result))
            return result

        with patch.object(group_editor, "pieces_of", side_effect=pieces):
            self.assertFalse(line.grab().isNull())
        self.assertEqual(drawn, [[2, 4, "こうい"]])
        text, _readings = annotated(doc, member, display=width())
        self.assertEqual(text, "ﾃﾛ行為(こうい)")
        self.assertEqual(re.findall(r"\(([^)]+)\)", text), [piece[2] for piece in drawn])
        packet_row = group_record(doc, doc.groups[0], 1, display=width())["original"][0]
        self.assertEqual(packet_row["annotated"], text)
        self.assertEqual(ann.parts, [[2, 4, "てろこうい"]])
        self.assertEqual(document_to_json(doc), before)

    def test_uncertain_and_hidden_review_markers_keep_their_status(self):
        doc, member, ann, _tpl = fixture(status="uncertain")
        text, readings = annotated(doc, member)
        self.assertEqual(text, "ﾃﾛ行為(こうい?)")
        self.assertEqual(readings[0]["status"], "uncertain")
        ann.status = "hidden"
        text, readings = annotated(doc, member)
        self.assertEqual(text, "ﾃﾛ行為")
        self.assertEqual(readings[0]["status"], "hidden")
        self.assertEqual(ann.parts, [[2, 4, "てろこうい"]])


if __name__ == "__main__":
    unittest.main()
