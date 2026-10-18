"""Review answer parsing, safe location and identity-preserving acceptance (Qt-free)."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.core import karaoke
from subkit.core.display import DisplayRules, traditional_text
from subkit.core.ops import Editor
from subkit.core.readings import anchor_of, visible_map
from subkit.core.review_import import (Answer, accept_all_readings, apply, match, parse_answers,
                                       read_answers, record_decision, write_answers)
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json


def answer(**kw):
    values = dict(group="g_1", lang="ja", kind="reading", target="私", old="わたし", new="わたくし", reason="复查")
    values.update(kw)
    return Answer(**values)


class ReviewImport(unittest.TestCase):
    def setUp(self):
        self.doc = Document("E1")
        self.member("私と私", "ja", "g_1", "m_1")
        self.member("银行软件", "zh-Hans", "g_1", "m_zh")
        self.annotation("m_1", 0, 1, "わたし")
        self.annotation("m_1", 2, 3, "あたし")
        self.ed = Editor(self.doc, {})
        self.rules = DisplayRules()

    def member(self, text, lang="ja", group="g_2", mid="m_2", runs=None):
        m = Member(mid, group, lang, "original", "dialogue", 100, 1000,
                   runs=runs or [Run(text, ins="i_" + mid, b=len(text))])
        self.doc.members[mid] = m
        g = next((g for g in self.doc.groups if g.id == group), None)
        if g is None:
            g = Group(group)
            self.doc.groups.append(g)
            self.doc.structure_changed()
        g.members.append(mid)
        return m

    def annotation(self, mid, a, b, text, kind="ruby", **kw):
        m = self.doc.members[mid]
        vt, ids = visible_map(m)
        aid = "a_" + str(len(self.doc.annotations))
        ann = Annotation(aid, kind, mid, anchor_of(ids[a:b]), vt.text[a:b], text, **kw)
        self.doc.annotations[aid] = ann
        return ann

    def match(self, item, **kw):
        return match(self.doc, {}, self.rules, item, **kw)

    def test_parse_jsonl_array_fences_prose_and_raw_errors(self):
        value = vars(answer())
        raw = json.dumps(value, ensure_ascii=False)
        text = "说明\n```json\n" + raw + "\n```\n" + json.dumps([value, value]) + '\n{"group":"bad"}\n{broken}\n'
        items, unreadable = parse_answers(text)
        self.assertEqual(len(items), 3)
        self.assertEqual(unreadable, ["说明", '{"group":"bad"}', "{broken}"])
        pretty = json.dumps(value, ensure_ascii=False, indent=2)
        self.assertEqual(parse_answers("```json\n" + pretty + "\n```"), ([answer()], []))
        self.assertEqual(parse_answers("[]"), ([], []))

    def test_normalize_unknown_kind_language_and_scalar(self):
        value = dict(vars(answer()), group=42, kind=" Surprise ", lang=" JA-jp ")
        items, errors = parse_answers(json.dumps(value))
        self.assertFalse(errors)
        self.assertEqual((items[0].group, items[0].kind, items[0].lang), ("42", "other", "ja"))
        value["target"] = {"unsafe": "value"}
        self.assertEqual(parse_answers(json.dumps(value))[0], [])

    def test_inline_json_and_bom_preserve_raw_prose(self):
        raw = json.dumps(vars(answer()), ensure_ascii=False)
        self.assertEqual(parse_answers("\ufeff" + raw), ([answer()], []))
        items, errors = parse_answers("解释 " + raw + " 结束")
        self.assertEqual(items, [answer()])
        self.assertEqual(errors, ["解释 " + raw + " 结束"])

    def test_match_classification_and_old_disambiguates(self):
        before = document_to_json(self.doc)
        found = self.match(answer())
        self.assertEqual((found.status, found.location.a, found.current), ("ok", 0, "わたし"))
        self.assertEqual(self.match(answer(old="不符")).status, "ambiguous")
        self.assertEqual(self.match(answer(target="不存在")).status, "not_found")
        self.assertEqual(self.match(answer(group="wrong")).status, "found_by_text")
        self.assertEqual(document_to_json(self.doc), before)

    def test_unique_mismatch_already_done_and_language_filter(self):
        self.member("猫")
        self.annotation("m_2", 0, 1, "ねこ")
        item = answer(group="g_2", target="猫", old="wrong", new="ねこ")
        self.assertEqual(self.match(item).status, "already_done")
        self.assertEqual(self.match(answer(group="g_2", target="猫", old="wrong")).status, "mismatch")
        self.assertEqual(self.match(answer(lang="en")).status, "not_found")
        self.member("猫", group="g_3", mid="m_3")
        self.assertEqual(self.match(answer(group="bad", target="猫", old="unknown")).status, "ambiguous")

    def test_correct_group_is_preferred_and_wrong_existing_id_falls_back(self):
        self.member("私")
        self.annotation("m_2", 0, 1, "わたし")
        self.assertEqual(self.match(answer()).location.member, "m_1")
        self.assertEqual(self.match(answer(group="g_2", target="銀行", lang="zh")).status, "not_found")
        self.assertEqual(self.match(answer(group="g_2", target="银行", lang="zh", kind="text", old="银行", new="行长")).status, "found_by_text")

    def test_ruby_accept_undo_redo_and_save_round_trip(self):
        before = copy.deepcopy(self.doc.annotations)
        self.assertTrue(apply(self.ed, self.match(answer())))
        self.assertEqual(len(self.ed.history.done), 1)
        self.assertEqual(self.match(answer()).status, "already_done")
        accepted = next(a for a in self.doc.annotations.values() if a.text == "わたくし")
        self.assertEqual((accepted.status, accepted.origin), ("confirmed", "manual"))
        saved = document_to_json(self.doc)
        self.assertEqual(document_to_json(document_from_json(saved)), saved)
        self.ed.history.undo()
        self.assertEqual(self.doc.annotations, before)
        self.ed.history.redo()
        self.assertEqual(document_to_json(self.doc)["annotations"], saved["annotations"])

    def test_pinyin_multiple_syllables_normalized_here_and_undo(self):
        self.annotation("m_zh", 0, 1, "yín", "pinyin")
        self.annotation("m_zh", 1, 2, "xíng", "pinyin")
        item = answer(lang="zh", target="银行", old="yin2 xing2", new="yin2 hang2")
        before = copy.deepcopy(self.doc.annotations)
        self.assertEqual(self.match(item).status, "ok")
        self.assertTrue(apply(self.ed, self.match(item)))
        self.assertEqual(self.match(item).status, "already_done")
        self.assertEqual(self.doc.rules, [])
        self.ed.history.undo()
        self.assertEqual(self.doc.annotations, before)

    def test_invalid_pinyin_rolls_back(self):
        item = answer(lang="zh", target="银行", old="", new="bad6")
        before = copy.deepcopy(self.doc.annotations)
        with self.assertRaises(ValueError):
            apply(self.ed, self.match(item))
        self.assertEqual(self.doc.annotations, before)
        self.assertEqual(len(self.ed.history.done), 0)

    def test_text_preserves_unselected_identities_readings_and_karaoke(self):
        m = self.member("猫犬鳥")
        keep = self.annotation(m.id, 2, 3, "とり", status="confirmed", origin="manual")
        m.role = "song"
        self.doc.group(m.group).kind = "song"
        beats = karaoke.derive(self.doc, m)
        for i, beat in enumerate(beats):
            beat.t0, beat.t1, beat.by = 100 + i * 200, 300 + i * 200, "tap"
        self.ed.set_karaoke(m.id, beats)
        before = copy.deepcopy(self.doc)
        ids = visible_map(m)[1]
        item = answer(group="g_2", kind="text", target="犬", old="犬", new="魚魚")
        self.assertTrue(apply(self.ed, self.match(item)))
        self.assertEqual(m.text, "猫魚魚鳥")
        self.assertEqual([visible_map(m)[1][0], visible_map(m)[1][-1]], [ids[0], ids[-1]])
        self.assertEqual(self.doc.annotations[keep.id], keep)
        view = karaoke.effective(self.doc, m)
        self.assertEqual((view.beats[-1].t0, view.beats[-1].t1), (beats[-1].t0, beats[-1].t1))
        self.assertEqual(len(self.ed.history.done), 2)
        self.ed.history.undo()
        self.assertEqual(document_to_json(self.doc)["members"], document_to_json(before)["members"])
        self.assertEqual(self.doc.karaoke, before.karaoke)

    def test_repeated_text_does_not_steal_suffix_identity(self):
        m = self.member("猫猫犬")
        ann = self.annotation(m.id, 0, 1, "ねこ", status="confirmed")
        old_ids = visible_map(m)[1]
        item = answer(group="g_2", kind="text", target="猫犬", old="猫犬", new="犬")
        self.assertTrue(apply(self.ed, self.match(item)))
        self.assertEqual(m.text, "猫犬")
        self.assertEqual(visible_map(m)[1], [old_ids[0], old_ids[2]])
        self.assertIn(ann.id, self.doc.annotations)

    def test_editing_offset_conversion_preserves_ass_tags_and_breaks(self):
        runs = [Run("猫", ins="i_a", b=1), Run("\\N", "sep"), Run("{\\i1}", "tag"),
                Run("犬", ins="i_b", b=1), Run("\\h", "sep"), Run("鳥", ins="i_c", b=1)]
        m = self.member("", runs=runs)
        item = answer(group="g_2", kind="text", target="鳥", old="鳥", new="魚")
        self.assertTrue(apply(self.ed, self.match(item)))
        self.assertEqual(m.text, "猫\\N{\\i1}犬\\h魚")
        item = answer(group="g_2", kind="text", target="猫犬", old="猫犬", new="虎")
        self.assertFalse(apply(self.ed, self.match(item)))

    def test_traditional_display_locates_original_and_writes_simplified(self):
        self.rules = DisplayRules(zh_traditional_enabled=True, zh_traditional_phrases=True)
        item = answer(lang="zh", kind="text", target="軟體", old="軟體", new="硬體")
        suggestion = self.match(item)
        self.assertEqual((suggestion.status, suggestion.location.a, suggestion.location.b), ("ok", 2, 4))
        self.assertTrue(apply(self.ed, suggestion))
        self.assertEqual(self.doc.members["m_zh"].text, "银行硬体")
        self.assertEqual(self.match(item).status, "already_done")
        self.ed.history.undo()
        self.assertEqual(self.doc.members["m_zh"].text, "银行软件")

    def test_phrase_expansion_has_safe_boundaries(self):
        m = self.member("互联网猫", "zh-Hans")
        self.rules = DisplayRules(zh_traditional_enabled=True, zh_traditional_phrases=True)
        _raw, bounds = traditional_text(m.text, self.rules)
        self.assertEqual(bounds, [0, 1, 3, 4, 5])
        item = answer(group="g_2", lang="zh", kind="text", target="網", old="網", new="線")
        suggestion = self.match(item)
        self.assertEqual(suggestion.status, "ambiguous")
        self.assertFalse(apply(self.ed, suggestion))
        item = answer(group="g_2", lang="zh", kind="text", target="網際網路", old="網際網路", new="網路")
        self.assertTrue(apply(self.ed, self.match(item)))
        self.assertEqual(m.text, "网路猫")

    def test_stale_and_informational_suggestions_never_apply(self):
        suggestion = self.match(answer())
        self.ed.insert_text("m_1", 0, "あ")
        self.assertFalse(apply(self.ed, suggestion))
        for kind in ("pairing", "speaker", "persona", "other"):
            suggestion = self.match(answer(kind=kind, target="と"))
            self.assertEqual(suggestion.status, "ok")
            self.assertFalse(apply(self.ed, suggestion))
        self.assertEqual(len(self.ed.history.done), 1)

    def test_accept_all_single_step_skips_fallback_mismatch_done_and_duplicate(self):
        first = self.match(answer())
        second = self.match(answer(old="あたし", new="わたし"))
        skip = self.match(answer(group="bad"))
        before = copy.deepcopy(self.doc.annotations)
        self.assertEqual(accept_all_readings(self.ed, [first, first, second, skip]), 2)
        self.assertEqual(len(self.ed.history.done), 1)
        self.ed.history.undo()
        self.assertEqual(self.doc.annotations, before)
        self.ed.history.redo()
        self.assertEqual(accept_all_readings(self.ed, [first, second]), 0)

    def test_accept_all_invalid_later_item_rolls_back_entire_step(self):
        good = self.match(answer())
        bad = self.match(answer(lang="zh", target="银行", old="", new="invalid"))
        before = copy.deepcopy(self.doc.annotations)
        with self.assertRaises(ValueError):
            accept_all_readings(self.ed, [good, bad])
        self.assertEqual(self.doc.annotations, before)
        self.assertEqual(self.ed.history.done, [])

    def test_ruby_piece_keeps_other_manual_piece(self):
        m = self.member("落ち着く")
        ann = self.annotation(m.id, 0, 4, "おちつく", status="confirmed", origin="manual",
                              parts=[[0, 1, "お"], [2, 3, "つ"]])
        item = answer(group="g_2", target="落", old="お", new="ら")
        self.assertTrue(apply(self.ed, self.match(item)))
        kept = next(a for a in self.doc.annotations.values() if a.member == m.id and a.base == "着")
        self.assertEqual((kept.text, kept.status, kept.origin), ("つ", "confirmed", "manual"))
        self.ed.history.undo()
        self.assertEqual(self.doc.annotations[ann.id], ann)

    def test_override_and_delete_text_preserve_original_source_tombstones(self):
        from subkit.model.source import Source, SourceEvent
        src = Source("s", "ja", "original", "srt", "unused", "unused", "hash", 9, "utf-8", "lf",
                     events=[SourceEvent(0, "Dialogue", 100, 1000, "猫犬鳥")])
        m = self.member("", runs=[Run("猫犬鳥", src="s", e=0, b=3)])
        self.ed = Editor(self.doc, {"s": src})
        item = answer(group="g_2", kind="text", target="犬", old="犬", new="魚")
        self.assertTrue(apply(self.ed, self.match(item), new_override=""))
        self.assertEqual(m.text, "猫鳥")
        self.assertEqual([(t.a, t.b, t.text) for t in self.doc.tombstones], [(1, 2, "犬")])
        self.assertEqual(visible_map(m)[1], [("s", "s", 0, 0), ("s", "s", 0, 2)])

    def test_decision_log_round_trip_reimport_and_reason_independent_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertEqual(read_answers(path), {})
            item = answer()
            record_decision(path, item, "rejected", at="2026-10-04T00:00:00+00:00")
            records = read_answers(path)
            self.assertEqual(self.match(item, decisions=records).decision, "rejected")
            self.assertEqual(self.match(item, decisions=records).status, "already_done")
            self.assertEqual(item.fingerprint, answer(reason="不同解释").fingerprint)
            record_decision(path, item, "accepted")
            record_decision(path, answer(kind="speaker"), "handled")
            records = read_answers(path)
            write_answers(path, records)
            self.assertEqual(read_answers(path), records)
            self.assertEqual(len(records), 2)
            data = json.loads((path / "review/answers.json").read_text(encoding="utf-8"))
            self.assertEqual((data["format"], data["format_version"]), ("subkit-review-answers", 1))
            with self.assertRaises(ValueError):
                record_decision(path, item, "unknown")


if __name__ == "__main__":
    unittest.main()
