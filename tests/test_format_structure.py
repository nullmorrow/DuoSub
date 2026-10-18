"""Character identity formatting survives structural edits and document persistence."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.core.ops import Editor
from subkit.core.readings import anchor_of, visible_map
from subkit.core.textformat import char_props
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json, load_document, save_document


class FormatStructure(unittest.TestCase):
    def make(self, texts=("abc", "def"), *, lane="original", source=False):
        doc = Document("format-structure")
        for i, text in enumerate(texts):
            gid, mid = f"g{i}", f"m{i}"
            run = Run(text, src="src", e=i, a=0, b=len(text)) if source else \
                Run(text, ins=f"i{i}", a=0, b=len(text))
            doc.groups.append(Group(gid, members=[mid]))
            doc.members[mid] = Member(mid, gid, "ja" if lane == "original" else "zh-Hans", lane,
                                      "dialogue", i * 1000, (i + 1) * 1000, runs=[run])
        sources = {"src": SimpleNamespace(events=[SimpleNamespace(text=t) for t in texts])} if source else {}
        return doc, Editor(doc, sources)

    def snapshot(self, doc):
        data = copy.deepcopy(document_to_json(doc))
        data.pop("revision", None)
        return data

    def roundtrip_and_history(self, doc, ed, before):
        after = self.snapshot(doc)
        path = Path(tempfile.mkdtemp(prefix="format-structure-", dir=os.environ.get("TEMP"))) / "document.json"
        save_document(doc, path)
        loaded = load_document(path)
        self.assertEqual(self.snapshot(loaded), after)
        for mid, m in doc.members.items():
            self.assertEqual(char_props(loaded.members[mid]), char_props(m))
        self.assertIsNotNone(ed.history.undo())
        self.assertEqual(self.snapshot(doc), before)
        self.assertIsNotNone(ed.history.redo())
        self.assertEqual(self.snapshot(doc), after)
        save_document(doc, path)
        self.assertEqual(self.snapshot(load_document(path)), after)

    def test_split_crossing_span_source_and_inserted_characters(self):
        for source in (False, True):
            with self.subTest(source=source):
                doc, ed = self.make(("abcdef",), source=source)
                ed.set_format("m0", {"b": 1, "c1": "&H0000FF&"}, 1, 5)
                before = self.snapshot(doc)
                gid = ed.split("g0", jp_at=3)
                left, right = doc.members["m0"], ed.lane_member(doc.group(gid), "original")
                props = {"b": 1, "c1": "&H0000FF&"}
                self.assertEqual(char_props(left), [None, props, props])
                self.assertEqual(char_props(right), [props, props, None])
                self.assertEqual(left.format[0]["anchor"][0]["b"], 3)
                self.assertEqual(right.format[0]["anchor"][0]["a"], 3)
                self.roundtrip_and_history(doc, ed, before)

    def test_split_copies_all_overrides_on_both_lanes(self):
        for lane in ("original", "translation"):
            with self.subTest(lane=lane):
                doc, ed = self.make(("abcdef",), lane=lane)
                ed.set_format("m0", {"fn": "Arial", "fs": 64, "b": 1, "i": 0,
                                     "c1": "&H0000FF&", "raw": r"\org(10,20)\t(0,500,\frz30)"})
                doc.members["m0"].overrides.update(pos=[20, 30], an=8, move=[20, 30, 40, 50])
                ed.set_format("m0", {"i": 1}, 2, 5)
                before = self.snapshot(doc)
                gid = ed.split("g0", **({"jp_at": 3} if lane == "original" else {"cn_at": 3}))
                left, right = doc.members["m0"], ed.lane_member(doc.group(gid), lane)
                self.assertEqual(right.overrides, left.overrides)
                self.assertIsNot(right.overrides, left.overrides)
                self.assertIsNot(right.overrides["pos"], left.overrides["pos"])
                self.assertEqual(char_props(right), [{"i": 1}, {"i": 1}, None])
                self.roundtrip_and_history(doc, ed, before)

    def test_merge_gpt_reproduction_keeps_second_span(self):
        doc, ed = self.make()
        ed.set_format("m1", {"b": 1}, 0, 3)
        before = self.snapshot(doc)
        ed.merge_with_next("g0")
        self.assertEqual(doc.members["m0"].text, "abc  def")
        self.assertEqual(char_props(doc.members["m0"]), [None] * 5 + [{"b": 1}] * 3)
        self.roundtrip_and_history(doc, ed, before)

    def test_merge_second_line_defaults_combine_with_character_overrides(self):
        doc, ed = self.make(source=True)
        ed.set_format("m0", {"fs": 40, "b": 0})
        ed.set_format("m0", {"c1": "&HFFFFFF&"}, 0, 1)
        ed.set_format("m1", {"fn": "Arial", "fs": 64, "b": 1, "i": 0, "c1": "&H0000FF&",
                             "raw": r"\frz15\bord2"})
        ed.set_format("m1", {"fs": 90, "i": 1, "raw": r"\blur1"}, 1, 2)
        before = self.snapshot(doc)
        ed.merge_with_next("g0")
        m = doc.members["m0"]
        self.assertEqual(m.overrides, {"fs": 40, "b": 0})
        props = char_props(m)
        self.assertEqual(props[0], {"c1": "&HFFFFFF&"})
        self.assertIsNone(props[4])  # joining separator has no character identity
        expected = {"fn": "Arial", "fs": 64, "b": 1, "i": 0, "c1": "&H0000FF&",
                    "raw": r"\frz15\bord2"}
        self.assertEqual(props[5], expected)
        self.assertEqual(props[6], {**expected, "fs": 90, "i": 1, "raw": r"\frz15\bord2\blur1"})
        self.assertEqual(props[7], expected)
        self.assertIsNone(ed.merge_note)
        self.roundtrip_and_history(doc, ed, before)

    def test_merge_conflicting_position_and_animation_keep_first(self):
        doc, ed = self.make()
        doc.members["m0"].overrides = {"pos": [10, 20], "an": 2, "raw": r"\t(0,500,\frz10)"}
        doc.members["m1"].overrides = {"move": [30, 40, 50, 60], "an": 8,
                                       "raw": r"\org(30,40)\t(0,500,\frz30)\bord3"}
        first = copy.deepcopy(doc.members["m0"].overrides)
        before = self.snapshot(doc)
        ed.merge_with_next("g0")
        self.assertEqual(doc.members["m0"].overrides, first)
        self.assertEqual(ed.merge_note, "合并：第二行的位置／动画设置没有保留")
        self.assertEqual(char_props(doc.members["m0"])[-3:], [{"raw": r"\bord3"}] * 3)
        self.roundtrip_and_history(doc, ed, before)
        ed.merge_with_next("g0")  # no neighbour: do not keep a stale warning
        self.assertIsNone(ed.merge_note)

    def test_merge_same_position_and_animation_has_no_warning(self):
        doc, ed = self.make()
        overrides = {"pos": [10, 20], "an": 2, "raw": r"\org(10,20)\t(0,500,\frz10)"}
        for m in doc.members.values():
            m.overrides = copy.deepcopy(overrides)
        before = self.snapshot(doc)
        ed.merge_with_next("g0")
        self.assertEqual(doc.members["m0"].overrides, overrides)
        self.assertIsNone(ed.merge_note)
        self.roundtrip_and_history(doc, ed, before)

    def test_move_both_directions_and_take_preserve_crossing_span(self):
        for direction, gid in (("head_to_prev", "g1"), ("tail_to_next", "g0"),
                               ("take_from_prev", "g1"), ("take_from_next", "g0")):
            for source in (False, True):
                with self.subTest(direction=direction, source=source):
                    doc, ed = self.make(("abcd", "efgh"), source=source)
                    ed.set_format("m0", {"b": 1}, 1, 3)
                    ed.set_format("m1", {"i": 1}, 1, 3)
                    wanted = {ident: props for m in doc.members.values()
                              for ident, props in zip(visible_map(m)[1], char_props(m)) if ident is not None}
                    before = self.snapshot(doc)
                    self.assertTrue(ed.move_phrase(gid, "original", direction, at=2))
                    for m in doc.members.values():
                        for ident, props in zip(visible_map(m)[1], char_props(m)):
                            self.assertEqual(props, wanted.get(ident))
                        owned = {ident for ident, *_ in m.identities() if ident is not None}
                        for span in m.format:
                            from subkit.core.textformat import _ids_of
                            self.assertTrue(set(_ids_of(span["anchor"])) <= owned)
                    self.roundtrip_and_history(doc, ed, before)

    def test_move_whole_line_gpt_reproduction(self):
        for direction, gid, mid in (("head_to_prev", "g1", "m1"), ("tail_to_next", "g0", "m0")):
            with self.subTest(direction=direction):
                doc, ed = self.make()
                ed.set_format(mid, {"b": 1}, 0, 3)
                before = self.snapshot(doc)
                self.assertTrue(ed.move_phrase(gid, "original", direction))
                self.assertNotIn(mid, doc.members)
                m = next(iter(doc.members.values()))
                props = char_props(m)
                self.assertEqual(props[-3:] if mid == "m1" else props[:3], [{"b": 1}] * 3)
                self.roundtrip_and_history(doc, ed, before)

    def test_move_to_group_without_original_member(self):
        doc, ed = self.make(("abcd", "翻译"))
        doc.members["m1"].lane = "translation"
        ed.set_format("m0", {"b": 1}, 1, 4)
        before = self.snapshot(doc)
        self.assertTrue(ed.move_phrase("g0", "original", "tail_to_next", at=2))
        moved = ed.lane_member(doc.group("g1"), "original")
        self.assertEqual(char_props(moved), [{"b": 1}] * 2)
        self.assertEqual(char_props(doc.members["m0"]), [None, {"b": 1}])
        self.roundtrip_and_history(doc, ed, before)

    def test_split_format_and_reading_use_same_character_ownership(self):
        doc, ed = self.make(("猫と犬",))
        ed.set_format("m0", {"b": 1}, 0, 3)
        ids = visible_map(doc.members["m0"])[1]
        doc.annotations["a1"] = Annotation("a1", "ruby", "m0", anchor_of(ids[2:]), "犬", "いぬ",
                                            status="confirmed", origin="manual")
        before = self.snapshot(doc)
        gid = ed.split("g0", jp_at=2)
        right = ed.lane_member(doc.group(gid), "original")
        self.assertEqual(doc.annotations["a1"].member, right.id)
        self.assertEqual(char_props(right), [{"b": 1}])
        self.roundtrip_and_history(doc, ed, before)

    def test_split_at_span_edges_and_text_edges(self):
        for cut in (0, 1, 4, 6):
            with self.subTest(cut=cut):
                doc, ed = self.make(("abcdef",))
                ed.set_format("m0", {"fs": 72}, 1, 4)
                before = self.snapshot(doc)
                gid = ed.split("g0", jp_at=cut)
                left = char_props(doc.members["m0"])
                right = ed.lane_member(doc.group(gid), "original")
                self.assertEqual(left + (char_props(right) if right else []),
                                 [None, {"fs": 72}, {"fs": 72}, {"fs": 72}, None, None])
                self.roundtrip_and_history(doc, ed, before)

    def test_merge_export_keeps_each_lines_text_properties(self):
        from subkit.ass.document import Style
        from subkit.ass.state import state_at
        from subkit.core.export import apply_overrides
        from subkit.core.textformat import apply_spans

        doc, ed = self.make()
        ed.set_format("m0", {"fs": 40, "b": 0})
        ed.set_format("m1", {"fs": 64, "b": 1, "raw": r"\fs80\b0"})
        ed.set_format("m1", {"fs": 90}, 1, 2)
        ed.merge_with_next("g0")
        m = doc.members["m0"]
        text = apply_spans(m, visible_map(m)[1], apply_overrides(m.text, m.overrides)[0], Style.new("Default"))
        for pos, size, bold in ((0, 40, 0), (5, 80, 0), (6, 90, 0), (7, 80, 0)):
            st, _ = state_at(text, pos, {"fs", "b"})
            self.assertEqual((st["fs"].num(), st["b"].num()), (size, bold))

    def test_merge_warning_is_shown_by_mainwindow(self):
        from subkit_app.mainwindow import MainWindow

        doc, ed = self.make()
        doc.members["m1"].overrides["pos"] = [10, 20]
        messages = []
        window = SimpleNamespace(_flush_group_selection=lambda: None,
                                 session=SimpleNamespace(editor=ed), current_gid="g0",
                                 _edit=lambda fn: (fn(), True)[1],
                                 statusBar=lambda: SimpleNamespace(showMessage=lambda *args: messages.append(args)))
        MainWindow._on_action(window, "merge")
        self.assertEqual(messages, [("合并：第二行的位置／动画设置没有保留", 6000)])


if __name__ == "__main__":
    unittest.main()
