"""S5: real Sudachi candidates, isolated project input history and offscreen popup wiring."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from subkit.reading.dictionary import ReadingDict
from subkit.reading.engine import Analyzer, Morph, ReadingEngine
from subkit_app.views.reading_popup import ReadingPopup
from tests import test_convert_ui as fixtures


class DictionaryCandidates(unittest.TestCase):
    def setUp(self):
        self.engine = ReadingEngine()

    def test_real_lookup_all_readings_and_analyzer_first(self):
        for surface, expected in (("他", {"ほか", "あだ", "た"}),
                                  ("私", {"わたくし", "わたし", "あたし", "し"}),
                                  ("方", {"ほう", "かた", "がた", "ぽう"}),
                                  ("不壊", {"ふえ"})):
            with self.subTest(surface=surface):
                result = self.engine.dictionary_readings(surface, 0, len(surface))
                self.assertTrue(expected.issubset(result), result)
                self.assertEqual(result[0], self.engine.analyzer.tokenize(surface)[0].reading)
                self.assertEqual(len(result), len(set(result)))

    def test_inflected_kanji_and_partial_okurigana(self):
        for text, a, b, reading in (("歩いて", 0, 1, "ある"), ("歩いて", 0, 2, "あるい"),
                                    ("弾いて", 0, 1, "はじ"), ("取り扱った", 2, 3, "あつか")):
            with self.subTest(text=text, range=(a, b)):
                self.assertIn(reading, self.engine.dictionary_readings(text, a, b))
        self.assertNotIn("あるく", self.engine.dictionary_readings("歩いて", 0, 1))

    def test_colloquial_long_vowel_candidates_keep_original_ranges(self):
        for mark in ("ー", "ｰ"):
            text = "ねえ 帰ろ" + mark
            with self.subTest(mark=mark):
                self.assertEqual(self.engine.dictionary_readings(text, 3, 4)[0], "かえ")
                self.assertEqual(self.engine.dictionary_readings(text, 3, 5)[0], "かえろ")
                readings = self.engine.dictionary_readings(text, 3, 6)
                self.assertEqual(readings[0], "かえろー")
                self.assertFalse(any("〜" in r or "きごう" in r or "？" in r for r in readings))
        self.assertEqual(self.engine.dictionary_readings("寒ーい", 0, 3)[0], "さむい")

    def test_multiple_morphs_and_visible_line_offsets(self):
        readings = self.engine.dictionary_readings("ここは迷宮都市", 3, 7)
        self.assertEqual(readings[0], "めいきゅうとし")
        self.assertIn("ほか", self.engine.dictionary_readings("ここは他だ", 3, 4))

    def test_lookup_cache_and_reject_non_kana(self):
        analyzer = Analyzer.__new__(Analyzer)
        analyzer._dict = Mock()
        analyzer._dict.lookup.return_value = [Mock(reading_form=Mock(return_value=r))
                                              for r in ("タ", "タ", "他", "A", "タ!", "", "ホカ")]
        self.assertEqual(analyzer.lookup_readings("他"), ["た", "他", "A", "た!", "ほか"])
        returned = analyzer.lookup_readings("他")
        returned.clear()
        self.assertEqual(analyzer.lookup_readings("他"), ["た", "他", "A", "た!", "ほか"])
        analyzer._dict.lookup.assert_called_once_with("他")

    def test_popup_rejects_non_kana_from_analyzer_and_dictionary(self):
        analyzer = Mock()
        analyzer.tokenize.return_value = [Morph("他", 0, 1, "他", (), "他")]
        analyzer.lookup_readings.return_value = ["他", "A", "た!", "た", "ほか"]
        self.assertEqual(ReadingEngine(analyzer=analyzer).dictionary_readings("他", 0, 1), ["た", "ほか"])

    def test_invalid_ranges(self):
        for a, b in ((-1, 1), (0, 2), (1, 1), (1, 0)):
            self.assertEqual(self.engine.dictionary_readings("他", a, b), [])

    def test_unaligned_partial_block_does_not_offer_whole_word(self):
        analyzer = Mock()
        analyzer.tokenize.return_value = [Morph("漢字", 0, 2, "かんじ", (), "漢字")]
        analyzer.lookup_readings.side_effect = lambda s: ["かんじ"] if s == "漢字" else ["かん"]
        self.assertEqual(ReadingEngine(analyzer=analyzer).dictionary_readings("漢字", 0, 1), ["かん"])


class TypedHistory(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp(dir=os.environ["TEMP"])) / "project" / "readings.json"

    def test_roundtrip_recent_dedup_limit_and_copy(self):
        rd = ReadingDict()
        for i in range(12):
            rd.add_typed(self.path, "不壊", str(i))
        rd.add_typed(self.path, "不壊", "5")
        expected = ["5", "11", "10", "9", "8", "7", "6", "4", "3", "2"]
        loaded = ReadingDict.load(self.path)
        self.assertEqual(loaded.typed_readings("不壊"), expected)
        loaded.typed_readings("不壊").clear()
        self.assertEqual(loaded.typed_readings("不壊"), expected)

    def test_every_other_json_value_preserved_and_words_preserve_typed(self):
        data = {"version": 7, "words": {"遠月": "とおつき"}, "readings": {"旧": "きゅう"},
                "speakers": {"人": {"私": "わたし"}}, "prefer": {"方": "ほう"},
                "candidates": {"他": ["た", "他", "A", "た!", "ほか"]}, "splits": {},
                "unknown": {"null": None, "flag": False, "list": [1, {"字": "値"}]},
                "typed": {"別": ["べつ"]}}
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        ReadingDict().add_typed(self.path, "不壊", "ふかい")
        after = json.loads(self.path.read_text(encoding="utf-8"))
        for key, value in data.items():
            self.assertEqual(after[key] if key != "typed" else after[key]["別"],
                             value if key != "typed" else value["別"])
        ReadingDict().add_word(self.path, "語", "ご")
        self.assertEqual(ReadingDict.load(self.path).typed_readings("不壊"), ["ふかい"])

    def test_global_history_ignored_project_isolated_and_auto_identical(self):
        global_path = self.path.parent.parent / "global.json"
        global_path.write_text('{"typed":{"他":["ぐろーばる"]}}', encoding="utf-8")
        global_bytes = global_path.read_bytes()
        before = self.engine_result(ReadingDict.load(global_path, self.path))
        ReadingDict().add_typed(self.path, "他", "たいぷ")
        rd = ReadingDict.load(global_path, self.path)
        self.assertEqual(rd.typed_readings("他"), ["たいぷ"])
        self.assertEqual(self.engine_result(rd), before)
        self.assertEqual(ReadingDict.load(global_path).typed_readings("他"), [])
        self.assertEqual(ReadingDict.load(self.path.parent.parent / "another.json").typed_readings("他"), [])
        self.assertEqual(global_path.read_bytes(), global_bytes)
        self.assertNotIn("他", rd.words)
        self.assertNotIn("他", rd.candidates)

    @staticmethod
    def engine_result(rd):
        return [(e.reading, e.alternatives, e.confidence, e.needs_review) for e in ReadingEngine(rd).analyze("他")]

    def test_manual_removal_seen_on_next_load(self):
        ReadingDict().add_typed(self.path, "不壊", "ふかい")
        self.path.write_text('{}', encoding="utf-8")
        self.assertEqual(ReadingDict.load(self.path).typed_readings("不壊"), [])


class PopupCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def popup(self, **kwargs):
        pop = ReadingPopup("他", "ほか", ["ほか", "た"] + [f"よみ{i}" for i in range(10)],
                           "analyzer", "uncertain", **kwargs)
        self.addCleanup(pop.deleteLater)
        self.addCleanup(pop.close)
        return pop

    def test_all_buttons_wrap_and_number_nine_selects(self):
        pop = self.popup()
        layout = pop.cand_box.layout()
        self.assertEqual(layout.count(), len(pop.cands))
        for i in range(len(pop.cands)):
            self.assertEqual(layout.getItemPosition(i), (i // 4, i % 4, 1, 1))
        got, typed = [], []
        pop.chosen.connect(lambda r, sc: got.append(r))
        pop.typedChosen.connect(lambda s, r: typed.append((s, r)))
        QTest.keyClick(pop, Qt.Key.Key_9)
        self.assertEqual(got, [pop.cands[8]])
        self.assertEqual(typed, [])

    def test_tenth_candidate_click_not_remembered(self):
        pop = self.popup()
        got, typed = [], []
        pop.chosen.connect(lambda r, sc: got.append(r))
        pop.typedChosen.connect(lambda s, r: typed.append((s, r)))
        pop.cand_box.layout().itemAt(9).widget().click()
        self.assertEqual(got, [pop.cands[9]])
        self.assertEqual(typed, [])

    def test_input_all_scopes_and_romaji(self):
        for sc in range(3):
            pop = self.popup()
            typed, got = [], []
            pop.typedChosen.connect(lambda s, r: typed.append((s, r)))
            pop.chosen.connect(lambda r, scope: got.append((r, scope)))
            pop.scope_group.button(sc).setChecked(True)
            pop.edit.setText("hoka")
            pop._accept_input()
            self.assertEqual(typed, [("他", "ほか")])
            self.assertEqual(got, [("ほか", sc)])

    def test_changed_range_history_uses_merged_surface(self):
        scope = {"text": "迷宮都市", "a": 0, "b": 2,
                 "spans": [{"a": 0, "b": 2, "reading": "めいきゅう"}, {"a": 2, "b": 4, "reading": "とし"}]}
        pop = ReadingPopup("迷宮", "めいきゅう", [], "manual", "confirmed", scope=scope)
        self.addCleanup(pop.deleteLater)
        self.addCleanup(pop.close)
        typed, got = [], []
        pop.typedChosen.connect(lambda s, r: typed.append((s, r)))
        pop.chosenRange.connect(lambda a, b, r, sc: got.append((a, b, r)))
        pop._extend(1)
        pop._accept_input()
        self.assertEqual(typed, [("迷宮都市", "めいきゅうとし")])
        self.assertEqual(got, [(0, 4, "めいきゅうとし")])

    def merged_popup(self, candidates_for):
        scope = {"text": "世間話をするくらいなら", "a": 0, "b": 2,
                 "spans": [{"a": 0, "b": 2, "reading": "せけん"},
                           {"a": 2, "b": 3, "reading": "はなし"}]}
        pop = ReadingPopup("世間", "せけん", ["せかん"], "analyzer", "uncertain", scope=scope,
                           candidates_for=candidates_for)
        self.addCleanup(pop.deleteLater)
        self.addCleanup(pop.close)
        return pop

    def test_merged_candidates_layout_click_and_all_scopes(self):
        engine = ReadingEngine()
        callback = Mock(side_effect=lambda a, b: engine.analyzer.lookup_readings("世間話をするくらいなら"[a:b])
                        + ["せけんはなし", "せけんばなし"] + [f"よみ{i}" for i in range(10)])
        for scope in range(3):
            pop = self.merged_popup(callback)
            got, typed = [], []
            pop.chosenRange.connect(lambda a, b, r, sc: got.append((a, b, r, sc)))
            pop.typedChosen.connect(lambda s, r: typed.append((s, r)))
            pop._extend(1)
            callback.assert_called_with(0, 3)
            self.assertTrue(pop.cand_box.isHidden())
            self.assertFalse(pop.range_cand_box.isHidden())
            self.assertEqual(pop.range_cands[0], "せけんばなし")
            self.assertEqual(pop.edit.text(), "せけんはなし")
            self.assertEqual(len(pop.range_cands), len(set(pop.range_cands)))
            layout = pop.range_cand_box.layout()
            self.assertEqual(layout.count(), len(pop.range_cands))
            for i in range(layout.count()):
                self.assertEqual(layout.getItemPosition(i), (i // 4, i % 4, 1, 1))
            self.assertIn("世間話", layout.itemAt(0).widget().text())
            pop.scope_group.button(scope).setChecked(True)
            layout.itemAt(0).widget().click()
            self.assertEqual(got, [(0, 3, "せけんばなし", scope)])
            self.assertEqual(typed, [])

    def test_merged_candidates_keys_work_with_prefill(self):
        for target_edit in (False, True):
            for key, index in ((Qt.Key.Key_1, 0), (Qt.Key.Key_9, 8)):
                readings = ["せけんばなし"] + [f"よみ{i}" for i in range(10)]
                pop = self.merged_popup(lambda a, b: readings)
                got = []
                pop.chosenRange.connect(lambda a, b, r, sc: got.append((a, b, r, sc)))
                pop._extend(1)
                QTest.keyClick(pop.edit if target_edit else pop, key)
                self.assertEqual(got, [(0, 3, readings[index], 0)])

    def test_merged_candidates_do_not_take_ime_composition_digits(self):
        from PySide6.QtGui import QInputMethodEvent

        pop = self.merged_popup(lambda a, b: ["せけんばなし"])
        pop._extend(1)
        chosen = Mock()
        pop.chosenRange.connect(chosen)
        QApplication.sendEvent(pop.edit, QInputMethodEvent("せ", []))
        QTest.keyClick(pop.edit, Qt.Key.Key_1)
        chosen.assert_not_called()
        QApplication.sendEvent(pop.edit, QInputMethodEvent("", []))
        QTest.keyClick(pop.edit, Qt.Key.Key_1)
        chosen.assert_called_once_with(0, 3, "せけんばなし", 0)

    def test_reset_range_restores_original_candidates(self):
        callback = Mock(return_value=["せけんばなし"])
        pop = self.merged_popup(callback)
        pop._extend(1)
        pop._reset_range()
        self.assertFalse(pop.cand_box.isHidden())
        self.assertTrue(pop.range_cand_box.isHidden())
        self.assertEqual(pop.cands, ["せけん", "せかん"])
        self.assertEqual(pop.base, "世間")
        self.assertEqual(pop.edit.text(), "")
        callback.assert_called_once_with(0, 3)

    def test_empty_or_missing_merged_candidates_keep_input_behavior(self):
        for callback in (None, lambda a, b: []):
            pop = self.merged_popup(callback)
            pop._extend(1)
            self.assertTrue(pop.cand_box.isHidden())
            self.assertTrue(pop.range_cand_box.isHidden())
            self.assertEqual(pop.edit.text(), "せけんはなし")
            got = []
            pop.chosenRange.connect(lambda a, b, r, sc: got.append((a, b, r, sc)))
            pop._accept_input()
            self.assertEqual(got, [(0, 3, "せけんはなし", 0)])

    def test_pinyin_layout_and_input_unchanged(self):
        pop = self.popup(kind="pinyin")
        self.assertEqual(len(pop.cand_box.findChildren(QPushButton)), 9)
        typed = Mock()
        pop.typedChosen.connect(typed)
        pop.edit.setText("ta1")
        pop._accept_input()
        typed.assert_not_called()


class CandidateWiring(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.project.dict_path = lambda kind: self.path / "dict" / "readings.json"
        self.m = self.group("ja", text="他")
        self.m.lang, self.m.lane = "ja", "original"
        self.session.engine = ReadingEngine()
        ReadingDict().add_typed(self.project.dict_path("readings"), "他", "たいぷ")

    def test_existing_and_new_popups_order_and_persist_input(self):
        from subkit.model.document import Annotation
        ann = Annotation("ruby", "ruby", self.m.id, [{"ins": self.m.id, "a": 0, "b": 1}],
                         "他", "げんざい", candidates=["るーる", "げんざい"], origin="manual")
        self.doc.annotations[ann.id] = ann
        with patch.object(ReadingPopup, "popup_at") as shown, patch.object(self.window, "_apply_reading"):
            for existing in (True, False):
                with patch("subkit_app.mainwindow.ReadingPopup", wraps=ReadingPopup) as factory:
                    if existing:
                        self.window._on_reading(ann.id, QPoint())
                    else:
                        self.window._on_new_reading(self.m.id, 0, 1, QPoint())
                    pop = self.window.findChildren(ReadingPopup)[-1]
                    self.addCleanup(pop.deleteLater)
                    self.addCleanup(pop.close)
                    self.assertIn("たいぷ", pop.cands)
                    self.assertIn("た", pop.cands)
                    self.assertLess(pop.cands.index("るーる"), pop.cands.index("たいぷ"))
                    self.assertLess(pop.cands.index("たいぷ"), pop.cands.index("た"))
                    if existing:
                        self.assertEqual(pop.cands[0], "げんざい")
                    pop.edit.setText("にゅうりょく")
                    pop._accept_input()
                    self.assertEqual(ReadingDict.load(self.project.dict_path("readings")).typed_readings("他")[0],
                                     "にゅうりょく")
            self.assertEqual(shown.call_count, 2)

    def test_existing_and_new_popups_merged_lexicon_history_parts_order(self):
        from subkit.model.document import Annotation, Run

        text = "世間話をするくらいなら"
        self.m.runs = [Run(text, ins=self.m.id, a=0, b=len(text))]
        ann = Annotation("merged", "ruby", self.m.id, [{"ins": self.m.id, "a": 0, "b": 2}],
                         "世間", "せけん", origin="analyzer")
        self.doc.annotations[ann.id] = ann
        ReadingDict().add_typed(self.project.dict_path("readings"), "世間話", "たいぷ")
        ReadingDict().add_typed(self.project.dict_path("readings"), "世間話", "せけんばなし")
        scope = {"text": text, "a": 0, "b": 2,
                 "spans": [{"a": 0, "b": 2, "reading": "せけん", "ann": ann.id},
                           {"a": 2, "b": 3, "reading": "はなし"}]}
        with patch.object(ReadingPopup, "popup_at"), patch.object(self.window, "_reading_scope", return_value=scope), \
                patch.object(self.window, "_apply_reading") as apply:
            for existing in (True, False):
                if existing:
                    self.window._on_reading(ann.id, QPoint())
                else:
                    self.window._on_new_reading(self.m.id, 0, 2, QPoint())
                pop = self.window.findChildren(ReadingPopup)[-1]
                self.addCleanup(pop.deleteLater)
                self.addCleanup(pop.close)
                with patch.object(self.session.engine, "dictionary_readings", return_value=["せけんはなし", "たいぷ"]):
                    pop._extend(1)
                self.assertEqual(pop.range_cands, ["せけんばなし", "たいぷ", "せけんはなし"])
                pop.scope_group.button(2).setChecked(True)
                pop.range_cand_box.layout().itemAt(0).widget().click()
                apply.assert_called_with(self.m.id, (0, 3), "せけんばなし", 2)

    def test_non_japanese_existing_popup_has_no_merged_callback(self):
        from subkit.model.document import Annotation

        self.m.lang = "en"
        ann = Annotation("other", "ruby", self.m.id, [{"ins": self.m.id, "a": 0, "b": 1}],
                         "他", "other", origin="manual")
        self.doc.annotations[ann.id] = ann
        with patch.object(ReadingPopup, "popup_at"):
            self.window._on_reading(ann.id, QPoint())
        pop = self.window.findChildren(ReadingPopup)[-1]
        self.addCleanup(pop.deleteLater)
        self.addCleanup(pop.close)
        self.assertIsNone(pop.candidates_for)

    def test_save_error_still_applies_reading(self):
        with patch("subkit_app.mainwindow.ReadingPopup", wraps=ReadingPopup), \
                patch.object(ReadingPopup, "popup_at"), patch.object(self.window, "_apply_reading") as apply:
            self.window._on_new_reading(self.m.id, 0, 1, QPoint())
            pop = self.window.findChildren(ReadingPopup)[-1]
            self.addCleanup(pop.deleteLater)
            self.addCleanup(pop.close)
            pop.edit.setText("ほか")
            with patch.object(ReadingDict, "add_typed", side_effect=OSError("write denied")):
                pop._accept_input()
            apply.assert_called_once()
            self.assertIn("手打读音未能记入工程词库", self.window.statusBar().currentMessage())
