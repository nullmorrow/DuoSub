"""English split edits and mapping links; Qt offscreen and dictionary doubles."""
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit.core import karaoke as kara
from subkit.model.serialize import document_from_json, document_to_json
from tests.test_lyrics_map import document


class EnglishSyllables(unittest.TestCase):
    def setUp(self):
        self.dictionary_loader = kara._english_dictionary
        self.fallback = patch.object(kara, "_english_dictionary", return_value=None)
        self.fallback.start()
        self.addCleanup(self.fallback.stop)

    def test_fallback_words_suffixes_and_contractions(self):
        counts = {"revolution": 4, "forever": 3, "missing": 2, "truth": 1, "take": 1,
                  "away": 2, "long": 1, "time": 1, "Let's": 1, "vision": 2,
                  "walked": 1, "wanted": 2, "table": 2}
        for word, count in counts.items():
            with self.subTest(word=word):
                parts = kara.english_syllables(word)
                self.assertEqual(len(parts), count)
                self.assertEqual("".join(parts), word)
                self.assertTrue(all(parts))
        self.assertEqual(kara.english_syllables("revolution")[-1], "tion")
        self.assertEqual(kara.english_syllables("vision")[-1], "sion")
        self.assertEqual(kara.english_syllables("日本語"), ["日本語"])

    def test_optional_dictionary_en_us_and_missing_import(self):
        dictionary = SimpleNamespace(inserted=lambda word: "Rev-o-lu-tion" if word == "Revolution" else word)
        factory = unittest.mock.Mock(return_value=dictionary)
        self.dictionary_loader.cache_clear()
        self.addCleanup(self.dictionary_loader.cache_clear)
        with patch.dict("sys.modules", {"pyphen": SimpleNamespace(Pyphen=factory)}):
            loaded = self.dictionary_loader()
            factory.assert_called_once_with(lang="en_US", left=1, right=1)
            with patch.object(kara, "_english_dictionary", return_value=loaded):
                self.assertEqual(kara.english_syllables("Revolution"), ["Rev", "o", "lu", "tion"])
                self.assertEqual(kara.english_syllables("Let's"), ["Let's"])
        typesetting = SimpleNamespace(inserted=lambda word: {"away": "away", "miracle": "mir-a-cle"}.get(word, word))
        with patch.object(kara, "_english_dictionary", return_value=typesetting):
            self.assertEqual(kara.english_syllables("away"), ["a", "way"])        # dictionary leaves it whole
            self.assertEqual(kara.english_syllables("miracle"), ["mir", "a", "cle"])  # same count: dictionary positions
        self.dictionary_loader.cache_clear()
        with patch.dict("sys.modules", {"pyphen": None}):
            self.assertIsNone(self.dictionary_loader())

    def test_two_lanes_one_undo_noop_save_and_source_identity(self):
        doc, ed = document("Revolution truth", "forever away Let's")
        before = copy.deepcopy(document_to_json(doc))
        self.assertEqual(ed.split_english_syllables("g"), 2)
        self.assertEqual(len(ed.history.done), 1)
        self.assertEqual(ed.history.done[-1].label, "英文按音节拆开")
        after = copy.deepcopy(doc.karaoke)
        for m in doc.members.values():
            self.assertEqual(document_to_json(doc)["members"][m.id], before["members"][m.id])
            self.assertFalse(kara.effective(doc, m).complete)
        self.assertEqual(ed.split_english_syllables("g"), 0)
        self.assertEqual(len(ed.history.done), 1)
        reopened = document_from_json(json.loads(json.dumps(document_to_json(doc))))
        self.assertEqual(reopened.karaoke, after)
        ed.history.undo()
        self.assertEqual(doc.karaoke, {})
        ed.history.redo()
        self.assertEqual(doc.karaoke, after)

    def test_timed_proportional_rounding_untouched_words_and_effect(self):
        doc, ed = document("revolution truth", "take")
        m = doc.members["jp"]
        beats = kara.derive(doc, m)
        for b, a, z in zip(beats, [1001, 1102], [1102, 1302]):
            b.t0, b.t1, b.by = a, z, "tap"
        ed.set_karaoke(m.id, beats, effect={"preset": "ko"})
        ed.split_english_syllables("g")
        view = kara.effective(doc, m)
        pieces = view.beats[:-1]
        lengths = [len(b.sung) for b in pieces]
        ends, used = [], 0
        for length in lengths:
            used += length
            ends.append(1001 + 101 * used // sum(lengths))
        self.assertEqual([b.t1 for b in pieces], ends)
        self.assertEqual([b.t0 for b in pieces], [1001] + ends[:-1])
        self.assertEqual([b.by for b in pieces], ["placeholder"] * 4)
        self.assertEqual((view.beats[-1].t0, view.beats[-1].t1, view.beats[-1].by), (1102, 1302, "tap"))
        self.assertEqual(view.kind, "ko")
        self.assertFalse(view.complete)
        # Export boundaries also use the edited character identities.
        from subkit.core.readings import visible_map
        self.assertEqual(len(kara.base_inserts(view, visible_map(m)[1], 100)), 5)

    def test_translation_split_keeps_outer_mapping_links_and_follows_retap(self):
        doc, ed = document("かなよ", "revolution")
        ed.apply_lyrics_mapping({"g": "revolution（かなよ）"})
        original_links = copy.deepcopy(kara.effective(doc, doc.members["cn"]).beats[0].link)
        ed.split_english_syllables("g")
        view = kara.effective(doc, doc.members["cn"])
        self.assertEqual(view.beats[0].link, {"start": original_links["start"]})
        self.assertEqual(view.beats[-1].link, {"end": original_links["end"]})
        self.assertFalse(any(b.link for b in view.beats[1:-1]))
        beats = kara.effective(doc, doc.members["jp"]).beats
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 2000 + i * 400, 2400 + i * 400, "tap"
        ed.set_karaoke("jp", beats)
        view = kara.effective(doc, doc.members["cn"])
        self.assertEqual((view.beats[0].t0, view.beats[-1].t1), (2000, 3200))
        self.assertFalse(view.complete)
        self.assertNotIn(kara.LINK_MISSING, view.issues)

    def test_dictionary_edits_and_merged_multiword_mapping(self):
        doc, ed = document("かな", "revolution forever")
        ed.apply_lyrics_mapping({"g": "revolution forever（かな）"})
        dictionary = SimpleNamespace(inserted=lambda w: {"revolution": "rev-o-lu-tion",
                                                         "forever": "for-ev-er"}.get(w, w))
        with patch.object(kara, "_english_dictionary", return_value=dictionary):
            self.assertEqual(ed.split_english_syllables("g"), 1)
            view = kara.effective(doc, doc.members["cn"])
            self.assertEqual([b.sung for b in view.beats], ["rev", "o", "lu", "tion", "for", "ev", "er"])
            self.assertEqual(ed.split_english_syllables("g"), 0)
        self.assertEqual(set(view.beats[0].link), {"start"})
        self.assertEqual(set(view.beats[-1].link), {"end"})

    def test_original_split_retargets_translation_mapping_in_same_transaction(self):
        doc, ed = document("revolution", "革命")
        ed.apply_lyrics_mapping({"g": "革命（revolution）"})
        before = copy.deepcopy(doc.karaoke)
        ed.split_english_syllables("g")
        originals = kara.effective(doc, doc.members["jp"]).beats
        translation = kara.effective(doc, doc.members["cn"]).beats[0]
        self.assertEqual(translation.link["start"], kara.beat_link(originals[0], "start"))
        self.assertEqual(translation.link["end"], kara.beat_link(originals[-1], "end"))
        self.assertNotIn(kara.LINK_MISSING, kara.effective(doc, doc.members["cn"]).issues)
        ed.history.undo()
        self.assertEqual(doc.karaoke, before)
        ed.history.redo()
        for i, b in enumerate(originals):
            b.t0, b.t1, b.by = 2000 + i * 300, 2300 + i * 300, "tap"
        ed.set_karaoke("jp", originals)
        translation = kara.effective(doc, doc.members["cn"]).beats[0]
        self.assertEqual((translation.t0, translation.t1), (2000, 3200))

    def test_manual_split_preserved_tiny_span_and_song_guard(self):
        doc, ed = document("missing", "truth")
        ed.split_beat("jp", kara.boundary_key(doc, doc.members["jp"], 1))
        before = copy.deepcopy(kara.edits_of(doc, doc.members["jp"]))
        ed.split_english_syllables("g")
        self.assertIn(before[0], kara.edits_of(doc, doc.members["jp"]))
        doc2, ed2 = document("away", "truth")
        beats = kara.derive(doc2, doc2.members["jp"])
        beats[0].t0, beats[0].t1, beats[0].by = 1000, 1001, "tap"
        ed2.set_karaoke("jp", beats)
        ed2.split_english_syllables("g")
        new = kara.effective(doc2, doc2.members["jp"]).beats
        self.assertEqual([(b.t0, b.t1) for b in new], [(1000, 1000), (1000, 1001)])
        self.assertFalse(any(b.timed for b in new))
        doc2.groups[0].kind = "dialogue"
        self.assertEqual(ed2.split_english_syllables("g"), 0)


class EnglishSyllablesUI(unittest.TestCase):
    def test_offscreen_button_refresh_dispatch_and_help(self):
        from PySide6.QtCore import qVersion
        from PySide6.QtWidgets import QApplication
        from subkit_app.mainwindow import MainWindow
        from subkit_app.views.waveform import WaveformPanel, WaveformView
        app = QApplication.instance() or QApplication([])
        self.assertEqual(app.platformName(), "offscreen")
        print("English syllables UI: Qt", qVersion(), "offscreen, player double")
        wave = WaveformView()
        panel = WaveformPanel(wave)
        self.addCleanup(panel.deleteLater)
        doc, ed = document("truth take long time Let's", "away")
        fake = SimpleNamespace(session=SimpleNamespace(doc=doc, editor=ed), current_gid="g", wave=wave,
            wave_panel=panel, project=SimpleNamespace(languages={"original": "ja", "translation": "zh-Hant"}),
            _k_unit_pref="syllable", _refresh_k_fx=lambda v: None, _flush_group_selection=lambda: None,
            _sync_k_menu=lambda: None, _edit=lambda fn: fn())
        fake._k_member = lambda g, lane: next((m for m in doc.members_of(g) if m.lane == lane), None)
        fake._refresh_k = lambda: MainWindow._refresh_k(fake)
        fake._refresh_k()
        self.assertTrue(panel.btn_kenglish.isEnabled())
        self.assertEqual(panel.btn_kenglish.help_anchor, "karaoke")
        panel.kAction.connect(lambda name: MainWindow._k_action(fake, name))
        panel.btn_kenglish.click()
        self.assertEqual(len(ed.history.done), 1)
        self.assertFalse(panel.btn_kenglish.isEnabled())
        self.assertIn("（这一行没有能拆的英文词）", panel.btn_kenglish.text())
        self.assertIn("（这一行没有能拆的英文词）", panel.btn_kenglish.toolTip())
        ed.history.undo()
        fake._refresh_k()
        self.assertTrue(panel.btn_kenglish.isEnabled())


if __name__ == "__main__":
    unittest.main()
