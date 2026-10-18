"""Pure token conversion, particle pronunciation, and lazy dependency discovery."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.kalign import available  # noqa: E402
from subkit.kalign.tokens import (kana_tokens, kata_to_hira, mms_tokens, particle_positions,
                                  romaji, sung_for_alignment)  # noqa: E402


class Tokens(unittest.TestCase):
    def test_base_kana(self):
        for kana, expected in (("あいうえお", "aiueo"), ("かきくけこ", "kakikukeko"),
                               ("しちつふじぢづ", "shichitsufujijizu"),
                               ("をゐゑんゔ", "oienvu"), ("がぎぐげご", "gagigugego")):
            with self.subTest(kana=kana):
                self.assertEqual(romaji(kana), expected)

    def test_yoon(self):
        for kana, expected in (("きゃきゅきょ", "kyakyukyo"), ("しゃしゅしょ", "shashusho"),
                               ("ちゃちゅちょ", "chachucho"), ("じゃじゅじょ", "jajujo"),
                               ("りゃにゅびょ", "ryanyubyo")):
            with self.subTest(kana=kana):
                self.assertEqual(romaji(kana), expected)

    def test_small_vowels_and_reference_spellings(self):
        for kana, expected in (("ふぁふぃふぇふぉ", "fafifefo"), ("てぃでぃとぅどぅ", "tiditudu"),
                               ("しぇちぇじぇ", "shecheje"), ("わぃわぇわぉ", "wiwewo"),
                               ("うぃうぇうぉ", "ieo"), ("ヴァヴィヴェヴォ", "vavivevo")):
            with self.subTest(kana=kana):
                self.assertEqual(romaji(kana), expected)

    def test_katakana_and_standalone_small_kana(self):
        self.assertEqual(romaji("キャットシェーン"), romaji("きゃっとしぇーん"))
        for kana, expected in zip("ぁぃぅぇぉゃゅょゎ", ("a", "i", "u", "e", "o", "ya", "yu", "yo", "wa")):
            with self.subTest(kana=kana):
                self.assertEqual(romaji(kana), expected)
        self.assertEqual(kata_to_hira("カナ漢字|ーABC"), "かな漢字|ーABC")

    def test_romaji_prototype_parity(self):
        spec = importlib.util.spec_from_file_location("romaji_prototype", ROOT / "tools/experiments/kalign/kana_romaji.py")
        prototype = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prototype)
        for base in prototype.BASE:
            for following in ("", "ゃ", "ゅ", "ょ", "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "っ", "ー", "ABC", "漢字"):
                kana = base + following
                with self.subTest(kana=kana):
                    self.assertEqual(romaji(kana), prototype.romaji(kana))

    def test_sokuon_is_silent(self):
        self.assertEqual(romaji("だっきっと"), "dakito")
        self.assertEqual(romaji("っッっ"), "")

    def test_long_mark_uses_last_vowel(self):
        self.assertEqual(romaji("ローンーー"), "roonoo")
        self.assertEqual(romaji("ーんー"), "n")
        self.assertEqual(romaji("AbCーー"), "abcaa")

    def test_ascii_and_unread_text(self):
        self.assertEqual(romaji("ABC-z 123 愛、é！"), "abcz")
        self.assertEqual(romaji("漢字！？"), "")
        self.assertEqual(romaji(""), "")

    def test_mms_tokens_keep_letter_order_and_repeats(self):
        vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", 1)}
        self.assertEqual(mms_tokens("キャーLove", vocab), [vocab[c] for c in "kyaalove"])
        self.assertEqual(mms_tokens("っ漢字", vocab), [])
        with self.assertRaises(KeyError):
            mms_tokens("か", {"a": 1})

    def test_kana_tokens_convert_filter_and_keep_sokuon(self):
        vocab = {"か": 6, "な": 2, "っ": 9, "|": 4, "ー": 5, "漢": 10, "A": 11}
        self.assertEqual(kana_tokens("カ|ナーッ漢字ABC", vocab), [6, 2, 9])
        self.assertEqual(kana_tokens("かか", vocab), [6, 6])
        self.assertEqual(kana_tokens("漢字ー|", vocab), [])

    def test_particle_pronunciation_never_changes_explicit_ruby(self):
        for sung, corrected in (("は", "わ"), ("へ", "え")):
            self.assertEqual(sung_for_alignment(sung, True, False), corrected)
            for particle, ruby in ((False, False), (False, True), (True, True)):
                self.assertEqual(sung_for_alignment(sung, particle, ruby), sung)
        self.assertEqual(sung_for_alignment("はな", True, False), "はな")

    def test_particle_offsets_full_dictionary_mode_c_and_close(self):
        def morph(surface, start, pos):
            return SimpleNamespace(surface=lambda: surface, begin=lambda: start,
                                   part_of_speech=lambda: (pos,))
        tokenizer = Mock()
        tokenizer.tokenize.return_value = [morph("は", 1, "助詞"), morph("へ", 4, "助詞"),
                                          morph("は", 7, "名詞"), morph("が", 8, "助詞")]
        dictionary = Mock()
        dictionary.create.return_value = tokenizer
        fake = SimpleNamespace(Dictionary=Mock(return_value=dictionary), SplitMode=SimpleNamespace(C=object()))
        with patch.dict(sys.modules, {"sudachipy": fake}):
            self.assertEqual(particle_positions("彼はへやへ行く"), {1, 4})
        fake.Dictionary.assert_called_once_with(dict="full")
        tokenizer.tokenize.assert_called_once_with("彼はへやへ行く", fake.SplitMode.C)
        dictionary.close.assert_called_once_with()

    def test_particle_dictionary_closed_on_tokenizer_failure(self):
        dictionary = Mock()
        dictionary.create.return_value.tokenize.side_effect = RuntimeError("tokenizer failed")
        fake = SimpleNamespace(Dictionary=Mock(return_value=dictionary), SplitMode=SimpleNamespace(C=0))
        with patch.dict(sys.modules, {"sudachipy": fake}), self.assertRaisesRegex(RuntimeError, "tokenizer failed"):
            particle_positions("は")
        dictionary.close.assert_called_once_with()

    def test_missing_sudachi_returns_empty(self):
        with patch.dict(sys.modules, {"sudachipy": None}):
            self.assertEqual(particle_positions("はへ"), set())

    @unittest.skipUnless(importlib.util.find_spec("sudachipy"), "SudachiPy is not installed")
    def test_real_particle_visible_offsets(self):
        self.assertEqual(particle_positions("彼はへやへ行く。　ははは"), {1, 4})
        self.assertEqual(particle_positions("猫は　空へ"), {1, 4})

    def test_availability_uses_discovery_only(self):
        with patch("subkit.kalign.find_spec", side_effect=lambda name: None if name in ("torch", "demucs") else object()) as probe:
            self.assertEqual(available(), (False, ["torch", "demucs"]))
        self.assertEqual([call.args[0] for call in probe.call_args_list],
                         ["torch", "torchaudio", "transformers", "demucs", "soundfile"])
        with patch("subkit.kalign.find_spec", return_value=object()):
            self.assertEqual(available(), (True, []))
        with patch("subkit.kalign.find_spec", side_effect=ImportError("unavailable")):
            self.assertEqual(len(available()[1]), 5)

    def test_fresh_import_and_probe_do_not_import_heavy_modules(self):
        code = "\n".join(("import sys", f"sys.path.insert(0, {str(ROOT / 'src')!r})",
                          "from subkit.kalign import available",
                          "from subkit.kalign import tokens, viterbi",
                          "available()",
                          "heavy = ('torch', 'torchaudio', 'transformers', 'demucs', 'sudachipy', 'soundfile', 'PySide6')",
                          "assert not any(n == p or n.startswith(p + '.') for n in sys.modules for p in heavy)"))
        proc = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code],
                              capture_output=True, text=True, encoding="utf-8", timeout=20,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()


class ContinuationMorae(unittest.TestCase):
    def test_spec_table_and_punctuation(self):
        from subkit.kalign.tokens import continuation_morae
        for a, b in [("こ", "う"), ("そ", "う"), ("ゆ", "う"), ("く", "う"), ("せ", "い"),
                     ("き", "い"), ("きょ", "う"), ("あ", "ぁ"), ("い", "ぃ"), ("う", "ぅ"),
                     ("え", "ぇ"), ("お", "ぉ"), ("ロ", "ー"), ("こ", "う…")]:
            with self.subTest(a=a, b=b):
                self.assertEqual(continuation_morae([a, b]), [False, True])
        for a, b in [("か", "う"), ("ん", "う"), ("っ", "ー"), ("こ", "ろー"), ("こ", "うた"), ("こ", "漢う")]:
            with self.subTest(a=a, b=b):
                self.assertEqual(continuation_morae([a, b]), [False, False])
        self.assertEqual(continuation_morae(["う", "う", "ー", "う"]), [False, True, True, True])
        self.assertEqual(continuation_morae([]), [])

    def test_word_start_is_never_a_continuation(self):
        from subkit.kalign.tokens import continuation_morae, word_starts
        # な｜い｜い(今)｜ま: the second い starts 今, after a double space.
        spans = [(5, False), (6, False), (9, True), (9, True)]
        self.assertEqual(word_starts(spans, "瞳には嘘のない  今"), [False, False, True, False])
        self.assertEqual(word_starts([(5, False), (6, False), (7, False)], "瞳には嘘のないい"), [False] * 3)
        self.assertEqual(word_starts([(0, True), (0, True)], "綺麗"), [False, False])   # inside one reading
        sung = ["な", "い", "い", "ま"]
        self.assertEqual(continuation_morae(sung), [False, False, True, False])
        self.assertEqual(continuation_morae(sung, word_starts(spans, "瞳には嘘のない  今")), [False] * 4)
        self.assertEqual(continuation_morae(["を", "う"], [False, True]), [False, False])   # を｜生(う)め
