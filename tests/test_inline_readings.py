"""W4: imported caption hints survive formatting, merge and ruby generation."""
import copy
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subkit.core.cc import analyze_event
from subkit.core.importer import import_source
from subkit.core.merge import MergeSettings, merge_episode
from subkit.core.readings import annotate_document
from subkit.reading.dictionary import ReadingDict
from subkit.reading.engine import ReadingEngine
from subkit.sources.jp import extract_inline_readings

LINES = ['<font size="20">白き雪よ　黄昏を前に風（うず）を巻け</font>',
         '<font size="20">間もなく 焔（ひ）は放たれる</font>']


class InlineReadings(unittest.TestCase):
    def test_full_and_halfwidth_brackets_with_font_and_fullwidth_space(self):
        for raw, surface, reading in zip(LINES, ("風", "焔"), ("うず", "ひ")):
            for text in (raw, raw.replace("（", "(").replace("）", ")")):
                with self.subTest(text=text):
                    cleaned, hints = extract_inline_readings(text)
                    self.assertNotIn(reading, cleaned)
                    self.assertEqual([(h.surface, h.reading) for h in hints], [(surface, reading)])
                    result = analyze_event(0, text)
                    hints = [s for s in result.spans if s.kind == "reading_hint"]
                    self.assertEqual(len(hints), 1)
                    self.assertEqual(hints[0].value, reading)
                    self.assertEqual(text[slice(*hints[0].base)], surface)
                    self.assertFalse(any(s.kind == "sfx" for s in result.spans))

    def test_import_merge_annotations_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ja_path, cn_path = root / "ja.srt", root / "cn.srt"
            text = "\n\n".join(f"{i}\n00:00:0{i},000 --> 00:00:0{i},900\n{line}"
                                  for i, line in enumerate(LINES, 1)) + "\n"
            ja_path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
            cn_path.write_text("1\n00:00:01,000 --> 00:00:02,900\n翻译\n", encoding="utf-8")
            original = ja_path.read_bytes()
            try:
                ja = import_source(ja_path, src_id="o", lang="ja", lane="original", episode_dir=root / "ep")
                cn = import_source(cn_path, src_id="t", lang="zh-Hans", lane="translation", episode_dir=root / "ep")
                before = copy.deepcopy(ja.events_json())
                doc = merge_episode("E", cn, ja, MergeSettings()).document
                annotate_document(doc, {"o": ja, "t": cn}, ReadingEngine(ReadingDict()))
                annotations = {(a.base, a.text) for a in doc.annotations.values()}
                self.assertIn(("風", "うず"), annotations)
                self.assertIn(("焔", "ひ"), annotations)
                for m in doc.members.values():
                    if m.lane == "original":
                        self.assertNotIn("（", m.text)
                        self.assertNotIn("<font", m.text)
                        self.assertIn("風を巻け", m.text)   # no separator where the bracket was
                        self.assertIn("焔は放たれる", m.text)
                self.assertEqual(ja.events_json(), before)
                self.assertEqual(ja_path.read_bytes(), original)
                self.assertEqual((root / "ep" / ja.copy).read_bytes(), original)
            finally:
                for p in (root / "ep").rglob("*"):
                    if p.is_file():
                        os.chmod(p, stat.S_IREAD | stat.S_IWRITE)

    def test_katakana_gloss_and_descriptions_unchanged(self):
        text = "眷属（ファミリア）だ（笑い）"
        spans = analyze_event(0, text).spans
        self.assertFalse(any(s.kind == "reading_hint" for s in spans))
        self.assertEqual([(s.kind, text[s.a:s.b]) for s in spans], [("sfx", "（笑い）")])
        self.assertEqual([(s.kind, s.value) for s in analyze_event(0, "音（うずの音）").spans],
                         [("sfx", "うずの音")])
