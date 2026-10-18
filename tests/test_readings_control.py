"""Furigana control: merge words into one reading (迷宮都市), cut a word, episode/project rules, the popup's range.

Run from the repository root:  python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_stage0 import CN_ASS, _write  # noqa: E402

from subkit.core.importer import import_source  # noqa: E402
from subkit.core.merge import MergeSettings, merge_episode  # noqa: E402
from subkit.core.ops import Editor  # noqa: E402
from subkit.core.readings import anchor_positions, reading_spans, visible_map  # noqa: E402
from subkit.model.document import Annotation  # noqa: E402
from subkit.reading.dictionary import ReadingDict  # noqa: E402
from subkit.reading.engine import ReadingEngine  # noqa: E402

SRT = """1
00:00:01,000 --> 00:00:03,000
ここは迷宮都市

2
00:00:05,000 --> 00:00:07,000
迷宮都市の朝
"""


class ReadingControl(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        d = self.tmp / "orig"
        d.mkdir()
        self.ep_dir = self.tmp / "episode"
        cn = import_source(_write(d, "c.ass", CN_ASS), src_id="t1", lang="zh-Hans", lane="translation",
                           episode_dir=self.ep_dir, cn_cfg={"dialogue": ["Default"]})
        self.ja = import_source(_write(d, "j.srt", SRT), src_id="o1", lang="ja", lane="original", episode_dir=self.ep_dir)
        self.sources = {"t1": cn, "o1": self.ja}
        self.doc = merge_episode("R", cn, self.ja, MergeSettings()).document
        self.ed = Editor(self.doc, self.sources)
        self.engine = ReadingEngine()
        self.m = next(m for m in self.doc.members.values() if m.lane == "original" and "ここは" in m.text)
        # the situation in the user's screenshot: 迷宮 has a reading, 都市 none
        for a in [a for a in self.doc.annotations.values() if a.member == self.m.id]:
            del self.doc.annotations[a.id]
        r = next(r for r in self.m.runs if r.src)
        self.doc.annotations["a_mk"] = Annotation("a_mk", "ruby", self.m.id, [{"src": r.src, "e": r.e, "a": r.a + 3, "b": r.a + 5}],
                                                  "迷宮", "めいきゅう", status="confirmed", parts=[[0, 2, "めいきゅう"]])

    def tearDown(self) -> None:
        for p in self.ep_dir.rglob("*"):
            if p.is_file():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _anns(self, m=None):
        m = m or self.m
        _vt, ids = visible_map(m)
        return sorted((anchor_positions(a, ids), a.base, a.text, a.parts) for a in self.doc.annotations.values() if a.member == m.id)

    # the words of a line, with readings for unannotated kanji from the analyzer: 迷宮 + 都市 → めいきゅう + とし
    def test_spans_prefill(self) -> None:
        spans = reading_spans(self.doc, self.m, self.engine)
        by = {(s["a"], s["b"]): s for s in spans}
        self.assertEqual(by[(3, 5)]["reading"], "めいきゅう")
        self.assertEqual(by[(3, 5)]["ann"], "a_mk")
        self.assertEqual(by[(5, 7)]["reading"], "とし")
        self.assertEqual("".join(s["reading"] for s in spans if 3 <= s["a"] and s["b"] <= 7), "めいきゅうとし")

    # one reading over 迷宮都市: one annotation, one ruby piece over all four kanji
    def test_merge_into_one_word(self) -> None:
        self.ed.add_annotation(self.m.id, 3, 7, "めいきゅうとし", engine=self.engine, source=self.ja)
        self.assertEqual(self._anns(), [((3, 7), "迷宮都市", "めいきゅうとし", [[0, 4, "めいきゅうとし"]])])
        self.ed.history.undo()
        self.assertEqual(self._anns()[0][1:3], ("迷宮", "めいきゅう"))

    # a reading for part of a word: the rest of the old word gets a fresh suggestion instead of losing its ruby
    def test_cut_word_rest_resuggested(self) -> None:
        self.ed.add_annotation(self.m.id, 3, 7, "めいきゅうとし")
        self.ed.add_annotation(self.m.id, 5, 7, "とし", engine=self.engine, source=self.ja)
        anns = self._anns()
        self.assertEqual([x[1] for x in anns], ["迷宮", "都市"])
        self.assertEqual(anns[0][2], "めいきゅう")

    # an episode rule: every 迷宮都市 in the episode gets one reading; confirmed readings stay; undo removes it all
    def test_episode_rule(self) -> None:
        from subkit.core.workspace import Episode

        other = next(m for m in self.doc.members.values() if m.lane == "original" and "の朝" in m.text)
        self.ed.add_reading_rule("迷宮都市", "めいきゅうとし")
        self.assertEqual(self.doc.rules, [{"kind": "reading", "surface": "迷宮都市", "reading": "めいきゅうとし"}])
        eng = Episode.reading_engine(_FakeEpisode(self.doc))
        self.ed.reannotate([other.id, self.m.id], eng, self.sources)
        self.assertIn(((0, 4), "迷宮都市", "めいきゅうとし", [[0, 4, "めいきゅうとし"]]), self._anns(other))
        self.assertIn(((3, 5), "迷宮", "めいきゅう", [[0, 2, "めいきゅう"]]), self._anns())     # confirmed: kept
        self.ed.history.undo()
        self.ed.history.undo()
        self.assertEqual(self.doc.rules, [])
        self.ed.history.redo()
        self.ed.remove_reading_rule("迷宮都市")
        self.assertEqual(self.doc.rules, [])

    def test_project_dictionary_word_removed(self) -> None:
        p = self.tmp / "readings.json"
        ReadingDict().add_word(p, "迷宮都市", "めいきゅうとし")
        self.assertIn("迷宮都市", json.loads(p.read_text(encoding="utf-8"))["words"])
        self.assertTrue(ReadingDict.remove_word(p, "迷宮都市"))
        self.assertNotIn("迷宮都市", json.loads(p.read_text(encoding="utf-8"))["words"])

    # the popup: ◀ ▶ change the range, the reading is pre-filled, choosing emits the new range and the scope
    def test_popup_range(self) -> None:
        from PySide6.QtWidgets import QApplication

        from subkit_app.views.reading_popup import ReadingPopup

        app = QApplication.instance() or QApplication([])
        scope = {"text": "ここは迷宮都市", "a": 3, "b": 5, "spans": reading_spans(self.doc, self.m, self.engine)}
        pop = ReadingPopup("迷宮", "めいきゅう", ["めいきゅう"], "manual", "confirmed", None, scope=scope)
        got = []
        pop.chosenRange.connect(lambda a, b, t, sc: got.append((a, b, t, sc)))
        pop._extend(+1)
        self.assertEqual(pop.base, "迷宮都市")
        self.assertEqual(pop.edit.text(), "めいきゅうとし")
        pop.scope_group.button(1).setChecked(True)
        pop._choose(pop._converted())
        self.assertEqual(got, [(3, 7, "めいきゅうとし", 1)])
        app.processEvents()

    # popups stay on screen: a point at the bottom edge puts the popup above it
    def test_popup_stays_on_screen(self) -> None:
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QApplication, QFrame

        from subkit_app.views.placement import place_near, screen_area

        QApplication.instance() or QApplication([])
        w = QFrame()
        w.resize(400, 200)
        area = screen_area(QPoint(10, 10), w)
        place_near(w, QPoint(area.right() - 5, area.bottom() - 5))
        self.assertGreaterEqual(w.x(), area.left())
        self.assertLessEqual(w.x() + w.width(), area.right() + 1)
        self.assertLessEqual(w.y() + w.height(), area.bottom() + 1)


class _FakeEpisode:
    """Just enough of an Episode for ``Episode.reading_engine`` (project dictionaries = none)."""

    def __init__(self, doc) -> None:
        self.document = doc
        self.project = _FakeProject()
        self.id = "S01E001"
        self.data = {"season": "S01"}


class _FakeProject:
    def reading_engine(self, episode=None, season=None):     # scopes of self-reference rules
        return ReadingEngine(ReadingDict())


if __name__ == "__main__":
    unittest.main()
