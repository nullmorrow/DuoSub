"""Song lines share one time per group, and one fade for a whole song (S01E005 OP, 2026-10-14)."""
from __future__ import annotations

import os
import unittest
import unittest.mock
from pathlib import Path

from subkit.core import karaoke as kara
from subkit.core.export import ExportOptions, build_ass
from subkit.core.ops import Editor
from subkit.model.document import Document, Group, Member, Run
from tests import test_convert_ui as fixtures
from tests.test_karaoke_fx import ktv_fixture


def song(start=93720, ends=(96060, 95790), fade=r"{\fad(300,300)}"):
    """One song group like the fansub source: the translation ends 270 ms before the original."""
    doc = Document("E5", groups=[Group("g", "song", ["jp", "cn"])])
    for (mid, text, lane, lang), end in zip((("jp", "足りないもの", "original", "ja"),
                                             ("cn", "為了尋覓", "translation", "zh-Hant")), ends):
        runs = ([Run(fade, "tag", ins="t_" + mid, b=len(fade))] if fade else []) + [Run(text, ins="i_" + mid, b=len(text))]
        doc.members[mid] = Member(mid, "g", lang, lane, "song", start, end, style="OP." + mid.upper(), runs=runs)
    return doc, Editor(doc, {})


class TranslationTimes(unittest.TestCase):
    def test_translation_takes_the_original_time_in_one_undo_step(self):
        doc, ed = song()
        self.assertEqual(ed.align_song_translation_times(["g"]), 1)
        self.assertEqual([(m.start, m.end) for m in doc.members.values()], [(93720, 96060)] * 2)
        self.assertEqual(doc.members["cn"].extra["time_src"], {"end": "manual"})
        self.assertEqual(ed.align_song_translation_times(["g"]), 0)             # already aligned: no undo step
        ed.history.undo()
        self.assertEqual(doc.members["cn"].end, 95790)

    def test_longer_translation_is_shortened_but_beats_stay_inside(self):
        doc, ed = song(ends=(96060, 96500))
        self.assertEqual(ed.align_song_translation_times(["g"]), 1)
        self.assertEqual(doc.members["cn"].end, 96060)
        doc, ed = song(ends=(96060, 96500))
        beats = kara.derive(doc, doc.members["cn"])
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 93720 + i * 600, 93720 + (i + 1) * 600, "manual"   # last beat ends at 96120
        ed.set_karaoke("cn", beats)
        ed.align_song_translation_times(["g"])
        self.assertEqual([(m.start, m.end) for m in doc.members.values()], [(93720, 96120)] * 2)

    def test_dialogue_and_original_only_groups_are_left_alone(self):
        doc, ed = song()
        doc.groups[0].kind = "dialogue"
        self.assertEqual(ed.align_song_translation_times(["g"]), 0)
        doc, ed = song()
        doc.groups[0].members.remove("cn")
        del doc.members["cn"]
        self.assertEqual(ed.align_song_translation_times(["g", "missing"]), 0)

    def test_auto_karaoke_widens_the_group_to_one_shared_edge(self):
        # Writing back auto K timing used to widen only the line a beat stuck out of: the translation then
        # ended (and faded out) 80 ms before the original.
        doc, tpl = ktv_fixture(starts=(1000,))
        jp, cn = doc.members["JP_0"], doc.members["CN_0"]
        jp.end, cn.end = 2300, 2100
        ed = Editor(doc, {})
        n = len(kara.effective(doc, jp).beats)
        beats = [(1000 + i * 1200 // n, 1000 + (i + 1) * 1200 // n, 1.0, False) for i in range(n)]   # ends at 2200
        report = ed.apply_auto_karaoke([{"mid": "JP_0", "beats": beats}], translations=False, overwrite=True)
        self.assertIn("CN_0", report["widened"])
        self.assertEqual((cn.start, cn.end), (jp.start, jp.end))


class SongFade(unittest.TestCase):
    def test_fade_replaces_the_source_fade_and_goes_back(self):
        doc, ed = song()
        self.assertEqual(ed.song_fade(["g"]), ((300, 300), False))
        self.assertEqual(ed.set_song_fade(["g"], (150, 0)), 2)
        self.assertEqual(ed.song_fade(["g"]), ((150, 0), True))
        self.assertEqual(ed.set_song_fade(["g"], (150, 0)), 0)
        doc.members["jp"].overrides["raw"] = r"\blur2\fad(150,0)"
        self.assertEqual(ed.set_song_fade(["g"], None), 2)                     # other tags stay
        self.assertEqual(doc.members["jp"].overrides, {"raw": r"\blur2"})
        self.assertNotIn("raw", doc.members["cn"].overrides)
        self.assertEqual(ed.song_fade(["g"]), ((300, 300), False))
        ed.history.undo()
        self.assertEqual(doc.members["cn"].overrides["raw"], r"\fad(150,0)")

    def test_export_uses_one_fade_per_line(self):
        _doc, tpl = ktv_fixture()
        for source in (r"{\fad(300,300)}", r"{\fade(255,0,255,0,300,2000,2340)}", ""):
            with self.subTest(source=source):
                doc, ed = song(fade=source)
                ed.set_song_fade(["g"], (150, 150))
                ass, _report = build_ass(doc, {}, tpl, Path(os.environ["TEMP"]), ExportOptions(playres=(960, 540)))
                texts = [e.text for e in ass.events if e.style in ("OP.JP", "OP.CN")]   # song lines carry \kf per beat
                self.assertEqual(len(texts), 2)
                for text in texts:
                    self.assertTrue(text.startswith(r"{\fad(150,150)}"), text)
                    self.assertNotIn("300", text)


class LyricsMenuUI(unittest.TestCase):
    """「更多 ▾ → 歌词」: 翻译时间对齐原声 ▸ / 淡入淡出 ▸ on a whole song (offscreen Qt, player double)."""

    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.project.pinyin_mode = "off"
        for gid, start in (("a", 1000), ("b", 4000)):
            cn = self.group(gid, kind="song", start=start)
            jp = Member("jp_" + gid, gid, "ja", "original", "song", start, start + 2300, style="OP.JP",
                        runs=[Run(r"{\fad(300,300)}", "tag", ins="t_" + gid, b=14), Run("うた", ins="jp_" + gid, b=2)])
            cn.runs.insert(0, Run(r"{\fad(300,300)}", "tag", ins="u_" + gid, b=14))
            self.doc.members[jp.id] = jp
            self.doc.groups[-1].members.insert(0, jp.id)
        self.doc.structure_changed()
        self.window.model.reset()
        self.window.select_group("a")

    def test_menus_align_whole_song_and_set_fade(self):
        from PySide6.QtWidgets import QDialog
        from subkit_app.views.song_fade import SongFadeDialog
        w = self.window
        w._refresh_redistribute_menu()
        panel = w.wave_panel
        self.assertEqual([m.title() for m in panel.lyrics_menu.findChildren(type(panel.lyrics_menu))
                          if m.title() in ("翻译时间对齐原声", "淡入淡出")], ["翻译时间对齐原声", "淡入淡出"])
        self.assertTrue(panel.scope_actions["align_times"]["song"].isEnabled())
        self.assertEqual(panel.scope_actions["fade"]["song"].text(), "整首歌…")
        self.assertFalse(panel.scope_actions["fade"]["selected"].isEnabled())        # needs two selected songs
        w._k_action("align_times:song")
        self.assertEqual({(m.start, m.end) for m in self.doc.members.values() if m.group == "a"}, {(1000, 3300)})
        self.assertEqual({(m.start, m.end) for m in self.doc.members.values() if m.group == "b"}, {(4000, 6300)})
        self.assertIn("已对齐 2 行", w.statusBar().currentMessage())
        seen = {}

        def accept(dialog):
            seen["restore"] = dialog.restore_button.isEnabled()
            seen["values"] = (dialog.fade_in.value(), dialog.fade_out.value())
            dialog.fade_in.setValue(100)
            dialog.fade_out.setValue(0)
            dialog.apply_button.click()
            return QDialog.DialogCode.Accepted

        with unittest.mock.patch.object(SongFadeDialog, "exec", accept):
            w._k_action("fade:song")
        self.assertEqual(seen, {"restore": False, "values": (300, 300)})
        self.assertEqual({m.overrides.get("raw") for m in self.doc.members.values()}, {r"\fad(100,0)"})
        self.session.editor.history.undo()
        self.assertFalse(any(m.overrides for m in self.doc.members.values()))


if __name__ == "__main__":
    unittest.main()
