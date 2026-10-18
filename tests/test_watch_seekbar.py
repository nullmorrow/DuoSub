"""F11 progress line drawn on the mpv OSD (it used to raise on every draw: an unescaped brace in the format)."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from subkit.ass.text import tokenize
from subkit_app.watch import SeekBar


class FakePlayer:
    def __init__(self):
        self.osd = {"w": 1920, "h": 1080}
        self.duration, self.time = 1421.0, 600.0
        self.overlays = {}

    def overlay(self, key, text):
        self.overlays[key] = text

    def clear_overlay(self, key):
        self.overlays.pop(key, None)

    def seek(self, t, exact=False):
        self.time = t


class SeekBarDraw(unittest.TestCase):
    def test_hover_on_the_bottom_edge_draws_valid_ass(self):
        self.app = QApplication.instance() or QApplication([])
        player = FakePlayer()
        bar = SeekBar(player)
        bar.enabled = True
        bar.mouse_move(QPointF(900, 1079))                  # touching the bottom edge shows and draws it
        self.assertTrue(bar.visible)
        (text,) = player.overlays.values()
        lines = text.split("\n")
        self.assertEqual(len(lines), 6)                      # outline, track, played part, knob, clock, hover time
        for line in lines[:3]:
            self.assertTrue(line.startswith("{\\an7\\pos(0,0)\\bord0\\shad0\\1c&H"), line)
            self.assertIn("\\p1}m ", line)
        self.assertTrue(all(tok.kind != "text" or tok.raw.strip() for line in lines for tok in tokenize(line)))
        bar.mouse_move(QPointF(900, 1079), on_subtitle=True)   # over a subtitle: no redraw, no error
        bar.hide()
        self.assertEqual(player.overlays, {})


if __name__ == "__main__":
    unittest.main()
