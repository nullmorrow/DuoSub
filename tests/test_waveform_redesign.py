"""S9: grouped waveform controls, persistent hints and shared K commands."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from tests import test_waveform_toolbar as toolbar
from tests import test_karaoke_ui as karaoke
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QFrame
from subkit_app.views.waveform import WaveformPanel, WaveformView


class ToolbarGroups(unittest.TestCase):
    setUpClass = toolbar.WaveformToolbar.__dict__["setUpClass"]
    setUp = toolbar.WaveformToolbar.setUp
    dispose = toolbar.WaveformToolbar.dispose

    def test_group_order_members_and_wrapping_geometry(self):
        p = self.panel
        expected = (("视图", (p.btn_waveform, p.btn_spectrum, p.zoom, p.lbl_span, p.btn_fit)),
                    ("播放", (p.speed_label, p.speed, p.btn_step)),
                    ("打轴", (p.btn_hint_lines, p.btn_adopt_time)), ("", (p.btn_settings,)))
        for group, (title, members) in zip(p.wave_groups, expected):
            labels = [w.text() for w in group.findChildren(QLabel)]
            if title:
                self.assertEqual(labels[0], title)
            for widget in members:
                self.assertIs(widget.parentWidget(), group)
        self.assertIn("缩放", [w.text() for w in p.wave_groups[0].findChildren(QLabel)])
        self.assertEqual(len([w for w in p.bar.findChildren(QFrame) if w.frameShape() == QFrame.Shape.VLine]), 3)
        for width in (600, 900, 1280):
            p.resize(width, 900)
            p.show()
            self.app.processEvents()
            self.assertEqual(p.width(), width)
            for group in p.wave_groups:
                self.assertLessEqual(group.geometry().right(), p.bar.width())
                centers = []
                for i in range(group.layout().count()):
                    widget = group.layout().itemAt(i).widget()
                    if not widget.isHidden():
                        self.assertGreaterEqual(widget.width(), widget.minimumSizeHint().width())
                        self.assertGreaterEqual(widget.x(), 0)
                        self.assertLessEqual(widget.geometry().right(), group.width())
                        centers.append(widget.geometry().center().y())
                self.assertLessEqual(max(centers) - min(centers), 1)
            self.assertEqual(p.btn_hint.geometry().right(), p.bar.width() - 7)
        p.resize(600, 900)
        self.app.processEvents()
        self.assertGreater(len({g.y() for g in p.wave_groups}), 1)

    def test_both_hint_buttons_default_persist_and_leave_legend_alone(self):
        p = self.panel
        with tempfile.TemporaryDirectory(prefix="s9_hints_") as tmp:
            path = str(Path(tmp) / "settings.ini")
            settings = QSettings(path, QSettings.Format.IniFormat)
            p.set_settings(settings)
            p.show_k(self.kview, has_tags=True)
            p.show()
            self.app.processEvents()
            legend = p.legend.isHidden()
            for key, button, hint in (("wave/hint_visible", p.btn_hint, p.hint),
                                      ("wave/khint_visible", p.btn_khint, p.khint)):
                self.assertTrue(button.isChecked())
                self.assertFalse(hint.isHidden())
                button.click()
                self.assertTrue(hint.isHidden())
                self.assertFalse(settings.value(key, type=bool))
                self.assertEqual(p.legend.isHidden(), legend)
            settings.sync()
            restored = WaveformPanel(WaveformView())
            try:
                restored.set_settings(QSettings(path, QSettings.Format.IniFormat))
                self.assertTrue(restored.hint.isHidden())
                self.assertTrue(restored.khint.isHidden())
                p.btn_hint.click()
                self.assertFalse(p.hint.isHidden())
                self.assertTrue(p.khint.isHidden())
            finally:
                restored.close()
                restored.deleteLater()
                self.app.processEvents()

    def test_hint_tooltips_keep_exact_legacy_text_in_each_mode(self):
        p = self.panel
        suffix = "  点刻度=跳转  PgUp/PgDn=上一组/下一组  粉线=画面切换"
        for mode, text in (("aegisub", "左键=开始  右键=结束  点上下色条=选中那一组"),
                           ("select", "左键点块=选中  Shift+左键=开始  Shift+右键=结束  Alt=不吸附")):
            p.set_timing_mode(mode)
            self.assertEqual(p.btn_hint.toolTip(), text + suffix)
            self.assertEqual(p.hint.text(), text + suffix)
        p.set_karaoke_mode(True)
        self.assertEqual(p.btn_hint.toolTip(), "左键=选中/跳转  右键=播放这一拍  拖边界=调拍  Alt=不吸附")
        self.assertEqual(p.btn_khint.toolTip(), p.khint.text())
        self.assertEqual(p.btn_hint.help_anchor, "timing")
        self.assertEqual(p.btn_khint.help_anchor, "karaoke")


class KBarActions(unittest.TestCase):
    setUpClass = karaoke.KaraokeUI.__dict__["setUpClass"]
    setUp = karaoke.KaraokeUI.setUp
    dispose = karaoke.KaraokeUI.dispose
    line = karaoke.KaraokeUI.line

    def test_shared_actions_once_and_immediate_enabled_state(self):
        self.line()
        w, p = self.window, self.window.wave_panel
        p.kAction.disconnect()
        called = Mock()
        p.kAction.connect(called)
        for name, button, value in (("tap", p.btn_k, "start"), ("import", p.btn_kimport, "import"),
                                    ("adopt", p.btn_kadopt, "adopt"), ("clear", p.btn_kclear, "clear"),
                                    ("confirm", p.btn_kconfirm, "confirm"), ("add_rest", p.btn_kaddrest, "add_rest"),
                                    ("remove_rest", p.btn_kdelrest, "remove_rest"),
                                    ("mapping", p.btn_kmap, "mapping"), ("english", p.btn_kenglish, "english")):
            action = w.action_hub.action("k." + name)
            self.assertIs(button.defaultAction(), action)
            button.setVisible(True)
            for enabled in (False, True):
                button.setEnabled(enabled)
                self.assertEqual(action.isEnabled(), enabled)
                action.setEnabled(not enabled)
                self.assertEqual(button.isEnabled(), not enabled, name)
            action.setEnabled(True)
            called.reset_mock()
            button.click()
            called.assert_called_once_with(value)
            called.reset_mock()
            action.trigger()
            called.assert_called_once_with(value)
            self.assertNotIn("\t", button.text())
        self.assertIs(p.btn_kparams.defaultAction(), w.action_hub.action("k.parameters"))
        for scope, action in p.kauto_scopes.items():
            self.assertIs(action, w.action_hub.action("k.auto_" + scope))
        for kind, actions in p.scope_actions.items():
            for scope, action in actions.items():
                self.assertIs(action, w.action_hub.action(f"k.{kind}_{scope}"))
        self.assertIs(p.k_controls.layout().itemAt(0).widget(), p.btn_karaoke)

    def test_karaoke_button_menu_and_real_f9_share_check_state(self):
        self.line()
        w, p = self.window, self.window.wave_panel
        action = w.action_hub.action("view.karaoke")
        self.assertIs(p.btn_karaoke.defaultAction(), action)
        self.assertIs(w.act_karaoke, action)
        w.show()
        w.activateWindow()
        w.player.setFocus()
        self.app.processEvents()
        p.btn_karaoke.click()
        self.assertEqual(w.edit_layout, "karaoke")
        self.assertTrue(action.isChecked())
        action.trigger()
        self.assertNotEqual(w.edit_layout, "karaoke")
        self.assertFalse(p.btn_karaoke.isChecked())
        w.player.setFocus()
        self.app.processEvents()
        QTest.keyClick(w.player, Qt.Key.Key_F9)
        self.app.processEvents()
        self.assertEqual(w.edit_layout, "karaoke")
        self.assertTrue(action.isChecked())
        self.assertTrue(p.btn_karaoke.isChecked())
        p.btn_karaoke.click()
        self.assertFalse(action.isChecked())
