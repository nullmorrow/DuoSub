"""S10 main toolbar, atomic K groups and the final offscreen layout matrix."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from tests import test_waveform_toolbar as waveform
from tests import test_editor_panel_redesign as editor
from tests import test_karaoke_ui as karaoke
from PySide6.QtCore import QTimer, Qt, qVersion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QLabel, QScrollBar, QToolButton
from subkit_app import mainwindow as mw


GROUPS = ((("file.save", "保存"), ("edit.undo", "撤销"), ("edit.redo", "重做")),
          (("play.toggle", "播放 / 暂停"), ("group.confirm_next", "✓ 确认并下一组")),
          (("view.watch_timing", "看片打轴"), ("view.watch_full", "全屏观看"),
           ("view.karaoke", "卡拉 OK")),
          (("view.source_panel", "原文"), ("view.people_panel", "人物")))


class MainToolbar(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose

    def test_order_separators_short_names_shared_actions_and_menu_text(self):
        w = self.window
        actions = w.toolbar.actions()
        self.assertEqual([w.toolbar.widgetForAction(a) for a in actions[:4]][1::2],
                         [w.ep_box, w.filter_box])
        self.assertEqual([w.toolbar.widgetForAction(actions[i]).text().strip() for i in (0, 2)],
                         ["集：", "列表："])
        expected = []
        for group in GROUPS:
            expected.append(None)
            expected.extend(action_id for action_id, _ in group)
        self.assertEqual([None if a.isSeparator() else a.property("action_id") for a in actions[4:]], expected)
        for group in GROUPS:
            for action_id, text in group:
                action = w.action_hub.action(action_id)
                button = w.toolbar.widgetForAction(action)
                self.assertIs(button.defaultAction(), action)
                self.assertEqual(button.text(), text)
                self.assertEqual(button.toolButtonStyle(), Qt.ToolButtonStyle.ToolButtonTextOnly)
                self.assertEqual(action.text(), w.action_hub.registry.lookup(action_id).text)
                # A later enable/state update must not restore the long menu label.
                action.setEnabled(not action.isEnabled())
                self.assertEqual(button.text(), text)
                self.assertEqual(action.isCheckable(), action_id.startswith("view."))

    def test_toolbar_and_menu_dispatch_once(self):
        direct = {"file.save": "save", "edit.undo": "undo", "edit.redo": "redo",
                  "play.toggle": "toggle_play", "group.confirm_next": "confirm_next",
                  "view.karaoke": "toggle_karaoke_mode"}
        with ExitStack() as stack:
            spies = {key: stack.enter_context(patch.object(mw.MainWindow, method)) for key, method in direct.items()}
            watch = stack.enter_context(patch.object(mw.MainWindow, "_watch_key"))
            w = mw.MainWindow()
            try:
                for action_id, spy in spies.items():
                    action = w.action_hub.action(action_id)
                    action.setEnabled(True)
                    for trigger in (w.toolbar.widgetForAction(action).click, action.trigger):
                        spy.reset_mock()
                        trigger()
                        spy.assert_called_once_with()
                for action_id, pinned in (("view.watch_timing", True), ("view.watch_full", False)):
                    action = w.action_hub.action(action_id)
                    for trigger in (w.toolbar.widgetForAction(action).click, action.trigger):
                        watch.reset_mock()
                        trigger()
                        watch.assert_called_once_with(pinned)
            finally:
                w.close()
                w.deleteLater()
                self.app.processEvents()

    def test_current_tooltips_rebinding_clear_and_menu_names_unchanged(self):
        w = self.window
        menu_names = {action_id: w.action_hub.action(action_id).text() for group in GROUPS for action_id, _ in group}
        for group in GROUPS:
            for action_id, text in group:
                keys = w.action_hub.registry.effective_keys(action_id, "main")
                self.assertEqual(w.toolbar.widgetForAction(w.action_hub.action(action_id)).toolTip(),
                                 text + ("（" + " / ".join(keys) + "）" if keys else ""))
        edited = w.action_hub.registry.copy()
        edited.set_keys("file.save", "main", "Ctrl+Alt+S")
        w._apply_shortcut_overrides(edited)
        button = w.toolbar.widgetForAction(w.action_hub.action("file.save"))
        self.assertEqual(button.toolTip(), "保存（Ctrl+Alt+S）")
        edited.set_keys("file.save", "main", "")
        w._apply_shortcut_overrides(edited)
        self.assertEqual(button.toolTip(), "保存")
        self.assertEqual(menu_names, {key: w.action_hub.action(key).text() for key in menu_names})

    def test_short_names_and_tooltips_survive_action_changes(self):
        # Regression (user screenshot, 0.33.5): after F11 the button showed the full menu text.
        w = self.window
        for action_id, (button, text) in w._toolbar_buttons.items():
            with self.subTest(action=action_id):
                action = w.action_hub.action(action_id)
                menu_text = action.text()
                action.setStatusTip("changed")         # any change makes the button re-read its action
                action.setToolTip("changed")
                if action.isCheckable():
                    action.setChecked(not action.isChecked())
                self.assertEqual(button.text(), text)
                self.assertTrue(button.toolTip().startswith(text), button.toolTip())
                self.assertEqual(action.text(), menu_text)

    def test_real_mode_keys_and_dock_toggles_sync_buttons(self):
        w = self.window
        w.player._hide_timer = QTimer(w.player)
        w.player.pause = Mock()
        self.project.settings = Mock(return_value={})
        w.show()
        w.activateWindow()
        self.app.processEvents()
        for action_id, key in (("view.watch_timing", Qt.Key.Key_F10),
                               ("view.watch_full", Qt.Key.Key_F11), ("view.karaoke", Qt.Key.Key_F9)):
            action = w.action_hub.action(action_id)
            button = w.toolbar.widgetForAction(action)
            for expected in (True, False):
                w.player.setFocus()
                self.app.processEvents()
                QTest.keyClick(w.player, key)
                self.app.processEvents()
                self.assertEqual(action.isChecked(), expected, action_id)
                self.assertEqual(button.isChecked(), expected, action_id)
            button.click()
            self.app.processEvents()
            self.assertTrue(action.isChecked())
            action.trigger()
            self.app.processEvents()
            self.assertFalse(button.isChecked())
        for action_id, dock in (("view.source_panel", w.dock_source), ("view.people_panel", w.people)):
            action = w.action_hub.action(action_id)
            button = w.toolbar.widgetForAction(action)
            for trigger in (button.click, action.trigger):
                old = dock.isVisible()
                trigger()
                self.app.processEvents()
                self.assertEqual(dock.isVisible(), not old)
                self.assertEqual(button.isChecked(), dock.isVisible())


class KGroups(unittest.TestCase):
    setUpClass = waveform.WaveformToolbar.__dict__["setUpClass"]
    setUp = waveform.WaveformToolbar.setUp
    dispose = waveform.WaveformToolbar.dispose

    def test_whole_groups_at_several_widths_and_empty_titles_hidden(self):
        p = self.panel
        p.show_k(self.kview, has_tags=True)
        for width in (600, 760, 900, 1280):
            p.resize(width, 1000)
            p.show()
            self.app.processEvents()
            self.assertEqual(p.width(), width)
            self.assertEqual(len(p.k_groups), 4)
            for group in p.k_groups:
                self.assertLessEqual(group.geometry().right(), p.k_controls.width())
                widgets = [group.layout().itemAt(i).widget() for i in range(group.layout().count())]
                widgets = [widget for widget in widgets if not widget.isHidden()]
                centers = [widget.geometry().center().y() for widget in widgets]
                self.assertLessEqual(max(centers) - min(centers), 1)
                for widget in widgets:
                    self.assertGreaterEqual(widget.width(), widget.minimumSizeHint().width())
                    self.assertLessEqual(widget.geometry().right(), group.width())
            self.assertIs(p.btn_kaddrest.parentWidget(), p.btn_kdelrest.parentWidget())
        for groups in (p.wave_groups[:3], p.k_groups):
            for group in groups:
                controls = [group.layout().itemAt(i).widget() for i in range(group.layout().count())]
                controls = [c for c in controls if isinstance(c, (QAbstractButton,)) or c is p.k_unit]
                if not controls:
                    continue
                for control in controls:
                    control.hide()
                # Wave view/play groups also have sliders/combos, so test all controls.
                for i in range(group.layout().count()):
                    control = group.layout().itemAt(i).widget()
                    if not isinstance(control, QLabel):
                        control.hide()
                self.app.processEvents()
                self.assertTrue(group.isHidden(), [type(c).__name__ for c in controls])
                controls[0].show()
                self.app.processEvents()
                self.assertFalse(group.isHidden())


class LayoutMatrix(unittest.TestCase):
    setUpClass = editor.EditorPanel.__dict__["setUpClass"]
    setUp = editor.EditorPanel.setUp
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    line = karaoke.KaraokeUI.line

    def test_editor_waveform_and_kbar_matrix(self):
        w = self.window
        self.wave = w.wave
        self.line(gid="song")
        results = []
        for width, height in ((1600, 1000), (1280, 800)):
            for layout in ("video", "list", "columns", "karaoke"):
                for gid in (("current", "song") if layout == "karaoke" else ("current",)):
                    with self.subTest(size=(width, height), layout=layout, group=gid):
                        w.set_edit_layout(layout)
                        w.select_group(gid, restore=True)
                        for section in w.editor.sections.values():
                            section.expanded = True
                            section._refresh()
                        w.editor._update_content_height()
                        w.resize(width, height)
                        w.show()
                        self.app.processEvents()
                        self.app.processEvents()
                        self.assertEqual((w.width(), w.height()), (width, height))
                        self.assertEqual(w.editor.group.id, gid)
                        checked = []
                        for panel in (w.editor, w.wave_panel.bar, w.wave_panel.k_controls):
                            for button in panel.findChildren(QAbstractButton):
                                if not button.isVisible() or not button.text():
                                    continue
                                text_width = button.fontMetrics().horizontalAdvance(button.text().replace("&", ""))
                                self.assertLessEqual(text_width, button.contentsRect().width(), button.text())
                                if panel is not w.editor:
                                    self.assertGreaterEqual(button.width(), button.sizeHint().width(), button.text())
                                parent = button.parentWidget()
                                while parent is not panel:
                                    self.assertGreaterEqual(button.mapTo(parent, button.rect().topLeft()).x(), 0, button.text())
                                    self.assertLessEqual(button.mapTo(parent, button.rect().topRight()).x(), parent.width(), button.text())
                                    parent = parent.parentWidget()
                                self.assertLessEqual(button.mapTo(panel, button.rect().topRight()).x(), panel.width(), button.text())
                                checked.append(button.text())
                            for scrollbar in panel.findChildren(QScrollBar):
                                if scrollbar.orientation() == Qt.Orientation.Horizontal:
                                    self.assertFalse(scrollbar.isVisible())
                        if gid == "song":
                            self.assertTrue(w.wave_panel.kbar.isVisible())
                            self.assertIn("后面加空拍", checked)
                        results.append({"size": [width, height], "layout": layout, "group": gid,
                                        "buttons": checked, "editor_width": w.editor.width(),
                                        "wave_width": w.wave_panel.bar.width()})
        path = Path(os.environ["TEMP"]) / "s10-layout-matrix.json"
        path.write_text(json.dumps({"qt": qVersion(), "platform": self.app.platformName(),
                                   "player": "test double", "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
