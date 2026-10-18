"""Q2 quick colours: real Qt widgets, project JSON and editor undo history."""
import copy
import unittest
from unittest.mock import patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QTextCursor
from PySide6.QtTest import QTest
from subkit.core.workspace import DEFAULT_QUICK_COLORS, Project
from subkit_app.views.quick_colours import QuickColours, QuickColourModel


class QuickColourTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select

    def setUp(self):
        fixtures.ConversionUI.setUp(self)
        self.stored = Project.create(self.path / "project", "颜色")
        self.other = Project.create(self.path / "other", "另一个")
        self.model = self.window.quick_colours
        self.model.set_project(self.stored)
        self.member = self.group("g", text="三个字")
        self.select("g")
        self.bar = self.window.editor.quick_colours
        self.second = QuickColours(lambda: QColor("#123456"))
        self.second.set_model(self.model)
        self.addCleanup(self.second.deleteLater)
        self.window.show()
        self.app.processEvents()

    def test_defaults_project_switch_empty_and_no_project(self):
        self.assertEqual(self.model.colours, list(DEFAULT_QUICK_COLORS))
        self.model.save(["#123456"])
        self.assertEqual(Project.open(self.stored.root).quick_colors, ["#123456"])
        self.assertEqual(len(self.second.buttons), 1)
        self.model.set_project(self.other)
        self.assertEqual(self.model.colours, list(DEFAULT_QUICK_COLORS))
        self.model.save([])
        self.model.set_project(Project.open(self.other.root))
        self.assertEqual(self.model.colours, [])
        self.assertTrue(self.bar.plus.isEnabled())
        self.model.set_project(self.stored)
        self.assertEqual(self.model.colours, ["#123456"])
        self.model.set_project(None)
        self.assertFalse(self.bar.isEnabled())
        self.assertFalse(self.second.isEnabled())

    def test_add_current_colour_cancel_edit_remove_move_persist_and_refresh(self):
        self.window.editor.fmt._colour = "&H563412&"
        with patch("subkit_app.views.quick_colours.QColorDialog.getColor", return_value=QColor("#123456")) as pick:
            self.bar.plus.click()
            self.assertEqual(pick.call_args.args[0].name(), "#123456")
        self.assertEqual(self.model.colours[-1], "#123456")
        with patch("subkit_app.views.quick_colours.QColorDialog.getColor", return_value=QColor("#abcdef")):
            self.second.edit_colour(4)
        self.bar.move_colour(4, -1)
        self.bar.move_colour(3, 1)
        self.assertEqual(Project.open(self.stored.root).quick_colors[-1], "#ABCDEF")
        with patch("subkit_app.views.quick_colours.QColorDialog.getColor", return_value=QColor()):
            self.bar.add_colour()
            self.bar.edit_colour(0)
        self.assertEqual(len(self.model.colours), 5)
        self.second.remove_colour(4)
        self.assertEqual(Project.open(self.stored.root).quick_colors, list(DEFAULT_QUICK_COLORS))
        self.assertEqual(len(self.bar.buttons), 4)
        self.assertEqual(self.bar.buttons[0].toolTip(), "#FFE36E：点一下给选中的字（没选就是整行）上色；右键改颜色或删除")
        self.assertEqual(self.bar.buttons[0].width(), self.window.editor.fmt.btn_c.sizeHint().height())
        self.assertEqual(self.bar.buttons[0].width(), self.bar.buttons[0].height())

    def test_mainwindow_project_loading_updates_both_widgets_and_help(self):
        from PySide6.QtWidgets import QColorDialog
        from subkit_app import help as app_help
        self.window.open_project(self.other.root)
        self.assertIs(self.model.project, self.window.project)
        self.assertEqual(self.model.colours, list(DEFAULT_QUICK_COLORS))
        self.model.save(["#101010"])
        self.assertEqual(len(self.second.buttons), 1)
        dialog = QColorDialog(self.bar)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(app_help._anchor(dialog), "format")
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        dialog.show()
        dialog.activateWindow()
        self.app.processEvents()
        with patch("subkit_app.help.open_manual", return_value=True) as opened:
            QTest.keyClick(dialog, Qt.Key.Key_F1)
            opened.assert_called_once_with("format")
        dialog.close()
        self.assertEqual(Project.open(self.other.root).quick_colors, ["#101010"])

    def test_max_twelve_and_boundary_moves(self):
        colours = [f"#{i:06X}" for i in range(12)]
        self.model.save(colours)
        self.assertFalse(self.bar.plus.isEnabled())
        self.assertEqual(self.bar.plus.toolTip(), "最多 12 个")
        with patch("subkit_app.views.quick_colours.QColorDialog.getColor") as pick:
            self.bar.add_colour()
            pick.assert_not_called()
        self.bar.move_colour(0, -1)
        self.bar.move_colour(11, 1)
        self.assertEqual(self.model.colours, colours)
        for invalid in (colours + ["#FFFFFF"], ["red"], ["#12345G"]):
            with self.assertRaises(ValueError):
                self.model.save(invalid)
        self.bar.remove_colour(11)
        self.assertTrue(self.bar.plus.isEnabled())

    def test_context_menu_text_boundaries_and_delete(self):
        from PySide6.QtWidgets import QMenu
        actions = []
        class Menu(QMenu):
            def exec(self, pos):
                actions.extend(self.actions())
                return self.actions()[1]
        with patch("subkit_app.views.quick_colours.QMenu", Menu):
            self.bar._menu(0, self.bar.mapToGlobal(self.bar.rect().center()))
        self.assertEqual([a.text() for a in actions], ["改颜色…", "删除", "往左移", "往右移"])
        self.assertFalse(actions[2].isEnabled())
        self.assertTrue(actions[3].isEnabled())
        self.assertEqual(self.model.colours, list(DEFAULT_QUICK_COLORS)[1:])

    def _apply_and_undo(self, inline, selection):
        w = self.window
        if inline:
            from PySide6.QtCore import QRect
            self.project.settings = lambda: {}
            w._inline_edit(self.member.id, QRect(100, 100, 250, 40), 24)
            editor = w._inline.edit
            bar = w._inline.fmt.quick_colours
            self.addCleanup(lambda: w._inline and w._inline.cancel())
        else:
            editor = w.editor.cn_edits[0][1]
            bar = self.bar
        w.activateWindow() if not inline else w._inline.activateWindow()
        editor.setFocus()
        cursor = editor.textCursor()
        cursor.setPosition(1 if selection else 0)
        if selection:
            cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        editor.setTextCursor(cursor)
        self.app.processEvents()
        before = copy.deepcopy((self.doc.members[self.member.id].format, self.doc.members[self.member.id].overrides))
        count_before = len(self.session.editor.history.done)
        QTest.mouseClick(bar.buttons[0], Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertEqual(len(self.session.editor.history.done), count_before + 1)
        member = self.doc.members[self.member.id]
        self.assertNotEqual((member.format, member.overrides), before)
        if selection:
            from subkit.core.textformat import char_props
            self.assertEqual(char_props(member), [None, {"c1": "&H6EE3FF&"}, None])
            self.assertNotIn("c1", member.overrides)
        else:
            self.assertEqual(member.overrides["c1"], "&H6EE3FF&")
            self.assertEqual(member.format, [])
        self.session.editor.history.undo()
        member = self.doc.members[self.member.id]
        self.assertEqual((member.format, member.overrides), before)

    def test_group_selection_one_undo(self):
        self._apply_and_undo(False, True)

    def test_group_whole_line_one_undo(self):
        self._apply_and_undo(False, False)

    def test_inline_selection_one_undo(self):
        self._apply_and_undo(True, True)

    def test_inline_whole_line_one_undo(self):
        self._apply_and_undo(True, False)
