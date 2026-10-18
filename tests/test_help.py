"""Offscreen help routing and user-manual consistency; no system browser launch."""

from __future__ import annotations

import ast
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QLineEdit, QMessageBox, QPushButton
from subkit_app import APP_VERSION, help as help_ui, mainwindow as mw
from subkit_app.views.dialogs import AddEpisodeDialog, ExportDialog, MergeDialog
from subkit_app.views.reading_popup import ReadingPopup
from tests.test_convert_ui import PlayerDouble


class ManualParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.text = set(), []

    def handle_starttag(self, tag, attrs):
        self.ids.update(v for k, v in attrs if k == "id")

    def handle_data(self, data):
        self.text.append(data)


def menu_name(text):
    text = re.sub(r"\(&.\)", "", text.split("\t")[0]).rstrip("…")
    return re.sub(r"（[^（）]*）$", "", text)


class HelpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if cls.app.platformName() != "offscreen":
            raise RuntimeError("Help tests require offscreen Qt")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub_help_")
        self.addCleanup(self.tmp.cleanup)
        self.settings = QSettings(str(Path(self.tmp.name) / "settings.ini"), QSettings.Format.IniFormat)
        for name, value in (("MpvWidget", PlayerDouble), ("QSettings", Mock(return_value=self.settings)),
                            ("MainWindow._open_last", Mock())):
            p = patch("subkit_app.mainwindow." + name, value)
            p.start()
            self.addCleanup(p.stop)
        self.window = mw.MainWindow()
        self.addCleanup(self.dispose)
        self.window.show()
        self.app.processEvents()

    def dispose(self):
        self.window.session = None
        self.window.watch.transform = None
        self.window.watch_mode = False
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.settings.sync()

    def actions(self, menu=None):
        for action in (menu or self.window.menuBar()).actions():
            if action.isSeparator():
                continue
            yield action
            submenu = action.menu()
            if submenu is not None:
                yield from self.actions(submenu)

    def test_manual_anchors_names_version_and_status_tips(self):
        parser = ManualParser()
        parser.feed(help_ui.manual_path().read_text(encoding="utf-8"))
        text = "".join(parser.text)
        self.assertIn("对应版本 " + APP_VERSION, text)
        from subkit_app.actions import ActionRegistry
        anchors = set(help_ui.HELP_ANCHORS) | {spec.help_anchor for spec in ActionRegistry()}
        for file in (ROOT / "src" / "subkit_app").rglob("*.py"):
            for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "help_anchor"
                                                       for t in node.targets):
                    anchors.add(ast.literal_eval(node.value))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                        "open_manual", "_help_message") and node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        anchors.add(node.args[0].value)
        self.assertFalse(anchors - parser.ids)
        design = (ROOT / "docs/ui-redesign/DESIGN.md").read_text(encoding="utf-8")
        design = re.sub(r"(开始时间|结束时间) −1 帧／\+1 帧", r"\1 −1 帧／\1 +1 帧", design)
        missing = []
        for action in self.actions():
            self.assertTrue(action.statusTip(), action.text())
            self.assertIn(action.property("help_anchor"), parser.ids, action.text())
            if action is self.window.dock_source.toggleViewAction():
                continue
            if menu_name(action.property("enabled_text") or action.text()) not in text + design:
                missing.append(menu_name(action.text()))
        self.assertEqual(missing, [])

    def test_menu_and_open_url(self):
        actions = self.window.help_menu.actions()
        self.assertEqual([menu_name(a.text()) for a in actions],
                         ["使用说明", "快捷键一览", "版本记录", "关于双语字幕工坊"])
        url = help_ui.manual_url("readings")
        self.assertEqual(Path(url.toLocalFile()), help_ui.manual_path())
        self.assertEqual(url.fragment(), "readings")
        redirect = Path(self.tmp.name) / "duosub-help.html"
        with patch.object(help_ui.QDesktopServices, "openUrl", return_value=True) as opener, \
                patch.object(help_ui, "redirect_path", return_value=redirect):
            self.assertTrue(help_ui.open_manual("readings"))
            self.assertEqual(Path(opener.call_args.args[0].toLocalFile()), redirect)
            page = redirect.read_text(encoding="utf-8")
            self.assertIn(url.toString(QUrl.ComponentFormattingOption.FullyEncoded), page)
            self.assertIn("location.replace(", page)
            self.assertTrue(help_ui.open_manual())
            self.assertEqual(Path(opener.call_args.args[0].toLocalFile()), help_ui.manual_path())
        with patch.object(help_ui, "manual_path", return_value=Path(self.tmp.name) / "missing.html"), \
                patch.object(QMessageBox, "warning") as warning:
            self.assertFalse(help_ui.open_manual())
            self.assertEqual(warning.call_args.args[2], f"找不到使用说明：{Path(self.tmp.name) / 'missing.html'}")

    def test_context_rows_and_precedence(self):
        w = self.window
        for widget, anchor in ((w.wave, "timing"), (w.wave_panel, "timing"), (w.editor.fmt, "format"),
                               (w.editor.position, "position"), (w.table, "review"), (w.editor, "review"),
                               (w.source, "review"), (w.player, None)):
            child = QLineEdit(widget)
            self.assertEqual(help_ui.resolve_anchor(w, active=w, focus=child), anchor)
            child.deleteLater()
        w.session = SimpleNamespace(doc=SimpleNamespace(group=lambda gid: SimpleNamespace(kind="song")))
        w.current_gid = "song"
        self.assertEqual(help_ui.resolve_anchor(w, active=w, focus=w.wave), "karaoke")
        w.watch_mode = True
        self.assertEqual(help_ui.resolve_anchor(w, active=w, focus=w.player), "karaoke")
        w.session.doc.group = lambda gid: SimpleNamespace(kind="dialogue")
        self.assertEqual(help_ui.resolve_anchor(w, active=w, focus=w.editor), "watch")
        w.watch.transform = object()
        self.assertEqual(help_ui.resolve_anchor(w, active=w, focus=w.editor), "transform")
        dialog = AddEpisodeDialog(w)
        self.assertEqual(help_ui.resolve_anchor(w, active=dialog, focus=dialog.eid), "add-episode")
        dialog.deleteLater()

    def test_dialog_buttons_and_modal_f1(self):
        for cls, anchor in ((AddEpisodeDialog, "add-episode"), (MergeDialog, "add-episode"),
                            (ExportDialog, "export")):
            dialog = cls(self.window)
            try:
                button = next(b for b in dialog.findChildren(QPushButton) if b.text() == "帮助")
                with patch.object(help_ui, "open_manual") as opener:
                    button.click()
                    opener.assert_called_once_with(anchor)
                    opener.reset_mock()
                    def press():
                        QTest.keyClick(dialog, Qt.Key.Key_F1)
                        dialog.reject()
                    QTimer.singleShot(0, press)
                    dialog.exec()
                    opener.assert_called_once_with(anchor)
            finally:
                dialog.deleteLater()
        popup = ReadingPopup("字", "じ", [], "auto", "auto", self.window)
        try:
            popup.show()
            popup.activateWindow()
            self.app.processEvents()
            with patch.object(help_ui, "open_manual") as opener:
                QTest.keyClick(popup.edit, Qt.Key.Key_F1)
                opener.assert_called_once_with("readings")
        finally:
            popup.close()
            popup.deleteLater()

    def test_scoped_message_help_does_not_accept_and_restores_scope(self):
        def run(parent, *args):
            box = QMessageBox(parent)
            box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            box.setDefaultButton(QMessageBox.StandardButton.No)
            def press():
                button = next(b for b in box.findChildren(QPushButton) if b.text() == "帮助")
                with patch.object(help_ui, "open_manual") as opener:
                    button.click()
                    opener.assert_called_once_with("fonts")
                    self.assertTrue(box.isVisible())
                    QTest.keyClick(box, Qt.Key.Key_F1)
                    self.assertEqual(opener.call_args.args, ("fonts",))
                box.button(QMessageBox.StandardButton.No).click()
            QTimer.singleShot(0, press)
            return box.exec()
        self.assertEqual(self.window._help_message("fonts", run), QMessageBox.StandardButton.No)
        self.assertIsNone(self.window._help_message_anchor)

    def test_about_uses_app_version(self):
        def inspect(box):
            self.assertIn("双语字幕工坊（DuoSub）" + APP_VERSION, box.text())
            self.assertEqual([b.text() for b in box.buttons()], ["关闭"])
            self.assertEqual(box._button_row.help_button.text(), "帮助")
        with patch.object(QMessageBox, "exec", inspect):
            self.window.show_about()

    def test_auto_results_f1_and_new_menu_anchors(self):
        from subkit_app.views.kalign_result import KalignResultDialog
        dialog = KalignResultDialog(self.window)
        self.addCleanup(dialog.deleteLater)
        dialog.show()
        dialog.activateWindow()
        self.app.processEvents()
        with patch.object(help_ui, "open_manual") as opener:
            QTest.keyClick(dialog, Qt.Key.Key_F1)
            opener.assert_called_once_with("karaoke-auto")
        p = self.window.wave_panel
        self.assertEqual(p.redistribute_menu.menuAction().property("help_anchor"), "karaoke")
        self.assertTrue(all(a.property("help_anchor") == "karaoke" for a in p.redistribute_actions.values()))
        fx = next(a.menu() for a in p.more_menu.actions() if a.text() == "特效")
        self.assertEqual(fx.help_anchor, "karaoke-fx")
        self.assertTrue(all(a.property("help_anchor") == "karaoke-fx" for a in fx.actions() if not a.isSeparator()))
        dialog.close()

    def test_long_note_menu_items_f1_open_auto_karaoke(self):
        menu = self.window.wave_panel.kauto_continuation
        self.assertEqual(menu.help_anchor, "karaoke-auto")
        for action in menu.actions():
            if action.isSeparator():
                continue
            self.assertEqual(action.property("help_anchor"), "karaoke-auto")
            menu.popup(self.window.pos())
            menu.setActiveAction(action)
            self.app.processEvents()
            with patch.object(help_ui, "open_manual") as opener:
                QTest.keyClick(menu, Qt.Key.Key_F1)
                opener.assert_called_once_with("karaoke-auto")
            menu.close()

    def test_remaining_dialog_mappings_and_buttons(self):
        from PySide6.QtWidgets import QWidget
        from subkit.ass import AssDocument
        from subkit.styles_default import default_styles_ass
        from subkit.model.document import Document
        from subkit.core.merge import MergeSettings
        from subkit_app.views.dialogs import ConvertToSongDialog
        from subkit_app.views import passages, template_editor
        template = AssDocument.parse(default_styles_ass())
        session = SimpleNamespace(episode=SimpleNamespace(id="E01", settings=lambda: {}), template=template)
        passage = dict(name="OP", category="OP", version="TV", start=0, duration=1000, groups=[],
                       summary=dict(groups=0, lines=0, karaoke=dict(partial=0), unconfirmed_lines=0))
        with patch.object(template_editor, "TemplateEditor", side_effect=lambda *a, **k: QWidget(a[0])):
            dialogs = [(template_editor.TemplateDialog(self.window), "template")]
        dialogs += [(template_editor.StyleImportDialog(AssDocument(), self.window,
                                                       translation_lang="zh-Hant", original_lang="ja"), "template"),
                    (ConvertToSongDialog(Document("help"), [], self.window, styles=MergeSettings().styles,
                                        template_styles=[], settings=self.settings), "karaoke"),
                    (passages.RecordPassageDialog(passage, self.window, karaoke_states=[]), "passages"),
                    (passages.PassageLibraryDialog(SimpleNamespace(list=lambda: []), self.window), "passages"),
                    (passages.PassagesDialog(session, [], self.window.player, self.window), "passages")]
        for dialog, anchor in dialogs:
            try:
                self.assertEqual(help_ui.resolve_anchor(self.window, active=dialog), anchor)
                button = next(b for b in dialog.findChildren(QPushButton) if b.text() == "帮助")
                with patch.object(help_ui, "open_manual") as opener:
                    button.click()
                    opener.assert_called_once_with(anchor)
            finally:
                dialog.deleteLater()
