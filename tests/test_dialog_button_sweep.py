"""U9b: inventory every dialog constructor and exercise rows offscreen."""
import ast
import importlib
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import Qt, qVersion
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton
from subkit_app import help as help_ui
from tests import test_dialog_buttons as rows

ROOT = Path(__file__).resolve().parents[1] / "src" / "subkit_app"


def call_name(node):
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""


class SourceSweep(unittest.TestCase):
    def test_every_dialog_constructor_uses_shared_policy(self):
        inventory = []
        for path in sorted(ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
                if not any(call_name(b) in ("QDialog", "QMessageBox", "QProgressDialog") for b in cls.bases):
                    continue
                label = f"{path.relative_to(ROOT)}:{cls.name}"
                inventory.append(label)
                init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
                calls = [n for n in ast.walk(init) if isinstance(n, ast.Call)]
                delegates = [n for n in calls if call_name(n.func) in ("dialog_buttons", "_accept_buttons")]
                with self.subTest(dialog=label):
                    self.assertTrue(delegates, "session/project/worker-dependent constructor must delegate its row")
                    self.assertTrue(any(n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == "self"
                                        for n in delegates))
                    self.assertFalse(any(call_name(n.func) == "add_help_button" for n in calls))
                    self.assertFalse(any(call_name(n.func) in ("setText", "QPushButton", "addButton")
                                         and n.args and isinstance(n.args[0], ast.Constant)
                                         and n.args[0].value in ("OK", "Ok", "Cancel", "Close", "Help") for n in calls))
            # Direct construction paths must arrange the particular instance, not another dialog.
            for scope in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
                calls = [n for n in ast.walk(scope) if isinstance(n, ast.Call)]
                for assign in (n for n in ast.walk(scope) if isinstance(n, ast.Assign)
                               and isinstance(n.value, ast.Call)
                               and call_name(n.value.func) in ("QDialog", "QMessageBox")):
                    variable = next((t.id for t in assign.targets if isinstance(t, ast.Name)), None)
                    with self.subTest(path=str(path), function=scope.name, line=assign.lineno):
                        self.assertTrue(any(call_name(n.func) in ("dialog_buttons", "add_help_button")
                                            and n.args and isinstance(n.args[0], ast.Name)
                                            and n.args[0].id == variable for n in calls))
        self.assertGreaterEqual(len(inventory), 53)
        print(f"U9b source inventory: {len(inventory)} classes; project/session/job-dependent classes use source rules")
        print("; ".join(inventory))


class OffscreenSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        print("U9b runtime: Qt", qVersion(), "offscreen; manual opening mocked")

    check_dialog = rows.DialogTests.check_dialog

    def test_simple_remaining_dialogs(self):
        factories = [
            ("ruby_settings", "RubySettingsDialog", ({},), ["确定", "取消", "帮助"]),
            ("reading_popup", "PinyinSettingsDialog", ("none",), ["确定", "取消", "帮助"]),
            ("review_answers", "ImportAnswersDialog", (), ["确定", "取消", "帮助"]),
            ("review_answers", "EditAnswerDialog", ("原文",), ["采用", "取消", "帮助"]),
            ("project_readings", "ProjectReadingRuleDialog", (), ["加入草稿", "取消", "帮助"]),
            ("passages", "_RenameDialog", ({"name": "OP", "version": "TV"},), ["改名…", "取消", "帮助"]),
            ("passages", "PasteLyricsDialog", (), ["复制提示词", "读入", "取消", "帮助"]),
            ("season_export", "SeasonReportDialog", ("报告", ROOT), ["打开输出文件夹", "关闭", "帮助"]),
            ("song_fade", "SongFadeDialog", (1, (100, 100), True), ["恢复原字幕的淡入淡出", "应用", "取消", "帮助"]),
        ]
        for module, name, args, expected in factories:
            with self.subTest(dialog=name):
                cls = getattr(importlib.import_module("subkit_app.views." + module), name)
                self.check_dialog(cls(*args), expected)

    def test_word_split_default_scope_and_destructive_action(self):
        from subkit_app.views.word_split_dialog import WordSplitDialog
        for scope in ("here", "project"):
            dialog = WordSplitDialog("世界征服", parts=[{"surface": "世界", "reading": "せかい"},
                                                       {"surface": "征服", "reading": "せいふく"}],
                                     scope=scope, existing_rule=True)
            self.check_dialog(dialog, ["删除此工程规则", "应用到当前句子", "应用到整个 Project", "取消", "帮助"])
            self.assertTrue((dialog.apply_project_button if scope == "project" else dialog.apply_here_button).isDefault())
            self.assertFalse(dialog.remove_button.isDefault())

    def test_lazy_input_dialog(self):
        from PySide6.QtWidgets import QInputDialog
        dialog = QInputDialog()
        dialog.help_anchor = "karaoke-auto"
        dialog.setInputMode(QInputDialog.InputMode.IntInput)
        help_ui.dialog_buttons(dialog)
        self.check_dialog(dialog, ["确定", "取消", "帮助"])
        dialog.show()
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)

    def test_progress_updates_resize_help_and_cancel(self):
        dialog = help_ui.ProgressDialog("进度", "取消", 0, 100)
        dialog.help_anchor = "karaoke-auto"
        self.addCleanup(dialog.deleteLater)
        dialog.setAutoClose(False)
        self.check_dialog(dialog, ["取消", "帮助"])
        dialog.show()
        for width in (300, 500, 300):
            dialog.resize(width, 140)
            dialog.setLabelText("更新进度")
            dialog.setValue(20)
            self.app.processEvents()
            buttons = rows.row_buttons(dialog)
            points = [b.mapTo(dialog, b.rect().center()) for b in buttons]
            self.assertEqual(points[0].y(), points[1].y())
            self.assertLess(points[0].x(), points[1].x())
        cancelled = []
        dialog.canceled.connect(lambda: cancelled.append(True))
        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        self.assertTrue(cancelled)
        self.assertTrue(dialog.wasCanceled())
        dialog.hide()


class SessionSweep(unittest.TestCase):
    setUpClass = classmethod(rows.fixtures.ConversionUI.setUpClass.__func__)
    setUp = rows.fixtures.ConversionUI.setUp
    _dispose = rows.fixtures.ConversionUI._dispose
    group = rows.fixtures.ConversionUI.group
    check_dialog = rows.DialogTests.check_dialog

    def test_session_rows(self):
        from types import SimpleNamespace
        from subkit_app.views import missing_language, shift_groups, source_copies, people, passages, review_answers
        member = self.group("g")
        languages = {"original": "ja", "translation": "zh-Hans"}
        book = self.session.editor.character_book
        dialogs = [
            (missing_language.MissingLanguageDialog(self.doc, ["g"], "original", languages), ["补上", "取消", "帮助"]),
            (shift_groups.ShiftGroupsDialog(self.doc, ["g"], self.settings), ["确定", "取消", "帮助"]),
            (source_copies.SourceCopiesDialog(self.project, problems=[]),
             ["从导入时的位置恢复", "在文件夹里查找…", "选择文件…", "关闭", "帮助"]),
            (people.NewCharacterDialog(self.window, languages), ["确定", "取消", "帮助"]),
            (people.CharacterChoiceDialog(self.window, "人物", [], str), ["确定", "取消", "帮助"]),
            (people.SpeakerPicker(self.window, book, member, [], settings=self.settings), ["关闭", "帮助"]),
            (people.BangumiFetchDialog(self.window, "0"), ["取消", "帮助"]),
            (passages.PassageLibraryDialog(SimpleNamespace(list=lambda: [])),
             ["导入旧版 OPED 资料…", "改名…", "删除", "关闭", "帮助"]),
            (passages.OPEDImportDialog(SimpleNamespace(), languages=languages), ["导入", "取消", "帮助"]),
            (review_answers.ReviewAnswersDialog(self.session, [], [], self.window), ["采用所有读音意见", "关闭", "帮助"]),
        ]
        for dialog, expected in dialogs:
            self.check_dialog(dialog, expected)
