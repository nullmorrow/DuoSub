"""X3 offscreen font names and private assfonts lookup with synthetic libraries only."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from subkit.ass import Style
from subkit.ass.text import tokenize
from subkit.core.fontcheck import FontCheck, FontLibrary, installed_font_aliases, prompt_text, report_lines
from subkit.core.textformat import with_span
from subkit.core.export import ExportOptions, build_ass
from subkit.model.document import Document, Group, Member, Run
from subkit_app import help as app_help, mainwindow as mw
from subkit_app.views.format_bar import FormatBar
from subkit_app.views.group_editor import GroupEditor
from subkit_app.views.dialogs import ExportDialog
from tests import test_convert_ui as conversion_ui


class FontBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        aliases = patch("subkit_app.views.format_bar.installed_font_aliases", return_value=frozenset())
        aliases.start()
        self.addCleanup(aliases.stop)
        self.editor = GroupEditor()
        self.addCleanup(self.dispose)
        self.style = Style.new("8para", Fontname="方正准圆_GBK")
        self.editor.style_lookup = lambda name: self.style if name == "8para" else None

    def dispose(self):
        self.editor.close()
        self.editor.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def show_line(self, text, overrides=None):
        runs = [Run(t.raw, "tag" if t.kind in ("tags", "comment") else "text", ins="i", a=t.start, b=t.end)
                for t in tokenize(text)]
        self.member = Member("m", "g", "zh-Hans", "translation", "sign", 1000, 2000,
                             style="8para", runs=runs, overrides=overrides or {})
        self.doc = Document("font-box", groups=[Group("g", "sign", ["m"])], members={"m": self.member})
        before = copy.deepcopy(self.doc)
        with patch("subkit_app.views.format_bar.QFontDatabase.families", return_value=["Arial", "Tahoma"]):
            self.editor.set_group(self.doc, self.doc.groups[0], None, None)
        self.assertEqual(self.doc, before)
        return self.editor.fmt.font_box.currentText()

    def test_soma_leading_inline_font_verbatim(self):
        changes = Mock()
        self.editor.formatRequested.connect(changes)
        self.assertEqual(self.show_line(r"{\fn方正小标宋_GBK\fscx88\fsp3\fs40}屏幕字"), "方正小标宋_GBK（未安装）")
        self.assertEqual(self.editor.fmt.target.text(), "整行")
        self.assertEqual(self.editor.fmt.font_box.toolTip(), "本机没有这个字体，画面会用替代字体显示")
        changes.assert_not_called()

    def test_style_font_and_late_inline_tag(self):
        self.assertEqual(self.show_line(r"第一字{\fnArial}后来"), "方正准圆_GBK（未安装）")

    def test_manual_font_wins_and_user_selection_emits_family(self):
        self.assertEqual(self.show_line(r"{\fn方正小标宋_GBK}屏幕字", {"fn": "Arial"}), "Arial")
        changes = Mock()
        self.editor.fmt.formatRequested.connect(changes)
        box = self.editor.fmt.font_box
        # Windows offscreen has no font model; emulate Qt's chosen installed font.
        with patch.object(box, "currentFont", return_value=QFont("Tahoma")):
            box.activated.emit(0)
        changes.assert_called_once_with({"fn": "Tahoma"})

    def test_font_reset_and_last_leading_tag(self):
        for tags in (r"\fnArial\fn", r"\fnArial\fn0", r"\fnArial\r"):
            with self.subTest(tags=tags):
                self.assertEqual(self.show_line("{" + tags + "}字"), "方正准圆_GBK（未安装）")
        self.assertEqual(self.show_line(r"{\fnArial}{\fnTahoma}字"), "Tahoma")

    def test_manual_raw_font_wins(self):
        self.assertEqual(self.show_line(r"{\fn方正小标宋_GBK}字", {"fn": "Arial", "raw": r"\fnTahoma"}), "Tahoma")

    def test_selected_span_font_and_inline_at_selection(self):
        self.show_line(r"A{\fnTahoma}B")
        self.member.format = with_span(self.member, 1, 2, {"fn": "Arial"})
        edit = self.editor._fmt_edit
        cursor = edit.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, cursor.MoveMode.KeepAnchor)
        edit.setTextCursor(cursor)
        with patch("subkit_app.views.format_bar.QFontDatabase.families", return_value=["Arial", "Tahoma"]):
            self.editor._show_fmt()
            self.assertEqual(self.editor.fmt.font_box.currentText(), "Arial")
            self.member.format = []
            self.editor._show_fmt()
            self.assertEqual(self.editor.fmt.font_box.currentText(), "Tahoma")

    def test_refresh_missing_name_never_emits_and_installed_clears_tooltip(self):
        bar = self.editor.fmt
        change = Mock()
        bar.formatRequested.connect(change)
        for name, expected in (("Not Installed X3", "Not Installed X3（未安装）"), ("Arial", "Arial")):
            with patch("subkit_app.views.format_bar.QFontDatabase.families", return_value=["Arial"]):
                bar.show_state(target="整行", fn=name, fs=40, bold=False, italic=False, colour=None)
            self.assertEqual(bar.font_box.currentText(), expected)
        self.assertEqual(bar.font_box.toolTip(), "字体（选中的字；没选就是整行）")
        change.assert_not_called()

    def test_installed_localized_alias_is_not_marked_missing(self):
        with patch("subkit_app.views.format_bar.installed_font_aliases", return_value={"方正小标宋_gbk"}):
            self.assertEqual(self.show_line(r"{\fn方正小标宋_GBK}字"), "方正小标宋_GBK")


class Library(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="x3_library_")
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "fonts.json"
        self.path.write_text(json.dumps([
            {"families": ["family"], "fullnames": ["FULL NAME"], "psnames": ["psname"], "path": "Z:/fonts/a.ttc", "index": 2},
            {"families": ["family"], "path": "Z:/fonts/a.ttc", "index": 3},
            {"families": ["family"], "path": "Z:/fonts/b.ttf"},
            {"families": ["方正小标宋_GBK"], "path": "Z:/fonts/小标宋.ttf"},
            {"families": None, "path": 123}, None], ensure_ascii=False), encoding="utf-8")

    def test_lazy_cached_lookup_all_aliases_lowercase_paths_deduplicated(self):
        library = FontLibrary(self.path)
        self.assertIsNone(library._names)
        with patch.object(Path, "open", autospec=True, wraps=Path.open) as opened:
            # Wrap manually so the synthetic file is actually parsed.
            opened.side_effect = lambda p, *a, **k: open(p, *a, **k)
            self.assertEqual(library.lookup("FaMiLy"), ["Z:/fonts/a.ttc", "Z:/fonts/b.ttf"])
            self.assertEqual(library.lookup("full name"), ["Z:/fonts/a.ttc"])
            self.assertEqual(library.lookup("PSNAME"), ["Z:/fonts/a.ttc"])
            self.assertEqual(library.lookup("absent"), [])
            opened.assert_called_once()
        found = library.lookup("family")
        found.clear()
        self.assertEqual(len(library.lookup("family")), 2)

    def test_report_and_export_prompt_include_each_found_missing_font(self):
        library = FontLibrary(self.path)
        fc = FontCheck(ran=True, missing={"方正小标宋_GBK": ["ref"], "Unknown": ["ref2"]})
        expected = "  方正小标宋_GBK — 字体库里有：Z:/fonts/小标宋.ttf"
        self.assertIn(expected, report_lines(fc, library=library))
        self.assertIn(expected, prompt_text(fc, library=library))
        self.assertNotIn("Unknown — 字体库里有", prompt_text(fc, library=library))
        self.assertEqual(fc.missing, {"方正小标宋_GBK": ["ref"], "Unknown": ["ref2"]})

    def test_success_or_unverified_does_not_load(self):
        library = FontLibrary(self.path)
        report_lines(FontCheck(ran=True), library=library)
        prompt_text(FontCheck(error="no tool"), library=library)
        self.assertIsNone(library._names)

    @unittest.skipUnless(os.name == "nt", "Windows font registry aliases")
    def test_installed_registry_alias_requires_existing_font_and_is_cached(self):
        import winreg

        installed_font_aliases.cache_clear()
        self.addCleanup(installed_font_aliases.cache_clear)
        font = self.path.parent / "font.ttf"
        font.write_bytes(b"synthetic font existence only")
        entries = [("方正小标宋_GBK (TrueType)", str(font), 1),
                   ("Absent (TrueType)", str(font.with_name("missing.ttf")), 1)]
        context = MagicMock()
        with patch.object(winreg, "OpenKey", return_value=context) as opened, \
                patch.object(winreg, "QueryInfoKey", return_value=(0, len(entries), 0)), \
                patch.object(winreg, "EnumValue", side_effect=lambda _key, i: entries[i]):
            self.assertEqual(installed_font_aliases(), frozenset({"方正小标宋_gbk"}))
            self.assertEqual(installed_font_aliases(), frozenset({"方正小标宋_gbk"}))
            self.assertEqual(opened.call_count, 2)

    def test_missing_corrupt_and_wrong_shape_do_not_break_font_report(self):
        for text in (None, "{", "{}"):
            with self.subTest(text=text):
                if text is None:
                    self.path.unlink()
                else:
                    self.path.write_text(text, encoding="utf-8")
                library = FontLibrary(self.path)
                lines = report_lines(FontCheck(ran=True, missing={"A": ["ref"]}), library=library)
                self.assertIn("没有安装：A", "\n".join(lines))
                self.assertEqual(library.lookup("A"), [])
                self.assertTrue(library.error)


class LibraryUI(conversion_ui.ConversionUI):
    def setUp(self):
        super().setUp()
        self.library_path = self.path / "fonts.json"
        self.library_path.write_text('[{"families":["Missing X3"],"path":"Z:/font.ttf"}]', encoding="utf-8")
        self.settings.setValue("fonts/library_path", str(self.library_path))

    def test_menu_next_to_fontcheck_and_file_dialog_help_save_cancel(self):
        tools = self.window.tools_menu
        check = next(a.menu() for a in tools.actions() if a.text() == "检查")
        advanced = next(a.menu() for a in tools.actions() if a.text() == "高级")
        self.assertEqual(check.actions()[1].text(), "字体（缺字体、缺字）")
        self.assertEqual(advanced.actions()[1].text(), "字体库位置…")
        self.assertEqual(advanced.actions()[1].property("help_anchor"), "fonts")
        previous = self.window._font_library_index()
        new_path = self.path / "new.json"

        def accept(dialog):
            self.assertEqual(dialog.help_anchor, "fonts")
            self.assertTrue(dialog.testOption(QFileDialog.Option.DontUseNativeDialog))
            with patch.object(app_help, "open_manual") as help_call:
                from PySide6.QtWidgets import QPushButton
                next(b for b in dialog.findChildren(QPushButton) if b.text() == "帮助").click()
                help_call.assert_called_once_with("fonts")
            return 1

        with patch.object(QFileDialog, "exec", accept), patch.object(QFileDialog, "selectedFiles", return_value=[str(new_path)]):
            self.window._choose_font_library()
        self.assertEqual(self.settings.value("fonts/library_path"), str(new_path))
        self.assertIsNot(self.window._font_library_index(), previous)
        with patch.object(QFileDialog, "exec", return_value=0):
            self.window._choose_font_library()
        self.assertEqual(self.settings.value("fonts/library_path"), str(new_path))

    def test_font_check_message_and_cursor_restored(self):
        self.episode.build_export = Mock(return_value=(None, None))
        with patch.object(mw, "check_fonts", return_value=FontCheck(ran=True, missing={"Missing X3": ["ref"]})), \
                patch.object(mw.FontCheckDialog, "exec", return_value=0) as message:
            self.window.check_episode_fonts()
        message.assert_called_once()
        self.assertIsNone(QApplication.overrideCursor())
        self.assertEqual(self.window.statusBar().currentMessage(), "")

    def test_export_prompt_cancel_and_written_report_use_library(self):
        self.group("g", text="测试")
        ass, report = build_ass(self.doc, {}, self.template, self.path, ExportOptions(ruby=False))
        out = self.path / "export.ass"
        self.episode.validate = Mock(return_value=SimpleNamespace(ok=True, summary=lambda: "通过"))
        self.episode.build_export = Mock(return_value=(ass, report))

        def write(result, path):
            result.save(path)
            return path

        self.episode.write_export = Mock(side_effect=write)

        def accept(dialog):
            dialog.path.setText(str(out))
            return 1

        fc = FontCheck(ran=True, missing={"Missing X3": ["ref"]})
        with patch.object(ExportDialog, "exec", accept), patch.object(mw, "check_fonts", return_value=fc), \
                patch.object(mw.FontCheckDialog, "exec", return_value=0) as question:
            self.window.export()
        question.assert_called_once()
        self.episode.write_export.assert_not_called()
        self.assertIsNone(QApplication.overrideCursor())
        with patch.object(ExportDialog, "exec", accept), patch.object(mw, "check_fonts", return_value=fc), \
                patch.object(mw.FontCheckDialog, "exec", return_value=1), \
                patch.object(QMessageBox, "information"):
            self.window.export()
        self.assertIn("字体库里有：Z:/font.ttf", out.with_suffix(".export-report.txt").read_text(encoding="utf-8"))
        self.assertIsNone(QApplication.overrideCursor())


if __name__ == "__main__":
    unittest.main()
