"""Offscreen private switch, shared font result dialog and safe layout reset."""
from pathlib import Path
from unittest.mock import patch, Mock

from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QPushButton, QMessageBox

from tests import test_convert_ui as conversion_ui
from subkit.core.fontcheck import FontCheck, FontLibrary
from subkit_app.views import fonts
from subkit_app import help as app_help


class PrivateSwitch(conversion_ui.ConversionUI):
    def test_default_toggle_persist_and_library_visibility(self):
        w = self.window
        self.assertTrue(w.private_enabled)
        self.assertTrue(w.act_private.isChecked())
        library = w._private_actions[0]
        w.act_private.setChecked(False)
        self.assertFalse(library.isVisible())
        self.assertFalse(self.settings.value("private/enabled", type=bool))
        w.act_private.setChecked(True)
        self.assertTrue(library.isVisible())

    def test_shared_export_check_and_private_library_gating(self):
        fc = FontCheck(ran=True, missing={"Absent": ["ref"]})
        for enabled in (True, False):
            self.window.act_private.setChecked(enabled)
            with patch.object(fonts.FontCheckDialog, "exec", return_value=1), \
                    patch.object(self.window, "_font_library_index", return_value=FontLibrary(self.path / "none.json")) as lookup:
                self.assertTrue(self.window._show_font_check(fc, exporting=True))
                self.assertEqual(lookup.call_count, int(enabled))

    def test_existing_stale_missing_paths_explorer_and_f1(self):
        file = self.path / "font with spaces.ttf"
        file.write_bytes(b"test")
        index = self.path / "fonts.json"
        import json
        index.write_text(json.dumps([{"families":["Found"], "path":str(file)},
                                     {"families":["Stale"], "path":str(self.path / "gone.ttf")}]), encoding="utf-8")
        dialog = fonts.FontCheckDialog(FontCheck(ran=True, missing={k:["ref"] for k in ("Found","Stale","Unknown")}),
                                       self.window, library=FontLibrary(index), exporting=True)
        try:
            labels = "\n".join(x.text() for x in dialog.findChildren(QLabel))
            self.assertIn("字体库里也没有", labels)
            self.assertIn("字体库里记的文件已经不在了", labels)
            buttons = dialog.findChildren(QPushButton)
            with patch.object(fonts.subprocess, "Popen") as run:
                next(b for b in buttons if b.text() == "在资源管理器中显示").click()
                run.assert_called_once_with(["explorer.exe", "/select,", str(file.resolve())])
                self.assertNotIn("startupinfo", run.call_args.kwargs)
            file.unlink()
            with patch.object(fonts.subprocess, "Popen") as run, patch.object(QMessageBox, "warning") as warning:
                dialog.show_file(file)
                run.assert_not_called()
                self.assertIn("已经不在了", warning.call_args.args[2])
            with patch.object(app_help, "open_manual") as opener:
                def press():
                    QTest.keyClick(dialog, Qt.Key.Key_F1)
                    dialog.reject()
                QTimer.singleShot(0, press)
                dialog.exec()
                opener.assert_called_once_with("fonts")
        finally:
            dialog.deleteLater()

    def test_clear_layout_cache_releases_jobs_and_refreshes(self):
        cache = self.path / "render.sqlite3"
        cache.write_bytes(b"old")
        self.session.export_cache.key = "old"
        self.window.watch.geo[0] = "old"
        self.window._clear_layout_cache()
        self.assertFalse(cache.exists())
        self.assertIsNone(self.session.export_cache.key)
        self.assertFalse(self.window.watch.geo)
        self.session.request_preview.assert_called_once()
        self.assertEqual(self.window.statusBar().currentMessage(), "排版缓存已清除，预览会重新测量。")

    def test_unstopped_job_keeps_cache(self):
        cache = self.path / "render.sqlite3"
        cache.write_bytes(b"old")
        with patch.object(self.session, "stop_preview", return_value=False), patch.object(QMessageBox, "warning") as warning:
            self.window._clear_layout_cache()
        self.assertEqual(cache.read_bytes(), b"old")
        warning.assert_called_once()
