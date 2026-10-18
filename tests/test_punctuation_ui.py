"""P3 offscreen settings, fake background renderer and gated host libass smoke."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton
from subkit import templates
from subkit.ass import AssDocument
from subkit.core.display import PunctuationRules, PUNCTUATION_CATEGORIES
from subkit.core.punctuation_profiles import PunctuationProfileStore, create_profile
from subkit.core.workspace import Project
from subkit.styles_default import default_styles_ass
from subkit_app import help as help_ui
from subkit_app.views import punctuation as ui


class PunctuationUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        assert cls.app.platformName() == "offscreen"

    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix="punctuation-ui-"))
        self.project = Project.create(self.path / "project", "UI", template_text=default_styles_ass())
        self.store = PunctuationProfileStore(self.path / "profiles")
        self.settings = QSettings(str(self.path / "settings.ini"), QSettings.Format.IniFormat)
        self.refresh, self.preview = Mock(), Mock()
        self.session = Mock()
        self.session.request_preview = self.preview
        self.session.episode.effective_playres.return_value = (1440, 1080)
        self.calls = []
        def fake(styles, png, **kw):
            self.calls.append((styles, kw, threading.get_ident()))
            image = QImage(320, 240, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.black)
            image.save(str(png))
            return templates.Preview(png, ["缺失字体"], {"Arial": "缺"})
        self.fake = fake
        self.dialogs = []

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.reject()
            self.wait(lambda: dialog._job is None)
            dialog.deleteLater()
        self.app.processEvents()
        self.settings.sync()

    def open(self, **kw):
        dialog = ui.PunctuationDialog(self.project, self.store, self.settings, session=self.session,
                    on_apply=self.refresh, renderer=kw.pop("renderer", self.fake), **kw)
        self.dialogs.append(dialog)
        return dialog

    def wait(self, predicate, seconds=3):
        end = time.monotonic() + seconds
        while not predicate() and time.monotonic() < end:
            self.app.processEvents()
            QTest.qWait(10)
        self.assertTrue(predicate(), "background UI operation did not finish")

    def test_two_tables_scopes_long_vowel_shared_model_and_completion(self):
        self.project.data["settings"].setdefault("display", {}).setdefault("punctuation", {})["original"] = {"font": "AbsentFont123"}
        with patch.object(ui.QFontDatabase, "families", return_value=["Arial", "Segoe UI"]) as families:
            d = self.open()
            families.assert_called_once_with()
        self.assertEqual(len(d.font_boxes), 26)
        boxes = list(d.font_boxes.values()) + list(d.all_fonts.values())
        self.assertEqual(len(boxes), 28)
        self.assertTrue(all(b.model() is d.font_model and b.isEditable() and b.completer() for b in boxes))
        self.assertEqual(d.font_boxes["original", "ellipsis"].currentText(), "AbsentFont123（未安装）")
        for lane, table in d.tables.items():
            self.assertEqual(table.rowCount(), 15)
            self.assertEqual([table.item(i, 0).text() for i in range(len(ui.ROWS))], [r[1] for r in ui.ROWS])
            row = next(i for i, r in enumerate(ui.ROWS) if r[0] == "long_vowel")
            self.assertIsNone(table.cellWidget(row, 1))
            self.assertEqual(table.item(row, 1).text(), "—")
            for layer in ("scope_chars", "scope_fonts"):
                self.assertEqual([k for k, _ in ui.MERGE_KINDS if d.scope_boxes[lane, layer, k].isChecked()],
                                 list(getattr(d._rules(lane), layer)))

    def test_halfwidth_kana_live_preview_scope_and_cancel(self):
        before = copy.deepcopy(self.project.data["settings"])
        d = self.open()
        row = next(i for i, item in enumerate(ui.ROWS) if item[0] == "halfwidth_kana")
        self.assertEqual(row, 0)
        self.assertIsNone(d.tables["original"].cellWidget(row, 2))
        box = d.rules["original", "halfwidth_kana"]
        self.assertEqual([box.itemText(i) for i in range(box.count())], ["不改", "转为全宽（含长音）"])
        box.setCurrentIndex(box.findData(True))
        self.assertTrue(d._rules("original").halfwidth_kana)
        self.assertFalse(d._rules("translation").halfwidth_kana)
        self.assertTrue(self.preview.called)
        sample = "超神ﾈｲｶﾞｰ"
        tpl = self.project.template()
        shown = ui.preview_events(tpl, "original", "dialogue", sample, d._rules("original"), (1920, 1080))
        self.assertIn("超神ネイガー", shown[1])
        song = ui.preview_events(tpl, "original", "song", sample, d._rules("original"), (1920, 1080))
        self.assertIn(sample, song[1])
        d.scope_boxes["original", "scope_chars", "song"].setChecked(True)
        song = ui.preview_events(tpl, "original", "song", sample, d._rules("original"), (1920, 1080))
        self.assertIn("超神ネイガー", song[1])
        d.reject()
        self.assertEqual(self.project.data["settings"], before)
        self.assertEqual(Project.open(self.project.root).data["settings"], before)

    def test_live_apply_all_categories_editable_fonts_and_exact_cancel(self):
        self.project.data["settings"]["unrelated"] = {"list": [1, {"x": 2}]}
        before = copy.deepcopy(self.project.data["settings"])
        d = self.open()
        d.rules["translation", "dash"].setCurrentIndex(1)
        self.assertEqual(self.project.settings()["display"]["punctuation"]["translation"]["dash"], "half")
        self.preview.assert_called_once_with()
        d.scope_boxes["translation", "scope_chars", "song"].setChecked(True)
        d.all_fonts["translation"].setEditText("TypedFont")
        d.apply_all("translation")
        fonts = dict(d._rules("translation").fonts)
        self.assertEqual(fonts, {category: "TypedFont" for category in PUNCTUATION_CATEGORIES})
        self.assertIn("song", d._rules("translation").scope_chars)
        d.reject()
        self.assertEqual(self.project.data["settings"], before)
        self.assertEqual(Project.open(self.project.root).data["settings"], before)
        self.assertTrue(self.refresh.called)

    def test_accept_keeps_and_escape_restores(self):
        d = self.open()
        d.rules["original", "tilde"].setCurrentIndex(2)
        d.accept()
        self.assertEqual(self.project.settings()["display"]["punctuation"]["original"]["tilde"], "half")
        before = copy.deepcopy(self.project.data["settings"])
        d = self.open()
        d.show()
        d.rules["original", "tilde"].setCurrentIndex(3)
        QTest.keyClick(d, Qt.Key.Key_Escape)
        self.assertEqual(self.project.data["settings"], before)

    def test_background_debounce_aspect_report_and_kind_styles(self):
        d = self.open()
        d.samples["original"].setPlainText("ハロー〜！")
        d.tabs.setCurrentIndex(1)
        d.kind_box.setCurrentIndex(1)
        self.assertEqual(self.calls, [])
        self.wait(lambda: len(self.calls) == 1 and d._job is None)
        _styles, kw, worker = self.calls[0]
        self.assertNotEqual(worker, threading.get_ident())
        self.assertEqual(kw["playres"], (1440, 1080))
        self.assertTrue(all(",OP.JP," in line for line in kw["event_lines"]))
        # The 原文/显示 labels are drawn by the dialog, never in the subtitle font (no false missing glyphs).
        self.assertTrue(kw["event_lines"][0].endswith("}ハロー〜！"))
        self.assertFalse(any("原文" in line or "显示" in line for line in kw["event_lines"]))
        self.assertEqual(kw["width"], ui.PREVIEW_WIDTH)
        pixmap = d.preview.pixmap()
        self.assertFalse(pixmap.isNull())
        self.assertLessEqual(pixmap.width(), max(200, d.preview.width() - 8) + 64)
        self.assertIn("缺失字体", d.warning.text())
        self.assertIn("缺", d.warning.text())

    def test_preview_layers_and_raw_use_core_semantics(self):
        template = self.project.template()
        rule = PunctuationRules(tilde="half", fonts=(("tilde", "Arial"), ("long_vowel", "Arial")),
                                scope_chars=("dialogue",), scope_fonts=("song",))
        lines = ui.preview_events(template, "original", "song", "ハロー〜{@Raw=〜}", rule, (1920, 1080))
        self.assertIn(r"\fnArial", lines[1])
        self.assertIn("〜", lines[1])
        self.assertTrue(ui.VisibleText(lines[1].split(",,", 1)[-1]).text.endswith("〜"))
        lines = ui.preview_events(template, "original", "dialogue", "危な〜い", rule, (1920, 1080))
        self.assertIn("危な~い", lines[1])
        self.assertNotIn(r"\fnArial", lines[1])
        lines = ui.preview_events(template, "translation", "other", "甲〜", rule, (1920, 1080))
        self.assertIn(",CN.Top,", lines[1])

    def test_profile_save_select_modified_default_export_import_delete(self):
        d = self.open()
        with patch.object(ui.QInputDialog, "getText", return_value=("标准", True)):
            d.save_profile()
        self.assertEqual(d.profile_box.currentText(), "标准")
        d.rules["original", "tilde"].setCurrentIndex(2)
        self.assertEqual(d.profile_box.currentText(), "标准（已修改）")
        with patch.object(ui.QInputDialog, "getText", return_value=("标准", True)), \
                patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            d.save_profile()
        self.assertEqual(d.profile_box.currentText(), "标准")
        d.default_profile()
        self.assertEqual(d.profile_box.currentText(), "标准（默认）")
        path = self.path / "export.json"
        with patch.object(ui.QFileDialog, "getSaveFileName", return_value=(str(path), "")):
            d.export_profile()
        self.assertTrue(path.is_file())
        before = copy.deepcopy(self.project.data["settings"])
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            d.delete_profile()
        self.assertEqual(self.project.data["settings"], before)
        self.assertEqual(d.profile_box.currentText(), "（未存为方案）")
        with patch.object(ui.QFileDialog, "getOpenFileName", return_value=(str(path), "")):
            d.import_profile()
        self.assertEqual(d.profile_box.currentData(), "标准")
        d.rules["original", "tilde"].setCurrentIndex(0)
        d.select_profile(d.profile_box.findData("标准"))
        self.assertEqual(d._rules("original").tilde, "half")

    def test_f1_and_button_help(self):
        d = self.open()
        d.show()
        self.app.processEvents()
        self.assertTrue(all(b.property("help_anchor") == "display-rules" for b in d.findChildren(QPushButton)))
        with patch.object(help_ui, "open_manual") as opened:
            QTest.keyClick(d.samples["translation"], Qt.Key.Key_F1)
            opened.assert_called_once_with("display-rules")

    def test_import_conflict_overwrite_rename_and_cancel(self):
        profile = create_profile("重复", PunctuationRules(tilde="half"), PunctuationRules())
        self.store.save(profile)
        path = self.path / "import.json"
        self.store.export_to("重复", path)
        changed = create_profile("重复", PunctuationRules(tilde="wave"), PunctuationRules())
        self.store.save(changed, overwrite=True)
        d = self.open()
        for choice in ("取消", "改名…", "覆盖"):
            def respond(box):
                next(b for b in box.buttons() if b.text() == choice).click()
                return 0
            with patch.object(ui.QFileDialog, "getOpenFileName", return_value=(str(path), "")), \
                    patch.object(QMessageBox, "exec", respond), \
                    patch.object(ui.QInputDialog, "getText", return_value=("改名方案", True)):
                d.import_profile()
            if choice == "取消":
                self.assertEqual(self.store.load("重复")["translation"]["tilde"], "wave")
            elif choice == "改名…":
                self.assertEqual(self.store.load("改名方案")["translation"]["tilde"], "half")
            else:
                self.assertEqual(self.store.load("重复")["translation"]["tilde"], "half")

    def test_use_current_and_restore_sample_do_not_change_settings(self):
        from types import SimpleNamespace
        self.session.doc = SimpleNamespace(group=lambda gid: SimpleNamespace(members=["a", "b"]),
            members={"a": SimpleNamespace(lane="translation", text="现在这一句～"),
                     "b": SimpleNamespace(lane="original", text="ハロー")})
        d = self.open(current_gid="g")
        before = copy.deepcopy(self.project.data["settings"])
        d.use_current("translation")
        self.assertEqual(d.samples["translation"].toPlainText(), "现在这一句～")
        d.samples["translation"].setPlainText(ui.SAMPLES["translation"])
        self.assertEqual(self.project.data["settings"], before)

    def test_close_running_job_keeps_reference_until_return(self):
        started, released = threading.Event(), threading.Event()
        def blocked(*args, **kw):
            started.set()
            released.wait(1)
            return self.fake(*args, **kw)
        d = self.open(renderer=blocked)
        d._render()
        self.wait(started.is_set)
        job = d._job
        self.assertFalse(job.autoDelete())
        d.reject()
        self.assertIs(d._job, job)
        released.set()
        self.wait(lambda: d._job is None)
        self.assertEqual(d.result(), d.DialogCode.Rejected)

    def test_open_time_with_real_font_database(self):
        started = time.perf_counter()
        d = self.open()
        d.show()
        self.app.processEvents()
        elapsed = time.perf_counter() - started
        print(f"P3 offscreen open (real QFontDatabase, fake renderer): {elapsed:.4f} s; families={d.font_model.rowCount()-1}")
        self.assertLess(elapsed, 2.0)

    def test_open_time_large_shared_font_model(self):
        with patch.object(ui.QFontDatabase, "families", return_value=[f"Family {i:04d}" for i in range(2000)]):
            started = time.perf_counter()
            d = self.open()
            d.show()
            self.app.processEvents()
            elapsed = time.perf_counter() - started
        self.assertEqual(d.font_model.rowCount(), 2001)
        self.assertEqual(d.scope_boxes["translation", "scope_fonts", "song"].isChecked(), True)
        print(f"P3 offscreen open (2000-family test double, fake renderer): {elapsed:.4f} s")
        self.assertLess(elapsed, 2.0)


@unittest.skipUnless(shutil.which("ffmpeg"), "requires host ffmpeg/libass")
class LibassSmoke(unittest.TestCase):
    def test_real_category_font_and_episode_aspect(self):
        path = Path(tempfile.mkdtemp(prefix="punctuation-libass-")) / "preview.png"
        styles = default_styles_ass()
        template = AssDocument.parse(styles)
        rule = PunctuationRules(tilde="half", fonts=(("tilde", "SubKit Missing Punctuation Font"),))
        lines = ui.preview_events(template, "original", "dialogue", "危な〜い！", rule, (1440, 1080))
        result = templates.render_scene(styles, path, event_lines=lines, playres=(1440, 1080), width=320, guides=False)
        self.assertTrue(path.exists() and path.stat().st_size > 1000)
        image = QImage(str(path))
        self.assertEqual((image.width(), image.height()), (320, 240))
        self.assertIn("SubKit Missing Punctuation Font", result.missing_fonts)
        print("P3 real ffmpeg/libass smoke: 320x240, inline missing punctuation font reported")
