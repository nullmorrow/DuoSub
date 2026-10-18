"""TP2: neutral language samples, host libass, and offscreen canvas backgrounds."""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6 import __version__ as QT_VERSION
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication
from subkit import templates as T
from subkit.ass import AssDocument
from subkit.ass.text import VisibleText
from subkit.proc import run
from subkit_app.views.dialogs import NewProjectDialog
from subkit_app.views.template_editor import TemplateDialog, TemplateEditor


class PreviewUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def editor(self, **kwargs):
        with patch.object(TemplateEditor, "_schedule"), patch.object(TemplateEditor, "_grab_frame"):
            editor = TemplateEditor(**kwargs)
        self.addCleanup(editor.close)
        return editor

    def test_language_combos_schedule_and_render_selected_languages(self):
        with patch.object(TemplateEditor, "_schedule") as schedule:
            dialog = NewProjectDialog()
            self.addCleanup(dialog.close)
            for code in ("ja", "zh-Hant", "ko", "en", "vi"):
                for combo in (dialog.original, dialog.translation):
                    if combo.currentData() == code:
                        combo.setCurrentIndex(combo.findData("en" if code != "en" else "ja"))
                    schedule.reset_mock()
                    combo.setCurrentIndex(combo.findData(code))
                    schedule.assert_called_once()
                    editor = dialog.editor
                    with patch.object(editor, "_run_job") as job, patch.object(T, "render_scene") as render:
                        editor._render()
                        job.call_args.args[0](None)
                        self.assertEqual(render.call_args.kwargs["original_lang"], dialog.original.currentData())
                        self.assertEqual(render.call_args.kwargs["translation_lang"], dialog.translation.currentData())
        self.assertEqual(T.preview_sentence("unknown", 0), T.preview_sentence("en", 0))

    def test_fixed_japanese_readings_and_all_spec_sentences(self):
        self.assertEqual(T.preview_sentence("ja", 0),
                         ("駅の前で待っています。", [(0, 1, "えき"), (2, 3, "まえ"), (4, 5, "ま")]))
        self.assertEqual(T.preview_sentence("ja", 2), ("三日後", [(0, 3, "みっかご")]))
        self.assertEqual(T.preview_sentence("ja", 3),
                         ("空に向かって歌おう", [(0, 1, "そら"), (2, 3, "む"), (6, 7, "うた")]))
        for line in (ROOT / "docs/TEMPLATE_PREVIEW.md").read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == 5 and cells[0].split()[-1] in T.PREVIEW_SENTENCES:
                self.assertEqual(tuple(cells[1:]), T.PREVIEW_SENTENCES[cells[0].split()[-1]])

    def test_background_defaults_and_initial_video_request(self):
        editor = self.editor()
        self.assertEqual(editor.bg_box.currentData(), "dark_gradient")
        self.assertEqual(editor.bg_box.itemText(editor.bg_box.findData("gradient")), "背景：浅灰渐变")
        with patch.object(TemplateEditor, "_schedule"), patch.object(TemplateEditor, "_grab_frame") as grab:
            dialog = TemplateDialog(original_lang="ko", translation_lang="vi", video=Path("episode.mp4"))
            self.addCleanup(dialog.close)
            self.assertEqual(dialog.editor.bg_box.currentData(), "video")
            self.assertEqual((dialog.editor.original_lang, dialog.editor.lang), ("ko", "vi"))
            grab.assert_called_once_with(Path("episode.mp4"))

    def test_file_template_uses_project_languages_and_available_episode_video(self):
        from subkit_app.mainwindow import MainWindow
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "episode.mp4"
            video.touch()
            project = SimpleNamespace(languages={"original": "vi", "translation": "ko"}, data={},
                                      template=lambda: None, episode_ids=lambda: ["E1"],
                                      episode=lambda eid: SimpleNamespace(video=video))
            window = SimpleNamespace(project=project, session=None)
            with patch("subkit_app.views.template_editor.TemplateDialog") as dialog:
                dialog.return_value.exec.return_value = 0
                dialog.DialogCode.Accepted = 1
                MainWindow.change_template(window)
                self.assertEqual(dialog.call_args.kwargs["video"], video)
                self.assertEqual(dialog.call_args.kwargs["original_lang"], "vi")
                self.assertEqual(dialog.call_args.kwargs["translation_lang"], "ko")

    def test_zoom_canvas_corners_with_gradients_and_video_frame(self):
        editor = self.editor(video=Path("episode.mp4"))
        editor.preview.resize(800, 450)
        editor.chk_zoom.setChecked(True)
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp) / "frame.png"
            image = QImage(1920, 1080, QImage.Format.Format_RGB32)
            image.fill(QColor("#705030"))
            self.assertTrue(image.save(str(frame)))
            editor._bg_frame = frame
            for key in ("dark_gradient", "gradient", "video"):
                editor.bg_box.setCurrentIndex(editor.bg_box.findData(key))
                editor._last_pm = QPixmap(960, 330)
                editor._last_pm.fill(QColor("#606060"))
                editor._display_preview()
                canvas = editor.preview.pixmap().toImage()
                self.assertEqual(canvas.size(), editor.preview.size())
                for x, y in ((0, 0), (799, 0), (0, 449), (799, 449)):
                    self.assertNotEqual(canvas.pixelColor(x, y), QColor("black"), key)
                    if key == "video":
                        self.assertEqual(canvas.pixelColor(x, y), QColor("#705030"))
        print(f"TP2 PySide6 {QT_VERSION} offscreen: zoom canvas 800x450, all 12 gradient/video corners filled")


@unittest.skipUnless(shutil.which("ffmpeg"), "requires host ffmpeg/libass")
class PreviewLibass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.cache = self.path / "measure.sqlite3"
        self.styles = T.build_styles(T.font_set("utena_serif"), T.palette("utena"))

    def events(self, scene="dialogue", **kwargs):
        lines = T.scene_events(self.styles, scene, measure_cache=self.cache, **kwargs)
        return AssDocument.parse(T._HEAD + self.styles + T._EVH + "\n".join(lines)).events

    def test_languages_swap_texts_and_annotations_on_either_lane(self):
        for lang in ("ja", "zh-Hant", "ko", "en", "vi", "unknown"):
            events = self.events(original_lang=lang, translation_lang=lang, pinyin=True)
            bases = [e for e in events if e.style in ("CN", "JP")]
            self.assertEqual([e.text for e in bases], [T.preview_sentence(lang, 0)[0]] * 2)
            annotation_styles = {e.style for e in events if e not in bases}
            expected = {"CN.Ruby", "JP.Ruby"} if lang == "ja" else {"CN.Pinyin", "JP.Pinyin"} if lang == "zh-Hant" else set()
            self.assertEqual(annotation_styles, expected)
        for scene in T.SCENES:
            for original, translation in (("ja", "zh-Hant"), ("zh-Hans", "ja")):
                events = self.events(scene, original_lang=original, translation_lang=translation, pinyin=True)
                ruby = [e for e in events if e.style.endswith(".Ruby")]
                pinyin = [e for e in events if e.style.endswith(".Pinyin")]
                self.assertTrue(ruby and pinyin, scene)
                self.assertTrue(all(("CN" in e.style.split(".")) == (translation == "ja") for e in ruby))
                self.assertTrue(all(("CN" in e.style.split(".")) == (translation.startswith("zh")) for e in pinyin))
                sample = 3 if scene == "song" else 2 if scene == "top" else 0
                for reading in T.preview_sentence("ja", sample)[1]:
                    self.assertTrue(any(reading[2] in VisibleText(e.text).text for e in ruby), (scene, reading))
        self.assertFalse(any(e.style.endswith(".Pinyin") for e in self.events(pinyin=False)))

    def test_swapped_annotation_styles_reach_actual_renderer(self):
        captured = []
        def capture_run(cmd, **kwargs):
            if "-vf" in cmd and "ass=t.ass" in cmd[cmd.index("-vf") + 1]:
                captured.append(Path(kwargs["cwd"], "t.ass").read_text(encoding="utf-8-sig"))
            kwargs.setdefault("capture_output", True)
            return run(cmd, **kwargs)
        runner = SimpleNamespace(run=capture_run, check_cancelled=lambda: None)
        T.render_scene(self.styles, self.path / "swapped.png", original_lang="zh-Hant", translation_lang="ja",
                       pinyin=True, guides=False, width=320, measure_cache=self.cache, runner=runner)
        self.assertEqual(len(captured), 1)
        ass = AssDocument.parse(captured[0])
        for name in ("CN.Ruby", "JP.Pinyin"):
            self.assertIsNotNone(ass.style(name))
            self.assertTrue(any(e.style == name for e in ass.events))

    def test_actual_zoom_render_corners_and_video_grab(self):
        from subkit_app.views.template_editor import _Job
        for key in ("dark_gradient", "gradient"):
            for region in ("bottom", "top"):
                res = T.render_scene(self.styles, self.path / "preview.png", width=320, region=region,
                                     background=key, guides=False, measure_cache=self.cache)
                image = QImage(str(res.png))
                self.assertFalse(image.isNull())
                for x, y in ((0, 0), (image.width()-1, 0), (0, image.height()-1), (image.width()-1, image.height()-1)):
                    self.assertGreater(image.pixelColor(x, y).lightness(), 0)
        video = self.path / "sample.mp4"
        result = run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=0x705030:s=320x180:r=5:d=2",
                      "-pix_fmt", "yuv420p", str(video)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with patch.object(TemplateEditor, "_schedule"), patch.object(TemplateEditor, "_grab_frame"):
            editor = TemplateEditor(video=video)
        self.addCleanup(editor.close)
        def sync_job(fn, done):
            runner = _Job(1, fn).runner
            done(fn(runner))
        with patch.object(editor, "_run_job", side_effect=sync_job), patch.object(editor, "_schedule"):
            editor._grab_frame(video)
        self.addCleanup(editor._bg_frame.unlink)
        self.assertTrue(editor._bg_frame.exists())
        self.assertEqual(editor._bg(), editor._bg_frame)
        res = T.render_scene(self.styles, self.path / "video.png", region="bottom", width=320,
                             background=editor._bg(), guides=False, measure_cache=self.cache)
        self.assertFalse(QImage(str(res.png)).isNull())
        version = run(["ffmpeg", "-version"], capture_output=True).stdout.decode("utf-8", "replace").splitlines()[0]
        print(f"TP2 host {version}: both gradients top/bottom corners nonblack; synthetic 2s video frame extracted/rendered")
