"""Traditional display rules: Windows host OpenCC/libass and offscreen Qt."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from subkit import templates as T
from subkit.ass.document import AssDocument
from subkit.ass.text import VisibleText
from subkit.core.display import DisplayRules, traditional_text
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.people import Character, CharacterBook
from subkit.core.review import build_packet
from subkit.core.workspace import Project
from tests.test_pinyin_export import document


def rule(variant="tw", phrases=False, enabled=True):
    return DisplayRules.from_settings({"display": {"zh_traditional": dict(
        enabled=enabled, variant=variant, phrases=phrases)}})


class TraditionalDisplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="c6-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.styles = T.build_styles(T.FontSet("test", "test", "Microsoft YaHei", "Microsoft YaHei", "Arial",
                                             pinyin_font="Arial"), T.palette("classic"))
        self.template = AssDocument.parse(self.styles)

    def export(self, doc, display=None, **kw):
        return build_ass(doc, {}, self.template, self.path, ExportOptions(display=display or rule(), **kw),
                         measure_cache=self.path / "render.sqlite3")

    def test_language_and_tags_drawings_are_untouched(self):
        raw = r"{\fn国語 软件}国语 软件{\p1}m 0 0 l 10 10 国语{\p0}广场\N发布"
        doc = document(raw)
        before = copy.deepcopy(doc)
        ass, rep = self.export(doc)
        self.assertEqual(ass.events[0].text,
                         r"{\fn国語 软件}國語 軟件{\p1}m 0 0 l 10 10 国语{\p0}廣場\N發佈")
        self.assertEqual(rep.traditional_lines, 1)
        self.assertEqual(doc, before)
        for lang in ("ja", "en", "yue"):
            text = "国語 発売 広場"
            ass, rep = self.export(document(text, lang=lang))
            self.assertEqual(ass.events[0].text, text)
            self.assertEqual(rep.traditional_lines, 0)
        for lang, lane in (("zh", "original"), ("zh-CN", "translation")):
            ass, _ = self.export(document("国语", lang=lang, lane=lane))
            self.assertEqual(ass.events[0].text, "國語")

    def test_variants_and_phrases_and_boundaries(self):
        self.assertEqual(rule("tw").convert("里面软件"), "裡面軟件")
        self.assertEqual(rule("hk").convert("里面软件"), "裏面軟件")
        self.assertEqual(rule("std").convert("里面软件"), "裏面軟件")
        for variant in ("tw", "hk", "std"):
            text, bounds = traditional_text("互联网软件", rule(variant, True))
            self.assertEqual(text, "網際網路軟體")
            self.assertEqual(len(bounds), 6)
            self.assertEqual((bounds[0], bounds[-1]), (0, 6))
            self.assertEqual(bounds, sorted(bounds))

    def test_names_and_review_packet(self):
        doc = document("国语软件")
        doc.members["m_1"].speaker = {"character": "chr_1", "label": "张"}
        book = CharacterBook(self.path / "characters.json")
        book.characters = [Character("chr_1", {"zh-Hans": "张软件", "zh-Hant": "專用繁中名"})]
        for format in ("short", "full", "given"):
            ass, _ = build_ass(doc, {}, self.template, self.path,
                               ExportOptions(display=rule(), name_format=format), character_book=book)
            self.assertEqual(ass.events[0].name, "專用繁中名")
        del book.characters[0].names["zh-Hant"]
        ass, _ = build_ass(doc, {}, self.template, self.path, ExportOptions(display=rule(phrases=True)),
                           character_book=book)
        self.assertEqual(ass.events[0].name, "張軟體")
        md, jl = build_packet(doc, episode="E1", chars=book, display=rule(phrases=True), pinyin_mode="PY")
        rec = json.loads(jl)
        self.assertEqual(rec["translation"][0]["text"], "國語軟體")
        self.assertIn("張軟體", md)
        self.assertEqual(doc.members["m_1"].text, "国语软件")

    def test_cache_toggle_and_soma_off_bytes(self):
        doc = document("国语")
        cache = ExportCache()
        def build(display):
            return build_ass(doc, {}, self.template, self.path, ExportOptions(display=display), cache=cache)
        off, _ = build(rule(enabled=False))
        on, rep = build(rule())
        self.assertEqual(rep.laid_out, 1)
        same, rep = build(rule())
        self.assertEqual(rep.laid_out, 0)
        self.assertEqual(same.serialize(), on.serialize())
        after, rep = build(rule(enabled=False))
        self.assertEqual(rep.laid_out, 1)
        self.assertEqual(off.serialize(), after.serialize())
        path = self.path / "soma"
        shutil.copytree(ROOT / "test_projects/soma", path, copy_function=shutil.copyfile)
        project = Project.open(path)
        episode = project.episode(project.episode_ids()[0])
        baseline, _ = episode.build_export()
        project.data.setdefault("settings", {}).setdefault("display", {})["zh_traditional"] = dict(
            enabled=False, variant="hk", phrases=True)
        after, _ = episode.build_export()
        baseline.save(self.path / "before.ass")
        after.save(self.path / "after.ass")
        self.assertEqual((self.path / "before.ass").read_bytes(), (self.path / "after.ass").read_bytes())
        print("C6 soma off: byte-identical, bytes=" + str((self.path / "after.ass").stat().st_size))

    def test_karaoke_timing_survives_phrase_expansion(self):
        from subkit.core import karaoke as kara
        from subkit.core.ops import Editor
        doc = document("互联网软件")
        m = doc.members["m_1"]
        doc.groups[0].kind = "song"
        m.role = "song_cn"
        beats = kara.derive(doc, m)
        for i, beat in enumerate(beats):
            beat.t0, beat.t1, beat.by = i * 137, (i + 1) * 137, "tap"
        Editor(doc, {}).set_karaoke(m.id, beats)
        saved = copy.deepcopy(doc)
        off, _ = self.export(doc, rule(enabled=False))
        on, _ = self.export(doc, rule(phrases=True))
        tags = re.findall(r"\\k[fo]?\d+", off.events[0].text)
        self.assertTrue(tags)
        self.assertEqual(tags, re.findall(r"\\k[fo]?\d+", on.events[0].text))
        self.assertEqual((off.events[0].start, off.events[0].end), (on.events[0].start, on.events[0].end))
        self.assertEqual(VisibleText(on.events[0].text).text, "網際網路軟體")
        self.assertEqual(doc, saved)

    def test_pinyin_measures_displayed_base_and_preserves_anchors(self):
        from subkit.core.pinyin_notes import annotate_pinyin
        doc = document("互联网")
        before = copy.deepcopy(doc)
        ass, rep = self.export(doc, rule(phrases=True), pinyin_mode="PY")
        self.assertEqual(VisibleText(ass.events[0].text).text, "網際網路")
        self.assertEqual(rep.pinyin_lines, 3)
        self.assertFalse(any("找不到底字" in w for w in rep.warnings))
        self.assertEqual(doc, before)
        reference = document("網際網路")
        reference.annotations = copy.deepcopy(doc.annotations)
        bounds = traditional_text(doc.members["m_1"].text, rule(phrases=True))[1]
        for a in reference.annotations.values():
            anchor = a.anchor[0]
            anchor["a"], anchor["b"] = bounds[anchor["a"]], bounds[anchor["b"]]
            a.base = reference.members["m_1"].text[anchor["a"]:anchor["b"]]
            a.parts = [[0, anchor["b"] - anchor["a"], a.text]]
        expected, expected_rep = self.export(reference, rule(enabled=False), pinyin_mode="PY")
        rows = [e.text for e, o in zip(ass.events, rep.owners) if o["kind"] == "pinyin"]
        expected_rows = [e.text for e, o in zip(expected.events, expected_rep.owners) if o["kind"] == "pinyin"]
        self.assertEqual(rows, expected_rows)
        print("C6 libass: converted-base pinyin layout equals explicitly converted reference (0 px)")

    def test_project_preview_and_offscreen_editor_menu(self):
        from PySide6.QtCore import QSettings, QRect
        from PySide6.QtWidgets import QApplication, QLabel
        from subkit_app import mainwindow as mw, help as help_ui
        from subkit_app.session import EpisodeSession, PreviewJob
        from subkit_app.watch import InlineTextEditor
        from tests.test_convert_ui import PlayerDouble
        app = QApplication.instance() or QApplication([])
        project = Project.create(self.path / "project", "C6", template_text=self.styles)
        episode = project.add_episode("E1")
        episode.document = document("国语软件")
        settings = QSettings(str(self.path / "settings.ini"), QSettings.Format.IniFormat)
        with patch.object(mw, "MpvWidget", PlayerDouble), patch.object(mw, "QSettings", return_value=settings), \
                patch.object(mw.MainWindow, "_open_last", Mock()):
            window = mw.MainWindow()
        session = EpisodeSession(project, episode)
        window.project, window.session = project, session
        try:
            with patch.object(session, "request_preview") as preview:
                window.act_traditional.setChecked(True)
                preview.assert_called_once()
            window.editor.set_group(episode.document, episode.document.groups[0], None, None)
            self.assertEqual(window.editor.cn_edits[0][1].toPlainText(), "国语软件")
            notes = [l for l in window.editor.findChildren(QLabel) if l.property("traditional_member")]
            self.assertTrue(any(not l.isHidden() for l in notes))
            window.traditional_actions["hk"].trigger()
            window.act_traditional_phrases.setChecked(True)
            saved = rule("hk", True)
            self.assertEqual(DisplayRules.from_settings(Project.open(project.root).settings()), saved)
            ass, rep = episode.build_export()
            self.assertEqual(ass.events[0].text, "國語軟體")
            self.assertEqual((rep.traditional_lines, rep.traditional_variant), (1, "香港用字"))
            job = PreviewJob(session, 0, self.path / "preview.ass")
            done, errors = [], []
            job.signals.done.connect(lambda *args: done.append(args))
            job.signals.failed.connect(lambda *args: errors.append(args))
            job.run()
            self.assertFalse(errors, errors)
            self.assertEqual(done[0][2].events[0].text, ass.events[0].text)
            inline = InlineTextEditor(episode.document, ["m_1"], "m_1", QRect(0, 0, 600, 60), 20, saved, window)
            try:
                self.assertEqual(inline.edit.text(), "国语软件")
                self.assertFalse(inline.traditional_note.isHidden())
            finally:
                inline._cancel = True
                inline.close()
                inline.deleteLater()
            self.assertEqual(help_ui.resolve_anchor(window, focus=window.traditional_menu), "readings")
            menu = window.act_traditional.associatedObjects()[-1]
            menu.setActiveAction(window.act_traditional)
            with patch.object(QApplication, "activePopupWidget", return_value=menu):
                self.assertEqual(help_ui.resolve_anchor(window), "readings")
        finally:
            session.close()
            window.session = None
            window.close()
            window.deleteLater()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
