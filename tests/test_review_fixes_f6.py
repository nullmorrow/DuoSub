"""F6: Windows host libass and offscreen Qt, isolated project copies."""
import copy
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass.document import AssDocument
from subkit.ass.text import VisibleText
from subkit.core.display import DisplayRules, single_line_text
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.readings import visible_map
from subkit.core.workspace import Project
from subkit.model.document import Annotation
from subkit.styles_default import default_styles_ass
from tests.test_pinyin_export import document

EVIDENCE = Path(os.environ["TEMP"]) / "f6-evidence"


def rules(lane="original", kind="dialogue", **kw):
    return DisplayRules(merge_lines=((lane, kind),), **kw)


class MergeDisplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="f6-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.template = AssDocument.parse(default_styles_ass())

    def build(self, doc, display=DisplayRules(), **kw):
        return build_ass(doc, {}, self.template, self.path, ExportOptions(display=display, **kw),
                         measure_cache=self.path / "measure.sqlite3")

    def test_boundaries_tags_soft_breaks_and_document_unchanged(self):
        raw = r"\N日{\i1}\N\N本\n語\h字\N"
        text, starts, ends = single_line_text(raw)
        self.assertEqual(text, r"  日{\i1}    本\n語\h字  ")
        self.assertEqual(starts, [2, 7, 8, 9, 10, 11, 14])
        self.assertEqual(ends, [0, 3, 8, 9, 10, 11, 12])
        doc = document(raw, lang="ja", lane="original", style="JP")
        before = copy.deepcopy(doc)
        ass, rep = self.build(doc, rules(), mode="preview")
        self.assertEqual(ass.events[0].text, text)
        original = visible_map(doc.members["m_1"])[1]
        shown = rep.owners[0]["ids"]
        for i, identity in enumerate(original):
            self.assertEqual(shown[starts[i]:ends[i+1]], [identity])
        self.assertTrue(all(shown[i] is None for i in (0, 1, 3, 4, 5, 6, 12, 13)))
        release, release_report = self.build(doc, rules())
        self.assertEqual(release.events[0].text, text.strip(" "))
        self.assertEqual(release_report.owners[0]["ids"], shown[2:-2])
        self.assertEqual(release_report.edge_spaces_removed, 4)
        self.assertEqual(doc, before)

    def test_each_kind_and_lane_changes_only_selected_lines(self):
        roles = ("dialogue", "top", "alt", "song", "song_jp", "song_cn", "sign", "info", "credits")
        for lane in ("original", "translation"):
            for role in roles:
                doc = document(r"一\N二\n三", lang="ja", lane=lane, style="JP")
                doc.members["m_1"].role = role
                for selected_lane in ("original", "translation"):
                    for kind in ("dialogue", "song", "other"):
                        rule = rules(selected_lane, kind)
                        ass, _ = self.build(doc, rule)
                        expected = r"一  二\n三" if rule.merge_for(doc.members["m_1"]) else r"一\N二\n三"
                        self.assertEqual(ass.events[0].text, expected, (lane, role, selected_lane, kind))

    def test_soma_off_export_byte_identical(self):
        path = self.path / "soma"
        shutil.copytree(ROOT / "test_projects/soma", path, copy_function=shutil.copyfile)
        project = Project.open(path)
        episode = project.episode(project.episode_ids()[0])
        before, _ = episode.build_export()
        project.data.setdefault("settings", {}).setdefault("display", {})["merge_lines"] = {
            lane: {kind: False for kind in ("dialogue", "song", "other")} for lane in ("original", "translation")}
        after, _ = episode.build_export()
        before.save(self.path / "before.ass"); after.save(self.path / "after.ass")
        self.assertEqual((self.path / "before.ass").read_bytes(), (self.path / "after.ass").read_bytes())
        print("F6 soma off byte-identical:", (self.path / "after.ass").stat().st_size, "bytes")

    def test_libass_ruby_matches_explicit_single_line_and_hit_identities(self):
        from subkit.core.layout_index import LayoutIndex
        doc = document(r"日\N本", lang="ja", lane="original", style="JP")
        doc.annotations = {
            "a": Annotation("a", "ruby", "m_1", [{"ins":"i_1", "a":0, "b":1}], "日", "にち", parts=[[0,1,"にち"]]),
            "b": Annotation("b", "ruby", "m_1", [{"ins":"i_1", "a":3, "b":4}], "本", "ほん", parts=[[0,1,"ほん"]])}
        before = copy.deepcopy(doc)
        ass, rep = self.build(doc, rules())
        reference = copy.deepcopy(doc)
        reference.members["m_1"].runs[0].text = "日  本"
        expected, expected_rep = self.build(reference)
        ruby = [e.text for e,o in zip(ass.events,rep.owners) if o["kind"] == "ruby"]
        self.assertEqual(len(ruby), 2)
        self.assertEqual(ruby, [e.text for e,o in zip(expected.events,expected_rep.owners) if o["kind"] == "ruby"])
        self.assertEqual(rep.owners[0]["ids"], [("i","i_1",0),None,None,("i","i_1",3)])
        index = LayoutIndex(ass, rep.owners, measure_cache=self.path / "measure.sqlite3")
        try:
            geoms = index.geometry(index.events_of_group("g_1"))
            self.assertTrue(geoms)
            base = next(g for g in geoms if g.kind == "base")
            for offset, identity in ((0,("i","i_1",0)), (3,("i","i_1",3))):
                box = base.chars[offset]
                hit = index.hit([base], box.cx, box.cy)
                self.assertEqual(hit.identity, identity)
        finally:
            index.close()
        self.assertEqual(doc, before)
        EVIDENCE.mkdir(exist_ok=True)
        ass.save(EVIDENCE / "merged-ruby.ass")
        print("F6 libass ruby: matches explicit single line, 0 px; identity mapping kept")

    def test_libass_pinyin_merge_matches_explicit_single_line(self):
        from subkit import templates as T
        self.template = AssDocument.parse(T.build_styles(T.FontSet("test","test","Microsoft YaHei",
            "Microsoft YaHei","Arial",pinyin_font="Arial"), T.palette("classic")))
        doc = document(r"行\N绿")
        before = copy.deepcopy(doc)
        ass, rep = self.build(doc, rules("translation"), pinyin_mode="PY")
        reference = copy.deepcopy(doc)
        reference.members["m_1"].runs[0].text = "行  绿"
        expected, expected_rep = self.build(reference, pinyin_mode="PY")
        rows = [e.text for e,o in zip(ass.events,rep.owners) if o["kind"] == "pinyin"]
        self.assertEqual(len(rows),2)
        self.assertEqual(rows,[e.text for e,o in zip(expected.events,expected_rep.owners) if o["kind"] == "pinyin"])
        self.assertEqual(doc,before)

    def test_kf_beats_after_break_and_traditional_composition(self):
        from subkit.core import karaoke as kara
        from subkit.core.ops import Editor
        doc = document(r"日\N本", lang="ja", lane="original", style="JP")
        doc.groups[0].kind = "song"; doc.members["m_1"].role = "song"
        beats = kara.derive(doc, doc.members["m_1"])
        for i, beat in enumerate(beats):
            beat.t0, beat.t1, beat.by = i*500, (i+1)*500, "tap"
        Editor(doc, {}).set_karaoke("m_1", beats)
        off, _ = self.build(doc)
        on, _ = self.build(doc, rules(kind="song"))
        self.assertEqual(re.findall(r"\\kf\d+", off.events[0].text), re.findall(r"\\kf\d+", on.events[0].text))
        self.assertRegex(on.events[0].text, r"日  \{\\kf\d+\}本")
        doc = document(r"互联网\N软件")
        ass, rep = self.build(doc, rules("translation", zh_traditional_enabled=True, zh_traditional_phrases=True))
        self.assertEqual(VisibleText(ass.events[0].text).text, "網際網路  軟體")
        self.assertEqual(rep.owners[0]["ids"][4:6], [None,None])

    def test_width_threshold_event_margins_and_cache_toggle(self):
        from subkit.ruby.measure import Measurer
        doc = document(r"Hello\Nworld", lang="en", lane="original", style="JP")
        m = doc.members["m_1"]
        with Measurer(self.template, wrap_style=2) as measure:
            width = measure.text_boxes([("JP", "Hello  world")])[0].width
        tx = 1920
        m.margins = [int(tx-width),0,0]
        self.template.style("JP").set("MarginR", 0)
        ass, rep = self.build(doc, rules())
        self.assertFalse(rep.warnings)
        m.margins[0] += 1
        cache = ExportCache()
        def build(display):
            return build_ass(doc, {}, self.template, self.path, ExportOptions(display=display), cache=cache,
                             measure_cache=self.path / "measure.sqlite3")
        on, rep = build(rules())
        self.assertEqual(rep.warnings, ["0:00:00.00 合成一行后太长，可能超出画面"])
        _, cached = build(rules())
        self.assertEqual(cached.laid_out, 0); self.assertEqual(cached.warnings, rep.warnings)
        off, rep = build(DisplayRules()); self.assertFalse(rep.warnings)
        self.assertEqual(off.events[0].text, m.text)
        print("F6 measured width", width, "px; equality passes, safe width minus 1 warns")

    def test_offscreen_project_settings_menu_preview_and_next_dialog(self):
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from subkit_app import mainwindow as mw, help as help_ui
        from subkit_app.session import EpisodeSession, PreviewJob
        from subkit_app.views.dialogs import AddEpisodeDialog
        from tests.test_convert_ui import PlayerDouble
        app = QApplication.instance() or QApplication([])
        project = Project.create(self.path / "project", "F6", template_text=default_styles_ass())
        episode = project.add_episode("E1")
        episode.document = document(r"日\N本", lang="ja", lane="original", style="JP")
        settings = QSettings(str(self.path / "settings.ini"), QSettings.Format.IniFormat)
        with patch.object(mw,"MpvWidget",PlayerDouble), patch.object(mw,"QSettings",return_value=settings), \
                patch.object(mw.MainWindow,"_open_last",Mock()):
            window = mw.MainWindow()
        session = EpisodeSession(project, episode)
        window.project, window.session = project, session
        before = copy.deepcopy(session.doc)
        try:
            with patch.object(session,"request_preview") as preview:
                window.merge_actions["original","dialogue"].setChecked(True)
                preview.assert_called_once()
            saved = Project.open(project.root).settings()["display"]
            dialog = AddEpisodeDialog(display_settings=saved, languages=project.languages)
            try:
                self.assertTrue(dialog.merge_checks["original","dialogue"].isChecked())
                self.assertEqual(sum(c.isChecked() for c in dialog.merge_checks.values()),1)
                self.assertEqual(dialog.merge_settings(), {"translation":dict(dialogue=False,song=False,other=False),
                    "original":dict(dialogue=True,song=False,other=False)})
            finally:
                dialog.close(); dialog.deleteLater()
            job = PreviewJob(session,0,self.path / "preview.ass")
            done, errors = [], []
            job.signals.done.connect(lambda *args:done.append(args)); job.signals.failed.connect(lambda *args:errors.append(args))
            job.run(); self.assertFalse(errors,errors)
            self.assertEqual(done[0][2].events[0].text,"日  本")
            self.assertEqual(help_ui.resolve_anchor(window,focus=window.merge_menu),"readings")
            self.assertEqual(session.doc,before)
            def accept(dialog):
                self.assertTrue(dialog.merge_checks["original","dialogue"].isChecked())
                dialog.merge_checks["translation","other"].setChecked(True)
                return AddEpisodeDialog.DialogCode.Accepted
            with patch.object(AddEpisodeDialog,"exec",accept), patch.object(window,"_start_episode_job"), \
                    patch.object(session,"request_preview"):
                window.add_episode()
            saved = Project.open(project.root).settings()["display"]
            next_dialog = AddEpisodeDialog(display_settings=saved)
            try:
                self.assertTrue(next_dialog.merge_checks["translation","other"].isChecked())
                self.assertTrue(next_dialog.merge_checks["original","dialogue"].isChecked())
            finally:
                next_dialog.close(); next_dialog.deleteLater()
        finally:
            session.close(); window.session=None; window.close(); window.deleteLater(); app.processEvents()


if __name__ == "__main__":
    unittest.main()
