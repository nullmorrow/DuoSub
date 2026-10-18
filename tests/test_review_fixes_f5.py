"""F5: offscreen result wording, shared group counts and main-window filtering."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from PySide6.QtCore import QModelIndex
from PySide6.QtWidgets import QApplication, QDialog, QLabel
from subkit.autotime import Segment
from subkit.core.merge import group_counts
from subkit.core.workspace import Project
from subkit.model.document import Document, Group, Member, Run
from subkit.model.validate import Report
from subkit_app.views.autotime import quality_text, AutoTimeResultDialog, AutoTimeDetailsDialog
from subkit_app.views.check_result import CheckResultDialog
from subkit_app.views.group_table import GroupFilter, GroupTableModel
from tests import test_convert_ui as fixtures


def sample():
    doc = Document("F5")
    def add(gid, lanes, kind="dialogue", state="auto"):
        g = Group(gid, kind, state=state); doc.groups.append(g)
        for i, (lane, text, role, source_kind) in enumerate(lanes):
            mid = gid + str(i)
            doc.members[mid] = Member(mid, gid, "en", lane, role, 1000, 2000,
                                      runs=[Run(text)], source_kind=source_kind)
            g.members.append(mid)
    body = lambda lane: (lane, "text", "dialogue", "Dialogue")
    add("both", [body("original"), body("translation")])
    add("translation", [body("translation"), ("original", "note", "comment", "Dialogue")], state="confirmed")
    add("original", [body("original"), ("translation", r"{\i1}\N  ", "top", "Dialogue")])
    add("source_comment", [body("original"), ("translation", "note", "dialogue", "Comment")])
    add("empty", [("original", r"{\i1}", "dialogue", "Dialogue")])
    add("song", [body("translation")], kind="song")
    add("sign", [body("translation")], kind="sign")
    add("info", [("translation", "note", "comment", "Comment")], kind="info")
    return doc


def episode(doc):
    return SimpleNamespace(document=doc, sources={}, project=SimpleNamespace(languages={"original":"en","translation":"ja"}),
                           data={"merge":{"summary":{"offset_ms":-205,"cuts":[(1200,500)],
                                                         "groups":{"translation_only":999}}}})


class ResultText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_counts_and_filter_visible_non_comment_lanes(self):
        doc = sample()
        self.assertEqual(group_counts(doc), dict(total=8, dialogue=5, both=1, translation_only=1,
                                                original_only=2, song=1, sign_info=2))
        proxy = GroupFilter(); model = GroupTableModel(doc); proxy.setSourceModel(model)
        proxy.set_mode("只有一种语言的对白")
        self.assertEqual([proxy.data(proxy.index(i,0), 256) for i in range(proxy.rowCount())],
                         ["translation", "original", "source_comment"])
        proxy.set_mode("只看对白"); self.assertEqual(proxy.rowCount(),5)
        proxy.set_mode("全部"); self.assertEqual(proxy.rowCount(),8)

    def test_merge_counts_fresh_warning_and_collapsible_ledger(self):
        ep = episode(sample())
        ep.sources = {"secret":SimpleNamespace(lane="original", events=[SimpleNamespace(text="abc")])}
        report = Report(ledger={"secret":{"paired":2,"unpaired":1}})
        dialog = CheckResultDialog(ep, report, merged=True)
        try:
            text = dialog.text.toPlainText()
            self.assertIn("这一集共 8 组：",text)
            self.assertIn("· 对白 5 组：两边都有 1 组，只有翻译（日语） 1 组，只有原声（英语） 2 组",text)
            self.assertIn("· 歌曲 1 组，屏幕字等 2 组（这些本来就不一定配对）",text)
            self.assertIn("整体差 -0.20 秒，已按翻译（日语）的时间对齐。",text)
            self.assertIn("两个版本剪辑不同",text)
            self.assertIn("检查：没有问题。",text)
            self.assertNotIn("每个字都有着落",text)
            self.assertNotIn("secret",text)
            self.assertTrue(dialog.single_language_warning.text().startswith("有 3 组对白只有一种语言"))
            self.assertTrue(dialog.details_text.isHidden())
            self.assertIn("原声（英语）字幕 3 字：配进组 2，没配上（单独成行） 1。",dialog.details_text.toPlainText())
            dialog.details_toggle.click(); self.assertFalse(dialog.details_text.isHidden())
            dialog.details_toggle.click(); self.assertTrue(dialog.details_text.isHidden())
        finally: dialog.deleteLater()

    def test_zero_non_dialogue_and_single_language_omitted(self):
        doc=sample(); doc.groups=doc.groups[:1]
        dialog=CheckResultDialog(episode(doc),Report(),merged=True)
        try:
            self.assertNotIn("歌曲",dialog.text.toPlainText())
            self.assertFalse(hasattr(dialog,"single_language_button"))
            self.assertFalse(dialog.details_toggle.isChecked())
        finally: dialog.deleteLater()

    def test_check_result_keeps_ledger_and_issue_categories(self):
        ep=episode(Document("test")); rep=Report()
        rep.add("error","ref.member","secret")
        dialog=CheckResultDialog(ep,rep)
        try:
            self.assertIn("每个字都有着落",dialog.text.toPlainText())
            self.assertIn("检查：发现 1 个问题",dialog.text.toPlainText())
            self.assertIn("内部数据（请把检查结果发给开发者）",dialog.text.toPlainText())
        finally: dialog.deleteLater()

    def test_merge_persists_structured_group_counts(self):
        with tempfile.TemporaryDirectory(dir=os.environ["TEMP"]) as folder:
            root=Path(folder); project=Project.create(root/"project","F5",original="en",translation="ja")
            ep=project.add_episode("E01")
            for lane in ("translation","original"):
                path=root/(lane+".srt"); path.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n",encoding="utf-8")
                ep.add_source(path,lane=lane,lang="en")
            ep.merge()
            self.assertEqual(ep.data["merge"]["summary"]["groups"],group_counts(ep.document))
            self.assertEqual(Project.open(project.root).episode("E01").data["merge"]["summary"]["groups"]["both"],1)

    def test_quality_grades_changed_unchanged_preview_and_unreliable(self):
        for grade, sentence in (("good","各句的时间都和声音吻合。"),
                                ("fair","少数句子本身的开始、结束时间不太准，自动调轴不改单句。"),
                                ("bad","不少句子本身的开始、结束时间就不准（不是整体偏移），自动调轴不改单句；需要的话在列表里逐句调。")):
            p=SimpleNamespace(quality=SimpleNamespace(max_offset_ms=512,grade=grade),
                              segments=[Segment(0,1000,0,confidence="high")],changes=(),excluded=())
            self.assertEqual(quality_text(p),"整体：已对齐（调之前最多差 0.51 秒）。\n单句："+sentence)
            self.assertEqual(quality_text(p,changed=False),"整体：本来就对得上（最多差 0.51 秒）。\n单句："+sentence)
            self.assertIn("预览",quality_text(p,preview_only=True))
            self.assertNotIn("少女革命",quality_text(p))
            p.segments=[Segment(0,1000,500,confidence="low")]
            self.assertIn("没有找到把握大的时间差，没有改动",quality_text(p,changed=False))
            p.segments=[Segment(0,1000,500,confidence="high")];p.excluded=(object(),)
            self.assertNotIn("本来就对得上",quality_text(p,changed=False))


class FilterFlow(unittest.TestCase):
    setUpClass=fixtures.ConversionUI.__dict__["setUpClass"]
    _dispose=fixtures.ConversionUI._dispose
    def setUp(self): fixtures.ConversionUI.setUp(self)

    def test_background_result_button_switches_main_combo_and_closes(self):
        self.doc=sample(); ep=episode(self.doc)
        self.window.model=GroupTableModel(self.doc)
        self.window.proxy.setSourceModel(self.window.model)
        self.window.session=None
        def inspect(dialog):
            dialog.single_language_button.click()
            self.assertEqual(dialog.result(),QDialog.DialogCode.Accepted)
            return dialog.result()
        with patch.object(self.window,"_finish_episode_job",return_value=SimpleNamespace(mode="merge")), \
             patch.object(self.window,"close_session"),patch.object(self.window,"_start_session"), \
             patch.object(CheckResultDialog,"exec",inspect):
            self.window._episode_done(ep,Report())
        self.assertEqual(self.window.filter_box.currentText(),"只有一种语言的对白")
        self.assertEqual(self.window.proxy.rowCount(),3)


class AutoTimeSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_result_header_and_two_quality_lines(self):
        from PySide6.QtWidgets import QMainWindow
        parent = QMainWindow()
        preview = SimpleNamespace(quality=SimpleNamespace(max_offset_ms=512,grade="bad"),
                                  segments=[Segment(0,10000,512,confidence="high")],
                                  changes=(),excluded=(),protected_count=0,low_confidence_count=0)
        session = SimpleNamespace(editor=SimpleNamespace(history=SimpleNamespace(done=[])))
        dialog = AutoTimeResultDialog(session,preview,(3,6),None,None,parent)
        try:
            self.assertIn("对齐结果",[label.text() for label in dialog.findChildren(QLabel)])
            self.assertTrue(dialog.quality_label.text().startswith("整体：已对齐"))
            self.assertIn("\n单句：不少句子",dialog.quality_label.text())
            self.assertNotIn("少女革命",dialog.quality_label.text())
        finally:
            dialog.deleteLater();parent.deleteLater()


class DetailsFooter(unittest.TestCase):
    from tests import test_autotime_ui as fixtures
    setUpClass = fixtures.AutoTimeUI.__dict__["setUpClass"]
    _dispose = fixtures.AutoTimeUI._dispose
    select = fixtures.AutoTimeUI.select
    def setUp(self): self.fixtures.AutoTimeUI.setUp(self)

    def test_details_footer_uses_preview_alignment_and_same_grade_sentence(self):
        with patch("subkit.autotime.estimate",return_value=self.map):
            dialog=AutoTimeDetailsDialog(self.session,self.window.wave.audio_features(),
                                         self.window.player,self.window.settings,self.window)
            self.fixtures.wait_preview(dialog)
        try:
            self.assertTrue(dialog.note.text().startswith("对齐结果\n整体：应用后将对齐"))
            self.assertIn("预览",dialog.note.text())
            self.assertIn("\n单句：",dialog.note.text())
            self.assertNotIn("少女革命",dialog.note.text())
        finally: dialog.deleteLater()
