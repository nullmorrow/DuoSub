"""F2/R21: offscreen geometry, time-shift boundaries and real libass colours."""
import copy
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QMenu
from tests import test_convert_ui as fixtures
from subkit.model.document import Member, Run, Annotation
from subkit.core.layout_index import LineGeom, Rect
from subkit.proc import run
from subkit.ass import AssDocument, Event
from subkit_app.watch import WatchController, VideoMapper
from subkit_app.views.shift_groups import ShiftGroupsDialog

EVIDENCE = Path(os.environ["TEMP"]) / "f2-evidence"


class ReviewF2(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select
    snapshot = fixtures.ConversionUI.snapshot
    context_menu = fixtures.ConversionUI.context_menu

    def bilingual(self):
        self.group("a", text=r"相信奇蹟…\N相信自己的心意能傳達到…", start=1000)
        m = Member("jp", "a", "ja", "original", "dialogue", 1031, 3000, style="JP",
                   runs=[Run("憎んでいるさ", ins="jp", a=0, b=6)])
        self.doc.members[m.id] = m
        self.doc.groups[0].members.append(m.id)
        self.doc.annotations["ruby"] = Annotation("ruby", "ruby", m.id, [{"ins":"jp","a":0,"b":1}], "憎", "にく", status="confirmed")
        self.doc.structure_changed()
        self.select("a")

    def test_editor_geometry_and_screenshots(self):
        for name in ("msyh.ttc", "msgothic.ttc"):
            QFontDatabase.addApplicationFont(str(Path(os.environ["WINDIR"]) / "Fonts" / name))
        self.app.setFont(QFont("Microsoft YaHei", 9))
        self.bilingual()
        w = self.window
        w.resize(1920, 1040)
        w.show()
        self.app.processEvents()
        EVIDENCE.mkdir(exist_ok=True)
        result = {}
        for opened in (False, True):
            if opened:
                w._dock_panels_right()
                w.people.raise_()
            else:
                w.dock_source.hide(); w.people.hide()
            self.app.processEvents()
            widgets = {"format": w.editor.fmt, "position": w.editor.position, "ruby": w.editor.ruby,
                       "original": w.editor.jp_edit, "translation": w.editor.cn_edits[0][1]}
            widgets.update({"alignment"+str(k):v for k,v in w.editor.position.alignments.items()})
            sizes = {}
            for key, widget in widgets.items():
                self.assertGreaterEqual(widget.height(), widget.sizeHint().height(), key)
                top = widget.mapTo(w.editor, QPoint()).y()
                self.assertGreaterEqual(top, 0, key)
                self.assertLessEqual(top + widget.height(), w.editor.height(), key)
                sizes[key] = [widget.height(), widget.sizeHint().height(), top]
            self.assertLessEqual(w.editor.ruby.geometry().bottom(), w.editor.jp_edit.mapTo(w.editor,QPoint()).y())
            self.assertEqual((w.width(),w.height()), (1920,1040))
            self.assertTrue(w.grab().save(str(EVIDENCE / ("docks-open.png" if opened else "docks-closed.png"))))
            if opened:
                self.assertGreater(w.people.widget().widget().height(),w.people.widget().viewport().height())
            result[str(opened)] = {"editor_height":w.editor.height(), "needed":w.editor.minimumHeight(),"controls":sizes}
        (EVIDENCE/"after-geometry.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print("F2 geometry:",result)

    def test_multishift_shared_beats_undo_redo_and_zero_boundary(self):
        a = self.group("a", "song", start=1000)
        b = self.group("b", "song", start=1031)
        c = self.group("c", "song", start=5000)
        shared = {"id":"shared", "beats":[{"core":[{"ins":a.id,"a":0,"b":1},{"ins":b.id,"a":0,"b":1}], "t0":1100,"t1":1200},
                                                  {"core":[{"ins":c.id,"a":0,"b":1}],"t0":5100,"t1":5200}]}
        self.doc.karaoke["shared"] = shared
        before = self.snapshot()
        ed = self.session.editor
        ed.shift_groups(["a","b","a"], -1000)
        self.assertEqual(len(ed.history.done),1)
        self.assertEqual(ed.history.done[0].label,"整体平移 2 组")
        self.assertEqual((a.start,b.start,c.start),(0,31,5000))
        self.assertEqual([(x["t0"],x["t1"]) for x in shared["beats"]],[(100,200),(5100,5200)])
        self.assertTrue(a.extra["time_manual"])
        after = self.snapshot()
        ed.history.undo(); self.assertEqual(self.snapshot(),before)
        ed.history.redo(); self.assertEqual(self.snapshot(),after)
        ed.shift_groups(["a","b"],-1)
        self.assertEqual(self.doc.members[a.id].start,0)
        self.assertEqual(self.doc.members[b.id].start,30)
        self.assertEqual(self.doc.members[c.id].start,5000)
        ed.history.undo(); self.assertEqual(self.snapshot(),after)
        ed.shift_groups(["a"],0); self.assertEqual(len(ed.history.done),1)

    def test_dialog_menu_boundary_remember_and_cancel(self):
        self.group("a",start=1031);self.group("b",start=5000);self.select("a","b")
        d = ShiftGroupsDialog(self.doc,["a","b"],self.settings,self.window)
        self.addCleanup(d.deleteLater)
        self.assertEqual(d.milliseconds.singleStep(),10)
        self.assertEqual((d.milliseconds.minimum(),d.milliseconds.maximum()),(0,600000))
        d.milliseconds.setValue(1032);self.assertTrue(d.ok.isEnabled())
        self.assertEqual(d.hint.text(),"开始和结束都夹到 0")
        d.milliseconds.setValue(1031);self.assertTrue(d.ok.isEnabled())
        d.later.setChecked(True);d.milliseconds.setValue(600000);self.assertTrue(d.ok.isEnabled())
        d.accept();self.assertEqual(self.settings.value("timing/shift_ms",type=int),600000)
        d2 = ShiftGroupsDialog(self.doc,["a"],self.settings,self.window);self.addCleanup(d2.deleteLater)
        self.assertEqual(d2.milliseconds.value(),600000);d2.milliseconds.setValue(0);d2.reject()
        self.assertEqual(self.settings.value("timing/shift_ms",type=int),600000)
        self.window._update_shift_action();self.assertTrue(self.window.act_shift_groups.isEnabled())
        received=[]
        self.window.table.shiftGroupsRequested.disconnect()
        self.window.table.shiftGroupsRequested.connect(lambda gids: received.append(gids))
        self.context_menu("整体平移")
        self.assertEqual(received,[["a","b"]])
        self.window.session=None;self.window._update_shift_action()
        self.assertIn("先打开一集",self.window.act_shift_groups.text())
        self.window.session=self.session

    def test_libass_name_uses_line_colour_question_only_red(self):
        ass = self.template
        ass.set_info("PlayResX", 1920)
        ass.set_info("PlayResY", 1080)
        st = ass.style("CN")
        st.set("PrimaryColour","&H0000FF00")
        st.set("OutlineColour","&H00FF0000")
        st.set("BackColour","&H0000FFFF")
        st.set("Bold","-1")
        ass.events=[Event(kind="Dialogue",start=0,end=300,style="CN",text="正文")]
        geom=LineGeom(0,"m","g","base",0,Rect(600,400,800,460))
        player=SimpleNamespace(overlay=Mock(),clear_overlay=Mock())
        watch=SimpleNamespace(player=player,ass=ass,transform=None,_speaker_cache={},
            parent=lambda:SimpleNamespace(settings=SimpleNamespace(value=lambda *args:"original")),_transform_preview=None,anchors={},
            _mapper=lambda:VideoMapper(dict(w=1920,h=1080),(1920,1080)),
            _doc=lambda:SimpleNamespace(members={"m":SimpleNamespace(lane="translation")},
                group=lambda gid:gid,members_of=lambda group:[]),
            _speakers_enabled=lambda:True,_current_geoms=lambda:[geom])
        EVIDENCE.mkdir(exist_ok=True)
        for name in ("名字：","? 名字","群众"):
            watch._speaker_cache.clear()
            watch._speaker_text=lambda gid:(name,"&H808080&")
            watch.active = lambda: [0]
            watch.owners = [dict(kind="base", member="m", group="g")]
            watch.geo = watch._speaker_geoms = {0: geom}
            WatchController._draw_speakers(watch)
            overlay=player.overlay.call_args.args[1]
            self.assertIn(r"\1c&H00FF00&",overlay)
            self.assertIn(r"\3c&HFF0000&",overlay)
            self.assertIn(r"\4c&H00FFFF&",overlay)
            self.assertIn(r"\b1",overlay)
            self.assertEqual(overlay.count(r"\1c&H0000FF&"),int(name.startswith("?")))
            rendered=AssDocument.parse(ass.serialize());rendered.events=[ass.events[0].copy(text=overlay)]
            target=EVIDENCE/("warning" if name.startswith("?") else "crowd" if name=="群众" else "name")
            target.with_suffix(".ass").write_text(rendered.serialize(),encoding="utf-8-sig")
            result=run(["ffmpeg","-v","error","-f","lavfi","-i","color=black:s=1920x1080:d=1",
                        "-vf",f"ass={target.name}.ass,format=rgb24","-frames:v","1","-f","rawvideo","-"],
                       cwd=EVIDENCE,capture_output=True,check=True)
            pixels=np.frombuffer(result.stdout,np.uint8).reshape(1080,1920,3)
            green=(pixels[:,:,1]>200)&(pixels[:,:,0]<50)&(pixels[:,:,2]<50)
            red=(pixels[:,:,0]>200)&(pixels[:,:,1]<50)&(pixels[:,:,2]<50)
            self.assertGreater(green.sum(),20)
            self.assertEqual(bool(red.any()),name.startswith("?"))
            print("F2 libass",name,"green",int(green.sum()),"red",int(red.sum()))
