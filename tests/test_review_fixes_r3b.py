"""R3b: host libass ink checks and offscreen template canvas, no visible player."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import re
import unittest
from unittest.mock import Mock, patch
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication
from subkit import templates as T
from subkit.ass import AssDocument, Event
from subkit.ass.text import line_tag
from subkit.core.layout_index import LayoutIndex
from subkit.proc import run
from subkit_app.views.template_editor import TemplateEditor, _Job
from subkit_app.watch import WatchController, VideoMapper



def _doc_double():
    """The document surface _draw_speakers uses (0.11.6 caches overlay text per group)."""
    member = SimpleNamespace(id="m", lane="translation", speaker=None)
    group = SimpleNamespace(id="g", members=["m"])
    return SimpleNamespace(members={"m": member}, group=lambda gid: group, members_of=lambda g: [member])


def _parent_double():
    return SimpleNamespace(settings=SimpleNamespace(value=lambda key, default=None, **kw: default))


class R3bTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="r3b-", dir=os.environ["TEMP"])
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.styles = T.build_styles(T.font_set("utena_serif"), T.palette("utena"), translation_lang="zh-Hant")

    def render(self, ass, filename):
        ass_path = self.path / (filename + ".ass")
        ass_path.write_text(ass.serialize(), encoding="utf-8-sig")
        w, h = ass.playres
        result = run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=black:s={w}x{h}:d=1",
                      "-vf", f"ass={ass_path.name},format=gray", "-frames:v", "1", "-f", "rawvideo", "-"],
                     cwd=self.path, capture_output=True, check=True)
        return np.frombuffer(result.stdout, np.uint8).reshape(h, w)

    def test_real_libass_names_left_of_first_line_selected_warning_and_16_9(self):
        measurements = []
        for width in (1440, 1920):
            for text in (r"相信奇蹟…\N相信自己的心意能傳達到…", "相信奇蹟…相信吧…"):
                ass = AssDocument.parse(T._preview_head((width, 1080)) + self.styles + T._EVH)
                ass.events = [Event(kind="Dialogue", start=0, end=300, style="CN", text=text)]
                owners = [dict(member="m", group="g", kind="base")]
                li = LayoutIndex(ass, owners, measure_cache=self.path / "measure.sqlite3")
                try:
                    geom = li.precise_geometry([0])[0]
                    anchor = li.anchor(0)
                finally:
                    li.close()
                base = self.render(ass, "base")
                # Restrict to the first visual line, rather than the whole block.
                y0, y1 = geom._frames[0][0].y0, geom._frames[0][0].y1
                first_x = int(np.nonzero(base[max(0,int(y0)-2):int(y1)+3] > 30)[1].min())
                for selected in (None, "g"):
                    for name, colour in (("有栖川樹璃：", "&HFFFFFF&"), ("? 高槻枝織", "&H0000FF&")):
                        player = SimpleNamespace(overlay=Mock(), clear_overlay=Mock())
                        watch = SimpleNamespace(player=player, ass=ass, transform=None, _transform_preview=None,
                            anchors={0: anchor}, selected=selected, _mapper=lambda: VideoMapper(dict(w=width,h=1080),(width,1080)),
                            _doc=_doc_double, parent=_parent_double, _speaker_cache={},
                            _speakers_enabled=lambda: True, _current_geoms=lambda: [geom],
                            _speaker_text=lambda gid: (name, colour))
                        watch.active = lambda: [0]
                        watch.owners = [dict(kind="base", member="m", group="g")]
                        watch.geo = watch._speaker_geoms = {0: geom}
                        WatchController._draw_speakers(watch)
                        overlay = player.overlay.call_args.args[1]
                        self.assertTrue(overlay)
                        name_ass = AssDocument.parse(T._preview_head((width,1080)) + self.styles + T._EVH)
                        name_ass.events = [ass.events[0].copy(text=overlay)]
                        ink = self.render(name_ass, "name")
                        right = int(np.nonzero(ink > 30)[1].max())
                        self.assertLess(right, first_x, (width,text,name,selected,right,first_x))
                        self.assertGreaterEqual(first_x-right, .3*68-2)
                        measurements.append((width, text.count(r"\N")+1, name, first_x-right))
        print("R3b host libass first-line/name ink gaps (px):", measurements)

    def test_name_shrinks_then_elides_without_crossing_first_ink(self):
        from subkit.core.layout_index import LineGeom, Rect
        for left in (160, 100, 1):
            ass = AssDocument.parse(T._HEAD + self.styles + T._EVH)
            ass.events = [Event(kind="Dialogue", start=0,end=300,style="CN",text="相信吧")]
            geom = LineGeom(0,"m","g","base",0,Rect(left,900,left+200,960))
            player = SimpleNamespace(overlay=Mock(),clear_overlay=Mock())
            watch = SimpleNamespace(player=player,ass=ass,transform=None,_transform_preview=None,anchors={},
                _mapper=lambda: VideoMapper(dict(w=1920,h=1080),(1920,1080)),
                _doc=_doc_double, parent=_parent_double, _speaker_cache={},
                _speakers_enabled=lambda:True,_current_geoms=lambda:[geom],
                _speaker_text=lambda gid:("? 有栖川樹璃有栖川樹璃", "&H0000FF&"))
            watch.active = lambda: [0]
            watch.owners = [dict(kind="base", member="m", group="g")]
            watch.geo = watch._speaker_geoms = {0: geom}
            WatchController._draw_speakers(watch)
            overlay = player.overlay.call_args.args[1]
            if left == 1:
                self.assertEqual(overlay, "")
            else:
                self.assertAlmostEqual(float(re.findall(r"\\fs([\d.]+)",overlay)[-1]),34,delta=.01)
                self.assertTrue(overlay.endswith("…"))
                name_ass = AssDocument.parse(T._HEAD+self.styles+T._EVH)
                name_ass.events=[ass.events[0].copy(text=overlay)]
                xs=np.nonzero(self.render(name_ass,"short")>30)[1]
                self.assertTrue(xs.size)
                self.assertLess(xs.max(),left)

    def test_video_frame_effective_playres_and_preview_matches_direct_libass(self):
        video = self.path / "four_three.mp4"
        run(["ffmpeg","-y","-v","error","-f","lavfi","-i","color=0x705030:s=320x240:r=5:d=2",
             "-pix_fmt","yuv420p",str(video)],capture_output=True,check=True)
        with patch.object(TemplateEditor,"_schedule"),patch.object(TemplateEditor,"_grab_frame"):
            editor=TemplateEditor(video=video)
        self.addCleanup(editor.close)
        def synchronous(fn,done):
            done(fn(_Job(1,fn).runner))
        with patch.object(editor,"_run_job",side_effect=synchronous),patch.object(editor,"_schedule"):
            editor._grab_frame(video)
        self.addCleanup(editor._bg_frame.unlink)
        self.assertEqual(editor._bg_playres,(1440,1080))
        self.assertEqual((QImage(str(editor._bg_frame)).width(),QImage(str(editor._bg_frame)).height()),(1440,1080))
        scripts=[]
        def capture(cmd,**kwargs):
            if "-vf" in cmd and "ass=t.ass" in cmd[cmd.index("-vf")+1]:
                scripts.append(Path(kwargs["cwd"],"t.ass").read_text(encoding="utf-8-sig"))
            kwargs.setdefault("capture_output",True)
            return run(cmd,**kwargs)
        png=self.path/"preview.png"
        T.render_scene(self.styles,png,background=editor._bg_frame,width=1920,guides=False,
                       runner=SimpleNamespace(run=capture,check_cancelled=lambda:None),measure_cache=self.path/"cache.sqlite3")
        self.assertEqual(AssDocument.parse(scripts[0]).playres,(1440,1080))
        ass_path=self.path/"direct.ass"; ass_path.write_text(scripts[0],encoding="utf-8-sig")
        direct=self.path/"direct.png"
        run(["ffmpeg","-y","-v","error","-loop","1","-i",str(editor._bg_frame),"-vf",
             "setsar=1,ass=direct.ass,pad=1920:1080:240:0","-frames:v","1",str(direct)],
            cwd=self.path,capture_output=True,check=True)
        a,b=QImage(str(png)),QImage(str(direct))
        self.assertEqual(a,b)  # zero pixel error, stricter than the required 1 px
        self.assertEqual(a.pixelColor(0,540),QColor("black"))
        self.assertNotEqual(a.pixelColor(240,540),QColor("black"))
        editor.preview.resize(800,450); editor._last_pm=QPixmap(str(png)); editor._display_preview()
        self.assertEqual(editor.preview.pixmap().toImage().pixelColor(0,225),QColor("black"))
        print("R3b 4:3 host ffmpeg/libass: frame 1440x1080; preview 1920x1080, 240 px bars, direct render pixel error=0")

    def test_16_9_matches_previous_renderer_bytes(self):
        # Keep the previous pipeline as an independent regression reference.
        styles=T._preview_styles(self.styles,"dialogue","ja","zh-Hans")
        lines=T.scene_events(styles,"dialogue",measure_cache=self.path/"cache.sqlite3")
        ass_path=self.path/"t.ass"; ass_path.write_text(T._HEAD+styles+T._EVH+"\n".join(lines)+"\n",encoding="utf-8-sig")
        image=QImage(1920,1080,QImage.Format.Format_RGB32); image.fill(QColor("#705030"))
        bg=self.path/"bg.png"; self.assertTrue(image.save(str(bg)))
        old=self.path/"old.png"; new=self.path/"new.png"
        run(["ffmpeg","-y","-hide_banner","-loglevel","verbose","-loop","1","-t","3","-i",str(bg),
             "-ss","1.00","-vf","scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=25,ass=t.ass,scale=800:-2",
             "-frames:v","1",str(old)],cwd=self.path,capture_output=True,check=True)
        T.render_scene(self.styles,new,width=800,guides=False,background=bg,measure_cache=self.path/"cache.sqlite3")
        self.assertEqual(old.read_bytes(),new.read_bytes())
        print("R3b 16:9 host libass: PNG byte-identical to previous pipeline")

    def test_pixel_aspect_rounding_and_wide_video_letterboxes(self):
        video=self.path/"anamorphic.mkv"
        run(["ffmpeg","-y","-v","error","-f","lavfi","-i","color=0x705030:s=720x576:r=1:d=1",
             "-vf","setsar=16/15","-c:v","ffv1",str(video)],capture_output=True,check=True)
        self.assertEqual(T.preview_playres(video),(1440,1080))
        probe=SimpleNamespace(run=Mock(return_value=SimpleNamespace(stdout=b'{"streams":[{"width":1441,"height":1080,"sample_aspect_ratio":"N/A"}]}')))
        self.assertEqual(T.preview_playres(video,probe),(1442,1080))
        image=QImage(2400,1080,QImage.Format.Format_RGB32); image.fill(QColor("#705030"))
        bg=self.path/"wide.png"; self.assertTrue(image.save(str(bg)))
        png=self.path/"wide_preview.png"
        T.render_scene(self.styles,png,width=800,guides=False,background=bg,original_lang="en",
                       measure_cache=self.path/"cache.sqlite3")
        result=QImage(str(png))
        self.assertEqual((result.width(),result.height()),(800,450))
        self.assertEqual(result.pixelColor(400,0),QColor("black"))
        self.assertNotEqual(result.pixelColor(400,100),QColor("black"))
