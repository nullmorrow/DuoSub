"""R3a: episode display coordinates, provenance-safe migration and real libass."""
from __future__ import annotations
import copy
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tests.ruby_policy import legacy_ruby_layout
from subkit.ass.document import AssDocument, Event, Style
from subkit.core.export import ExportOptions, build_ass
from subkit.core.layout_index import LayoutIndex
from subkit.core.workspace import Project
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import save_document
from subkit.proc import run


def fixture():
    doc = Document("R3a", groups=[Group("g", members=["m"])])
    doc.members["m"] = Member("m", "g", "ja", "original", "dialogue", 0, 3000,
                               style="Default", runs=[Run("世界", ins="i", b=2)])
    tpl = AssDocument()
    tpl.set_info("PlayResX", 1920); tpl.set_info("PlayResY", 1080)
    tpl.styles = [Style.new("Default", Fontname="Microsoft YaHei", Fontsize=60,
                           MarginL=120, MarginR=160, MarginV=100, Spacing=0,
                           Outline=0, Shadow=0)]
    return doc, tpl


class EpisodeFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="r3a-", dir=os.environ["TEMP"])
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.project = Project.create(self.path / "project", "R3a")
        self.ep = self.project.add_episode("S01E001")

    def video(self, width=1440, height=1080, sar="1:1"):
        self.ep.data["media"]["video"] = {"path": "unavailable.mkv", "geometry": {
            "width": width, "height": height, "sample_aspect_ratio": sar}}
        self.ep.save_meta()



class EpisodeCoordinates(EpisodeFixture):
    def test_effective_resolution_sar_height_rounding_and_no_video(self):
        self.assertEqual(self.ep.effective_playres(), (1920, 1080))
        self.video(720, 576, "16:15")
        self.assertEqual(self.ep.effective_playres(), (1440, 1080))
        self.ep.data["settings"]["playres"] = [1280, 720]
        self.assertEqual(self.ep.effective_playres(), (960, 720))
        self.video(1441, 1080, "N/A")
        self.assertEqual(self.ep.effective_playres(), (962, 720))
        self.ep.data["media"]["video"] = None
        self.assertEqual(self.ep.effective_playres(), (1280, 720))

    def test_probe_is_cached_in_metadata_and_missing_video_uses_cache(self):
        video = self.path / "video.mkv"; video.write_bytes(b"fixture")
        result = type("Result", (), {"stdout": b'{"streams":[{"width":720,"height":576,"sample_aspect_ratio":"16:15"}]}', "returncode": 0})()
        with patch("subkit.proc.run", return_value=result) as probe:
            self.ep.set_video(video)
            self.assertEqual(self.ep.effective_playres(), (1440, 1080))
            reopened = self.project.episode(self.ep.id)
            self.assertEqual(reopened.effective_playres(), (1440, 1080))
            self.assertEqual(probe.call_count, 1)
        video.unlink()
        self.assertEqual(reopened.effective_playres(), (1440, 1080))

    def test_unreadable_video_keeps_recorded_coordinates_and_opens(self):
        import subprocess
        video = self.path / "video.mkv"; video.write_bytes(b"fixture")
        self.ep.data["media"]["video"] = {"path": str(video)}
        self.ep.save_meta()
        doc, _ = fixture(); doc.extra["coordinate_playres"] = [1440, 1080]
        doc.members["m"].overrides = {"pos": [600, 300]}
        self.ep.document = doc; save_document(doc, self.ep.dir / "document.json")
        before = (self.ep.dir / "document.json").read_bytes()
        failure = subprocess.CalledProcessError(1, ["ffprobe"])
        with patch("subkit.proc.run", side_effect=failure):
            opened = self.project.episode(self.ep.id)
            self.assertEqual(opened.effective_playres(), (1440, 1080))
        self.assertEqual(opened.document.members["m"].overrides["pos"], [600, 300])
        self.assertEqual((self.ep.dir / "document.json").read_bytes(), before)
        self.assertNotIn("geometry", opened.data["media"]["video"])
        video.unlink()
        self.assertEqual(self.project.episode(self.ep.id).effective_playres(), (1440, 1080))

    def test_migration_backup_once_and_source_bytes_untouched(self):
        self.video()
        doc, _ = fixture(); m = doc.members["m"]
        m.overrides = {"pos": [800, 300], "move": [800, 300, 1000, 400, 10, 2000],
                       "org": [900, 350], "clip": [100, 200, 1200, 700],
                       "raw": r"\org(900,350)\clip(100,200,1200,700)\t(0,500,\iclip(100,200,1200,700))\fs60\fsp4"}
        source = Run(r"{\pos(800,300)}", "tag", "source", 0, 0, 15)
        inserted = Run(r"{\pos(1000,300)\clip(2,m 0 0 l 1200 400)}", "tag", ins="tag", b=50)
        m.runs = [source, inserted, *m.runs]
        m.format = [{"anchor": [{"ins":"i","a":0,"b":2}], "props": {"raw": r"\org(900,350)"}}]
        self.ep.document = doc; save_document(doc, self.ep.dir / "document.json")
        before = (self.ep.dir / "document.json").read_bytes()
        opened = self.project.episode(self.ep.id); new = opened.document.members["m"]
        self.assertEqual(opened.document.extra["coordinate_playres"], [1440,1080])
        self.assertEqual(new.overrides["pos"], [600,300])
        self.assertEqual(new.overrides["move"], [600,300,750,400,10,2000])
        self.assertEqual(new.overrides["org"], [675,350])
        self.assertEqual(new.overrides["clip"], [75,200,900,700])
        self.assertIn(r"\org(675,350)", new.overrides["raw"])
        self.assertIn(r"\fs60\fsp4", new.overrides["raw"])
        self.assertEqual(new.runs[0], source)
        self.assertIn(r"\pos(750,300)", new.runs[1].text)
        self.assertIn("l 900 400", new.runs[1].text)
        self.assertEqual(new.format[0]["props"]["raw"], r"\org(675,350)")
        backups = list((self.ep.dir / "journal/backups").glob("*coordinates*.json"))
        self.assertEqual(len(backups),1); self.assertEqual(backups[0].read_bytes(),before)
        once = (self.ep.dir / "document.json").read_bytes()
        self.project.episode(self.ep.id)
        self.assertEqual((self.ep.dir / "document.json").read_bytes(),once)
        self.assertEqual(len(list(backups[0].parent.glob("*coordinates*.json"))),1)

    def test_recorded_width_change_and_backup_failure_leave_document_untouched(self):
        self.video(); doc,_ = fixture(); doc.extra["coordinate_playres"] = [1600,1080]
        doc.members["m"].overrides = {"pos":[1000,300]}
        self.ep.document = doc; save_document(doc,self.ep.dir / "document.json")
        original = (self.ep.dir / "document.json").read_bytes()
        with patch("subkit.core.workspace.Episode.backup", side_effect=PermissionError("denied backup")):
            with self.assertRaises(PermissionError): self.project.episode(self.ep.id)
        self.assertEqual((self.ep.dir / "document.json").read_bytes(),original)
        opened=self.project.episode(self.ep.id)
        self.assertEqual(opened.document.members["m"].overrides["pos"],[900,300])

    def test_template_horizontal_metrics_only_and_no_mutation(self):
        doc,tpl=fixture(); tpl.styles[0].set("Spacing",4)
        before=tpl.serialize()
        ass,_=build_ass(doc,{},tpl,self.path,ExportOptions(playres=(1440,1080),ruby=False))
        st=ass.styles[0]
        self.assertEqual((st.fontsize,st.num("MarginV"),st.num("MarginL"),st.num("MarginR"),st.spacing),(60,100,90,120,3))
        self.assertEqual(tpl.serialize(),before)


@unittest.skipUnless(shutil.which("ffmpeg"), "requires host ffmpeg/libass")
class RenderProofs(EpisodeFixture):
    def render(self, ass, label):
        path=self.path / (label+".ass"); ass.save(path)
        result=run(["ffmpeg","-hide_banner","-loglevel","verbose","-f","lavfi","-i",
                    "color=c=black:s=1440x1080:r=1:d=1","-vf",f"ass={path.name},format=gray",
                    "-frames:v","1","-f","rawvideo","-pix_fmt","gray","-"],cwd=self.path,
                   capture_output=True,timeout=30)
        log=result.stderr.decode("utf-8","replace")
        self.assertEqual(result.returncode,0,log)
        print(next((l for l in log.splitlines() if "libass API" in l),""))
        return np.frombuffer(result.stdout,dtype=np.uint8).reshape(1080,1440)

    def test_layout_and_ruby_x_match_1440_ink_left_center_right(self):
        doc,tpl=fixture(); m=doc.members["m"]
        doc.annotations["a"]=Annotation("a","ruby","m",[{"ins":"i","a":0,"b":2}],"世界","せかい", parts=[[0,2,"せかい"]])
        errors=[]
        for an in (1,2,3):
            m.overrides={"an":an}
            ass,report=build_ass(doc,{},tpl,self.path,ExportOptions(playres=(1440,1080)))
            self.assertEqual(report.ruby_lines,1)
            index=LayoutIndex(ass,report.owners)
            try: geoms=index.precise_geometry(range(len(ass.events)))
            finally: index.close()
            for geom in geoms:
                single=ass.copy(); single.events=[ass.events[geom.index]]
                frame=self.render(single,f"layout-{an}-{geom.index}")
                yy,xx=np.where(frame>24)
                actual=np.array([xx.min(),xx.max()+1])
                predicted=np.array([geom.rect.x0,geom.rect.x1])
                error=float(np.abs(actual-predicted).max()); errors.append(error)
                self.assertLessEqual(error,1,(an,geom.kind,actual,predicted))
        print(f"R3a 1440x1080 left/center/right base+ruby: {len(errors)} ink comparisons, max x error={max(errors):.3f}px")

    def test_migration_render_positions_within_one_pixel(self):
        self.video(); doc,tpl=fixture(); m=doc.members["m"]
        for an in (1,2,3):
            m.overrides={"an":an,"pos":[960,650]}
            save_document(doc,self.ep.dir / "document.json")
            before,_=build_ass(doc,{},tpl,self.path,ExportOptions(ruby=False))
            old=self.render(before,"old")
            opened=self.project.episode(self.ep.id)
            after,_=build_ass(opened.document,{},tpl,self.path,ExportOptions(playres=opened.effective_playres(),ruby=False))
            new=self.render(after,"new")
            oy,ox=np.where(old>24); ny,nx=np.where(new>24)
            error=max(abs(ox.min()-nx.min()),abs(ox.max()-nx.max()))
            self.assertLessEqual(error,1)
            print(f"R3a migration alignment {an}: x ink drift={error}px")

    @legacy_ruby_layout()
    def test_soma_169_export_byte_identity(self):
        from subkit.core.export import ExportCache
        path=self.path / "soma"
        shutil.copytree(ROOT / "test_projects/soma",path,copy_function=shutil.copyfile)
        project=Project.open(path); ep=project.episode(project.episode_ids()[0])
        # The pre-R3a byte fingerprint predates the default X11 width limits.
        opts=ExportOptions(name_format="label",playres=ep.effective_playres(), ruby_max_width={
            "dialogue": {"enabled": False}, "song": {"enabled": False}})
        new,_=build_ass(ep.document,ep.sources,project.template(),ep.dir,opts,
                        measure_cache=project.cache_dir/"render.sqlite3",character_book=project.character_book())
        self.assertEqual(ep.effective_playres(),(1920,1080))
        b=new.serialize().encode("utf-8")
        # Measured with the pre-R3a source on the same soma temp copy (0.10.8); 0.20.0 only moves the 75 OP.CN
        # song lines one layer above their original (layer 1 -> 2), every other byte unchanged.
        self.assertEqual(hashlib.sha256(b).hexdigest(), "ea14c12c709dc840f74f09c9f9a3d83e36bced633b6cfc80644e29e93aba723f")
        print(f"R3a soma 16:9 export byte-identical: {len(b)} UTF8 bytes SHA256={hashlib.sha256(b).hexdigest()}")

    def test_source_positions_resampled_and_inserted_tags_keep_episode_coordinates(self):
        doc,tpl=fixture()
        source_ass=tpl.copy(); source_ass.events=[Event(text=r"{\pos(960,650)}??")]
        source_path=self.path / "source.ass"; source_ass.save(source_path)
        src=self.ep.add_source(source_path,lang="ja",lane="original")
        m=doc.members["m"]
        m.runs=[Run(r"{\pos(960,650)}","tag",src.id,0,0,15),
                Run(r"{\org(720,650)}","tag",ins="owned",b=15),
                Run("世界",src=src.id,e=0,a=15,b=17)]
        m.margins=[120,160,100]
        ass,_=build_ass(doc,{src.id:src},tpl,self.ep.dir,ExportOptions(playres=(1440,1080),ruby=False))
        self.assertIn(r"\pos(720,650)",ass.events[0].text)
        self.assertIn(r"\org(720,650)",ass.events[0].text)
        self.assertEqual((ass.events[0].margin_l,ass.events[0].margin_r),(90,120))
        self.assertIn(r"\pos(960,650)",m.runs[0].text)
