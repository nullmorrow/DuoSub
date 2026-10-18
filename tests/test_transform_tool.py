"""Qt-free free-transform session tests; real libass for clear and accumulation."""

from __future__ import annotations

import math
import shutil
import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass.document import Event
from subkit.core.export import apply_overrides
from subkit.core.layout_index import LayoutIndex, LineGeom, Rect
from subkit.core.transform import apply_group_transform, source_transform_tags, transform_of
from subkit_app.transform_tool import (DISABLED_3D_REASON, HINT, Drag, TransformMember, TransformTool)
from tests.test_transform import LibassHarness, document, ink_box, member_state, merged


def member(doc, mid="m", *, text=r"{\an7\pos(100,100)}HH", overrides=None,
           original=False, box=(100,100,200,160), origins=((100,100),)):
    state = member_state(doc, Event(text=text), mid, overrides)
    return TransformMember(mid, Event(text=text), doc.styles[0], state.anchor, state.overrides,
                           (Rect(*box),), origins,
                           (state.transform.fscx or 100, state.transform.fscy or 100),
                           source_transform_tags(text), original)


def drag_handle(tool, name, dx=0, dy=0, **mods):
    point = next(h.point for h in tool.handles() if h.name == name)
    return Drag(name, point, (point[0]+dx, point[1]+dy), **mods)


class TransformToolTests(unittest.TestCase):
    def setUp(self):
        self.doc = document()
        self.tool = TransformTool(self.doc, [member(self.doc)])

    def state(self, tool=None, mid="m"):
        return next(s for s in (tool or self.tool)._states() if s.member == mid)

    def test_corner_opposite_fixed_and_shift_ratio(self):
        op = self.tool.operation(drag_handle(self.tool,"se",50,15))
        self.assertEqual((op["sx"],op["sy"],op["fixed"]), (1.5,1.25,(100,100)))
        self.tool.end_drag(drag_handle(self.tool,"se",50,15,shift=True))
        self.assertEqual((self.state().transform.fscx,self.state().transform.fscy), (150,150))
        self.assertEqual(self.tool.box()[0],(100,100))

    def test_alt_corner_keeps_pivot_fixed(self):
        pivot = self.tool.pivot()
        self.tool.end_drag(drag_handle(self.tool,"se",25,15,alt=True))
        self.assertEqual(self.tool.pivot(),pivot)
        self.assertEqual(self.tool.box(), ((75,85),(225,85),(225,175),(75,175)))

    def test_all_corners_and_edges(self):
        for name,delta in (("nw",(-10,-6)),("ne",(10,-6)),("se",(10,6)),("sw",(-10,6))):
            op = self.tool.operation(drag_handle(self.tool,name,*delta))
            self.assertAlmostEqual(op["sx"],1.1)
            self.assertAlmostEqual(op["sy"],1.1)
        for name,delta,expected in (("n",(0,-30),(1,1.5)),("s",(0,30),(1,1.5)),
                                     ("e",(50,0),(1.5,1)),("w",(-50,0),(1.5,1))):
            with self.subTest(edge=name):
                op = self.tool.operation(drag_handle(self.tool,name,*delta))
                self.assertEqual((op["sx"],op["sy"]),expected)
                self.tool.preview(drag_handle(self.tool,name,*delta))

    def test_ctrl_edges_shear_opposite_edge_fixed(self):
        for name,delta,axis,expected in (("n",(12,0),"kx",-.2),("s",(12,0),"kx",.2),
                                        ("e",(0,20),"ky",.2),("w",(0,20),"ky",-.2)):
            with self.subTest(edge=name):
                op = self.tool.operation(drag_handle(self.tool,name,*delta,ctrl=True))
                self.assertEqual(op["kind"],"shear")
                self.assertAlmostEqual(op[axis],expected)
                changes = self.tool.preview(drag_handle(self.tool,name,*delta,ctrl=True))
                self.assertIn("raw",changes["m"])

    def test_rotation_and_shift_snap_absolute_angle(self):
        pivot = self.tool.pivot()
        start = (pivot[0]+100,pivot[1])
        current = (pivot[0]+100*math.cos(math.radians(22)),pivot[1]-100*math.sin(math.radians(22)))
        drag = Drag("rotate_ne",start,current)
        self.assertAlmostEqual(self.tool.operation(drag)["delta"],22)
        self.tool.end_drag(replace(drag,shift=True))
        self.assertEqual(self.state().transform.frz,15)
        self.assertEqual(self.state().transform.org,pivot)
        self.assertEqual(self.tool.operation(replace(drag,shift=True))["delta"],15)

    def test_pivot_drag_preserves_rotated_quad(self):
        tool = TransformTool(self.doc,[member(self.doc,overrides={"raw":r"\frz30"})])
        before = tool.box()
        tool.end_drag(drag_handle(tool,"pivot",25,-10))
        self.assertLess(np.abs(np.asarray(tool.box())-before).max(),.1)
        self.assertEqual(self.state(tool).transform.org,(tool.pivot()[0],tool.pivot()[1]))

    def test_inside_move_preview_is_pure_and_accumulates(self):
        before = deepcopy(self.tool.current_overrides)
        move = Drag("inside",(120,110),(130,130))
        preview = self.tool.preview(move)
        self.assertEqual(self.tool.current_overrides,before)
        self.assertEqual(self.tool.end_drag(move),preview)
        self.tool.end_drag(move)
        self.assertEqual(self.state().anchor["x"],120)
        self.assertEqual(self.state().anchor["y"],140)

    def test_scale_rotate_shear_move_accumulation_and_origins(self):
        self.tool.end_drag(drag_handle(self.tool,"se",100,60))
        self.tool.end_drag(drag_handle(self.tool,"s",20,0,ctrl=True))
        self.tool.end_drag(Drag("inside",(0,0),(10,-5)))
        state = self.state()
        self.assertEqual((state.transform.fscx,state.transform.fscy,state.transform.fax),(200,200,.17))
        self.assertEqual(state.anchor["x"],110)
        self.assertEqual(state.layout_origin,(110,95))
        pivot = self.tool.pivot()
        self.tool.end_drag(Drag("rotate_ne",(pivot[0]+100,pivot[1]),(pivot[0],pivot[1]-100)))
        self.assertEqual(self.state().transform.frz,90)
        self.assertEqual(self.state().transform.fscx,200)

    def test_scaled_layout_bearing_rebuilt_for_later_shear(self):
        tool = TransformTool(self.doc,[member(self.doc,origins=((90,80),))])
        tool.end_drag(drag_handle(tool,"se",100,60))
        state = self.state(tool)
        self.assertEqual(state.layout_origin,(80,60))
        expected = apply_group_transform([state], tool.operation(drag_handle(tool,"s",12,0,ctrl=True)))
        self.assertEqual(tool.preview(drag_handle(tool,"s",12,0,ctrl=True)),expected)

    def test_cancel_restores_exact_entry_and_changes_merge_round_trip(self):
        baseline = {"raw":r"\bord2\fr20\1c&H112233&", "fs":55,"pos":[100,100]}
        tool = TransformTool(self.doc,[member(self.doc,overrides=baseline)])
        tool.clear()
        tool.end_drag(Drag("inside",(0,0),(12,7)))
        changes = tool.changes()
        self.assertNotIn("fs",changes["m"])
        self.assertEqual(merged(baseline,changes["m"]),tool.current_overrides["m"])
        tool.cancel()
        self.assertEqual(tool.changes(),{})
        self.assertEqual(tool.current_overrides["m"],baseline)

    def test_clear_returns_none_removal_and_keeps_positions(self):
        baseline = {"raw":r"\frz30\fax0.2\org(80,80)","move":[100,100,200,150,20,900],"fs":50}
        tool = TransformTool(self.doc,[member(self.doc,overrides=baseline)])
        tool.clear()
        self.assertEqual(tool.changes(),{"m":{"raw":None}})
        self.assertEqual(tool.current_overrides["m"],{k:v for k,v in baseline.items() if k!="raw"})
        self.assertIsNone(self.state(tool).transform.org)
        tool.cancel()
        self.assertEqual(tool.current_overrides["m"],baseline)

    def test_3d_corner_screen_pixels_and_accumulation(self):
        self.assertTrue(self.tool.toggle_3d())
        self.tool.end_drag(drag_handle(self.tool,"ne",-5,10,screen_scale=(2,3)))
        self.assertEqual((self.state().transform.frx,self.state().transform.fry),(15,-5))
        self.tool.end_drag(drag_handle(self.tool,"ne",2,4,screen_scale=(2,3)))
        self.assertEqual((self.state().transform.frx,self.state().transform.fry),(21,-3))
        self.assertIn(" · X 21° Y −3°",self.tool.readout())
        self.assertFalse(self.tool.toggle_3d())

    def test_3d_mixed_selection_disabled_handles_and_clear_reenables(self):
        tool = TransformTool(self.doc,[member(self.doc),member(self.doc,"tilt",overrides={"raw":r"\frx10"})])
        handles = {h.name:h for h in tool.handles()}
        for name in ("nw","ne","se","sw","n","e","s","w","pivot"):
            self.assertFalse(handles[name].enabled)
            self.assertEqual(handles[name].reason,DISABLED_3D_REASON)
            with self.assertRaisesRegex(ValueError,"^3D:"):
                tool.preview(drag_handle(tool,name,1,1))
        self.assertTrue(handles["rotate_ne"].enabled)
        tool.preview(Drag("inside",(0,0),(2,3)))
        tool.toggle_3d()
        self.assertTrue(next(h for h in tool.handles() if h.name=="ne").enabled)
        tool.clear()
        self.assertTrue(all(h.enabled for h in tool.handles()))

    def test_rotated_hit_test_handles_rotation_zones_pivot_and_inside(self):
        tool = TransformTool(self.doc,[member(self.doc,overrides={"raw":r"\frz40\org(150,130)"})])
        for handle in tool.handles(tolerance=3):
            self.assertEqual(tool.hit_test(handle.point,3).name,handle.name)
        self.assertEqual(tool.hit_test((145,125),1).name,"inside")
        self.assertIsNone(tool.hit_test((100,100),1))
        self.assertIsNone(tool.hit_test((1000,1000),3))
        handle = tool.handles(3)[0]
        self.assertEqual(tool.hit_test((handle.point[0]+2,handle.point[1]),3).name,handle.name)

    def test_rotated_corner_scale_uses_local_axes(self):
        tool = TransformTool(self.doc,[member(self.doc,overrides={"raw":r"\frz90\org(100,100)"})])
        op = tool.operation(drag_handle(tool,"se",30,-50))
        self.assertAlmostEqual(op["sx"],1.5)
        self.assertAlmostEqual(op["sy"],1.5)
        self.assertEqual(op["frame_angle"],90)

    def test_reference_original_union_and_entry_copy(self):
        members = [member(self.doc,"cn"),member(self.doc,"jp",original=True,box=(100,200,180,240),origins=((100,200),))]
        tool = TransformTool(self.doc,members)
        self.assertEqual(tool.reference,"jp")
        self.assertEqual(tool.box(),((100,100),(200,100),(200,240),(100,240)))
        members[0].frames[0].x0=-999
        self.doc.styles[0].set("Angle",99)
        returned = tool.current_overrides
        returned["cn"]["raw"]=r"\frz99"
        self.assertEqual(tool.box()[0],(100,100))
        self.assertEqual(self.state(tool,"cn").transform.frz,0)

    def test_readout_hint_and_invalid_input(self):
        tool = TransformTool(self.doc,[member(self.doc,overrides={"raw":r"\frz12.5\fscx120\fax0.1"})])
        self.assertEqual(tool.readout(),"旋转 12.5° · 缩放 120% × 100% · 斜切 0.10 / 0.00")
        self.assertEqual(HINT,"自由变换：拖手柄调整，Enter 确定，Esc 取消，Backspace 清除变形")
        for action in (lambda:TransformTool(self.doc,[]),lambda:tool.hit_test((0,0),-1),
                       lambda:tool.preview(Drag("bad",(0,0),(1,1))),
                       lambda:tool.preview(Drag("inside",(0,0),(math.nan,1))),
                       lambda:TransformTool(self.doc,[replace(member(self.doc),layout_origins=())])):
            with self.assertRaises(ValueError): action()


class ClearTransforms(unittest.TestCase):
    def test_clear_style_defaults_source_tags_and_unrelated_raw(self):
        doc = document()
        doc.styles[0].set("Angle",12)
        doc.styles[0].set("ScaleX",130)
        doc.styles[0].set("ScaleY",90)
        source = r"{\pos(300,200)\fr30\fscx160\frx20}HH{\fax.2\fry-10}H"
        baseline = {"raw":r"\bord2\org(200,150)\fscy150\fax0\t(\frz90)\1c&H112233&", "fs":55}
        state = replace(member_state(doc,Event(text=source),overrides=baseline),source_tags=source_transform_tags(source))
        change = apply_group_transform([state],{"kind":"clear"})["m"]
        self.assertEqual(change,{"raw":r"\bord2\t(\frz90)\1c&H112233&\frz12\frx0\fry0\fscx130\fax0"})
        out = apply_overrides(source,merged(baseline,change))[0]
        tf = transform_of(doc,Event(text=out))
        self.assertEqual((tf.frz,tf.frx,tf.fry,tf.fscx,tf.fscy,tf.fax,tf.org),(12,0,0,130,90,0,None))
        self.assertIn(r"\pos(300,200)",out)
        self.assertEqual(baseline["fs"],55)


@unittest.skipUnless(shutil.which("ffmpeg"),"needs ffmpeg/libass")
class LibassTool(LibassHarness):
    def test_clear_matches_style_render_and_keeps_move(self):
        doc = document()
        doc.styles[0].set("Angle",12)
        doc.styles[0].set("ScaleX",130)
        doc.styles[0].set("ScaleY",90)
        source = r"{\an7\move(350,220,360,230,0,900)\frz30\frx20\fry-15\fscx150\fax.2}██"
        state = replace(member_state(doc,Event(text=source),overrides={"raw":r"\org(400,270)\fay.1"}),
                        source_tags=source_transform_tags(source))
        change = apply_group_transform([state],{"kind":"clear"})["m"]
        actual = apply_overrides(source,merged(state.overrides,change))[0]
        expected = r"{\an7\move(350,220,360,230,0,900)}██"
        frames = self.render(doc,[actual,expected],"tool-clear")
        self.assertTrue(np.array_equal(frames[0],frames[1]))
        print("clear libass: style-default render pixel-identical; move preserved; " +
              next(line for line in self.logs[-1].splitlines() if "libass source:" in line))

    def test_precise_geom_capture_accumulated_tool_render(self):
        doc = document()
        source = r"{\an7\pos(350,220)}██"
        doc.events=[Event(text=source)]
        li=LayoutIndex(doc,[{"member":"m","group":"g","kind":"base"}])
        try:
            # A plain selected line also needs precise frames on tool entry.
            li._transformed_geoms([0])
            geom=li.geometry([0])[0]
            capture=TransformMember.from_geom(geom,doc.events[0],doc.styles[0],li.anchor(0),{},measured_scale=(100,100))
        finally:
            li.close()
        tool=TransformTool(doc,[capture])
        tool.end_drag(drag_handle(tool,"se",(geom.rect.x1-geom.rect.x0)*.5,(geom.rect.y1-geom.rect.y0)*.5))
        tool.end_drag(Drag("inside",(0,0),(10,-5)))
        output=apply_overrides(source,tool.current_overrides["m"])[0]
        frame=self.render(doc,[output],"tool-accumulated",scale=4)[0]
        quad=np.asarray(tool.box())
        predicted=(quad[:,0].min(),quad[:,1].min(),quad[:,0].max(),quad[:,1].max())
        error=float(np.abs(np.asarray(predicted)-ink_box(frame,4)).max())
        self.assertLessEqual(error,1)
        self.assertEqual(merged({},tool.changes()["m"]),tool.current_overrides["m"])
        print(f"tool accumulated scale/move: max libass bounds error {error:.6f} px")


if __name__ == "__main__":
    unittest.main()
