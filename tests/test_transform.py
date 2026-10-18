"""Static transform semantics, checked against real ffmpeg/libass ink."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass.document import AssDocument, Event, Style
from subkit.ass.state import check_raw
from subkit.core.export import apply_overrides
from subkit.core.layout_index import LayoutIndex
from subkit.core.transform import (LineTransform, TransformState, apply_group_transform, line_quad,
                                   line_quads, measurement_copy, source_transform_tags, transform_of)
from subkit.proc import run
from subkit.ruby.generate import RubyGenerator, RubyJob, RubyPiece
from subkit.ruby.measure import LineRequest, Measurer


def document() -> AssDocument:
    doc = AssDocument()
    for key, value in (("PlayResX", 960), ("PlayResY", 540), ("WrapStyle", 2),
                       ("ScaledBorderAndShadow", "yes")):
        doc.set_info(key, value)
    doc.styles = [Style.new("Default", Fontname="Arial", Fontsize="80", Outline="0", Shadow="0",
                            MarginL="180", MarginR="180", MarginV="160", Alignment="7")]
    return doc


def ink_box(frame, scale=1):
    y, x = np.where(frame > 24)
    if not x.size:
        raise AssertionError("libass rendered no ink")
    return (float(x.min())/scale, float(y.min())/scale, float(x.max() + 1)/scale, float(y.max() + 1)/scale)


class TransformParsing(unittest.TestCase):
    def test_style_defaults(self):
        doc = document()
        doc.styles[0].set("Angle", 12)
        doc.styles[0].set("ScaleX", 130)
        doc.styles[0].set("ScaleY", 90)
        self.assertEqual(transform_of(doc, Event(text="Hi")), LineTransform(frz=12, fscx=130, fscy=90))
        self.assertEqual(transform_of(doc, Event(style="Absent", text="Hi")), LineTransform())

    def test_leading_blocks_order_alias_and_reset(self):
        doc = document()
        doc.styles.append(Style.new("Other", Angle="13", ScaleX="140"))
        ev = Event(text=r"{\frz10\fr20\fscx150}{comment}{\fscx120}Hi{\frz90}")
        self.assertEqual(transform_of(doc, ev), LineTransform(frz=20, fscx=120))
        ev.text = r"{\frz10\rOther\fscy80\frz}Hi"
        self.assertEqual(transform_of(doc, ev), LineTransform(frz=13, fscx=140, fscy=80))

    def test_animation_ignored_and_flagged(self):
        ev = Event(text=r"{\frz9\t(0,1000,\frz90\fscx200)\fscx120}Hi")
        self.assertEqual(transform_of(document(), ev), LineTransform(frz=9, fscx=120, animated=True))
        ev.text = r"Hi{\t(\frz30)}!"
        self.assertTrue(transform_of(document(), ev).animated)

    def test_numeric_fractions_exponents_and_negative_scale(self):
        tf = transform_of(document(), Event(text=r"{\fax.5\fay1e-1\frz-2.5e1\fscx-20}HH"))
        self.assertEqual(tf, LineTransform(fax=.5, fay=.1, frz=-25, fscx=0))

    def test_org_is_first_valid_line_value(self):
        ev = Event(text=r"{\org(bad)\org(1,2)\org(3,4)}Hi{\org(5,6)}")
        self.assertEqual(transform_of(document(), ev).org, (1, 2))
        self.assertEqual(transform_of(document(), Event(text=r"Hi{\org(5,6)}")).org, (5, 6))

    def test_measurement_does_not_mutate_or_double_scale(self):
        doc = document()
        doc.styles[0].set("Angle", 30)
        ev = Event(text=r"{\frz20\fscx150\fax.5}HH")
        before = doc.serialize(), ev.text
        copy, plain = measurement_copy(doc, ev)
        self.assertEqual(copy.styles[0].num("Angle"), 0)
        self.assertEqual(plain.text, r"{\fscx150}HH")
        self.assertEqual((doc.serialize(), ev.text), before)
        quad = line_quad((10, 20, 30, 40), (20, 40), LineTransform(fscx=150), measured_scale=(150, 100))
        self.assertEqual(quad, ((10, 20), (30, 20), (30, 40), (10, 40)))

    def test_requires_layout_metrics_and_valid_geometry(self):
        with self.assertRaisesRegex(ValueError, "layout origin"):
            line_quad((0, 0, 10, 10), (0, 0), LineTransform(fax=.5))
        with self.assertRaises(ValueError):
            line_quads([(0, 0, 10, 10)], (0, 0), LineTransform(), layout_origins=[])
        with self.assertRaises(ValueError):
            line_quad((0, 0, 10, 10), (0, 0), LineTransform(), measured_scale=(0, 100))

    def test_3d_projection_context_and_camera_plane(self):
        box, anchor, tf = (0, 0, 100, 80), (0, 0), LineTransform(fry=30)
        quad = line_quad(box, anchor, tf)
        factor = 312.5 / (312.5 + 50)
        self.assertAlmostEqual(quad[2][0], 100 * math.cos(math.radians(30)) * factor)
        self.assertAlmostEqual(quad[2][1], 80 * factor)
        distant = line_quad(box, anchor, tf, perspective_distance=625)
        self.assertGreater(distant[2][0], quad[2][0])
        self.assertEqual(line_quads([box], anchor, tf, layout_origins=[anchor],
                                    perspective_distance=625), [distant])
        for bad in (0, -1, float("inf")):
            with self.assertRaises(ValueError):
                line_quad(box, anchor, tf, perspective_distance=bad)
        with self.assertRaises(ValueError):
            line_quad(box, anchor, LineTransform(frx=float("nan")))
        with self.assertRaisesRegex(ValueError, "^3D:"):
            line_quad((0, 0, 500, 80), anchor, LineTransform(fry=-60))


def member_state(doc, event, mid="m", overrides=None, *, baseline=None, layout=None):
    overrides = dict(overrides or {})
    ev = event.copy(text=apply_overrides(event.text, overrides)[0])
    copy = doc.copy()
    copy.events = [ev]
    anchor = LayoutIndex(copy, [{"member": mid, "group": "g", "kind": "base"}]).anchor(0)
    return TransformState(mid, transform_of(copy, ev), copy.style(ev.style), anchor, overrides,
                          layout_origin=layout, baseline_overrides=baseline)


def merged(overrides, change):
    result = dict(overrides)
    for key, value in change.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


class GroupTransforms(unittest.TestCase):
    def test_style_position_materialised_for_all_alignments(self):
        doc = document()
        for an in range(1, 10):
            ev = Event(text=f"{{\\an{an}}}HH")
            state = member_state(doc, ev)
            change = apply_group_transform([state], {"kind": "move", "dx": 12.34, "dy": -4.56})["m"]
            self.assertEqual(change["an"], an)
            self.assertEqual(change["pos"], [round(state.anchor["x"] + 12.34, 1),
                                             round(state.anchor["y"] - 4.56, 1)])
            self.assertEqual(state.overrides, {})

    def test_move_preserves_endpoint_times_and_moves_explicit_origin(self):
        doc = document()
        state = member_state(doc, Event(text=r"{\move(300,200,400,250,20,900)}HH"),
                             overrides={"raw": r"\org(200,150)\frz30\bord2"})
        change = apply_group_transform([state], {"kind": "move", "dx": 10, "dy": -20})["m"]
        self.assertEqual(change["move"], [310, 180, 410, 230, 20, 900])
        self.assertIn(r"\org(210,130)", change["raw"])
        self.assertIn(r"\frz30", change["raw"])

    def test_repivot_rotated_move_compensates_both_ends(self):
        doc = document()
        state = member_state(doc, Event(text=r"{\move(300,200,400,250)}HH"),
                             overrides={"raw": r"\frz30"})
        change = apply_group_transform([state], {"kind": "set_pivot", "pivot": (200, 100)})["m"]
        updated = merged(state.overrides, change)
        tf = transform_of(doc, Event(text=apply_overrides("HH", updated)[0]))
        for old, new in zip([(300, 200), (400, 250)], [change["move"][:2], change["move"][2:4]]):
            quad = line_quad((new[0], new[1], new[0]+10, new[1]+10), tuple(new), tf)
            self.assertLessEqual(max(abs(quad[0][k]-old[k]) for k in (0, 1)), .1)

    def test_scale_rotated_move_and_origin(self):
        doc = document()
        state = member_state(doc, Event(text=r"{\move(300,200,400,250,0,1000)}HH"),
                             overrides={"raw": r"\frz30\org(200,100)"})
        change = apply_group_transform([state], {"kind": "scale", "sx": 1.5, "sy": 1.5,
                                                 "fixed": (100, 50), "frame_angle": 30})["m"]
        self.assertEqual(change["move"], [400, 275, 550, 350, 0, 1000])
        self.assertIn(r"\org(250,125)", change["raw"])
        self.assertIn(r"\fscx150\fscy150", change["raw"])

    def test_raw_keeps_order_alias_animation_and_drops_changed_style_values(self):
        doc = document()
        doc.styles[0].set("ScaleX", 130)
        doc.styles[0].set("Angle", 12)
        raw = r"\bord2\fr32\1c&H112233&\t(0,500,\frz90\bord4)\fax0\bord3\fscx130\frx0"
        state = member_state(doc, Event(text="HH"), overrides={"raw": raw})
        change = apply_group_transform([state], {"kind": "rotate", "delta": -20, "pivot": (200, 100)})["m"]
        # the rotation reaches the style's 12° and is left out; \fax0 and \fscx130 were not changed: kept as typed
        expected = r"\bord2\1c&H112233&\t(0,500,\frz90\bord4)\fax0\bord3\fscx130\frx0\org(200,100)"
        self.assertEqual(change["raw"], expected)
        self.assertEqual(check_raw(expected, span=False), (expected, ""))
        self.assertEqual(state.overrides["raw"], raw)
        identity = member_state(document(), Event(text="HH"), overrides={"raw": r"\frz30"})
        self.assertEqual(apply_group_transform([identity], {"kind": "rotate", "delta": -30,
                                                          "pivot": (480, 160)})["m"]["raw"], r"\org(480,160)")

    def test_style_value_is_written_when_the_source_has_that_tag(self):
        # leaving \frz out would let the source's own \frz30 back into the export
        doc = document()
        ev = Event(text=r"{\pos(300,200)\fr30}HH{\fax0.2}H")
        self.assertEqual(source_transform_tags(ev.text), frozenset({"frz", "fax"}))
        self.assertEqual(source_transform_tags(r"{\t(0,500,\frz90)}HH"), frozenset())
        state = replace(member_state(doc, ev), source_tags=source_transform_tags(ev.text))
        change = apply_group_transform([state], {"kind": "rotate", "delta": -30, "pivot": (300, 200)})["m"]
        self.assertIn(r"\frz0", change["raw"])
        self.assertNotIn("fax", change["raw"])               # untouched: the source's mid-line shear stays
        out = apply_overrides(ev.text, merged(state.overrides, change))[0]
        self.assertNotIn(r"\fr30", out)
        self.assertIn(r"\fax0.2", out)
        self.assertEqual(transform_of(doc, Event(text=out)).frz, 0)
        # without a source tag the style value is still left out
        plain = member_state(doc, Event(text=r"{\pos(300,200)}HH"), overrides={"raw": r"\frz30"})
        self.assertNotIn("frz", apply_group_transform([plain], {"kind": "rotate", "delta": -30,
                                                               "pivot": (300, 200)})["m"]["raw"])

    def test_clear_masks_a_source_org_only_while_the_style_still_rotates(self):
        ev = Event(text=r"{\pos(300,200)\org(100,100)}HH")
        tags = source_transform_tags(ev.text)
        self.assertIn("org", tags)
        tilted = document()
        tilted.styles[0].set("Angle", 20)
        state = replace(member_state(tilted, ev, overrides={"raw": r"\frz40"}), source_tags=tags)
        raw = apply_group_transform([state], {"kind": "clear"})["m"]["raw"]
        self.assertEqual(raw, r"\org(300,200)")                    # libass's default origin, written out
        out = apply_overrides(ev.text, {"raw": raw})[0]
        self.assertNotIn(r"\org(100,100)", out)
        self.assertEqual(transform_of(tilted, Event(text=out)).frz, 20)
        level = replace(member_state(document(), ev, overrides={"raw": r"\frz40"}), source_tags=tags)
        self.assertIsNone(apply_group_transform([level], {"kind": "clear"})["m"]["raw"])   # no rotation left

    def test_rounding_and_zero_scale(self):
        doc = document()
        state = member_state(doc, Event(text="HH"), overrides={"raw": r"\frz12.34\fax.126"})
        change = apply_group_transform([state], {"kind": "scale", "sx": 1.23456, "sy": 0,
                                                 "fixed": (180, 160)})["m"]
        self.assertEqual(change["raw"], r"\frz12.34\fax.126\fscx123.5\fscy0")      # untouched tags stay as typed
        self.assertEqual(change["pos"], [180, 160])

    def test_shear_requires_layout_and_composes_existing_shears(self):
        doc = document()
        state = member_state(doc, Event(text="HH"), overrides={"raw": r"\fax0.2\fay0.3"}, layout=(180, 160))
        change = apply_group_transform([state], {"kind": "shear", "kx": .4, "fixed": (180, 160)})["m"]
        self.assertEqual(change["raw"], r"\fay0.3\fscx112\fax0.54")       # untouched \fay stays where it was
        self.assertEqual(change["pos"], [180, 160])
        with self.assertRaisesRegex(ValueError, "layout origin"):
            apply_group_transform([replace(state, layout_origin=None)], {"kind": "shear", "ky": .2, "fixed": (0, 0)})

    def test_reset_restores_export_bytes_and_unrelated_overrides(self):
        doc = document()
        ev = Event(text=r"{\an2\pos(500,350)\frz8}HH")
        baseline = {"raw": r"\bord2\fr20\1c&H112233&", "fs": 55, "move": [300, 200, 400, 220, 0, 900]}
        before = apply_overrides(ev.text, baseline)[0].encode("utf-8")
        state = member_state(doc, ev, overrides=baseline)
        changed = merged(baseline, apply_group_transform([state], {"kind": "rotate", "delta": 30,
                                                                 "pivot": (400, 270)})["m"])
        state = member_state(doc, ev, overrides=changed, baseline=baseline)
        restored = merged(changed, apply_group_transform([state], {"kind": "reset"})["m"])
        self.assertEqual(restored, baseline)
        self.assertEqual(apply_overrides(ev.text, restored)[0].encode("utf-8"), before)

    def test_invalid_operations_are_explicit(self):
        state = member_state(document(), Event(text="HH"))
        for op in ({"kind": "move", "dx": float("nan")}, {"kind": "scale", "sx": -1, "fixed": (0, 0)},
                   {"kind": "shear", "kx": .1, "ky": .2, "fixed": (0, 0)}, {"kind": "unknown"}):
            with self.assertRaises(ValueError):
                apply_group_transform([state], op)
        for name in ("dfrx", "dfry"):
            with self.assertRaises(ValueError):
                apply_group_transform([state], {"kind": "rotate3d", name: float("nan"), "pivot": (0, 0)})

    def test_rotate3d_mixed_group_compensates_only_new_3d_members(self):
        doc = document()
        flat = member_state(doc, Event(text=r"{\pos(300,200)}HH"), mid="flat",
                            overrides={"raw": r"\frz30\org(200,150)\bord2"})
        tilted = member_state(doc, Event(text=r"{\pos(400,260)}HH"), mid="tilted",
                              overrides={"raw": r"\org(220.00,170.00)\frx10\fry-5\frz12.34"})
        changes = apply_group_transform([flat, tilted], {"kind": "rotate3d", "dfrx": 5.26,
                                                        "dfry": -7.24, "pivot": (350, 270)})
        for state, expected in ((flat, (5.3, -7.2, (350, 270))), (tilted, (15.3, -12.2, (220, 170)))):
            updated = merged(state.overrides, changes[state.member])
            tf = transform_of(doc, Event(text=apply_overrides("HH", updated)[0]))
            self.assertEqual((tf.frx, tf.fry, tf.org), expected)
        self.assertEqual(changes["tilted"]["pos"], [400, 260])
        self.assertIn(r"\org(220.00,170.00)", changes["tilted"]["raw"])
        self.assertIn(r"\frz12.34", changes["tilted"]["raw"])
        self.assertIn(r"\bord2", changes["flat"]["raw"])
        before = line_quad((300,200,320,220), (300,200), flat.transform)
        pos = changes["flat"]["pos"]
        tf = transform_of(doc, Event(text=apply_overrides("HH", merged(flat.overrides, changes["flat"]))[0]))
        after = line_quad((pos[0],pos[1],pos[0]+20,pos[1]+20), tuple(pos), replace(tf, frx=0, fry=0))
        self.assertLessEqual(float(np.abs(np.asarray(before)-after).max()), .1)

    def test_existing_3d_move_and_rotate_keep_explicit_or_implicit_origin(self):
        doc = document()
        for origin in ("", r"\org(200.00,150.00)"):
            state = member_state(doc, Event(text=r"{\move(300,200,400,250,20,900)}HH"),
                                 overrides={"raw": origin + r"\frx20\fry-10\frz15"})
            for kind in ("rotate", "rotate3d"):
                change = apply_group_transform([state], {"kind": kind, "delta": 30, "dfrx": 5,
                                                         "dfry": -3, "pivot": (900, 500)})["m"]
                self.assertEqual(change["move"], [300,200,400,250,20,900])
                tf = transform_of(doc, Event(text=apply_overrides("HH", merged(state.overrides, change))[0]))
                self.assertEqual(tf.org, state.transform.org)
                self.assertEqual(tf.frz, 45 if kind == "rotate" else 15)
                self.assertEqual((tf.frx,tf.fry), (25,-13) if kind == "rotate3d" else (20,-10))
            moved = apply_group_transform([state], {"kind": "move", "dx": 12, "dy": -7})["m"]
            self.assertEqual(moved["move"], [312,193,412,243,20,900])
            tf = transform_of(doc, Event(text=apply_overrides("HH", merged(state.overrides, moved))[0]))
            self.assertEqual(tf.org, (212,143) if origin else None)

    def test_existing_3d_rejects_pivot_scale_and_shear(self):
        state = member_state(document(), Event(text="HH"))
        for tf in (LineTransform(frx=20), LineTransform(fry=-10)):
            for op in ({"kind":"set_pivot", "pivot":(100,100)},
                       {"kind":"scale", "fixed":(100,100)},
                       {"kind":"shear", "kx":.1, "fixed":(100,100)}):
                with self.assertRaisesRegex(ValueError, "^3D:"):
                    apply_group_transform([replace(state, transform=tf)], op)

    def test_3d_source_identity_tags_and_reset_preserve_export(self):
        doc = document()
        ev = Event(text=r"{\pos(300,200)\frx10\fry-20}HH")
        tags = source_transform_tags(ev.text)
        self.assertEqual(tags, frozenset({"frx", "fry"}))
        state = replace(member_state(doc, ev), source_tags=tags)
        change = apply_group_transform([state], {"kind":"rotate3d", "dfrx":-10,
                                                 "dfry":20, "pivot":(100,100)})["m"]
        self.assertEqual(change["raw"], r"\frx0\fry0")
        updated = merged(state.overrides, change)
        exported = apply_overrides(ev.text, updated)[0]
        self.assertEqual((transform_of(doc, Event(text=exported)).frx,
                          transform_of(doc, Event(text=exported)).fry), (0,0))
        snapshot = member_state(doc, ev, overrides=updated, baseline=state.overrides)
        restored = merged(updated, apply_group_transform([snapshot], {"kind":"reset"})["m"])
        self.assertEqual(apply_overrides(ev.text, restored)[0].encode("utf-8"), ev.text.encode("utf-8"))
        plain = member_state(doc, Event(text="HH"), overrides={"raw":r"\frx10\fry-20\t(\frx90)"})
        change = apply_group_transform([plain], {"kind":"rotate3d", "dfrx":-10,
                                                 "dfry":20, "pivot":(100,100)})["m"]
        self.assertEqual(change["raw"], r"\t(\frx90)")


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg/libass")
class LibassHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="subkit_transform_", dir=os.environ.get("TEMP"))
        cls.path = Path(cls.tmp.name)
        cls.results = []
        cls.logs = []

    @classmethod
    def tearDownClass(cls):
        evidence = os.environ.get("SUBKIT_TRANSFORM_EVIDENCE")
        if evidence:
            dest = Path(evidence)
            dest.mkdir(parents=True, exist_ok=True)
            prefix = "group-transform" if cls.__name__ == "LibassGroupTransforms" else "transform"
            (dest / (prefix + "-measurements.json")).write_text(json.dumps(cls.results, ensure_ascii=False, indent=2), encoding="utf-8")
            (dest / (prefix + "-libass.log")).write_text("\n".join(cls.logs), encoding="utf-8")
            for p in cls.path.glob("*.ass"):
                shutil.copyfile(p, dest / p.name)
        cls.tmp.cleanup()

    def render(self, doc, texts, stem, scale=1):
        copy = doc.copy()
        copy.events = [Event(style="Default", start=i * 100, end=(i + 1) * 100, text=t)
                       for i, group in enumerate(texts)
                       for t in ([group] if isinstance(group, str) else group)]
        (self.path / (stem + ".ass")).write_text(copy.serialize(), encoding="utf-8-sig")
        proc = run(["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                    f"color=c=black:s={960*scale}x{540*scale}:r=1:d={len(texts)}", "-vf", f"ass={stem}.ass,format=gray",
                    "-frames:v", str(len(texts)), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                   cwd=self.path, capture_output=True, timeout=60)
        self.logs.append(proc.stderr.decode("utf-8", "replace"))
        self.assertEqual(proc.returncode, 0, self.logs[-1])
        self.assertEqual(len(proc.stdout), len(texts) * 960 * 540 * scale * scale)
        return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, 540*scale, 960*scale)

    def assert_quad(self, base, drawn, anchor, tf, origin, label, *, base_scale=1, measured_scale=(100,100), screen_check=False):
        box = base if isinstance(base, tuple) else ink_box(base, base_scale)
        quad = line_quad(box, anchor, tf, layout_origin=origin, measured_scale=measured_scale)
        # Inverse-map actual ink to the original axes: works for skew/rotation,
        # unlike comparing only an axis-aligned output bounding rectangle.
        q = np.asarray(quad)
        matrix = np.column_stack(((q[1] - q[0]) / (box[2] - box[0]),
                                  (q[3] - q[0]) / (box[3] - box[1])))
        y, x = np.where(drawn > 24)
        local = (np.column_stack((x + .5, y + .5)) - q[0]) @ np.linalg.inv(matrix).T
        extrema = [local[:, 0].min(), local[:, 1].min(), local[:, 0].max(), local[:, 1].max()]
        error = float(np.max(np.abs(np.asarray(extrema) - [0.5, 0.5, box[2]-box[0]-.5, box[3]-box[1]-.5])))
        bounds = (q[:,0].min(), q[:,1].min(), q[:,0].max(), q[:,1].max())
        screen_error = float(np.max(np.abs(np.asarray(bounds) - ink_box(drawn))))
        self.results.append({"case": label, "base_box": box, "drawn_box": ink_box(drawn),
                             "layout_origin": origin, "quad": quad, "base_render_scale": base_scale,
                             "max_inverse_ink_error_px": error, "max_screen_bound_error_px": screen_error})
        if screen_check:
            self.assertLessEqual(screen_error, 1.0, (label, screen_error, bounds, ink_box(drawn)))
        else:
            self.assertLessEqual(error, 1.0, (label, error, extrema, box, quad))

class LibassGeometry(LibassHarness):
    def test_scale_stretch_rotation_shears_combination_nine_alignments(self):
        doc = document()
        tags = [r"\fscx150\fscy150", r"\fscx150", r"\fscy120", r"\frz30",
                r"\fax0.5", r"\fay0.3", r"\fscx130\fscy120\fax0.3\fay0.2\frz25"]
        texts, cases = [], []
        base_texts = [r"{\an7\pos(480,270)}██"]
        # A high-contrast rectangular string exposes all four ink corners.
        for an in range(1, 10):
            for positioned in (False, True):
                pos = r"\pos(480,270)" if positioned else ""
                head = "\\an" + str(an) + pos
                reference = 0
                base = len(base_texts)
                base_texts.append("{" + head + "}██")
                for tag in tags:
                    for explicit in (False, True):
                        # Style alignment anchors (margins are deliberately safe).
                        ax = 180 if (an-1)%3 == 0 else 480 if (an-1)%3 == 1 else 780
                        ay = 380 if an <= 3 else 270 if an <= 6 else 160
                        anchor = (480, 270) if positioned else (ax, ay)
                        org = r"\org(480,270)" if explicit else ""
                        index = len(texts)
                        text = "{" + head + tag + org + "}██"
                        texts.append(text)
                        cases.append((reference, base, index, anchor, text, an, positioned, explicit))
        frames = self.render(doc, texts, "all-alignments")
        # Native integer probe boxes lose up to a pixel before any transform.
        # Measure the reference at 4x to separate raster quantisation from math.
        bases = self.render(doc, base_texts, "all-alignments-reference", scale=4)
        boxes = [ink_box(f,4) for f in bases]
        del bases
        for reference, base, index, anchor, text, an, positioned, explicit in cases:
            rb, bb = boxes[reference], boxes[base]
            origin = (480 + bb[0]-rb[0], 270 + bb[1]-rb[1])
            with self.subTest(an=an, pos=positioned, org=explicit, text=text):
                self.assert_quad(boxes[base], frames[index], anchor, transform_of(doc, Event(text=text)), origin, text,
                                 base_scale=4, screen_check=True)

    def test_already_scaled_ink_is_not_scaled_twice(self):
        doc = document()
        plain = r"{\an7\pos(400,220)\fscx130\fscy120}██"
        transformed = r"{\an7\pos(400,220)\fscx130\fscy120\fax0.3\fay0.2\frz25}██"
        base = self.render(doc, [plain], "already-scaled-reference", scale=4)[0]
        drawn = self.render(doc, [transformed], "already-scaled")[0]
        self.assert_quad(base, drawn, (400,220), transform_of(doc, Event(text=transformed)), (400,220),
                         "already-scaled", base_scale=4, measured_scale=(130,120), screen_check=True)

    def test_hard_break_resets_shear_frame(self):
        doc = document()
        tags = r"\fax0.5\fay0.3\frz20"
        texts = [r"{\an7\pos(400,220)}H\NH", "{\\an7\\pos(400,220)" + tags + r"}H\NH"]
        frames = self.render(doc, texts, "hard-break")
        # Independently rendered visual lines use their own left/top, while
        # both rotate about the event anchor.
        separate = self.render(doc, [r"{\an7\pos(400,220)\org(400,220)" + tags + "}H",
                                     r"{\an7\pos(400,300)\org(400,220)" + tags + "}H"], "separate-lines")
        self.assertLessEqual(int(np.abs(frames[1].astype(int) - np.maximum(*separate).astype(int)).max()), 1)
        boxes = [ink_box(frames[0][:300]), ink_box(frames[0][300:])]
        boxes[1] = (boxes[1][0], boxes[1][1]+300, boxes[1][2], boxes[1][3]+300)
        tf = transform_of(doc, Event(text=texts[1]))
        quads = line_quads(boxes, (400, 220), tf, layout_origins=[(400,220),(400,300)])
        self.assertEqual(len(quads), 2)
        for n, box in enumerate(boxes):
            base = np.zeros_like(frames[0])
            lo, hi = (0,300) if n == 0 else (300,540)
            base[lo:hi] = frames[0][lo:hi]
            self.assert_quad(base, separate[n], (400,220), tf, (400,220+n*80), f"hard-break-{n}")

    def test_real_measurer_scales_and_style_angle(self):
        doc = document()
        m = Measurer(doc)
        requests = [LineRequest("Default", text, 7, 300) for text in
                    ("HH", r"{\fscx150\fscy120}HH", r"{\frz30\fax0.5\fay0.3}HH")]
        plain, scaled, transformed = m.line_char_boxes(requests)
        self.assertEqual(plain, transformed)
        self.assertGreater(scaled[-1].x1- scaled[0].x0, (plain[-1].x1-plain[0].x0)*1.4)
        self.assertGreater(scaled[0].height, plain[0].height*1.1)
        doc.styles[0].set("Angle", 30)
        rotated = Measurer(doc).line_char_boxes([requests[0]])[0]
        self.assertNotEqual(plain, rotated)
        neutral, ev = measurement_copy(doc, Event(text="HH"))
        self.assertEqual(Measurer(neutral).line_char_boxes([LineRequest(ev.style, ev.text, 7, 300)])[0], plain)

    def test_ruby_uses_its_position_and_shared_rotation_origin(self):
        doc = document()
        doc.styles.append(Style.new("Ruby", Fontname="Arial", Fontsize="40", Outline="0", Shadow="0", Alignment="2"))
        base = Event(style="Default", start=0, end=100, text=r"{\an2\pos(480,350)\frz30}HHHH")
        job = RubyJob(base, "Ruby", [RubyPiece(1, 3, "HH")])
        RubyGenerator(doc, Measurer(doc)).run([job])
        self.assertEqual(len(job.out), 1)
        ev = job.out[0]
        tf = transform_of(doc, ev)
        self.assertEqual(tf.org, (480,350))
        self.assertEqual(tf.frz, 30)
        # Render the generated event as Default with the actual ruby style.
        rd = doc.copy()
        rd.styles = [doc.style("Ruby").copy("Default")]
        plain = measurement_copy(rd, ev)[1].text
        frames = self.render(rd, [plain, ev.text], "ruby")
        from subkit.ass.text import line_tag
        anchor = tuple(line_tag(ev.text, "pos").float_args())
        self.assert_quad(frames[0], frames[1], anchor, tf, anchor, "generated-ruby-frz30")

    def test_org_first_wins_even_after_text(self):
        doc = document()
        frames = self.render(doc, [r"{\an7\pos(400,220)\frz30\org(480,270)}HH",
                                   r"{\an7\pos(400,220)\frz30\org(480,270)\org(1,2)}HH",
                                   r"{\an7\pos(400,220)\frz30}HH{\org(480,270)}"], "org-order")
        self.assertTrue(np.array_equal(frames[0], frames[1]))
        self.assertTrue(np.array_equal(frames[0], frames[2]))

    def test_numeric_boundaries_match_libass(self):
        doc = document()
        doc.styles[0].set("ScaleX", 130)
        tags = ["", r"\fscx", r"\fscx-20", r"\fscx0", r"\fax.5", r"\fax0.5", r"\fay1e-1", r"\fay0.1"]
        frames = self.render(doc, [r"{\an7\pos(400,220)" + t + "}HH" for t in tags], "number-boundaries")
        self.assertTrue(np.array_equal(frames[0], frames[1]))
        self.assertFalse(np.any(frames[2] > 24))
        self.assertFalse(np.any(frames[3] > 24))
        self.assertTrue(np.array_equal(frames[4], frames[5]))
        self.assertTrue(np.array_equal(frames[6], frames[7]))


class LibassGroupTransforms(LibassHarness):
    @staticmethod
    def matrix(angle):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        return np.asarray([[c, s], [-s, c]])

    def assert_bounds(self, quad, frame, label, scale=1):
        q = np.asarray(quad)
        bounds = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max())
        drawn = ink_box(frame, scale)
        error = float(np.abs(np.asarray(bounds) - drawn).max())
        self.results.append({"case": label, "expected_bounds": bounds, "drawn_bounds": drawn,
                             "max_error_px": error, "render_scale": scale})
        self.assertLessEqual(error, 1, (label, error, bounds, drawn))

    def test_rotate_two_line_group_rigid_and_reset_exact(self):
        doc = document()
        originals = ["██", r"{\an7\pos(260,280)}██"]  # CN above JP; first uses style margins.
        states = [member_state(doc, Event(text=text), mid=mid)
                  for mid, text in zip(("cn", "jp"), originals)]
        pivot = np.asarray((400, 270))
        changes = apply_group_transform(states, {"kind": "rotate", "delta": 30, "pivot": pivot})
        transformed = [apply_overrides(text, changes[state.member])[0] for text, state in zip(originals, states)]
        bases = self.render(doc, originals, "group-rotate-reference", scale=4)
        frames = self.render(doc, [*transformed, transformed], "group-rotate")
        quads = []
        for n, base in enumerate(bases):
            x0, y0, x1, y1 = ink_box(base, 4)
            q = (np.asarray([(x0,y0), (x1,y0), (x1,y1), (x0,y1)]) - pivot) @ self.matrix(30).T + pivot
            quads.append(q)
            self.assert_bounds(q, frames[n], f"group-rotate-{n}")
        self.assert_bounds(np.concatenate(quads), frames[2], "group-rotate-combined")
        # Positions introduced solely for transformation disappear on reset.
        updated = [member_state(doc, Event(text=text), mid=state.member,
                                overrides=merged(state.overrides, changes[state.member]), baseline=state.overrides)
                   for text, state in zip(originals, states)]
        reset = apply_group_transform(updated, {"kind": "reset"})
        restored = [apply_overrides(text, merged(state.overrides, reset[state.member]))[0]
                    for text, state in zip(originals, updated)]
        self.assertEqual([s.encode("utf-8") for s in restored], [s.encode("utf-8") for s in originals])
        self.assertTrue(np.array_equal(self.render(doc, [originals], "group-reset-before")[0],
                                       self.render(doc, [restored], "group-reset-after")[0]))

    def test_scale_corner_fixed_and_stretch_rotated_frame(self):
        doc = document()
        source = r"{\an7\pos(350,220)}██"
        base = self.render(doc, [source], "group-scale-reference", scale=4)[0]
        box = ink_box(base, 4)
        for n, raw in enumerate(("", r"\frz30\org(400,270)")):
            state = member_state(doc, Event(text=source), overrides={"raw": raw} if raw else {})
            quad = np.asarray(line_quad(box, (350, 220), state.transform))
            fixed = quad[0]
            op = {"kind": "scale", "sx": 1.5, "sy": 1.5 if n == 0 else 1.2,
                  "fixed": fixed, "frame_angle": state.transform.frz}
            change = apply_group_transform([state], op)["m"]
            out = apply_overrides(source, merged(state.overrides, change))[0]
            rotation = self.matrix(state.transform.frz)
            expected = (quad-fixed) @ rotation @ np.diag([op["sx"], op["sy"]]) @ rotation.T + fixed
            drawn = self.render(doc, [out], f"group-scale-{n}", scale=4)[0]
            self.assert_bounds(expected, drawn, f"group-scale-{n}", scale=4)
            if n == 0:
                corner_error = float(np.abs(np.asarray(ink_box(drawn, 4)[:2])-fixed).max())
                self.results.append({"case": "group-scale-fixed-corner", "max_error_px": corner_error})
                self.assertLessEqual(corner_error, 1)

    def test_set_pivot_rotated_group_moves_no_ink(self):
        doc = document()
        sources = [r"{\an7\pos(350,160)}██", r"{\an7\pos(380,280)}██"]
        states = [member_state(doc, Event(text=text), mid=str(n),
                               overrides={"raw": r"\frz30\org(400,270)\fscx130\fax0.2"})
                  for n, text in enumerate(sources)]
        changes = apply_group_transform(states, {"kind": "set_pivot", "pivot": (320.3, 250.7)})
        before = [apply_overrides(text, state.overrides)[0] for text, state in zip(sources, states)]
        after = [apply_overrides(text, merged(state.overrides, changes[state.member]))[0]
                 for text, state in zip(sources, states)]
        frames = self.render(doc, [before, after], "group-pivot", scale=4)
        error = float(np.abs(np.asarray(ink_box(frames[0], 4))-ink_box(frames[1], 4)).max())
        self.results.append({"case": "group-pivot-no-ink-move", "max_error_px": error, "render_scale": 4})
        self.assertLessEqual(error, 1)
        # Check individual lines too; a stable union alone could hide movement.
        frames = self.render(doc, [*before, *after], "group-pivot-members", scale=4)
        for n in range(2):
            error = float(np.abs(np.asarray(ink_box(frames[n], 4))-ink_box(frames[n+2], 4)).max())
            self.results.append({"case": f"group-pivot-member-{n}", "max_error_px": error})
            self.assertLessEqual(error, 1)

    def test_shear_composes_in_rotated_scaled_frame(self):
        doc = document()
        source = r"{\an7\pos(350,220)}██"
        base = self.render(doc, [source], "group-shear-reference", scale=4)[0]
        box = ink_box(base, 4)
        for axis in ("kx", "ky"):
            state = member_state(doc, Event(text=source),
                                 overrides={"raw": r"\frz30\org(400,270)\fscx130\fscy120\fax0.2\fay0.1"},
                                 layout=(350, 220))
            quad = np.asarray(line_quad(box, (350, 220), state.transform, layout_origin=(350, 220)))
            op = {"kind": "shear", axis: .25, "fixed": (350, 220), "frame_angle": 30}
            change = apply_group_transform([state], op)["m"]
            out = apply_overrides(source, merged(state.overrides, change))[0]
            rotation = self.matrix(30)
            shear = np.asarray([[1, .25 if axis == "kx" else 0], [.25 if axis == "ky" else 0, 1]])
            expected = (quad-op["fixed"]) @ rotation @ shear.T @ rotation.T + op["fixed"]
            drawn = self.render(doc, [out], "group-shear-"+axis)[0]
            self.assert_bounds(expected, drawn, "group-shear-"+axis)


class Libass3D(LibassHarness):
    @classmethod
    def tearDownClass(cls):
        # Measurements and original libass version messages stay in test logs.
        try:
            print("3D measurements: " + json.dumps(cls.results, ensure_ascii=False))
            if cls.logs:
                print("\n".join(line for line in cls.logs[0].splitlines()
                                if "libass" in line or "fontselect" in line))
        finally:
            super().tearDownClass()

    def assert_3d_bounds(self, quad, frame, label, *, scale=1):
        q = np.asarray(quad)
        predicted = (q[:,0].min(), q[:,1].min(), q[:,0].max(), q[:,1].max())
        actual = ink_box(frame, scale)
        error = float(np.abs(np.asarray(predicted)-actual).max())
        self.results.append({"case":label, "quad":quad, "drawn_bounds":actual,
                             "max_error_px":error, "render_scale":scale})
        self.assertLessEqual(error, 1, (label, error, predicted, actual))

    def test_perspective_xyz_scale_shear_and_all_alignments(self):
        doc = document()
        transforms = [r"\frx30", r"\frx-30", r"\fry30", r"\fry-30",
                      r"\frx45\fry30\frz25", r"\frx60\fry-45",
                      r"\frx-40\fry55\frz-20\fscx130\fscy120\fax0.2\fay0.1",
                      r"\frx120\fry-25\frz15"]
        bases, texts, cases = [], [], []
        for an in range(1,10):
            # Explicit and style-margin anchors are both tested.
            for positioned in (False, True):
                head = f"\\an{an}" + (r"\pos(480,270)" if positioned else "")
                anchor = ((480,270) if positioned else
                          (180 if (an-1)%3 == 0 else 480 if (an-1)%3 == 1 else 780,
                           380 if an <= 3 else 270 if an <= 6 else 160))
                base_index = len(bases)
                bases.append("{"+head+"}██")
                for tag in transforms:
                    for explicit in (False, True):
                        # Keep the distant camera-plane cases separate below.
                        org = f"\\org({anchor[0]+30},{anchor[1]+20})" if explicit else ""
                        text = "{"+head+tag+org+"}██"
                        cases.append((base_index, anchor, text))
                        texts.append(text)
        boxes = [ink_box(frame,4) for frame in self.render(doc, bases, "3d-reference", scale=4)]
        # Measure layout offsets independently for each alignment.
        reference = boxes[12]  # an7 without pos: top-left margins (180,160).
        # Layout origin = anchor plus the alignment displacement, not ink bearing.
        origins = [(180+box[0]-reference[0],160+box[1]-reference[1]) for box in boxes]
        frames = self.render(doc, texts, "3d-perspective")
        for (n, anchor, text), frame in zip(cases, frames):
            with self.subTest(text=text):
                quad = line_quad(boxes[n], anchor, transform_of(doc, Event(text=text)), layout_origin=origins[n])
                self.assert_3d_bounds(quad, frame, text)

    def test_video_storage_size_changes_camera_distance(self):
        doc = document()
        head = r"\an7\pos(400,220)"
        box = ink_box(self.render(doc, ["{"+head+"}██"], "3d-storage-base", scale=4)[0],4)
        text = "{"+head+r"\frx30\fry-25\frz15}██"
        # ffmpeg's ass filter sets storage size to the input frame size.
        # At 2x the camera distance in script units is halved, not unchanged.
        frame = self.render(doc, [text], "3d-storage-2x", scale=2)[0]
        quad = line_quad(box, (400,220), transform_of(doc, Event(text=text)), perspective_distance=156.25)
        self.assert_3d_bounds(quad, frame, "storage-1920x1080", scale=2)
        wrong = np.asarray(line_quad(box, (400,220), transform_of(doc, Event(text=text))))
        wrong_bounds = (wrong[:,0].min(),wrong[:,1].min(),wrong[:,0].max(),wrong[:,1].max())
        self.assertGreater(float(np.abs(np.asarray(wrong_bounds)-ink_box(frame,2)).max()), 3)

    def test_rotate3d_zero_delta_does_not_jump(self):
        doc = document()
        sources = [r"{\an7\pos(350,160)}██", r"{\an7\pos(380,280)}██",
                   r"{\an7\move(320,240,350,270,0,100)}██", "██"]
        raws = [r"\frz30\org(400,270)\fscx130\fax0.2", r"\frz-20",
                r"\frx25\fry-15\frz10\org(400.00,270.00)", r"\frx20\frz15"]
        states = [member_state(doc, Event(text=text), mid=str(n), overrides={"raw":raw})
                  for n,(text,raw) in enumerate(zip(sources,raws))]
        changes = apply_group_transform(states, {"kind":"rotate3d", "dfrx":0, "dfry":0,
                                                 "pivot":(320.3,250.7)})
        before = [apply_overrides(text,state.overrides)[0] for text,state in zip(sources,states)]
        after = [apply_overrides(text,merged(state.overrides,changes[state.member]))[0]
                 for text,state in zip(sources,states)]
        # Separate frames catch member movement that a stable union could hide.
        frames = self.render(doc, [*before,*after], "3d-zero-delta")
        for n in range(len(states)):
            error = float(np.abs(np.asarray(ink_box(frames[n]))-ink_box(frames[n+len(states)])).max())
            self.results.append({"case":f"rotate3d-zero-{n}", "max_error_px":error})
            self.assertLessEqual(error,1)


if __name__ == "__main__":
    unittest.main()
