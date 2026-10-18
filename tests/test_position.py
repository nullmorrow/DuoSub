"""Numeric positioning, undo/source safety, and real libass layout invariants."""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
import stat
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass.document import AssDocument, Event, Style
from subkit.ass.text import tokenize
from subkit.core.export import apply_overrides
from subkit.core.layout_index import LayoutIndex, Rect
from subkit.core.ops import Editor
from subkit.core.position import (PositionTarget, preset_side, restore, set_alignment, set_xy,
                                  source_positioned)
from subkit.core.transform import LineTransform, transform_of
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.source import Source, SourceEvent
from subkit.proc import run


def document():
    doc = AssDocument()
    for key, value in (("PlayResX", 960), ("PlayResY", 540), ("WrapStyle", 2), ("ScaledBorderAndShadow", "yes")):
        doc.set_info(key, value)
    doc.styles = [Style.new("Default", Fontname="Arial", Fontsize="80", Outline="0", Shadow="0",
                            MarginL="180", MarginR="180", MarginV="160", Alignment="2")]
    return doc


def merged(overrides, change):
    result = copy.deepcopy(overrides)
    for key, value in change.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def target(mid="m", *, an=2, x=480., y=380., move=None, overrides=None, transform=LineTransform()):
    width, height = 100., 80.
    left = x - (an - 1) % 3 * width / 2
    top = y - (1 - (an - 1) // 3 / 2) * height
    return PositionTarget(mid, "original", "dialogue", document().styles[0], (0, 0, 0),
                          copy.deepcopy(overrides or {}), False,
                          {"an": an, "x": x, "y": y, "move": move}, Rect(left, top, left + width, top + height), transform)


class PositionMath(unittest.TestCase):
    def test_group_delta_line_scope_missing_axis_and_rounding(self):
        cn, jp = target("cn", y=300), target("jp", y=380)
        changes = set_xy([cn, jp], jp, x=500.26, y=400.24)
        self.assertEqual(changes["cn"]["pos"], [500.3, 320.2])
        self.assertEqual(changes["jp"]["pos"], [500.3, 400.2])
        changes = set_xy([cn, jp], "jp", x=-5.26, scope="line")
        self.assertEqual(set(changes), {"jp"})
        self.assertEqual(changes["jp"]["pos"], [-5.3, 380])
        self.assertEqual(set_xy([cn], cn), {})
        self.assertEqual(cn.overrides, {})

    def test_move_times_and_explicit_origin_follow_xy(self):
        raw = r"\bord2\org(200,150)\frz30\t(0,500,\bord4)"
        t = target(move=[480, 380, 520, 400, 20, 900], overrides={"raw": raw, "pos": [1, 2]},
                   transform=LineTransform(frz=30, org=(200, 150)))
        change = set_xy([t], t, x=500, y=370)["m"]
        self.assertEqual(change["move"], [500, 370, 540, 390, 20, 900])
        self.assertIsNone(change["pos"])
        self.assertEqual(change["raw"], r"\bord2\frz30\t(0,500,\bord4)\org(220,140)")
        self.assertEqual(t.overrides["raw"], raw)

    def test_alignment_points_scope_and_move_shift(self):
        a = target("a", move=[480, 380, 520, 400, 20, 900])
        b = target("b")
        change = set_alignment([a, b], 7, "line")["a"]
        self.assertEqual(change["an"], 7)
        self.assertEqual(change["move"], [430, 300, 470, 320, 20, 900])
        self.assertEqual(set(set_alignment([a, b], 9, "line", reference="b")), {"b"})
        self.assertEqual(set(set_alignment([a, b], 8)), {"a", "b"})
        self.assertEqual(set_alignment([b], 2), {"b": {}})

    def test_rotations_pin_implicit_org_and_keep_source_org(self):
        for tf in (LineTransform(frz=30), LineTransform(frx=20), LineTransform(fry=-20)):
            t = target(transform=tf, overrides={"raw": r"\blur1"})
            self.assertEqual(set_alignment([t], 7)["m"]["raw"], r"\blur1\org(480,380)")
        t = target(transform=LineTransform(frz=30, org=(200, 100)))
        self.assertEqual(set_alignment([t], 7)["m"]["raw"], r"\org(200,100)")
        t = replace(t, overrides={"raw": r"\org(200.00,100.00)\blur1"})
        self.assertNotIn("raw", set_alignment([t], 7)["m"])

    def test_side_preserves_each_alignment_row_and_member_margins(self):
        for an in range(1, 10):
            t = replace(target(an=an), margins=(25, 35, 0))
            for side, x, col in (("left", 25, 1), ("right", 925, 3)):
                ch = preset_side([t], side, (960, 540))["m"]
                self.assertEqual(ch["an"], (an - 1) // 3 * 3 + col)
                self.assertEqual(ch["pos"], [x, 380])
        t = target()
        self.assertEqual(preset_side([t], "left", (960, 540))["m"]["pos"], [180, 380])
        t = target(move=[480, 380, 520, 400, 20, 900], transform=LineTransform(org=(100, 100)))
        ch = preset_side([t], "right", (960, 540))["m"]
        self.assertEqual(ch["move"], [780, 380, 820, 400, 20, 900])
        self.assertEqual(ch["raw"], r"\org(400,100)")

    def test_restore_only_removes_position_overrides(self):
        overrides = {"an": 9, "pos": [100, 200], "move": [1, 2, 3, 4], "raw": r"\bord2", "fs": 90}
        t = target(overrides=overrides)
        ch = restore([t])["m"]
        self.assertEqual(ch, {"an": None, "pos": None, "move": None})
        restored = merged(overrides, ch)
        self.assertEqual(restored, {"raw": r"\bord2", "fs": 90})
        source = r"{\an7\pos(300,200)}Hello"
        text = apply_overrides(source, restored)[0]
        self.assertIn(r"\an7\pos(300,200)", text)
        self.assertNotIn(r"\an9", text)

    def test_invalid_inputs(self):
        t = target()
        for an in (0, 10, 2.5, True):
            with self.assertRaises(ValueError):
                set_alignment([t], an)
        for call in (lambda: set_xy([t], t, x=float("nan")),
                     lambda: set_xy([t], t, y=float("inf")),
                     lambda: set_xy([t], "absent", x=1),
                     lambda: set_xy([t, t], t, x=1),
                     lambda: set_alignment([t], 7, "unknown"),
                     lambda: preset_side([t], "unknown", (960, 540)),
                     lambda: set_xy([replace(t, anchor={**t.anchor, "move": [1, 2, 3]})], t, x=1)):
            with self.assertRaises(ValueError):
                call()


class PositionEdits(unittest.TestCase):
    def setUp(self):
        self.source = Source("s", "ja", "original", "ass", "", "", "", 0, "utf-8", "lf")
        self.doc = Document("position")
        self.ed = Editor(self.doc, {"s": self.source})

    def member(self, mid, text="Hello", *, lane="original", role="dialogue", kind="dialogue", gid=None):
        gid = gid or "g_" + mid
        e = len(self.source.events)
        self.source.events.append(SourceEvent(e, "Dialogue", 0, 2000, text))
        runs = [Run(tok.raw, "tag" if tok.kind == "tags" else "text", "s", e, tok.start, tok.end)
                for tok in tokenize(text)]
        m = Member(mid, gid, "ja", lane, role, 0, 2000, style="CN" if lane == "translation" else "JP", runs=runs)
        self.doc.members[mid] = m
        g = next((g for g in self.doc.groups if g.id == gid), None)
        if g is None:
            self.doc.groups.append(Group(gid, kind, [mid]))
            self.doc.structure_changed()
        else:
            g.members.append(mid)
        return m

    def test_top_bottom_roundtrip_undo_and_nonposition_fields(self):
        a = self.member("cn", lane="translation", gid="g")
        b = self.member("jp", role="alt", gid="g")
        a.overrides = {"an": 9, "pos": [1, 2], "raw": r"\bord2", "fs": 90}
        b.overrides = {"move": [1, 2, 3, 4, 20, 900]}
        self.doc.annotations["a"] = Annotation("a", "ruby", b.id, [], "H", "h")
        self.doc.karaoke["k"] = {"id": "k", "member": a.id, "beats": [{"t0": 0, "t1": 500}]}
        source_before = copy.deepcopy(self.source)
        members_before = copy.deepcopy(self.doc.members)
        others = copy.deepcopy((self.doc.groups, self.doc.annotations, self.doc.karaoke))
        result = self.ed.set_vertical(["g", "g"], "top")
        self.assertEqual((result.gids, result.members, result.skipped), (["g"], ["cn", "jp"], []))
        self.assertEqual((a.role, a.style, b.role, b.style), ("top", "CN.Top", "top", "JP.Top"))
        self.assertEqual(a.overrides, {"raw": r"\bord2", "fs": 90})
        self.assertEqual(b.overrides, {})
        self.assertEqual(len(self.ed.history.done), 1)
        self.assertTrue(self.ed.history.done[-1].keeps_text)
        self.assertEqual(self.ed.history.done[-1].label, "移到顶部")
        self.ed.history.undo()
        self.assertEqual(self.doc.members, members_before)
        self.ed.history.redo()
        self.ed.set_vertical(["g"], "bottom")
        self.assertEqual(len(self.ed.history.done), 2)
        self.assertEqual(self.ed.history.done[-1].label, "移到底部")
        self.assertEqual([(m.role, m.style) for m in self.doc.members.values()], [("dialogue", "CN"), ("dialogue", "JP")])
        # 底部第二人: the template's CN.Alt / JP.Alt, also with a stored style map that predates the alt keys
        self.ed.set_vertical(["g"], "alt", styles={"translation/dialogue": "CN", "original/dialogue": "JP"})
        self.assertEqual(self.ed.history.done[-1].label, "底部第二人")
        self.assertEqual([(m.role, m.style) for m in self.doc.members.values()], [("alt", "CN.Alt"), ("alt", "JP.Alt")])
        self.ed.set_vertical(["g"], "bottom")
        self.assertEqual((self.doc.groups, self.doc.annotations, self.doc.karaoke), others)
        self.assertEqual(self.source, source_before)
        for mid in members_before:
            self.assertEqual(self.doc.members[mid].runs, members_before[mid].runs)

    def test_source_event_tags_skip_even_if_runs_omit_them(self):
        for n, text in enumerate((r"{\pos(1,2)}H", r"H{\an7}", r"{\move(1,2,3,4)}H")):
            m = self.member(str(n), text)
            m.runs = [r for r in m.runs if r.kind != "tag"]
            m.overrides = {"an": 2, "pos": [480, 380]}
            self.assertTrue(source_positioned(m, self.ed.sources))
        before = copy.deepcopy(self.doc.members)
        result = self.ed.set_vertical([g.id for g in self.doc.groups])
        self.assertEqual(len(result.skipped), 3)
        self.assertTrue(all(s.reason == "原字幕自带位置，不改位置" and s.member for s in result.skipped))
        self.assertEqual(self.doc.members, before)
        self.assertEqual(len(self.ed.history.done), 0)

    def test_protected_members_groups_custom_styles_and_validation(self):
        m = self.member("plain", gid="mixed")
        for role in ("song", "sign", "info", "comment"):
            self.member(role, role=role, gid="mixed")
        for kind in ("song", "sign", "info"):
            self.member("group_" + kind, kind=kind)
        before = copy.deepcopy(self.doc.members)
        result = self.ed.set_vertical([g.id for g in self.doc.groups] + ["absent"], styles={"original/top": "Custom.Top"})
        self.assertEqual(result.members, [m.id])
        self.assertEqual(len(result.skipped), 8)
        self.assertEqual(m.style, "Custom.Top")
        for mid in before.keys() - {m.id}:
            self.assertEqual(self.doc.members[mid], before[mid])
        with self.assertRaises(ValueError):
            self.ed.set_vertical([m.group], "side")
        # Missing keys fall back to the default style map (stored maps predate the alt keys).
        self.ed.set_vertical([m.group], "bottom", styles={})
        self.assertEqual((m.role, m.style), ("dialogue", "JP"))
        self.assertEqual(len(self.ed.history.done), 2)

    def test_each_numeric_action_one_undo_and_source_file_untouched(self):
        with tempfile.TemporaryDirectory(prefix="subkit_position_source_", dir=os.environ.get("TEMP")) as scratch:
            path = Path(scratch) / "source.ass"
            path.write_text(r"{\an7\pos(300,200)}Hello", encoding="utf-8")
            path.chmod(stat.S_IREAD)
            try:
                digest = hashlib.sha256(path.read_bytes()).digest()
                m = self.member("m", path.read_text(encoding="utf-8"))
                self.source.path = str(path)
                for action in (lambda t: set_xy([t], t, x=500.1), lambda t: set_alignment([t], 7),
                               lambda t: preset_side([t], "right", (960, 540)), lambda t: restore([t])):
                    m = self.doc.members["m"]
                    t = target(overrides=m.overrides)
                    before = copy.deepcopy(m)
                    count = len(self.ed.history.done)
                    self.ed.set_overrides(action(t))
                    self.assertEqual(len(self.ed.history.done), count + 1)
                    self.assertEqual(self.doc.members["m"].runs, before.runs)
                    self.ed.history.undo()
                    self.assertEqual(self.doc.members["m"], before)
                    self.ed.history.redo()
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)
            finally:
                # Only this test's private copy is made writable for cleanup.
                path.chmod(stat.S_IREAD | stat.S_IWRITE)


def ink_box(frame, scale=1):
    y, x = np.where(frame > 24)
    if not x.size:
        raise AssertionError("libass rendered no ink")
    return np.asarray((x.min(), y.min(), x.max() + 1, y.max() + 1), dtype=float) / scale


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
class LibassPosition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="subkit_position_", dir=os.environ.get("TEMP"))
        cls.path = Path(cls.tmp.name)
        cls.max_error = 0.
        cls.cases = 0
        cls.version = []
        cls.started = time.perf_counter()

    @classmethod
    def tearDownClass(cls):
        print(f"Position libass: {cls.cases} comparisons; max ink/block error {cls.max_error:.3f} px; "
              f"{time.perf_counter() - cls.started:.3f}s")
        print("\n".join(cls.version))
        cls.tmp.cleanup()

    def snapshots(self, doc, texts):
        doc = doc.copy()
        doc.events = [Event(text=t) for t in texts]
        owners = [{"member": str(i), "group": "g", "kind": "base"} for i in range(len(texts))]
        index = LayoutIndex(doc, owners)
        try:
            index.layout_boxes(list(range(len(texts))))
            return [PositionTarget.from_layout(index, i, Member(str(i), "g", "ja", "original", "dialogue", 0, 1000), {})
                    for i in range(len(texts))]
        finally:
            index.close()

    def render(self, doc, texts):
        copy_doc = doc.copy()
        copy_doc.events = [Event(start=i * 100, end=(i + 1) * 100, text=t) for i, t in enumerate(texts)]
        (self.path / "render.ass").write_text(copy_doc.serialize(), encoding="utf-8-sig")
        proc = run(["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                    f"color=c=black:s=960x540:r=1:d={len(texts)}", "-vf", "ass=render.ass,format=gray",
                    "-frames:v", str(len(texts)), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                   cwd=self.path, capture_output=True, timeout=60)
        log = proc.stderr.decode("utf-8", "replace")
        if not self.version:
            type(self).version = [line for line in log.splitlines() if "libass" in line or "fontselect" in line]
        self.assertEqual(proc.returncode, 0, log)
        self.assertEqual(len(proc.stdout), len(texts) * 960 * 540)
        return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, 540, 960)

    def error(self, before, after, label):
        error = float(np.abs(np.asarray(before) - after).max())
        type(self).max_error = max(self.max_error, error)
        type(self).cases += 1
        self.assertLessEqual(error, 1., (label, error, before, after))

    def test_single_line_all_nine_alignments_positions_move_and_rotations(self):
        for mode, tags in (("style", ""), ("pos", r"\pos(480,270)"),
                           ("move", r"\move(480,270,510,290,0,10000)"),
                           ("frz", r"\pos(480,270)\frz25"), ("style-angle", r"\pos(480,270)"),
                           ("frx", r"\pos(480,270)\frx25"), ("fry", r"\pos(480,270)\fry-25"),
                           ("org", r"\pos(480,270)\frz20\org(420,240)")):
            doc = document()
            if mode == "style-angle":
                doc.styles[0].set("Angle", 20)
            originals = [f"{{\\an{an}{tags}}}HH" for an in range(1, 10)]
            states = self.snapshots(doc, originals)
            texts = list(originals)
            for t, source in zip(states, originals):
                for an in range(1, 10):
                    texts.append(apply_overrides(source, set_alignment([t], an)[t.member])[0])
            frames = self.render(doc, texts)
            for i in range(9):
                for an in range(1, 10):
                    self.error(ink_box(frames[i]), ink_box(frames[9 + i * 9 + an - 1]), (mode, i + 1, an))

    def test_multiline_layout_block_and_widest_line_stay_in_place(self):
        doc = document()
        originals = [f"{{\\an{an}\\pos(480,270)}}HHHH\\NH" for an in range(1, 10)]
        states = self.snapshots(doc, originals)
        changed = [apply_overrides(source, set_alignment([t], an)[t.member])[0]
                   for t, source in zip(states, originals) for an in range(1, 10)]
        after = self.snapshots(doc, changed)
        frames = self.render(doc, originals + changed)
        for i, t in enumerate(states):
            split = int(t.layout_box.cy)
            for an in range(1, 10):
                new = after[i * 9 + an - 1]
                self.error(t.layout_box.corners, new.layout_box.corners, ("block", i + 1, an))
                self.error(ink_box(frames[i][:split]), ink_box(frames[9 + i * 9 + an - 1][:split]),
                           ("widest", i + 1, an))

    def test_alignment_after_side_presets_near_screen_edges(self):
        doc = document()
        originals = [r"{\an7\pos(40,200)}HH", r"{\an9\pos(920,200)}HH"]
        states = self.snapshots(doc, originals)
        changed = [apply_overrides(text, set_alignment([t], 8)[t.member])[0]
                   for t, text in zip(states, originals)]
        frames = self.render(doc, originals + changed)
        for i in range(2):
            self.error(ink_box(frames[i]), ink_box(frames[2 + i]), ("edge-alignment", i))

    def test_side_preserves_vertical_bounds_stacking_and_xy_group_delta(self):
        doc = document()
        for an in range(1, 10):
            # CN above JP at every vertical alignment row.
            texts = [f"{{\\an{an}\\pos(480,{y})}}HH" for y in (220, 340)]
            states = self.snapshots(doc, texts)
            states = [replace(t, margins=(40, 60, 0)) for t in states]
            all_texts = list(texts)
            for side in ("left", "right"):
                changes = preset_side(states, side, (960, 540))
                all_texts.extend(apply_overrides(text, changes[t.member])[0] for t, text in zip(states, texts))
            changes = set_xy(states, states[1], x=510.1, y=360.2)
            all_texts.extend(apply_overrides(text, changes[t.member])[0] for t, text in zip(states, texts))
            frames = self.render(doc, all_texts)
            for i in range(2):
                for offset in (2, 4):
                    self.error(ink_box(frames[i])[[1, 3]], ink_box(frames[offset + i])[[1, 3]], ("side-y", an, i))
                self.error(ink_box(frames[i]) + [30.1, 20.2, 30.1, 20.2], ink_box(frames[6 + i]), ("xy", an, i))
            for offset in (0, 2, 4, 6):
                self.assertLess(ink_box(frames[offset])[3], ink_box(frames[offset + 1])[1])


if __name__ == "__main__":
    unittest.main()
