"""R5: song placeholders, unfinished effects, and unchanged dialogue exports."""
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
from subkit.ass.document import AssDocument, Style
from subkit.ass.text import tokenize
from subkit.core import karaoke as kara
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.karaoke_fx import EventLimit, get_effect
from subkit.model.document import Document, Group, Member, Run
from subkit.proc import run


def fixture(kind="song"):
    doc = Document("R5", groups=[Group("g", kind)])
    for mid, lane, lang, text in (("jp", "original", "ja", "あいう"),
                                  ("cn", "translation", "zh-Hant", "世界")):
        m = Member(mid, "g", lang, lane, "song" if kind == "song" else "dialogue",
                   1003, 3008, style="Default", runs=[Run(text, ins=mid, b=len(text))])
        doc.members[mid] = m
        doc.groups[0].members.append(mid)
    tpl = AssDocument()
    tpl.styles = [Style.new("Default", Fontname="Yu Gothic UI", Fontsize="48", Shadow="0",
                            PrimaryColour="&H00FFFFFF", SecondaryColour="&H000000FF")]
    return doc, tpl


class SongPlaceholders(unittest.TestCase):
    def test_no_tracks_both_lanes_and_release_checks(self):
        doc, tpl = fixture()
        before = copy.deepcopy(doc)
        self.assertFalse(kara.KIndex(doc))
        for m in doc.members.values():
            view = kara.effective(doc, m)
            self.assertEqual(view.state, "placeholder")
            self.assertFalse(view.complete)
            self.assertTrue(all(b.by == "placeholder" for b in view.beats))
        self.assertEqual([(b.t0, b.t1) for b in kara.effective(doc, doc.members["jp"]).beats],
                         [(1003, 1671), (1671, 2339), (2339, 3008)])
        outputs = []
        for mode in ("preview", "release", "work"):
            ass, report = build_ass(doc, {}, tpl, ROOT, ExportOptions(mode=mode, ruby=False))
            self.assertEqual(report.karaoke_lines, 2)
            self.assertEqual(len(report.karaoke_unfinished), 2)
            for ev in ass.events:
                self.assertIn("\\kf", ev.text)
                self.assertEqual(sum(d for _, d in kara.parse_k_runs(ev.text)), ev.end-ev.start)
            outputs.append(ass.serialize())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])
        self.assertEqual(set(dict(kara.unfinished(doc))), {"jp", "cn"})
        self.assertEqual(doc, before)

    def test_exclusions_preserve_source_tags_and_dialogue(self):
        for case in ("comment_role", "comment_source", "off", "tags", "empty", "drawing", "dialogue"):
            with self.subTest(case=case):
                doc, tpl = fixture("dialogue" if case == "dialogue" else "song")
                m = doc.members["jp"]
                if case == "comment_role": m.role = "comment"
                if case == "comment_source": m.source_kind = "Comment"
                if case == "off": m.extra["karaoke_off"] = True
                text = {"tags": "{\\kf20}あ{\\kf30}い", "empty": "{\\bord2}  ",
                        "drawing": "{\\p1}m 0 0 l 20 20"}.get(case)
                if text is not None:
                    m.runs = [Run(t.raw, "tag" if t.kind == "tags" else "text", ins=m.id,
                                  a=t.start, b=t.end) for t in tokenize(text)]
                self.assertEqual(kara.effective(doc, m).state, "none")
                ass, report = build_ass(doc, {}, tpl, ROOT, ExportOptions(mode="work", ruby=False))
                ev = next(e for e, owner in zip(ass.events, report.owners) if owner["member"] == m.id)
                self.assertEqual(ev.text, m.text)
                self.assertNotIn(m.id, dict(kara.unfinished(doc)))

    def test_dialogue_export_matches_pre_r5_bytes(self):
        doc, tpl = fixture("dialogue")
        doc.members["jp"].overrides = {"an": 8, "pos": [320, 50]}
        doc.members["cn"].speaker = {"label": "Alice"}
        ass, _ = build_ass(doc, {}, tpl, ROOT, ExportOptions(ruby=False))
        self.assertEqual(hashlib.sha256(ass.serialize().encode("utf-8")).hexdigest(), "d7d2abaab578b8ba1d47a9071be89a3533867260b1c38b51a247a1315d17ff86")


@unittest.skipUnless(shutil.which("ffmpeg"), "requires host ffmpeg/libass")
class UnfinishedEffects(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="r5-", dir=os.environ["TEMP"])
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def export(self, doc, tpl, mode="release", cache=None):
        return build_ass(doc, {}, tpl, self.path, ExportOptions(mode=mode, ruby=False, playres=(960, 540)),
                         cache=cache, measure_cache=self.path / "measure.sqlite3")

    def track(self, doc, preset, state="placeholder", beats=None, params=None):
        m = doc.members["jp"]
        doc.karaoke["k"] = kara.beats_to_track(m, beats or [], state=state,
                                               effect={"preset": preset, "params": params or {}})

    def test_placeholder_effect_changes_export_and_translation_follows(self):
        doc, tpl = fixture()
        self.track(doc, "particles")
        before = copy.deepcopy(doc)
        cache = ExportCache()
        ass, report = self.export(doc, tpl, "preview", cache)
        self.assertEqual(report.karaoke_fx_lines, 2)
        for mid in ("jp", "cn"):
            events = [e for e, o in zip(ass.events, report.owners) if o["member"] == mid]
            self.assertTrue(any("\\p1" in e.text for e in events))
        self.assertEqual(len(report.karaoke_unfinished), 2)
        same, cached = self.export(doc, tpl, "preview", cache)
        self.assertEqual(cached.laid_out, 0)
        self.assertEqual(same.serialize(), ass.serialize())
        release, _ = self.export(doc, tpl)
        self.assertEqual(release.serialize(), ass.serialize())
        self.assertEqual(doc, before)
        self.track(doc, "glow")
        changed, _ = self.export(doc, tpl, "preview", cache)
        self.assertNotEqual(changed.serialize(), same.serialize())
        self.assertTrue(any("\\blur" in e.text for e in changed.events))

    def test_follow_switch_and_translation_own_effect(self):
        doc, tpl = fixture()
        self.track(doc, "particles", params={"follow_translation": False})
        ass, report = self.export(doc, tpl)
        self.assertEqual(report.karaoke_fx_lines, 1)
        cn = next(e for e, o in zip(ass.events, report.owners) if o["member"] == "cn")
        self.assertIn("\\kf", cn.text)
        doc.karaoke["cn"] = kara.beats_to_track(doc.members["cn"], [], effect={"preset": "glow"})
        self.track(doc, "particles")
        ass, report = self.export(doc, tpl)
        cn = [e for e, o in zip(ass.events, report.owners) if o["member"] == "cn"]
        self.assertTrue(any("\\blur" in e.text for e in cn))
        self.assertFalse(any("\\p1" in e.text for e in cn))

    def test_partial_and_suggested_beats_keep_times_and_unfinished_status(self):
        for state in ("partial", "suggested"):
            with self.subTest(state=state):
                doc, tpl = fixture()
                beats = kara.derive(doc, doc.members["jp"])
                for i, b in enumerate(beats):
                    b.t0, b.t1, b.by = 1100+i*500, 1600+i*500, "tap" if state == "partial" else "onset"
                self.track(doc, "pop", "timed" if state == "partial" else "suggested",
                           beats[:1] if state == "partial" else beats)
                before = copy.deepcopy(doc)
                ass, report = self.export(doc, tpl)
                self.assertEqual(report.karaoke_fx_lines, 2)
                self.assertTrue(any("\\fscx150" in e.text for e in ass.events))
                self.assertEqual((kara.effective(doc, doc.members["jp"]).beats[0].t0,
                                  kara.effective(doc, doc.members["jp"]).beats[0].t1), (1100, 1600))
                self.assertEqual(len(report.karaoke_unfinished), 2)
                self.assertEqual(set(dict(kara.unfinished(doc))), {"jp", "cn"})
                self.assertEqual(doc, before)

    def test_placeholder_event_limit_falls_back_to_identical_sweep(self):
        doc, tpl = fixture()
        self.track(doc, "particles")
        with patch.object(get_effect("particles"), "generate", side_effect=EventLimit):
            fallback, report = self.export(doc, tpl)
        self.track(doc, "kf")
        normal, _ = self.export(doc, tpl)
        self.assertEqual(fallback.serialize(), normal.serialize())
        self.assertEqual(report.karaoke_fx_lines, 0)
        self.assertEqual(sum("已改用渐变扫过" in w for w in report.warnings), 2)

    def test_libass_placeholder_preset_changes_visible_frame(self):
        doc, tpl = fixture()
        frames = []
        for name in ("kf", "particles"):
            self.track(doc, name)
            ass, _ = self.export(doc, tpl)
            (self.path / "frame.ass").write_text(ass.serialize(), encoding="utf-8-sig")
            proc = run(["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                        "color=c=black:s=960x540:r=100:d=0.01", "-vf",
                        "setpts=PTS+1.2/TB,ass=frame.ass,format=rgb24", "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], cwd=self.path,
                       capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
            frame = np.frombuffer(proc.stdout, dtype=np.uint8)
            self.assertTrue(frame.any())
            frames.append(frame)
            print(next((s for s in proc.stderr.decode("utf-8", "replace").splitlines()
                        if "libass API" in s), ""))
        changed = np.count_nonzero(frames[0] != frames[1])
        self.assertGreater(changed, 100)
        print(f"R5 libass placeholder kf/particles at 1.2 s: {changed} RGB components differ")
