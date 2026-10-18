"""Portable passage snapshots, suggestions and single-transaction application."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.audio_match import Match, Template
from subkit.core import karaoke as kara
from subkit.core.merge import MergeSettings
from subkit.core.ops import Editor
from subkit.core.passages import PassageStore, already_applied, plan, snapshot
from subkit.core.readings import anchor_of, visible_map
from subkit.core.workspace import Project
from subkit.model.document import Annotation, Document, Group, Member, Run
from subkit.model.serialize import document_to_json
from subkit.model.source import Source, SourceEvent
from subkit.model.validate import validate


def member(doc, gid, mid, text, *, start=1000, end=5000, lane="original", layer=0, role="song", style="OP.JP"):
    m = Member(mid, gid, "ja" if lane == "original" else "zh-Hant", lane, role, start, end,
               layer=layer, style=style, runs=[Run(text, ins="i_" + mid, a=0, b=len(text))])
    doc.members[mid] = m
    next(g for g in doc.groups if g.id == gid).members.append(mid)
    return m


def annotation(doc, m, a, b, text, status="confirmed", kind="ruby"):
    vt, ids = visible_map(m)
    aid = f"a_{len(doc.annotations)}"
    ann = Annotation(aid, kind, m.id, anchor_of(ids[a:b]), vt.text[a:b], text,
                     status=status, parts=[[0, b - a, text]] if text else [])
    doc.annotations[aid] = ann
    return ann


def recorded(doc, gids=None, category="OP"):
    return snapshot(doc, gids or [g.id for g in doc.groups], name="段落", category=category, version="TV", episode_id="E1")


def match(start=10000, covered=(0, 4000), scale=1.0):
    return Match(start, .95, covered, [], 40, scale)


def actions(line):
    return {a.kind: a for a in line.actions}


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document("E1", groups=[Group("g_1", "song")])

    def test_tags_breaks_and_identity_independent_readings_format(self):
        m = member(self.doc, "g_1", "m_1", "")
        m.runs = [Run(r"{\an8}", "tag", ins="i_tag", b=6), Run(r"学\N校", ins="i_text", b=4)]
        annotation(self.doc, m, 0, 2, "がっこう")
        m.overrides = {"pos": [100, 200]}
        m.format = [{"anchor": anchor_of(visible_map(m)[1][1:]), "props": {"fs": 50}}]
        before = copy.deepcopy(document_to_json(self.doc))
        p = recorded(self.doc)
        s = p["groups"][0]["members"][0]
        self.assertEqual((s["text"], s["visible"], s["inline_tags"]), (r"学\N校", "学校", r"{\an8}"))
        self.assertEqual((s["annotations"][0]["a"], s["annotations"][0]["b"]), (0, 2))
        self.assertEqual(s["format"], [{"a": 1, "b": 2, "props": {"fs": 50}}])
        for r in m.runs:
            r.ins = "new_" + r.ins
        self.doc.annotations.clear()
        annotation(self.doc, m, 0, 2, "がっこう")
        m.format[0]["anchor"] = anchor_of(visible_map(m)[1][1:])
        s2 = recorded(self.doc)["groups"][0]["members"][0]
        self.assertEqual(s, s2)
        s["overrides"]["pos"][0] = 999
        self.assertEqual(m.overrides["pos"][0], 100)
        self.assertEqual(before["members"][m.id]["overrides"], m.overrides)

    def test_only_confirmed_hidden_and_summary(self):
        m = member(self.doc, "g_1", "m_1", "学学校校")
        for i, status in enumerate(("confirmed", "hidden", "auto", "uncertain")):
            annotation(self.doc, m, i, i + 1, "がく", status)
        s = recorded(self.doc)
        self.assertEqual([a["status"] for a in s["groups"][0]["members"][0]["annotations"]], ["confirmed", "hidden"])
        self.assertEqual((s["summary"]["readings"], s["summary"]["readings_skipped"], s["summary"]["unconfirmed_lines"]), (2, 2, 1))
        # A group the user marked confirmed was reviewed with its furigana: every reading is recorded as confirmed.
        self.doc.group("g_1").state = "confirmed"
        s = recorded(self.doc)
        self.assertEqual([a["status"] for a in s["groups"][0]["members"][0]["annotations"]],
                         ["confirmed", "hidden", "confirmed", "confirmed"])
        self.assertEqual((s["summary"]["readings"], s["summary"]["readings_skipped"], s["summary"]["unconfirmed_lines"]), (4, 0, 0))
        self.assertEqual([a.status for a in self.doc.annotations.values()], ["confirmed", "hidden", "auto", "uncertain"])

    def test_ignores_comments_empty_groups_and_preserves_document_order(self):
        m = member(self.doc, "g_1", "m_1", "あ", start=1500, end=4500)
        member(self.doc, "g_1", "m_comment", "注释", start=0, end=10000, role="comment")
        member(self.doc, "g_1", "m_empty", " ")
        self.doc.groups.append(Group("g_2", "dialogue"))
        member(self.doc, "g_2", "m_2", "い", start=500, end=3500)
        self.doc.groups.append(Group("g_empty"))
        p = recorded(self.doc, ["g_empty", "g_2", "g_1", "g_1"])
        self.assertTrue(p["id"].startswith("psg_"))
        self.assertEqual((p["start"], p["duration"]), (500, 4000))
        self.assertEqual([g["members"][0]["text"] for g in p["groups"]], ["あ", "い"])
        self.assertEqual(p["groups"][0]["members"][0]["start"], 1000)

    def test_empty_and_invalid_category(self):
        with self.assertRaises(ValueError):
            recorded(self.doc)
        member(self.doc, "g_1", "m_1", "あ")
        with self.assertRaises(ValueError):
            recorded(self.doc, category="BAD")
        self.assertEqual(recorded(self.doc, category="NA")["category"], "NA")

    def test_placeholder_and_suggested_beats_excluded(self):
        m = member(self.doc, "g_1", "m_1", "あいう")
        beats = kara.derive(self.doc, m)
        beats[0].t0, beats[0].t1, beats[0].by = 1100, 1500, "tap"
        tr = kara.beats_to_track(m, beats)
        self.doc.karaoke[tr["id"]] = tr
        p = recorded(self.doc)
        k = p["groups"][0]["members"][0]["karaoke"]
        self.assertEqual(len(k["beats"]), 1)
        self.assertEqual((k["beats"][0]["t0"], k["beats"][0]["t1"]), (100, 500))
        self.assertEqual(p["summary"]["karaoke"], {"none": 0, "done": 0, "partial": 1})
        for i, beat in enumerate(beats):
            beat.t0, beat.t1, beat.by = 1100 + i * 400, 1500 + i * 400, "onset"
        tr = kara.beats_to_track(m, beats, state="suggested")
        self.doc.karaoke = {tr["id"]: tr}
        # Auto K timing not yet confirmed is recorded with its state (it used to be dropped: untimed lyrics later).
        p = recorded(self.doc)
        k = p["groups"][0]["members"][0]["karaoke"]
        self.assertEqual((k["state"], len(k["beats"]), k["beats"][0]["t0"]), ("suggested", 3, 100))
        self.assertEqual(p["summary"]["karaoke"], {"none": 0, "done": 1, "partial": 0})

    def test_edits_links_and_beats_remap_to_new_identities(self):
        original = member(self.doc, "g_1", "m_o", "ガー")
        translated = member(self.doc, "g_1", "m_t", "天空", lane="translation")
        edit = {"op": "split", "at": kara.boundary_key(self.doc, original, 1)}
        ob = kara.derive(self.doc, original, edits=[edit])
        tb = kara.derive(self.doc, translated)
        for i, beat in enumerate(ob + tb):
            beat.t0, beat.t1, beat.by = 1100 + i * 100, 1200 + i * 100, "manual"
        tb[0].link = {"end": kara.beat_link(ob[1], "start")}
        for m, beats, edits in ((original, ob, [edit]), (translated, tb, [])):
            tr = kara.beats_to_track(m, beats, edits=edits, effect={"preset": "ko", "params": {"x": 1}})
            self.doc.karaoke[tr["id"]] = tr
        p = recorded(self.doc)
        ko, kt = [m["karaoke"] for m in p["groups"][0]["members"]]
        self.assertEqual(ko["edits"][0]["at"]["core"], [[1, 2]])
        self.assertEqual(kt["beats"][0]["link"]["end"]["core"], [[1, 2]])
        self.assertEqual(kt["beats"][0]["link"]["end"]["member_index"], 0)
        other = Document("E2", groups=[Group("g_new", "song")])
        dst = member(other, "g_new", "m_new", "ガー")
        ids = visible_map(dst)[1]
        new_edit = copy.deepcopy(ko["edits"])
        for e in new_edit:
            e["at"]["core"] = anchor_of([ident for a, b in e["at"]["core"] for ident in ids[a:b]])
        derived = kara.derive(other, dst, ko["unit"], new_edit)
        self.assertEqual([b.sung for b in derived], [b["sung"] for b in ko["beats"]])
        for beat, saved in zip(derived, ko["beats"]):
            self.assertEqual(list(beat.core), [ident for a, b in saved["core"] for ident in ids[a:b]])
        self.assertEqual(p["summary"]["karaoke"]["done"], 2)
        self.assertNotIn("i_m_o", json.dumps(p, ensure_ascii=False))


class StoreTests(unittest.TestCase):
    def test_add_rename_replace_remove_utf8_and_custom_path(self):
        for custom in (False, True):
            with self.subTest(custom=custom), tempfile.TemporaryDirectory(dir=os.environ["TEMP"]) as scratch:
                root = Path(scratch)
                data = {"dictionaries": {"passages": "library/saved.json"} if custom else {}}
                project = Project(root, data)
                manifest = root / "subkit-project.json"
                manifest.write_text(json.dumps(data), encoding="utf-8")
                before = manifest.read_bytes()
                doc = Document("E1", groups=[Group("g_1", "song")])
                member(doc, "g_1", "m_1", "学校")
                p = recorded(doc)
                store = PassageStore(project)
                self.assertEqual(store.list(), [])
                self.assertFalse(store.path.exists())
                t = Template(np.arange(300 * 128, dtype=np.uint8).reshape(300, 128), np.zeros(300, np.float32))
                id = store.add(p, t)
                store = PassageStore(project)
                self.assertEqual(store.get(id), p)
                self.assertTrue(store.path.read_bytes().startswith(b'{'))
                self.assertIn("段落".encode("utf-8"), store.path.read_bytes())
                np.testing.assert_array_equal(store.template(id).spectrogram, t.spectrogram)
                store.rename(id, "新版", "完整版")
                self.assertEqual(store.get(id)["name"], "新版")
                replacement = copy.deepcopy(p)
                replacement["id"] = "psg_new"
                replacement["name"] = "替换"
                t.spectrogram[:] = 99
                self.assertEqual(store.add(replacement, t, replace_id=id), id)
                self.assertEqual(len(store.list()), 1)
                self.assertEqual(store.get(id)["name"], "替换")
                np.testing.assert_array_equal(store.template(id).spectrogram, t.spectrogram)
                self.assertFalse((root / "dict/passages/psg_new.npz").exists())
                detached = store.get(id)
                detached["name"] = "wrong"
                self.assertEqual(store.get(id)["name"], "替换")
                store.remove(id)
                self.assertEqual(store.list(), [])
                self.assertFalse((root / "dict/passages" / f"{id}.npz").exists())
                self.assertEqual(manifest.read_bytes(), before)
                self.assertEqual(project.data, data)
                with self.assertRaises(KeyError):
                    store.get(id)
                with self.assertRaises(ValueError):
                    store.add({"id": "psg_../escape"}, t)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.source = Document("E1", groups=[Group("g_s", "song")])
        self.sm = member(self.source, "g_s", "m_s", "学校")
        annotation(self.source, self.sm, 0, 2, "がっこう")
        self.sm.overrides = {"pos": [100, 200]}
        self.sm.format = [{"anchor": anchor_of(visible_map(self.sm)[1]), "props": {"fs": 48}}]
        beats = kara.derive(self.source, self.sm)
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1100 + i * 500, 1600 + i * 500, "tap"
        track = kara.beats_to_track(self.sm, beats)
        self.source.karaoke[track["id"]] = track
        self.p = recorded(self.source)
        self.doc = Document("E2", groups=[Group("g_t", "dialogue")])
        self.tm = member(self.doc, "g_t", "m_t", "学校", start=10100, end=14100, role="dialogue", style="JP")

    def test_exact_actions_portable_and_pure(self):
        before = copy.deepcopy(document_to_json(self.doc))
        stored = copy.deepcopy(self.p)
        result = plan(self.doc, self.p, match())
        line = result.lines[0]
        self.assertEqual(line.status, "matched")
        a = actions(line)
        self.assertEqual(set(a), {"time", "readings", "karaoke", "convert", "position"})
        self.assertTrue(all(x.checked for x in a.values()))
        self.assertEqual((a["time"].data["start"], a["time"].data["end"]), (10000, 14000))
        self.assertEqual(a["karaoke"].data["beats"][0]["t0"], 10100)
        self.assertTrue(line.readings_differ)
        self.assertTrue(all(x.target_member_id == self.tm.id and x.target_group_id == "g_t" for x in a.values()))
        self.assertEqual(document_to_json(self.doc), before)
        self.assertEqual(self.p, stored)
        self.assertEqual([x.id for x in line.actions], [x.id for x in plan(self.doc, self.p, match()).lines[0].actions])
        a["position"].data["overrides"]["pos"][0] = 999
        self.assertEqual(self.p, stored)

    def test_human_reading_beats_position_conflicts_unchecked(self):
        annotation(self.doc, self.tm, 0, 2, "がくこう", "hidden")
        b = kara.derive(self.doc, self.tm)
        for bt in b:
            bt.t0, bt.t1, bt.by = 10200, 10500, "drag"
        tr = kara.beats_to_track(self.tm, b)
        self.doc.karaoke[tr["id"]] = tr
        self.tm.overrides = {"pos": [300, 400]}
        result = plan(self.doc, self.p, match())
        a = actions(result.lines[0])
        for kind in ("readings_conflict", "karaoke", "position"):
            self.assertFalse(a[kind].checked)
            self.assertTrue(a[kind].conflict)
        self.assertNotIn("readings", a)

    def test_safe_and_conflicting_readings_separate_actions(self):
        self.source.annotations.clear()
        annotation(self.source, self.sm, 0, 1, "がく")
        annotation(self.source, self.sm, 1, 2, "こう", "hidden")
        self.p = recorded(self.source)
        annotation(self.doc, self.tm, 0, 1, "まな", "confirmed")
        a = actions(plan(self.doc, self.p, match()).lines[0])
        self.assertTrue(a["readings"].checked)
        self.assertEqual(a["readings"].data["annotations"][0]["a"], 1)
        self.assertFalse(a["readings_conflict"].checked)

    def test_identical_confirmed_readings_not_offered_and_no_time_within_tolerance(self):
        annotation(self.doc, self.tm, 0, 2, "がっこう")
        self.tm.start, self.tm.end = 10010, 13990
        result = plan(self.doc, self.p, match())
        a = actions(result.lines[0])
        self.assertNotIn("readings", a)
        self.assertNotIn("readings_conflict", a)
        self.assertNotIn("time", a)
        self.assertFalse(result.lines[0].readings_differ)
        self.tm.end = 13989
        self.assertIn("time", actions(plan(self.doc, self.p, match()).lines[0]))

    def test_text_differs_only_time_unchecked(self):
        self.tm.runs = [Run("別の文", ins="i_different", b=3)]
        result = plan(self.doc, self.p, match())
        self.assertEqual(result.lines[0].status, "text_differs")
        self.assertEqual(set(actions(result.lines[0])), {"time"})
        self.assertFalse(result.lines[0].actions[0].checked)
        self.assertEqual(result.lines[0].notes, ["文字和库里不同：库里「学校」"])

    def test_nearest_lane_use_once_and_glow_layer_preference(self):
        twin = member(self.doc, "g_t", "m_twin", "学校", start=10110, end=14110, layer=1)
        self.p["groups"][0]["members"][0]["layer"] = 1
        self.assertEqual(plan(self.doc, self.p, match()).lines[0].target_member_id, twin.id)
        twin.start, twin.end = 12000, 16000
        self.assertEqual(plan(self.doc, self.p, match()).lines[0].target_member_id, self.tm.id)
        self.p["groups"][0]["members"].append(copy.deepcopy(self.p["groups"][0]["members"][0]))
        result = plan(self.doc, self.p, match())
        self.assertEqual(len({l.target_member_id for l in result.lines}), 2)
        self.tm.lane = twin.lane = "translation"
        self.assertTrue(all(l.status == "missing_group" for l in plan(self.doc, self.p, match()).lines))

    def test_group_members_prefer_one_target_group_over_identical_caption(self):
        # Stored pair JP+CN; the target has a JP CC caption of the same lyric (closer in time, its own group) and
        # the fansub's JP+CN pair.  The original goes with its translation, the caption is left alone.
        cn = member(self.source, "g_s", "m_cn", "學校", lane="translation", style="OP.CN")
        p = recorded(self.source)
        doc = Document("E2", groups=[Group("g_cc", "dialogue"), Group("g_f", "song")])
        member(doc, "g_cc", "m_cc", "学校", start=10000, end=14000, role="dialogue", style="JP")
        member(doc, "g_f", "m_fj", "学校", start=10300, end=14300)
        member(doc, "g_f", "m_fc", "學校", start=10300, end=14300, lane="translation", style="OP.CN")
        result = plan(doc, p, match())
        self.assertEqual([(l.target_member_id, l.target_group_id) for l in result.lines],
                         [("m_fj", "g_f"), ("m_fc", "g_f")])
        self.assertFalse(any(a.kind == "convert" and a.target_group_id == "g_cc" for l in result.lines for a in l.actions))
        del doc.members["m_fc"]
        doc.groups[1].members.remove("m_fc")                       # no pair to join: nearest caption as before
        self.assertEqual(plan(doc, p, match()).lines[0].target_member_id, "m_cc")
        self.assertIsNotNone(cn)

    def test_missing_member_matched_later_sibling_and_missing_group(self):
        self.p["groups"][0]["members"].insert(0, {**copy.deepcopy(self.p["groups"][0]["members"][0]), "lane": "translation", "text": "天空", "visible": "天空"})
        result = plan(self.doc, self.p, match())
        self.assertEqual(result.lines[0].status, "missing_member")
        insert = result.lines[0].actions[0]
        self.assertEqual((insert.kind, insert.target_group_id), ("insert_member", "g_t"))
        self.assertEqual(insert.data["target_members"], {1: self.tm.id})
        self.doc = Document("E2")
        result = plan(self.doc, self.p, match())
        self.assertTrue(all(l.status == "missing_group" for l in result.lines))
        self.assertEqual(result.lines[0].actions[0].id, result.lines[1].actions[0].id)
        self.assertEqual(len(result.lines[0].actions[0].data["members"]), 2)

    def test_partial_play_unknown_is_opt_in_and_extra_lines_only_info(self):
        member(self.source, "g_s", "m_later", "あ", start=5500, end=7000)
        self.p = recorded(self.source)
        member(self.doc, "g_t", "m_extra", "別", start=13000, end=13500, lane="translation")
        result = plan(self.doc, self.p, match(covered=(0, 4000)))
        self.assertEqual(result.lines[1].coverage, "unknown")
        self.assertFalse(result.lines[1].actions[0].checked)
        self.assertEqual(result.info, ["这一段里还有 1 行库里没有，不会改动。", "覆盖未知：1 行默认不勾选，可手动勾选。"])
        # A blank line in range is not counted and must not break the count (E005: int + str TypeError).
        member(self.doc, "g_t", "m_blank", "", start=12000, end=12500, lane="translation")
        result = plan(self.doc, self.p, match(covered=(0, 4000)))
        self.assertEqual(result.info[0], "这一段里还有 1 行库里没有，不会改动。")
        self.doc = Document("E2")
        result = plan(self.doc, self.p, match(covered=(0, 4000)))
        self.assertEqual(len(result.lines[0].actions[0].data["members"]), 1)

    def test_coverage_boundary_and_match_start_window_boundaries(self):
        self.doc = Document("E2")
        self.assertEqual(plan(self.doc, self.p, match(covered=(0, 4000))).lines[0].coverage, "direct")
        for bounds in ((1, 4000), (0, 3999), (500, 3500)):
            line = plan(self.doc, self.p, match(covered=bounds)).lines[0]
            self.assertEqual(line.coverage, "unknown")
            self.assertFalse(line.actions[0].checked)
        self.doc = Document("E2", groups=[Group("g_t", "dialogue")], members={self.tm.id: self.tm})
        self.doc.groups[0].members.append(self.tm.id)
        self.tm.start, self.tm.end = 13000, 17000
        self.assertEqual(plan(self.doc, self.p, match()).lines[0].status, "matched")
        self.tm.start, self.tm.end = 13001, 17001
        self.assertEqual(plan(self.doc, self.p, match()).lines[0].status, "missing_group")

    def test_opt_in_scaling_anchor_and_beats_round_ties_even(self):
        plain = plan(self.doc, self.p, match(scale=1.01))
        scaled = plan(self.doc, self.p, match(scale=1.01), scale=True)
        self.assertEqual((plain.lines[0].start, plain.lines[0].end), (10000, 14000))
        self.assertEqual((scaled.lines[0].start, scaled.lines[0].end), (10000, 14040))
        self.assertEqual(actions(scaled.lines[0])["karaoke"].data["beats"][0]["t0"], 10101)
        self.p["groups"][0]["members"][0]["end"] = 4001
        self.assertEqual(plan(self.doc, self.p, match(scale=1.5), scale=True).lines[0].end, 16002)

    def test_style_after_conversion_and_template_filter(self):
        self.p["groups"][0]["members"][0]["style"] = "Custom"
        a = actions(plan(self.doc, self.p, match(), template_styles=["Custom"]).lines[0])
        self.assertEqual(a["style"].label, "样式 OP.JP → Custom")
        self.assertNotIn("style", actions(plan(self.doc, self.p, match(), template_styles=[]).lines[0]))
        self.doc.groups[0].kind = "song"
        self.assertNotIn("convert", actions(plan(self.doc, self.p, match()).lines[0]))

    def test_exact_and_text_differs_outside_coverage_still_match(self):
        result = plan(self.doc, self.p, match(covered=(10000, 20000)))
        self.assertEqual(result.lines[0].status, "matched")
        self.tm.runs = [Run("別の文", ins="i_different", b=3)]
        result = plan(self.doc, self.p, match(covered=(10000, 20000)))
        self.assertEqual(result.lines[0].status, "text_differs")
        self.assertEqual(set(actions(result.lines[0])), {"time"})

    def test_project_style_mapping_matches_conversion(self):
        styles = {**MergeSettings().styles, "original/song": "{fam}.ProjectJP"}
        self.p["groups"][0]["members"][0]["style"] = "OP.ProjectJP"
        result = plan(self.doc, self.p, match(), styles=styles)
        self.assertNotIn("style", actions(result.lines[0]))
        Editor(self.doc, {}).apply_passage(result, [actions(result.lines[0])["convert"].id])
        self.assertEqual(self.tm.style, "OP.ProjectJP")
        self.p["groups"][0]["members"][0]["style"] = "Custom"
        self.doc.groups[0].kind, self.tm.role = "dialogue", "dialogue"
        self.tm.extra.clear()
        result = plan(self.doc, self.p, match(), styles=styles)
        self.assertEqual(actions(result.lines[0])["style"].label, "样式 OP.ProjectJP → Custom")

    def test_applied_markers_are_group_local_and_tolerance(self):
        self.doc.extra["passage"] = {"id": self.p["id"], "at": 10000}
        self.assertFalse(already_applied(self.doc, self.p["id"], 10000))
        self.doc.groups[0].extra["passage"] = {"id": self.p["id"], "at": 12000}
        self.assertTrue(plan(self.doc, self.p, match()).already_applied)
        self.assertFalse(already_applied(self.doc, self.p["id"], 9999))


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.source = Document("E1", groups=[Group("g_s", "song")])
        self.sm = member(self.source, "g_s", "m_s", "学校")
        annotation(self.source, self.sm, 0, 2, "がっこう")
        self.sm.overrides = {"pos": [120, 230]}
        self.sm.format = [{"anchor": anchor_of(visible_map(self.sm)[1][1:]), "props": {"fs": 42}}]
        self.timed(self.source, self.sm)
        self.p = recorded(self.source)
        self.doc = Document("E2", groups=[Group("g_t", "dialogue")])
        self.tm = member(self.doc, "g_t", "m_t", "学校", start=10100, end=14100, role="dialogue", style="JP")
        self.ed = Editor(self.doc, {})

    def timed(self, doc, m, by="tap", edits=None):
        beats = kara.derive(doc, m, edits=edits)
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = m.start + 100 + i * 500, m.start + 600 + i * 500, by
        track = kara.beats_to_track(m, beats, edits=edits, effect={"preset": "ko", "params": {"x": 1}})
        doc.karaoke[track["id"]] = track
        return track

    def serialized(self):
        data = copy.deepcopy(document_to_json(self.doc))
        data.pop("revision")  # History deliberately advances this on do/undo/redo.
        return json.dumps(data, ensure_ascii=False, sort_keys=True)

    def planned(self, **kwargs):
        return plan(self.doc, self.p, match(scale=1.01), **kwargs)

    def chosen(self, result, kinds=None):
        return {a.id for line in result.lines for a in line.actions
                if a.checked and (kinds is None or a.kind in kinds)}

    def roundtrip(self, result, chosen):
        before = self.serialized()
        summary = self.ed.apply_passage(result, chosen)
        after = self.serialized()
        self.assertEqual(len(self.ed.history.done), 1)
        self.ed.history.undo()
        self.assertEqual(self.serialized(), before)
        self.ed.history.redo()
        self.assertEqual(self.serialized(), after)
        self.assertTrue(validate(self.doc, self.ed.sources).ok)
        return summary

    def test_apply_undo_redo_readings_identities_format_memory_and_marker(self):
        result = self.planned()
        summary = self.roundtrip(result, self.chosen(result))
        self.assertEqual((summary.time_lines, summary.readings, summary.karaoke_lines, summary.inserted_lines), (1, 1, 1, 0))
        m = self.doc.members[self.tm.id]
        self.assertEqual((m.start, m.end, m.role, m.style), (10000, 14000, "song", "OP.JP"))
        ann = self.doc.annotations_of(m.id)[0]
        self.assertEqual((ann.origin, ann.status, ann.parts), ("passage", "confirmed", [[0, 2, "がっこう"]]))
        self.assertEqual(ann.anchor, anchor_of(visible_map(m)[1]))
        track = kara.tracks_of(self.doc, m.id)[0]
        self.assertEqual(track["state"], "timed")
        self.assertTrue(all(b["by"] == "memory" for b in track["beats"]))
        self.assertNotIn("i_m_s", json.dumps(track))
        self.assertEqual(m.format[0]["anchor"], anchor_of(visible_map(m)[1][1:]))
        self.assertEqual(self.doc.group("g_t").extra["passage"]["id"], result.id)
        self.assertTrue(self.ed.history.done[0].keeps_text)
        self.assertEqual(self.ed.history.done[0].label, "套用重复段落《段落》")

    def test_unchosen_actions_inert_including_revision_and_history(self):
        before = document_to_json(self.doc)
        summary = self.ed.apply_passage(self.planned(), [])
        self.assertEqual(document_to_json(self.doc), before)
        self.assertEqual(summary.readings, 0)
        self.assertFalse(self.ed.history.done)
        result = self.planned()
        self.roundtrip(result, self.chosen(result, {"time"}))
        self.assertEqual(self.doc.members[self.tm.id].role, "dialogue")
        self.assertFalse(self.doc.annotations)
        self.assertFalse(self.doc.karaoke)
        self.assertEqual(self.doc.members[self.tm.id].overrides, {})

    def test_conflicts_preserved_by_default_and_explicitly_replaceable(self):
        old = annotation(self.doc, self.tm, 0, 2, "がくこう", "hidden")
        track = self.timed(self.doc, self.tm)
        self.tm.overrides = {"pos": [500, 600]}
        result = self.planned()
        summary = self.ed.apply_passage(result, self.chosen(result))
        self.assertEqual(self.doc.annotations[old.id], old)
        self.assertEqual(self.doc.members[self.tm.id].overrides, {"pos": [500, 600]})
        self.assertTrue(all(b["by"] == "tap" for b in self.doc.karaoke[track["id"]]["beats"]))
        self.assertEqual(summary.karaoke_lines, 0)
        self.ed.history.undo()
        self.ed.history.done.clear()
        summary = self.roundtrip(result, {a.id for l in result.lines for a in l.actions})
        self.assertEqual((summary.readings, summary.karaoke_lines), (1, 1))
        self.assertNotIn(old.id, self.doc.annotations)

    def test_suggested_auto_timing_applies_and_stays_unconfirmed(self):
        track = next(iter(self.source.karaoke.values()))
        track["state"] = "suggested"
        for b in track["beats"]:
            b["by"] = "align"
        self.p = recorded(self.source)
        result = self.planned()
        summary = self.ed.apply_passage(result, [x.id for x in result.lines[0].actions])
        self.assertEqual((summary.karaoke_lines, summary.skipped), (1, []))
        view = kara.effective(self.doc, self.tm)
        self.assertEqual(view.state, "suggested")
        self.assertTrue(view.beats and all(b.timed for b in view.beats))

    def test_skip_reason_reported_once_per_line(self):
        from subkit.core.passages import ApplySummary
        summary = ApplySummary()
        for _ in range(18):
            summary.skip("p:2:0:karaoke", "karaoke link target is absent from the target group")
        summary.skip("p:2:0:karaoke", "sung sequence differs after applying readings")
        self.assertEqual(len(summary.skipped), 2)

    def test_karaoke_mismatch_skips_and_time_still_shifts_existing_track(self):
        track = self.timed(self.doc, self.tm, by="memory")
        old = copy.deepcopy(track)
        result = self.planned()
        summary = self.roundtrip(result, self.chosen(result, {"time", "karaoke"}))
        self.assertEqual(summary.karaoke_lines, 0)
        self.assertIn("sung sequence", summary.skipped[0]["reason"])
        current = self.doc.karaoke[track["id"]]
        self.assertEqual([b["t0"] for b in current["beats"]], [b["t0"] - 100 for b in old["beats"]])
        self.assertEqual([b["by"] for b in current["beats"]], [b["by"] for b in old["beats"]])

    def test_complete_recording_rejects_extra_syllable_after_same_prefix(self):
        annotation(self.doc, self.tm, 0, 2, "がっこうう")
        result = self.planned()
        summary = self.roundtrip(result, self.chosen(result, {"karaoke"}))
        self.assertEqual(summary.karaoke_lines, 0)
        self.assertFalse(self.doc.karaoke)
        self.assertIn("sung sequence", summary.skipped[0]["reason"])

    def test_readings_remove_auto_overlaps_preserve_other_annotations(self):
        auto = annotation(self.doc, self.tm, 0, 2, "がくこう", "auto")
        gloss = annotation(self.doc, self.tm, 0, 1, "school", "confirmed", "gloss")
        result = self.planned()
        summary = self.roundtrip(result, self.chosen(result, {"readings"}))
        self.assertEqual(summary.readings, 1)
        self.assertNotIn(auto.id, self.doc.annotations)
        self.assertEqual(self.doc.annotations[gloss.id], gloss)

    def test_plan_staleness_protects_new_manual_readings_karaoke_position(self):
        result = self.planned()
        old = annotation(self.doc, self.tm, 0, 2, "がくこう", "confirmed")
        track = self.timed(self.doc, self.tm)
        self.tm.overrides = {"pos": [500, 600]}
        summary = self.roundtrip(result, self.chosen(result))
        self.assertEqual(self.doc.annotations[old.id], old)
        self.assertEqual(self.doc.karaoke[track["id"]]["beats"][0]["by"], "tap")
        self.assertEqual(self.doc.members[self.tm.id].overrides, {"pos": [500, 600]})
        self.assertEqual(len(summary.skipped), 3)

    def test_scaling_consistent_times_and_beats(self):
        result = self.planned(scale=True)
        summary = self.roundtrip(result, self.chosen(result))
        self.assertEqual(summary.time_lines, 1)
        self.assertEqual(self.doc.members[self.tm.id].end, 14040)
        track = kara.tracks_of(self.doc, self.tm.id)[0]
        self.assertEqual(track["beats"][0]["t0"], 10101)
        self.assertEqual(track["beats"][0]["t1"], 10606)

    def test_convert_once_for_bilingual_group_preserves_before_convert(self):
        member(self.source, "g_s", "m_st", "天空", lane="translation", style="OP.CN")
        tm = member(self.doc, "g_t", "m_tt", "天空", lane="translation", start=10100, end=14100,
                    role="top", style="CN.Top")
        self.p = recorded(self.source)
        result = self.planned()
        with patch.object(self.ed, "_convert_transaction", wraps=self.ed._convert_transaction) as convert:
            self.roundtrip(result, self.chosen(result, {"convert"}))
            self.assertEqual(convert.call_count, 1)
        self.assertEqual(self.doc.group("g_t").extra["before_convert"], {"kind": "dialogue"})
        self.assertEqual(self.doc.members[tm.id].extra["before_convert"], {"role": "top", "style": "CN.Top"})

    def test_insert_group_once_tags_new_identities_order_and_scaling(self):
        member(self.source, "g_s", "m_st", "天空", lane="translation", style="OP.CN")
        self.sm.runs.insert(0, Run(r"{\an8}", "tag", ins="i_tag", b=6))
        self.p = recorded(self.source)
        self.doc.members[self.tm.id].start, self.doc.members[self.tm.id].end = 50000, 54000
        result = self.planned(scale=True)
        summary = self.roundtrip(result, self.chosen(result))
        self.assertEqual(summary.inserted_lines, 2)
        self.assertEqual(len(self.doc.groups), 2)
        g = self.doc.groups[0]
        self.assertEqual((g.kind, g.state, g.origin), ("song", "auto", {"by": "passage", "passage": self.p["id"]}))
        ms = self.doc.members_of(g)
        self.assertEqual([m.lane for m in ms], ["translation", "original"])
        inserted = ms[1]
        self.assertTrue(inserted.id.startswith("m_"))
        self.assertEqual(inserted.runs[0].kind, "tag")
        self.assertEqual(inserted.end, 14040)
        self.assertTrue(all(r.src is None and r.ins.startswith("i_") for r in inserted.runs))
        self.assertFalse(self.ed.history.done[0].keeps_text)

    def test_insert_member_and_restore_link_to_existing_original(self):
        translated = member(self.source, "g_s", "m_st", "天空", lane="translation", style="OP.CN")
        track = self.timed(self.source, translated)
        ob = kara.effective(self.source, self.sm).beats
        track["beats"][0]["link"] = {"end": kara.beat_link(ob[1], "start")}
        self.p = recorded(self.source)
        result = self.planned()
        summary = self.roundtrip(result, self.chosen(result))
        self.assertEqual(summary.inserted_lines, 1)
        g = self.doc.group("g_t")
        self.assertEqual([self.doc.members[mid].lane for mid in g.members], ["translation", "original"])
        t = kara.tracks_of(self.doc, g.members[0])[0]
        link = t["beats"][0]["link"]["end"]
        self.assertEqual(link["core"], anchor_of(visible_map(self.doc.members[self.tm.id])[1]))
        self.assertNotIn("member_index", link)
        self.assertNotIn("i_m_s", json.dumps(link))

    def test_group_insert_links_resolve_after_both_members_and_edits_map(self):
        self.sm.runs = [Run("ガー", ins="i_m_s", b=2)]
        self.source.annotations.clear()
        self.source.karaoke.clear()
        edits = [{"op": "split", "at": kara.boundary_key(self.source, self.sm, 1)}]
        self.timed(self.source, self.sm, edits=edits)
        translated = member(self.source, "g_s", "m_st", "天空", lane="translation", style="OP.CN")
        track = self.timed(self.source, translated)
        track["beats"][0]["link"] = {"end": kara.beat_link(kara.effective(self.source, self.sm).beats[1], "start")}
        self.p = recorded(self.source)
        self.doc.groups.clear()
        self.doc.members.clear()
        self.doc.structure_changed()
        result = self.planned()
        summary = self.roundtrip(result, self.chosen(result))
        self.assertEqual((summary.inserted_lines, summary.karaoke_lines), (2, 2))
        g = self.doc.groups[0]
        original = next(m for m in self.doc.members_of(g) if m.lane == "original")
        t = kara.tracks_of(self.doc, original.id)[0]
        self.assertEqual(t["edits"][0]["at"]["core"], anchor_of(visible_map(original)[1][1:]))
        translated = next(m for m in self.doc.members_of(g) if m.lane == "translation")
        link = kara.tracks_of(self.doc, translated.id)[0]["beats"][0]["link"]["end"]
        self.assertEqual(link["core"], anchor_of(visible_map(original)[1][1:]))

    def test_partial_karaoke_only_timed_keys_transferred(self):
        self.sm.runs = [Run("あいう", ins="i_m_s", b=3)]
        self.source.annotations.clear()
        self.source.karaoke.clear()
        self.timed(self.source, self.sm)
        self.tm.runs = [Run("あいう", ins="i_m_t", b=3)]
        tr = next(iter(self.source.karaoke.values()))
        tr["beats"] = tr["beats"][:1]
        self.p = recorded(self.source)
        result = self.planned()
        summary = self.roundtrip(result, self.chosen(result))
        self.assertEqual(summary.karaoke_lines, 1)
        track = kara.tracks_of(self.doc, self.tm.id)[0]
        self.assertEqual(len(track["beats"]), 1)
        self.assertEqual(track["beats"][0]["by"], "memory")

    def test_existing_translation_links_to_newly_inserted_original(self):
        translated = member(self.source, "g_s", "m_st", "天空", lane="translation", style="OP.CN")
        tr = self.timed(self.source, translated)
        tr["beats"][0]["link"] = {"end": kara.beat_link(kara.effective(self.source, self.sm).beats[1], "start")}
        self.p = recorded(self.source)
        self.tm.lane, self.tm.lang = "translation", "zh-Hant"
        self.tm.runs = [Run("天空", ins="i_m_t", b=2)]
        result = self.planned()
        summary = self.roundtrip(result, self.chosen(result))
        self.assertEqual(summary.inserted_lines, 1)
        original = next(m for m in self.doc.members_of(self.doc.group("g_t")) if m.lane == "original")
        link = kara.tracks_of(self.doc, self.tm.id)[0]["beats"][0]["link"]["end"]
        self.assertEqual(link["core"], anchor_of(visible_map(original)[1]))
        self.assertEqual(summary.skipped, [])

    def test_source_ledger_and_order_unchanged_by_inserts_and_group_ties(self):
        source = Source("ja1", "ja", "original", "srt", "readonly.srt", "source.srt", "", 0, "utf-8", "lf",
                        events=[SourceEvent(0, "Dialogue", 10000, 14000, "別文"),
                                SourceEvent(1, "Dialogue", 20000, 24000, "後文")])
        self.tm.runs = [Run("別文", src="ja1", e=0, b=2)]
        self.tm.start, self.tm.end = 10000, 14000
        self.doc.groups.append(Group("g_later"))
        later = member(self.doc, "g_later", "m_later", "後文", start=20000, end=24000)
        later.runs = [Run("後文", src="ja1", e=1, b=2)]
        self.doc.structure_changed()
        self.ed.sources = {"ja1": source}
        # A different lane avoids a text-differs suggestion and permits a new group.
        self.p["groups"][0]["members"][0]["lane"] = "translation"
        before = validate(self.doc, self.ed.sources)
        self.assertTrue(before.ok)
        result = self.planned()
        self.roundtrip(result, self.chosen(result))
        self.assertEqual(self.doc.groups[0].id, "g_t")
        self.assertEqual(self.doc.groups[-1].id, "g_later")
        self.assertEqual(validate(self.doc, self.ed.sources).ledger, before.ledger)

    def test_validation_failure_rolls_back_transaction(self):
        result = self.planned()
        actions(result.lines[0])["time"].data["end"] = -1
        before = self.serialized()
        with self.assertRaisesRegex(ValueError, "validation errors"):
            self.ed.apply_passage(result, self.chosen(result))
        self.assertEqual(self.serialized(), before)
        self.assertFalse(self.ed.history.done)



class CoverageTests(unittest.TestCase):
    def passage(self, spans, *, together=False):
        source = Document("source")
        for i, (start, end) in enumerate(spans):
            gid = "g_0" if together else f"g_{i}"
            if not source.groups or not together:
                source.groups.append(Group(gid, "song"))
            member(source, gid, f"m_{i}", f"line{i}", start=start, end=end)
        return recorded(source)

    def test_short_gap_threshold_boundary_and_conservative_crossing(self):
        p = self.passage([(0, 10000), (10000, 15000), (15000, 25000), (9000, 16000), (25000, 25001)])
        m = Match(5000, .99, (0, 25000), [], 250, 1.01, ((0, 10000), (15000, 25000)))
        for scale in (False, True):
            result = plan(Document("target"), p, m, scale=scale)
            self.assertEqual([l.coverage for l in result.lines], ["direct", "inferred", "direct", "inferred", "unknown"])
            self.assertEqual([l.actions[0].checked for l in result.lines], [True, True, True, True, False])
        for gap, state in ((8000, "inferred"), (8001, "unknown")):
            m.covered_intervals = ((0, 10000), (10000 + gap, 30000))
            self.assertEqual(plan(Document("target"), p, m).lines[1].coverage, state)
        m.covered_intervals = ()
        self.assertTrue(all(l.coverage == "unknown" for l in plan(Document("target"), p, m).lines))

    def test_mixed_group_defaults_manual_opt_in_order_and_undo_redo(self):
        p = self.passage([(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000)], together=True)
        m = Match(5000, 1., (0, 40000), [], 0, 1., ((0, 10000), (30000, 40000)))
        for all_lines in (False, True):
            doc = Document("target")
            editor = Editor(doc, {})
            before = document_to_json(doc)
            result = plan(doc, p, m)
            chosen = {a.id for l in result.lines for a in l.actions if all_lines or a.checked}
            summary = editor.apply_passage(result, chosen)
            self.assertEqual(summary.inserted_lines, 4 if all_lines else 2)
            self.assertEqual(len(doc.groups), 1)
            self.assertEqual([doc.members[mid].text for mid in doc.groups[0].members],
                             ["line0", "line1", "line2", "line3"] if all_lines else ["line0", "line3"])
            self.assertTrue(validate(doc, {}).ok)
            editor.history.undo()
            after = document_to_json(doc); after["revision"] = before["revision"]
            self.assertEqual(after, before)
            editor.history.redo()
            self.assertEqual(len(doc.members), summary.inserted_lines)

    def test_unknown_missing_sibling_keeps_matched_actions_unchanged(self):
        p = self.passage([(0, 10000), (10000, 20000)], together=True)
        p["groups"][0]["members"][1]["lane"] = "translation"
        doc = Document("target", groups=[Group("g_target", "song")])
        member(doc, "g_target", "m_target", "line0", start=5000, end=15000)
        m = Match(5000, 1., (0, 20000), [], 0, 1., ((0, 10000),))
        result = plan(doc, p, m)
        self.assertEqual(result.lines[0].status, "matched")
        self.assertEqual(result.lines[0].actions, [])
        self.assertEqual(result.lines[1].actions[0].kind, "insert_member")
        self.assertFalse(result.lines[1].actions[0].checked)

if __name__ == "__main__":
    unittest.main()
