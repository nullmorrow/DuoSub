"""Dialogue/song conversion: exact round trips, dormant timing, source safety and export caches."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_stage0 import CN_ASS, JP_SRT, _write  # noqa: E402

from subkit.ass import AssDocument  # noqa: E402
from subkit.ass.text import tokenize  # noqa: E402
from subkit.core import karaoke as kara  # noqa: E402
from subkit.core.convert import (K_HINT, dialogue_text, export_stripped_text, plan_conversion, song_style_text,
                                 song_text, status_text)  # noqa: E402
from subkit.core.export import ExportCache, ExportOptions, build_ass  # noqa: E402
from subkit.core.importer import import_source  # noqa: E402
from subkit.core.merge import MergeSettings, merge_episode  # noqa: E402
from subkit.core.ops import Editor  # noqa: E402
from subkit.model.document import Annotation, Document, Group, Member, Run  # noqa: E402
from subkit.model.serialize import document_to_json, load_document, save_document  # noqa: E402
from subkit.model.validate import validate  # noqa: E402
from subkit.styles_default import default_styles_ass  # noqa: E402


class Conversion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.doc = Document("conversion")
        self.ed = Editor(self.doc, {})
        self.template = AssDocument.parse(default_styles_ass())

    def group(self, gid: str, *, kind: str = "dialogue", role: str = "dialogue", style: str = "JP",
              lane: str = "original", text: str = "かな", start: int = 1000, end: int = 3000,
              identity: str = "") -> Member:
        mid = "m_" + gid
        runs = [Run(t.raw, "tag" if t.kind in ("tags", "comment") else "text", ins=identity or mid,
                    a=t.start, b=t.end) for t in tokenize(text)]
        m = Member(mid, gid, "ja" if lane == "original" else "zh-Hans", lane, role, start, end,
                   style=style, runs=runs)
        self.doc.groups.append(Group(gid, kind, [mid]))
        self.doc.members[mid] = m
        self.doc.structure_changed()
        return m

    def timed(self, m: Member, *, unit: str = "syllable", twins: bool = False) -> str:
        beats = kara.derive(self.doc, m, unit)
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = m.start + i * 200, m.start + (i + 1) * 200, "tap"
        return self.ed.set_karaoke(m.id, beats, unit=unit, twins=twins, effect={"preset": "ko", "params": {"x": 1}})

    def snapshot(self) -> dict:
        d = copy.deepcopy(document_to_json(self.doc))
        d.pop("revision")
        return d

    def export(self, cache=None, *, mode="release"):
        return build_ass(self.doc, {}, self.template, self.path,
                         ExportOptions(mode=mode, ruby=False), cache=cache)

    def test_sign_roundtrip_readings_export_and_undo(self) -> None:
        m = self.group("sign", kind="sign", role="sign", style="Sign", text=r"{\pos(50,60)\move(1,2,3,4)\an8\org(20,30)\i1}私が")
        m.extra["source_style"] = True
        m.margins = [15, 25, 35]
        before = self.snapshot()
        before_export = self.export()[0].events
        self.ed.convert_to_dialogue([m.group])
        self.assertEqual((m.role, m.style, m.margins), ("dialogue", "JP", [0, 0, 0]))
        self.assertTrue(self.doc.annotations)
        self.assertEqual(m.text, before["members"][m.id]["runs"][0]["text"] + "私が")
        from subkit.core.autotime_doc import speech_member, estimation_intervals
        self.assertTrue(speech_member(self.doc.group(m.group), m, {}))
        self.assertEqual(estimation_intervals(self.doc, {}), [(m.start, m.end)])
        converted, report = self.export()
        self.assertEqual(converted.events[0].style, "JP")
        for tag in (r"\pos", r"\move", r"\an", r"\org"):
            self.assertNotIn(tag, converted.events[0].text)
        self.assertIn(r"\i1", converted.events[0].text)
        self.assertEqual(report.positions_overridden, 1)
        self.assertTrue(validate(self.doc, {}).ok)
        after = self.snapshot()
        self.ed.history.undo()
        self.assertEqual(self.snapshot(), before)
        self.ed.history.redo()
        self.assertEqual(self.snapshot(), after)
        self.ed.convert_to_sign([m.group])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.export()[0].events, before_export)
        self.assertTrue(validate(self.doc, {}).ok)

    def test_sign_multiselect_and_protected_members(self) -> None:
        a = self.group("sign_a", kind="sign", role="sign", lane="translation", text="我")
        b = self.group("sign_b", kind="sign", role="sign", lane="translation", text="你")
        song = self.group("song", kind="song", role="song", lane="translation", text="我")
        info = self.group("info", kind="info", role="comment")
        self.doc.members["protected"] = Member("protected", a.group, "ja", "original", "song", 1000, 3000)
        self.doc.group(a.group).members.append("protected")
        protected = copy.deepcopy(self.doc.members["protected"])
        p = self.ed.convert_to_dialogue([b.group, a.group, info.group])
        self.assertEqual(p.eligible, [a.group, b.group])
        self.assertEqual(p.gids, [a.group, b.group])
        self.assertEqual(self.doc.members["protected"], protected)
        self.assertEqual(len(self.ed.history.done), 1)
        self.ed.convert_to_sign([a.group, b.group, song.group, info.group])
        self.assertEqual((self.doc.group(song.group).kind, self.doc.group(info.group).kind), ("song", "info"))
        self.assertEqual(self.doc.members["protected"], protected)
        self.ed.history.undo()
        self.assertEqual(self.doc.group(a.group).kind, "dialogue")
        self.ed.history.redo()
        self.assertEqual(self.doc.group(a.group).kind, "sign")

    def test_dialogue_sign_uses_source_style_and_exact_return(self) -> None:
        cn_path, ja_path = self.path / "cn.ass", self.path / "ja.srt"
        _write(self.path, "cn.ass", CN_ASS)
        _write(self.path, "ja.srt", JP_SRT)
        ep = self.path / "episode"
        cn = import_source(cn_path, src_id="t1", lang="zh-Hans", lane="translation", episode_dir=ep,
                           cn_cfg={"dialogue": ["Default"]})
        ja = import_source(ja_path, src_id="o1", lang="ja", lane="original", episode_dir=ep)
        for src in (cn, ja):
            self.addCleanup(os.chmod, ep / src.copy, stat.S_IREAD | stat.S_IWRITE)
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cn_path, ja_path, ep / cn.copy, ep / ja.copy)}
        self.doc = merge_episode("E1", cn, ja).document
        self.ed = Editor(self.doc, {cn.id: cn, ja.id: ja})
        sign = next(g for g in self.doc.groups if g.kind == "sign")
        before = self.snapshot()
        original = build_ass(self.doc, self.ed.sources, self.template, ep, ExportOptions(ruby=False))[0].events
        self.ed.convert_to_dialogue([sign.id])
        save_document(self.doc, ep / "document.json")
        loaded = load_document(ep / "document.json")
        Editor(loaded, self.ed.sources).convert_to_sign([sign.id])
        loaded_json = document_to_json(loaded); loaded_json.pop("revision")
        self.assertEqual(loaded_json, before)
        self.ed.convert_to_sign([sign.id])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(build_ass(self.doc, self.ed.sources, self.template, ep, ExportOptions(ruby=False))[0].events, original)
        g = next(g for g in self.doc.groups if g.kind == "dialogue")
        member = next(m for m in self.doc.members_of(g) if m.lane == "translation")
        before = self.snapshot()
        self.ed.convert_to_sign([g.id])
        self.assertEqual(member.style, "Default")
        self.assertTrue(member.extra["source_style"])
        self.ed.convert_to_dialogue([g.id])
        self.assertEqual(self.snapshot(), before)
        self.assertTrue(validate(self.doc, self.ed.sources).ok)
        self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes}, hashes)

    def test_sign_conversion_enters_dialogue_tools_and_preserves_existing_readings(self) -> None:
        from subkit.core.merge import dialogue_lanes
        from subkit.core.people import needs_speaker
        m = self.group("sign", kind="sign", role="sign", text="私")
        self.doc.annotations["manual"] = Annotation("manual", "ruby", m.id,
            [{"ins": m.id, "a": 0, "b": 1}], "私", "わたくし", status="confirmed", origin="manual")
        annotations = copy.deepcopy(self.doc.annotations)
        g = self.doc.group(m.group)
        self.assertEqual(dialogue_lanes(self.doc, g), {"original"})
        self.assertFalse(needs_speaker(g, m))
        self.ed.convert_to_dialogue([g.id])
        self.assertEqual(dialogue_lanes(self.doc, g), {"original"})
        self.assertTrue(needs_speaker(g, m))
        self.assertEqual(self.doc.annotations, annotations)
        self.ed.convert_to_sign([g.id])
        self.assertEqual(self.doc.annotations, annotations)
        self.assertFalse(needs_speaker(g, m))

    def test_sign_manual_readings_survive_return(self) -> None:
        m = self.group("sign", kind="sign", role="sign", text="私")
        self.ed.convert_to_dialogue([m.group])
        ann = next(iter(self.doc.annotations.values()))
        self.ed.set_reading(ann.id, "わたくし")
        self.ed.convert_to_sign([m.group])
        self.assertEqual(self.doc.annotations[ann.id].text, "わたくし")

    def test_dialogue_roundtrips_restore_top_alt_and_leave_other_fields(self) -> None:
        m = self.group("dialogue", role="top", style="CN.Top", lane="translation", text="心")
        g = self.doc.group(m.group)
        g.state, g.flags, g.origin, g.extra = "confirmed", ["order_exception"], {"by": "user"}, {"other": 1}
        m.layer, m.margins, m.effect = 7, [10, 20, 30], "fx"
        m.speaker, m.overrides = {"label": "太郎"}, {"an": 8, "pos": [100, 200]}
        m.format = [{"anchor": [{"ins": m.id, "a": 0, "b": 1}], "props": {"fs": 80}}]
        m.source_kind, m.extra = "Comment", {"other": "keep"}
        alt = Member("alt", g.id, "ja", "original", "alt", 1010, 3010, style="JP.Special",
                     runs=[Run("心", ins="alt", b=1)])
        g.members.append(alt.id)
        self.doc.members[alt.id] = alt
        for role in ("sign", "info", "comment"):
            protected = Member(role, g.id, "ja", "original", role, 1000, 3000, style="Default")
            self.doc.members[role] = protected
            g.members.append(role)
        self.doc.annotations["ruby"] = Annotation("ruby", "ruby", m.id, [{"ins": m.id, "a": 0, "b": 1}],
                                                  "心", "こころ", status="confirmed")
        members = copy.deepcopy(self.doc.members)
        annotations = copy.deepcopy(self.doc.annotations)
        initial_group = dataclasses.asdict(g)
        for _ in range(3):
            p = self.ed.convert_to_song([g.id], "ED")
            self.assertEqual(p.members_changed, 2)
            self.assertEqual((m.role, m.style, alt.role, alt.style), ("song", "ED.CN", "song", "ED.JP"))
            self.ed.convert_to_dialogue([g.id])
            self.assertEqual((g.kind, m.role, m.style, alt.role, alt.style),
                             ("dialogue", "top", "CN.Top", "alt", "JP.Special"))
            for mid, before in members.items():
                now = dataclasses.asdict(self.doc.members[mid])
                original = dataclasses.asdict(before)
                for field in ("rev", "extra"):
                    now.pop(field)
                    original.pop(field)
                self.assertEqual(now, original)
            self.assertEqual(self.doc.annotations, annotations)
            self.assertEqual({k: v for k, v in dataclasses.asdict(g).items() if k != "extra"},
                             {k: v for k, v in initial_group.items() if k != "extra"})
            self.assertEqual(m.extra["other"], "keep")
        for role in ("sign", "info", "comment"):
            self.assertEqual(self.doc.members[role], members[role])

    def test_each_conversion_is_one_exact_undo_redo(self) -> None:
        m = self.group("song", kind="song", role="song", style="OP.JP")
        self.timed(m)
        before = self.snapshot()
        self.ed.convert_to_dialogue([m.group])
        after = self.snapshot()
        self.assertEqual(len(self.ed.history.done), 2)
        self.assertEqual(self.ed.history.done[-1].label, "转为普通台词")
        self.ed.history.undo()
        self.assertEqual(self.snapshot(), before)
        self.ed.history.redo()
        self.assertEqual(self.snapshot(), after)
        self.ed.convert_to_song([m.group], "OP")
        song = self.snapshot()
        self.assertEqual(self.ed.history.done[-1].label, "转为歌曲")
        self.ed.history.undo()
        self.assertEqual(self.snapshot(), after)
        self.ed.history.redo()
        self.assertEqual(self.snapshot(), song)

    def test_imported_song_roundtrip_preserves_all_tracks_and_beats(self) -> None:
        m = self.group("song", kind="song", role="song", style="OP.JP", text="ガーデン")
        cn = self.group("cn", kind="song", role="song", lane="translation", style="OP.CN", text="世界")
        kid = self.timed(m, unit="mora")
        self.timed(cn)
        second = copy.deepcopy(self.doc.karaoke[kid])
        second["id"] = "k_second"
        self.doc.karaoke[second["id"]] = second
        saved = copy.deepcopy(self.doc.karaoke)
        self.ed.convert_to_dialogue([m.group, cn.group])
        self.assertTrue(all(t["dormant"] for t in self.doc.karaoke.values()))
        idx = kara.KIndex(self.doc)
        self.assertFalse(idx)
        self.assertEqual((idx.beats, idx.by_core, idx.ids, idx.members), ({}, {}, set(), set()))
        self.assertFalse(idx.has(m))
        self.assertEqual(kara.tracks_of(self.doc, m.id), [])
        self.assertEqual(len(kara.tracks_of(self.doc, m.id, dormant=True)), 2)
        self.assertEqual(kara.effective(self.doc, m).state, "none")
        self.assertEqual(kara.unfinished(self.doc), [])
        p = plan_conversion(self.doc, [m.group, cn.group], "song")
        self.assertEqual(p.default_family, "OP")
        self.assertEqual(p.dormant_members, [m.id, cn.id])
        self.ed.convert_to_song([m.group, cn.group], "OP")
        self.assertEqual((m.style, cn.style), ("OP.JP", "OP.CN"))
        self.assertEqual(self.doc.karaoke, saved)
        self.assertTrue(kara.KIndex(self.doc).has(m))
        self.assertTrue(kara.effective(self.doc, m).complete)

    def test_dormant_track_cannot_apply_through_shared_identities(self) -> None:
        m = self.group("song", kind="song", role="song", style="OP.JP", identity="shared")
        other = self.group("sign", kind="sign", role="sign", identity="shared")
        self.timed(m)
        self.assertTrue(kara.KIndex(self.doc).has(other))
        self.ed.convert_to_dialogue([m.group])
        idx = kara.KIndex(self.doc)
        self.assertFalse(idx.has(other))
        self.assertFalse(any(b.timed for b in kara.effective(self.doc, other, idx=idx).beats))

    def test_dialogue_off_blocks_other_active_track_with_shared_identities(self) -> None:
        m = self.group("song", kind="song", role="song", style="OP.JP", identity="shared")
        other = self.group("sign", kind="sign", role="sign", identity="shared")
        self.timed(other)
        self.ed.convert_to_dialogue([m.group])
        idx = kara.KIndex(self.doc)
        self.assertTrue(idx.has(other))
        self.assertFalse(idx.has(m))
        self.assertFalse(kara.has_karaoke(self.doc, m, idx))
        self.assertEqual(kara.effective(self.doc, m, idx=idx).state, "none")
        self.assertFalse(any(b.timed for b in kara.effective(self.doc, m, idx=idx).beats))
        out, rep = self.export()
        self.assertEqual(out.events[0].text, "かな")
        self.assertEqual(rep.karaoke_lines, 1)

    def test_unfinished_ignores_dormant_and_returns_after_reactivation(self) -> None:
        m = self.group("song", kind="song", role="song", style="IN.JP")
        self.timed(m)
        self.ed.set_member_time(m.id, end=1100)
        self.assertTrue(kara.unfinished(self.doc))
        self.ed.convert_to_dialogue([m.group])
        self.assertEqual(kara.unfinished(self.doc), [])
        self.ed.convert_to_song([m.group], "IN")
        self.assertTrue(kara.unfinished(self.doc))

    def test_retiming_reuses_reactivated_track_id(self) -> None:
        m = self.group("song", kind="song", role="song", style="ED.JP")
        kid = self.timed(m)
        self.ed.convert_to_dialogue([m.group])
        self.ed.convert_to_song([m.group], "ED")
        before = copy.deepcopy(self.doc.karaoke)
        beats = kara.effective(self.doc, m).beats
        beats[0].t1 += 20
        self.assertEqual(self.ed.set_karaoke(m.id, beats), kid)
        self.assertEqual(list(self.doc.karaoke), [kid])
        self.assertNotIn("dormant", self.doc.karaoke[kid])
        self.ed.history.undo()
        self.assertEqual(self.doc.karaoke, before)

    def test_family_custom_mappings_and_saved_song_style(self) -> None:
        m = self.group("song", kind="song", role="song", style="OP.Special.JP")
        styles = dict(MergeSettings().styles, **{"original/song": "{fam}.Custom.JP", "original/dialogue": "Talk"})
        self.ed.convert_to_dialogue([m.group], styles)
        self.assertEqual((m.role, m.style), ("dialogue", "Talk"))
        p = plan_conversion(self.doc, [m.group], "song", styles=styles, template_styles={"OP.CN"})
        self.assertEqual(p.default_family, "OP")
        self.assertIn("OP.Special.JP", p.missing_styles)
        self.assertEqual(song_style_text(p), "样式：OP.CN / OP.Custom.JP / OP.Special.JP")
        self.ed.convert_to_song([m.group], "OP", styles)
        self.assertEqual(m.style, "OP.Special.JP")
        self.ed.convert_to_dialogue([m.group], styles)
        self.ed.convert_to_song([m.group], "ED", styles)
        self.assertEqual(m.style, "ED.Custom.JP")
        self.ed.convert_to_dialogue([m.group], styles)
        self.assertEqual(m.style, "Talk")
        cn = self.group("cn", lane="translation", style="CN.Top", role="top", text="別")
        self.ed.convert_to_song([cn.group], "IN", styles)
        self.assertEqual(cn.style, "IN.CN")

    def test_twin_groups_are_in_document_order_with_inclusive_20ms_boundary(self) -> None:
        twin = self.group("twin", text=r"{\blur2}かな", role="alt", start=1020, end=2980)
        m = self.group("selected")
        far = self.group("far", start=1040, end=2960)
        edge = self.group("edge", start=1021)
        lane = self.group("lane", lane="translation")
        sign = self.group("sign", kind="sign", role="sign")
        other = self.group("other", text="かに")
        p = plan_conversion(self.doc, [m.group, m.group], "song", family="OP")
        self.assertEqual(p.eligible, [m.group])
        self.assertEqual(p.twins, [twin.group])
        self.assertEqual(p.gids, [twin.group, m.group])
        self.assertEqual(p.members_changed, 2)
        self.ed.convert_to_song([m.group], "OP")
        self.assertEqual(self.doc.group(twin.group).kind, "song")
        for untouched in (far, edge, lane, sign, other):
            self.assertNotEqual(untouched.role, "song")
        back = self.ed.convert_to_dialogue([m.group])
        self.assertEqual(back.twins, [twin.group])
        self.assertEqual(twin.role, "alt")

    def test_selected_order_skips_and_noop_do_not_add_history(self) -> None:
        a = self.group("a")
        b = self.group("b", text="べつ")
        song = self.group("song", kind="song", role="song")
        sign = self.group("sign", kind="sign", role="sign")
        info = self.group("info", kind="info", role="comment")
        p = plan_conversion(self.doc, [b.group, song.group, a.group, info.group, sign.group, "missing"], "song")
        self.assertEqual(p.eligible, [a.group, b.group])
        self.assertEqual([gid for gid, _reason in p.skipped], [song.group, sign.group, info.group, "missing"])
        self.assertTrue(all(reason for _gid, reason in p.skipped))
        reverse = plan_conversion(self.doc, [a.group, b.group, song.group, info.group, sign.group], "dialogue")
        self.assertEqual(reverse.eligible, [song.group, sign.group])
        self.assertEqual(len(reverse.skipped), 3)
        before = self.snapshot()
        self.ed.convert_to_song([song.group, sign.group], "IN")
        self.ed.convert_to_dialogue([])
        self.assertEqual(len(self.ed.history.done), 0)
        self.assertEqual(self.snapshot(), before)
        with self.assertRaises(ValueError):
            self.ed.convert_to_song([a.group], "BAD")
        with self.assertRaises(ValueError):
            plan_conversion(self.doc, [], "invalid")
        self.assertEqual(self.snapshot(), before)

    def test_plan_and_chinese_text_are_pure_and_cover_counts(self) -> None:
        m = self.group("song", kind="song", role="song", style="OP.JP", text=r"{\k20}か{\K20}な")
        twin = self.group("twin", kind="song", role="song", style="OP.JP", text="かな", start=1010)
        self.timed(m)
        sign = self.group("sign", kind="sign", role="sign")
        before = self.snapshot()
        p = plan_conversion(self.doc, [m.group, sign.group], "dialogue")
        self.assertEqual((p.timed_members, p.source_k_members, p.twins), ([m.id], [m.id], [twin.group]))
        txt = dialogue_text(p)
        for s in ("把选中的 2 组", "已打的 K 轴（1 行）", "自带的 \\k（1 行）", "另外 1 组", "（可以撤销）"):
            self.assertIn(s, txt)
        self.assertEqual(self.snapshot(), before)
        self.ed.convert_to_dialogue([m.group])
        before = self.snapshot()
        p = plan_conversion(self.doc, [m.group, sign.group], "song", template_styles={"OP.JP"})
        txt = song_text(p)
        for s in ("会恢复 1 行", "另外 1 组", "1 组不是普通台词", "模板里没有 OP.CN"):
            self.assertIn(s, txt)
        self.assertEqual(song_style_text(p), "样式：OP.CN / OP.JP")
        self.assertEqual(status_text(p), "已转为歌曲：2 组（Ctrl+Z 撤销）")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(K_HINT, "这一组不是歌曲，不能打 K 轴（右键 → 转为歌曲）")
        self.assertEqual(export_stripped_text([]), "")
        report = export_stripped_text(str(i) for i in range(8))
        self.assertIn("8 行已转为普通台词", report)
        self.assertEqual(report.splitlines()[1:], [str(i) for i in range(6)])

    def test_save_reopen_preserves_swap_memory_off_and_dormant(self) -> None:
        m = self.group("song", kind="song", role="song", style="OP.JP")
        self.timed(m)
        saved = copy.deepcopy(self.doc.karaoke)
        self.ed.convert_to_dialogue([m.group])
        path = self.path / "document.json"
        save_document(self.doc, path)
        loaded = load_document(path)
        self.assertEqual(document_to_json(loaded), document_to_json(self.doc))
        self.assertIn('"before_convert"', path.read_text(encoding="utf-8"))
        self.assertTrue(loaded.members[m.id].extra["karaoke_off"])
        self.assertEqual(loaded.group(m.group).extra["before_convert"], {"kind": "song"})
        Editor(loaded, {}).convert_to_song([m.group], "OP")
        self.assertEqual(loaded.members[m.id].style, "OP.JP")
        self.assertEqual(loaded.karaoke, saved)

    def test_export_all_source_tags_and_subkit_timing_roundtrip_byte_identical(self) -> None:
        m = self.group("song", kind="song", role="song", style="OP.JP",
                       text=r"{\pos(100,200)\k20}か{\kf20}な{\ko20}よ{\K20}ね")
        self.timed(m)
        before, _ = self.export()
        source = copy.deepcopy(m.runs)
        for mode in ("release", "preview", "work"):
            self.ed.convert_to_dialogue([m.group])
            out, rep = self.export(mode=mode)
            self.assertEqual(out.events[0].text, r"{\pos(100,200)}かなよね")
            self.assertEqual(rep.karaoke_lines, 0)
            self.assertEqual(rep.karaoke_unfinished, [])
            self.assertEqual(rep.karaoke_stripped, ["0:00:01.00–0:00:03.00 かなよね"])
            self.ed.convert_to_song([m.group], "OP")
            after, rep = self.export()
            self.assertEqual(after.serialize().encode("utf-8"), before.serialize().encode("utf-8"))
            self.assertEqual(rep.karaoke_stripped, [])
            self.assertEqual(m.runs, source)

    def test_source_tags_without_tracks_restore_and_untagged_lines_are_not_reported(self) -> None:
        m = self.group("song", kind="song", role="song", style="ED.JP", text=r"{\k20}か{\kf0}な{\ko30}よ{\K20}ね")
        clean = self.group("clean", kind="song", role="song", style="ED.JP", text=r"{\pos(100,200)}よ")
        before, _ = self.export()
        self.ed.convert_to_dialogue([m.group, clean.group])
        out, rep = self.export()
        self.assertEqual(out.events[0].text, "かなよね")
        self.assertEqual(out.events[1].text, clean.text)
        self.assertEqual(len(rep.karaoke_stripped), 1)
        self.ed.convert_to_song([m.group, clean.group], "ED")
        after, _ = self.export()
        self.assertEqual(after.serialize().encode("utf-8"), before.serialize().encode("utf-8"))

    def test_preview_cache_changes_on_conversion_undo_redo_and_reports_hits(self) -> None:
        m = self.group("song", kind="song", role="song", style="IN.JP", text=r"{\kf20}かな")
        self.timed(m)
        cache = ExportCache()
        before, _ = self.export(cache, mode="preview")
        self.ed.convert_to_dialogue([m.group])
        after, rep = self.export(cache, mode="preview")
        self.assertEqual(after.events[0].text, "かな")
        self.assertEqual((rep.laid_out, len(rep.karaoke_stripped)), (1, 1))
        hit, rep = self.export(cache, mode="preview")
        self.assertEqual(hit.serialize(), after.serialize())
        self.assertEqual((rep.laid_out, len(rep.karaoke_stripped)), (0, 1))
        self.ed.history.undo()
        undo, rep = self.export(cache, mode="preview")
        self.assertEqual(undo.serialize(), before.serialize())
        self.assertEqual((rep.laid_out, rep.karaoke_stripped), (1, []))
        self.ed.history.redo()
        redo, rep = self.export(cache, mode="preview")
        self.assertEqual(redo.serialize(), after.serialize())
        self.assertEqual(len(rep.karaoke_stripped), 1)

    def test_cache_signature_covers_karaoke_off_even_when_role_and_style_are_same(self) -> None:
        m = self.group("song", kind="song", role="song", style="JP", text=r"{\k20}かな")
        m.extra["before_convert"] = {"role": "dialogue", "style": "JP"}
        cache = ExportCache()
        before, _ = self.export(cache)
        self.ed.convert_to_dialogue([m.group])
        after, _ = self.export(cache)
        self.assertNotEqual(before.events[0].text, after.events[0].text)
        # Isolate the flag from the other signature fields through an undoable edit.
        def apply(tx):
            tx.member(m.id).extra.pop("karaoke_off")
        self.ed.history.run("flag", apply)
        restored, rep = self.export(cache)
        self.assertEqual(restored.events[0].text, m.text)
        self.assertEqual(rep.laid_out, 1)

    def test_real_import_sources_and_readonly_copies_are_byte_identical(self) -> None:
        orig = self.path / "orig"
        orig.mkdir()
        ep = self.path / "episode"
        cn_path = _write(orig, "cn.ass", CN_ASS.replace("你好啊", r"{\k20}你好啊"))
        ja_path = _write(orig, "ja.srt", JP_SRT)
        cn = import_source(cn_path, src_id="t1", lang="zh-Hans", lane="translation", episode_dir=ep,
                           cn_cfg={"dialogue": ["Default"]})
        ja = import_source(ja_path, src_id="o1", lang="ja", lane="original", episode_dir=ep)
        # Only read-only fixture copies created by this test need Windows attribute cleanup.
        for src in (cn, ja):
            self.addCleanup(os.chmod, ep / src.copy, stat.S_IREAD | stat.S_IWRITE)
        paths = [cn_path, ja_path, ep / cn.copy, ep / ja.copy]
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        doc = merge_episode("E1", cn, ja).document
        ed = Editor(doc, {cn.id: cn, ja.id: ja})
        gids = [g.id for g in doc.groups if g.kind == "dialogue"]
        for _ in range(2):
            ed.convert_to_song(gids, "OP")
            ed.convert_to_dialogue(gids)
            self.assertTrue(validate(doc, ed.sources).ok)
        save_document(doc, ep / "document.json")
        build_ass(doc, ed.sources, self.template, ep, ExportOptions(ruby=False))
        self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, hashes)


if __name__ == "__main__":
    unittest.main()
