"""Season batch boundaries, lossless lane filtering and real release pipeline."""

from __future__ import annotations

import copy
import subprocess
from types import SimpleNamespace
import numpy as np
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subkit import templates
from subkit.ass.document import AssDocument, Event
from subkit.core.batch import (BatchOptions, effect_styles, run_season, scan_season,
                               source_roles, song_families, unused_path, _hide_effect_overlap,
                               adopt_suggestions, embed_fonts, EpisodeResult, _attachment_families, _attachment_names,
                               _ass_encode, _ass_decode, _subset_document, _font_entries,
                               _attachment_cmaps, _carried_gaps, _is_interjection_line,
                               _hide_unpaired_interjections)
from subkit.audio import AudioFeatures, load_features, waveform_cache_file
from subkit.model.document import Document, Group, Member, Run
from subkit.core.importer import import_source
from subkit.core.ops import Editor
from subkit.core.workspace import Project
from subkit.sources.cn import classify


def ass_text(events, *, names=("Chinese", "Japanese", "OP")):
    doc = AssDocument.parse(templates.build_styles(templates.font_set("utena_serif"), templates.palette("utena")))
    doc.set_info("PlayResX", 1920)
    doc.set_info("PlayResY", 1080)
    base = doc.style("CN")
    doc.styles = []
    for name in names:
        style = base.copy(name)
        style.set("Fontname", "Arial")
        style.set("Fontsize", "48" if name == "Japanese" else "60")
        doc.styles.append(style)
    doc.events = [Event(kind="Dialogue", start=start, end=end, style=style, text=text)
                  for start, end, style, text in events]
    return doc.serialize()


class BatchCoreTests(unittest.TestCase):
    def setUp(self):
        # The bridge owns this run's scratch lifetime, including read-only source copies.
        self.root = Path(tempfile.mkdtemp(prefix="batch_test_", dir=os.environ.get("TEMP")))
        self.season = self.root / "Season 01"
        self.season.mkdir()
        template = AssDocument.parse(templates.build_styles(templates.font_set("utena_serif"), templates.palette("utena")))
        for name, size in (("CN", 64), ("JP", 48), ("JP.Ruby", 38.4)):
            template.style(name).set("Fontname", "Arial")
            template.style(name).set("Fontsize", str(size))
        self.project = Project.create(self.root / "Existing Project", "Chosen project",
                                      translation="zh-Hans", template_text=template.serialize())

    def put(self, name, text=""):
        path = self.season / name
        path.write_text(text, encoding="utf-8")
        return path

    def pair(self, key="S01E001", bilingual=False):
        events = [(100, 300, "Chinese", "你好"), (500, 600, "OP", r"{\pos(320,100)\kf20}特效")]
        if bilingual:
            events += [(100, 300, "Japanese", "こんにちは"), (500, 600, "Japanese", "♪うた")]
        cn = self.put(f"{key}.ass", ass_text(events))
        jp = cn if bilingual else self.put(f"{key}.srt", "1\n00:00:01,000 --> 00:00:03,000\n(太郎)こんにちは\n\n"
                                          "2\n00:00:05,000 --> 00:00:06,000\n♪うた♪\n")
        return cn, jp

    def run_fast(self, options, **kw):
        with patch("subkit.templates.fit_layout", side_effect=lambda f, p, **k: (f, {})):
            return run_season(self.season, self.project, __import__("dataclasses").replace(options, assfonts=None), **kw)

    def song_doc(self, *, dense=False, ruby=False, drawing=False, shared=False):
        names = ("Chinese", "Japanese", "OPJP", "OPCN", "EDJP-V", "EDCN-V", "EDJP-furi", "Screen", "DRAW", "IN")
        doc = AssDocument.parse(ass_text([(100, 200, "Chinese", "中文"),
                                         (100, 200, "Japanese", "こんにちは")], names=names))
        for style, text in (("OPJP", r"{\k100}夢を見る"), ("OPCN", "看见梦想"),
                            ("EDJP-V", "明日へ行く"), ("EDCN-V", "走向明天")):
            doc.events.append(Event(kind="Comment", start=500 if style.startswith("OP") else 1000,
                                    end=700 if style.startswith("OP") else 1200,
                                    style=style, effect="karaoke", text=text))
            for _ in range(51 if dense and style == "OPJP" else 2):
                doc.events.append(Event(start=500 if style.startswith("OP") else 1000,
                                        end=700 if style.startswith("OP") else 1200,
                                        style=style, effect="fx", text=r"{\pos(10,20)}特效"))
        for name, alignment in (("EDJP-V", 3), ("EDCN-V", 7)):
            doc.style(name).set("Fontname", "@MS Gothic")
            doc.style(name).set("Angle", "270")
            doc.style(name).set("Alignment", alignment)
        doc.events += [Event(start=500, end=700, style="Screen", text=r"{\pos(50,50)}屏幕"),
                       Event(start=500, end=700, style="DRAW", effect="fx", text=r"{\p1}m 0 0 l 1 1")]
        if ruby:
            doc.events.append(Event(start=1000, end=1200, style="EDJP-furi", effect="fx", text="あした"))
        if drawing:
            doc.events.append(Event(start=500, end=700, style="OPJP", effect="fx", text=r"{\p1}m 0 0 l 1 1"))
        if shared:
            doc.events += [Event(kind="Comment", start=1300, end=1500, style="IN", effect="karaoke", text="夢を見た"),
                           Event(kind="Comment", start=1300, end=1500, style="IN", effect="karaoke", text="看见梦想")]
        return doc

    def test_song_families_nonbiri_defaults_and_shared_kana(self):
        doc = self.song_doc(shared=True)
        families = {f.family: f for f in song_families(doc)}
        self.assertEqual(set(families), {"OP", "ED", "IN"})
        self.assertEqual({k: f.default for k, f in families.items()},
                         {"OP": "transcribe", "ED": "transcribe_vertical", "IN": "transcribe"})
        self.assertEqual(families["OP"].karaoke_lines, {"OPJP": 1, "OPCN": 1})
        self.assertEqual((families["OP"].fx_count, families["OP"].fx_ratio), (4, 2))
        self.assertTrue(families["ED"].vertical)
        cfg, jp = source_roles(doc, "5.2")
        self.assertEqual(jp, ["Japanese"])
        self.assertEqual([cfg["event_roles"][i] for i in range(len(doc.events)-2, len(doc.events))], ["song_jp", "song_cn"])
        self.assertIn("Screen", cfg["keep"])
        self.assertIn("DRAW", cfg["keep"])

    def test_song_defaults_berserk_and_ratio_boundary(self):
        families = {f.family: f for f in song_families(self.song_doc(dense=True, ruby=True))}
        self.assertEqual({k: f.default for k, f in families.items()}, {"OP": "keep", "ED": "transcribe_vertical"})
        self.assertTrue(families["ED"].own_furigana)
        self.assertEqual(families["OP"].fx_ratio, 51)
        doc = self.song_doc(dense=True)
        doc.events.pop(next(i for i, e in enumerate(doc.events) if e.style == "OPJP" and e.effect == "fx"))
        self.assertEqual(song_families(doc)[0].default, "transcribe")
        self.assertTrue(song_families(self.song_doc(drawing=True))[0].has_drawing)
        self.assertEqual(song_families(self.song_doc(drawing=True))[0].default, "keep")

    def test_vertical_detection_requires_both_font_and_angle(self):
        for font, angle, expected in (("@MS Gothic", 270, True), ("@MS Gothic", 90, True),
                                      ("@MS Gothic", 0, False), ("MS Gothic", 270, False)):
            doc = self.song_doc()
            for name in ("EDJP-V", "EDCN-V"):
                doc.style(name).set("Fontname", font)
                doc.style(name).set("Angle", angle)
            family = next(f for f in song_families(doc) if f.family == "ED")
            self.assertEqual(family.vertical, expected)
            self.assertEqual(family.default, "transcribe_vertical" if expected else "transcribe")

    def test_vertical_transcribe_source_styles_readings_fx_and_embedding_input(self):
        doc = self.song_doc()
        path = self.put("S01E001.ass", doc.serialize())
        before = path.read_bytes()
        template = self.project.template()
        template.styles.append(doc.style("EDJP-V").copy())
        template.style("EDJP-V").set("Angle", 0)
        self.project.replace_template(template.serialize())
        row = self.run_fast(BatchOptions(mode="5.2", require_video=False)).episodes[0]
        self.assertEqual(row.status, "完成", row.reason)
        ep = Project.open(self.project.root).episode(row.key)
        members = [m for m in ep.document.members.values() if m.style in ("EDJP-V", "EDCN-V")]
        self.assertEqual(len(members), 2)
        self.assertTrue(all(m.extra.get("source_style") for m in members))
        jp = next(m for m in members if m.lang == "ja")
        self.assertTrue(any(a.member == jp.id for a in ep.document.annotations.values()))
        self.assertFalse(any(a.member == m.id for m in members if m.lang != "ja"
                             for a in ep.document.annotations.values()))
        out = AssDocument.load(row.output)
        for name, alignment in (("src.EDJP-V", 3), ("EDCN-V", 7)):
            self.assertEqual(out.style(name).get("Fontname"), "@MS Gothic")
            self.assertEqual(out.style(name).num("Angle"), 270)
            self.assertEqual(out.style(name).alignment, alignment)
            self.assertTrue(any(e.style == name for e in out.dialogues()))
        self.assertTrue(any(e.style.startswith("src.EDJP-V") and e.style.endswith(".Ruby")
                            for e in out.dialogues()))
        self.assertFalse(any(e.effect == "fx" and e.style.startswith("ED") for e in out.events))
        subset = _subset_document(out, set())
        self.assertTrue(any(s.get("Fontname") == "@MS Gothic" for s in subset.styles))
        self.assertTrue(ep.validate().ok, ep.validate().summary())
        self.assertEqual(path.read_bytes(), before)

    def test_original_mixed_lyric_and_dialogue_cue_stays_dialogue(self):
        path = self.put("S01E001.srt", "1\n00:00:01,000 --> 00:00:02,000\n♪夢を見る♪\n普通の台詞\n")
        src = import_source(path, src_id="o1", lang="ja", lane="original", episode_dir=self.root / "mixed",
                            song_intervals=[(1000, 2000)])
        self.assertEqual(src.roles[0], "dialogue")
        self.assertEqual(src.events[0].text, "♪夢を見る♪\n普通の台詞")

    def test_plain_song_roles_and_line_count(self):
        doc = AssDocument.parse(ass_text([(100, 200, "Chinese", "中文"), (100, 200, "Japanese", "こんにちは"),
                                         (500, 700, "OPJP", "夢を見る"), (500, 700, "OPCN", "看见梦想")],
                                        names=("Chinese", "Japanese", "OPJP", "OPCN")))
        family = song_families(doc)[0]
        self.assertEqual((family.default, family.karaoke_count, family.line_count), ("transcribe", 0, 1))
        cfg, _ = source_roles(doc, "5.2")
        self.assertEqual([cfg["event_roles"][i] for i in (2, 3)], ["song_jp", "song_cn"])

    def test_karaoke_transcribe_readings_fx_exclusion_and_comment_provenance(self):
        doc = self.song_doc(ruby=True)
        source = self.put("S01E001.ass", doc.serialize())
        before = source.read_bytes()
        result = self.run_fast(BatchOptions(mode="5.2", require_video=False, song_choices={"ED": "transcribe"}))
        row = result.episodes[0]
        self.assertEqual(row.status, "完成", row.reason)
        ep = Project.open(result.project).episode(row.key)
        self.assertTrue(ep.validate().ok, ep.validate().summary())
        songs = [g for g in ep.document.groups if g.kind == "song"]
        self.assertEqual(len(songs), 2)
        self.assertTrue(all(len(g.members) == 2 for g in songs))
        original = [m for g in songs for m in ep.document.members_of(g) if m.lane == "original"]
        self.assertTrue(all(any(a.member == m.id for a in ep.document.annotations.values()) for m in original))
        for m in original:
            self.assertTrue(all(ep.sources[r.src].event(r.e).kind == "Comment" for r in m.runs if r.src))
        ledger = ep.source_of_lane("translation")
        self.assertTrue(all(ledger.roles[e.i] == "excluded" for e in ledger.events
                            if e.effect == "fx" and e.style.startswith(("OP", "ED"))))
        self.assertTrue(any((sp.data or {}).get("reason") == "song_transcribe" for sp in ledger.spans))
        out = AssDocument.load(row.output)
        self.assertFalse(any(e.effect == "fx" and e.style.startswith(("OP", "ED")) for e in out.events))
        self.assertTrue(any(e.style == "OP.JP.Ruby" for e in out.events))
        self.assertTrue(any(e.style == "ED.JP.Ruby" for e in out.events))
        self.assertIn(r"{\k100}", next(m.text for m in original if m.style == "OP.JP"))
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual((ep.dir / ledger.copy).read_bytes(), before)

    def test_keep_retains_fx_verbatim_and_no_song_readings(self):
        doc = self.song_doc(dense=True, ruby=True)
        self.put("S01E001.ass", doc.serialize())
        row = self.run_fast(BatchOptions(mode="5.2", require_video=False, song_choices={"ED": "keep"})).episodes[0]
        self.assertEqual(row.status, "完成", row.reason)
        self.assertEqual(row.songs, {"OP": "keep", "ED": "keep"})
        ep = Project.open(self.project.root).episode(row.key)
        self.assertFalse(any(g.kind == "song" for g in ep.document.groups))
        out = AssDocument.load(row.output)
        self.assertCountEqual([(e.style, e.effect, e.text, e.start, e.end) for e in out.events if e.effect == "fx"],
                              [(e.style, e.effect, e.text, e.start, e.end) for e in doc.events if e.effect == "fx"])
        self.assertFalse(any(e.style in ("OP.Ruby", "ED.Ruby") for e in out.events))

    def test_51_transcribed_song_pairs_srt_kept_song_only_hides(self):
        doc = self.song_doc(ruby=True)
        doc.events = [e for e in doc.events if e.style != "Japanese"]
        self.put("S01E001.ass", doc.serialize())
        self.put("S01E001.srt", "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n\n"
                               "2\n00:00:05,000 --> 00:00:07,000\n♪夢を見る♪\n\n"
                               "3\n00:00:10,000 --> 00:00:12,000\n♪明日へ行く♪\n\n"
                               "4\n00:00:15,000 --> 00:00:17,000\n♪学校へ♪\n")
        doc.events.append(Event(start=1500, end=1700, style="Screen", text=r"{\pos(50,50)}屏幕"))
        self.put("S01E001.ass", doc.serialize())
        self.project.data["settings"] = {"merge": {"pairing": "same_time"}}
        self.project.save()
        row = self.run_fast(BatchOptions(require_video=False, song_choices={"ED": "keep"})).episodes[0]
        self.assertEqual(row.status, "完成", row.reason)
        self.assertEqual(row.hidden, 1)
        ep = Project.open(self.project.root).episode(row.key)
        self.assertTrue(ep.validate().ok, ep.validate().summary())
        songs = [g for g in ep.document.groups if g.kind == "song"]
        self.assertEqual(len(songs), 1)
        self.assertEqual({m.lane for m in ep.document.members_of(songs[0])}, {"original", "translation"})
        self.assertTrue(any(m.text == "学校へ" for m in ep.document.members.values()))
        self.assertTrue(any(a.member in songs[0].members for a in ep.document.annotations.values()))

    def test_interjection_line_examples_and_content(self):
        for text in ("ウゥ…", "グア…", "グアー！", "うわー！", "ウゥ… ギャ！",
                     "オラ オラ オラ！", "ひょう ひょう！", "「ウゥ」、（ギャ！）",
                     r"{\i1}ウゥ…\Nギャ！"):
            with self.subTest(text=text):
                self.assertTrue(_is_interjection_line(text))
        for text in ("なんだ", "待て！", "ふざけるな", "はい", "そうか", "あっ 待て！",
                     "うわー 学校", "ウゥ… なんだ", "オラ ひょう", "漢", "𠮷", "", "…！？"):
            with self.subTest(text=text):
                self.assertFalse(_is_interjection_line(text))

    def test_unpaired_interjections_preserve_other_groups_and_remove_readings(self):
        doc = Document("cries")
        def member(gid, text, lane="original", role="dialogue", kind="dialogue", source_kind="Dialogue"):
            group = next((g for g in doc.groups if g.id == gid), None)
            if group is None:
                group = Group(gid, kind, [])
                doc.groups.append(group)
            mid = "m" + str(len(doc.members))
            group.members.append(mid)
            doc.members[mid] = Member(mid, gid, "ja" if lane == "original" else "zh-Hans", lane,
                                      role, 1000, 2000, source_kind=source_kind,
                                      runs=[Run(text, ins=mid, b=len(text))])
            return doc.members[mid]
        cry = member("cry", "ウゥ… ギャ！")
        member("paired", "ウゥ…")
        member("paired", "啊", "translation")
        member("cn", "ウゥ…", "translation")
        member("song", "ウゥ…", kind="song")
        member("keep", "ウゥ…", role="keep", kind="keep")
        member("comment", "ウゥ…", source_kind="Comment")
        member("mixed", "ウゥ…")
        member("mixed", "なんだ")
        doc.annotations["a"] = SimpleNamespace(id="a", member=cry.id)
        ep = SimpleNamespace(document=doc, sources={})
        before = {mid: m.text for mid, m in doc.members.items()}
        self.assertEqual(_hide_unpaired_interjections(ep), 1)
        self.assertEqual(cry.text, "")
        self.assertFalse(doc.annotations)
        self.assertEqual({mid: m.text for mid, m in doc.members.items() if mid != cry.id},
                         {mid: text for mid, text in before.items() if mid != cry.id})
        self.assertEqual(_hide_unpaired_interjections(ep), 0)
        Editor(doc, {}).restore_comment(next(iter(doc.comments)))
        self.assertEqual(cry.text, before[cry.id])
        self.assertFalse(doc.tombstones)

    def test_interjection_option_both_modes_report_restore_and_source_hashes(self):
        for mode in ("5.1", "5.2"):
            for enabled in (False, True):
                with self.subTest(mode=mode, enabled=enabled):
                    key = f"S01E00{1 + (mode == '5.2') * 2 + enabled}"
                    cn = self.put(key + ".ass", ass_text([
                        (100, 300, "Chinese", "啊"), (100, 300, "Japanese", "ウゥ…"),
                        (1000, 1100, "Japanese", "オラ オラ オラ！")]
                        if mode == "5.2" else [(100, 300, "Chinese", "啊")]))
                    jp = cn if mode == "5.2" else self.put(key + ".srt",
                        "1\n00:00:01,000 --> 00:00:03,000\nウゥ…\n\n"
                        "2\n00:00:10,000 --> 00:00:11,000\nオラ オラ オラ！\n")
                    before = {p: p.read_bytes() for p in (cn, jp)}
                    result = self.run_fast(BatchOptions(mode=mode, require_video=False,
                                                       hide_unpaired_interjections=enabled))
                    row = next(r for r in result.episodes if r.key == key)
                    self.assertEqual(row.status, "完成", row.reason)
                    self.assertEqual(row.hidden_interjections, int(enabled))
                    self.assertIn(f"隐藏没有中文的叫喊和语气词：{int(enabled)} 行",
                                  result.report.read_text(encoding="utf-8"))
                    ep = Project.open(result.project).episode(key)
                    self.assertTrue(ep.validate().ok, ep.validate().summary())
                    self.assertIn("ウゥ…", [m.text for m in ep.document.members.values()])
                    if enabled:
                        Editor(ep.document, ep.sources).restore_comment(next(iter(ep.document.comments)))
                    self.assertIn("オラ オラ オラ！", [m.text for m in ep.document.members.values()])
                    self.assertTrue(ep.validate().ok, ep.validate().summary())
                    self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_scan_statuses_and_ignored_files(self):
        self.pair()
        self.put("S01E002.mkv")
        self.put("S01E003.ass")
        self.put("S01E003.mp4")
        self.put("S01E004.zip")
        self.put("not-an-episode.ass")
        rows = scan_season(self.season)
        self.assertEqual([r.status for r in rows], ["缺视频", "缺中文字幕", "缺日文字幕"])
        self.assertEqual([r.key for r in rows], ["S01E001", "S01E002", "S01E003"])

    def test_conflicts_list_actual_candidates(self):
        self.pair()
        self.put("S01E001-second.ass")
        status = scan_season(self.season)[0].status
        self.assertTrue(status.startswith("有多个候选："))
        self.assertIn("S01E001.ass", status)
        self.assertIn("S01E001-second.ass", status)

    def test_ja_ass_is_original_and_ssa_is_translation(self):
        self.put("S01E001.mkv")
        self.put("S01E001.ssa")
        self.put("S01E001.ja.ass")
        row = scan_season(self.season)[0]
        self.assertEqual(row.status, "就绪")
        self.assertEqual(row.original.name, "S01E001.ja.ass")

    def test_bilingual_scan_uses_one_file_twice(self):
        cn, _ = self.pair(bilingual=True)
        row = scan_season(self.season, BatchOptions(mode="5.2", require_video=False))[0]
        self.assertEqual(row.status, "就绪")
        self.assertEqual(row.original, cn)
        self.assertEqual(row.translation, cn)

    def test_effect_threshold_and_keep_overrides_song(self):
        doc = AssDocument.parse(ass_text([(100, 200, "OP", r"{\move(1,2,3,4)}A"),
                                         (200, 300, "OP", "B"), (300, 400, "Chinese", "中文")]))
        self.assertEqual(effect_styles(doc), ["OP"])
        classified = classify(doc, {"keep": ["OP"], "dialogue": ["OP", "Chinese"]})
        self.assertEqual([classified.roles[i] for i in (0, 1)], ["keep", "keep"])
        self.assertEqual(classified.roles[2], "dialogue")

    def test_below_half_remains_song(self):
        doc = AssDocument.parse(ass_text([(100, 200, "OP", r"{\pos(1,2)}A"),
                                         (200, 300, "OP", "B"), (300, 400, "OP", "C")]))
        self.assertEqual(effect_styles(doc), [])

    def test_bilingual_kana_fallback_and_no_jp_error(self):
        doc = AssDocument.parse(ass_text([(100, 300, "Chinese", "你好"),
                                         (100, 300, "Japanese", "こんにちは")]))
        cfg, jp = source_roles(doc, "5.2")
        self.assertEqual(jp, ["Japanese"])
        self.assertEqual(cfg["dialogue"], ["Chinese"])
        doc.events = doc.events[:1]
        with self.assertRaisesRegex(ValueError, "日文"):
            source_roles(doc, "5.2")

    def test_filtered_import_preserves_all_events_and_bytes(self):
        cn, _ = self.pair(bilingual=True)
        src = import_source(cn, src_id="o1", lang="ja", lane="original", episode_dir=self.root / "ep",
                            ass_styles=["Japanese"])
        self.assertEqual(len(src.events), 4)
        self.assertEqual([src.roles[i] for i in range(4)], ["excluded", "excluded", "dialogue", "dialogue"])
        self.assertEqual((self.root / "ep" / src.copy).read_bytes(), cn.read_bytes())
        self.assertEqual([s.e for s in src.spans if (s.data or {}).get("reason") == "style_filter"], [0, 1])

    def test_empty_filter_selects_no_styles(self):
        cn, _ = self.pair(bilingual=True)
        src = import_source(cn, src_id="o1", lang="ja", lane="original", episode_dir=self.root / "ep", ass_styles=[])
        self.assertEqual(set(src.roles.values()), {"excluded"})

    def test_51_pipeline_effects_hidden_recoverable_report_hashes(self):
        cn, jp = self.pair()
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (cn, jp)}
        result = self.run_fast(BatchOptions(require_video=False, song_choices={"OP": "keep"}))
        row = result.episodes[0]
        self.assertEqual(row.status, "完成", row.reason)
        self.assertEqual(row.effects, 1)
        self.assertEqual(row.hidden, 1)
        self.assertEqual(row.counts["both"], 1)
        output = AssDocument.load(row.output)
        self.assertIn(r"{\pos(320,100)\kf20}特效", [e.text for e in output.events])
        self.assertNotIn("うた", [e.text for e in output.events])
        self.assertEqual({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}, before)
        self.assertIn("S01E001：完成", result.report.read_text(encoding="utf-8"))
        ep = Project.open(result.project).episode("S01E001")
        self.assertTrue(ep.validate().ok, ep.validate().summary())
        comment = next(c for c in ep.document.comments.values() if c.kind == "user_hidden")
        Editor(ep.document, ep.sources).restore_comment(comment.id)
        self.assertIn("うた", [m.text for m in ep.document.members.values()])
        self.assertTrue(ep.validate().ok, ep.validate().summary())

    def test_52_split_no_duplicate_cn_or_jp_and_ledger_valid(self):
        self.pair(bilingual=True)
        result = self.run_fast(BatchOptions(mode="5.2", require_video=False))
        row = result.episodes[0]
        self.assertEqual(row.status, "完成", row.reason)
        ep = Project.open(result.project).episode(row.key)
        self.assertTrue(ep.validate().ok, ep.validate().summary())
        out = AssDocument.load(row.output)
        self.assertEqual(sum(e.text == "你好" for e in out.events), 1)
        self.assertEqual(sum(e.text == "こんにちは" for e in out.events), 1)
        self.assertAlmostEqual(ep.project.template().style("JP.Ruby").fontsize, 48 * .8)
        self.assertEqual(ep.project.template().style("JP").fontname, "Arial")
        ep.reanalyze_sources()
        self.assertTrue(ep.validate().ok, ep.validate().summary())

    def test_52_import_bypasses_aligner_and_preserves_one_sided_lines(self):
        cn, _ = self.pair(bilingual=True)
        before = cn.read_bytes()
        with patch("subkit.core.merge.Aligner2", side_effect=AssertionError("must not align bilingual source")):
            result = self.run_fast(BatchOptions(mode="5.2", require_video=False))
        row = result.episodes[0]
        self.assertEqual(row.status, "完成", row.reason)
        self.assertEqual((row.counts["both"], row.counts["original_only"], row.counts["translation_only"]), (1, 1, 0))
        self.assertEqual(row.hidden, 0)
        ep = Project.open(result.project).episode(row.key)
        self.assertEqual(ep.data["merge"]["settings"]["pairing"], "same_time")
        self.assertEqual(ep.data["merge"]["log"], ["按相同时间配对：配对 1 组，只有中文 0 行，只有日文 1 行"])
        self.assertTrue(ep.validate().ok, ep.validate().summary())
        self.assertEqual(cn.read_bytes(), before)

    def test_outputs_never_overwrite_and_target_object_is_untouched(self):
        self.pair()
        out_dir = self.season / "DuoSub"
        out_dir.mkdir()
        first = out_dir / "S01E001.zh-ja.ass"
        first.write_bytes(b"existing")
        before = copy.deepcopy(self.project.data)
        result = self.run_fast(BatchOptions(require_video=False))
        self.assertEqual(result.episodes[0].status, "完成", result.episodes[0].reason)
        self.assertEqual(first.read_bytes(), b"existing")
        self.assertEqual(result.episodes[0].output.name, "S01E001.zh-ja (2).ass")
        self.assertEqual(result.project, self.project.root)
        self.assertFalse((out_dir / "工程").exists())
        self.assertEqual(self.project.data, before)
        self.assertEqual(Project.open(result.project).episode_ids(), ["S01E001"])

    def test_readings_are_generated_by_merge_and_exported(self):
        self.pair()
        self.put("S01E001.srt", "1\n00:00:01,000 --> 00:00:03,000\n学校へ行く\n")
        result = self.run_fast(BatchOptions(require_video=False))
        row = result.episodes[0]
        self.assertEqual(row.status, "完成", row.reason)
        ep = Project.open(result.project).episode(row.key)
        self.assertTrue(ep.document.annotations)
        output = AssDocument.load(row.output)
        self.assertTrue(any(e.style.endswith(".Ruby") for e in output.events))
        self.assertTrue(ep.validate().ok, ep.validate().summary())

    def test_cancel_between_episodes_keeps_finished_output(self):
        self.pair()
        self.pair("S01E002")
        steps = []
        def progress(i, n, key, step):
            steps.append((i, n, key, step))
        result = self.run_fast(BatchOptions(require_video=False), progress=progress,
                               cancel_check=lambda: bool(list((self.season / "DuoSub").glob("*.ass"))))
        self.assertTrue(result.cancelled)
        self.assertEqual([r.status for r in result.episodes], ["完成", "跳过"])
        self.assertEqual(result.episodes[1].reason, "已取消")
        self.assertEqual([s[3] for s in steps], ["读取字幕", "合并", "读音", "导出"])
        self.assertTrue(result.episodes[0].output.exists())

    def test_rerun_only_adds_new_keys_and_preserves_existing_bytes(self):
        self.pair()
        result = self.run_fast(BatchOptions(require_video=False))
        ep = Project.open(result.project).episode("S01E001")
        editor = Editor(ep.document, ep.sources)
        editor.set_group_time(ep.document.groups[0].id, start=1200)
        ep.save()
        before = {p.relative_to(ep.dir): p.read_bytes() for p in ep.dir.rglob("*") if p.is_file()}
        template = Project.open(result.project).template_path().read_bytes()
        output = result.episodes[0].output.read_bytes()
        self.pair("S01E002")
        again = self.run_fast(BatchOptions(require_video=False))
        self.assertEqual([r.status for r in again.episodes], ["跳过", "完成"])
        self.assertEqual(again.episodes[0].reason, "工程里已有这一集，未改动")
        self.assertEqual({p.relative_to(ep.dir): p.read_bytes() for p in ep.dir.rglob("*") if p.is_file()}, before)
        self.assertEqual(Project.open(result.project).template_path().read_bytes(), template)
        self.assertEqual(result.episodes[0].output.read_bytes(), output)
        self.assertEqual(Project.open(result.project).episode_ids(), ["S01E001", "S01E002"])

    def test_effect_overlap_preserves_paired_dialogue_and_half_threshold(self):
        self.pair(bilingual=True)
        result = self.run_fast(BatchOptions(mode="5.2", require_video=False))
        self.assertEqual(result.episodes[0].hidden, 0)      # 5.2 JP lines were shown in the source file
        ep = Project.open(result.project).episode("S01E001")
        paired = next(m for m in ep.document.members.values() if m.lane == "original" and m.text == "こんにちは")
        self.assertEqual(_hide_effect_overlap(ep, [(paired.start, paired.end)]), 0)
        self.assertEqual(paired.text, "こんにちは")
        self.assertEqual(_hide_effect_overlap(ep, [(5000, 5249), (5000, 5249)]), 0)
        self.assertEqual(_hide_effect_overlap(ep, [(5000, 5250), (5250, 5500)]), 1)

    def test_adoption_protects_manual_confirmed_and_undo_redo(self):
        doc = Document("adopt")
        for i, times in enumerate([(3600, 5000), (5600, 6500), (7000, 7500)]):
            gid, mid = f"g{i}", f"m{i}"
            doc.groups.append(Group(gid, "dialogue", [mid]))
            doc.members[mid] = Member(mid, gid, "ja", "original", "dialogue", *times,
                                      runs=[Run("line", ins=mid, b=4)])
        doc.groups[1].state = "confirmed"
        doc.members["m2"].extra["time_manual"] = True
        before = copy.deepcopy(doc.members)
        curve = np.zeros(800, np.float32); curve[355:500] = .8
        features = AudioFeatures(np.zeros(800), np.zeros(800), np.zeros((800,128)), np.zeros(800), snap_activity=curve)
        editor = Editor(doc, {})
        count = adopt_suggestions(editor, features, BatchOptions(lead_in_ms=270, lead_out_ms=160))
        self.assertEqual(count, 1)
        self.assertEqual((doc.members["m0"].start, doc.members["m0"].end), (3280, 5160))
        self.assertEqual(doc.members["m0"].extra["time_src"], {"start": "adopted", "end": "adopted"})
        self.assertEqual(doc.members["m1"], before["m1"])
        self.assertEqual(doc.members["m2"], before["m2"])
        self.assertEqual(len(editor.history.done), 1)
        editor.history.undo(); self.assertEqual(doc.members, before)
        editor.history.redo(); self.assertEqual(doc.members["m0"].start, 3280)

    def test_feature_provider_once_per_episode_and_shared_cache(self):
        self.pair()
        self.put("S01E001.mkv")
        curve = np.zeros(800, np.float32); curve[120:270] = .8
        features = AudioFeatures(np.zeros(800), np.zeros(800), np.zeros((800,128)), np.zeros(800), snap_activity=curve)
        calls = []
        def provider(video, track, cache):
            calls.append((video, track, cache))
            return features
        result = self.run_fast(BatchOptions(adopt_suggested_times=True, lead_in_ms=100, lead_out_ms=100),
                               feature_provider=provider)
        self.assertEqual(result.episodes[0].status, "完成", result.episodes[0].reason)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], Project.open(result.project).cache_dir / "waveform")
        self.assertGreater(result.episodes[0].adopted, 0)

    def test_audio_cache_decodes_once_and_waveform_reads_identical_format(self):
        from subkit_app.views.waveform import _waveform_cache_file, PeakJob
        video = self.put("video.mp4")
        raw = np.zeros(16000, dtype="<i2").tobytes()
        cache = self.root / "waveform"
        with patch("subkit.proc.run", return_value=subprocess.CompletedProcess([], 0, raw, b"")) as runner:
            first = load_features(video, None, cache)
            second = load_features(video, None, cache)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(_waveform_cache_file(video, None, cache), waveform_cache_file(video, None, cache))
        np.testing.assert_array_equal(first.spectrogram, second.spectrogram)
        job = PeakJob(video, None, cache)
        seen = []; job.signals.done.connect(lambda f, g: seen.append(f))
        with patch("subkit_app.views.waveform._decode_samples", side_effect=AssertionError("must reuse cache")):
            job.run()
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0].mins, first.mins)

    def test_assfonts_success_final_name_no_extra_duosub_copies(self):
        path = self.put("output.ass", ass_text([(100,200,"Chinese","Hello")]))
        row = EpisodeResult("S01E001", "完成")
        def runner(cmd, **kw):
            self.assertIn("-d", cmd)
            self.assertIn("-s", cmd)
            src = Path(cmd[cmd.index("-i") + 1]); dest = Path(cmd[cmd.index("-o") + 1])
            (dest / "Arial_subset.ttf").write_bytes(Path(r"C:\Windows\Fonts\arial.ttf").read_bytes())
            return subprocess.CompletedProcess(cmd, 0, b'[INFO] embedded', b'')
        with patch("subkit.proc.run", side_effect=runner):
            embed_fonts(path, [], self.root / "assfonts.exe", row)
        self.assertEqual(len(row.fonts_embedded), 1)
        self.assertIn("Arial_subset.ttf", row.fonts_embedded[0])
        self.assertIn("Arial", row.fonts_embedded[0])
        self.assertIn("Arial_subset.ttf", path.read_text(encoding="utf-8-sig"))
        self.assertEqual([p.name for p in self.season.iterdir()], ["output.ass"])

    def test_assfonts_zero_exit_missing_fonts_carries_source_entries_verbatim(self):
        lines = ["fontname: MissingFont_0.ttf", "!!ABCD", "  EFGH  "]
        source = AssDocument.parse(ass_text([(100,200,"Chinese","Hello")]) +
                                  "\n[Fonts]\n" + "\n".join(lines))
        doc = AssDocument.parse(ass_text([(100,200,"Chinese","Hello"), (100,200,"Japanese","Hi")]))
        doc.style("Chinese").set("Fontname", "MissingFont")
        path = self.put("output.ass", doc.serialize())
        row = EpisodeResult("S01E001", "完成")
        (self.root / "fonts.json").write_text('[{"families":["MissingFont"],"path":"Z:/gone/font.ttf"}]',encoding="utf-8")
        log = b'[WARN] Missing the font: "MissingFont" (700,0)\n[ERROR] Found missing fonts.'
        calls = []
        def runner(cmd, **kw):
            staged = AssDocument.load(cmd[cmd.index("-i") + 1])
            calls.append(staged)
            if len(calls) == 1:
                return subprocess.CompletedProcess([], 0, log, b"")
            self.assertNotIn("MissingFont", [s.fontname for s in staged.styles])
            dest = Path(cmd[cmd.index("-o") + 1])
            (dest / "Arial.ttf").write_bytes(Path(r"C:\Windows\Fonts\arial.ttf").read_bytes())
            return subprocess.CompletedProcess([], 0, b"subset complete", b"")
        with patch("subkit.proc.run", side_effect=runner):
            embed_fonts(path, [source, source], self.root / "assfonts.exe", row)
        self.assertEqual(len(calls), 2)
        self.assertEqual(row.fonts_missing, {"MissingFont": ["Z:/gone/font.ttf"]})
        self.assertEqual(row.fonts_carried, ["MissingFont_0.ttf"])
        self.assertIn("fontname: MissingFont_0.ttf\n!!ABCD\n  EFGH  ", path.read_text(encoding="utf-8-sig"))
        self.assertEqual(len(AssDocument.load(path).events), 2)
        self.assertEqual(AssDocument.load(path).styles, doc.styles)
        self.assertTrue(row.warnings)  # Invalid attachment names are explicitly reported.

    def test_inaccessible_library_file_is_excluded_like_a_missing_font(self):
        doc = AssDocument.parse(ass_text([(100,200,"Chinese","Hello"), (100,200,"Japanese","Hi")]))
        doc.style("Japanese").set("Fontname", "Unmapped Font")
        path = self.put("output.ass", doc.serialize())
        row = EpisodeResult("S01E001", "完成")
        log = (b'[INFO] Found font: "Unmapped Font" (400,0) --> "Z:\\fonts\\unmapped.otf"[0]\n'
               b'[ERROR] "Z:\\fonts\\unmapped.otf" is inaccessible\n')
        calls = []
        def runner(cmd, **kw):
            staged = AssDocument.load(cmd[cmd.index("-i") + 1])
            calls.append(staged)
            if len(calls) == 1:
                return subprocess.CompletedProcess([], 0, log, b"")
            self.assertNotIn("Unmapped Font", [s.fontname for s in staged.styles])
            dest = Path(cmd[cmd.index("-o") + 1])
            (dest / "Arial.ttf").write_bytes(Path(r"C:\Windows\Fonts\arial.ttf").read_bytes())
            return subprocess.CompletedProcess([], 0, b"subset complete", b"")
        with patch("subkit.proc.run", side_effect=runner):
            embed_fonts(path, [], self.root / "assfonts.exe", row)
        self.assertEqual(len(calls), 2)
        self.assertIn("Unmapped Font", row.fonts_missing)
        self.assertEqual(len(AssDocument.load(path).events), 2)

    def test_attachment_family_name_decode_and_recovery(self):
        import struct
        family = "MissingFont".encode("utf-16-be")
        name = struct.pack(">HHH", 0, 1, 18) + struct.pack(">HHHHHH", 3, 1, 1033, 1, len(family), 0) + family
        sfnt = struct.pack(">IHHHH", 0x10000, 1, 0, 0, 0) + struct.pack(">4sIII", b"name", 0, 28, len(name)) + name
        encoded = ""
        for i in range(0, len(sfnt), 3):
            chunk = sfnt[i:i+3]; n = int.from_bytes(chunk.ljust(3, b"\0"), "big")
            encoded += "".join(chr(((n >> shift) & 63) + 33) for shift in (18,12,6,0))[:len(chunk)+1]
        lines = ["fontname: arbitrary-file.ttf", encoded]
        self.assertEqual(_attachment_families(lines), {"missingfont"})
        source = AssDocument.parse(ass_text([(100,200,"Chinese","Hello")]) + "\n[Fonts]\n" + "\n".join(lines))
        output = AssDocument.parse(ass_text([(100,200,"Chinese","Hello")]))
        output.style("Chinese").set("Fontname", "@MISSINGFONT")
        path = self.put("out.ass", output.serialize())
        row = EpisodeResult("S01E001", "完成")
        log = b'[WARN] Missing the font: "@MISSINGFONT"\n[WARN] Missing the font: "Arial"\n'
        with patch("subkit.proc.run", return_value=subprocess.CompletedProcess([], 0, log, b"")) as runner:
            embed_fonts(path, [source], self.root / "assfonts.exe", row)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(row.fonts_recovered, ["MISSINGFONT"])
        self.assertEqual(AssDocument.load(path).raw_section("[Fonts]")[:2], lines)

    def test_retry_excludes_each_reported_font_beyond_three_attempts(self):
        doc = AssDocument.parse(ass_text([(100, 200, "Chinese",
            r"A{\fnMissing0}B{\fnMissing1}C{\fnMissing2}D{\fnMissing3}E")] ))
        path = self.put("output.ass", doc.serialize())
        row = EpisodeResult("S01E001", "完成")
        def runner(cmd, **kw):
            staged = AssDocument.load(cmd[cmd.index("-i") + 1])
            for i in range(4):
                family = f"Missing{i}"
                if family in staged.serialize():
                    log = f'[WARN] Missing the font: "{family}"\n'.encode("utf-8")
                    return subprocess.CompletedProcess(cmd, 0, log, b"")
            dest = Path(cmd[cmd.index("-o") + 1])
            (dest / "Arial.ttf").write_bytes(Path(r"C:\Windows\Fonts\arial.ttf").read_bytes())
            return subprocess.CompletedProcess(cmd, 0, b"subset complete", b"")
        with patch("subkit.proc.run", side_effect=runner) as tool:
            embed_fonts(path, [], self.root / "assfonts.exe", row)
        self.assertEqual(tool.call_count, 5)
        self.assertEqual(set(row.fonts_missing), {f"Missing{i}" for i in range(4)})
        self.assertEqual(len(row.fonts_embedded), 1)
        self.assertFalse(row.warnings, row.warnings)

    def subset_arial(self, chars="ABC"):
        from fontTools import subset
        font = subset.load_font(r"C:\Windows\Fonts\arial.ttf", subset.Options())
        try:
            subsetter = subset.Subsetter()
            subsetter.populate(text=chars)
            subsetter.subset(font)
            target = self.root / "small.ttf"
            font.save(target)
        finally:
            font.close()
        return ["fontname: small.ttf", *_ass_encode(target.read_bytes())]

    def test_assfonts_family_wins_over_carried_subset(self):
        lines = self.subset_arial()
        source = AssDocument.parse(ass_text([]) + "\n[Fonts]\n" + "\n".join(lines))
        path = self.put("output.ass", ass_text([(100, 200, "Chinese", "ABCD")]))
        row = EpisodeResult("S01E001", "完成")
        def runner(cmd, **kw):
            staged = AssDocument.load(cmd[cmd.index("-i") + 1])
            self.assertEqual(staged.events[0].text, "ABCD")
            self.assertIsNone(staged.raw_section("[Fonts]"))
            dest = Path(cmd[cmd.index("-o") + 1])
            (dest / "complete.ttf").write_bytes(Path(r"C:\Windows\Fonts\arial.ttf").read_bytes())
            return subprocess.CompletedProcess(cmd, 0, b"subset complete", b"")
        with patch("subkit.proc.run", side_effect=runner) as tool:
            embed_fonts(path, [source], self.root / "assfonts.exe", row)
        self.assertEqual(tool.call_count, 1)
        self.assertEqual([name for name, _ in _font_entries(AssDocument.load(path))], ["complete.ttf"])
        self.assertFalse(row.fonts_carried)
        self.assertFalse(row.warnings, row.warnings)

    def test_carried_subset_reports_exact_missing_glyphs_and_ruby(self):
        lines = self.subset_arial()
        source = AssDocument.parse(ass_text([]) + "\n[Fonts]\n" + "\n".join(lines))
        doc = AssDocument.parse(ass_text([(100, 200, "Chinese", r"ABC{\fnArial}D\N\hD{ignored}"),
                                         (100, 200, "Ruby", "E")], names=("Chinese", "Ruby")))
        last = doc.style("Chinese").copy()
        doc.style("Chinese").set("Fontname", "Times New Roman")
        doc.styles.append(last)
        path = self.put("output.ass", doc.serialize())
        row = EpisodeResult("S01E001", "完成")
        log = b'[WARN] Missing the font: "Arial"\n'
        with patch("subkit.proc.run", return_value=subprocess.CompletedProcess([], 0, log, b"")):
            embed_fonts(path, [source], self.root / "assfonts.exe", row)
        self.assertEqual(row.fonts_carried, ["small.ttf"])
        self.assertEqual(row.fonts_recovered, ["Arial"])
        self.assertFalse(row.fonts_embedded)
        self.assertEqual(row.warnings, ["Arial：字形不全：2 个字（DE）"])
        self.assertEqual(_ass_decode(_font_entries(AssDocument.load(path))[0][1][1:]), _ass_decode(lines[1:]))
        self.assertEqual(AssDocument.load(path).events, doc.events)

    def test_coverage_font_runs_resets_drawings_and_limit(self):
        doc = AssDocument.parse(ass_text([(100, 200, "Chinese",
            r"A{\fnOther}B{\rJapanese}C{\fn}D{\p1}m 0 0 l 1 1{\p0}E\N\h")]))
        doc.style("Chinese").set("Fontname", "Arial")
        doc.style("Japanese").set("Fontname", "Times New Roman")
        gaps = _carried_gaps(doc, {"arial": {ord("A")}, "other": set(), "times new roman": set()})
        self.assertEqual(gaps, ["Arial：字形不全：2 个字（DE）", "Other：字形不全：1 个字（B）",
                                "Times New Roman：字形不全：1 个字（C）"])
        doc.events = [Event(kind="Dialogue", style="Chinese", text="abcdefghijklmnopqrstuvwxyz")]
        self.assertEqual(_carried_gaps(doc, {"arial": set()}),
                         ["Arial：字形不全：26 个字（abcdefghijklmnopqrst）"])

    def test_ass_encoding_roundtrip_boundaries_and_real_ttf(self):
        for length in (0, 1, 2, 3, 59, 60, 61, 62, 255, 256):
            data = bytes(range(256))[:length]
            lines = _ass_encode(data)
            self.assertEqual(_ass_decode(lines), data)
            self.assertTrue(all(len(line) <= 80 for line in lines))
        self.assertEqual(_ass_encode(b"\0"), ["!!"])
        self.assertEqual(_ass_encode(b"\xff\xff"), ["``]"])
        with self.assertRaises(ValueError):
            _ass_decode(["!"])
        with self.assertRaises(ValueError):
            _ass_decode(["~~"])
        copied = self.root / "arial.ttf"
        copied.write_bytes(Path(r"C:\Windows\Fonts\arial.ttf").read_bytes())
        lines = ["fontname: arial.ttf", *_ass_encode(copied.read_bytes())]
        self.assertEqual(_ass_decode(lines[1:]), copied.read_bytes())
        self.assertIn("Arial", _attachment_names(lines))

    def test_ttc_checks_all_faces(self):
        from fontTools.ttLib import TTFont, TTCollection
        collection = TTCollection()
        collection.fonts = [TTFont(r"C:\Windows\Fonts\arial.ttf"), TTFont(r"C:\Windows\Fonts\times.ttf")]
        for table in collection.fonts[0]["cmap"].tables:
            table.cmap.pop(ord("Z"), None)
        path = self.root / "faces.ttc"
        try:
            collection.save(path)
        finally:
            collection.close()
        lines = ["fontname: faces.ttc", *_ass_encode(path.read_bytes())]
        coverage = _attachment_cmaps(lines)
        self.assertIn(ord("A"), coverage["arial"])
        self.assertIn(ord("B"), coverage["times new roman"])
        self.assertNotIn(ord("Z"), coverage["arial"])
        self.assertIn(ord("Z"), coverage["times new roman"])
        names = _attachment_families(lines)
        self.assertIn("arial", names)
        self.assertIn("times new roman", names)

    def test_subset_split_inline_vertical_reset_and_original_unchanged(self):
        doc = AssDocument.parse(ass_text([(100,200,"Chinese","covered"),
            (100,200,"Japanese",r"a{\fn@Covered}b{\fnOther}c{\rChinese}d"),
            (100,200,"Chinese",r"a{\fnOther}b")]))
        doc.style("Chinese").set("Fontname", "@COVERED")
        doc.style("Japanese").set("Fontname", "Uncovered")
        before = doc.serialize()
        subset = _subset_document(doc, {"covered"})
        self.assertEqual(len(subset.events), 2)
        self.assertNotIn("covered", [s.fontname.casefold() for s in subset.styles])
        self.assertNotIn(r"\fn@Covered", subset.serialize())
        self.assertNotIn(r"\rChinese", subset.serialize())
        self.assertIn(r"\fnOther", subset.serialize())
        self.assertEqual(doc.serialize(), before)

    def test_carried_dedup_filename_and_content_fonts_before_events(self):
        first = ["fontname: same.ttf", *_ass_encode(Path(r"C:\Windows\Fonts\arial.ttf").read_bytes())]
        second = ["fontname: same.ttf", *_ass_encode(Path(r"C:\Windows\Fonts\times.ttf").read_bytes())]
        source = AssDocument.parse(ass_text([]) + "\n[Fonts]\n" + "\n".join(first + second))
        path = self.put("output.ass", ass_text([(100,200,"Chinese","Hello")]))
        row = EpisodeResult("S01E001", "完成")
        log = b'[WARN] Missing the font: "Arial"\n'
        with patch("subkit.proc.run", return_value=subprocess.CompletedProcess([], 0, log, b"")):
            embed_fonts(path, [source, source], self.root / "assfonts.exe", row)
        output = AssDocument.load(path)
        entries = _font_entries(output)
        self.assertEqual(len(entries), 2)
        self.assertEqual(_ass_decode(entries[0][1][1:]), Path(r"C:\Windows\Fonts\arial.ttf").read_bytes())
        self.assertEqual(_ass_decode(entries[1][1][1:]), Path(r"C:\Windows\Fonts\times.ttf").read_bytes())
        self.assertLess(output.serialize().index("[Fonts]"), output.serialize().index("[Events]"))
        self.assertEqual(set(row.fonts_missing), {"Arial"})

    def test_fake_cli_subset_and_abort_are_real_child_processes(self):
        fake = self.root / "fake_assfonts.py"
        fake.write_text("""import sys, pathlib, shutil
args=sys.argv[1:]
assert '-s' in args
src=pathlib.Path(args[args.index('-i')+1])
out=pathlib.Path(args[args.index('-o')+1])
if 'AbsentFont' in src.read_text(encoding='utf-8-sig'):
    print('[WARN] Missing the font: "AbsentFont" (400,0)')
    print('[ERROR] Found missing fonts')
else:
    shutil.copyfile(r'C:\\Windows\\Fonts\\arial.ttf',out/'fake-subset.ttf')
    print('[INFO] subset complete')
""", encoding="utf-8")
        doc = AssDocument.parse(ass_text([(100,200,"Chinese","Hello"), (100,200,"Japanese","Hi")]))
        doc.style("Chinese").set("Fontname", "AbsentFont")
        path = self.put("fake.ass", doc.serialize())
        row = EpisodeResult("S01E001", "完成")
        from subkit import proc
        run = proc.run
        with patch("subkit.proc.run", side_effect=lambda cmd, **kw: run([sys.executable, str(fake), *cmd[1:]], **kw)):
            embed_fonts(path, [], self.root / "fake.exe", row)
        self.assertEqual(row.fonts_missing, {"AbsentFont": []})
        self.assertEqual(len(row.fonts_embedded), 1)
        self.assertEqual(AssDocument.load(path).events, doc.events)
        self.assertFalse(row.warnings, row.warnings)

    def test_second_abort_preserves_plain_bytes(self):
        path = self.put("failed.ass", ass_text([(100,200,"Chinese","Hello")]))
        before = path.read_bytes()
        row = EpisodeResult("S01E001", "完成")
        log = b'[WARN] Missing the font: "Unknown"\n[ERROR] Found missing fonts'
        with patch("subkit.proc.run", return_value=subprocess.CompletedProcess([], 0, log, b"")) as runner:
            embed_fonts(path, [], self.root / "fake.exe", row)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue(row.warnings)
        self.assertFalse(row.fonts_carried)

    def test_failed_atomic_write_preserves_plain_and_reports_no_attachments(self):
        lines = ["fontname: Arial.ttf", *_ass_encode(Path(r"C:\Windows\Fonts\arial.ttf").read_bytes())]
        source = AssDocument.parse(ass_text([]) + "\n[Fonts]\n" + "\n".join(lines))
        path = self.put("atomic.ass", ass_text([(100,200,"Chinese","Hello")]))
        before = path.read_bytes()
        row = EpisodeResult("S01E001", "完成")
        with patch.object(AssDocument, "save", side_effect=OSError("write denied")):
            embed_fonts(path, [source], self.root / "fake.exe", row)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(row.fonts_embedded)
        self.assertFalse(row.fonts_carried)
        self.assertTrue(row.warnings)

    def test_adoption_uses_updated_previous_neighbour(self):
        doc = Document("neighbours")
        for i in range(2):
            gid, mid = f"g{i}", f"m{i}"
            doc.groups.append(Group(gid, "dialogue", [mid]))
            doc.members[mid] = Member(mid, gid, "ja", "original", "dialogue", 1000 + i*2000, 2000+i*2000,
                                      runs=[Run("line", ins=mid, b=4)])
        features = AudioFeatures(np.zeros(500), np.zeros(500), np.zeros((500,128)), np.zeros(500), snap_activity=np.zeros(500))
        with patch("subkit.core.batch.suggest_group_times", side_effect=[(1100, 2100),(3100,4100)]) as suggest:
            self.assertEqual(adopt_suggestions(Editor(doc, {}), features, BatchOptions()), 2)
        self.assertEqual(suggest.call_args_list[1].kwargs["previous"], (1100,2100))

    def test_assfonts_failure_keeps_plain_bytes(self):
        path = self.put("output.ass", ass_text([(100,200,"Chinese","Hello")]))
        before = path.read_bytes(); row = EpisodeResult("S01E001", "完成")
        with patch("subkit.proc.run", side_effect=OSError("tool failed")):
            embed_fonts(path, [], self.root / "missing.exe", row)
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue(row.warnings)

    def test_failure_does_not_stop_next_episode(self):
        self.put("S01E001.ass", "bad")
        self.put("S01E001.srt", "bad")
        self.pair("S01E002")
        result = self.run_fast(BatchOptions(require_video=False))
        self.assertEqual([r.status for r in result.episodes], ["失败", "完成"])

    def test_missing_inputs_are_reported(self):
        self.put("S01E001.mkv")
        result = run_season(self.season, self.project.root)
        self.assertEqual(result.episodes[0].reason, "缺中文字幕")
        self.assertIn("缺中文字幕", result.report.read_text(encoding="utf-8"))
        self.assertEqual(result.project, self.project.root)

    def test_option_limits_and_name_suffix(self):
        with self.assertRaises(ValueError):
            BatchOptions(mode="unknown")
        path = self.put("out.ass")
        self.put("out (2).ass")
        self.assertEqual(unused_path(path).name, "out (3).ass")

    def test_output_uses_project_styles_and_ruby_size(self):
        self.pair()
        self.put("S01E001.srt", "1\n00:00:01,000 --> 00:00:03,000\n学校へ行く\n")
        before = self.project.template_path().read_bytes()
        result = self.run_fast(BatchOptions(require_video=False))
        row = result.episodes[0]
        self.assertEqual(row.status, "完成", row.reason)
        out = AssDocument.load(row.output)
        for name in ("CN", "JP", "JP.Ruby"):
            self.assertEqual(out.style(name).to_line(out.style_format), self.project.template().style(name).to_line(out.style_format))
        self.assertTrue(any(e.style == "JP.Ruby" for e in out.events))
        self.assertEqual(self.project.template_path().read_bytes(), before)
        self.assertIn("振假名大小按工程模板 JP.Ruby", result.report.read_text(encoding="utf-8"))

    def test_target_must_already_be_a_project(self):
        with self.assertRaises(FileNotFoundError):
            run_season(self.season, self.root / "absent")
        self.assertFalse((self.root / "absent").exists())
        self.assertFalse((self.season / "DuoSub").exists())


if __name__ == "__main__":
    unittest.main()
