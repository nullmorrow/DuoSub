"""S1: source-safe exact copies, scoped exact milliseconds and karaoke timing."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from subkit.ass import AssDocument
from subkit.core import karaoke as kara
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.importer import import_source
from subkit.core.ops import Editor
from subkit.core.people import file_name_speaker
from subkit.core.position import source_positioned
from subkit.core.readings import anchor_of, visible_map
from subkit.core.time_input import format_time, parse_time
from subkit.model.document import Annotation, Comment, Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json, save_document, load_document
from subkit.model.source import Source, SourceEvent
from subkit.model.validate import validate
from subkit.styles_default import default_styles_ass


def fixture(kind="song"):
    doc = Document("E1", groups=[Group("g", kind, ["cn", "ja"])])
    sources = {}
    for mid, text, lang, lane, style in (("cn", "去学校", "zh-Hans", "translation", "CN"),
                                         ("ja", "学校に行く", "ja", "original", "JP")):
        sources[mid] = Source(mid, lang, lane, "srt", "", "", "", 0, "utf-8", "lf",
                              events=[SourceEvent(0, "Dialogue", 1000, 4000, text)])
        m = Member(mid, "g", lang, lane, "song" if kind == "song" else kind, 1000, 4000, style=style,
                   layer=2, margins=[11, 22, 33], effect="test", speaker={"label": "人物", "by": "user"},
                   runs=[Run(text, src=mid, e=0, a=0, b=len(text))],
                   overrides={"an": 8, "pos": [123, 234], "raw": r"\frz10"})
        doc.members[mid] = m
    ids = visible_map(doc.members["ja"])[1]
    doc.annotations["a"] = Annotation("a", "ruby", "ja", anchor_of(ids[:2]), "学校", "がっこう",
                                      status="uncertain", origin="analyzer", parts=[[0, 2, "がっこう"]],
                                      layout={"gap": 3}, candidates=["がっこう"])
    doc.members["ja"].format = [{"anchor": anchor_of(ids[2:4]), "props": {"c1": "&HFF0000&", "fs": 40}}]
    cnids = visible_map(doc.members["cn"])[1]
    doc.annotations["p"] = Annotation("p", "pinyin", "cn", anchor_of(cnids[1:2]), "学", "xué",
                                      status="confirmed", origin="manual", parts=[[0, 1, "xué"]])
    return doc, sources, Editor(doc, sources)


def snapshot(doc):
    value = copy.deepcopy(document_to_json(doc))
    value.pop("revision")
    return value


class DuplicateGroups(unittest.TestCase):
    def test_copy_all_fields_new_identity_and_ledger(self):
        doc, sources, ed = fixture()
        group = doc.groups[0]
        group.state, group.flags = "confirmed", ["hidden", "marked", "boundary_uncertain"]
        group.extra = {"mark_note": "复查", "song_family": "OP", "unknown": {"x": 1}}
        doc.comments["c"] = Comment("c", "g", "note", runs=[Run("注", ins="i_note", b=1)], position={"after": "ja"})
        group.comments = ["c"]
        before, source_before = snapshot(doc), copy.deepcopy(sources)
        copied = ed.duplicate_groups(["g"])
        self.assertEqual(len(copied), 1)
        new = doc.group(copied[0])
        self.assertEqual([g.id for g in doc.groups], ["g", new.id])
        self.assertEqual(new.kind, group.kind)
        self.assertEqual(new.state, "auto")
        self.assertEqual(new.flags, ["boundary_uncertain"])
        self.assertNotIn("mark_note", new.extra)
        self.assertEqual(new.extra["song_family"], "OP")
        for oldmid, mid in zip(group.members, new.members):
            old, m = doc.members[oldmid], doc.members[mid]
            for attr in ("text", "lang", "lane", "role", "start", "end", "style", "layer", "margins", "effect",
                         "speaker", "overrides", "source_kind"):
                self.assertEqual(getattr(m, attr), getattr(old, attr), attr)
            self.assertTrue(all(r.src is None and r.ins for r in m.runs))
            self.assertTrue(set(visible_map(m)[1]).isdisjoint(visible_map(old)[1]))
            self.assertEqual([a.text for a in doc.annotations_of(m.id)], [a.text for a in doc.annotations_of(old.id)])
            self.assertEqual([a.status for a in doc.annotations_of(m.id)], [a.status for a in doc.annotations_of(old.id)])
        self.assertEqual(doc.comments[new.comments[0]].position, {"after": new.members[1]})
        self.assertTrue(validate(doc, sources).ok, validate(doc, sources).summary())
        self.assertFalse(any(i.code == "annotation.detached" for i in validate(doc, sources).issues))
        self.assertEqual(sources, source_before)
        self.assertEqual(ed.history.done[-1].label, "复制一组")
        self.assertTrue(ed.history.done[-1].keeps_text)
        after = snapshot(doc)
        ed.history.undo()
        self.assertEqual(snapshot(doc), before)
        ed.history.redo()
        self.assertEqual(snapshot(doc), after)

    def test_multiselect_document_order_repeat_copy_and_mutable_independence(self):
        doc, _, ed = fixture("dialogue")
        doc.groups.append(Group("g2", "sign", []))
        doc.structure_changed()
        new = ed.duplicate_groups(["g2", "g", "g", "missing"])
        self.assertEqual([g.id for g in doc.groups], ["g", new[0], "g2", new[1]])
        nested = ed.duplicate_groups([new[0]])[0]
        m1, m2 = doc.members[doc.group(new[0]).members[1]], doc.members[doc.group(nested).members[1]]
        m2.overrides["pos"][0] = 999
        m2.format[0]["props"]["fs"] = 99
        self.assertEqual(m1.overrides["pos"][0], 123)
        self.assertEqual(m1.format[0]["props"]["fs"], 40)
        self.assertTrue(set(visible_map(m1)[1]).isdisjoint(visible_map(m2)[1]))
        self.assertEqual(len(ed.history.done), 2)
        self.assertEqual(ed.duplicate_groups(["missing"]), [])
        self.assertEqual(len(ed.history.done), 2)

    def test_render_context_survives_run_slice_comment_restore_and_format_guard(self):
        doc, sources, ed = fixture("dialogue")
        m=doc.members["ja"]
        # A source prefix hidden before copying has no second source occurrence.
        cid=ed.hide_range("ja",0,1)
        self.assertIsNotNone(cid)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"document.json"
            save_document(doc,path)
            original=path.read_bytes()
            self.assertEqual(document_to_json(doc)["format_version"],1)
            gid=ed.duplicate_groups(["g"])[0]
            new=doc.group(gid)
            copied=doc.members[new.members[1]]
            self.assertEqual(copied.runs[0].slice(0,1).render_src,"ja")
            save_document(doc,path)
            self.assertEqual(path.with_name("document.v1.bak.json").read_bytes(),original)
            self.assertEqual(document_to_json(doc)["format_version"],2)
            loaded=load_document(path)
            self.assertEqual(snapshot(loaded),snapshot(doc))
            # This is the exact old reader's version guard, not an old-writer simulation.
            from subkit.model import serialize
            from unittest.mock import patch
            with patch.object(serialize,"FORMAT_VERSION",1), self.assertRaisesRegex(ValueError,"newer"):
                document_from_json(document_to_json(doc))
            ed.restore_comment(new.comments[0])
            ed.restore_comment(cid)
            self.assertEqual(doc.members[copied.id].text,doc.members["ja"].text)
            self.assertTrue(validate(doc,sources).ok)

    def test_karaoke_boundaries_effect_links_and_independence(self):
        doc, _, ed = fixture()
        for mid in ("ja", "cn"):
            beats = kara.derive(doc, doc.members[mid])
            for i, b in enumerate(beats):
                b.t0, b.t1, b.by = 1100 + i * 200, 1300 + i * 200, "tap"
            if mid == "cn":
                original = kara.effective(doc, doc.members["ja"]).beats[0]
                beats[0].link = {"start": kara.beat_link(original, "start")}
            ed.set_karaoke(mid, beats, effect={"preset": "ko", "params": {"test": 7}})
        key = kara.boundary_key(doc, doc.members["ja"], 0, 1)
        ed.split_beat("ja", key, 1150)
        old = {mid: kara.effective(doc, doc.members[mid]) for mid in ("ja", "cn")}
        before = snapshot(doc)
        new = doc.group(ed.duplicate_groups(["g"])[0])
        for oldmid, mid in zip(("cn", "ja"), new.members):
            view = kara.effective(doc, doc.members[mid])
            self.assertEqual([(b.sung, b.t0, b.t1, b.by) for b in view.beats],
                             [(b.sung, b.t0, b.t1, b.by) for b in old[oldmid].beats])
            self.assertEqual((view.kind, view.params, view.unit, view.state),
                             (old[oldmid].kind, old[oldmid].params, old[oldmid].unit, old[oldmid].state))
            self.assertTrue(all(not b.link for b in view.beats))
            self.assertTrue(kara.tracks_of(doc, mid))
        loaded = document_from_json(document_to_json(doc))
        self.assertEqual(snapshot(loaded), snapshot(doc))
        ed.history.undo()
        self.assertEqual(snapshot(doc), before)

    def test_moved_character_timing_is_copied_without_own_track(self):
        doc, _, ed = fixture()
        beats = kara.derive(doc, doc.members["ja"])
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = 1000 + i * 100, 1100 + i * 100, "manual"
        ed.set_karaoke("ja", beats)
        track = kara.tracks_of(doc, "ja")[0]
        track["member"] = "retired-member"  # timing is still resolved by character identities
        new = doc.group(ed.duplicate_groups(["g"])[0])
        self.assertEqual([(b.t0, b.t1) for b in kara.effective(doc, doc.members[new.members[1]]).beats],
                         [(b.t0, b.t1) for b in beats])

    def test_copy_conversion_cleanup_never_deletes_original_readings(self):
        from dataclasses import asdict
        doc,_,ed=fixture("dialogue")
        m=doc.members["ja"]
        doc.groups[0].extra["before_sign_convert"]={"kind":"sign"}
        m.extra["before_sign_convert"]={"role":"sign","style":"JP","margins":[0,0,0],
            "overrides":{},"rev":0,"source_style":True,"generated":{"a":asdict(doc.annotations["a"])}}
        original=copy.deepcopy(doc.annotations["a"])
        gid=ed.duplicate_groups(["g"])[0]
        copied_mid=doc.group(gid).members[1]
        copied_ann=doc.annotations_of(copied_mid)[0].id
        ed.convert_to_sign([gid])
        self.assertEqual(doc.annotations["a"],original)
        self.assertNotIn(copied_ann,doc.annotations)
        self.assertEqual(doc.members["ja"].role,"dialogue")

    def test_source_style_rescale_resets_inserted_tags_and_copy_of_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            source_doc = AssDocument.parse(default_styles_ass())
            source_doc.set_info("PlayResX", "1280")
            source_doc.set_info("PlayResY", "720")
            style = source_doc.style("CN").copy("SourceOnly")
            style.set("Fontsize", 30)
            source_doc.styles.append(style)
            # Import a real source copy so the normal export source-style loader runs.
            from subkit.ass.document import Event
            text = r"{\pos(100,200)\rSourceOnly}屏幕字"
            source_doc.events = [Event(text=text, style="SourceOnly", start=100, end=400)]
            original_file = path / "original.ass"
            source_doc.save(original_file)
            original_bytes = original_file.read_bytes()
            src = import_source(original_file, src_id="s", lang="zh-Hans", lane="translation", episode_dir=path / "ep")
            tag_end = text.index("}") + 1
            m = Member("m", "g", "zh-Hans", "translation", "sign", 1000, 4000, style="SourceOnly", margins=[10,20,30],
                       speaker={"label":"制作署名"},
                       runs=[Run(text[:tag_end], "tag", src="s", e=0, a=0, b=tag_end),
                             Run(text[tag_end:], src="s", e=0, a=tag_end, b=len(text)),
                             Run(r"{\fs20}手打", ins="i", b=len(r"{\fs20}手打"))],
                       extra={"source_style": True})
            doc = Document("E", groups=[Group("g", "sign", ["m"])], members={"m":m})
            ed = Editor(doc, {"s":src})
            gid = ed.duplicate_groups(["g"])[0]
            gid2 = ed.duplicate_groups([gid])[0]
            copies = [doc.members[doc.group(g).members[0]] for g in (gid, gid2)]
            for copied in copies:
                self.assertTrue(source_positioned(copied, {"s":src}))
                self.assertTrue(file_name_speaker(copied, {"s":{"制作署名"}}))
            template = AssDocument.parse(default_styles_ass())
            cache = ExportCache()
            ass, report = build_ass(doc, {"s":src}, template, path/"ep", ExportOptions(ruby=False), cache=cache)
            by_mid = {owner["member"]: event for event, owner in zip(ass.events, report.owners) if owner["kind"] == "base"}
            original = by_mid["m"]
            self.assertIn(r"\pos(150,300)", original.text)
            self.assertIn(r"\fs20", original.text)
            self.assertEqual(original.margin_l, 15)
            for copied in copies:
                self.assertEqual(by_mid[copied.id], original)
            ass2, report2 = build_ass(doc, {"s":src}, template, path/"ep", ExportOptions(ruby=False), cache=cache)
            self.assertEqual(ass2.events, ass.events)
            self.assertEqual(report2.laid_out, 0)
            self.assertEqual(original_file.read_bytes(), original_bytes)

            # Changing the effective video width scales hand-written coordinates,
            # while copied ASS tags stay in source coordinates until export.
            from subkit.core.workspace import Episode
            from unittest.mock import Mock
            from types import SimpleNamespace
            doc.extra["coordinate_playres"]=[1920,1080]
            ep=SimpleNamespace(document=doc,dir=path/"ep",effective_playres=lambda:(1440,1080),
                               settings=lambda:{"playres":[1920,1080]},backup=Mock())
            Episode._sync_coordinates(ep)
            for copied in copies:
                self.assertIn(r"\pos(100,200)",copied.text)
            resized,report=build_ass(doc,{"s":src},template,path/"ep",ExportOptions(ruby=False,playres=(1440,1080)))
            events={o["member"]:e for o,e in zip(report.owners,resized.events) if o["kind"]=="base"}
            for copied in copies:
                self.assertEqual(events[copied.id],events["m"])

    def test_libass_copy_pixels_match_original_ruby_pinyin_and_karaoke(self):
        import hashlib
        import numpy as np
        from subkit.proc import run
        doc,sources,ed=fixture()
        for mid in ("ja","cn"):
            beats=kara.derive(doc,doc.members[mid])
            for i,b in enumerate(beats): b.t0,b.t1,b.by=1100+i*250,1350+i*250,"tap"
            ed.set_karaoke(mid,beats)
        gid=ed.duplicate_groups(["g"])[0]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            ass,report=build_ass(doc,sources,AssDocument.parse(default_styles_ass()),path,
                                 ExportOptions(pinyin_mode="PY"),measure_cache=path/"measure.sqlite3")
            frames=[]
            for group in ("g",gid):
                isolated=copy.deepcopy(ass)
                isolated.events=[e for e,o in zip(ass.events,report.owners) if o["group"]==group]
                isolated.save(path/"copy.ass")
                result=run(["ffmpeg","-hide_banner","-v","error","-f","lavfi","-i",
                            "color=black:s=1920x1080:r=10:d=2","-vf","ass=copy.ass","-ss","1.5",
                            "-frames:v","1","-f","rawvideo","-pix_fmt","gray","pipe:1"],
                           cwd=path,capture_output=True,check=True,timeout=30)
                frame=np.frombuffer(result.stdout,dtype=np.uint8)
                self.assertEqual(frame.size,1920*1080)
                self.assertGreater(np.count_nonzero(frame>24),100)
                frames.append(result.stdout)
            self.assertEqual(frames[0],frames[1])
            print("S1 real libass original/copy frame SHA-256:",hashlib.sha256(frames[0]).hexdigest())


class ExactGroupTime(unittest.TestCase):
    def test_parser_formats_precision_and_invalid(self):
        for text, expected in (("0:01:23.45",83450), ("0:01:23.456",83456), ("1:23.45",83450),
                               (" 83456 ",83456), ("0",0), ("1:02:03",3723000), ("0:59.5",59500)):
            self.assertEqual(parse_time(text), expected)
            self.assertEqual(parse_time(format_time(expected)), expected)
        for text in ("", "-1", "1.2", "0:60:00", "0:00:60.00", "1:00.1234", "abc", "1::23", "nan", "inf"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_time(text)

    def test_per_row_delta_manual_edges_and_one_undo(self):
        doc, _, ed = fixture()
        doc.members["cn"].start, doc.members["cn"].end = 1200, 4200
        for mid in ("cn", "ja"):
            beats = kara.derive(doc, doc.members[mid])
            for i, b in enumerate(beats):
                b.t0, b.t1, b.by = doc.members[mid].start + i*100, doc.members[mid].start+(i+1)*100, "tap"
            ed.set_karaoke(mid, beats)
        before = snapshot(doc)
        original_times = {mid:[(b.t0,b.t1) for b in kara.effective(doc,doc.members[mid]).beats] for mid in ("cn","ja")}
        self.assertEqual(ed.set_exact_group_time("g",83456,85555),2)
        for mid, delta in (("cn",82256),("ja",82456)):
            self.assertEqual((doc.members[mid].start,doc.members[mid].end),(83456,85555))
            self.assertEqual(doc.members[mid].extra["time_src"],{"start":"manual","end":"manual"})
            self.assertEqual([(b.t0,b.t1) for b in kara.effective(doc,doc.members[mid]).beats],
                             [(a+delta,b+delta) for a,b in original_times[mid]])
        self.assertEqual(ed.history.done[-1].label,"设置时间")
        after = snapshot(doc)
        ed.history.undo()
        self.assertEqual(snapshot(doc),before)
        ed.history.redo()
        self.assertEqual(snapshot(doc),after)

    def test_single_lane_never_follows_translation_links_and_box_off(self):
        doc, _, ed = fixture()
        for mid in ("ja","cn"):
            beats = kara.derive(doc,doc.members[mid])
            for i,b in enumerate(beats): b.t0,b.t1,b.by = 1000+i*200,1200+i*200,"tap"
            if mid == "cn": beats[0].link={"start":kara.beat_link(kara.effective(doc,doc.members['ja']).beats[0],"start")}
            ed.set_karaoke(mid,beats)
        cn_before = copy.deepcopy(doc.members["cn"])
        cn_track = copy.deepcopy(kara.tracks_of(doc,"cn"))
        ed.set_exact_group_time("g",2000,4000,target="original")
        self.assertEqual(doc.members["cn"],cn_before)
        self.assertEqual(kara.tracks_of(doc,"cn"),cn_track)
        old_track = copy.deepcopy(doc.karaoke)
        ed.set_exact_group_time("g",10000,11000,target="original",shift_beats=False)
        self.assertEqual(doc.karaoke,old_track)
        self.assertIn("有拍落在这一行的时间外",kara.effective(doc,doc.members["ja"]).issues)

    def test_accept_same_time_protects_and_invalid_does_not_edit(self):
        doc, _, ed = fixture()
        ed.set_exact_group_time("g",1000,4000,target="translation")
        self.assertTrue(doc.members["cn"].extra["time_manual"])
        self.assertNotIn("time_manual",doc.members["ja"].extra)
        before = snapshot(doc)
        for a,b,target in ((10,10,"group"),(-1,100,"group"),(0,100,"wrong")):
            with self.assertRaises(ValueError): ed.set_exact_group_time("g",a,b,target=target)
        self.assertEqual(snapshot(doc),before)

    def test_source_k_tags_box_off_keeps_absolute_beats(self):
        doc, _, ed = fixture()
        m=doc.members["ja"]
        text=r"{\ko30}あ{\ko40}い"
        from subkit.ass.text import tokenize
        m.runs=[Run(t.raw,"tag" if t.kind=="tags" else "text",ins="i_raw",a=t.start,b=t.end)
                for t in tokenize(text)]
        for aid in [a.id for a in doc.annotations_of(m.id)]: del doc.annotations[aid]
        old=[(b.t0,b.t1) for b in kara.import_from_tags(doc,m)]
        ed.set_exact_group_time("g",2000,4000,target="original",shift_beats=False)
        view=kara.effective(doc,m)
        self.assertEqual([(b.t0,b.t1) for b in view.beats],old)
        self.assertEqual(view.kind,"ko")

    def test_uppercase_source_K_keeps_sweeping_effect(self):
        doc,_,ed=fixture()
        m=doc.members["ja"]
        text=r"{\K30}あ{\K40}い"
        from subkit.ass.text import tokenize
        m.runs=[Run(t.raw,"tag" if t.kind=="tags" else "text",ins="i_raw",a=t.start,b=t.end)
                for t in tokenize(text)]
        doc.annotations.pop("a")
        ed.set_exact_group_time("g",2000,4000,target="original",shift_beats=False)
        self.assertEqual(kara.effective(doc,m).kind,"kf")


class PerKanjiReadings(unittest.TestCase):
    def test_manual_single_char_anchors_hidden_empty_and_undo(self):
        doc, sources, ed = fixture()
        before = snapshot(doc)
        old_track = copy.deepcopy(doc.karaoke)
        self.assertEqual(ed.split_kanji_readings("ja",0,2,["がっ",""]),2)
        anns=doc.annotations_of("ja")
        self.assertEqual([(a.base,a.text,a.status,a.origin) for a in anns],
                         [("学","がっ","confirmed","manual"),("校","","hidden","manual")])
        self.assertTrue(all(a.layout=={"gap":3} for a in anns))
        self.assertTrue(all(len(kara._ids_of_anchor(a.anchor))==1 for a in anns))
        self.assertTrue(validate(doc,sources).ok)
        self.assertEqual(doc.karaoke,old_track)
        self.assertEqual(doc.rules,[])
        self.assertEqual(doc.annotations["p"].member,"cn")
        self.assertEqual(ed.history.done[-1].label,"按单个汉字拆开")
        after=snapshot(doc)
        ed.history.undo()
        self.assertEqual(snapshot(doc),before)
        ed.history.redo()
        self.assertEqual(snapshot(doc),after)
        self.assertEqual(snapshot(document_from_json(document_to_json(doc))),after)

    def test_empty_is_not_revived_by_automatic_refresh(self):
        from subkit.reading.engine import ReadingEngine
        doc,sources,ed=fixture()
        ed.split_kanji_readings("ja",0,2,["",""])
        ed.reannotate(["ja"],ReadingEngine(),sources)
        hidden=[a for a in doc.annotations_of("ja") if a.base in ("学","校")]
        self.assertEqual([(a.base,a.status,a.origin) for a in hidden],
                         [("学","hidden","manual"),("校","hidden","manual")])

    def test_written_kana_are_not_annotated_and_other_occurrence_unchanged(self):
        doc,_,ed=fixture()
        m=doc.members["ja"]
        text="取り組み　取り組み"
        m.runs=[Run(text,ins="i_typed",b=len(text))]
        doc.annotations.pop("a")
        ed.add_annotation("ja",0,4,"とりくみ")
        other=ed.add_annotation("ja",5,9,"とりくみ")
        other_before=copy.deepcopy(doc.annotations[other])
        ed.split_kanji_readings("ja",0,4,["と","く"])
        new=[a for a in doc.annotations_of("ja") if a.id!=other]
        self.assertEqual([(a.base,a.text) for a in new],[("取","と"),("組","く")])
        self.assertEqual(doc.annotations[other],other_before)
        self.assertEqual(m.text,text)

    def test_invalid_range_and_partial_old_reading_do_not_drop_manual_data(self):
        doc,_,ed=fixture()
        ed.add_annotation("ja",0,4,"がっこうにい")
        before=snapshot(doc)
        with self.assertRaisesRegex(ValueError,"整个读音范围"):
            ed.split_kanji_readings("ja",0,2,["がっ","こう"])
        self.assertEqual(snapshot(doc),before)
        for a,b,readings in ((-1,2,["a","b"]),(0,2,["a"]),(0,1,["a"])):
            with self.assertRaises(ValueError): ed.split_kanji_readings("ja",a,b,readings)
        self.assertEqual(snapshot(doc),before)


if __name__ == "__main__":
    unittest.main()
