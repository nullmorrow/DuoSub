"""Text edits preserve existing ASS controls and rendering origin without source forgery."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.ass.document import AssDocument, Event, Style
from subkit.ass.text import line_tag, tokenize
from subkit.core.editor_text import editing_text, is_private_run
from subkit.core.export import ExportCache, ExportOptions, build_ass
from subkit.core.importer import import_source
from subkit.core.merge import runs_of_event
from subkit.core.ops import Editor
from subkit.core.position import source_positioned
from subkit.core.readings import visible_map
from subkit.core.render_origin import source_events, source_id
from subkit.model.document import Document, Group, Member, Run
from subkit.model.serialize import document_from_json, document_to_json
from subkit.model.source import Source, SourceEvent
from subkit.model.validate import validate


PREFIX = r"{\an7\c&HD0A6D7&\pos(965.508,128.354)}"


def member_of(source, *, mid="m", gid="g", event=0):
    ev = source.events[event]
    return Member(mid, gid, source.lang, source.lane, "sign", ev.start, ev.end,
                  layer=ev.layer or 0, style=ev.style or "Screen", margins=list(ev.margins or (0, 0, 0)),
                  effect=ev.effect or "", runs=runs_of_event(source, event),
                  extra={"source_style": True})


def fixture(text=PREFIX + "智力"):
    source = Source("s", "zh-Hant", "translation", "ass", "", "", "", 0, "utf-8", "lf",
                    events=[SourceEvent(0, "Dialogue", 1000, 4000, text, style="Screen")],
                    roles={0: "sign"}, playres=[1920, 1080])
    doc = Document("text-rendering", groups=[Group("g", "sign", ["m"])],
                   members={"m": member_of(source)})
    sources = {source.id: source}
    return doc, sources, Editor(doc, sources)


def controls(member):
    """Existing hidden runs and their editing-text offsets, in stable order."""
    offset, result = 0, []
    for run in member.runs:
        if run.kind == "tag":
            if not is_private_run(run):
                result.append((offset, run))
        else:
            offset += len(run.text)
    return result


def snapshot(doc):
    state = copy.deepcopy(document_to_json(doc))
    state.pop("revision", None)
    return state


class TextRenderingCore(unittest.TestCase):
    def assert_valid(self, doc, sources):
        report = validate(doc, sources)
        self.assertTrue(report.ok, report.summary())

    def assert_source_unchanged(self, sources, saved):
        self.assertEqual({sid: source.events_json() for sid, source in sources.items()}, saved)

    def test_full_replacement_keeps_source_position_colour_and_tag_identity(self):
        doc, sources, editor = fixture()
        member = doc.members["m"]
        saved = {sid: copy.deepcopy(source.events_json()) for sid, source in sources.items()}
        original_tag = copy.deepcopy(member.runs[0])
        original_ids = visible_map(member)[1]
        editor.replace_editor_text("m", "力量")
        self.assertEqual(member.text, PREFIX + "力量")
        self.assertEqual(controls(member), [(0, original_tag)])
        self.assertTrue(source_positioned(member, sources))
        self.assertEqual(source_id(member), "s")
        typed = [run for run in member.runs if run.kind == "text" and run.ins is not None]
        self.assertTrue(typed)
        self.assertTrue(all(run.src is None and run.e is None and run.ins for run in typed))
        self.assertTrue(all(run.render_src is None and run.render_e is None for run in typed))
        self.assertEqual(document_to_json(doc)["format_version"], 1)
        self.assertEqual(visible_map(member)[1][0], original_ids[1])
        self.assertNotIn(visible_map(member)[1][1], original_ids)
        self.assertEqual("".join(stone.text for stone in doc.tombstones), "智")
        self.assert_source_unchanged(sources, saved)
        self.assert_valid(doc, sources)

    def test_multiple_leading_blocks_and_all_inline_states_survive_equal_length_replacement(self):
        leading = r"{\an7}{\pos(100,200)\fad(100,720)\clip(10,20,300,400)}"
        inline = r"{\fnMS Gothic\fs70\c&H123456&\t(0,900,\frz20\fscx150)}"
        reset = r"{\rScreen\bord2\alpha&H20&}"
        doc, sources, editor = fixture(leading + "AB" + inline + "C" + reset + "D")
        old = copy.deepcopy(controls(doc.members["m"]))
        editor.replace_text("m", "WXYZ")
        self.assertEqual(doc.members["m"].text, leading + "WX" + inline + "Y" + reset + "Z")
        self.assertEqual(controls(doc.members["m"]), old)
        self.assert_valid(doc, sources)

    def test_inline_controls_map_proportionally_inside_shorter_replacement(self):
        tag = r"{\fs90\c&H00FF00&}"
        doc, sources, editor = fixture(PREFIX + "AB" + tag + "CD")
        editor.replace_text("m", "XYZ")
        self.assertEqual(doc.members["m"].text, PREFIX + "X" + tag + "YZ")
        self.assertEqual([offset for offset, _run in controls(doc.members["m"])], [0, 1])
        self.assert_valid(doc, sources)

    def test_partial_selected_replacement_preserves_unselected_ids_and_boundary_controls(self):
        doc, sources, editor = fixture(PREFIX + r"AA{\i1}AA{\i0}AA")
        before_ids = visible_map(doc.members["m"])[1]
        before_controls = copy.deepcopy(controls(doc.members["m"]))
        editor.replace_text("m", "AABBAA", span=(2, 4))
        member = doc.members["m"]
        self.assertEqual(member.text, PREFIX + r"AA{\i1}BB{\i0}AA")
        self.assertEqual(controls(member), before_controls)
        after_ids = visible_map(member)[1]
        self.assertEqual(after_ids[:2], before_ids[:2])
        self.assertEqual(after_ids[4:], before_ids[4:])
        self.assertTrue(set(after_ids[2:4]).isdisjoint(before_ids))
        self.assert_valid(doc, sources)

    def test_delete_range_keeps_inline_controls_and_does_not_tombstone_them(self):
        doc, sources, editor = fixture(PREFIX + r"ab{\c&HFF0000&}cd{\rScreen}ef")
        old_tags = [copy.deepcopy(run) for _offset, run in controls(doc.members["m"])]
        editor.delete_range("m", 1, 3)
        self.assertEqual(doc.members["m"].text, PREFIX + r"a{\c&HFF0000&}d{\rScreen}ef")
        self.assertEqual([run for _offset, run in controls(doc.members["m"])], old_tags)
        self.assertEqual([offset for offset, _run in controls(doc.members["m"])], [0, 1, 2])
        self.assertEqual("".join(stone.text for stone in doc.tombstones), "bc")
        self.assert_valid(doc, sources)

    def test_replacement_deletion_and_delete_range_have_identical_surviving_source_runs(self):
        first, sources, delete = fixture(PREFIX + r"ab{\b1}cd{\b0}ef")
        second = copy.deepcopy(first)
        replace = Editor(second, sources)
        delete.delete_range("m", 1, 3)
        replace.replace_text("m", "adef")
        self.assertEqual(first.members["m"].runs, second.members["m"].runs)
        self.assert_valid(first, sources)
        self.assert_valid(second, sources)

    def test_first_and_last_insertions_keep_prefix_and_original_character_ids(self):
        for method in ("insert", "replace"):
            with self.subTest(method=method):
                doc, sources, editor = fixture()
                member = doc.members["m"]
                before_ids = visible_map(member)[1]
                if method == "insert":
                    editor.insert_text("m", 0, "超")
                    editor.insert_text("m", 3, "強")
                else:
                    editor.replace_text("m", "超智力")
                    editor.replace_text("m", "超智力強")
                self.assertEqual(member.text, PREFIX + "超智力強")
                self.assertEqual(visible_map(member)[1][1:3], before_ids)
                inserted = [run for run in member.runs if run.ins is not None and run.kind == "text"]
                self.assertTrue(all(run.src is None and run.render_src is None and run.render_e is None
                                    for run in inserted))
                self.assertFalse(doc.tombstones)
                self.assert_valid(doc, sources)

    def test_insertion_at_inline_boundary_keeps_following_controls_without_forging_context(self):
        doc, sources, editor = fixture("AB")
        other = Source("other", "zh-Hant", "translation", "ass", "", "", "", 0, "utf-8", "lf",
                       events=[SourceEvent(0, "Dialogue", 1000, 4000, r"{\c&HFF0000&}CD")])
        sources[other.id] = other
        doc.members["m"].runs.extend(runs_of_event(other, 0))
        editor.insert_text("m", 2, "X")
        self.assertEqual(doc.members["m"].text, r"AB{\c&HFF0000&}XCD")
        inserted = next(run for run in doc.members["m"].runs if run.ins is not None)
        self.assertIsNone(inserted.render_src)
        self.assertIsNone(inserted.render_e)
        self.assertEqual(source_events(doc.members["m"]), {("s", 0), ("other", 0)})
        self.assert_valid(doc, sources)

    def test_empty_text_retains_tags_and_retyping_retains_rendering_context(self):
        doc, sources, editor = fixture()
        tag = copy.deepcopy(doc.members["m"].runs[0])
        editor.replace_text("m", "")
        self.assertEqual(doc.members["m"].runs, [tag])
        self.assertTrue(source_positioned(doc.members["m"], sources))
        editor.insert_text("m", 0, "力量")
        self.assertEqual(doc.members["m"].text, PREFIX + "力量")
        text = next(run for run in doc.members["m"].runs if run.kind == "text")
        self.assertIsNone(text.render_src)
        self.assertEqual(source_id(doc.members["m"]), "s")
        self.assert_valid(doc, sources)

    def test_inserted_only_text_does_not_acquire_fake_source_or_rendering_context(self):
        member = Member("m", "g", "zh-Hant", "translation", "dialogue", 1000, 4000,
                        runs=[Run("AB", ins="original-insertion", b=2)])
        doc = Document("typed", groups=[Group("g", members=["m"])], members={"m": member})
        editor = Editor(doc, {})
        editor.replace_text("m", "XY")
        editor.insert_text("m", 0, "Z")
        self.assertTrue(all(run.src is None and run.e is None and run.render_src is None and run.render_e is None
                            for run in member.runs))
        self.assertIsNone(source_id(member))
        self.assert_valid(doc, {})

    def test_typed_ass_looking_braces_stay_editable_text(self):
        doc, sources, editor = fixture()
        new = r"力量{ordinary}{\b1}{@private:x}{@注:unfinished"
        editor.replace_editor_text("m", new)
        self.assertEqual(editing_text(doc.members["m"]), new)
        self.assertEqual([run.text for _pos, run in controls(doc.members["m"])], [PREFIX])
        self.assertTrue(all(run.kind == "text" for run in doc.members["m"].runs if run.text != PREFIX))
        self.assert_valid(doc, sources)

    def test_raw_wrapping_and_unwrapping_keep_existing_ass_and_character_ids(self):
        doc, sources, editor = fixture()
        member = doc.members["m"]
        ids = visible_map(member)[1]
        tag = copy.deepcopy(member.runs[0])
        editor.replace_editor_text("m", "{@Raw=智力}")
        self.assertEqual(controls(member), [(0, tag)])
        self.assertEqual(editing_text(member), "{@Raw=智力}")
        self.assertEqual(visible_map(member)[1], ids)
        self.assertTrue(editor.history.done[-1].keeps_text)
        editor.replace_editor_text("m", "智力")
        self.assertEqual(member.text, PREFIX + "智力")
        self.assertEqual(visible_map(member)[1], ids)
        self.assertFalse(doc.tombstones)
        self.assert_valid(doc, sources)

    def test_gloss_mark_editing_uses_current_marks_and_keeps_existing_ass_once(self):
        doc, sources, editor = fixture()
        editor.replace_editor_text("m", "智{@注:=First}{@g:=Second}力")
        self.assertEqual(editing_text(doc.members["m"]), "智{@注:=First}{@g:=Second}力")
        editor.replace_editor_text("m", "力{@g:=New}量")
        self.assertEqual(editing_text(doc.members["m"]), "力{@g:=New}量")
        self.assertEqual([run.text for _pos, run in controls(doc.members["m"])], [PREFIX])
        self.assertNotIn("First", doc.members["m"].text)
        self.assertNotIn("Second", doc.members["m"].text)
        self.assert_valid(doc, sources)

    def test_first_mixed_source_replacement_does_not_switch_to_later_real_source(self):
        doc, sources, editor = fixture("AB")
        other = Source("other", "zh-Hant", "translation", "ass", "", "", "", 0, "utf-8", "lf",
                       events=[SourceEvent(0, "Dialogue", 1000, 4000, "CD")])
        sources[other.id] = other
        doc.members["m"].runs.extend(runs_of_event(other, 0))
        editor.replace_text("m", "WXCD", span=(0, 2))
        self.assertEqual(source_id(doc.members["m"]), "s")
        self.assertEqual(source_events(doc.members["m"]), {("s", 0), ("other", 0)})
        self.assertEqual((doc.members["m"].runs[0].render_src, doc.members["m"].runs[0].render_e), ("s", 0))
        self.assert_valid(doc, sources)

    def test_copy_then_replace_keeps_render_context_and_independent_insertions(self):
        doc, sources, editor = fixture()
        original = copy.deepcopy(doc.members["m"])
        copied_gid = editor.duplicate_groups(["g"])[0]
        copied = doc.members[doc.group(copied_gid).members[0]]
        copied_ids = set(visible_map(copied)[1])
        original_ids = set(visible_map(doc.members["m"])[1])
        editor.replace_text(copied.id, "力量")
        self.assertEqual(copied.text, PREFIX + "力量")
        self.assertEqual(doc.members["m"], original)
        self.assertTrue(all(run.src is None and run.ins for run in copied.runs))
        self.assertEqual(source_id(copied), "s")
        new_runs = [run for run in copied.runs if run.kind == "text"
                    and any(run.identity(index) not in copied_ids for index in range(len(run.text)))]
        self.assertTrue(new_runs)
        self.assertTrue(all(run.render_src is None and run.render_e is None for run in new_runs))
        self.assertTrue(original_ids.isdisjoint(visible_map(copied)[1]))
        self.assertTrue(source_positioned(copied, sources))
        self.assertFalse(doc.tombstones)
        self.assert_valid(doc, sources)

    def test_undo_redo_and_serialization_keep_tags_context_and_ledger_exactly(self):
        doc, sources, editor = fixture("ABCD")
        before = snapshot(doc)
        editor.replace_text("m", "XYZ")
        after = snapshot(doc)
        self.assertEqual(document_to_json(doc)["format_version"], 2)
        context = doc.members["m"].runs[0]
        self.assertEqual((context.text, context.kind, context.render_src, context.render_e), ("", "tag", "s", 0))
        self.assertEqual(context.a, context.b)
        self.assertTrue(context.ins)
        self.assertIsNone(context.src)
        self.assertEqual(snapshot(document_from_json(document_to_json(doc))), after)
        editor.history.undo()
        self.assertEqual(snapshot(doc), before)
        editor.history.redo()
        self.assertEqual(snapshot(doc), after)
        self.assert_valid(doc, sources)


class TextRenderingExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="text_edit_rendering_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.episode = self.root / "episode"

    def source(self, text=PREFIX + "智力", *, sid="s", playres=(1280, 720), fontsize=30):
        ass = AssDocument()
        ass.set_info("PlayResX", playres[0])
        ass.set_info("PlayResY", playres[1])
        ass.set_info("ScaledBorderAndShadow", "yes")
        ass.styles = [Style.new("Screen", Fontname="MS Gothic", Fontsize=fontsize,
                                Alignment=7, Outline=2, Shadow=1, PrimaryColour="&H00123456"),
                      Style.new("Accent", Fontname="Arial", Fontsize=40, Italic=-1)]
        ass.events = [Event(text=text, style="Screen", start=100, end=400,
                            layer=3, margin_l=20, margin_r=30, margin_v=40, effect="original-effect")]
        path = self.root / f"original-{sid}.ass"
        ass.save(path)
        source = import_source(path, src_id=sid, lang="zh-Hant", lane="translation", episode_dir=self.episode,
                               cn_cfg={"style_roles": {"Screen": "sign"}})
        doc = Document("export", groups=[Group("g", "sign", ["m"])], members={"m": member_of(source)})
        sources = {sid: source}
        return doc, sources, Editor(doc, sources)

    def build(self, doc, sources, *, mode="release", cache=None):
        template = AssDocument()
        template.set_info("PlayResX", 1920)
        template.set_info("PlayResY", 1080)
        template.styles = [Style.new("Screen", Fontsize=80), Style.new("Accent", Fontsize=90)]
        return build_ass(doc, sources, template, self.episode,
                         ExportOptions(mode=mode, ruby=False), cache=cache)

    def base(self, ass, report, mid="m"):
        return next(event for event, owner in zip(ass.events, report.owners)
                    if owner["member"] == mid and owner["kind"] == "base")

    def test_full_edit_retains_custom_source_style_resampling_and_event_fields(self):
        doc, sources, editor = self.source(r"{\an7\pos(100,200)\c&HBC89C4&}智力")
        source_before = copy.deepcopy(sources["s"].events_json())
        original_bytes = Path(sources["s"].path).read_bytes()
        copy_bytes = (self.episode / sources["s"].copy).read_bytes()
        old_ass, old_report = self.build(doc, sources)
        old = self.base(old_ass, old_report)
        editor.replace_editor_text("m", "力量")
        ass, report = self.build(doc, sources)
        event = self.base(ass, report)
        self.assertEqual(event.text, old.text.replace("智力", "力量"))
        self.assertEqual(line_tag(event.text, "pos").float_args(), [150, 300])
        self.assertEqual(event.style, "src.Screen")
        self.assertEqual(ass.style(event.style).fontsize, 45)
        self.assertEqual(ass.style(event.style).fontname, "MS Gothic")
        self.assertEqual((event.layer, event.start, event.end, event.margin_l, event.margin_r, event.margin_v,
                          event.effect), (3, 100, 400, 30, 45, 60, "original-effect"))
        self.assertEqual(sources["s"].events_json(), source_before)
        self.assertEqual(Path(sources["s"].path).read_bytes(), original_bytes)
        self.assertEqual((self.episode / sources["s"].copy).read_bytes(), copy_bytes)
        self.assertTrue(validate(doc, sources).ok)

    def test_control_bytes_reset_alias_and_animations_survive_export_after_replacement(self):
        tags = r"{\an7\move(100,200,200,300,20,900)\org(30,40)\fad(100,720)}"
        first = r"{\clip(10,20,300,400)\fnArial\fs30\c&H123456&\t(0,900,\frz30\fs60)}"
        second = r"{\rAccent\iclip(20,30,200,250)\alpha&H40&}"
        doc, sources, editor = self.source(tags + "AB" + first + "C" + second + "D")
        old_ass, old_report = self.build(doc, sources)
        old = self.base(old_ass, old_report)
        editor.replace_text("m", "WXYZ")
        ass, report = self.build(doc, sources)
        event = self.base(ass, report)
        self.assertEqual([tok.raw for tok in tokenize(event.text) if tok.kind == "tags"],
                         [tok.raw for tok in tokenize(old.text) if tok.kind == "tags"])
        self.assertIn(r"\rsrc.Accent", event.text)
        self.assertEqual(ass.style("src.Accent").fontsize, 60)
        self.assertEqual(line_tag(event.text, "move").float_args(), [150, 300, 300, 450, 20, 900])
        self.assertEqual(line_tag(event.text, "fad").float_args(), [100, 720])
        self.assertTrue(validate(doc, sources).ok)

    def test_plain_source_without_tags_still_keeps_source_style_when_all_characters_change(self):
        doc, sources, editor = self.source("智力")
        editor.replace_text("m", "體能")
        self.assertTrue(all(run.src is None for run in doc.members["m"].runs))
        self.assertEqual(source_id(doc.members["m"]), "s")
        ass, report = self.build(doc, sources)
        event = self.base(ass, report)
        self.assertEqual(event.style, "src.Screen")
        self.assertEqual(ass.style(event.style).fontsize, 45)
        self.assertEqual(event.text, "體能")

    def test_typed_ass_tags_stay_in_episode_coordinates_after_source_context_is_preserved(self):
        doc, sources, editor = self.source("智力")
        typed = r"{\pos(100,200)\fs20}體能"
        editor.replace_editor_text("m", typed)
        ass, report = self.build(doc, sources)
        event = self.base(ass, report)
        self.assertEqual(event.text, typed)
        self.assertEqual(line_tag(event.text, "pos").float_args(), [100, 200])
        self.assertEqual(line_tag(event.text, "fs").num(), 20)
        self.assertEqual(ass.style(event.style).fontsize, 45)
        self.assertTrue(all(run.render_src is None for run in doc.members["m"].runs if run.kind == "text"))

    def test_typed_font_override_is_not_rescaled_with_retained_source_position(self):
        doc, sources, editor = self.source(r"{\an7\pos(100,200)}智力")
        editor.replace_editor_text("m", r"力量{\fs20}手打")
        ass, report = self.build(doc, sources)
        event = self.base(ass, report)
        self.assertEqual(line_tag(event.text, "pos").float_args(), [150, 300])
        self.assertEqual(line_tag(event.text, "fs").num(), 20)
        self.assertEqual(document_to_json(doc)["format_version"], 1)

    def test_warm_preview_cache_after_edit_matches_cold_preview_and_release(self):
        doc, sources, editor = self.source()
        cache = ExportCache()
        self.build(doc, sources, mode="preview", cache=cache)
        editor.replace_text("m", "力量")
        warm, warm_report = self.build(doc, sources, mode="preview", cache=cache)
        cached, cached_report = self.build(doc, sources, mode="preview", cache=cache)
        cold, cold_report = self.build(doc, sources, mode="preview")
        release, release_report = self.build(doc, sources)
        self.assertEqual(warm_report.laid_out, 1)
        self.assertEqual(cached_report.laid_out, 0)
        self.assertEqual(warm.serialize(), cached.serialize())
        self.assertEqual(warm.serialize(), cold.serialize())
        self.assertEqual(warm.events, release.events)
        self.assertEqual(warm.styles, release.styles)
        self.assertEqual(warm_report.owners, cold_report.owners)
        self.assertEqual(warm_report.owners, release_report.owners)

    def test_copy_edit_keeps_source_resolution_and_original_member_unchanged(self):
        doc, sources, editor = self.source(r"{\an7\pos(100,200)}智力")
        original = copy.deepcopy(doc.members["m"])
        copied_group = editor.duplicate_groups(["g"])[0]
        copied = doc.members[doc.group(copied_group).members[0]]
        editor.replace_text(copied.id, "力量")
        ass, report = self.build(doc, sources)
        source_event, copy_event = self.base(ass, report), self.base(ass, report, copied.id)
        self.assertEqual(copy_event.text, source_event.text.replace("智力", "力量"))
        self.assertEqual(copy_event.style, source_event.style)
        self.assertEqual(line_tag(copy_event.text, "pos").float_args(), [150, 300])
        self.assertEqual(doc.members["m"], original)
        self.assertTrue(validate(doc, sources).ok)

    def test_mixed_real_and_render_origins_keep_first_source_style_and_resolution(self):
        doc, sources, editor = self.source("AB")
        _other_doc, other_sources, _other_editor = self.source("CD", sid="other", playres=(640, 360), fontsize=50)
        sources.update(other_sources)
        doc.members["m"].runs.extend(runs_of_event(sources["other"], 0))
        old_ass, old_report = self.build(doc, sources)
        editor.replace_text("m", "WXCD", span=(0, 2))
        ass, report = self.build(doc, sources)
        event = self.base(ass, report)
        self.assertEqual(source_id(doc.members["m"]), "s")
        self.assertEqual(event.style, self.base(old_ass, old_report).style)
        self.assertEqual(ass.style(event.style).fontsize, 45)
        self.assertEqual((event.margin_l, event.margin_r, event.margin_v), (30, 45, 60))
        self.assertTrue(validate(doc, sources).ok)

    def test_clearing_text_omits_empty_release_line_then_retyping_restores_position(self):
        doc, sources, editor = self.source(r"{\an7\pos(100,200)}智力")
        editor.replace_text("m", "")
        empty, _report = self.build(doc, sources)
        self.assertFalse(empty.events)
        editor.replace_text("m", "力量")
        ass, report = self.build(doc, sources)
        event = self.base(ass, report)
        self.assertEqual(line_tag(event.text, "pos").float_args(), [150, 300])
        self.assertEqual(event.style, "src.Screen")
        self.assertTrue(validate(doc, sources).ok)


if __name__ == "__main__":
    unittest.main()
