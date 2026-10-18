"""Document → ASS (preview / work / release), FORMAT-v1 §11.

One generator serves the live preview in the player and the release export, so
what the user reviews is what gets published.  The release differs only in
what it leaves out:

* never written: project IDs, hidden comments, uncertain-reading marks;
* optional: the translation file's own Comment lines (``include_source_comments``),
  the speaker in the Name field (``include_actor``);
* kept unchanged: ASS override tags and any ``@`` / ``#`` inside dialogue text.

Members coming from an ASS source at another resolution (e.g. 1280×720 signs)
are rescaled to the template resolution with the baseline resampler.

User position settings (``member.overrides``: ``an`` / ``pos``) become line tags;
if the source line had its own \\an/\\pos/\\move, the user's setting wins in the
output (the source itself is never changed) and the report counts it.

Live preview: an :class:`ExportCache` keeps every member's output (its event, its
ruby lines and their geometry), so after an edit only the changed members are laid
out again.  ``ExportReport.owners`` says which member (and which annotation) every
output event belongs to — the player uses it to find what the user clicked.
"""

from __future__ import annotations

import re

import copy
from itertools import groupby
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from ..ass.document import AssDocument, Event, Style
from ..ass.resample import scale_text
from ..ass.tags import KARAOKE_TAGS
from ..ass.text import (insert_at_visible, line_tag, raw_mark_errors, remove_private, remove_tags_named,
                        split_karaoke_overrides, tokenize)
from ..gloss import Gloss, gloss_anchor, parse_glosses
from ..languages import get_language
from ..karaoke_fx import EventLimit, ExportError, FxContext, FxGlyph, INLINE, get_effect
from ..karaoke_fx import ktv
from ..model.document import Document, Member
from ..model.source import Source
from ..pinyin import display as display_pinyin
from ..ruby.generate import RubyGenerator, RubyJob, RubyPiece, derive_ruby_style
from ..ruby.measure import Measurer, measurement_environment
from . import karaoke as kara
from .display import (DisplayRules, display_name, display_runs, punctuation_fonts, punctuation_layout,
                      single_line_text, display_text)
from .readings import anchor_positions, display_annotations, pieces_of, visible_map
from .render_origin import run_source, source_id
from .fades import inherited_fades

POSITION_TAGS = ("an", "pos", "move")


@dataclass
class ExportOptions:
    mode: str = "release"                  # release | preview | work
    include_actor: bool = True
    include_source_comments: bool = False
    ruby: bool = True
    blur: float = 0.0
    ruby_gap: float = 2.0
    ruby_min_scale: float = 60.0
    ruby_overhang: float = 0.5
    playres: tuple[int, int] = (1920, 1080)   # project resolution (a style-only template has no PlayRes)
    template_playres: Optional[tuple[int, int]] = None  # style-only template uses episode settings
    display: DisplayRules = field(default_factory=DisplayRules)   # user-chosen display rules (never change the document)
    pinyin_mode: str = "off"
    name_format: str = "short"              # short | full | family | given | label
    ruby_offset: float = 0.0                # project-wide px at 1080p
    pinyin_offset: float = 0.0
    ruby_max_width: dict = field(default_factory=lambda: {
        "dialogue": {"enabled": True, "percent": 150}, "song": {"enabled": True, "percent": 150}})
    reference_enabled: bool = False         # export-only: never edits the project
    reference_path: Optional[Path] = None   # file, directory, or automatic beside original sources
    annotation_order: str = "end"          # release file: end | after_parent

    def ruby_width_percent(self, kind: str) -> Optional[float]:
        setting = self.ruby_max_width.get("song" if kind == "song" else "dialogue", {})
        return float(setting.get("percent", 150)) if setting.get("enabled", True) else None

    def __post_init__(self) -> None:
        if self.name_format not in ("short", "full", "family", "given", "label"):
            raise ValueError(self.name_format)
        if self.annotation_order not in ("end", "after_parent"):
            raise ValueError(f"未知注音行顺序：{self.annotation_order}")

    def key(self) -> tuple:
        d = self.display
        return (self.mode, self.include_actor, self.include_source_comments, self.ruby, self.ruby_gap, self.ruby_min_scale,
                self.ruby_overhang, tuple(self.playres), self.template_playres, tuple((key, tuple(sorted(value.items())) if isinstance(value, dict) else value)
                    for key, value in sorted(vars(d).items())) if d is not None else None,
                self.pinyin_mode, self.name_format, self.ruby_offset, self.pinyin_offset,
                self.ruby_width_percent("dialogue"), self.ruby_width_percent("song"),
                self.reference_enabled, str(self.reference_path or ""), self.annotation_order)


@dataclass
class ExportReport:
    events: int = 0
    ruby_lines: int = 0
    pinyin_lines: int = 0
    not_rendered: list[str] = field(default_factory=list)      # hidden fillers / hidden text
    hidden_groups: int = 0
    hidden_by_rules: int = 0                                     # characters a display rule did not show
    zh_lines: int = 0
    zh_script: str = "off"
    zh_variant: str = ""
    uncertain: int = 0
    positions_overridden: int = 0                                # source \an/\pos/\move replaced by the user's position
    karaoke_lines: int = 0                                       # lines with SubKit karaoke
    karaoke_fx_lines: int = 0
    karaoke_fx_events: int = 0
    karaoke_unfinished: list[str] = field(default_factory=list)  # placeholder / changed beats (release check)
    karaoke_stripped: list[str] = field(default_factory=list)    # converted dialogue whose source tags were omitted
    # lines with no time on screen (end ≤ start, or entirely before 0): written as Comment, not as Dialogue — a
    # zero-length Dialogue makes mpv treat the whole track as "unknown duration" and cut other lines short
    not_timed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    owners: list[dict[str, Any]] = field(default_factory=list)  # parallel to the output events
    laid_out: int = 0                                            # members laid out in this run (the rest came from the cache)
    elapsed: float = 0.0                                         # seconds (filled in by the preview job)
    reference: Optional[dict[str, Any]] = None                    # detached audit payload for optional HTML
    replacement_lines: int = 0
    replacement_matches: int = 0
    replacement_deleted: int = 0
    replacement_key: Optional[tuple] = None
    edge_spaces_removed: int = 0
    empty_events_removed: int = 0
    annotation_order: str = "end"
    order_layer_events: int = 0

    @property
    def traditional_lines(self) -> int:
        return self.zh_lines if self.zh_script == "traditional" else 0

    @property
    def traditional_variant(self) -> str:
        return self.zh_variant if self.zh_script == "traditional" else ""


@dataclass
class _MemberOut:
    event: Event
    source_text: str = ""                 # exported text before whole-line member overrides
    ruby_font_context: Optional[str] = None  # line font before display-only punctuation fonts
    ruby: list[Event] = field(default_factory=list)
    ruby_style: str = ""
    ruby_geoms: list[dict[str, Any]] = field(default_factory=list)   # per ruby event: ann, x, y, w, h
    pinyin: list[Event] = field(default_factory=list)
    pinyin_geoms: list[dict[str, Any]] = field(default_factory=list)
    ids: list = field(default_factory=list)                          # identity of every visible char of event.text
    hidden_by_rules: int = 0
    converted: bool = False
    display_bounds: list[int] = field(default_factory=list)
    display_ends: list[int] = field(default_factory=list)
    merged: bool = False
    annotation_ids: list = field(default_factory=list)
    uncertain: int = 0
    overridden: bool = False
    warnings: list[str] = field(default_factory=list)
    kview: Any = None                                                # karaoke view (beats with timing) or None
    untimed: bool = False                                            # no time on screen: written as a Comment
    karaoke_stripped: bool = False
    fx_events: Optional[list[Event]] = None
    fx_owners: list[dict] = field(default_factory=list)
    fx_ruby: bool = False
    ktv_slot: Any = None
    glosses: list[Gloss] = field(default_factory=list)
    gloss_events: list[Event] = field(default_factory=list)
    replacements: int = 0
    replacement_deleted: set[int] = field(default_factory=set)
    replacement_ranges: list[tuple[int, int]] = field(default_factory=list)


class ExportCache:
    """Per-member output of the last exports (live preview).  Not thread-safe: one job at a time."""

    def __init__(self) -> None:
        self.key: Any = None
        self.members: dict[str, tuple[Any, _MemberOut]] = {}
        self.sources: dict[str, tuple[Any, Optional[AssDocument]]] = {}
        self.rescaled: dict[tuple, str] = {}

    def clear(self) -> None:
        self.key = None
        self.members.clear()
        self.rescaled.clear()


def ms_to_cs(ms: float) -> int:
    return int((int(round(ms)) + 5) // 10)


def _source_doc_for(src: Source, episode_dir: Path, cache: Optional[ExportCache] = None) -> Optional[AssDocument]:
    if src.format not in ("ass", "ssa"):
        return None
    p = Path(episode_dir) / src.copy
    if not p.exists():
        return None
    if cache is not None:
        st = p.stat()
        sig = (str(p), st.st_size, st.st_mtime_ns)
        hit = cache.sources.get(src.id)
        if hit and hit[0] == sig:
            return hit[1]
        doc = AssDocument.load(p)
        cache.sources[src.id] = (sig, doc)
        return doc
    return AssDocument.load(p)


def apply_overrides(text: str, overrides: dict[str, Any]) -> tuple[str, bool]:
    """Line tags from the user's position, whole-line format and raw tag settings, in one leading block (position,
    then format, then the raw tags, so a raw tag wins over a format setting of the same name).  Returns (text,
    replaced a source position tag).  Tags of the same names elsewhere in the source text are left out of the
    output (the user's setting is for the whole line) — except \\t animations; the source itself is never changed."""
    if not overrides:
        return text, False
    from . import textformat

    head: list[str] = []
    an = overrides.get("an")
    pos = overrides.get("pos")
    move = overrides.get("move")
    if an:
        head.append(f"\\an{int(an)}")
    if move and len(move) >= 4:
        head.append("\\move(" + ",".join(_num(v) if k < 4 else str(int(v)) for k, v in enumerate(move)) + ")")
    elif pos and len(pos) >= 2:
        head.append(f"\\pos({_num(pos[0])},{_num(pos[1])})")
    source_off = bool(overrides.get("source_position_off"))
    positioned = bool(head) or source_off
    fmt = textformat.line_props(overrides)
    if fmt:
        head.append(textformat.tags(fmt))
    if not head and not source_off:
        return text, False
    position_names = set(POSITION_TAGS) | ({"org", "a"} if source_off else set())
    names = textformat.tag_names(fmt) | (position_names if positioned else set())
    stripped = remove_tags_named(text, names)
    replaced = positioned and remove_tags_named(text, position_names) != text
    return ("{" + "".join(head) + "}" if head else "") + stripped, replaced


def _num(v: float) -> str:
    s = f"{float(v):.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def _reset_styles(text: str) -> set[str]:
    return {tag.arg for tok in tokenize(text) if tok.kind == "tags"
            for tag in tok.tags if tag.name == "r" and tag.arg}


def _alias_resets(text: str, aliases: dict[str, str]) -> str:
    blocks = []
    for tok in tokenize(text):
        if tok.kind != "tags":
            blocks.append(tok.raw)
            continue
        parts, at = [], 0
        for tag in tok.tags:
            start = tok.raw.find(tag.raw, at)
            parts.append(tok.raw[at:start])
            parts.append("\\r" + aliases[tag.arg] if tag.name == "r" and tag.arg in aliases else tag.raw)
            at = start + len(tag.raw)
        blocks.append("".join(parts) + tok.raw[at:])
    return "".join(blocks)


def _member_sig(m: Member, anns: list, alias: Optional[str], include_actor: bool, ksig: tuple = (),
                pinyin_mode: str = "off") -> tuple:
    sp = m.speaker or {}
    return (m.group, m.start, m.end, m.layer, m.style, tuple(m.margins or ()), m.effect, m.role, m.lane, m.source_kind,
            tuple((r.text, r.kind, r.src, r.e, r.a, r.b, r.ins, r.render_src, r.render_e) for r in m.runs),
            repr(sorted(m.overrides.items())) if m.overrides else "", repr(m.format) if m.format else "",
            bool(m.extra.get("source_style")), bool(m.extra.get("karaoke_off")), alias,
            (sp.get("display"), sp.get("label"), sp.get("character")) if include_actor else None,
            tuple((a.id, a.kind, a.text, a.status, repr(a.parts), repr(sorted((a.layout or {}).items())), repr(a.anchor))
                  for a in anns), ksig, m.lang, pinyin_mode)


def build_ass(doc: Document, sources: dict[str, Source], template: AssDocument, episode_dir: Path,
              opts: Optional[ExportOptions] = None, *, measure_cache: Optional[Path] = None,
              video_height: Optional[int] = None, cache: Optional[ExportCache] = None,
              character_book=None, runner=None) -> tuple[AssDocument, ExportReport]:
    opts = opts or ExportOptions()
    rep = ExportReport(hidden_groups=sum("hidden" in g.flags for g in doc.groups))
    rep.replacement_key = opts.display.replacements
    reference_rows = {}
    if opts.reference_enabled:
        from .reference_text import prepare_reference
        rep.reference = prepare_reference(doc, sources, episode_dir, opts.display, opts.reference_path, runner=runner)
        reference_rows = {row["member"]: row for row in rep.reference["rows"]}
        # External reference bytes may change between exports. Avoid retaining
        # layouts keyed only by the project revision/path; default previews keep
        # their existing per-member cache unchanged.
        cache = None
    from .people import file_name_speaker, file_name_labels
    source_labels = {sid: file_name_labels(src) for sid, src in sources.items()}
    if character_book is not None:
        character_book.set_sources(sources)
    out = AssDocument()
    tx, ty = opts.playres
    out.script_info = [(None, "; 双语字幕工坊（DuoSub）— 由工程导出，不要直接编辑（编辑请回到双语字幕工坊）"), ("Title", doc.episode),
                       ("ScriptType", "v4.00+"), ("WrapStyle", "0"), ("ScaledBorderAndShadow", "yes"),
                       ("YCbCr Matrix", "TV.709"), ("PlayResX", str(tx)), ("PlayResY", str(ty))]
    out.style_format = list(template.style_format)
    out.styles = [Style(dict(s.values)) for s in template.styles if not s.name.startswith("@")]
    template_res = opts.template_playres
    if template_res is None:
        template_res = template.playres if template.info("PlayResX") or template.info("PlayResY") else opts.playres
    template_x = template_res[0]
    if template_x != tx:
        sx = tx / template_x
        for style in out.styles:
            for key in ("MarginL", "MarginR", "Spacing"):
                value = style.num(key) * sx
                style.set(key, round(value, 2) if key == "Spacing" else round(value))
    template_names = {s.name for s in out.styles}
    if cache is not None:
        gkey = (opts.key(), video_height, tuple(s.to_line(out.style_format) for s in out.styles),
                measurement_environment())
        if cache.key != gkey:
            cache.clear()
            cache.key = gkey

    # ---- members coming from ASS sources at another resolution: rescale texts and source styles
    member_src: dict[str, str] = {}
    for m in doc.members.values():
        sid = source_id(m)
        if sid is not None:
            member_src[m.id] = sid
    rescale: dict[str, tuple[float, float, float]] = {}       # source id → (sx, sy, sb)
    style_alias: dict[tuple[str, str], str] = {}
    src_of_member: dict[str, str] = {}
    for sid, src in sources.items():
        sdoc = _source_doc_for(src, episode_dir, cache)
        if sdoc is None:
            continue
        owners = [mid for mid, s in member_src.items() if s == sid]
        if not owners:
            continue
        for mid in owners:
            src_of_member[mid] = sid
        tmp = AssDocument()
        tmp.script_info = list(sdoc.script_info)
        tmp.style_format = list(sdoc.style_format)
        tmp.styles = [Style(dict(s.values)) for s in sdoc.styles]
        if tmp.playres != (tx, ty):
            from ..ass.resample import resample

            old_x, old_y = tmp.playres
            sb = (ty / old_y) if tmp.scaled_border else ty / (video_height or ty)
            rescale[sid] = (tx / old_x, ty / old_y, sb)
            resample(tmp, tx, ty, video_height=video_height)
        used = {doc.members[mid].style for mid in owners if doc.members[mid].extra.get("source_style")}
        used.update(name for mid in owners for name in _reset_styles(doc.members[mid].text))
        for st in tmp.styles:
            if st.name not in used:
                continue
            name = st.name if st.name not in template_names else f"src.{st.name}"
            style_alias[(sid, st.name)] = name
            if out.style(name) is None:
                out.styles.append(st.copy(name))

    # ---- per member: from the cache, or laid out now
    ann_by_member: dict[str, list] = {}
    for a in doc.annotations.values():
        ann_by_member.setdefault(a.member, []).append(a)
    kidx = kara.KIndex(doc)
    kdefault = "kf"
    visible_mids = {mid for g in doc.groups if "hidden" not in g.flags for mid in g.members}
    views = {m.id: kara.effective(doc, m, idx=kidx, default_kind=kdefault)
             for m in doc.members.values() if m.id in visible_mids and (kidx.has(m) or kara.placeholder_eligible(doc, m))}
    for view in views.values():
        get_effect(view.kind)   # unknown presets are errors even on placeholder tracks
    for g in doc.groups:
        if "hidden" in g.flags:
            continue
        original = next((views[mid] for mid in g.members if doc.members[mid].lane == "original"
                         and mid in views and views[mid].kind not in INLINE), None)
        if original is not None and get_effect(original.kind).resolve(ktv.effect_params(original.params))["follow_translation"]:
            for mid in g.members:
                if doc.members[mid].lane == "translation" and mid in views and views[mid].kind in INLINE:
                    views[mid] = replace(views[mid], kind=original.kind, params=dict(original.params))
    slots = ktv.song_slots(doc, views)
    if slots:
        styles = {mid: style_alias.get((src_of_member.get(mid), doc.members[mid].style), doc.members[mid].style)
                  if doc.members[mid].extra.get("source_style") else doc.members[mid].style for mid in slots}
        with Measurer(out, cache_path=measure_cache, runner=runner) as measurer:
            slots = ktv.position_slots(doc, slots, out, styles, RubyGenerator(out, measurer), opts.ruby_gap)
        for mid, slot in slots.items():
            if mid == slot.original and ms_to_cs(slot.start) >= ms_to_cs(doc.members[mid].start):
                rep.warnings.append(f"{_fmt_ms(doc.members[mid].start)} 双行卡拉 OK：换行提前已缩短为 0 毫秒")
    results: dict[str, _MemberOut] = {}
    fades = inherited_fades(doc)
    jobs: list[tuple[str, str, RubyJob]] = []
    merged_jobs: list[str] = []
    gloss_jobs: list[str] = []
    for g in doc.groups:
        if "hidden" in g.flags:
            continue
        for mid in g.members:
            m = doc.members[mid]
            if m.role == "comment" and opts.mode == "release" and not opts.include_source_comments:
                continue
            sid = src_of_member.get(m.id)
            alias = style_alias.get((sid, m.style)) if (sid and m.extra.get("source_style")) else None
            anns = display_annotations(doc, m, sorted(ann_by_member.get(m.id, ()), key=lambda a: a.id))
            kview = views.get(mid)
            if kview is not None and kview.state == "none":
                kview = None
            actor = "" if file_name_speaker(m, source_labels) else None
            if opts.include_actor and character_book is not None:
                sp = m.speaker or {}
                actor = display_name(m, opts.display, character_book, opts.name_format)
            slot = slots.get(mid)
            fade = fades.get(mid)
            fade_raw = fade.raw if fade is not None else ""
            sig = (_member_sig(m, anns, alias, opts.include_actor, kara.view_signature(kview), opts.pinyin_mode), actor, slot, g.kind, fade_raw) if cache is not None else None
            if cache is not None:
                hit = cache.members.get(m.id)
                if hit is not None and hit[0] == sig:
                    results[m.id] = hit[1]
                    continue
            shown = m
            if fade_raw:
                shown = copy.copy(m)
                shown.overrides = {**m.overrides, "raw": str(m.overrides.get("raw") or "") + fade_raw}
            aliases = {name: target for (source, name), target in style_alias.items() if source == sid}
            if aliases:
                shown = copy.copy(shown)
                shown.runs = [r.copy(text=_alias_resets(r.text, aliases)) if run_source(m, r) == sid else r for r in m.runs]
            mo = _layout_member(doc, shown, anns, sid, alias, rescale, opts, cache, kview, out.style(alias or m.style), actor, slot, out.style_map() if opts.display.punctuation_layers_for(m)[1] else None,
                                reference_row=reference_rows.get(mid))
            rep.laid_out += 1
            results[m.id] = mo
            if mo.glosses and mo.event.kind == "Dialogue":
                gloss_jobs.append(m.id)
            if mo.merged and mo.event.kind == "Dialogue":
                merged_jobs.append(m.id)
            if cache is not None:
                cache.members[m.id] = (sig, mo)
            if opts.ruby and get_language(m.lang).annotation_system == "ruby" and mo.event.kind == "Dialogue":
                pieces = _ruby_pieces(m, mo, anns)
                if pieces:
                    mo.ruby_style = _derived(out, mo.event.style, "Ruby")
                    jobs.append((m.id, "ruby", RubyJob(mo.event.copy(), mo.ruby_style, pieces, offset=opts.ruby_offset,
                                                      max_width_percent=opts.ruby_width_percent(g.kind), prefer_center=True,
                                                      font_context=mo.event.copy(text=mo.ruby_font_context)
                                                      if mo.ruby_font_context is not None else None)))
            if opts.pinyin_mode != "off" and get_language(m.lang).annotation_system == "pinyin" and mo.event.kind == "Dialogue":
                pieces = _ruby_pieces(m, mo, anns, kind="pinyin", pinyin_mode=opts.pinyin_mode)
                if pieces:
                    pname = _pinyin_style(out, template, mo.event.style, mo.warnings)
                    if pname is not None:
                        jobs.append((m.id, "pinyin", RubyJob(mo.event.copy(), pname, pieces, offset=opts.pinyin_offset,
                                                          font_context=mo.event.copy(text=mo.ruby_font_context)
                                                          if mo.ruby_font_context is not None else None)))
    if cache is not None:
        live = set(doc.members)
        for mid in [k for k in cache.members if k not in live]:
            del cache.members[mid]

    if merged_jobs:
        _warn_merged_width(out, doc, results, merged_jobs, measure_cache, runner)

    # ---- generated ruby lines for the members laid out now
    if jobs:
        with Measurer(out, cache_path=measure_cache, runner=runner) as measurer:
            gen = RubyGenerator(out, measurer, gap=opts.ruby_gap, min_scale=opts.ruby_min_scale, overhang=opts.ruby_overhang)
            gen.run([j for _mid, _kind, j in jobs])
        for mid, kind, j in jobs:
            mo = results[mid]
            if kind == "ruby":
                mo.event = j.event
            for e, gm in zip(j.out, j.geoms):
                e.effect = ""
                events, geoms = (mo.pinyin, mo.pinyin_geoms) if kind == "pinyin" else (mo.ruby, mo.ruby_geoms)
                events.append(e)
                geoms.append(gm)
            mo.warnings.extend(j.warnings)

    if gloss_jobs:
        _generate_glosses(out, results, gloss_jobs, measure_cache, gap=opts.ruby_gap, runner=runner)

    advanced = [mid for mid, mo in results.items() if mo.kview is not None and mo.kview.kind not in INLINE
                and mo.kview.state != "none" and mo.event.kind == "Dialogue" and mo.fx_events is None]
    if advanced:
        for mid in advanced:
            if results[mid].ruby:
                _derived(out, results[mid].event.style, "Ruby")
        _generate_fx(out, doc, results, advanced, measure_cache, runner=runner)

    # ---- assemble: dialogue in document order, then the ruby lines (like the baseline)
    ruby_events: list[tuple[Event, dict]] = []
    for g in doc.groups:
        if "hidden" in g.flags:
            continue
        for mid in g.members:
            mo = results.get(mid)
            if mo is None:
                continue
            if mo.fx_events:
                if mo.fx_ruby:
                    _derived(out, mo.event.style, "Ruby")
                out.events.extend(e.copy() for e in mo.fx_events)
                rep.owners.extend(dict(o) for o in mo.fx_owners)
                rep.events += len(mo.fx_events)
                rep.karaoke_fx_lines += 1
                rep.karaoke_fx_events += len(mo.fx_events)
                rep.ruby_lines += sum(o["kind"] == "ruby" for o in mo.fx_owners)
            else:
                out.events.append(mo.event.copy())
                rep.owners.append({"member": mid, "group": g.id, "kind": "base", "ids": mo.ids,
                                   "source_text": mo.source_text,
                                   **({"replacement_ranges": [(mo.display_bounds[a], mo.display_ends[b])
                                        for a, b in mo.replacement_ranges],
                                       "replacement_deleted_ids": [mo.annotation_ids[i] for i in sorted(mo.replacement_deleted)]}
                                      if mo.replacements else {})})
                rep.events += 1
            rep.hidden_by_rules += mo.hidden_by_rules
            rep.replacement_lines += int(mo.replacements > 0)
            rep.replacement_matches += mo.replacements
            rep.replacement_deleted += len(mo.replacement_deleted)
            rep.zh_lines += int(mo.converted)
            rep.zh_script = opts.display.direction
            if opts.display.direction == "traditional":
                rep.zh_variant = opts.display.traditional_label
            rep.uncertain += mo.uncertain
            rep.positions_overridden += int(mo.overridden)
            if mo.untimed:
                m_ = doc.members[mid]
                rep.not_timed.append(f"{_fmt_ms(m_.start)}–{_fmt_ms(m_.end)} {visible_map(m_)[0].text[:24]}")
            if mo.karaoke_stripped:
                m_ = doc.members[mid]
                rep.karaoke_stripped.append(f"{_fmt_ms(m_.start)}–{_fmt_ms(m_.end)} {visible_map(m_)[0].text[:24]}")
            # Typed text runs only (gloss marks are tag runs); visible_map would itself read {…} as a tag.
            member = doc.members[mid]
            shown = "".join(r.text for r in member.runs if r.kind == "text"
                            or r.kind == "tag" and r.text in ("{@Raw=", "}"))
            if "{@Raw" in shown:
                shown = remove_private(shown)
            if "{" in _PRIVATE_MARK.sub("", shown) or "}" in _PRIVATE_MARK.sub("", shown):
                # Kept as typed: players read {…} as an override block, so the braces and their text vanish.
                rep.warnings.append(f"{_fmt_ms(doc.members[mid].start)} 这一行的大括号「{{」「}}」会被播放器当成标签，"
                                    f"括号里的字不会显示（需要时手动改成全角｛｝或 \\{{ \\}}）：{shown[:24]}")
            if mo.kview is not None:
                rep.karaoke_lines += 1
                if not mo.kview.complete or mo.kview.issues:
                    rep.karaoke_unfinished.append(f"{_fmt_ms(doc.members[mid].start)} {visible_map(doc.members[mid])[0].text[:24]}："
                                                  + ("；".join(mo.kview.issues) or "有占位拍"))
            rep.warnings.extend(mo.warnings)
            if mo.gloss_events:
                _derived(out, mo.event.style, "Gloss")
                for event in mo.gloss_events:
                    out.events.append(event.copy())
                    rep.owners.append({"member": mid, "group": g.id, "kind": "gloss"})
                    rep.events += 1
            if mo.ruby and not mo.fx_ruby:
                _derived(out, mo.event.style, "Ruby")
                for e, gm in zip(mo.ruby, mo.ruby_geoms):
                    ruby_events.append((e, {"member": mid, "group": g.id, "kind": "ruby", **gm}))
            if mo.pinyin:
                _pinyin_style(out, template, mo.event.style)
                for e, gm in zip(mo.pinyin, mo.pinyin_geoms):
                    ruby_events.append((e, {"member": mid, "group": g.id, "kind": "pinyin", **gm}))
        for cid in g.comments:
            c = doc.comments[cid]
            if c.kind in ("hidden_filler", "user_hidden"):
                rep.not_rendered.append(c.content)
    for e, owner in ruby_events:
        out.events.append(e.copy())
        rep.owners.append(owner)
        if owner["kind"] == "pinyin":
            rep.pinyin_lines += 1
        else:
            rep.ruby_lines += 1
    _finish_song_events(doc, out.events, rep.owners, results)
    if opts.mode == "release":
        from .export_presentation import finish_release
        finish_release(out, rep, opts.annotation_order)
        if rep.order_layer_events:
            rep.warnings.append(f"为保留原特效叠画次序，已重编 {rep.order_layer_events} 条输出事件的图层编号（工程原图层不变）。")
    used_styles = {e.style for e in out.events}
    used_styles.update(name for e in out.events for name in _reset_styles(e.text))
    out.styles = [s for s in out.styles if s.name in used_styles]
    if opts.blur > 0:
        _apply_blur(out, opts.blur)
    if rep.karaoke_fx_lines:
        rep.warnings.append(f"卡拉 OK 特效：{rep.karaoke_fx_lines} 行生成了 {rep.karaoke_fx_events} 条字幕行")
    if rep.reference is not None:
        from ..ass.text import plain_text
        rep.reference["rows"] = [row for row in rep.reference["rows"] if row["member"] in results]
        for row in rep.reference["rows"]:
            text = results[row["member"]].event.text
            if opts.mode == "release":
                from .export_presentation import clean_text
                text = clean_text(text).text
            row["final"] = plain_text(text, breaks="\n")
    return out, rep


# ASS events use centiseconds. Longer overlaps are intentional simultaneous lyrics.
SONG_OVERLAP_LIMIT_CS = 100


def _finish_song_events(doc, events, owners, results):
    """Adjust final copies only: cached layout and document times stay intact."""
    by_member = {}
    bases = {}
    for event, owner in zip(events, owners):
        mid = owner["member"]
        by_member.setdefault(mid, []).append(event)
        if results[mid].fx_events is None and owner["kind"] in ("base", "ruby"):
            event.text = split_karaoke_overrides(event.text)
        if owner["kind"] == "base" and event.kind == "Dialogue":
            bases[mid] = event

    def positioned(event):
        return line_tag(event.text, "pos") is not None or line_tag(event.text, "move") is not None

    def template_layout(mid):
        return not doc.members[mid].extra.get("source_style") and not positioned(bases[mid])

    lanes = {}
    for group in doc.groups:
        # Dialogue too: a sign converted to dialogue keeps the sign's layer for its translation, so the two
        # lines collide and libass pushes one past the other (top: the original above its translation).
        if group.kind not in ("song", "dialogue") or "hidden" in group.flags:
            continue
        originals = [mid for mid in group.members if mid in bases
                     and doc.members[mid].lane == "original" and template_layout(mid)]
        original_layers = {bases[mid].layer for mid in originals}
        for mid in group.members:
            if mid not in bases:
                continue
            base = bases[mid]
            member = doc.members[mid]
            if member.lane == "translation" and template_layout(mid) and base.layer in original_layers:
                layer = base.layer
                for event in by_member[mid]:
                    if event.layer == layer:
                        event.layer += 1
            if group.kind == "song":                    # dialogue may overlap on purpose (two speakers)
                lanes.setdefault(member.lane, []).append((mid, base))
    for lane in lanes.values():
        lane.sort(key=lambda pair: pair[1].start)
        for (mid, earlier), (_later_mid, later) in zip(lane, lane[1:]):
            overlap = earlier.end - later.start
            shorter = min(earlier.end - earlier.start, later.end - later.start)
            if (earlier.style == later.style and not positioned(earlier) and not positioned(later)
                    and 0 < overlap <= SONG_OVERLAP_LIMIT_CS and 2 * overlap < shorter
                    and earlier.start < later.start):
                for event in by_member[mid]:
                    event.end = min(event.end, later.start)


def _warn_merged_width(out, doc, results, mids, measure_cache, runner=None):
    from ..ruby.measure import layout_tokens

    # An unwrapped wider probe avoids clipping at the width being checked.
    probe = AssDocument.parse(out.serialize())
    probe.set_info("PlayResX", str(out.playres[0] * 2))
    texts = [(results[mid].event.style, "".join(raw for _kind, raw in layout_tokens(results[mid].event.text)))
             for mid in mids]
    with Measurer(probe, cache_path=measure_cache, wrap_style=2, runner=runner) as measurer:
        boxes = measurer.text_boxes(texts)
    for mid, box in zip(mids, boxes):
        ev = results[mid].event
        style = out.style(ev.style) or Style.new(ev.style)
        safe = max(0, out.playres[0] - (ev.margin_l or style.num("MarginL"))
                   - (ev.margin_r or style.num("MarginR")))
        if box is not None and box.width > safe:
            results[mid].warnings.append(f"{_fmt_ms(doc.members[mid].start)} 合成一行后太长，可能超出画面")


def _generate_fx(out, doc, results, mids, measure_cache, runner=None):
    """Place isolated glyphs using full-line and isolated libass ink bearings."""
    from ..ass.state import PAINT, TRANSFORM, state_at
    from ..ass.tags import FONT_TAGS
    from ..ass.text import VisibleText, line_tag
    from ..ruby.measure import LineRequest

    # Subpixel bearings keep isolated glyphs within 1 px of the original line.
    with Measurer(out, cache_path=measure_cache, raster_scale=4, runner=runner) as measurer:
        gen = RubyGenerator(out, measurer)
        for mid in mids:
            mo, m = results[mid], doc.members[mid]
            effect = get_effect(mo.kview.kind)
            params = effect.resolve(ktv.effect_params(mo.kview.params))
            if mo.ktv_slot is not None:
                params["lead_in_ms"] = max(0, m.start - mo.event.start*10)
            rows = [(mo.event, "base", None)]
            if params["follow_furigana"]:
                rows += [(e, "ruby", gm) for e, gm in zip(mo.ruby, mo.ruby_geoms)]
            glyphs = []
            aliases = {}
            if mo.display_bounds:
                for i, identity in enumerate(mo.annotation_ids):
                    if identity is not None:
                        for j in range(mo.display_bounds[i], mo.display_ends[i+1]):
                            aliases.setdefault(j, []).append(identity)
            for ev, kind, gm in rows:
                vt = VisibleText(ev.text)
                an, x = gen._anchor(ev)
                boxes = measurer.line_char_boxes([LineRequest(ev.style, ev.text, an, x)])[0]
                top = gen.block_top(ev, vt.line_count)
                fonts = []
                for i, ch in enumerate(vt.chars):
                    st, _ = state_at(ev.text, i, FONT_TAGS | set(PAINT) | set(TRANSFORM) |
                                     {"u", "s", "bord", "shad", "xbord", "ybord", "xshad", "yshad"})
                    tags = "".join(t.raw for t in st.values())
                    fonts.append((tags, st, ev.text[ch.raw_start:ch.raw_end]))
                probes = [LineRequest(ev.style, "{" + tags + "}" + char, 7, 0) for tags, _, char in fonts]
                isolated = measurer.line_char_boxes(probes)
                ruby_times = {}
                if kind == "ruby":
                    cursor, duration, visible = ev.start*10, 0, 0
                    for tok in tokenize(ev.text):
                        if tok.kind == "tags":
                            for tag in tok.tags:
                                if tag.name in KARAOKE_TAGS:
                                    cursor += duration
                                    duration = int(tag.num() or 0)*10
                        elif tok.kind in ("text", "hardspace", "softbreak"):
                            count = len(tok.raw) if tok.kind == "text" else 1
                            for k in range(count):
                                ruby_times[visible+k] = (cursor, cursor+duration)
                            visible += count
                source_ids = visible_map(m)[1]
                for i, (box, probe, (tags, st, char)) in enumerate(zip(boxes, isolated, fonts)):
                    if box is None or not probe or probe[0] is None:
                        continue
                    local = probe[0]
                    times = []
                    if kind == "base":
                        identity = mo.ids[i] if i < len(mo.ids) else None
                        anchors = aliases.get(i, [identity] if identity is not None else [])
                        times = [(j, b.t0, b.t1) for j, b in enumerate(mo.kview.beats)
                                 if any(anchor in source_ids[b.a:b.b] for anchor in anchors)]
                    elif i in ruby_times:
                        a, z = ruby_times[i]
                        j = next((j for j, b in enumerate(mo.kview.beats) if abs(b.t0-a) <= 10), 0)
                        times = [(j, a, z)]
                    if not times:
                        times = [(0, m.start, m.end)]
                    owner = {"member": mid, "group": m.group, "kind": kind,
                             "ids": [mo.ids[i]] if kind == "base" and i < len(mo.ids) else [],
                             "source_text": mo.source_text}
                    if mo.replacements:
                        owner["replacement_ranges"] = [(0, 1)]
                    gx, gy = box.x0-local.x0, top+box.y0-local.y0
                    if gm is not None:
                        owner.update(ann=gm.get("ann"), x=(box.x0+box.x1)/2, y=top+box.y1,
                                     w=box.width, h=box.height)
                    org = line_tag(ev.text, "org")
                    if org is not None:
                        tags += org.raw
                    elif any(k in st for k in TRANSFORM) or (out.style(ev.style) or Style.new(ev.style)).num("Angle"):
                        ox, oy = gen.origin(ev, vt.line_count)
                        tags += f"\\org({ox:.2f},{oy:.2f})"
                    style = out.style(ev.style) or Style.new(ev.style)
                    move = line_tag(ev.text, "move")
                    mv = move.float_args() if move is not None else None
                    movement = None
                    if mv and len(mv) >= 4:
                        movement = (mv[2]-mv[0], mv[3]-mv[1], int(mv[4]) if len(mv) >= 6 else 0,
                                    int(mv[5]) if len(mv) >= 6 else ev.duration*10)
                    glyphs.append(FxGlyph(ev, (box.x0, top+box.y0, box.x1, top+box.y1), tags, char, gx, gy, times, owner,
                                          st["fscx"].num() if "fscx" in st else style.scale_x,
                                          st["fscy"].num() if "fscy" in st else style.scale_y, movement,
                                          tuple(identity for identity in (aliases.get(i, owner["ids"]) if gm is None else
                                                mo.ids[gm["base_start"]:gm["base_end"]]) if identity is not None)))
            an, _ = gen._anchor(mo.event)
            ctx = FxContext(mid, max(m.start, mo.event.start*10) if mo.ktv_slot is not None else m.start,
                            mo.event.end*10 if mo.ktv_slot is not None else m.end,
                            out.style(mo.event.style) or Style.new(mo.event.style),
                            glyphs, params, an, gen.origin(mo.event, VisibleText(mo.event.text).line_count),
                            FxContext.seeded(mid, {"preset": effect.name, **params}))
            try:
                mo.fx_events = effect.generate(ctx)
                mo.warnings.extend(ctx.warnings)
                mo.fx_owners = ctx.owners
                if mo.ktv_slot is not None:
                    # Clip generated tails/particles and entry variants to the occupied slot.
                    pairs = [(e.copy(start=max(e.start, mo.event.start), end=min(e.end, mo.event.end)), o)
                             for e, o in zip(mo.fx_events, mo.fx_owners)
                             if min(e.end, mo.event.end) > max(e.start, mo.event.start)]
                    mo.fx_events = [e for e, _ in pairs]
                    mo.fx_owners = [o for _, o in pairs]
                mo.fx_ruby = params["follow_furigana"] and bool(mo.fx_events)
            except EventLimit:
                mo.fx_events = []
                mo.warnings.append(f"{_fmt_ms(m.start)} 这一行特效太复杂，已改用渐变扫过")


_PRIVATE_MARK = re.compile(r"\{@[^{}]*\}")   # gloss and other private marks are not literal braces


def _fmt_ms(ms: int) -> str:
    sign, ms = ("-" if ms < 0 else ""), abs(int(ms))
    return f"{sign}{ms // 3600000}:{ms // 60000 % 60:02d}:{ms % 60000 / 1000:05.2f}"


def _generate_glosses(out, results, mids, measure_cache, *, gap=2.0, runner=None):
    from ..ass.state import restore_tags, state_at
    from ..ass.text import VisibleText, line_tag
    from ..ass.tags import parse_block
    from ..ruby.measure import Box, LineRequest

    with Measurer(out, cache_path=measure_cache, runner=runner, raster_scale=4) as measurer:
        gen = RubyGenerator(out, measurer)
        requests, pending, ink_requests = [], [], []
        probe_x = out.playres[0] / 2.0
        for mid in mids:
            ev = results[mid].event
            an, x = gen._anchor(ev)
            requests.append(LineRequest(ev.style, ev.text + r"\N{\r}" + ev.text, an, x))
        for mid, measured in zip(mids, measurer.line_char_boxes(requests)):
            mo = results[mid]
            ev = mo.event
            vis = VisibleText(ev.text)
            count = len(vis.text)
            boxes = measured[:count]
            top = gen.block_top(ev, vis.line_count)
            pair = next(((a, b) for a, b in zip(boxes, measured[count:])
                         if a is not None and b is not None), None)
            if pair is not None:
                height = pair[1].y0 - pair[0].y0
                reference_height = gen._metrics[gen.metric_key(ev)][0] * vis.line_count
                an, _x = gen._anchor(ev)
                top += (reference_height - height) * (1.0, 0.5, 0.0)[(an - 1) // 3]
            base = out.style(ev.style) or Style.new(ev.style)
            style = _derived(out, ev.style, "Gloss")
            for gloss in mo.glosses:
                effective_style = base
                cursor = 0
                for token in tokenize(ev.text):
                    if token.kind == "tags":
                        for tag in token.tags:
                            if tag.name == "r":
                                effective_style = out.style(tag.arg) or base
                    elif token.kind in ("text", "hardspace", "softbreak"):
                        cursor += len(token.raw) if token.kind == "text" else 1
                        if cursor > gloss.start:
                            break
                span = [b for b in boxes[gloss.start:gloss.end] if b is not None]
                if not span:
                    mo.warnings.append(f"无法测量：{gloss.target}")
                    continue
                box = Box(min(b.x0 for b in span), max(b.x1 for b in span),
                          top + min(b.y0 for b in span), top + max(b.y1 for b in span))
                follow, needs_org, _dropped = gen._follow(ev.text, gloss.start)
                inherited = restore_tags(ev.text, gloss.start,
                                         ("fn", "b", "i", "u", "s", "fsp", "fscx", "fscy", "frz", "frx", "fry", "fax", "fay",
                                          "1c", "2c", "3c", "4c", "alpha", "blur", "be"), effective_style)
                state, _ = state_at(ev.text, gloss.start, ("fs",))
                size = state["fs"].num() if "fs" in state else effective_style.fontsize
                size = size if size is not None and size > 0 else effective_style.fontsize
                actual = gloss.size if gloss.size is not None else size * 0.5
                # Like furigana, the edge comes from the .Gloss style; only the line's own \bord / \shad
                # (or a \r to another style) is carried over, scaled with the size.
                own, _ = state_at(ev.text, gloss.start, ("bord", "xbord", "ybord", "shad", "xshad", "yshad"))
                restyled = effective_style is not base
                edges = [n for group, names in ((("bord", "xbord", "ybord"), ("xbord", "ybord")),
                                               (("shad", "xshad", "yshad"), ("xshad", "yshad")))
                         if restyled or any(k in own for k in group) for n in names]
                borders = restore_tags(ev.text, gloss.start, edges, effective_style) if edges else ""
                borders = "".join(f"\\{t.name}{_num((t.num() or 0) * actual / size)}" for t in parse_block(borders))
                tags = inherited + follow + borders + f"\\fs{_num(actual)}"
                if gloss.font is not None:
                    tags += "\\fn" + gloss.font
                if gloss.color is not None:
                    rgb = gloss.color[1:]
                    tags += "\\1c&H" + rgb[4:6] + rgb[2:4] + rgb[0:2] + "&"
                if gloss.scale is not None:
                    tags += f"\\fscx{_num(gloss.scale)}"
                tags += "".join(t.raw for t in (line_tag(ev.text, name) for name in ("fad", "fade", "clip", "iclip"))
                                if t is not None)
                an = 10 - gloss.position
                measured_text = "{" + tags + "}" + gloss.text
                # Repeat the text to measure its block height with its actual font and overrides.
                ink_requests.append(LineRequest(style, measured_text + r"\N" + measured_text, an, probe_x))
                pending.append((mo, ev, gloss, box, boxes, vis, style, tags, an, needs_org))
        for item, measured in zip(pending, measurer.line_char_boxes(ink_requests)):
            mo, ev, gloss, box, boxes, vis, style, tags, an, needs_org = item
            count = len(VisibleText(gloss.text).text)
            ink = [b for b in measured[:count] if b is not None]
            pair = next(((a, b) for a, b in zip(measured[:count], measured[count:])
                         if a is not None and b is not None), None)
            if not ink or pair is None:
                mo.warnings.append(f"无法测量：{gloss.text}")
                continue
            height = pair[1].y0 - pair[0].y0
            top = -height * (1.0, 0.5, 0.0)[(an - 1) // 3]
            relative = Box(min(b.x0 for b in ink) - probe_x, max(b.x1 for b in ink) - probe_x,
                           top + min(b.y0 for b in ink), top + max(b.y1 for b in ink))
            x, y = gloss_anchor(box, relative, gloss.position, gap + gloss.distance)
            if any(t.name == "fay" for token in tokenize(ev.text) for t in token.tags):
                y += gen._fay_shift(ev.text, boxes, vis, gloss.start, x)
            move = line_tag(ev.text, "move")
            mv = move.float_args() if move is not None else []
            position = f"\\an{an}"
            if len(mv) >= 4:
                timing = "".join("," + _num(v) for v in mv[4:6])
                position += f"\\move({_num(x)},{_num(y)},{_num(x+mv[2]-mv[0])},{_num(y+mv[3]-mv[1])}{timing})"
            else:
                position += f"\\pos({_num(x)},{_num(y)})"
            if needs_org or any(t.name == "frz" and t.num() for t in parse_block(tags)):
                org = line_tag(ev.text, "org")
                ox, oy = gen.origin(ev, vis.line_count)
                position += org.raw if org is not None else f"\\org({_num(ox)},{_num(oy)})"
            mo.gloss_events.append(ev.copy(style=style, name="", effect="", text="{" + position + tags + "}" + gloss.text))


def _layout_member(doc: Document, m: Member, anns: list, sid: Optional[str], alias: Optional[str],
                   rescale: dict[str, tuple[float, float, float]], opts: ExportOptions,
                   cache: Optional[ExportCache], kview=None, style_obj=None, actor=None, ktv_slot=None, styles=None,
                   *, reference_row=None) -> _MemberOut:
    shown = m
    hidden = 0
    punctuation, fonts = opts.display.punctuation_layers_for(m)
    has_punctuation = punctuation.active or bool(fonts)
    if opts.display.active and not has_punctuation and sid not in rescale:
        runs, n_hidden = display_runs(m, opts.display)
        if n_hidden:
            shown = copy.copy(m)
            shown.runs = runs
            hidden = n_hidden
    if sid in rescale:
        sx, sy, sb = rescale[sid]
        key = (sid, tuple((run_source(m, r), r.text) for r in m.runs), sx, sy, sb)
        text = cache.rescaled.get(key) if cache is not None else None
        if text is None:
            # Source tags use source coordinates; inserted tags already use episode coordinates.
            text = "".join(scale_text(part, sx, sy, sb) if sourced else part
                           for sourced, runs in groupby(m.runs, key=lambda r: run_source(m, r) is not None)
                           for part in ["".join(r.text for r in runs)])
            if cache is not None:
                cache.rescaled[key] = text
    else:
        text = shown.text
    _vt, ids = visible_map(shown)
    warnings = []
    glosses = parse_glosses(text, warnings)
    conversion_changed = gloss_changed = False
    for gloss in glosses:
        if opts.display.gloss_for(m, gloss.text):
            note = opts.display.convert(gloss.text)
            gloss_changed |= note != gloss.text
            gloss.text = note
    warnings.extend(f"Raw 写法有误：{mark}（应写为 {{@Raw=文字}}，文字里不能含 }}）"
                    for mark in raw_mark_errors(text))
    text = remove_private(text, keep_raw=True)
    annotation_ids = ids
    bounds, ends = [], []
    from .replacements import replace_ass_text
    replacements = replace_ass_text(text, opts.display.replacements, lane=m.lane)
    if replacements.matches:
        text = replacements.text
        bounds, ends = replacements.bounds, list(replacements.bounds)
        if reference_row is not None:
            from ..ass.text import plain_text
            generated = display_text(text, opts.display)[0] if opts.display.converted_for(m) else text
            reference_row.update(strategy="replacement", generated=plain_text(generated, breaks="\n"),
                reason="已应用工程批量替换；使用替换后的正文，避免原配字幕恢复已替换或删除的文字。")
    merged = opts.display.merge_for(m) and any(t.kind == "break" and not t.protected for t in tokenize(text))
    converted = opts.display.converted_for(m)
    reference_applied = False
    if converted and reference_row is not None and reference_row["strategy"] in ("reference", "merged"):
        from .reference_text import _canonical, replace_visible
        try:
            if (kview is not None or (opts.pinyin_mode != "off" and any(a.kind == "pinyin" for a in anns))):
                from ..ass.text import VisibleText
                if _canonical(VisibleText(text).text) != _canonical(reference_row["generated"]):
                    raise ValueError("原配词语不同，保留当前正文及已有逐字注音／卡拉 OK 对应后自动转换。")
            replaced, bounds = replace_visible(text, reference_row["generated"])
        except ValueError as exc:
            reference_row.update(strategy="opencc", generated=reference_row["automatic"], reason=str(exc))
        else:
            conversion_changed |= replaced != text
            text, ends = replaced, bounds
            reference_applied = True
            from ..ass.text import plain_text
            reference_row["generated"] = plain_text(text, breaks="\n")
    font_positions: dict[str, set[int]] = {}
    if has_punctuation:
        if converted and not reference_applied:
            converted_text, converted_bounds = display_text(text, opts.display)
            conversion_changed |= converted_text != text
            text = converted_text
            ends = [converted_bounds[i] for i in ends] if bounds else converted_bounds
            bounds = [converted_bounds[i] for i in bounds] if bounds else converted_bounds
        text, punctuation_bounds, punctuation_ends, font_positions, hidden = punctuation_layout(
            text, punctuation, hide_final_period=(opts.display.hide_final_period
                and m.lane == "original" and m.role in ("dialogue", "top", "alt")))
        ends = [punctuation_ends[i] for i in ends] if bounds else punctuation_ends
        bounds = [punctuation_bounds[i] for i in bounds] if bounds else punctuation_bounds
        if merged:
            text, merge_bounds, merge_ends = single_line_text(text)
            font_positions = {category: {merge_bounds[i] for i in positions}
                              for category, positions in font_positions.items()}
            bounds = [merge_bounds[i] for i in bounds]
            ends = [merge_ends[i] for i in ends]
    else:
        if merged:
            text, merge_bounds, merge_ends = single_line_text(text)
            if bounds:
                bounds = [merge_bounds[i] for i in bounds]
                ends = [merge_ends[i] for i in ends]
            else:
                bounds, ends = merge_bounds, merge_ends
        if converted and not reference_applied:
            converted_text, conversion_bounds = display_text(text, opts.display)
            conversion_changed |= converted_text != text
            text = converted_text
            if bounds:
                bounds = [conversion_bounds[i] for i in bounds]
                ends = [conversion_bounds[i] for i in ends]
            else:
                bounds = conversion_bounds
                ends = conversion_bounds
    if bounds:
        for gloss in glosses:
            gloss.start, gloss.end = bounds[gloss.start], ends[gloss.end]
        mapped_ids = [None] * bounds[-1]
        for i, identity in enumerate(ids):
            for j in range(bounds[i], ends[i + 1]):
                if mapped_ids[j] is None:
                    mapped_ids[j] = identity
        ids = mapped_ids
    text = remove_private(text)
    if kview is not None:            # SubKit karaoke replaces any \k the source line had (the source is unchanged)
        text = remove_tags_named(text, KARAOKE_TAGS)
        inserts = kara.base_inserts(kview, annotation_ids, ms_to_cs(ktv_slot.start if ktv_slot is not None else m.start),
                                    display_starts=bounds, display_ends=ends)
        if bounds:
            mapped = {}
            for i, value in inserts.items():
                at = bounds[i]
                mapped[at] = mapped.get(at, "") + value
            inserts = mapped
        text = insert_at_visible(text, inserts)
    source_text = text
    text, overridden = apply_overrides(text, m.overrides)
    if m.format:                          # part-of-line formatting: open the span, restore after it
        from . import textformat

        text = textformat.apply_spans(m, ids, text, style_obj)
        source_text = textformat.apply_spans(m, ids, source_text, style_obj)
    ruby_font_context = text if fonts else None
    if fonts and font_positions:
        text = punctuation_fonts(text, font_positions, fonts, style_obj, styles or {})
        source_text = punctuation_fonts(source_text, font_positions, fonts, style_obj, styles or {})
    stripped = False
    if kview is None and m.extra.get("karaoke_off") and any(
            tag.name in KARAOKE_TAGS for tok in tokenize(text) if tok.kind == "tags" for tag in tok.tags):
        text = remove_tags_named(text, KARAOKE_TAGS)
        source_text = remove_tags_named(source_text, KARAOKE_TAGS)
        stripped = True
    style = alias or m.style
    name = ""
    if opts.include_actor and m.speaker:
        name = actor if actor is not None else display_name(m, opts.display, format=opts.name_format)
    start, end = max(0, ms_to_cs(m.start)), max(0, ms_to_cs(m.end))
    if ktv_slot is not None:
        start, end = max(0, ms_to_cs(ktv_slot.start)), min(end, max(0, ms_to_cs(ktv_slot.end)))
        text, _ = apply_overrides(text, {"an": ktv_slot.an, "pos": [ktv_slot.x, ktv_slot.y]})
        if kview is None or kview.kind == "none":
            from ..karaoke_fx.base import color
            st = style_obj or Style.new(style)
            wait = max(0, ms_to_cs(m.start)*10 - start*10)
            text = (f"{{\\1c{color(st.get('SecondaryColour'))}\\t({wait},{wait+1},"
                    f"\\1c{color(st.get('PrimaryColour'))})}}" + text)
    kind = "Comment" if (m.role == "comment" or m.source_kind == "Comment") else "Dialogue"
    untimed = kind == "Dialogue" and (end <= start or max(0, ms_to_cs(m.end)) <= max(0, ms_to_cs(m.start)))
    if untimed:
        kind = "Comment"
    margins = list(m.margins or [0, 0, 0])
    if sid in rescale:
        sx, sy, _sb = rescale[sid]
        margins = [round(margins[0] * sx), round(margins[1] * sx), round(margins[2] * sy)]
    ev = Event(kind=kind, layer=m.layer, start=start, end=end, style=style, name=name,
               margin_l=margins[0], margin_r=margins[1],
               margin_v=margins[2], effect=m.effect or "", text=text)
    return _MemberOut(event=ev, source_text=source_text, ruby_font_context=ruby_font_context, ids=ids, hidden_by_rules=hidden, overridden=overridden, kview=kview, untimed=untimed,
                      karaoke_stripped=stripped, converted=conversion_changed or (kind == "Dialogue" and gloss_changed),
                      display_bounds=bounds, display_ends=ends,
                      annotation_ids=annotation_ids, merged=merged,
                      ktv_slot=ktv_slot, glosses=glosses, warnings=warnings,
                      replacements=replacements.matches, replacement_deleted=replacements.deleted,
                      replacement_ranges=replacements.ranges)


def _derived(out: AssDocument, base: str, suffix: str) -> str:
    name = f"{base}.{suffix}"
    if out.style(name) is None:
        b = out.style(base)
        if suffix == "Gloss":
            derived = (b or Style.new(base)).copy(name)
            derived.set("Fontsize", (b or Style.new(base)).fontsize * 0.5)
            derived.set("Outline", round(max(0.0, derived.outline * 0.6), 2))   # as derive_ruby_style
            derived.set("Shadow", round(derived.shadow * 0.5, 2))
            out.styles.append(derived)
            return name
        family = out.style(f"{base.split('.')[0]}.{suffix}")
        if family is not None:
            out.styles.append(family.copy(name))
        elif b is not None:
            out.styles.append(derive_ruby_style(b, name, ratio=0.5 if suffix != "Gloss" else 0.45))
    return name


def _pinyin_style(out: AssDocument, template: AssDocument, base: str,
                  warnings: Optional[list[str]] = None) -> Optional[str]:
    name = f"{base}.Pinyin"
    if template.style(name) is not None:
        return name
    fallback = template.style("CN.Pinyin")
    if warnings is not None:
        warnings.append(f"模板缺少拼音样式：{name}" + ("；使用 CN.Pinyin。" if fallback is not None else "；未导出拼音。"))
    if fallback is not None:
        if out.style(name) is None:
            out.styles.append(fallback.copy(name))
        return name
    return None


def _ruby_pieces(m: Member, mo: _MemberOut, anns: list, *, kind: str = "ruby",
                 pinyin_mode: str = "PY") -> list[RubyPiece]:
    ids = mo.annotation_ids
    pieces: list[RubyPiece] = []
    for a in anns:
        if a.kind != kind or a.status == "hidden" or not a.text:
            continue
        pos = anchor_positions(a, ids)
        if pos is None:
            mo.warnings.append(f"注音找不到底字：{a.base}（{a.text}）")
            continue
        if a.status == "uncertain":
            mo.uncertain += 1
        scale = (a.layout or {}).get("scale")
        parts = (a.parts or [[0, pos[1] - pos[0], a.text]]) if kind == "pinyin" else pieces_of(a)
        for k, (pa, pb, text) in enumerate(parts):          # kanji blocks only: okurigana stays without ruby
            if kind == "pinyin":
                text = " ".join(display_pinyin(syl, pinyin_mode) for syl in text.split())
            kt = kara.ruby_text(mo.kview, a.id, k, text, mo.event.start if mo.ktv_slot is not None else ms_to_cs(m.start)) if kind == "ruby" and mo.kview is not None else None
            start, end = pos[0] + pa, pos[0] + pb
            if mo.replacement_deleted.intersection(range(start, end)):
                mo.warnings.append("批量替换删除了注音底字，这段注音未显示（工程里的原注音保留）。")
                continue
            if mo.display_bounds:
                start, end = mo.display_bounds[start], mo.display_ends[end]
                if end <= start:
                    if mo.replacements:
                        continue
                    start = max(0, start - 1)
                    end = start + 1
            if kind == "ruby" and any(gloss.position in (7, 8, 9) and gloss.start < end and start < gloss.end
                                      for gloss in mo.glosses):
                continue
            pieces.append(RubyPiece(start, end, text, scale, tag=a.id, k_text=kt))
    pieces.sort(key=lambda p: p.start)
    return pieces


def _apply_blur(out: AssDocument, value: float) -> None:
    from ..ass.text import tokenize

    for ev in out.events:
        if ev.kind != "Dialogue":
            continue
        toks = tokenize(ev.text)
        if any(t.kind == "tags" and any(tag.name in ("blur", "be") for tag in t.tags) for t in toks):
            continue
        tag = f"\\blur{value:g}"
        ev.text = ("{" + tag + ev.text[1:]) if (toks and toks[0].kind == "tags") else ("{" + tag + "}" + ev.text)
