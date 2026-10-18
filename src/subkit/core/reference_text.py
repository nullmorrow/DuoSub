"""Export-only Chinese reference decisions; the document and its sources stay immutable.

Three texts have distinct jobs: an original source-language baseline, today's
editable text, and its original counterpart. Only provable one-event members
participate. Missing/ambiguous provenance is an ordinary, reported conversion
fallback, never a reason to replace an edit with an older translation.
"""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from ..ass.text import VisibleText, plain_text, tokenize
from .display import display_text
from .reference_sources import resolve_reference
from .render_origin import source_events


def _check(runner):
    if runner is not None:
        runner.check_cancelled()


def _flat(text: str) -> str:
    return text.replace("\n", "")


@lru_cache(maxsize=1)
def _canonical_converter():
    from opencc import OpenCC
    return OpenCC("t2s")


def _canonical(text: str) -> str:
    # Used to align the two originals, never as the exported reference text.
    return _canonical_converter().convert(text)


def _boundary_map(original: str, counterpart: str) -> dict[int, int]:
    a, b = _canonical(original), _canonical(counterpart)
    if len(a) != len(original) or len(b) != len(counterpart):
        return {}
    if a == b:
        return {i: i for i in range(len(a) + 1)}
    candidates = defaultdict(set)
    for tag, i, j, x, y in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            continue
        for k in range(j - i + 1):
            candidates[i + k].add(x + k)
    return {i: next(iter(values)) for i, values in candidates.items() if len(values) == 1}


def merge_reference(baseline: str, current: str, reference: str, convert) -> tuple[str, str, str]:
    """Reuse unedited counterpart spans; convert the user's edits at exact boundaries.

    All arguments are visible text without hard line breaks. When a patch cannot
    be located on both sides, the complete current sentence is converted instead.
    """
    automatic = convert(current)
    if current == baseline:
        return reference, "reference", "正文未改动，复用可靠对应的原配字幕；时间轴沿用当前工程。"
    ops = SequenceMatcher(None, baseline, current, autojunk=False).get_opcodes()
    if not any(tag == "equal" and j > i for tag, i, j, _x, _y in ops):
        return automatic, "opencc", "整句正文已改写，自动转换当前正文，避免恢复原版译文。"
    mapping = _boundary_map(baseline, reference)
    patches = []
    for tag, i, j, x, y in ops:
        if tag == "equal":
            continue
        if i not in mapping or j not in mapping or mapping[i] > mapping[j]:
            return automatic, "opencc", "改动边界无法在原配正文中唯一定位，自动转换当前正文。"
        left, right = mapping[i], mapping[j]
        if _canonical(baseline[i:j]) != _canonical(reference[left:right]):
            return automatic, "opencc", "改动区域与原配译法不同，无法可靠迁移，自动转换当前正文。"
        # An edit touching a different reference phrase can otherwise create
        # hybrid words (軟體 -> 软件, editing 軟 to 韌 must not produce 韧件).
        # Require an unchanged neighbouring glyph on both sides of a patch.
        if ((i > 0 and (mapping.get(i - 1) != left - 1 or
                        _canonical(baseline[i - 1:i]) != _canonical(reference[left - 1:left]))) or
                (j < len(baseline) and (mapping.get(j + 1) != right + 1 or
                        _canonical(baseline[j:j + 1]) != _canonical(reference[right:right + 1])))):
            return automatic, "opencc", "改动紧邻不同译法的区域，无法可靠迁移词语，自动转换当前正文。"
        # A deletion in repeated text has several equally valid diff positions.
        # Do not silently choose between distinct traditional spellings.
        old = baseline[i:j]
        if old and baseline.count(old) > 1:
            versions = set()
            at = baseline.find(old)
            while at >= 0:
                if at in mapping and at + len(old) in mapping:
                    versions.add(reference[mapping[at]:mapping[at + len(old)]])
                at = baseline.find(old, at + 1)
            if len(versions) > 1:
                return automatic, "opencc", "重复文字存在多种原配字形，改动位置不唯一，自动转换当前正文。"
        patches.append((left, right, convert(current[x:y])))
    result = reference
    last = len(reference) + 1
    for left, right, value in reversed(patches):
        if right > last:
            return automatic, "opencc", "改动区间在原配字幕中重叠，自动转换当前正文。"
        result = result[:left] + value + result[right:]
        last = left
    return result, "merged", "保留原配字幕未改动的部分，仅将当前正文的修改自动转换后迁移。"


def replace_visible(text: str, replacement: str) -> tuple[str, list[int]]:
    """Replace glyphs while retaining current ASS tags and character boundaries.

    A length-changing edit must stay inside one ordinary text token. Crossing a
    style/karaoke/break/Raw boundary is deliberately rejected, so downstream span
    formatting and annotation anchors cannot acquire guessed associations.
    """
    vt = VisibleText(text)
    if replacement == vt.text:
        return text, list(range(len(vt) + 1))
    if any(ch in replacement for ch in "{}\\\r\n\x00"):
        raise ValueError("原配正文含 ASS 结构字符，保留当前正文进行转换。")
    tokens = tokenize(text)
    if any(t.protected or t.kind == "drawing" for t in tokens):
        raise ValueError("当前行含 Raw 或绘图，保留其范围及当前正文进行转换。")
    token_at = {}
    for ti, token in enumerate(tokens):
        if token.kind == "text":
            for pos in range(token.start, token.end):
                token_at[pos] = ti
    chunks = [ch.ch for ch in vt.chars]
    counts = [1] * len(chunks)
    for tag, i, j, x, y in SequenceMatcher(None, vt.text, replacement, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        value = replacement[x:y]
        if j - i == y - x:
            for k, ch in enumerate(value, i):
                if vt.chars[k].raw_start not in token_at and ch != vt.chars[k].ch:
                    raise ValueError("原配正文会改变硬空格或软换行，自动转换当前正文。")
                chunks[k] = ch
            continue
        if i == j:
            if not vt.chars:
                raise ValueError("没有可定位的正文。")
            pos = min(i, len(vt.chars) - 1)
            owner = token_at.get(vt.chars[pos].raw_start)
            if owner is None or (0 < i < len(vt.chars) and
                                token_at.get(vt.chars[i - 1].raw_start) != owner):
                raise ValueError("原配新增文字跨越特效或换行边界，自动转换当前正文。")
            if i == len(vt.chars):
                chunks[-1] += value
                counts[-1] += len(value)
            else:
                chunks[i] = value + chunks[i]
                counts[i] += len(value)
            continue
        owners = {token_at.get(c.raw_start) for c in vt.chars[i:j]}
        if None in owners or len(owners) != 1:
            raise ValueError("原配字数变化跨越特效或换行边界，自动转换当前正文。")
        for k in range(i, j):
            a = (k - i) * len(value) // (j - i)
            b = (k + 1 - i) * len(value) // (j - i)
            chunks[k] = value[a:b]
            counts[k] = b - a
    out, at = [], 0
    for ch, value in zip(vt.chars, chunks):
        out.append(text[at:ch.raw_start])
        # Preserve the literal spelling of \h / \n when it is unchanged.
        out.append(text[ch.raw_start:ch.raw_end] if ch.ch == value else value)
        at = ch.raw_end
    out.append(text[at:])
    bounds = [0]
    for count in counts:
        bounds.append(bounds[-1] + count)
    return "".join(out), bounds


def prepare_reference(doc, sources, episode_dir: Path, rules, reference_path=None, *, runner=None) -> dict:
    """Build a detached audit payload. It also supplies export-only replacements."""
    data = dict(schema=1, episode=doc.episode, direction=rules.direction,
                sources=[], rows=[], issues=[], protected_input_paths=[])
    protected = {str(Path(s.path)) for s in sources.values() if s.path}
    protected.update(str(Path(episode_dir) / s.copy) for s in sources.values() if s.copy)
    if reference_path:
        protected.add(str(Path(reference_path)))
    memberships = defaultdict(set)
    origins = {}
    event_ids = {sid: {event.i for event in source.events} for sid, source in sources.items()}
    stones = defaultdict(set)
    for stone in doc.tombstones:
        if stone.member:
            stones[stone.member].add((stone.src, stone.e))
    live = {mid for group in doc.groups if "hidden" not in group.flags for mid in group.members}
    for mid in live:
        m = doc.members[mid]
        if not m.lang.startswith("zh"):
            continue
        events = source_events(m) | stones[mid]
        events = {(sid, ei) for sid, ei in events if sid in sources and sources[sid].lang.startswith("zh")
                  and ei in event_ids[sid]}
        origins[mid] = events
        for origin in events:
            memberships[origin].add(mid)
    selections = {}
    for sid, src in sources.items():
        _check(runner)
        if not src.lang.startswith("zh") or rules.direction == "off":
            continue
        selection = resolve_reference(src, rules.direction, reference_path,
                                      episode_id=doc.episode, runner=runner)
        selections[sid] = selection
        protected.update(selection.protected_input_paths)
        data["sources"].append(dict(source_id=sid, source_path=src.path, source_sha256=src.sha256,
            target_path=selection.target.path if selection.target else "",
            target_sha256=selection.target.sha256 if selection.target else "",
            baseline_path=selection.baseline.path if selection.baseline else "",
            baseline_sha256=selection.baseline.sha256 if selection.baseline else "",
            issues=list(selection.issues)))
    for group in doc.groups:
        if "hidden" in group.flags:
            continue
        for mid in group.members:
            _check(runner)
            m = doc.members[mid]
            if mid not in origins:
                continue
            current = plain_text(m.text, breaks="\n")
            automatic = plain_text(display_text(m.text, rules)[0], breaks="\n") if rules.converted_for(m) else current
            row = dict(group=group.id, member=mid, start=m.start, end=m.end, lang=m.lang,
                source_path="", source_event=None, imported="", baseline="", current=current,
                reference="", automatic=automatic, generated=automatic, final="", strategy="opencc",
                reason="没有可唯一核对的原始事件，自动转换当前正文。", changed=False, preimport_changed=False)
            events = origins[mid]
            if len(events) == 1:
                sid, ei = next(iter(events))
                src = sources[sid]
                ev = next(event for event in src.events if event.i == ei)
                row.update(source_path=src.path, source_event=ei,
                           imported=plain_text(ev.text, breaks="\n"))
                row["baseline"] = row["imported"]
                row["changed"] = _flat(current) != _flat(row["imported"])
                selection = selections.get(sid)
                if selection:
                    if selection.baseline and ei in selection.baseline_matches:
                        be = next(e for e in selection.baseline.events if e.i == selection.baseline_matches[ei])
                        row["baseline"] = be.text
                        row["preimport_changed"] = _flat(be.text) != _flat(row["imported"])
                    if selection.target and ei in selection.target_matches:
                        te = next(e for e in selection.target.events if e.i == selection.target_matches[ei])
                        row["reference"] = te.text
                # A source split across members (or only partially retained) cannot
                # inherit the entire old sentence. Include tombstones for full rewrites.
                offsets = {r.a + k for r in m.runs if r.src == sid and r.e == ei
                           for k in range(len(r.text))}
                offsets.update(k for t in doc.tombstones if t.member == mid and t.src == sid and t.e == ei
                               for k in range(t.a, t.b))
                expected = {ch.raw_start for ch in VisibleText(ev.text).chars}
                if len(memberships[(sid, ei)]) > 1 or not expected.issubset(offsets):
                    row["reason"] = "原始句子已拆分、合并或仅保留片段，自动转换当前片段。"
                elif not row["reference"]:
                    row["reason"] = "未找到可靠对应的目标字形原配句，自动转换当前正文。"
                elif (not selection or not selection.baseline or ei not in selection.baseline_matches) and (
                        _canonical(_flat(row["baseline"])) != _canonical(_flat(row["reference"]))):
                    row["reason"] = "原配译文与导入正文不同且缺少原版同字形底稿，无法判断导入前改译，自动转换当前正文。"
                else:
                    generated, strategy, reason = merge_reference(_flat(row["baseline"]), _flat(current),
                                                                  _flat(row["reference"]), rules.convert)
                    if strategy == "opencc":
                        generated = automatic  # honour per-token/Raw conversion scope
                    row.update(generated=generated, strategy=strategy, reason=reason)
            elif len(events) > 1:
                row["reason"] = "当前句子来自多个原始事件，自动转换当前正文。"
            if not rules.converted_for(m):
                row.update(generated=current, strategy="unchanged", reason=(
                    "未启用中文目标字形转换，保持当前正文。" if rules.direction == "off" else
                    "当前类别未启用中文转换，保持当前正文。"))
            data["rows"].append(row)
    data["protected_input_paths"] = sorted(protected)
    return data
