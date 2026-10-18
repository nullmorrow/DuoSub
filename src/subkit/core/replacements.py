"""Project-wide, literal display replacements. Never writes subtitle text."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import re

from ..ass.text import VisibleText, tokenize


@dataclass(frozen=True)
class ReplacementRule:
    source: str
    target: str = ""
    lane: str = "both"
    enabled: bool = True


def normalize_rules(rules, *, strict=True) -> list[dict]:
    if not isinstance(rules, (list, tuple)):
        if strict:
            raise ValueError("批量替换必须是一个替换列表。")
        return []
    result = []
    for i, raw in enumerate(rules, 1):
        try:
            item = asdict(raw) if isinstance(raw, ReplacementRule) else raw
            if not isinstance(item, dict):
                raise ValueError("不是有效的替换规则")
            source, target = item.get("source"), item.get("target", "")
            lane, enabled = item.get("lane", "both"), item.get("enabled", True)
            if not isinstance(source, str) or not source:
                raise ValueError("词 A 不能为空")
            if not isinstance(target, str):
                raise ValueError("词 B 必须是文字，留空表示删除")
            if lane not in ("original", "translation", "both"):
                raise ValueError("作用轴必须是原声、翻译或两轴")
            if not isinstance(enabled, bool):
                raise ValueError("启用状态无效")
            if any(ch in source + target for ch in "{}\\\r\n\x00"):
                raise ValueError("只能替换正文，不能含反斜杠、半角大括号、换行或 NUL（不能写 ASS 标签）")
            result.append(dict(enabled=enabled, source=source, target=target, lane=lane))
        except ValueError as exc:
            if strict:
                raise ValueError(f"第 {i} 条：{exc}") from exc
    return result


def frozen_rules(rules) -> tuple[ReplacementRule, ...]:
    return tuple(ReplacementRule(**r) for r in normalize_rules(rules, strict=False))


def load_rules(project) -> list[dict]:
    return normalize_rules(project.data.get("display_replacements", []), strict=False)


def save_rules(project, rules) -> bool:
    normalized = normalize_rules(rules)
    if normalized == load_rules(project):
        return False
    previous = copy.deepcopy(project.data)
    project.data["display_replacements"] = normalized
    try:
        project.save()
    except Exception:
        project.data = previous
        raise
    return True


@lru_cache(maxsize=32)
def _matcher(rules: tuple[ReplacementRule, ...], lane: str):
    selected = {}
    for rule in rules:
        if rule.enabled and rule.lane in (lane, "both"):
            selected.setdefault(rule.source, rule.target)
    pattern = re.compile("|".join(re.escape(text) for text in selected)) if selected else None
    return pattern, selected


def apply_replacements(text: str, rules, *, lane: str) -> str:
    """Plain-text preview: match original positions once, in list priority order."""
    pattern, choices = _matcher(frozen_rules(rules), lane)
    return pattern.sub(lambda match: choices[match.group()], text) if pattern is not None else text


@dataclass
class ReplacementResult:
    text: str
    bounds: list[int]
    matches: int = 0
    deleted: set[int] = field(default_factory=set)   # explicit empty-B ranges, in original visible positions
    ranges: list[tuple[int, int]] = field(default_factory=list)


def replace_ass_text(text: str, rules, *, lane: str) -> ReplacementResult:
    """Replace visible words across inline styles, never across Raw/drawing/hard breaks.

    Original ASS tokens stay in order. Expansions share original character IDs;
    contractions retain a monotonic boundary map for ruby, formatting and hits.
    Empty-B deletion records its exact original extent for annotation protection.
    """
    vt = VisibleText(text)
    result = ReplacementResult(text, list(range(len(vt) + 1)))
    pattern, choices = _matcher(frozen_rules(rules), lane)
    if pattern is None or not vt.chars:
        return result
    by_raw = {ch.raw_start: i for i, ch in enumerate(vt.chars)}
    segments, segment = [], []
    for token in tokenize(text):
        if token.protected or token.kind in ("break", "drawing"):
            if segment:
                segments.append(segment)
                segment = []
        elif token.kind == "text":
            segment.extend(by_raw[i] for i in range(token.start, token.end))
        elif token.kind in ("softbreak", "hardspace"):
            segment.append(by_raw[token.start])
    if segment:
        segments.append(segment)
    chunks = [ch.ch for ch in vt.chars]
    counts = [1] * len(vt)
    for segment in segments:
        visible = "".join(vt.chars[i].ch for i in segment)
        for match in pattern.finditer(visible):
            target = choices[match.group()]
            if target == match.group():
                continue
            indices = segment[match.start():match.end()]
            result.matches += 1
            result.ranges.append((indices[0], indices[-1] + 1))
            for k, i in enumerate(indices):
                a, b = k * len(target) // len(indices), (k + 1) * len(target) // len(indices)
                chunks[i], counts[i] = target[a:b], b - a
            if not target:
                result.deleted.update(indices)
    if not result.matches:
        return result
    out, at = [], 0
    for ch, chunk in zip(vt.chars, chunks):
        out.append(text[at:ch.raw_start])
        out.append(text[ch.raw_start:ch.raw_end] if chunk == ch.ch else chunk)
        at = ch.raw_end
    out.append(text[at:])
    result.text = "".join(out)
    result.bounds = [0]
    for n in counts:
        result.bounds.append(result.bounds[-1] + n)
    return result
