"""Pre-release check of a work file: what still needs attention."""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable

from .ass.document import AssDocument
from .ass.text import plain_text
from .project import Episode
from .workfile import PERSIST_NOTE, STYLE_NOTE, STYLE_RECORD, TRANSIENT_NOTE, collect_rows, role_of_style

Log = Callable[[str], None]


def check(ep: Episode, log: Log = print) -> int:
    doc = AssDocument.load(ep.work_ass)
    rows, _ = collect_rows(doc)
    problems = 0
    unreviewed = []
    persist = Counter()
    info = Counter()
    transient = []
    empty_jp = []
    orphan_jp = []
    for rid, r in sorted(rows.items()):
        if r.record is not None and "?" in r.record.text:
            unreviewed.append((r.jp.start if r.jp else 0, re.findall(r"\?(\S+?)（", r.record.text)))
        for n in r.notes:
            if n.text.startswith(PERSIST_NOTE + "已省略") or n.text.startswith(PERSIST_NOTE + "🔊按配音"):
                info[n.text[1:].split("：")[0]] += 1   # informational: nothing to do
            elif n.text.startswith(PERSIST_NOTE):
                persist[re.split(r"[：（，,]", n.text[1:])[0][:14]] += 1
            elif n.text.startswith(TRANSIENT_NOTE):
                transient.append(n.text[1:])
        if r.cn is not None and r.jp is not None and plain_text(r.cn.text).strip() and not plain_text(r.jp.text).strip():
            empty_jp.append(r.cn.start)
        if r.jp is not None and plain_text(r.jp.text).strip() and (r.cn is None or not plain_text(r.cn.text).strip()):
            orphan_jp.append(r.jp.start)
    # overlapping CN lines in the same style (both visible at once, same place)
    cn = sorted((e for e in doc.events if e.is_dialogue and role_of_style(e.style) and role_of_style(e.style)[0] == "cn"),
                key=lambda e: e.start)
    overlaps = 0
    for a, b in zip(cn, cn[1:]):
        if a.style == b.style and b.start < a.end - 5:
            overlaps += 1

    def ts(cs: int) -> str:
        return f"{cs // 6000}:{cs % 6000 / 100:05.2f}"

    log(f"[check] {ep.id}: {len(rows)} 行")
    if unreviewed:
        problems += len(unreviewed)
        log(f"  · 读音待确认（?）{len(unreviewed)} 行：" + "；".join(f"{ts(t)} {'、'.join(w)}" for t, w in unreviewed[:15]))
    if persist:
        log("  · 未处理的 ※ 备注：" + "，".join(f"{k} ×{v}" for k, v in persist.most_common()) + "（处理完请删除备注行）")
    if info:
        log("  · 仅供参考的 ※ 备注（不用处理）：" + "，".join(f"{k} ×{v}" for k, v in info.most_common()))
    if transient:
        problems += len(transient)
        grouped = Counter(re.sub(r"[：:].*$", "", t) if t.startswith(("振假名放不下", "词库优先", "无法测量")) else t for t in transient)
        log(f"  · ⚙ 提示 {len(transient)} 条：" + "；".join(f"{k} ×{v}" if v > 1 else k for k, v in grouped.most_common(8)))
    if empty_jp:
        log(f"  · 有中文没日文的行 {len(empty_jp)}：" + "、".join(ts(t) for t in empty_jp[:20]))
    if orphan_jp:
        log(f"  · 有日文没中文的行 {len(orphan_jp)}：" + "、".join(ts(t) for t in orphan_jp[:20]))
    if overlaps:
        log(f"  · 同一样式的中文行时间重叠 {overlaps} 处（会上下堆叠显示）")
    if not (unreviewed or persist or transient or empty_jp or orphan_jp or overlaps):
        log("  · 没有发现问题")
    return problems
