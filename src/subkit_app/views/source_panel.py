"""Original captions around the current group (REQUIREMENTS 二.4: the original is always visible)."""

from __future__ import annotations

import html
from collections import defaultdict
from typing import Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtWidgets import QTextBrowser

from subkit.model.document import Document, Group
from subkit.model.source import Source


def _fmt(ms: int) -> str:
    return f"{ms // 60000}:{ms % 60000 / 1000:06.3f}"


class SourcePanel(QTextBrowser):
    help_anchor = "review"
    groupRequested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setOpenLinks(False)
        self.anchorClicked.connect(self._clicked)
        self.doc: Optional[Document] = None
        self.sources: dict[str, Source] = {}
        self._owner: dict[tuple[str, int, int], str] = {}

    def set_data(self, doc: Document, sources: dict[str, Source]) -> None:
        self.doc, self.sources = doc, sources

    def _clicked(self, url: QUrl) -> None:
        gid = url.toString().removeprefix("g:")
        if gid:
            self.groupRequested.emit(gid)

    def show_group(self, g: Optional[Group], *, context: int = 3) -> None:
        if self.doc is None or g is None:
            self.setHtml("")
            return
        doc = self.doc
        # character ownership for the events we show
        used: dict[tuple[str, int], list[tuple[int, int, str, str]]] = defaultdict(list)
        events: dict[str, set[int]] = defaultdict(set)
        for mid in g.members:
            for r in doc.members[mid].runs:
                if r.src is not None:
                    events[r.src].add(r.e)
        for cid in g.comments:
            for r in doc.comments[cid].runs:
                if r.src is not None:
                    events[r.src].add(r.e)
        for grp in doc.groups:
            for mid in grp.members:
                for r in doc.members[mid].runs:
                    if r.src is not None and r.kind != "sep":
                        used[(r.src, r.e)].append((r.a, r.b, grp.id, "cur" if grp.id == g.id else "other"))
            for cid in grp.comments:
                for r in doc.comments[cid].runs:
                    if r.src is not None:
                        used[(r.src, r.e)].append((r.a, r.b, grp.id, "hidden"))
        for t in doc.tombstones:
            used[(t.src, t.e)].append((t.a, t.b, "", "deleted"))
        parts = ["<div style='font-family:\"Yu Gothic UI\",\"Microsoft YaHei UI\"; font-size:12pt'>"]
        for sid in sorted(events, key=lambda s: (self.sources[s].lane != "original", s)):
            src = self.sources.get(sid)
            if src is None:
                continue
            idx = sorted(events[sid])
            lo, hi = max(0, idx[0] - context), min(len(src.events), idx[-1] + context + 1)
            lane = "原声" if src.lane == "original" else "翻译"
            parts.append(f"<p style='color:#57606a;margin:4px 0'>【{lane}原文 {html.escape(src.id)}：{html.escape(src.path.split('/')[-1])}】</p>")
            for e in range(lo, hi):
                ev = src.events[e]
                spans = {k: s for s in src.spans_of(e) for k in range(s.a, s.b)}
                own = used.get((sid, e), [])
                link_gid = next((x[2] for x in own if x[2]), "")
                cue = ev.cue or str(e + 1)
                head = f"#{cue}  {_fmt(ev.start)}–{_fmt(ev.end)}"
                bg = "#eef6ff" if e in events[sid] else "transparent"
                html_chars = []
                for k, ch in enumerate(ev.text):
                    c = html.escape(ch) if ch != "\n" else "<br>"
                    kinds = [x[3] for x in own if x[0] <= k < x[1]]
                    if k in spans:
                        c = f"<span style='color:#8c959f'>{c}</span>"
                    elif "deleted" in kinds:
                        c = f"<span style='color:#cf222e;text-decoration:line-through'>{c}</span>"
                    elif "hidden" in kinds:
                        c = f"<span style='color:#b35900;text-decoration:underline'>{c}</span>"
                    elif "cur" in kinds:
                        c = f"<span style='color:#0550ae;font-weight:600'>{c}</span>"
                    html_chars.append(c)
                body = "".join(html_chars)
                parts.append(f"<div style='background:{bg};padding:2px 4px;margin:1px 0'>"
                             f"<a href='g:{link_gid}' style='color:#6e7781;text-decoration:none'>{head}</a><br>{body}</div>")
        parts.append("<p style='color:#8c959f;font-size:9pt'>蓝色：本组使用的文字；灰色：说话人标签等元数据；橙色下划线：未渲染（可恢复）；"
                     "红色删除线：已删除。点时间跳到那一组。</p></div>")
        self.setHtml("".join(parts))
