"""Project-wide review list, read directly from saved episode documents."""

from __future__ import annotations

import csv
import html
from pathlib import Path

from ..model.serialize import load_document
from .readings import visible_map

HEADERS = ("组号", "开始–结束", "翻译", "原声", "状态", "备注")


def _time(ms: int) -> str:
    sign = "-" if ms < 0 else ""
    hours, remainder = divmod(abs(ms), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{sign}{hours}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def collect_review(project) -> list[tuple[str, str, list[tuple[str, ...]]]]:
    """Read all saved documents without loading sessions or source/media files."""
    episodes = []
    for entry in project.data.get("episodes", []):
        path = project.root / entry["path"] / "document.json"
        if not path.exists():
            continue
        doc = load_document(path)
        rows = []
        for number, group in enumerate(doc.groups, 1):
            hidden, marked = "hidden" in group.flags, "marked" in group.flags
            if not (hidden or marked):
                continue
            start, end = doc.group_time(group)
            texts = [" ／ ".join(visible_map(member)[0].text for member in doc.members_of(group)
                                 if member.lane == lane) for lane in ("translation", "original")]
            state = "标记+隐藏" if hidden and marked else "隐藏" if hidden else "标记"
            rows.append((str(number), f"{_time(start)}–{_time(end)}", *texts, state,
                         group.extra.get("mark_note", "")))
        if rows:
            episodes.append((entry["id"], entry.get("title", ""), rows))
    return episodes


def write_review(episodes, path: Path) -> int:
    """Write a self-contained HTML and a sibling BOM CSV; empty input writes nothing."""
    if not episodes:
        return 0
    escape = html.escape
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>复查清单</title><style>:root{color-scheme:light dark}'
             'body{font:16px sans-serif;background:#fff;color:#202124;margin:2rem}'
             'table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;'
             'padding:.5rem;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}'
             '@media(prefers-color-scheme:dark){body{background:#181a1b;color:#eee}'
             'th,td{border-color:#666}}</style><body><h1>复查清单</h1>']
    count = 0
    for eid, title, rows in episodes:
        parts.append(f'<section><h2>{escape(eid)} {escape(title)}</h2><table><thead><tr>')
        parts.extend(f'<th>{escape(label)}</th>' for label in HEADERS)
        parts.append('</tr></thead><tbody>')
        for row in rows:
            parts.append('<tr>' + ''.join(f'<td>{escape(cell)}</td>' for cell in row) + '</tr>')
        parts.append('</tbody></table></section>')
        count += len(rows)
    parts.append('</body></html>')
    path.write_text('\n'.join(parts), encoding="utf-8")
    with path.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(("集号", "标题", *HEADERS))
        for eid, title, rows in episodes:
            writer.writerows((eid, title, *row) for row in rows)
    return count
