"""Extract the user's final per-line readings (振假名 Comment records) from the
finished Utena episodes → eval/data/utena/ruby_ENN.json

Each item: {"jp": visible JP text, "entries": [[surface, reading], ...]}
Readings are normalised to hiragana unless the user deliberately wrote katakana.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from subkit.ass import AssDocument  # noqa: E402
from subkit.ass.text import plain_text  # noqa: E402

from build_utena import OUT, final_path  # noqa: E402

ENTRY = re.compile(r"([^\s（）()+]+)[（(]([^（）()]*)[）)]")


def parse_entries(text: str) -> list[tuple[str, str]]:
    out = []
    for m in ENTRY.finditer(text):
        surface = m.group(1).strip()
        body = m.group(2).strip()
        reading = re.split(r"[,，]", body)[0].strip()
        if surface and reading:
            out.append((surface, reading))
    return out


def build(ep: int) -> list[dict]:
    doc = AssDocument.load(final_path(ep))
    jp_lines = {}
    for e in doc.events:
        if e.is_dialogue and e.style in ("JP", "JPSecondary", "JPTop", "JPOPED"):
            jp_lines.setdefault((e.start, e.end), []).append((e.style, plain_text(e.text, breaks="")))
    items = []
    for e in doc.events:
        if e.is_comment and e.style == "振假名" and "Action" not in e.text:
            cands = jp_lines.get((e.start, e.end))
            if not cands:
                continue
            style, jp = cands[0]
            items.append({"jp": jp, "style": style, "entries": parse_entries(e.text)})
    return items


if __name__ == "__main__":
    for ep in range(1, 11):
        items = build(ep)
        (OUT / f"ruby_E{ep:02d}.json").write_text(json.dumps(items, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"E{ep:02d}: lines={len(items)} entries={sum(len(i['entries']) for i in items)}")
