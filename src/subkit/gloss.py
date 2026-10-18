"""Glosses (旁注) — the old CalUP and English annotations, unified.

Write the mark anywhere in a CN or JP dialogue line (usually right after the word)::

    光差す校庭{@注:校庭=ガーデン} 手をとり合い        → ガーデン above 校庭 (JP line: 校庭 gets no furigana)
    我们去食戟{@注:食戟=料理对决}吧                  → 料理对决 above 食戟 (CN line)
    {@注:=第一话}                                    → empty target: centred above the whole line
    {@注:校庭=ガーデン,80}                           → \\fscx80 for this gloss

Aliases: ``{@注:…}`` ``{@g:…}`` ``{@gloss:…}``.  If the target occurs several
times, the occurrence closest to the mark is used.  The gloss is rendered in
``<base style>.Gloss`` (derived from the base style when missing).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from .ass.text import VisibleText, private_blocks
from .textutil import is_kana

_NAME = re.compile(r"^(?:注|g|gloss)(?=\s*[:：]|$)", re.IGNORECASE)
_GLOSS = re.compile(r"^(?:注|g|gloss)\s*[:：]\s*(.*?)\s*[=＝]\s*(.*?)\s*$", re.IGNORECASE)


@dataclass
class Gloss:
    start: int        # visible range of the target (whole line when target empty)
    end: int
    target: str
    text: str
    scale: Optional[float] = None
    position: int = 8
    distance: float = 0.0
    font: Optional[str] = None
    size: Optional[float] = None
    color: Optional[str] = None


def contains_kana(text: str) -> bool:
    """Include half-width and supplementary kana before treating a note as Chinese."""
    return any(is_kana(ch) or "ｦ" <= ch <= "ﾟ"
               or 0x1AFF0 <= ord(ch) <= 0x1AFFF or 0x1B000 <= ord(ch) <= 0x1B16F
               for ch in text)


def _number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(value)
    return number


def parse_glosses(raw_text: str, warnings: Optional[list[str]] = None) -> list[Gloss]:
    vis = VisibleText(raw_text).text
    out: list[Gloss] = []
    for pos, body, tok in private_blocks(raw_text):
        if not _NAME.match(body):
            continue
        m = _GLOSS.match(body)
        if not m:
            if warnings is not None:
                warnings.append(f"旁注写法有误：{tok.raw}（不认识的选项「{body}」）")
            continue
        target, value = m.groups()
        fields = re.split(r"[;；]", value)
        label = fields[0].strip()
        scale = None
        mm = re.match(r"^(.*?)[,，]\s*(\d+(?:\.\d+)?)$", label)
        if mm:
            label, scale = mm.group(1), float(mm.group(2))
        gloss = Gloss(0, len(vis), target, label, scale)
        reason, positional = "", 0
        for option in fields[1:]:
            option = option.strip()
            pair = re.split(r"[=＝]", option, maxsplit=1)
            if len(pair) == 2:
                name, val = pair[0].strip().lower(), pair[1].strip()
                if name == "font":
                    gloss.font = val
                elif name == "size":
                    try:
                        gloss.size = _number(val)
                        if gloss.size <= 0:
                            raise ValueError(val)
                    except ValueError:
                        reason = "字号要是正数"
                elif name == "color":
                    if re.fullmatch(r"#[0-9a-fA-F]{6}", val):
                        gloss.color = val
                    else:
                        reason = "颜色要写成 #RRGGBB"
                else:
                    reason = f"不认识的选项「{pair[0].strip()}」"
            elif positional == 0:
                if option:
                    if re.fullmatch(r"[1-9]", option):
                        gloss.position = int(option)
                    else:
                        reason = "位置要写 1–9"
                positional += 1
            elif positional == 1:
                try:
                    gloss.distance = _number(option) if option else 0.0
                except ValueError:
                    reason = "距离要是数字"
                positional += 1
            else:
                reason = f"不认识的选项「{option}」"
            if reason:
                break
        if not reason and target:
            candidates = [p for p in range(len(vis)) if vis.startswith(target, p)]
            if not candidates:
                reason = f"找不到被注的字「{target}」"
            else:
                best = min(candidates, key=lambda p: (min(abs(pos-p), abs(pos-p-len(target))),
                                                      p+len(target) > pos, p))
                gloss.start, gloss.end = best, best + len(target)
        if reason and warnings is not None:
            warnings.append(f"旁注写法有误：{tok.raw}（{reason}）")
        elif not reason and label:
            out.append(gloss)
    return out


def gloss_anchor(box, ink, position: int, distance: float = 0.0) -> tuple[float, float]:
    """Place the opposite edge of gloss ink at the target's nine-grid point.

    ``ink`` is measured relative to the gloss event's opposite-alignment anchor.
    Coordinates stay unrotated, as in furigana placement.
    """
    column, row = (position - 1) % 3, (position - 1) // 3
    x = (box.x0 - distance, (box.x0 + box.x1) / 2, box.x1 + distance)[column]
    y = (box.y1 + distance, (box.y0 + box.y1) / 2, box.y0 - distance)[row]
    ix = (ink.x1, (ink.x0 + ink.x1) / 2, ink.x0)[column]
    iy = (ink.y0, (ink.y0 + ink.y1) / 2, ink.y1)[row]
    return x - ix, y - iy
