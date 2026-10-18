"""Lossless SRT reading: exact text, millisecond times, block order and line numbers.

Unlike the baseline reader this keeps millisecond precision (no rounding to
centiseconds), keeps blank lines that occur *inside* a block, and never edits
the text (HTML-like tags such as ``<i>`` stay as written).  Only line
terminators are unified to ``\\n``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TIME = r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
_ARROW = re.compile(_TIME + r"\s*-->\s*" + _TIME)


@dataclass
class SrtBlock:
    order: int          # 0-based position in the file
    number: str         # the block number as written ("" when missing)
    start: int          # ms
    end: int            # ms
    text: str           # lines joined with "\n"
    line: int           # 1-based line number of the timing line
    timing_tail: str    # anything after the end time on the timing line (position hints), verbatim


def _ms(h: str, m: str, s: str, frac: str) -> int:
    # "5" → 500 ms, "05" → 50 ms, "050" → 50 ms (digits are a decimal fraction)
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(round(float("0." + frac) * 1000))


def parse_srt(text: str) -> list[SrtBlock]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[0].startswith("﻿"):
        lines[0] = lines[0][1:]
    timing = [i for i, ln in enumerate(lines) if _ARROW.search(ln)]
    blocks: list[SrtBlock] = []
    for k, ti in enumerate(timing):
        m = _ARROW.search(lines[ti])
        g = m.groups()
        start, end = _ms(*g[:4]), _ms(*g[4:])
        number = ""
        if ti > 0 and lines[ti - 1].strip().isdigit():
            number = lines[ti - 1].strip()
        stop = timing[k + 1] if k + 1 < len(timing) else len(lines)
        body = lines[ti + 1: stop]
        # the next block's number line belongs to the next block
        if k + 1 < len(timing) and body and body[-1].strip().isdigit():
            body = body[:-1]
        while body and not body[-1].strip():
            body.pop()
        blocks.append(SrtBlock(len(blocks), number, start, end, "\n".join(body), ti + 1,
                               lines[ti][m.end():]))
    return blocks
