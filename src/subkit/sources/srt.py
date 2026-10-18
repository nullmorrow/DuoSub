"""SRT reading/writing (tolerant: missing indices, extra blank lines, ',' or '.' millis)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..ass.timefmt import cs_to_srt, srt_to_cs
from ..encoding import read_text, write_text

_ARROW = re.compile(r"(\d+:\d{1,2}:\d{1,2}[,.]\d{1,3})\s*-->\s*(\d+:\d{1,2}:\d{1,2}[,.]\d{1,3})")
_HTML = re.compile(r"</?(?:i|b|u|font)[^>]*>", re.IGNORECASE)


@dataclass
class SrtCue:
    index: int
    start: int  # cs
    end: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def parse_srt(text: str) -> list[SrtCue]:
    cues: list[SrtCue] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    n = len(lines)
    while i < n:
        m = _ARROW.search(lines[i])
        if not m:
            i += 1
            continue
        start, end = srt_to_cs(m.group(1)), srt_to_cs(m.group(2))
        body: list[str] = []
        i += 1
        while i < n:
            line = lines[i]
            if not line.strip():
                # a blank line ends the cue unless the next non-blank line is text
                # that is not an index followed by a timestamp
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                if j >= n:
                    i = j
                    break
                nxt_is_cue = bool(_ARROW.search(lines[j])) or (
                    lines[j].strip().isdigit() and j + 1 < n and _ARROW.search(lines[j + 1])
                )
                if nxt_is_cue:
                    i = j
                    break
                i = j
                continue
            if _ARROW.search(line):
                break
            if line.strip().isdigit() and i + 1 < n and _ARROW.search(lines[i + 1]):
                i += 1
                break
            body.append(_HTML.sub("", line.lstrip("﻿")))
            i += 1
        cues.append(SrtCue(len(cues) + 1, start, end, body))
    return cues


def load_srt(path: Path | str) -> list[SrtCue]:
    return parse_srt(read_text(path).text)


def save_srt(path: Path | str, cues: list[SrtCue]) -> None:
    out = []
    for k, c in enumerate(cues, 1):
        out.append(str(k))
        out.append(f"{cs_to_srt(c.start)} --> {cs_to_srt(c.end)}")
        out.extend(c.lines or [""])
        out.append("")
    write_text(path, "\n".join(out), newline="\r\n", bom=True)
