"""ASS / SRT time conversion.  Internally every time is an integer in centiseconds."""

from __future__ import annotations

import re

_ASS_TIME = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2})(?:[.,](\d+))?\s*$")


def ass_to_cs(value: str) -> int:
    m = _ASS_TIME.match(value or "")
    if not m:
        raise ValueError(f"bad ASS time: {value!r}")
    h, mi, s, frac = m.groups()
    frac = frac or "0"
    # ASS uses centiseconds; tolerate milliseconds (rounded) and single digits.
    if len(frac) == 1:
        cs = int(frac) * 10
    elif len(frac) == 2:
        cs = int(frac)
    else:
        cs = int(round(int(frac[:3].ljust(3, "0")) / 10.0))
    return ((int(h) * 60 + int(mi)) * 60 + int(s)) * 100 + cs


def cs_to_ass(cs: int) -> str:
    cs = max(0, int(cs))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


_SRT_TIME = re.compile(r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})")


def srt_to_cs(value: str) -> int:
    m = _SRT_TIME.search(value or "")
    if not m:
        raise ValueError(f"bad SRT time: {value!r}")
    h, mi, s, ms = m.groups()
    ms = int(ms.ljust(3, "0"))
    return ((int(h) * 60 + int(mi)) * 60 + int(s)) * 100 + (ms + 5) // 10


def cs_to_srt(cs: int) -> str:
    ms = max(0, int(cs)) * 10
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cs_to_seconds(cs: int) -> float:
    return cs / 100.0
