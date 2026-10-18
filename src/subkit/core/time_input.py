"""Exact milliseconds from the time dialog; independent of ASS centisecond rounding."""

from __future__ import annotations

import re

TIME_FORMAT = "时间格式：0:01:23.45"
_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?")


def parse_time(value: str) -> int:
    """Accept h:mm:ss.cc[c], m:ss.cc[c], or a nonnegative integer in ms."""
    value = value.strip()
    if re.fullmatch(r"[0-9]+", value):
        return int(value)
    match = _TIME.fullmatch(value)
    if not match:
        raise ValueError(TIME_FORMAT)
    hour, minute, second, fraction = match.groups()
    if int(second) >= 60 or (hour is not None and int(minute) >= 60):
        raise ValueError(TIME_FORMAT)
    return ((int(hour or 0) * 60 + int(minute)) * 60 + int(second)) * 1000 + int((fraction or "0").ljust(3, "0"))


def format_time(ms: int) -> str:
    """Use the list's format, retaining a third fractional digit when necessary."""
    hour, rest = divmod(ms, 3_600_000)
    minute, rest = divmod(rest, 60_000)
    second, fraction = divmod(rest, 1000)
    tail = f"{fraction:03d}" if fraction % 10 else f"{fraction // 10:02d}"
    return f"{hour}:{minute:02d}:{second:02d}.{tail}"
