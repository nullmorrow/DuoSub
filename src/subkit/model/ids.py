"""Opaque, time-sortable IDs (FORMAT-v1 §3): prefix + 12 hex ms timestamp + 8 hex random."""

from __future__ import annotations

import os
import threading
import time

_lock = threading.Lock()
_last_ms = 0
_seq = 0

PREFIXES = ("prj_", "g_", "m_", "c_", "a_", "k_", "i_", "op_", "chr_", "rule_", "tpl_")


def new_id(prefix: str) -> str:
    """A new unique ID.  Within one millisecond a counter keeps the order monotonic."""
    global _last_ms, _seq
    with _lock:
        now = int(time.time() * 1000)
        if now <= _last_ms:
            _seq += 1
            now = _last_ms
        else:
            _seq = 0
            _last_ms = now
        rnd = (int.from_bytes(os.urandom(3), "big") << 8) | (_seq & 0xFF)
    return f"{prefix}{now:012x}{rnd:08x}"


def kind_of(id_: str) -> str:
    """The prefix of an ID ('g_', 'm_', …) or '' if unknown."""
    for p in PREFIXES:
        if id_.startswith(p):
            return p
    return ""
