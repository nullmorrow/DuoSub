"""Read-only import of subtitle sources (FORMAT-v1 §5).

``import_source`` never writes to the original file.  It hashes the bytes,
stores a byte-identical copy in the project (marked read-only), decodes with the
detected encoding, parses every event with its exact text, classifies roles
(dialogue / top / song / sign / comment) and — for Japanese captions — records
structural metadata spans instead of deleting anything.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Optional

from ..ass.document import AssDocument
from ..encoding import decode_bytes
from ..model.source import Source, SourceEvent, Span
from .sdh_en import analyzer_for
from .srt import parse_srt

_EVENT_LINE = re.compile(r"^(Dialogue|Comment|Picture|Sound|Movie|Command)\s*:", re.IGNORECASE)


class SourceChanged(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_format(path: Path, text: str) -> str:
    ext = path.suffix.lower()
    if ext in (".srt", ".ass", ".ssa"):
        return ext[1:]
    head = text[:4000].lower()
    if "[script info]" in head:
        return "ssa" if "[v4 styles]" in head else "ass"
    return "srt"


def _newline_kind(data: bytes) -> str:
    crlf = data.count(b"\r\n")
    lf = data.count(b"\n") - crlf
    cr = data.count(b"\r") - crlf
    kinds = [k for k, n in (("crlf", crlf), ("lf", lf), ("cr", cr)) if n]
    if not kinds:
        return "none"
    return kinds[0] if len(kinds) == 1 else "mixed"


def _store_copy(data: bytes, sha: str, target: Path) -> None:
    """Byte-identical copy, read-only.  An existing copy must have the same hash."""
    if target.exists():
        if sha256_bytes(target.read_bytes()) != sha:
            raise SourceChanged(f"{target} already exists with different content; refusing to overwrite")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, target)
    if sha256_bytes(target.read_bytes()) != sha:
        raise SourceChanged(f"copy of source to {target} does not verify")
    os.chmod(target, stat.S_IREAD)


def _ass_event_lines(text: str) -> list[int]:
    """1-based line numbers of the event lines in [Events], in file order."""
    out, in_events = [], False
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.lstrip("﻿").strip()
        if line.startswith("[") and line.endswith("]"):
            in_events = line.lower() == "[events]"
            continue
        if in_events and _EVENT_LINE.match(line):
            out.append(n)
    return out


def _cn_roles(doc: AssDocument, cn_cfg: Optional[dict[str, Any]]) -> tuple[dict[int, str], list[str], str]:
    from ..sources.cn import classify, detect_cn_script, _is_positioned  # baseline classifier (style roles)

    c = classify(doc, dict(cn_cfg or {}))
    roles: dict[int, str] = {}
    for i, ev in enumerate(doc.events):
        r = c.roles.get(i, "keep")
        if r == "drop":
            roles[i] = "excluded"
        elif r == "song_jp":
            roles[i] = "song_original"
        elif r == "song_cn":
            roles[i] = "song_translation"
        elif not ev.is_dialogue:
            roles[i] = "comment" if ev.is_comment else "sign"
        elif r == "dialogue":
            roles[i] = "dialogue"
        elif r == "top":
            roles[i] = "top"
        else:
            roles[i] = "sign"          # positioned / drawing / staff / unknown styles: kept as they are
    return roles, list(c.notes), c.lang


def import_source(path: Path | str, *, src_id: str, lang: str, lane: str, episode_dir: Path,
                  cn_cfg: Optional[dict[str, Any]] = None,
                  ass_styles: Optional[list[str]] = None,
                  song_intervals: Optional[list[tuple[int, int]]] = None) -> Source:
    path = Path(path)
    data = path.read_bytes()                       # the only access to the original: read
    sha = sha256_bytes(data)
    dec = decode_bytes(data)
    fmt = detect_format(path, dec.text)
    copy_rel = f"sources/{src_id}{path.suffix.lower() or '.' + fmt}"
    _store_copy(data, sha, Path(episode_dir) / copy_rel)
    src = Source(id=src_id, lang=lang, lane=lane, format=fmt, path=str(path).replace("\\", "/"), copy=copy_rel,
                 sha256=sha, bytes=len(data), encoding=dec.encoding + ("-sig" if dec.bom else ""),
                 newline=_newline_kind(data), imported_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if fmt == "srt":
        for blk in parse_srt(dec.text):
            extra = {"timing_tail": blk.timing_tail} if blk.timing_tail.strip() else {}
            src.events.append(SourceEvent(blk.order, "Dialogue", blk.start, blk.end, blk.text,
                                          line=blk.line, cue=blk.number or None, extra=extra))
            src.roles[blk.order] = "dialogue"
    else:
        doc = AssDocument.parse(dec.text)
        lines = _ass_event_lines(dec.text)
        src.playres = list(doc.playres) if doc.playres else None
        for i, ev in enumerate(doc.events):
            src.events.append(SourceEvent(
                i, ev.kind, ev.start * 10, ev.end * 10, ev.text, style=ev.style, name=ev.name, layer=ev.layer,
                margins=[ev.margin_l, ev.margin_r, ev.margin_v], effect=ev.effect,
                line=lines[i] if len(lines) == len(doc.events) else None, extra=dict(ev.extra)))
        if lane == "translation":
            roles, notes, _detected = _cn_roles(doc, cn_cfg)
            src.roles = roles
            src.doubts.extend({"e": None, "text": n} for n in notes)
        else:
            src.roles = {i: ("excluded" if ass_styles is not None and
                              (not ev.is_dialogue or ev.style not in ass_styles) else
                              "comment" if ev.is_comment else "dialogue")
                         for i, ev in enumerate(doc.events)}
        for ev in src.events:
            if src.roles.get(ev.i) == "excluded":
                src.spans.append(Span(ev.i, 0, len(ev.text), "other",
                                      data={"reason": (cn_cfg or {}).get("excluded_reasons", {}).get(ev.i, "style_filter"),
                                            "style": ev.style}))
    analyzer, analyze = analyzer_for(lang)
    src.analyzer = dict(analyzer)
    for ev in src.events:
        if src.roles.get(ev.i) not in ("dialogue", "top"):
            continue
        an = analyze(ev.i, ev.text, fmt)
        src.spans.extend(an.spans)
        src.doubts.extend({"e": ev.i, "text": d} for d in an.doubts)
        visible_lines = [line for line in an.lines if line.kind in ("speech", "lyric")]
        if lane == "original" and song_intervals and visible_lines and all(line.kind == "lyric" for line in visible_lines):
            if any(min(ev.end, end) > max(ev.start, start) for start, end in song_intervals):
                src.roles[ev.i] = "song_original"
    return src


def verify_copy(src: Source, episode_dir: Path) -> bool:
    """True when the project's copy still matches the recorded hash."""
    return copy_state(Path(episode_dir) / src.copy, src.sha256) == "ok"


def copy_state(path: Path, sha: str) -> str:
    """``ok`` / ``missing`` / ``changed`` for a project copy of an imported subtitle file."""
    path = Path(path)
    if not path.is_file():
        return "missing"
    return "ok" if sha256_bytes(path.read_bytes()) == sha else "changed"


def restore_copy(target: Path, sha: str, candidate: Path) -> None:
    """Put back a project copy from ``candidate``, which must be byte-identical to what was imported.
    A copy whose bytes changed is kept beside it as ``<name>.changed-<time>``, never deleted."""
    data = Path(candidate).read_bytes()
    if sha256_bytes(data) != sha:
        raise SourceChanged("这个文件的内容和导入时不同（SHA-256 不一致），不能当作原字幕副本")
    target = Path(target)
    if copy_state(target, sha) == "ok":
        return
    if target.exists():
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
        target.replace(target.with_name(f"{target.name}.changed-{time.strftime('%Y%m%d-%H%M%S')}"))
    _store_copy(data, sha, target)
