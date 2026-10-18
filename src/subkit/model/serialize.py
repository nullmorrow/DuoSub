"""JSON (de)serialisation of the working document (FORMAT-v1 §6).

Unknown fields are kept in each object's ``extra`` and written back unchanged,
so a document written by another implementation survives a load/save cycle
(FORMAT-v1 §13).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import fields
from itertools import chain
from pathlib import Path
from typing import Any

from .document import (FORMAT_DOCUMENT, Annotation, Comment, Document, Group, Member, Run, Tombstone)

FORMAT_VERSION = 2  # v2 adds rendering-only context retained by copies and text edits.


def _split(cls, data: dict[str, Any], skip: tuple[str, ...] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    names = {f.name for f in fields(cls)} - {"extra"} - set(skip)
    known = {k: v for k, v in data.items() if k in names}
    extra = {k: v for k, v in data.items() if k not in names and k not in skip}
    return known, extra


def run_to_json(r: Run) -> dict[str, Any]:
    d: dict[str, Any] = {}
    if r.kind != "text":
        d["kind"] = r.kind
    if r.src is not None:
        d.update(src=r.src, e=r.e, a=r.a, b=r.b)
    elif r.ins is not None:
        d.update(ins=r.ins, a=r.a, b=r.b)
    if r.render_src is not None:
        d.update(render_src=r.render_src, render_e=r.render_e)
    d["text"] = r.text
    return d


def run_from_json(d: dict[str, Any]) -> Run:
    return Run(text=d.get("text", ""), kind=d.get("kind", "text"), src=d.get("src"), e=d.get("e"),
               a=int(d.get("a", 0)), b=int(d.get("b", 0)), ins=d.get("ins"),
               render_src=d.get("render_src"), render_e=d.get("render_e"))


def _obj(o: Any, drop_default: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(o):
        if f.name.startswith("_") or f.name == "extra":
            continue
        v = getattr(o, f.name)
        if f.name == "runs":
            v = [run_to_json(r) for r in v]
        if drop_default and f.name in drop_default and v == drop_default[f.name]:
            continue
        out[f.name] = v
    out.update(getattr(o, "extra", {}) or {})
    return out


def document_to_json(doc: Document) -> dict[str, Any]:
    # Keep existing documents at v1. Only a retained source context needs v2, which
    # makes older writers refuse rather than silently dropping its ASS context.
    version = 2 if any(r.render_src is not None for owner in chain(doc.members.values(), doc.comments.values())
                       for r in owner.runs) else 1
    return {
        "format": FORMAT_DOCUMENT, "format_version": version,
        "episode": doc.episode, "revision": doc.revision,
        "groups": [_obj(g) for g in doc.groups],
        "members": {k: _obj(m, {"rev": 0}) for k, m in doc.members.items()},
        "comments": {k: _obj(c) for k, c in doc.comments.items()},
        "annotations": {k: _obj(a) for k, a in doc.annotations.items()},
        "karaoke": doc.karaoke,
        "tombstones": [_obj(t) for t in doc.tombstones],
        "rules": doc.rules,
        **doc.extra,
    }


def document_from_json(data: dict[str, Any]) -> Document:
    if data.get("format") != FORMAT_DOCUMENT:
        raise ValueError(f"not a SubKit document: format={data.get('format')!r}")
    ver = int(data.get("format_version", 0))
    if ver > FORMAT_VERSION:
        raise ValueError(f"document format_version {ver} is newer than this program ({FORMAT_VERSION}); open read-only")
    top_known = {"format", "format_version", "episode", "revision", "groups", "members", "comments", "annotations",
                 "karaoke", "tombstones", "rules"}
    doc = Document(episode=data.get("episode", ""), revision=int(data.get("revision", 0)))
    doc.extra = {k: v for k, v in data.items() if k not in top_known}
    for g in data.get("groups", []):
        known, extra = _split(Group, g)
        doc.groups.append(Group(**known, extra=extra))
    for k, m in data.get("members", {}).items():
        known, extra = _split(Member, m, skip=("runs",))
        mem = Member(**known, extra=extra)
        mem.runs = [run_from_json(r) for r in m.get("runs", [])]
        doc.members[k] = mem
    for k, c in data.get("comments", {}).items():
        known, extra = _split(Comment, c, skip=("runs",))
        com = Comment(**known, extra=extra)
        com.runs = [run_from_json(r) for r in c.get("runs", [])]
        doc.comments[k] = com
    for k, a in data.get("annotations", {}).items():
        known, extra = _split(Annotation, a)
        doc.annotations[k] = Annotation(**known, extra=extra)
    doc.karaoke = dict(data.get("karaoke", {}))
    for t in data.get("tombstones", []):
        known, _extra = _split(Tombstone, t)
        doc.tombstones.append(Tombstone(**known))
    doc.rules = list(data.get("rules", []))
    doc.structure_changed()
    return doc


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON via a temp file + rename, so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=1)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_document(doc: Document, path: Path) -> None:
    path = Path(path)
    data = document_to_json(doc)
    if data["format_version"] > 1 and path.is_file():
        original = path.read_bytes()
        previous = int(json.loads(original).get("format_version", 1))
        if previous < data["format_version"]:
            # Preserve the exact last v1 file before a source-copy upgrades it.
            suffix = 0
            while True:
                tail = f"-{suffix}" if suffix else ""
                backup = path.with_name(f"{path.stem}.v{previous}{tail}.bak.json")
                try:
                    with backup.open("xb") as stream:
                        stream.write(original)
                        stream.flush()
                        os.fsync(stream.fileno())
                    break
                except FileExistsError:
                    suffix += 1
    write_json_atomic(path, data)


def load_document(path: Path) -> Document:
    return document_from_json(read_json(path))
