"""Read-only source records (FORMAT-v1 §5).

A :class:`Source` is one imported subtitle file: its identity (path, SHA-256,
encoding) plus the parsed events with their *exact* text and the structural
metadata spans an analyzer found.  Nothing here is ever written back to the
original file; the project keeps a byte-identical copy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

FORMAT_EVENTS = "subkit-source-events"

SPAN_KINDS = ("speaker_label", "marker", "reading_hint", "sfx", "music", "flashback_mark", "other")


@dataclass
class SourceEvent:
    i: int                      # 0-based order in the file (Dialogue and Comment lines / SRT blocks)
    kind: str                   # Dialogue | Comment | … (SRT: Dialogue)
    start: int                  # ms
    end: int                    # ms
    text: str                   # exact text (ASS Text field verbatim; SRT lines joined with "\n")
    style: Optional[str] = None
    name: Optional[str] = None
    layer: Optional[int] = None
    margins: Optional[list[int]] = None
    effect: Optional[str] = None
    line: Optional[int] = None  # 1-based line number in the file where the event starts
    cue: Optional[str] = None   # SRT block number as written
    extra: dict[str, Any] = field(default_factory=dict)   # unknown fields, kept verbatim


@dataclass
class Span:
    """A piece of an event's text that is structural metadata, not dialogue."""
    e: int
    a: int
    b: int
    kind: str
    value: Optional[str] = None
    base: Optional[list[int]] = None    # reading_hint: [a, b) of the annotated characters
    data: Optional[dict[str, Any]] = None   # e.g. speaker_label: {"readings": [[surface, kana], …], "applies_to_line": n}


@dataclass
class Source:
    id: str                      # episode-local short id: cn1, ja1, en1 …
    lang: str                    # BCP 47: zh-Hans, zh-Hant, ja, en
    lane: str                    # original (spoken language) | translation | reference
    format: str                  # srt | ass | ssa
    path: str                    # original location (read-only)
    copy: str                    # byte-identical copy inside the project (relative to the episode dir)
    sha256: str
    bytes: int
    encoding: str
    newline: str                 # crlf | lf | cr | mixed
    imported_at: str = ""
    events: list[SourceEvent] = field(default_factory=list)
    spans: list[Span] = field(default_factory=list)
    analyzer: dict[str, Any] = field(default_factory=dict)
    roles: dict[int, str] = field(default_factory=dict)   # event index → dialogue | top | song | sign | info | comment …
    doubts: list[dict[str, Any]] = field(default_factory=list)
    playres: Optional[list[int]] = None

    def event(self, i: int) -> SourceEvent:
        return self.events[i]

    def key(self, i: int) -> str:
        return f"{self.id}:{i}"

    def spans_of(self, i: int) -> list[Span]:
        return [s for s in self.spans if s.e == i]

    # ------------------------------------------------------------ json
    def entry_json(self) -> dict[str, Any]:
        """The short entry stored in episode.json."""
        return {"id": self.id, "lang": self.lang, "lane": self.lane, "format": self.format, "path": self.path,
                "copy": self.copy, "sha256": self.sha256, "bytes": self.bytes, "encoding": self.encoding,
                "newline": self.newline, "imported_at": self.imported_at}

    def events_json(self) -> dict[str, Any]:
        def ev(e: SourceEvent) -> dict[str, Any]:
            d = asdict(e)
            if not d["extra"]:
                d.pop("extra")
            return d

        def sp(s: Span) -> dict[str, Any]:
            d = asdict(s)
            for k in ("value", "base", "data"):
                if d[k] is None:
                    d.pop(k)
            return d

        return {"format": FORMAT_EVENTS, "format_version": 1, "source": self.id, "sha256": self.sha256,
                "playres": self.playres,
                "events": [ev(e) for e in self.events], "spans": [sp(s) for s in self.spans],
                "roles": {str(k): v for k, v in sorted(self.roles.items())},
                "doubts": self.doubts, "analyzer": self.analyzer}

    @classmethod
    def from_json(cls, entry: dict[str, Any], events: dict[str, Any]) -> "Source":
        src = cls(**{k: entry.get(k, "") for k in ("id", "lang", "lane", "format", "path", "copy", "sha256",
                                                    "encoding", "newline", "imported_at")},
                  bytes=int(entry.get("bytes", 0)))
        src.events = [SourceEvent(**e) for e in events.get("events", [])]
        src.spans = [Span(**s) for s in events.get("spans", [])]
        src.analyzer = dict(events.get("analyzer", {}))
        src.roles = {int(k): v for k, v in events.get("roles", {}).items()}
        src.doubts = list(events.get("doubts", []))
        src.playres = events.get("playres")
        return src
