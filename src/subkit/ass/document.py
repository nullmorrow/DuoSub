"""Lossless-ish ASS document model.

Everything the tool does not understand is preserved verbatim: unknown
sections, comments in [Script Info], unknown event types and extra fields.
Known sections are re-serialised from the model so edits are easy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ..encoding import DecodedText, decode_bytes, write_text
from .timefmt import ass_to_cs, cs_to_ass

DEFAULT_STYLE_FORMAT = [
    "Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour", "OutlineColour", "BackColour",
    "Bold", "Italic", "Underline", "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle",
    "Outline", "Shadow", "Alignment", "MarginL", "MarginR", "MarginV", "Encoding",
]
DEFAULT_EVENT_FORMAT = ["Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text"]

_STYLE_DEFAULTS = {
    "Name": "Default", "Fontname": "Arial", "Fontsize": "48", "PrimaryColour": "&H00FFFFFF",
    "SecondaryColour": "&H000000FF", "OutlineColour": "&H00000000", "BackColour": "&H00000000",
    "Bold": "0", "Italic": "0", "Underline": "0", "StrikeOut": "0", "ScaleX": "100", "ScaleY": "100",
    "Spacing": "0", "Angle": "0", "BorderStyle": "1", "Outline": "2", "Shadow": "0", "Alignment": "2",
    "MarginL": "10", "MarginR": "10", "MarginV": "10", "Encoding": "1",
}

_EVENT_KINDS = ("Dialogue", "Comment", "Picture", "Sound", "Movie", "Command")


def _canon_field(name: str) -> str:
    """Canonical event/style field names (Actor == Name)."""
    n = name.strip()
    low = n.lower()
    if low in ("actor", "name"):
        return "Name"
    for known in DEFAULT_STYLE_FORMAT + DEFAULT_EVENT_FORMAT + ["Marked"]:
        if known.lower() == low:
            return known
    return n


def _num(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _fmt_num(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


@dataclass
class Style:
    values: dict[str, str] = field(default_factory=dict)

    @classmethod
    def new(cls, name: str, **overrides: str) -> "Style":
        values = dict(_STYLE_DEFAULTS)
        values["Name"] = name
        for k, v in overrides.items():
            values[_canon_field(k)] = str(v)
        return cls(values)

    @property
    def name(self) -> str:
        return self.values.get("Name", "")

    @name.setter
    def name(self, value: str) -> None:
        self.values["Name"] = value

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(_canon_field(key), default)

    def set(self, key: str, value) -> None:
        self.values[_canon_field(key)] = value if isinstance(value, str) else _fmt_num(float(value))

    def num(self, key: str, default: float = 0.0) -> float:
        return _num(self.get(key, ""), default)

    # convenient typed accessors
    @property
    def fontname(self) -> str:
        return self.get("Fontname").lstrip("@")

    @property
    def fontsize(self) -> float:
        return self.num("Fontsize", 48.0)

    @property
    def scale_x(self) -> float:
        return self.num("ScaleX", 100.0)

    @property
    def scale_y(self) -> float:
        return self.num("ScaleY", 100.0)

    @property
    def spacing(self) -> float:
        return self.num("Spacing", 0.0)

    @property
    def outline(self) -> float:
        return self.num("Outline", 0.0)

    @property
    def shadow(self) -> float:
        return self.num("Shadow", 0.0)

    @property
    def alignment(self) -> int:
        return int(self.num("Alignment", 2))

    @property
    def bold(self) -> bool:
        return self.num("Bold", 0) != 0

    @property
    def italic(self) -> bool:
        return self.num("Italic", 0) != 0

    def copy(self, name: Optional[str] = None) -> "Style":
        s = Style(dict(self.values))
        if name is not None:
            s.name = name
        return s

    def to_line(self, fmt: list[str]) -> str:
        return "Style: " + ",".join(self.values.get(_canon_field(k), _STYLE_DEFAULTS.get(_canon_field(k), "")) for k in fmt)


@dataclass
class Event:
    kind: str = "Dialogue"
    layer: int = 0
    start: int = 0  # centiseconds
    end: int = 0
    style: str = "Default"
    name: str = ""
    margin_l: int = 0
    margin_r: int = 0
    margin_v: int = 0
    effect: str = ""
    text: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def is_comment(self) -> bool:
        return self.kind == "Comment"

    @property
    def is_dialogue(self) -> bool:
        return self.kind == "Dialogue"

    @property
    def duration(self) -> int:
        return self.end - self.start

    def copy(self, **changes) -> "Event":
        ev = copy.copy(self)
        ev.extra = dict(self.extra)
        for k, v in changes.items():
            setattr(ev, k, v)
        return ev

    def to_line(self, fmt: list[str]) -> str:
        vals = []
        for key in fmt:
            k = _canon_field(key)
            if k == "Layer":
                vals.append(str(self.layer))
            elif k == "Start":
                vals.append(cs_to_ass(self.start))
            elif k == "End":
                vals.append(cs_to_ass(self.end))
            elif k == "Style":
                vals.append(self.style)
            elif k == "Name":
                vals.append(self.name)
            elif k == "MarginL":
                vals.append(str(self.margin_l))
            elif k == "MarginR":
                vals.append(str(self.margin_r))
            elif k == "MarginV":
                vals.append(str(self.margin_v))
            elif k == "Effect":
                vals.append(self.effect)
            elif k == "Text":
                vals.append(self.text)
            else:
                vals.append(self.extra.get(k, "Marked=0" if k == "Marked" else ""))
        return f"{self.kind}: " + ",".join(vals)


def _parse_int(value: str) -> int:
    try:
        return int(float(value.strip()))
    except (ValueError, AttributeError):
        return 0


@dataclass
class _Section:
    name: str  # as written, e.g. "[Events]"
    raw_lines: list[str] = field(default_factory=list)  # only for sections not modelled


class AssDocument:
    """An ASS file.  ``script_info`` keeps (key, value) pairs and raw comment lines in order."""

    def __init__(self) -> None:
        self.script_info: list[tuple[Optional[str], str]] = []  # (key, value) or (None, raw line)
        self.style_format: list[str] = list(DEFAULT_STYLE_FORMAT)
        self.styles: list[Style] = []
        self.event_format: list[str] = list(DEFAULT_EVENT_FORMAT)
        self.events: list[Event] = []
        self._sections: list[_Section] = []
        self.styles_section_name = "[V4+ Styles]"
        self.newline = "\r\n"
        self.bom = True
        self.encoding = "utf-8"
        self.source_path: Optional[Path] = None

    # ------------------------------------------------------------------ parse
    @classmethod
    def load(cls, path: Path | str) -> "AssDocument":
        path = Path(path)
        decoded = decode_bytes(path.read_bytes())
        doc = cls.parse(decoded.text)
        doc.newline = decoded.newline
        doc.bom = bool(decoded.bom) or decoded.encoding == "utf-8"
        doc.encoding = "utf-8" if decoded.encoding.startswith("utf-16") else decoded.encoding
        doc.source_path = path
        return doc

    @classmethod
    def parse(cls, text: str) -> "AssDocument":
        doc = cls()
        doc._sections = []
        current: Optional[_Section] = None
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for raw in lines:
            line = raw.lstrip("﻿")
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                current = _Section(stripped)
                doc._sections.append(current)
                low = stripped.lower()
                if low in ("[v4+ styles]", "[v4 styles]", "[v4 styles+]"):
                    doc.styles_section_name = "[V4+ Styles]"
                continue
            if current is None:
                if stripped:
                    # junk before the first section: keep it in Script Info
                    current = _Section("[Script Info]")
                    doc._sections.append(current)
                else:
                    continue
            low = current.name.lower()
            if low == "[script info]":
                if not stripped:
                    continue
                if stripped.startswith(";") or ":" not in stripped:
                    doc.script_info.append((None, line))
                else:
                    key, value = line.split(":", 1)
                    doc.script_info.append((key.strip(), value.strip()))
            elif low in ("[v4+ styles]", "[v4 styles]", "[v4 styles+]"):
                if not stripped or stripped.startswith(";"):
                    continue
                if stripped.lower().startswith("format:"):
                    doc.style_format = [_canon_field(x) for x in stripped.split(":", 1)[1].split(",")]
                elif stripped.lower().startswith("style:"):
                    body = stripped.split(":", 1)[1].strip()
                    parts = body.split(",", len(doc.style_format) - 1)
                    values = {}
                    for i, key in enumerate(doc.style_format):
                        values[key] = parts[i].strip() if i < len(parts) else _STYLE_DEFAULTS.get(key, "")
                    doc.styles.append(Style(values))
            elif low == "[events]":
                if not stripped or stripped.startswith(";"):
                    continue
                if stripped.lower().startswith("format:"):
                    doc.event_format = [_canon_field(x) for x in stripped.split(":", 1)[1].split(",")]
                    continue
                kind, sep, body = line.partition(":")
                kind = kind.strip()
                if not sep or kind not in _EVENT_KINDS:
                    continue
                doc.events.append(doc._parse_event(kind, body.lstrip()))
            else:
                current.raw_lines.append(raw)
        return doc

    def _parse_event(self, kind: str, body: str) -> Event:
        fmt = self.event_format
        parts = body.split(",", len(fmt) - 1)
        while len(parts) < len(fmt):
            parts.append("")
        ev = Event(kind=kind)
        for key, value in zip(fmt, parts):
            if key == "Layer":
                ev.layer = _parse_int(value)
            elif key == "Start":
                ev.start = ass_to_cs(value)
            elif key == "End":
                ev.end = ass_to_cs(value)
            elif key == "Style":
                ev.style = value.strip()
            elif key == "Name":
                ev.name = value.strip()
            elif key == "MarginL":
                ev.margin_l = _parse_int(value)
            elif key == "MarginR":
                ev.margin_r = _parse_int(value)
            elif key == "MarginV":
                ev.margin_v = _parse_int(value)
            elif key == "Effect":
                ev.effect = value.strip()
            elif key == "Text":
                ev.text = value
            else:
                ev.extra[key] = value
        return ev

    # ------------------------------------------------------------- script info
    def info(self, key: str, default: Optional[str] = None) -> Optional[str]:
        low = key.lower()
        for k, v in self.script_info:
            if k is not None and k.lower() == low:
                return v
        return default

    def set_info(self, key: str, value) -> None:
        value = str(value)
        low = key.lower()
        for i, (k, _v) in enumerate(self.script_info):
            if k is not None and k.lower() == low:
                self.script_info[i] = (k, value)
                return
        self.script_info.append((key, value))

    def remove_info(self, key: str) -> None:
        low = key.lower()
        self.script_info = [(k, v) for k, v in self.script_info if k is None or k.lower() != low]

    @property
    def playres(self) -> tuple[int, int]:
        x = _parse_int(self.info("PlayResX", "0") or "0")
        y = _parse_int(self.info("PlayResY", "0") or "0")
        if x <= 0 and y <= 0:
            return 384, 288  # the ASS spec default
        if x <= 0:
            x = 1280 if y == 1024 else int(round(y * 4 / 3))
        if y <= 0:
            y = 1024 if x == 1280 else int(round(x * 3 / 4))
        return x, y

    @property
    def scaled_border(self) -> bool:
        return (self.info("ScaledBorderAndShadow", "no") or "no").strip().lower() in ("yes", "1", "true")

    # ------------------------------------------------------------------ styles
    def style(self, name: str) -> Optional[Style]:
        for s in self.styles:
            if s.name == name:
                return s
        low = name.lstrip("*").lower()
        for s in self.styles:
            if s.name.lower() == low:
                return s
        return None

    def style_map(self) -> dict[str, Style]:
        return {s.name: s for s in self.styles}

    def upsert_style(self, style: Style) -> None:
        for i, s in enumerate(self.styles):
            if s.name == style.name:
                self.styles[i] = style
                return
        self.styles.append(style)

    # ------------------------------------------------------------- sections
    def raw_section(self, name: str) -> Optional[list[str]]:
        for sec in self._sections:
            if sec.name.lower() == name.lower():
                return sec.raw_lines
        return None

    def drop_section(self, name: str) -> None:
        self._sections = [s for s in self._sections if s.name.lower() != name.lower()]

    # -------------------------------------------------------------- serialize
    def serialize(self) -> str:
        out: list[str] = []
        names = [s.name.lower() for s in self._sections]
        order: list[_Section] = list(self._sections)
        if "[script info]" not in names:
            order.insert(0, _Section("[Script Info]"))
        if not any(n in ("[v4+ styles]", "[v4 styles]", "[v4 styles+]") for n in names):
            idx = 1 if order and order[0].name.lower() == "[script info]" else 0
            order.insert(idx, _Section("[V4+ Styles]"))
        if "[events]" not in names:
            order.append(_Section("[Events]"))
        for sec in order:
            low = sec.name.lower()
            if out:
                out.append("")
            if low == "[script info]":
                out.append("[Script Info]")
                for k, v in self.script_info:
                    out.append(v if k is None else f"{k}: {v}")
            elif low in ("[v4+ styles]", "[v4 styles]", "[v4 styles+]"):
                out.append(self.styles_section_name)
                out.append("Format: " + ", ".join(self.style_format))
                for s in self.styles:
                    out.append(s.to_line(self.style_format))
            elif low == "[events]":
                out.append("[Events]")
                out.append("Format: " + ", ".join(self.event_format))
                for ev in self.events:
                    out.append(ev.to_line(self.event_format))
            else:
                out.append(sec.name)
                raw = list(sec.raw_lines)
                while raw and not raw[-1].strip():
                    raw.pop()
                out.extend(raw)
        return "\n".join(out) + "\n"

    def save(self, path: Path | str, *, newline: Optional[str] = None) -> None:
        write_text(path, self.serialize(), newline=newline or self.newline or "\r\n", bom=True, encoding="utf-8")

    # ------------------------------------------------------------- helpers
    def dialogues(self, styles: Optional[Iterable[str]] = None) -> Iterator[Event]:
        wanted = set(styles) if styles is not None else None
        for ev in self.events:
            if ev.is_dialogue and (wanted is None or ev.style in wanted):
                yield ev

    def copy(self) -> "AssDocument":
        return copy.deepcopy(self)
