"""Read-only release font checks, measured by the local ffmpeg/libass."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ..ass import AssDocument, Event
from ..ass.text import VisibleText, tokenize
from ..ass.timefmt import cs_to_ass
from ..proc import KillableRun, run
from ..templates import FontLog, parse_font_log

_FRAME_CS = 4                         # 25 fps, exactly representable in ASS centiseconds
_EVENTS_PER_FRAME = 32                # overlap does not affect font selection
DEFAULT_FONT_LIBRARY = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "assfonts" / "fonts.json"


def installed_font_files() -> dict[str, list[Path]]:
    """Fresh registered files, including localized aliases and face variants."""
    files: dict[str, list[Path]] = {}
    if os.name == "nt":
        import winreg

        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts") as key:
                    for i in range(winreg.QueryInfoKey(key)[1]):
                        name, value, _kind = winreg.EnumValue(key, i)
                        if not isinstance(value, str):
                            continue
                        path = Path(value)
                        if not path.is_absolute():
                            path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / path
                        name = re.sub(r"\s+\([^)]*\)$", "", name)
                        for alias in name.split(" & "):
                            files.setdefault(alias.casefold(), []).append(path)
            except OSError:
                continue
    return files


@lru_cache(maxsize=1)
def installed_font_aliases() -> frozenset[str]:
    """Windows registers localized names that Qt's family list can omit."""
    names: set[str] = set()
    if os.name == "nt":
        import winreg

        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts") as key:
                    for i in range(winreg.QueryInfoKey(key)[1]):
                        name, value, _kind = winreg.EnumValue(key, i)
                        if not isinstance(value, str):
                            continue
                        path = Path(value)
                        if not path.is_absolute():
                            path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / path
                        if path.is_file():
                            name = re.sub(r"\s+\([^)]*\)$", "", name)
                            names.update(alias.casefold() for alias in name.split(" & "))
            except OSError:
                continue
    return frozenset(names)


class FontLibrary:
    """Private assfonts index: read lazily once per configured path, never install fonts."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._names: Optional[dict[str, list[str]]] = None
        self.error = ""

    def lookup(self, name: str) -> list[str]:
        if self._names is None:
            self._names = {}
            try:
                with self.path.open(encoding="utf-8-sig") as stream:
                    entries = json.load(stream)
                if not isinstance(entries, list):
                    raise ValueError("字体库必须是 JSON 列表")
                for entry in entries:
                    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                        continue
                    for key in ("families", "fullnames", "psnames"):
                        names = entry.get(key) or []
                        if isinstance(names, str):
                            names = [names]
                        if not isinstance(names, list):
                            continue
                        for alias in names:
                            if isinstance(alias, str):
                                paths = self._names.setdefault(alias.lower(), [])
                                if entry["path"] not in paths:
                                    paths.append(entry["path"])
            except (OSError, ValueError) as exc:
                self.error = str(exc)
        return self._names.get(name.lower(), []).copy()


@dataclass
class FontCheck:
    ran: bool = False
    error: str = ""
    families: list[str] = field(default_factory=list)
    missing: dict[str, list[str]] = field(default_factory=dict)
    gaps: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    undrawable: dict[str, tuple[str, list[str]]] = field(default_factory=dict)
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.ran and not (self.missing or self.gaps or self.undrawable)


def _family(name: str) -> str:
    return name.strip().lstrip("@")


def _family_key(name: str) -> str:
    return _family(name).casefold()


def font_runs(event: Event, doc: AssDocument) -> Iterator[tuple[str, str]]:
    """The requested family of each text run; drawings and comments request no glyphs."""
    style = doc.style(event.style) or doc.style("Default")
    base = _family(style.fontname) if style is not None else ""
    family, drawing = base, False
    for tok in tokenize(event.text):
        if tok.kind == "tags":
            for tag in tok.tags:
                arg = tag.arg.strip()
                if tag.name == "fn":
                    family = base if arg in ("", "0") else _family(arg)
                elif tag.name == "r":
                    reset = doc.style(arg) if arg else None
                    family = _family(reset.fontname) if reset is not None else base
                elif tag.name == "p":
                    v = tag.num()
                    drawing = bool(v and v > 0)
        elif tok.kind == "text":
            yield family, tok.raw
        elif tok.kind == "hardspace" and not drawing:
            yield family, "\u00a0"


def _slot(index: int) -> tuple[int, int]:
    """Start and end (cs) of the index-th probe event.  ffmpeg's ass filter truncates ``pts * 0.04 * 1000`` to whole
    milliseconds, which lands 1 ms early on some frames (803, 811, …): each slot opens 1 cs before its frame."""
    frame = (index // _EVENTS_PER_FRAME) * _FRAME_CS
    return max(0, frame - 1), frame + _FRAME_CS - 1


def probe_ass(doc: AssDocument) -> str:
    """Retime a deep copy so every Dialogue is active at a probe frame timestamp."""
    probe = doc.copy()
    probe.events = [ev for ev in probe.events if ev.is_dialogue]
    for i, ev in enumerate(probe.events):
        ev.start, ev.end = _slot(i)
    return probe.serialize()


def _ref(event: Event, doc: AssDocument) -> str:
    text = VisibleText(event.text, wrap_style=int(doc.info("WrapStyle", "0") or "0")).text[:24]
    return f"{cs_to_ass(event.start)} [{event.style}] {text}"


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _attribute(doc: AssDocument, log: FontLog) -> FontCheck:
    names: dict[str, str] = {}
    usage: list[tuple[str, dict[str, str]]] = []
    for ev in doc.dialogues():
        chars: dict[str, str] = {}
        for family, text in font_runs(ev, doc):
            key = _family_key(family)
            if key:
                names.setdefault(key, _family(family))
                chars[key] = chars.get(key, "") + text
        usage.append((_ref(ev, doc), chars))

    selected: dict[str, str] = {}
    for family, ps in log.selected.items():
        key = _family_key(family)
        names.setdefault(key, _family(family))
        selected.setdefault(key, ps)
    missing = {key for key, ps in selected.items() if ps.lower().startswith("arial") and "arial" not in key}

    def merge(source: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for family, chars in source.items():
            key = _family_key(family)
            names.setdefault(key, _family(family))
            out[key] = "".join(_unique(out.get(key, "") + chars))
        return out

    gaps, undrawable = merge(log.gaps), merge(log.undrawable)
    fc = FontCheck(ran=True, families=sorted(names.values(), key=str.casefold))
    for key in sorted(missing):
        fc.missing[names[key]] = _unique(ref for ref, chars in usage if key in chars)

    def glyph_refs(source: dict[str, str], target: dict[str, tuple[str, list[str]]]) -> None:
        for key, chars in source.items():
            if chars:
                refs = _unique(ref for ref, runs in usage if any(ch in runs.get(key, "") for ch in chars))
                target[names[key]] = (chars, refs)

    # Failed fallbacks cannot also be described as characters displayed by another font.
    gaps = {key: "".join(ch for ch in chars if ch not in undrawable.get(key, ""))
            for key, chars in gaps.items() if key not in missing}
    glyph_refs(gaps, fc.gaps)
    glyph_refs(undrawable, fc.undrawable)
    return fc


def check_fonts(doc: AssDocument, *, ffmpeg: str = "ffmpeg", runner: Optional[KillableRun] = None) -> FontCheck:
    """Probe every exported Dialogue once; failures remain an explicitly unverified check."""
    start = time.perf_counter()
    if runner is not None:
        runner.check_cancelled()
    frames = max(1, (sum(1 for _ in doc.dialogues()) + _EVENTS_PER_FRAME - 1) // _EVENTS_PER_FRAME)
    try:
        with tempfile.TemporaryDirectory(prefix="subkit_fonts_") as tmp:
            Path(tmp, "probe.ass").write_text(probe_ass(doc), encoding="utf-8-sig")
            try:
                result = (runner.run if runner is not None else run)([ffmpeg, "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                              f"color=c=black:s=320x180:r=25:d={frames * _FRAME_CS / 100:.2f}",
                              "-vf", "ass=probe.ass", "-frames:v", str(frames), "-f", "null", "-"],
                             cwd=tmp, capture_output=True)
            except FileNotFoundError:
                return FontCheck(error=f"找不到 ffmpeg：{ffmpeg}", elapsed=time.perf_counter() - start)
        if runner is not None:
            runner.check_cancelled()
        log = result.stderr.decode("utf-8", "replace")
        if result.returncode:
            errors = [ln.strip() for ln in log.splitlines() if "error" in ln.lower() or "failed" in ln.lower()]
            reason = " / ".join(errors or log.splitlines()[-2:])[-240:]
            return FontCheck(error=f"ffmpeg 退出码 {result.returncode}：{reason}", elapsed=time.perf_counter() - start)
    except OSError as exc:
        return FontCheck(error=f"无法运行字体检查：{str(exc)[-240:]}", elapsed=time.perf_counter() - start)
    fc = _attribute(doc, parse_font_log(log))
    fc.elapsed = time.perf_counter() - start
    return fc


def _show_chars(chars: str) -> str:
    def invisible(ch: str) -> bool:
        cp = ord(ch)
        return (ch.isspace() or not ch.isprintable() or ch in "\u034f\u115f\u1160\u2800\u3164\uffa0"
                or 0xFE00 <= cp <= 0xFE0F or 0xE0100 <= cp <= 0xE01EF)

    return "".join(f"U+{ord(ch):04X}" if invisible(ch) else ch for ch in _unique(chars))


def _problem_lines(fc: FontCheck, limit: Optional[int] = None, library: Optional[FontLibrary] = None) -> list[str]:
    lines = ["字体检查（本机 libass 实测）发现问题："]
    remaining = None if limit is None else max(0, limit)

    def examples(refs: list[str]) -> str:
        nonlocal remaining
        refs = _unique(refs)
        shown = refs if remaining is None else refs[:remaining]
        if remaining is not None:
            remaining -= len(shown)
        return "；".join(shown) + ("…" if len(shown) < len(refs) else "")

    if fc.missing:
        refs = _unique(ref for group in fc.missing.values() for ref in group)
        lines.append("没有安装：" + "、".join(fc.missing)
                     + f"——用到的 {len(refs)} 行会换成别的字体显示，振假名位置也是按替代字体测量的。")
        lines.append("  例：" + examples(refs))
        if library is not None:
            for family in fc.missing:
                for path in library.lookup(family):
                    lines.append(f"  {family} — 字体库里有：{path}")
            if library.error and library.path.exists():
                lines.append(f"字体库读取失败：{library.error}")
    for family, (chars, refs) in fc.gaps.items():
        lines.append(f"{family} 缺字：「{_show_chars(chars)}」——这些字由别的字体代替显示"
                     + f"（{len(_unique(refs))} 行，例：{examples(refs)}）")
    if fc.undrawable:
        chars = "".join(chars for chars, _refs in fc.undrawable.values())
        refs = _unique(ref for _chars, group in fc.undrawable.values() for ref in group)
        lines.append(f"没有任何字体能显示：「{_show_chars(chars)}」（{len(refs)} 行，例：{examples(refs)}）")
    return lines


def report_lines(fc: FontCheck, *, library: Optional[FontLibrary] = None) -> list[str]:
    if not fc.ran:
        return [f"字体检查没有运行：{fc.error}（缺字体、缺字没有验证）"]
    lines = ["字体：本机都已安装，没有缺字（libass 实测）"] if fc.ok else _problem_lines(fc, library=library)
    return lines + ["用到的字体：" + "、".join(fc.families)]


def prompt_text(fc: FontCheck, limit: int = 12, *, library: Optional[FontLibrary] = None) -> str:
    if not fc.ran or fc.ok:
        return "\n".join(report_lines(fc, library=library))
    return "\n".join(_problem_lines(fc, limit, library)) + "\n\n仍然导出吗？（会写进导出报告；双语字幕工坊不会替换字体）"
