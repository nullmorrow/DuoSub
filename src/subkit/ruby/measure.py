"""Exact text measurement with libass (through ffmpeg), batched and cached.

Every measurement is a *probe*: one dialogue line rendered in its own horizontal
band of a tall canvas.  Many probes are packed into one frame and all frames
are produced by a single ffmpeg run whose raw gray output is read from a pipe.

Per-character boxes of a line come from 3 probes: characters k ≡ 0,1,2 (mod 3)
visible.  Visible glyphs are then ≥2 characters apart, so their ink can be
separated by column projection even for glyphs with internal gaps (い, 川, ハ).
If the number of ink groups does not match (rare), that phase falls back to one
probe per character.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..ass.document import AssDocument, Style
from ..ass.tags import FONT_TAGS
from ..ass.text import tokenize
from ..proc import CancelledError, KillableRun, tool
from ..core.fontcheck import installed_font_files

CACHE_VERSION = "m4"      # m4: installed font files and the resolved ffmpeg binary invalidate probes
_LAYOUT_TAGS = FONT_TAGS | {"fe"}


@lru_cache(maxsize=8192)
def _resolved_file(path: str, size: int, mtime: int, device: int, inode: int) -> str:
    return str(Path(path).resolve())


def file_signature(path: Path | str) -> tuple:
    path = Path(path)
    try:
        stat = path.stat()
        resolved = _resolved_file(str(path), stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)
        return resolved, stat.st_size, stat.st_mtime_ns
    except OSError:
        return str(path.resolve()), "not installed"


_ENV_RECHECK_S = 10.0       # registry list + font folder times: cheap, checked at most this often
_ENV_RESTAT_S = 300.0       # every file signature: ~0.5–2 s with a few thousand fonts
_environments: dict[str, tuple[float, float, tuple, tuple]] = {}   # binary -> (checked, stat'ed, marker, value)


def _font_folders() -> tuple:
    folders = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
    if os.environ.get("LOCALAPPDATA"):
        folders.append(Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Windows" / "Fonts")
    return tuple(file_signature(folder) for folder in folders)


def forget_measurement_environment() -> None:
    _environments.clear()


def measurement_environment(ffmpeg: str = "ffmpeg") -> tuple:
    """Include fallback/variant files too: libass can select them for individual glyphs.

    Every preview asks for this, so the full per-file signature is recomputed only when
    the registered font list, the font folders or ffmpeg changed, or every few minutes.
    """
    binary = tool(ffmpeg) if ffmpeg == "ffmpeg" else ffmpeg
    now = time.monotonic()
    hit = _environments.get(binary)
    if hit is not None and now - hit[0] < _ENV_RECHECK_S:
        return hit[3]
    registered = installed_font_files()
    marker = (file_signature(binary), _font_folders(),
              tuple(sorted((name, tuple(map(str, paths))) for name, paths in registered.items())))
    if hit is not None and hit[2] == marker and now - hit[1] < _ENV_RESTAT_S:
        _environments[binary] = (now, hit[1], marker, hit[3])
        return hit[3]
    signatures = {p: file_signature(p) for paths in registered.values() for p in paths}
    fonts = tuple(sorted((name, tuple(sorted(signatures[p] for p in paths)))
                         for name, paths in registered.items()))
    value = (file_signature(binary), fonts)
    _environments[binary] = (now, now, marker, value)
    return value


@dataclass
class Box:
    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def h_align(an: int) -> int:
    return {1: 0, 4: 0, 7: 0, 2: 1, 5: 1, 8: 1, 3: 2, 6: 2, 9: 2}.get(an, 1)


def layout_tokens(text: str) -> list[tuple[str, str]]:
    """("tags", "{…}") / ("char", ch) / ("break", raw) with only layout-relevant tags kept."""
    out: list[tuple[str, str]] = []
    for tok in tokenize(text):
        if tok.kind == "tags":
            kept = "".join(t.raw for t in tok.tags if t.name in _LAYOUT_TAGS)
            if kept:   # as an override block: bare "\fs45" would be drawn as (invisible) text and shift the line
                out.append(("tags", "{" + kept + "}"))
        elif tok.kind == "text":
            out.extend(("char", ch) for ch in tok.raw)
        elif tok.kind == "hardspace":
            out.append(("char", "\\h"))
        elif tok.kind in ("break", "softbreak"):
            out.append(("break", tok.raw))
    return out


@dataclass
class LineRequest:
    style: str
    text: str          # original dialogue text (tags allowed)
    an: int
    x: float           # anchor x the line is laid out against


class Measurer:
    def __init__(self, doc: AssDocument, *, ffmpeg: str = "ffmpeg", cache_path: Optional[Path] = None,
                 wrap_style: int = 0, raster_scale: int = 1, runner: Optional[KillableRun] = None, renderer=None) -> None:
        if not isinstance(raster_scale, int) or raster_scale < 1:
            raise ValueError("raster_scale must be a positive integer")
        self.doc = doc
        self.renderer = renderer
        self.runner = runner if runner is not None else KillableRun()
        self._owns_runner = runner is None
        # Only retry directories explicitly abandoned after their child exited.
        for pending in Path(tempfile.gettempdir()).glob("subkit_probe_*/.cleanup-pending"):
            shutil.rmtree(pending.parent, ignore_errors=True)
        self.raster_scale = raster_scale
        self.ffmpeg = tool(ffmpeg) if ffmpeg == "ffmpeg" else ffmpeg
        self._environment = hashlib.sha1(json.dumps(measurement_environment(self.ffmpeg),
                                                     ensure_ascii=False).encode("utf-8")).hexdigest()
        self.playres_x, self.playres_y = doc.playres
        self.wrap_style = wrap_style
        self._mem: dict[str, list[Box]] = {}
        self._db: Optional[sqlite3.Connection] = None
        self.font_log: set[str] = set()
        self.renders = 0
        self._style_lines: dict[str, str] = {}   # style lines are part of every cache key: build each once
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(cache_path))
            self._db.execute("CREATE TABLE IF NOT EXISTS probe (k TEXT PRIMARY KEY, v TEXT)")

    def close(self) -> None:
        if self._owns_runner:
            self.cancel()
        if self._db is not None:
            self._db.close()
            self._db = None

    def cancel(self) -> None:
        self.runner.cancel()

    def __enter__(self) -> "Measurer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ helpers
    def anchor_x(self, style: str, an: int, margin_l: int = 0, margin_r: int = 0, pos_x: Optional[float] = None) -> float:
        if pos_x is not None:
            return pos_x
        st = self.doc.style(style) or Style.new(style)
        ml = margin_l or int(st.num("MarginL"))
        mr = margin_r or int(st.num("MarginR"))
        h = h_align(an)
        if h == 0:
            return float(ml)
        if h == 2:
            return float(self.playres_x - mr)
        return ml + (self.playres_x - ml - mr) / 2.0

    def _style_line(self, style: str) -> str:
        line = self._style_lines.get(style)
        if line is None:
            st = self.doc.style(style)
            line = st.to_line(self.doc.style_format) if st else style
            self._style_lines[style] = line
        return line

    def _key(self, style: str, probe: str, an_h: int, x: float) -> str:
        version = CACHE_VERSION if self.raster_scale == 1 else f"{CACHE_VERSION}@{self.raster_scale}x"
        if self.raster_scale != 1:
            x = float(x)  # 400 and 400.0 describe the same precise probe.
        if self.renderer is not None:
            version += ":" + self.renderer.cache_id
        raw = json.dumps([version, self.playres_x, self.wrap_style, self._style_line(style),
                          probe, an_h, round(x, 2), self._environment], ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _get(self, key: str) -> Optional[list[Box]]:
        if key in self._mem:
            return self._mem[key]
        if self._db is not None:
            row = self._db.execute("SELECT v FROM probe WHERE k=?", (key,)).fetchone()
            if row:
                boxes = [Box(*b) for b in json.loads(row[0])]
                self._mem[key] = boxes
                return boxes
        return None

    # ------------------------------------------------------------ public API
    def line_char_boxes(self, reqs: Sequence[LineRequest]) -> list[list[Optional[Box]]]:
        """Per visible character ink boxes for each line (None for spaces/no ink)."""
        plans = []  # per request: list of (phase_chars, key)
        todo: dict[str, tuple[str, str, int, float]] = {}
        for rq in reqs:
            toks = layout_tokens(rq.text)
            chars = [v for k, v in toks if k == "char"]
            phases = []
            for phase in range(3):
                vis_idx = [i for i, ch in enumerate(chars) if i % 3 == phase and ch not in (" ", "\u3000", "\\h", "\t")]
                probe = self._probe_text(toks, set(vis_idx))
                key = self._key(rq.style, probe, h_align(rq.an), rq.x)
                phases.append((vis_idx, key, probe))
                if self._get(key) is None:
                    todo[key] = (rq.style, probe, h_align(rq.an), rq.x)
            plans.append((toks, chars, phases))
        self._render(todo)
        # validation + per-char fallback
        todo2: dict[str, tuple[str, str, int, float]] = {}
        singles: list[list[tuple[int, str]]] = []
        for rq, (toks, chars, phases) in zip(reqs, plans):
            need: list[tuple[int, str]] = []
            for vis_idx, key, _probe in phases:
                boxes = self._get(key) or []
                if len(boxes) != len(vis_idx):
                    for i in vis_idx:
                        probe = self._probe_text(toks, {i})
                        k2 = self._key(rq.style, probe, h_align(rq.an), rq.x)
                        need.append((i, k2))
                        if self._get(k2) is None:
                            todo2[k2] = (rq.style, probe, h_align(rq.an), rq.x)
            singles.append(need)
        self._render(todo2)
        out = []
        for (toks, chars, phases), need in zip(plans, singles):
            res: list[Optional[Box]] = [None] * len(chars)
            for vis_idx, key, _probe in phases:
                boxes = self._get(key) or []
                if len(boxes) == len(vis_idx):
                    for i, b in zip(vis_idx, boxes):
                        res[i] = b
            for i, k2 in need:
                boxes = self._get(k2) or []
                if boxes:
                    res[i] = Box(min(b.x0 for b in boxes), max(b.x1 for b in boxes), min(b.y0 for b in boxes), max(b.y1 for b in boxes))
            out.append(res)
        return out

    def text_boxes(self, items: Sequence[tuple[str, str]]) -> list[Optional[Box]]:
        """Ink box of whole short texts [(style, text)] (e.g. ruby strings), centred at screen middle."""
        x = self.playres_x / 2.0
        keys = []
        todo: dict[str, tuple[str, str, int, float]] = {}
        for style, text in items:
            probe = "{\\bord0\\shad0\\blur0\\be0\\1c&HFFFFFF&\\alpha&H00&}" + text
            key = self._key(style, "T" + probe, 1, x)
            keys.append(key)
            if self._get(key) is None:
                todo[key] = (style, probe, 1, x)
        self._render(todo, merge=True)
        out = []
        for key in keys:
            bs = self._get(key) or []
            out.append(bs[0] if bs else None)
        return out

    def _span_text(self, style: str, text: str, start: int, end: int) -> Optional[str]:
        """A single-line span with its inherited font state, without paint tags."""
        if start < 0 or end <= start:
            return None
        base = self.doc.style(style) or Style.new(style)

        def reset(name):
            st = self.doc.style(name) or base
            fields = (("fn", "Fontname"), ("fs", "Fontsize"), ("fscx", "ScaleX"),
                      ("fscy", "ScaleY"), ("fsp", "Spacing"), ("b", "Bold"),
                      ("i", "Italic"), ("fe", "Encoding"))
            return "{" + "".join("\\" + tag + st.get(field) for tag, field in fields) + "}"

        parts, index, included = [], 0, False
        for tok in tokenize(text):
            if tok.kind == "tags":
                for tag in tok.tags:
                    if tag.name == "r":
                        # Expanding resets keeps named style metrics in the key,
                        # without resetting the probe's white ink or alignment.
                        parts.append(reset(tag.arg))
                    elif tag.name in _LAYOUT_TAGS:
                        parts.append("{" + tag.raw + "}")
                continue
            if tok.kind == "break" or (tok.kind == "softbreak" and self.wrap_style == 2):
                if included:
                    return None
                continue
            chars = list(tok.raw) if tok.kind == "text" else ["\\h"] if tok.kind in ("hardspace", "softbreak") else []
            for char in chars:
                if start <= index < end:
                    parts.append(char)
                    included = True
                index += 1
                if index == end:
                    return "".join(parts)
        return None

    def span_advances(self, spans: Sequence[tuple[str, str, int, int]]) -> list[Optional[float]]:
        """Libass layout width of each base span, including its font/size/spacing.

        Left/right-aligned copies of identical ink differ by the layout advance;
        glyph bearings cancel. A small inset avoids clipping italic overhangs.
        Fourfold probes reduce threshold rounding; bounded batches keep whole
        episodes from allocating all high-resolution frames at once.
        """
        previous_scale = self.raster_scale
        self.raster_scale = max(4, previous_scale)
        try:
            plans, todo = [], {}
            inset = min(64.0, self.playres_x / 8.0)
            for style, text, start, end in spans:
                fragment = self._span_text(style, text, start, end)
                if fragment is None:
                    plans.append(None)
                    continue
                probe = "{\\q2\\frz0\\bord0\\shad0\\blur0\\be0\\1c&HFFFFFF&\\alpha&H00&}" + fragment
                keys = []
                for align, x in ((0, inset), (2, self.playres_x - inset)):
                    key = self._key(style, "A1" + probe, align, x)
                    keys.append(key)
                    if self._get(key) is None:
                        todo[key] = (style, probe, align, x)
                plans.append(keys)
            pending = list(todo.items())
            for offset in range(0, len(pending), 8):
                self.runner.check_cancelled()
                self._render(dict(pending[offset:offset + 8]), merge=True)
            widths = []
            for keys in plans:
                left, right = ((self._get(key) or []) for key in keys) if keys else ([], [])
                if not left or not right or min(left[0].x0, right[0].x0) <= 0 or max(left[0].x1, right[0].x1) >= self.playres_x:
                    widths.append(None)
                    continue
                width = self.playres_x - 2 * inset + left[0].x0 - right[0].x0
                widths.append(width if width > 0 and abs(left[0].width - right[0].width) <= 0.5 else None)
            return widths
        finally:
            self.raster_scale = previous_scale

    def _probe_text(self, toks: list[tuple[str, str]], visible: set[int]) -> str:
        parts = ["{\\alpha&HFF&\\bord0\\shad0\\blur0\\be0\\1c&HFFFFFF&}"]   # ink must be visible on the black canvas
        i = 0
        cur_on = False
        for kind, val in toks:
            if kind == "char":
                on = i in visible
                if on != cur_on:
                    parts.append("{\\alpha&H00&}" if on else "{\\alpha&HFF&}")
                    cur_on = on
                parts.append(val)
                i += 1
            else:
                parts.append(val)
        return "".join(parts)

    # ------------------------------------------------------------ rendering
    def _render(self, todo: dict[str, tuple[str, str, int, float]], merge: bool = False) -> None:
        if not todo:
            return
        items = list(todo.items())
        max_fs = 10.0
        for _k, (style, text, _a, _x) in items:
            st = self.doc.style(style)
            fs = (st.fontsize if st else 48) * (1 + text.count("\\N"))
            if self.raster_scale != 1:
                # Large inline scales/fonts must not spill into the next band.
                sizes = [st.fontsize if st else 48]
                scales = [st.scale_y if st else 100]
                for tok in tokenize(text):
                    if tok.kind == "tags":
                        for tag in tok.tags:
                            if tag.name in ("fs", "fscy") and tag.num() is not None:
                                (sizes if tag.name == "fs" else scales).append(tag.num())
                fs = max(sizes) * max(scales) / 100 * (1 + text.count("\\N"))
            max_fs = max(max_fs, fs)
        row_h = int(math.ceil(max_fs * 2.0)) + 10
        rows_per_frame = max(1, min(4320 // row_h, len(items)))
        frame_h = rows_per_frame * row_h
        frame_h += frame_h % 2
        n_frames = int(math.ceil(len(items) / rows_per_frame))
        scale = self.raster_scale
        raster_w, raster_h = self.playres_x * scale, frame_h * scale
        lines = []
        for idx, (key, (style, text, an_h, x)) in enumerate(items):
            f, r = divmod(idx, rows_per_frame)
            an = {0: 7, 1: 8, 2: 9}[an_h]
            lines.append(f"Dialogue: 0,{_t(f)},{_t(f + 1)},{style},,0,0,0,,{{\\an{an}\\pos({x:.2f},{r * row_h + 5})}}{text}")
        styles = "\n".join(s.to_line(self.doc.style_format) for s in self.doc.styles)
        content = (
            f"[Script Info]\nScriptType: v4.00+\nWrapStyle: {self.wrap_style}\nScaledBorderAndShadow: yes\n"
            f"PlayResX: {self.playres_x}\nPlayResY: {frame_h}\n\n[V4+ Styles]\nFormat: {', '.join(self.doc.style_format)}\n{styles}\n\n"
            f"[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n" + "\n".join(lines) + "\n"
        )
        self.runner.check_cancelled()
        if self.renderer is None:
            frames = self._render_ffmpeg(content, raster_w, raster_h, n_frames)
        else:
            try:
                frames, fonts = self.renderer.render(content, raster_w, raster_h, n_frames, self.runner)
                self.font_log.update(fonts)
            except CancelledError:
                raise
            except Exception:
                self.runner.check_cancelled()
                frames = self._render_ffmpeg(content, raster_w, raster_h, n_frames)
        self.renders += 1
        for idx, (key, (style, text, an_h, x)) in enumerate(items):
            f, r = divmod(idx, rows_per_frame)
            band = frames[f, r * row_h * scale : (r + 1) * row_h * scale]
            st = self.doc.style(style)
            fs = (st.fontsize if st else 48) * ((st.scale_x if st else 100) / 100.0)
            boxes = _segment(band, fs * scale, merge_all=merge, probe_y=5 * scale)
            if scale != 1:
                boxes = [Box(b.x0 / scale, b.x1 / scale, b.y0 / scale, b.y1 / scale) for b in boxes]
            self._mem[key] = boxes
            if self._db is not None:
                self._db.execute("INSERT OR REPLACE INTO probe VALUES (?,?)", (key, json.dumps([[b.x0, b.x1, b.y0, b.y1] for b in boxes])))
        if self._db is not None:
            self._db.commit()

    def _render_ffmpeg(self, content: str, raster_w: int, raster_h: int, n_frames: int) -> np.ndarray:
        tmp = tempfile.mkdtemp(prefix="subkit_probe_")
        try:
            Path(tmp, "probe.ass").write_text(content, encoding="utf-8-sig")
            cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "verbose", "-f", "lavfi", "-i",
                   f"color=c=black:s={raster_w}x{raster_h}:r=1:d={n_frames}", "-vf", "ass=probe.ass,format=gray",
                   "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
            proc = self.runner.run(cmd, cwd=tmp)
        finally:
            # No interpreter-exit finalizer or recursive permission retry. The
            # runner has reaped the child before its working directory is removed.
            shutil.rmtree(tmp, ignore_errors=True)
            if Path(tmp).exists():
                try:
                    Path(tmp, ".cleanup-pending").touch()
                except OSError:
                    pass
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg/libass measurement failed: " + proc.stderr.decode("utf-8", "replace")[-800:])
        for line in proc.stderr.decode("utf-8", "replace").splitlines():
            if "fontselect" in line:
                self.font_log.add(line.split("]", 1)[-1].strip())
        expected = n_frames * raster_h * raster_w
        data = np.frombuffer(proc.stdout, dtype=np.uint8)
        if data.size < expected:
            raise RuntimeError(f"ffmpeg returned {data.size} bytes, expected {expected}")
        return data[:expected].reshape(n_frames, raster_h, raster_w)


def _t(sec: int) -> str:
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}.00"


def _segment(band: np.ndarray, fs: float, threshold: int = 24, merge_all: bool = False,
             probe_y: float = 5) -> list[Box]:
    """Split the ink of a probe band into glyph boxes by column projection.

    y values are relative to the band top (the probe was anchored at y=5 with \\an7/8/9)."""
    mask = band > threshold
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return []
    gap = max(3.0, fs * 0.45)
    groups: list[list[int]] = [[int(cols[0]), int(cols[0])]]
    for c in cols[1:]:
        c = int(c)
        if not merge_all and c - groups[-1][1] > gap:
            groups.append([c, c])
        else:
            groups[-1][1] = c
    boxes = []
    for a, b in groups:
        rows = np.where(mask[:, a : b + 1].any(axis=1))[0]
        boxes.append(Box(float(a), float(b + 1), float(rows[0] - probe_y), float(rows[-1] + 1 - probe_y)))
    return boxes
