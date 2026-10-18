"""Whole-season release export: output names and conflicts (the GUI builds, asks and writes).

Each episode is written as ``<video file name without .mkv>.<language suffix>.ass`` into one folder;
an episode without a video uses its episode id.  Names are compared case-insensitively (Windows).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

from ..model.serialize import read_json

INVALID = set('<>:"/\\|?*')
SUFFIXES = ("zh-Hant", "zh-Hans", "zh-TW", "zh-HK", "zh-CN", "cht", "chs", "ja")


@dataclass
class SeasonEntry:
    eid: str
    video: Optional[Path]
    merged: bool
    sources: tuple[Path, ...] = ()


@dataclass
class SeasonItem:
    eid: str
    target: Optional[Path]           # None: skipped (not merged, or the name is an imported source subtitle)
    note: str = ""                    # 没有视频，用集名 / 还没有合并，跳过 / 和原字幕同名，不能覆盖
    exists: bool = False
    duplicate_of: str = ""            # another episode with the same output name
    direction: Optional[str] = None


def check_suffix(text: str) -> tuple[str, str]:
    """The language suffix as written into names (spaces and outer dots dropped), or an error."""
    suffix = text.strip().strip(".").strip()
    if not suffix:
        return "", "请填写语言后缀，例如 zh-Hant"
    bad = sorted({c for c in suffix if c in INVALID or c.isspace() or ord(c) < 32})
    if bad:
        return "", "语言后缀里不能有：" + " ".join(repr(c)[1:-1] if c.isspace() else c for c in bad)
    return suffix, ""


def season_entries(project) -> list[SeasonEntry]:
    """Every episode's video, merge state and imported subtitle files, read without loading documents."""
    from .workspace import DOCUMENT_FILE, EPISODE_FILE

    out = []
    for entry in project.data.get("episodes", []):
        folder = project.root / entry["path"]
        try:
            data = read_json(folder / EPISODE_FILE)
        except (OSError, ValueError):
            data = {}
        video = ((data.get("media") or {}).get("video") or {}).get("path")
        sources = tuple(Path(s["path"]) for s in data.get("sources", []) if s.get("path"))
        out.append(SeasonEntry(entry["id"], Path(video) if video else None, (folder / DOCUMENT_FILE).exists(), sources))
    return out


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))
    except OSError:
        return False


def target_name(eid: str, video: Optional[Path], suffix: str) -> str:
    stem = video.name[:-len(video.suffix)] if video is not None and video.suffix else (video.name if video else eid)
    return f"{stem}.{suffix}.ass"


def plan(entries: list[SeasonEntry], folder: Path, suffix: str, *, check_files: bool = True,
         traditional_suffix: Optional[str] = None) -> list[SeasonItem]:
    """Output names in episode order; ``check_files`` False only names them (no folder chosen yet)."""
    if traditional_suffix is not None:
        suffix, error = check_suffix(suffix)
        traditional_suffix, other_error = check_suffix(traditional_suffix)
        if error or other_error:
            raise ValueError(error or other_error)
        if suffix.casefold() == traditional_suffix.casefold():
            raise ValueError("简体后缀和繁体后缀不能相同")
        items = []
        for entry in entries:
            for ending, direction in ((suffix, "simplified"), (traditional_suffix, "traditional")):
                item = plan([entry], folder, ending, check_files=check_files)[0]
                item.direction = direction
                items.append(item)
        seen = {}
        for item in items:
            if item.target is not None:
                key = item.target.name.casefold()
                if key in seen:
                    item.duplicate_of = seen[key]
                else:
                    seen[key] = item.eid
        return items
    items, seen = [], {}
    for entry in entries:
        eid = entry.eid
        if not entry.merged:
            items.append(SeasonItem(eid, None, "还没有合并，跳过"))
            continue
        target = Path(folder) / target_name(eid, entry.video, suffix)
        if check_files and any(_same_file(target, source) for source in entry.sources):
            items.append(SeasonItem(eid, None, f"「{target.name}」是这一集的原字幕，不能覆盖，跳过"))
            continue
        item = SeasonItem(eid, target, "" if entry.video else "没有视频，用集名", check_files and target.exists())
        key = target.name.casefold()
        if key in seen:
            item.duplicate_of = seen[key]
        else:
            seen[key] = eid
        items.append(item)
    return items


def script_targets(path: Path) -> list[tuple[str, Path]]:
    """Replace a final language suffix, retaining an existing version number."""
    path = Path(path)
    stem = path.stem
    pattern = r"\.(?:zh(?:-[A-Za-z]+)?|cht|chs|ja|jpn|en|eng|ko|kor)(?P<version>\.v\d+)?$"
    match = re.search(pattern, stem, re.IGNORECASE)
    version = (match.group("version") or "") if match else ""
    if match:
        stem = stem[:match.start()]
    return [(direction, path.with_name(f"{stem}.{suffix}{version}.ass"))
            for direction, suffix in (("simplified", "zh-Hans"), ("traditional", "zh-Hant"))]
