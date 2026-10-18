"""Where each episode's video is now (集 → 视频文件位置…).

Videos and imported subtitles may be moved away after an episode is made; the project keeps its own
subtitle copies, but the video is only referenced.  When it is missing, a folder can be searched: first a
file with the same name, then one with the recorded size (a moved file keeps it even when renamed), then
one whose name carries the episode number.  One file serves one episode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..model.serialize import read_json, write_json_atomic

VIDEO_SUFFIXES = {".mkv", ".mp4", ".m4v", ".avi", ".m2ts", ".ts", ".webm", ".mov", ".wmv", ".flv"}
RULES = {"name": "同名", "size": "文件大小相同", "number": "集号"}


@dataclass
class VideoEntry:
    eid: str
    folder: Path                     # the episode folder
    number: Optional[int]
    recorded: Optional[Path]
    size: Optional[int]
    track: Optional[int]

    @property
    def missing(self) -> bool:
        return self.recorded is not None and not self.recorded.is_file()


def video_entries(project) -> list[VideoEntry]:
    from .workspace import EPISODE_FILE

    out = []
    for entry in project.data.get("episodes", []):
        folder = project.root / entry["path"]
        try:
            media = read_json(folder / EPISODE_FILE).get("media") or {}
        except (OSError, ValueError):
            media = {}
        video = media.get("video") or {}
        number = entry.get("number")
        if not isinstance(number, int):
            found = re.search(r"(\d+)$", entry["id"])
            number = int(found.group(1)) if found else None
        out.append(VideoEntry(entry["id"], folder, number, Path(video["path"]) if video.get("path") else None,
                              video.get("size"), media.get("audio_track")))
    return out


def _number_pattern(n: int) -> re.Pattern:
    # E05 / EP05 / S01E005, or 05 standing alone (" - 05 ", "[05]", "第05話"); not 1080 or 10bit.
    return re.compile(rf"(?:[Ee][Pp]?0*{n}(?!\d))|(?:(?:^|[\s\[\(_#第\-])0*{n}(?=$|[\s\]\)_\-話话集vV.]))")


def match_folder(entries: list[VideoEntry], folder: Path) -> dict[str, tuple[Path, str]]:
    """episode id → (video file under ``folder``, rule) for the given entries, one file per episode."""
    try:
        files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file()]
    except OSError:
        files = []
    taken: set[Path] = set()
    found: dict[str, tuple[Path, str]] = {}

    def claim(entry, path, rule):
        found[entry.eid] = (path, rule)
        taken.add(path)

    by_name = {}
    for p in files:
        by_name.setdefault(p.name.casefold(), []).append(p)
    for e in entries:
        same = [p for p in by_name.get(e.recorded.name.casefold(), []) if p not in taken] if e.recorded else []
        if len(same) == 1:
            claim(e, same[0], "name")
    sizes = {}
    for p in files:
        try:
            sizes.setdefault(p.stat().st_size, []).append(p)
        except OSError:
            continue
    for e in entries:
        if e.eid in found or not e.size:
            continue
        same = [p for p in sizes.get(e.size, []) if p not in taken]
        if len(same) == 1:
            claim(e, same[0], "size")
    for e in entries:
        if e.eid in found or e.number is None:
            continue
        pattern = _number_pattern(e.number)
        same = [p for p in files if p not in taken and pattern.search(p.stem)]
        if len(same) == 1:
            claim(e, same[0], "number")
    return found


def relink(entry: VideoEntry, video: Path, cache_dir: Optional[Path] = None) -> None:
    """Record ``video`` for the episode (episode.json only; the audio track stays) and carry over the
    waveform and dialogue-alignment caches made under the old path."""
    from .workspace import EPISODE_FILE

    video = Path(video)
    st = video.stat()
    path = entry.folder / EPISODE_FILE
    data = read_json(path)
    media = data.setdefault("media", {})
    media["video"] = {"path": str(video).replace("\\", "/"), "size": st.st_size, "mtime": int(st.st_mtime)}
    write_json_atomic(path, data)
    if cache_dir is not None and entry.recorded is not None:
        carry_caches(Path(cache_dir), entry.recorded, video, entry.track)
    entry.recorded, entry.size = video, st.st_size


def carry_caches(cache_dir: Path, old: Path, video: Path, track: Optional[int]) -> None:
    """Rename caches written under the video's old path to their path-independent names (best effort)."""
    from ..audio import adopt_legacy, ffmpeg_audio_index, legacy_waveform_cache_file, waveform_cache_file
    from ..kalign import cache as kcache

    try:
        adopt_legacy(legacy_waveform_cache_file(video, track, cache_dir / "waveform", recorded=old),
                     waveform_cache_file(video, track, cache_dir / "waveform"))
        kcache.cache_directory(cache_dir / "kalign", video, ffmpeg_audio_index(track), None, recorded=old)
    except (OSError, ImportError):
        pass
