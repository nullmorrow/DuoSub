"""The project's own copies of the imported subtitle files: find missing / changed ones and put them back.

A project never reads subtitles from where they were imported from: ``sources/<id>.<ext>`` in the episode
folder is a byte-identical, read-only copy (recorded SHA-256).  When that copy is missing or its bytes
changed, only a byte-identical file can replace it — from the original path when it is still there, or
any file with the same content found in a folder the user chooses (names may differ).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..model.serialize import read_json
from .importer import copy_state, restore_copy, sha256_bytes

SUBTITLE_SUFFIXES = {".ass", ".ssa", ".srt", ".vtt", ".txt", ".sub", ".lrc"}


@dataclass
class CopyProblem:
    eid: str
    sid: str
    lane: str
    copy: Path                         # where the copy belongs
    sha: str
    size: Optional[int]
    original: Path                     # where it was imported from (may be gone)
    state: str                         # missing | changed
    fixed_from: Optional[Path] = None


def scan_episode(eid: str, folder: Path) -> list[CopyProblem]:
    from .workspace import EPISODE_FILE

    out = []
    for s in read_json(Path(folder) / EPISODE_FILE).get("sources", []):
        if not s.get("copy") or not s.get("sha256"):
            continue
        copy = Path(folder) / s["copy"]
        state = copy_state(copy, s["sha256"])
        if state != "ok":
            out.append(CopyProblem(eid, s["id"], s.get("lane", ""), copy, s["sha256"], s.get("bytes"),
                                   Path(s.get("path") or ""), state))
    return out


def scan_project(project) -> list[CopyProblem]:
    """Every episode's copy problems, read from episode.json (documents are not loaded)."""
    out = []
    for entry in project.data.get("episodes", []):
        try:
            out.extend(scan_episode(entry["id"], project.root / entry["path"]))
        except (OSError, ValueError):
            continue
    return out


def fix_from_original(problem: CopyProblem) -> bool:
    """Restore from the import path when that file still has the imported bytes."""
    try:
        if problem.original.name and problem.original.is_file() and \
                sha256_bytes(problem.original.read_bytes()) == problem.sha:
            restore_copy(problem.copy, problem.sha, problem.original)
            problem.fixed_from = problem.original
            return True
    except OSError:
        pass
    return False


def find_in_folder(folder: Path, problems: list[CopyProblem], *,
                   cancel_check: Callable[[], bool] = lambda: False) -> dict[int, Path]:
    """index in ``problems`` → a file under ``folder`` (subfolders included) with the same bytes.
    Only subtitle-like files of a recorded size are hashed."""
    wanted: dict[str, list[int]] = {}
    sizes = set()
    for i, p in enumerate(problems):
        wanted.setdefault(p.sha, []).append(i)
        sizes.add(p.size)
    found: dict[int, Path] = {}
    for path in Path(folder).rglob("*"):
        if cancel_check():
            break
        try:
            if path.suffix.lower() not in SUBTITLE_SUFFIXES or not path.is_file():
                continue
            if None not in sizes and path.stat().st_size not in sizes:
                continue
            sha = sha256_bytes(path.read_bytes())
        except OSError:
            continue
        for i in wanted.get(sha, ()):
            found.setdefault(i, path)
        if len(found) == len(problems):
            break
    return found


def fix_from_file(problem: CopyProblem, candidate: Path) -> None:
    """Raises SourceChanged (with the reason in Chinese) when the bytes differ."""
    restore_copy(problem.copy, problem.sha, candidate)
    problem.fixed_from = Path(candidate)
