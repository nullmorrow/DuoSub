"""Project root, episode ids and source resolution."""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config

_EP = re.compile(r"(?:S(\d+))?\s*E(\d+)|^(\d+)x(\d+)$|^(\d+)$", re.IGNORECASE)


def parse_episode(spec: str) -> tuple[int, int]:
    """'S01E001', 's1e1', '1x01', '5' (season 1) → (season, episode)."""
    s = spec.strip()
    m = re.search(r"S(\d+)\s*E(\d+)", s, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d+)x(\d+)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^E?(\d+)$", s, re.IGNORECASE)
    if m:
        return 1, int(m.group(1))
    raise ValueError(f"无法识别的集数：{spec}（例：S01E001、1x01、5）")


def episode_id(season: int, episode: int) -> str:
    return f"S{season:02d}E{episode:03d}"


def find_root(start: Optional[Path] = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "subproj.toml").exists():
            return cand
    raise FileNotFoundError("找不到 subproj.toml：请在工程目录里运行，或先执行 subkit init")


@dataclass
class Episode:
    season: int
    number: int
    project: "Project"

    @property
    def id(self) -> str:
        return episode_id(self.season, self.number)

    @property
    def work_dir(self) -> Path:
        return self.project.config.work_dir / self.id

    @property
    def work_ass(self) -> Path:
        return self.work_dir / "work.ass"

    @property
    def state_path(self) -> Path:
        return self.work_dir / "state.json"

    def _fmt(self, pattern: str, extra: dict) -> str:
        return pattern.format(S=self.season, E=self.number, **extra)

    def _find(self, patterns, extra: dict) -> Optional[Path]:
        if isinstance(patterns, str):
            patterns = [patterns]
        for pat in patterns:
            text = self._fmt(pat, extra)
            path = Path(text)
            if not path.is_absolute():
                path = self.project.root / path
            hits = sorted(glob.glob(_unescape_wild(str(path))))
            if hits:
                return Path(hits[0])
        return None

    def video(self) -> Optional[Path]:
        return self._find(self.project.config["episodes"]["video"], {})

    def sources(self) -> dict[str, Optional[Path]]:
        video = self.video()
        extra = {"video_dir": str(video.parent) if video else "", "video_stem": video.stem if video else ""}
        cn = self._find(self.project.config["episodes"]["cn"], extra) if video else None
        jp = self._find(self.project.config["episodes"]["jp"], extra)
        return {"video": video, "cn": cn, "jp": jp}

    def release_path(self, cn_lang: str) -> Optional[Path]:
        video = self.video()
        if not video:
            return None
        extra = {"video_dir": str(video.parent), "video_stem": video.stem, "cn_lang": cn_lang}
        p = Path(self._fmt(self.project.config["episodes"]["release"], extra))
        return p if p.is_absolute() else self.project.root / p

    def load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"version": 1, "episode": self.id, "next_id": 1, "rows": {}}

    def save_state(self, state: dict) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)


def _unescape_wild(path: str) -> str:
    """Escape glob metacharacters except our own '*' wildcards ([VCB-Studio] contains '[')."""
    out = []
    for ch in path:
        if ch == "[":
            out.append("[[]")
        elif ch == "]":
            out.append("[]]")
        else:
            out.append(ch)
    return "".join(out)


class Project:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = Config.load(root)

    @classmethod
    def open(cls, start: Optional[Path] = None) -> "Project":
        return cls(find_root(start))

    def episode(self, spec: str) -> Episode:
        s, e = parse_episode(spec)
        return Episode(s, e, self)

    @property
    def cache_dir(self) -> Path:
        return self.root / ".cache"
