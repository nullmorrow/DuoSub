"""Project configuration (subproj.toml) with documented defaults.

Only ``tomllib`` (stdlib) is used for reading; the template is written as text
so comments survive.
"""

from __future__ import annotations

import copy
import os
import re
import tomllib
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "project": {
        "name": "",
        "playres": [1920, 1080],
        "styles": "styles.ass",
        "dict": "dict/readings.json",
        "work_dir": "work",
    },
    "tools": {
        "aegisub": "C:/Program Files/Aegisub/aegisub.exe",
        "ffmpeg": "ffmpeg",
        "ffprobe": "ffprobe",
        "fonts_dirs": ["%LOCALAPPDATA%/Microsoft/Windows/Fonts", "%WINDIR%/Fonts"],
        "global_dict": "%APPDATA%/subkit/global_readings.json",
    },
    "episodes": {
        # {S} season, {E} episode; candidates are tried in order, globs allowed
        "video": ["Season {S:02}/S{S:02}E{E:03} - *.mkv"],
        "cn": ["{video_dir}/{video_stem}.tc.ass", "{video_dir}/{video_stem}.ass"],
        "jp": ["sources/jp/S{S:02}E{E:02}.ja.srt"],
        "release": "{video_dir}/{video_stem}.{cn_lang}.ass",
    },
    "cn": {
        "dialogue": [],          # empty = auto (most frequent plain style)
        "top": [],
        "song_jp": [],
        "song_cn": [],
        "drop": [],
        "linebreak": "  ",       # CN \N inside a dialogue line becomes this
        "space": "  ",           # runs of spaces become this
    },
    "jp": {
        "drop_sfx": True,         # whole-line (效果音) descriptions
        "strip_speaker": True,    # leading (说话人) labels
        "strip_period": True,     # sentence-final 。
        "space": "  ",            # separator between phrases in a JP line
        "halfwidth_alnum": False, # ３人 → 3人
    },
    "align": {
        "embedding": "auto",      # auto | none | labse | bge-m3 | ollama:<model>
        "max_shift_rows": 2,      # how far the semantic step may move a fragment
        "allow_swaps": True,
    },
    "reading": {
        "analyzer": "sudachi",
        "sudachi_dict": "full",
        "audio": "off",           # off | whisper  (verify ambiguous readings against the voice track)
        "whisper_model": "medium",
    },
    "ruby": {
        "gap": 2.0,               # px between base ink top and ruby ink bottom
        "min_scale": 60,          # never compress ruby below this \fscx
        "overhang": 0.5,          # ruby may overhang neighbouring kana by this many ruby-char widths
        "mono_ruby": True,        # split compound readings per kanji when it fits (熟語ルビ)
    },
    "release": {
        "blur": 0.0,              # 0 = no extra \blur
        "blur_styles": ["CN*", "JP*", "Song*"],
        "keep_comments": False,
    },
    "pinyin": {"enabled": False, "mode": "PY"},
}


_SEASON_KEY = re.compile(r"^S\d+$")


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def expand_path(value: str) -> str:
    value = os.path.expandvars(value)
    # %VAR% on Windows when expandvars did not catch it
    for key, val in os.environ.items():
        value = value.replace(f"%{key}%", val)
    return os.path.expanduser(value)


class Config:
    def __init__(self, data: dict[str, Any], root: Path) -> None:
        self.data = data
        self.root = root

    @classmethod
    def load(cls, root: Path) -> "Config":
        path = root / "subproj.toml"
        user = {}
        if path.exists():
            with open(path, "rb") as fh:
                user = tomllib.load(fh)
        return cls(_merge(DEFAULTS, user), root)

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.data[section]

    def section(self, name: str, season: int | None = None) -> dict[str, Any]:
        """``[name]`` with the per-season table ``[name.S02]`` merged in (a later season
        subbed by another group uses other style names)."""
        data = self.data.get(name, {})
        out = {k: v for k, v in data.items() if not (isinstance(v, dict) and _SEASON_KEY.match(k))}
        if season is not None:
            for key in (f"S{season:02d}", f"S{season}"):
                if isinstance(data.get(key), dict):
                    out = _merge(out, data[key])
                    break
        return out

    def path(self, value: str) -> Path:
        p = Path(expand_path(value))
        return p if p.is_absolute() else self.root / p

    @property
    def playres(self) -> tuple[int, int]:
        x, y = self.data["project"]["playres"]
        return int(x), int(y)

    @property
    def work_dir(self) -> Path:
        return self.path(self.data["project"]["work_dir"])


TEMPLATE = """# subkit 工程配置。没写的项目使用默认值（见 _tool/subkit/config.py 的 DEFAULTS）。

[project]
name = "{name}"
playres = [1920, 1080]          # 工作文件统一缩放到这个分辨率（与视频一致最好）
styles = "styles.ass"           # 样式方案
dict = "dict/readings.json"     # 番组词库（人名、术语；优先级最高）

[tools]
aegisub = "C:/Program Files/Aegisub/aegisub.exe"

[episodes]
# {{S}} 季，{{E}} 集；按顺序尝试，支持通配符
video = ["Season {{S:02}}/S{{S:02}}E{{E:03}} - *.mkv"]
cn = ["{{video_dir}}/{{video_stem}}.tc.ass", "{{video_dir}}/{{video_stem}}.ass"]
jp = ["sources/jp/S{{S:02}}E{{E:02}}.ja.srt"]
release = "{{video_dir}}/{{video_stem}}.{{cn_lang}}.ass"

[cn]
dialogue = []      # 中文对白 Style；留空 = 自动（出现最多的普通 Style）
top = []           # 顶部对白 Style
song_jp = []       # 原字幕里的日文歌词 Style
song_cn = []       # 原字幕里的中文歌词 Style

[jp]
drop_sfx = true        # 删除整行（效果音）描述
strip_speaker = true   # 删除行首（说话人）
strip_period = true    # 删除句末「。」

[ruby]
gap = 2.0
min_scale = 60
"""
