"""Song memory (dict/songs.json): lyrics repeat every episode, so work done once
— corrected readings, karaoke timing, CN↔kana mapping — is reused.

    {"生まれ変わる": {"record": "生まれ変わる（うまれかわる）", "k": "{\\k41}う…", "k_cn": "", "episode": "S01E001"}}

Keys are the visible lyric text.  A stored ``k`` is rescaled to the new line's
duration when the line length differs (BD episodes cut songs identically, so
usually it is used as is).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


class SongMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, dict] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                self.data = {}
        self.dirty = False

    def get(self, lyric: str) -> Optional[dict]:
        return self.data.get(lyric.strip())

    def put(self, lyric: str, **fields) -> None:
        key = lyric.strip()
        cur = dict(self.data.get(key, {}))
        changed = False
        for k, v in fields.items():
            if v and cur.get(k) != v:
                cur[k] = v
                changed = True
        if changed:
            self.data[key] = cur
            self.dirty = True

    def save(self) -> None:
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            self.dirty = False


def rescale_k(k_text: str, new_duration: int) -> str:
    durs = [int(x) for x in re.findall(r"\\k[fo]?(\d+)", k_text)]
    total = sum(durs)
    if total <= 0 or abs(total - new_duration) <= 5:
        return k_text
    f = new_duration / total
    return re.sub(r"(\\k[fo]?)(\d+)", lambda m: f"{m.group(1)}{max(1, int(round(int(m.group(2)) * f)))}", k_text)
