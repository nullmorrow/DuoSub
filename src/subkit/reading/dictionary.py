"""Reading dictionary (global + project).

``dict/readings.json``::

    {
      "words":      {"幸平 創真": "ゆきひら そうま", "遠月": "とおつき", "行う": "おこなう"},
      "speakers":   {"七実": {"私": "わたくし"}},
      "prefer":     {"辛い": "からい"},
      "candidates": {"上手": ["じょうず", "うわて", "かみて"]}
    }

* ``words`` — locked readings, highest priority (beat per-line records unless the
  record entry is forced with ``!``).  Spaces split a word into separately
  placed ruby groups: "幸平 創真" → 幸平(ゆきひら) 創真(そうま).
* ``speakers`` — locked readings that only apply to lines of that speaker
  (speaker labels come from the Japanese captions): the わたし/わたくし problem.
* ``prefer`` — project default for an ambiguous word (a cooking show reads 辛い
  as からい); still a candidate list, so context/audio can disagree.
* ``candidates`` — extra heteronym candidates the resolver should consider.
* ``typed`` — project input-box history, used only by the reading popup.
* word_splits — explicit Project word boundaries and readings; each part is a separate entry.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def normalize_word_split(surface: str, parts: list[dict]) -> list[dict[str, str]]:
    """Validate exact word boundaries; spaces in a reading never create words."""
    if not isinstance(surface, str) or not surface or not surface.strip():
        raise ValueError("拆词表记不能为空")
    if not isinstance(parts, (list, tuple)) or len(parts) < 2:
        raise ValueError("至少拆成两个词")
    out = []
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("拆词的每一段必须包含表记和读音")
        word, reading = part.get("surface"), part.get("reading")
        if not isinstance(word, str) or not word or word != word.strip():
            raise ValueError("每个词的表记不能为空，也不能在词的两端留空格")
        if not isinstance(reading, str) or not reading.strip():
            raise ValueError("每个词都要填写读音；不会自动猜测空读音")
        if any(ch in word + reading for ch in "\r\n\t"):
            raise ValueError("词和读音不能包含换行或制表符")
        out.append({"surface": word, "reading": reading.strip()})
    if "".join(part["surface"] for part in out) != surface:
        raise ValueError("分段表记必须完整拼回选中文字，不能丢字、改字或增加字")
    return out


def write_dictionary_bytes(path: Path, data: bytes) -> None:
    """Atomic replacement, also used by the document's external-file undo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(data)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _read_update(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes() if path.exists() else b""
    data = json.loads(raw.decode("utf-8-sig")) if raw else {}
    if not isinstance(data, dict):
        raise ValueError("工程读音词库必须是 JSON 对象")
    for key in ("words", "readings", "splits", "word_splits"):
        if key in data and not isinstance(data[key], dict):
            raise ValueError(f"{key} must be a JSON object")
    return data, raw


def _write_update(path: Path, data: dict, original: bytes) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if b"\r\n" in original:
        text = text.replace("\n", "\r\n")
    encoding = "utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8"
    write_dictionary_bytes(path, text.encode(encoding))


def _remove_surface(data: dict, tables: tuple[str, ...], surface: str) -> None:
    key = "".join(surface.split())
    for name in tables:
        table = data.get(name, {})
        for old in list(table):
            if "".join(old.split()) == key:
                del table[old]


@dataclass
class DictWord:
    surface: str                 # without spaces
    reading: str                 # without spaces
    groups: list[tuple[str, str]]  # [(surface part, reading part)]
    origin: str = "project"
    separate_words: bool = False       # legacy groups remain one Entry


@dataclass
class ReadingDict:
    words: dict[str, DictWord] = field(default_factory=dict)
    speakers: dict[str, dict[str, DictWord]] = field(default_factory=dict)
    prefer: dict[str, str] = field(default_factory=dict)
    candidates: dict[str, list[str]] = field(default_factory=dict)
    sources: list[Path] = field(default_factory=list)
    characters: object = None
    character_book: object = None
    typed: dict[str, list[str]] = field(default_factory=dict)
    word_splits: dict[str, DictWord] = field(default_factory=dict)

    @staticmethod
    def _word(key: str, value: str, origin: str) -> DictWord:
        sk = key.split()
        rv = value.split()
        surface = "".join(sk)
        reading = "".join(rv)
        if len(sk) == len(rv) and len(sk) > 1:
            groups = list(zip(sk, rv))
        else:
            groups = [(surface, reading)]
        return DictWord(surface, reading, groups, origin)

    def merge_json(self, data: dict, origin: str) -> None:
        for k, v in (data.get("words") or {}).items():
            if isinstance(v, str) and k.strip() and v.strip():
                w = self._word(k, v, origin)
                self.words[w.surface] = w
                self.word_splits.pop(w.surface, None)
        # legacy furigana_phrases.json
        for k, v in (data.get("readings") or {}).items():
            if isinstance(v, str) and k.strip() and v.strip():
                w = self._word(k, v, origin)
                self.words[w.surface] = w
                self.word_splits.pop(w.surface, None)
        for k, v in (data.get("splits") or {}).items():
            if isinstance(v, list):
                parts = [(x.get("surface", ""), x.get("reading", "")) for x in v if isinstance(x, dict)]
                parts = [(s, r) for s, r in parts if s and r]
                if parts and "".join(s for s, _ in parts) == k:
                    self.words[k] = DictWord(k, "".join(r for _, r in parts), parts, origin)
                    self.word_splits.pop(k, None)
        split_table = data.get("word_splits", {})
        if not isinstance(split_table, dict):
            raise ValueError("word_splits must be a JSON object")
        for surface, value in split_table.items():
            parts = normalize_word_split(surface, value)
            self.word_splits[surface] = DictWord(
                surface, "".join(p["reading"] for p in parts),
                [(p["surface"], p["reading"]) for p in parts], origin, separate_words=True)
            self.words.pop(surface, None)
        for spk, table in (data.get("speakers") or {}).items():
            if isinstance(table, dict):
                dst = self.speakers.setdefault(spk, {})
                for k, v in table.items():
                    if isinstance(v, str):
                        w = self._word(k, v, origin)
                        dst[w.surface] = w
        for k, v in (data.get("prefer") or {}).items():
            if isinstance(v, str):
                self.prefer[k] = v
        for k, v in (data.get("candidates") or {}).items():
            if isinstance(v, list):
                self.candidates[k] = [x for x in v if isinstance(x, str) and x]
            elif isinstance(v, str):
                self.candidates[k] = [v]

        if origin == "project" and isinstance(data.get("typed"), dict):
            for surface, readings in data["typed"].items():
                if isinstance(readings, list):
                    self.typed[surface] = list(dict.fromkeys(r for r in readings if isinstance(r, str) and r))[:10]

    def typed_readings(self, surface: str) -> list[str]:
        """Recent project input-box readings; not automatic reading candidates."""
        return list(self.typed.get(surface, []))

    def add_typed(self, path: Path, surface: str, reading: str) -> None:
        """Update input history in the caller's PROJECT readings.json, preserving other values."""
        if not surface or not reading:
            return
        data = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
        table = data.setdefault("typed", {})
        if not isinstance(table, dict):
            raise ValueError("typed must be a JSON object")
        previous = table.get(surface, [])
        if not isinstance(previous, list):
            previous = []
        readings = list(dict.fromkeys([reading] + [r for r in previous if isinstance(r, str) and r]))[:10]
        table[surface] = readings
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.typed[surface] = readings

    @classmethod
    def load(cls, *paths: Optional[Path]) -> "ReadingDict":
        """Later paths override earlier ones (pass global first, project last)."""
        d = cls()
        for p in paths:
            if p is None or not Path(p).exists():
                continue
            data = json.loads(Path(p).read_text(encoding="utf-8-sig"))
            d.merge_json(data, "global" if "global" in str(p).lower() else "project")
            d.sources.append(Path(p))
        return d

    def merge_characters(self, table) -> None:
        """Character table: names become locked words; 私的读法 becomes a speaker rule."""
        self._merge_name_words(table.words())
        for ch in table.chars:
            if ch.self_reading:
                self.speakers.setdefault(ch.name, {})["私"] = DictWord("私", ch.self_reading, [("私", ch.self_reading)], "characters")
        self.characters = table

    def merge_character_book(self, book) -> None:
        self._merge_name_words(book.words())
        self.character_book = book

    def _merge_name_words(self, words) -> None:
        for key, reading in words:
            surface = key.replace(" ", "")
            if len(surface) == 1:
                # one-kanji names (司, 恵, 薊) would break 寿司 / 上司 / 恵まれる as surface
                # matches: they apply only where the analyzer sees the name as a word
                self.prefer.setdefault(surface, reading.replace(" ", ""))
                continue
            w = self._word(key, reading, "characters")
            self.words.setdefault(w.surface, w)

    def match(self, text: str, speaker: Optional[str] = None, *, character: Optional[str] = None,
              line: Optional[str] = None, episode: Optional[str] = None,
              season: Optional[str] = None) -> list[tuple[int, int, DictWord]]:
        """Longest-first, left-to-right, non-overlapping occurrences."""
        table: dict[str, DictWord] = {**self.words, **self.word_splits}
        if self.character_book is not None:
            ch = next((c for c in self.character_book.characters if c.id == character), None)
            if ch is not None:
                for label in [*(ch.full_name(lang) for lang in ch.names), *ch.aliases]:
                    table.update(self.speakers.get(label, {}))
                selected = {}
                for ref in ch.self_refs:
                    scope = ref.get("scope") or {}
                    if "line" in scope:
                        priority, applies = 3, scope["line"] == line
                    elif "episodes" in scope:
                        priority, applies = 2, episode in scope["episodes"]
                    elif "season" in scope:
                        priority, applies = 1, scope["season"] == season
                    else:
                        priority, applies = 0, not scope
                    surface, reading = ref.get("surface", ""), ref.get("reading", "")
                    if applies and surface and reading and priority >= selected.get(surface, (-1,))[0]:
                        selected[surface] = (priority, self._word(surface, reading, "characters"))
                table.update({surface: word for surface, (_priority, word) in selected.items()})
        elif speaker:
            for name, sub in self.speakers.items():
                if name == speaker or (speaker and (speaker.endswith(name) or name.endswith(speaker))):
                    table.update(sub)
        if not table:
            return []
        keys = sorted(table, key=len, reverse=True)
        out: list[tuple[int, int, DictWord]] = []
        i = 0
        n = len(text)
        while i < n:
            for k in keys:
                if text.startswith(k, i):
                    out.append((i, i + len(k), table[k]))
                    i += len(k)
                    break
            else:
                i += 1
        return out

    @staticmethod
    def remove_word(path: Path, surface: str) -> bool:
        """Remove a locked word from a dictionary file (keys may contain group spaces).  Returns True if found."""
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        words = data.get("words") or {}
        keys = [k for k in words if k.replace(" ", "") == surface]
        for k in keys:
            del words[k]
        if keys:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return bool(keys)

    def add_word(self, path: Path, key: str, value: str) -> None:
        data, original = _read_update(path)
        data.setdefault("version", 2)
        _remove_surface(data, ("words", "readings", "splits", "word_splits"), key)
        data.setdefault("words", {})[key] = value
        _write_update(path, data, original)
        w = self._word(key, value, "project")
        self.word_splits.pop(w.surface, None)
        self.words[w.surface] = w

    def set_word_split(self, path: Path, surface: str, parts: list[dict]) -> None:
        parts = normalize_word_split(surface, parts)
        data, original = _read_update(path)
        data.setdefault("version", 2)
        _remove_surface(data, ("words", "readings", "splits", "word_splits"), surface)
        data.setdefault("word_splits", {})[surface] = parts
        _write_update(path, data, original)
        self.words.pop(surface, None)
        self.word_splits[surface] = DictWord(
            surface, "".join(p["reading"] for p in parts),
            [(p["surface"], p["reading"]) for p in parts], "project", separate_words=True)

    def remove_word_split(self, path: Path, surface: str) -> bool:
        if not path.exists():
            return False
        data, original = _read_update(path)
        table = data.get("word_splits", {})
        if surface not in table:
            return False
        del table[surface]
        _write_update(path, data, original)
        self.word_splits.pop(surface, None)
        return True


TEMPLATE = {
    "version": 2,
    "words": {},
    "speakers": {},
    "prefer": {},
    "candidates": {},
}
