"""Project-only reading-rule drafts, committed once with byte-exact document undo."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

from ..reading.dictionary import ReadingDict, _read_update, _write_update, normalize_word_split

TABLES = ("words", "readings", "splits", "word_splits")
KIND_NAMES = {"word": "整词读音／合词", "segments": "整词分段注音", "word_split": "固定拆词"}


def _surface(value: str) -> str:
    return "".join(value.split())


def _parts(value) -> tuple[str, ...]:
    values = re.split(r"[/／]", value) if isinstance(value, str) else value
    return tuple(str(part).strip() for part in values)


def normalize_rule_input(kind: str, surfaces, readings) -> tuple[tuple[str, str], ...]:
    if kind not in KIND_NAMES:
        raise ValueError("请选择规则类型")
    words, values = _parts(surfaces), _parts(readings)
    if len(words) != len(values) or not words or any(not x for x in (*words, *values)):
        raise ValueError("表记与读音都要填写；用 / 一一对应，不能留空段")
    if any(any(ch.isspace() for ch in x) for x in (*words, *values)):
        raise ValueError("段内不能有空白；分段请用 /，整词不分段")
    if kind == "word" and len(words) != 1:
        raise ValueError("整词读音只填一段；分段注音请明确选择对应类型")
    if kind != "word" and len(words) < 2:
        raise ValueError("分段注音或固定拆词至少需要两段")
    if kind == "word_split":
        normalize_word_split("".join(words), [{"surface": w, "reading": r} for w, r in zip(words, values)])
    return tuple(zip(words, values))


@dataclass(frozen=True)
class ProjectReadingRule:
    table: str
    key: str
    kind: str
    parts: tuple[tuple[str, str], ...]
    error: str = ""

    @property
    def surface(self) -> str:
        return _surface(self.key)

    @property
    def surface_parts(self) -> str:
        return "/".join(word for word, _reading in self.parts)

    @property
    def reading_parts(self) -> str:
        return "/".join(reading for _word, reading in self.parts)


@dataclass(frozen=True)
class ProjectReadingsSave:
    changed_surfaces: tuple[str, ...] = ()
    updated_annotations: int = 0
    backup: Path | None = None


class ProjectReadingsDraft:
    def __init__(self, project):
        self.project = project
        self.path = Path(project.dict_path("readings"))
        if not self.path.resolve().is_relative_to(Path(project.root).resolve()):
            raise ValueError("只管理本工程目录里的读音词库，不能修改全局或其他工程词库")
        self.existed = self.path.exists()
        self.original_data, self.original_bytes = _read_update(self.path)
        self.data = copy.deepcopy(self.original_data)

    @property
    def dirty(self) -> bool:
        return self.data != self.original_data

    def rows(self) -> list[ProjectReadingRule]:
        out = []
        for table in TABLES:
            for key, value in self.data.get(table, {}).items():
                kind, parts, error = "word", (), ""
                try:
                    if table in ("words", "readings"):
                        if not isinstance(value, str) or not value.strip() or not key.strip():
                            raise ValueError("旧规则表记或读音无效；未编辑时原样保留")
                        word = ReadingDict._word(key, value, "project")
                        parts = tuple(word.groups)
                        kind = "segments" if len(parts) > 1 else "word"
                    else:
                        if not isinstance(value, list) or not value:
                            raise ValueError("旧分段格式无效；未编辑时原样保留")
                        parts = tuple((p["surface"], p["reading"]) for p in value)
                        if not all(isinstance(w, str) and isinstance(r, str) and w and r for w, r in parts):
                            raise ValueError("旧分段内容无效；未编辑时原样保留")
                        if "".join(w for w, _r in parts) != key:
                            raise ValueError("旧分段不能拼回表记；未编辑时原样保留")
                        kind = "word_split" if table == "word_splits" else "segments" if len(parts) > 1 else "word"
                        if table == "word_splits":
                            normalize_word_split(key, value)
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                    parts = ((key, str(value)),)
                    kind = "word_split" if table == "word_splits" else "segments" if table == "splits" else "word"
                out.append(ProjectReadingRule(table, key, kind, parts, error))
        return out

    def remove_rule(self, rule: ProjectReadingRule) -> None:
        if rule.key not in self.data.get(rule.table, {}):
            raise ValueError("草稿里的规则已改变，请重新选择")
        del self.data[rule.table][rule.key]
        if not self.data[rule.table] and rule.table not in self.original_data:
            del self.data[rule.table]

    def set_rule(self, kind: str, surface_parts, reading_parts, *, original: ProjectReadingRule | None = None) -> ProjectReadingRule:
        parts = normalize_rule_input(kind, surface_parts, reading_parts)
        surface = "".join(word for word, _reading in parts)
        duplicates = [r for r in self.rows() if r.surface == surface]
        if original is None and duplicates:
            raise ValueError("这个表记已有工程规则，请选中它再编辑；同一表记不能同时保存不同类型")
        if original is not None:
            self.remove_rule(original)
        # Editing explicitly replaces competing types for this same surface.
        for table in TABLES:
            for key in list(self.data.get(table, {})):
                if _surface(key) == surface:
                    del self.data[table][key]
        if kind == "word_split":
            table, key = "word_splits", surface
            value = [{"surface": word, "reading": reading} for word, reading in parts]
        elif original is not None and original.kind == kind and original.table == "splits":
            table, key = "splits", surface
            value = [{"surface": word, "reading": reading} for word, reading in parts]
        else:
            table = original.table if original is not None and original.kind == kind and original.table in ("words", "readings") else "words"
            key, value = " ".join(word for word, _reading in parts), " ".join(reading for _word, reading in parts)
        self.data.setdefault(table, {})[key] = value
        return next(r for r in self.rows() if r.table == table and r.key == key)

    def changed_surfaces(self) -> tuple[str, ...]:
        changed = set()
        for table in TABLES:
            old, new = self.original_data.get(table, {}), self.data.get(table, {})
            for key in old.keys() | new.keys():
                if key not in old or key not in new or old[key] != new[key]:
                    changed.add(_surface(key))
        return tuple(sorted(changed, key=lambda text: (-len(text), text)))

    def _check_current(self) -> None:
        current = self.path.read_bytes() if self.path.exists() else b""
        if self.path.exists() != self.existed or current != self.original_bytes:
            raise RuntimeError("工程词库在窗口打开后被其他操作改过；为避免覆盖，请关闭后重新打开管理窗口")
        if not self.path.resolve().is_relative_to(Path(self.project.root).resolve()):
            raise RuntimeError("工程词库路径已改变，拒绝保存")

    def save(self, *, editor=None) -> ProjectReadingsSave:
        if not self.dirty:
            return ProjectReadingsSave()
        self._check_current()
        changed = self.changed_surfaces()
        backup = None
        if editor is not None:
            def apply(tx):
                tx.keeps_text = True
                path = editor._tx_reading_dictionary(tx, self.project)
                _write_update(path, self.data, self.original_bytes)
                engine = editor._word_split_engine(self.project)
                return sum(editor._tx_word_split_refresh(tx, surface, engine) for surface in changed)
            count = editor.history.run("管理工程词语与分词规则", apply)
        else:
            if self.existed:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                backup = self.path.with_name(self.path.name + f".before-manager-{stamp}.bak")
                with backup.open("xb") as stream:
                    stream.write(self.original_bytes)
            self._check_current()
            _write_update(self.path, self.data, self.original_bytes)
            count = 0
        self.original_data, self.original_bytes = _read_update(self.path)
        self.existed = True
        self.data = copy.deepcopy(self.original_data)
        return ProjectReadingsSave(changed, count, backup)
