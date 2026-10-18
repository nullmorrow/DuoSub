"""Project-local effect presets; tracks store resolved effects, never library references."""
from __future__ import annotations

import copy
from pathlib import Path

from ..model.serialize import read_json, write_json_atomic
from . import get_effect
from . import ktv


def validate_params(effect, params):
    get_effect(effect).resolve(ktv.effect_params(params))
    for key, spec in ktv.PARAMETERS.items():
        if key in params:
            spec.validate(params[key])
    return copy.deepcopy(params)


class PresetStore:
    def __init__(self, root):
        self.path = Path(root) / "dict" / "karaoke_fx.json"

    def list(self):
        if not self.path.exists():
            return []
        data = read_json(self.path)
        if data.get("format") != "subkit-karaoke-fx" or data.get("version") != 1:
            raise ValueError("不支持的卡拉 OK 特效预设格式")
        entries = data["presets"]
        for entry in entries:
            validate_params(entry["effect"], entry["params"])
        return copy.deepcopy(entries)

    def _write(self, entries):
        write_json_atomic(self.path, {"format": "subkit-karaoke-fx", "version": 1, "presets": entries})

    def save(self, name, effect, params):
        name = name.strip()
        if not name:
            raise ValueError("请输入预设名称")
        entries = self.list()
        if any(e["name"] == name for e in entries):
            raise ValueError("预设名称已存在")
        entries.append({"name": name, "effect": effect, "params": validate_params(effect, params)})
        self._write(entries)

    def rename(self, name, new_name):
        new_name = new_name.strip()
        entries = self.list()
        if not new_name or any(e["name"] == new_name and e["name"] != name for e in entries):
            raise ValueError("预设名称为空或已存在")
        entry = next(e for e in entries if e["name"] == name)
        entry["name"] = new_name
        self._write(entries)

    def delete(self, name):
        self._write([e for e in self.list() if e["name"] != name])
