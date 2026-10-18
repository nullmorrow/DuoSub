"""User-level punctuation profiles; no Qt or project-file dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .display import MERGE_KINDS, PUNCTUATION_CATEGORIES, PunctuationRules
from ..model.serialize import write_json_atomic

FORMAT = "duosub-punctuation-profile"
LANES = ("translation", "original")


class ProfileError(ValueError):
    """Invalid profile contents or name."""


class ProfileConflictError(ProfileError):
    """A profile name already exists (the UI may offer overwrite/rename)."""


def _name(name: str) -> str:
    # Names are filenames on Windows, including when imported from another OS.
    if (not isinstance(name, str) or not name.strip() or name != name.strip()
            or name.endswith(".") or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name in (".", "..")
            or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", name)):
        raise ProfileError("方案名称无效：请使用可作为文件名的非空名称。")
    return name


def _lane(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ProfileError("方案规则必须是 JSON 对象。")
    choices = {
        "comma": ("off", "spaces", "ideographic", True, False),
        "ellipsis": ("off", "single", "double", "keep_count"),
        "exclaim": ("off", "full", "half"), "question": ("off", "full", "half"),
        "quotes": ("off", "corner", "white_corner", "nested", "nested_inverse", True, False),
        "tilde": ("off", "full", "half", "wave"),
        "dash": ("off", "half", "em", "choon", "bar", "fullwidth_hyphen"),
    }
    for key, value in data.items():
        valid = False
        if key in choices:
            valid = any(type(value) is type(option) and value == option for option in choices[key])
        elif key in ("dunhao", "period", "spaces", "separator", "halfwidth_kana"):
            valid = isinstance(value, bool)
        elif key == "font":
            valid = value is None or isinstance(value, str)
        elif key == "fonts":
            valid = isinstance(value, dict) and all(
                category in PUNCTUATION_CATEGORIES and (font is None or isinstance(font, str))
                for category, font in value.items())
        elif key in ("scope_chars", "scope_fonts"):
            valid = isinstance(value, (list, tuple)) and all(
                kind in tuple(k for k, _ in MERGE_KINDS) for kind in value)
        if not valid:
            raise ProfileError(f"方案规则无效：{key}。")
    return PunctuationRules.from_settings(data).to_settings()


def validate_profile(profile: dict) -> dict:
    """Validate external data and return a detached canonical profile."""
    if not isinstance(profile, dict) or profile.get("format") != FORMAT:
        raise ProfileError("不是标点方案文件（duosub-punctuation-profile）。")
    if type(profile.get("version")) is not int or profile["version"] != 1:
        raise ProfileError("不支持的标点方案版本（需要版本 1）。")
    name = _name(profile.get("name"))
    if any(lane not in profile for lane in LANES):
        raise ProfileError("方案缺少翻译或原声规则。")
    return {"format": FORMAT, "version": 1, "name": name,
            **{lane: _lane(profile[lane]) for lane in LANES}}


def create_profile(name: str, translation: PunctuationRules, original: PunctuationRules) -> dict:
    return validate_profile({"format": FORMAT, "version": 1, "name": name,
                             "translation": translation.to_settings(), "original": original.to_settings()})


def matches(settings: dict, profile: dict) -> bool:
    """Compare project settings through legacy migration, independent of profile name."""
    profile = validate_profile(profile)
    punctuation = settings.get("display", {}).get("punctuation", {})
    return all(PunctuationRules.from_settings(punctuation.get(lane, {})).to_settings() == profile[lane]
               for lane in LANES)


def apply(settings: dict, profile: dict) -> None:
    """Replace both lanes, retaining unrelated settings and a name hint only."""
    profile = validate_profile(profile)
    display = settings.setdefault("display", {})
    display["punctuation"] = {lane: profile[lane] for lane in LANES}
    display["punctuation_profile"] = profile["name"]


class PunctuationProfileStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, name: str) -> Path:
        name = _name(name)
        # Windows names conflict regardless of case; keep this behaviour in portable tests too.
        for path in self.directory.glob("*.json"):
            if path.stem.casefold() == name.casefold():
                return path
        return self.directory / (name + ".json")

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            return validate_profile(json.loads(path.read_text(encoding="utf-8-sig")))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ProfileError(f"无法读取标点方案 JSON：{path.name}（{exc}）") from exc

    def list(self) -> list[dict]:
        profiles = [self.load(path.stem) for path in self.directory.glob("*.json")]
        return sorted(profiles, key=lambda profile: profile["name"])

    def load(self, name: str) -> dict:
        profile = self._read(self._path(name))
        if profile["name"].casefold() != name.casefold():
            raise ProfileError("方案名称与文件名不一致。")
        return profile

    def save(self, profile: dict, *, overwrite: bool = False) -> dict:
        profile = validate_profile(profile)
        path = self._path(profile["name"])
        if path.exists() and not overwrite:
            raise ProfileConflictError(f"已有同名方案：{profile['name']}")
        self.directory.mkdir(parents=True, exist_ok=True)
        # Preserve the existing spelling on a case-insensitive overwrite.
        profile["name"] = path.stem
        write_json_atomic(path, profile)
        return profile

    def delete(self, name: str) -> None:
        self._path(name).unlink()

    def rename(self, name: str, new_name: str) -> dict:
        old_path = self._path(name)
        profile = self.load(name)
        new_path = self._path(new_name)
        if new_path.exists() and new_path != old_path:
            raise ProfileConflictError(f"已有同名方案：{new_name}")
        profile["name"] = _name(new_name)
        if new_path == old_path:
            new_path = self.directory / (new_name + ".json")
            write_json_atomic(old_path, profile)
            if new_path.name != old_path.name:
                old_path.rename(new_path)
        else:
            write_json_atomic(new_path, profile)
            old_path.unlink()
        return profile

    def export_to(self, name: str, path: Path) -> None:
        write_json_atomic(Path(path), self.load(name))

    def import_from(self, path: Path, *, overwrite: bool = False, name: str | None = None) -> dict:
        profile = self._read(Path(path))
        if name is not None:
            profile["name"] = _name(name)
        return self.save(profile, overwrite=overwrite)

    def apply_default(self, settings: dict, name: str | None) -> bool:
        if not name or not self._path(name).exists():
            return False
        apply(settings, self.load(name))
        return True
