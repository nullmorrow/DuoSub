"""Project character identities and exact subtitle-label linking (Qt-free)."""

from __future__ import annotations

import copy
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..model.ids import new_id
from ..model.serialize import read_json, write_json_atomic
from ..languages import get_language, normalize_language_code
from ..reading.characters import CharacterTable
from .cc import normalize_speaker

PERSONA_FIELDS = ("self", "address", "catchphrases", "tone", "note_ja", "note_tr")


def suggest_character(book: "CharacterBook", label: str) -> Optional[str]:
    """Suggest one identity from exact name parts/aliases, without linking it."""
    key = normalize_speaker(label)
    if not key:
        return None
    matches = set()
    for ch in book.characters:
        candidates = list(ch.aliases)
        for form in ch.names.values():
            candidates.extend((form.family, form.given))
            full = form.compose()
            if re.search(r"[・·]", full):
                candidates.extend(re.split(r"[・·]", full))
        if any(value and normalize_speaker(value) == key for value in candidates):
            matches.add(ch.id)
    return next(iter(matches)) if len(matches) == 1 else None


@dataclass
class NameForm:
    family: str = ""
    given: str = ""
    family_reading: str = ""
    given_reading: str = ""
    order: Optional[str] = None
    separator: Optional[str] = None
    short: str = "full"
    short_custom: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def with_defaults(self, lang: str) -> "NameForm":
        language = get_language(lang)
        if self.order is None:
            self.order = language.name_order
        if self.separator is None:
            self.separator = language.name_separator
        if self.order not in ("family_first", "given_first") or self.short not in ("full", "family", "given", "custom"):
            raise ValueError("Invalid name order or short form")
        if language.annotation_system == "pinyin":
            from .pinyin_notes import normalize_rule

            for part in ("family", "given"):
                reading = getattr(self, part + "_reading")
                if reading:
                    setattr(self, part + "_reading", normalize_rule(getattr(self, part), reading)[1])
        return self

    def parts(self, *, readings: bool = False) -> list[str]:
        values = (self.family_reading, self.given_reading) if readings else (self.family, self.given)
        return [x for x in (values if self.order == "family_first" else values[::-1]) if x]

    def compose(self, format: str = "full") -> str:
        if format == "short":
            return self.short_custom if self.short == "custom" else self.compose(self.short)
        if format in ("family", "given"):
            return getattr(self, format)
        if format != "full":
            raise ValueError(format)
        return (self.separator or "").join(self.parts())

    def to_json(self) -> dict:
        data = asdict(self)
        data.update(data.pop("extra"))
        return data

    @classmethod
    def from_json(cls, lang: str, data: dict) -> "NameForm":
        fields = cls.__dataclass_fields__.keys() - {"extra"}
        return cls(**{k: v for k, v in data.items() if k in fields},
                   extra={k: v for k, v in data.items() if k not in fields}).with_defaults(lang)


def _name_forms(names: dict, parts=(), readings=()) -> dict[str, NameForm]:
    """Accept legacy callers and migrate v1 without guessing foreign name boundaries."""
    out = {}
    for lang, value in names.items():
        if lang == "ja-kana":
            continue
        if isinstance(value, NameForm):
            out[lang] = copy.deepcopy(value).with_defaults(lang)
        elif isinstance(value, dict):
            out[lang] = NameForm.from_json(lang, value)
        else:
            if lang == "ja":
                kana = names.get("ja-kana", "")
                np = list(parts) if "".join(parts) == "".join(value.split()) else value.split()
                rp = list(readings) if "".join(readings) == "".join(kana.split()) else kana.split()
                family, given = (np[0], " ".join(np[1:])) if len(np) > 1 else ("", value)
                fr, gr = (rp[0], " ".join(rp[1:])) if len(rp) > 1 else ("", kana)
                # An unsplit reading cannot be assigned to two name parts safely.
                if family and kana and len(rp) != len(np):
                    family, given, fr, gr = "", value.replace(" ", ""), "", kana.replace(" ", "")
                out[lang] = NameForm(family, given, fr, gr).with_defaults(lang)
            else:
                out[lang] = NameForm(given=value).with_defaults(lang)
    return out


@dataclass
class Character:
    id: str
    names: dict[str, NameForm] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    self_refs: list[dict[str, Any]] = field(default_factory=list)
    persona: dict[str, str] = field(default_factory=lambda: dict.fromkeys(PERSONA_FIELDS, ""))
    style: Optional[str] = None
    name_parts: list[str] = field(default_factory=list)
    kana_parts: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.names = _name_forms(self.names, self.name_parts, self.kana_parts)

    def name_form(self, lang: str) -> Optional[NameForm]:
        key = lang if lang in self.names else normalize_language_code(lang)
        keys = [key]
        if key in ("zh-Hans", "zh-Hant"):
            keys += ["zh-Hant" if key == "zh-Hans" else "zh-Hans", "zh"]
        return next((self.names[k] for k in keys if k in self.names and self.names[k].compose()), None)

    def full_name(self, lang: str) -> str:
        form = self.name_form(lang)
        return form.compose() if form else ""

    def name(self, lang: str, format: str = "short") -> str:
        form = self.name_form(lang)
        return form.compose(format) if form else ""

    def to_json(self) -> dict:
        data = asdict(self)
        data["names"] = {lang: form.to_json() for lang, form in self.names.items()}
        data.update(data.pop("extra"))
        if self.style is None:
            data.pop("style")
        return data

    @classmethod
    def from_json(cls, data: dict) -> "Character":
        fields = cls.__dataclass_fields__.keys() - {"extra"}
        return cls(**copy.deepcopy({k: v for k, v in data.items() if k in fields}),
                   extra=copy.deepcopy({k: v for k, v in data.items() if k not in fields}))


def character_display_name(character: Optional[Character], lang: str, label: str = "",
                           format: str = "full") -> str:
    """Human-readable name, without using an internal character identity."""
    if format == "label":
        return label
    if character is not None:
        forms = [character.name_form(lang), character.name_form("ja"), *character.names.values()]
        for form in forms:
            if form is not None:
                name = form.compose(format) or form.compose()
                if name.strip():
                    return name
        alias = next((alias for alias in character.aliases if alias.strip()), "")
        if alias:
            return alias
    return label or "（未命名人物）"


def needs_speaker(group, member):
    return (group.kind == "dialogue" and member.role in ("dialogue", "top", "alt")
            and member.source_kind != "Comment")


def shown_speaker_label(doc, group, preferred_lane="original", book=None) -> str:
    """Return the group's displayed label, preferring the selected speaker lane."""
    members = [m for m in doc.members_of(group) if needs_speaker(group, m)
               and (book is None or not book.speaker_ignored(m))
               and (m.speaker or {}).get("label")]
    source = next((m for m in members if m.lane == preferred_lane), members[0] if members else None)
    return (source.speaker or {}).get("label", "") if source is not None else ""


def file_name_labels(source):
    """Common ASS Name credits are file metadata; source records stay untouched."""
    events = [e for e in source.events if e.kind == "Dialogue"]
    counts = Counter(normalize_speaker(e.name) for e in events if e.name)
    return {name for name, count in counts.items() if count >= 20 and count * 2 >= len(events)}


def file_name_speaker(member, labels):
    if (member.speaker or {}).get("by") == "user":
        return False
    label = normalize_speaker((member.speaker or {}).get("label", ""))
    from .render_origin import run_source

    # File-level Name decisions do not require a source event number.
    return bool(label and any(label in labels.get(sid, set()) for run in member.runs
                              if (sid := run_source(member, run))))


class CharacterBook:
    def __init__(self, path: Path, *, original: str = "") -> None:
        self.original = normalize_language_code(original) if original else ""
        self.source_labels = {}
        self.path = Path(path)
        self.characters: list[Character] = []
        self.crowd: list[str] = []
        self.not_person: list[str] = []
        self.not_name: list[str] = []
        self.pending: list[dict] = []
        self.extra: dict[str, Any] = {}

    @classmethod
    def load(cls, path: Path, *, tsv: Optional[Path] = None,
             translation: str = "zh-Hant", original: str = "") -> "CharacterBook":
        book = cls(path, original=original)
        if book.path.exists():
            data = read_json(book.path)
            book._load_data(data)
            if data.get("format_version") == 1:
                backup = book.path.with_name(book.path.stem + ".v1.bak.json")
                if not backup.exists():
                    with backup.open("xb") as fh:
                        fh.write(book.path.read_bytes())
                book.save()
        elif tsv is not None and Path(tsv).exists():
            for old in CharacterTable.load(Path(tsv)).chars:
                names = {"ja": old.name}
                if old.kana:
                    names["ja-kana"] = old.kana
                if old.cn:
                    names[translation] = old.cn
                book.characters.append(Character(
                    new_id("chr_"), names, list(old.aliases),
                    [{"surface": "私", "reading": old.self_reading, "scope": {}}] if old.self_reading else [],
                    {**dict.fromkeys(PERSONA_FIELDS, ""), "note_ja": old.note},
                    name_parts=list(old.name_parts), kana_parts=list(old.kana_parts)))
            book.save()                    # one conversion; the TSV is never written
        if book.original == "ja" and not book.extra.get("default_self_refs_filled"):
            for ch in book.characters:
                book.ensure_default(ch)
            book.extra["default_self_refs_filled"] = True
            book.save()
        return book

    def ensure_default(self, ch):
        if (self.original == "ja" and not ch.extra.get("default_self_ref_removed")
                and not any(r.get("surface") == "私" and not r.get("scope") for r in ch.self_refs)):
            ch.self_refs.insert(0, {"surface": "私", "reading": "わたし", "scope": {}})

    def set_sources(self, sources):
        self.source_labels = {sid: file_name_labels(src) for sid, src in sources.items()}

    def speaker_ignored(self, member):
        sp = member.speaker or {}
        return (sp.get("character") == "nobody" or file_name_speaker(member, self.source_labels)
                or (sp.get("by") != "user" and any(normalize_speaker(sp.get("label", ""))
                    == normalize_speaker(label) for label in self.not_person)))

    def speaker_label(self, member):
        return "" if self.speaker_ignored(member) else (member.speaker or {}).get("label", "")

    def _load_data(self, data: dict) -> None:
        if data.get("format") != "subkit-characters" or data.get("format_version") not in (1, 2):
            raise ValueError("subkit-characters / format_version 1 or 2 required")
        chars = [Character.from_json(c) for c in data.get("characters", [])]
        ids = [c.id for c in chars]
        if any(not cid.startswith("chr_") for cid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Character IDs must be unique chr_ identities")
        self.characters = chars
        self.crowd = list(data.get("crowd", []))
        self.not_person = list(data.get("not_person", []))
        self.not_name = list(data.get("not_name", []))
        self.pending = copy.deepcopy(data.get("pending", []))
        self.extra = copy.deepcopy({k: v for k, v in data.items() if k not in
                                   {"format", "format_version", "characters", "crowd", "not_person", "not_name", "pending"}})

    def to_json(self) -> dict:
        return {**self.extra, "format": "subkit-characters", "format_version": 2,
                "characters": [c.to_json() for c in self.characters], "crowd": list(self.crowd),
                "not_person": list(self.not_person), "not_name": list(self.not_name),
                "pending": copy.deepcopy(self.pending)}

    def save(self) -> None:
        write_json_atomic(self.path, self.to_json())

    def snapshot(self) -> tuple[dict, Optional[bytes]]:
        return self.to_json(), self.path.read_bytes() if self.path.exists() else None

    def restore(self, state: tuple[dict, Optional[bytes]]) -> None:
        data, raw = state
        if raw is None:
            self.path.unlink(missing_ok=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(raw)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            finally:
                Path(tmp).unlink(missing_ok=True)
        self._load_data(data)

    def character(self, cid: str) -> Character:
        return next(c for c in self.characters if c.id == cid)

    def display_name(self, cid: Optional[str], lang: str, label: str = "", format: str = "full") -> str:
        if cid == "nobody":
            return ""
        ch = next((c for c in self.characters if c.id == cid), None)
        if ch is None:
            return label
        return character_display_name(ch, lang, label, format)

    def words(self) -> list[tuple[str, str]]:
        """Japanese full names and parts; ReadingDict retains single-kanji safeguards."""
        out = []
        for ch in self.characters:
            form = ch.name_form("ja")
            if form is None:
                continue
            pairs = [(form.family, form.family_reading), (form.given, form.given_reading)]
            if form.order == "given_first":
                pairs.reverse()
            filled = [(s, r) for s, r in pairs if s]
            if filled and all(r for s, r in filled):
                separator = form.separator or " "
                out.append((separator.join(s for s, r in filled), separator.join(r for s, r in filled)))
            out.extend((s, r) for s, r in filled if r)
        return out

    def pinyin_words(self, lang: Optional[str] = None) -> dict[str, str]:
        from .pinyin_notes import normalize_rule
        from ..textutil import is_han

        out = {}
        for ch in self.characters:
            forms = [ch.name_form(lang)] if lang else [f for code, f in ch.names.items()
                                                      if get_language(code).annotation_system == "pinyin"]
            for form in forms:
                if form is None:
                    continue
                for surface, reading in ((form.family, form.family_reading), (form.given, form.given_reading)):
                    if surface and reading:
                        out.setdefault(*normalize_rule(surface, reading))
                if form.compose() and all(is_han(c) for c in form.compose()) and len(form.parts()) == len(form.parts(readings=True)):
                    reading = " ".join(form.parts(readings=True))
                    if reading:
                        out.setdefault(*normalize_rule(form.compose(), reading))
        return out

    def resolve(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return None
        key = normalize_speaker(label)
        if any(normalize_speaker(x) == key for x in self.not_person):
            return None
        if any(normalize_speaker(x) == key for x in self.crowd):
            return "crowd"
        matches = {c.id for c in self.characters for x in [*(c.full_name(lang) for lang in c.names), *c.aliases]
                   if x and normalize_speaker(x) == key}
        return next(iter(matches)) if len(matches) == 1 else None  # ambiguous labels stay pending

    def _prune_pending(self) -> None:
        ignored = {normalize_speaker(x) for x in self.not_person}
        self.pending = [p for p in self.pending if self.resolve(p["label"]) is None
                        and normalize_speaker(p["label"]) not in ignored]

    def add_character(self, names: dict, **fields) -> Character:
        ch = Character(new_id("chr_"), dict(names), **copy.deepcopy(fields))
        self.ensure_default(ch)
        self.characters.append(ch)
        self._prune_pending()
        self.save()
        return ch

    def update_character(self, cid: str, **fields) -> Character:
        ch = self.character(cid)
        allowed = Character.__dataclass_fields__.keys() - {"id", "extra"}
        if fields.keys() - allowed:
            raise ValueError("Unknown or immutable character field")
        if "self_refs" in fields and any(r.get("surface") == "私" and not r.get("scope") for r in ch.self_refs):
            if not any(r.get("surface") == "私" and not r.get("scope") for r in fields["self_refs"]):
                ch.extra["default_self_ref_removed"] = True
        for key, value in fields.items():
            setattr(ch, key, _name_forms(value) if key == "names" else copy.deepcopy(value))
        self._prune_pending()
        self.save()
        return ch

    def rename_character(self, cid: str, lang: str, name: NameForm | dict | str) -> Character:
        names = dict(self.character(cid).names)
        names[lang] = name
        return self.update_character(cid, names=names)

    def mark_label(self, label: str, *, crowd: bool) -> None:
        key = normalize_speaker(label)
        self.crowd = [x for x in self.crowd if normalize_speaker(x) != key]
        self.not_person = [x for x in self.not_person if normalize_speaker(x) != key]
        (self.crowd if crowd else self.not_person).append(label)
        self._prune_pending()
        self.save()

    def mark_not_name(self, text: str) -> None:
        """Remember a rejected dialogue-name suggestion without changing speaker links."""
        key = normalize_speaker(text)
        if key and not any(normalize_speaker(x) == key for x in self.not_name):
            self.not_name.append(text)
            self.save()

    def resolve_pending(self, label: str, cid: str) -> None:
        ch = self.character(cid)
        key = normalize_speaker(label)
        self.crowd = [x for x in self.crowd if normalize_speaker(x) != key]
        self.not_person = [x for x in self.not_person if normalize_speaker(x) != key]
        if label not in ch.aliases:
            ch.aliases.append(label)
        self._prune_pending()
        self.save()

    def delete_character(self, cid: str, *, editor=None) -> None:
        if editor is not None:
            editor.edit_characters(lambda book: book.delete_character(cid), book=self,
                                   replacements={cid: None})
            return
        self.character(cid)
        self.characters = [c for c in self.characters if c.id != cid]
        self.save()

    def merge_characters(self, source: str, target: str, *, editor=None) -> None:
        if source == target:
            raise ValueError("Cannot merge a character into itself")
        if editor is not None:
            editor.edit_characters(lambda book: book.merge_characters(source, target), book=self,
                                   replacements={source: target})
            return
        a, b = self.character(source), self.character(target)
        b.aliases = list(dict.fromkeys([*b.aliases, *a.aliases, *(a.full_name(lang) for lang in a.names)]))
        for lang, name in a.names.items():
            b.names.setdefault(lang, name)
        for ref in a.self_refs:
            if ref not in b.self_refs:
                b.self_refs.append(copy.deepcopy(ref))
        for key, value in a.persona.items():
            if not b.persona.get(key):
                b.persona[key] = value
        if b.style is None:
            b.style = a.style
        self.characters.remove(a)
        self._prune_pending()
        self.save()

    def link_document(self, doc, *, tx=None, sources=None) -> int:
        """Refresh auto links and replace this episode's pending counts, never user picks."""
        if sources is not None:
            self.set_sources(sources)
        changed = 0
        counts: dict[str, dict] = {}
        ignored = {normalize_speaker(x) for x in self.not_person}
        for m in doc.members.values():
            if (not m.speaker or m.speaker.get("by") == "user"
                    or not needs_speaker(doc.group(m.group), m) or self.speaker_ignored(m)):
                continue
            label = m.speaker.get("label", "")
            cid = self.resolve(label)
            if m.speaker.get("character") != cid:
                if tx is not None:
                    tx.member(m.id)
                m.speaker = {**m.speaker, "character": cid}
                m.rev += 1
                changed += 1
            if label and cid is None and normalize_speaker(label) not in ignored:
                p = counts.setdefault(label, {"count": 0, "examples": []})
                p["count"] += 1
                example = "".join(r.text for r in m.runs if r.kind != "tag")
                if example and example not in p["examples"] and len(p["examples"]) < 3:
                    p["examples"].append(example)
        pending = {p["label"]: copy.deepcopy(p) for p in self.pending}
        for p in pending.values():
            p.setdefault("by_episode", {}).pop(doc.episode, None)
        for label, stats in counts.items():
            p = pending.setdefault(label, {"label": label, "by_episode": {}})
            p["by_episode"][doc.episode] = stats
        self.pending = []
        for p in pending.values():
            episodes = p["by_episode"]
            if not episodes:
                continue
            p.update(count=sum(s["count"] for s in episodes.values()), episodes=list(episodes),
                     examples=list(dict.fromkeys(x for s in episodes.values() for x in s["examples"]))[:3])
            self.pending.append(p)
        self._prune_pending()
        self.save()
        return changed


def preserve_user_speakers(previous, document) -> None:
    """Transfer explicit picks by source ranges when a whole merge generates new IDs."""
    if previous is None:
        return
    ranges: dict[tuple, list] = {}
    users = {m.id for m in previous.members.values() if m.speaker and m.speaker.get("by") == "user"}
    for mid in users:
        m = previous.members[mid]
        for r in m.runs:
            if r.src is not None and r.a < r.b:
                ranges.setdefault((m.lane, r.src, r.e), []).append((r.a, r.b, m))
    matched = set()
    for m in document.members.values():
        picks = {}
        for r in m.runs:
            for a, b, old in ranges.get((m.lane, r.src, r.e), []):
                if r.a < b and a < r.b:
                    picks[old.speaker.get("character")] = old.speaker
                    matched.add(old.id)
        if len(picks) > 1:
            raise RuntimeError("重新合并会把不同的人工说话人合到一行；请保留现有配对")
        if picks:
            old = next(iter(picks.values()))
            m.speaker = {**(m.speaker or {"label": old.get("label", "")}),
                         "character": old.get("character"), "by": "user"}
    if users - matched:
        raise RuntimeError("重新合并无法保留部分人工说话人；请保留现有配对")
