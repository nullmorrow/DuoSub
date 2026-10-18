"""Explicit, Qt-free Bangumi fetching, read-only previews and character import."""

from __future__ import annotations

import copy
import json
import re
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ..languages import normalize_language_code
from ..model.ids import new_id
from ..textutil import kata_to_hira
from .people import Character, CharacterBook, NameForm

try:
    from subkit_app import APP_VERSION
except ImportError:
    APP_VERSION = "0.7.1"

USER_AGENT = f"DuoSub/{APP_VERSION} (https://github.com/nullmorrow/DuoSub)"
API = "https://api.bgm.tv/v0"


class BangumiError(ValueError):
    def __init__(self, reason: str):
        super().__init__(f"无法从 Bangumi 读取：{reason}")


class BangumiCancelled(Exception):
    """An explicit user cancellation between requests; no partial import."""


def parse_subject_id(text: str) -> int:
    """Accept a positive ASCII ID or a subject URL on the three Bangumi hosts."""
    text = text.strip()
    if re.fullmatch(r"[0-9]+", text):
        value = int(text)
    else:
        try:
            url = urlsplit(text)
            match = re.fullmatch(r"/subject/([0-9]+)/?", url.path)
            if (url.scheme not in ("http", "https") or url.netloc.lower() not in
                    ("bangumi.tv", "bgm.tv", "chii.in") or not match):
                raise ValueError
            value = int(match[1])
        except ValueError:
            raise BangumiError("请输入作品链接或编号") from None
    if value <= 0:
        raise BangumiError("作品编号必须大于 0")
    return value


def _read(path: str, timeout: float):
    try:
        with urlopen(Request(API + path, headers={"User-Agent": USER_AGENT}), timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        exc.close()
        raise BangumiError(f"HTTP {exc.code}（{path}）") from exc
    except (URLError, TimeoutError, OSError, UnicodeError, ValueError) as exc:
        raise BangumiError(str(exc)) from exc


def fetch_subject(text: str, *, timeout: float = 10, delay: float = 0.2,
                  cancelled=None, progress=None) -> tuple[dict, list[dict], dict]:
    """Fetch only on explicit invocation; retain names/credits/infobox, no artwork or summaries.

    Returns (subject, characters, details keyed by integer character ID). A failed
    detail request rejects the batch rather than presenting an incomplete import.
    """
    def checkpoint():
        if cancelled is not None and cancelled():
            raise BangumiCancelled()

    checkpoint()
    sid = parse_subject_id(text)
    if timeout <= 0 or delay < 0:
        raise BangumiError("超时必须大于 0，请求间隔不能小于 0")
    subject = _read(f"/subjects/{sid}", timeout)
    checkpoint()
    characters = _read(f"/subjects/{sid}/characters", timeout)
    if not isinstance(subject, dict) or not isinstance(characters, list):
        raise BangumiError("作品资料格式无效")
    rows, details = [], {}
    for character in characters:
        checkpoint()
        if not isinstance(character, dict) or not isinstance(character.get("id"), int):
            raise BangumiError("人物资料格式无效")
        cid = character["id"]
        if cid not in details:
            time.sleep(delay)
            checkpoint()
            if progress is not None:
                progress(len(rows), len(characters))
            detail = _read(f"/characters/{cid}", timeout)
            if not isinstance(detail, dict):
                raise BangumiError(f"人物 {cid} 的资料格式无效")
            details[cid] = {key: detail.get(key) for key in ("id", "name", "infobox")}
        actors = character.get("actors") or []
        if not isinstance(actors, list) or any(not isinstance(a, dict) for a in actors):
            raise BangumiError(f"人物 {cid} 的声优资料格式无效")
        rows.append({"id": cid, "name": character.get("name", ""),
                     "relation": character.get("relation", ""),
                     "actors": [{"id": a.get("id"), "name": a.get("name", "")}
                                for a in actors]})
    checkpoint()
    return ({"id": sid, "name": subject.get("name", ""), "name_cn": subject.get("name_cn", "")},
            rows, details)


def _info(detail: dict) -> dict[str, str]:
    out = {}
    for entry in detail.get("infobox") or []:
        value = entry.get("value")
        if isinstance(value, str):
            out.setdefault(entry.get("key", ""), value.strip())
        elif isinstance(value, list):
            for alias in value:
                if isinstance(alias, dict) and isinstance(alias.get("v"), str):
                    out.setdefault(alias.get("k", ""), alias["v"].strip())
    return out


def _kana_key(text: str) -> str:
    # Hepburn macrons / omitted long vowels occur frequently in name romanizations.
    text = kata_to_hira(text).replace("ー", "")
    return re.sub(r"([おこそとのほもよろごぞどぼぽょ])(?:う|お)|([えけせてねへめれげぜでべぺ])い",
                  lambda m: m[1] or m[2], text)


def _romaji_matches(romaji: str, family: str, given: str) -> bool:
    from ..reading.records import romaji_to_kana

    plain = "".join(c for c in unicodedata.normalize("NFD", romaji) if not unicodedata.combining(c))
    words = re.findall(r"[A-Za-z']+", plain)
    if len(words) != 2:
        return False
    actual = [_kana_key(romaji_to_kana(w)) for w in words]
    expected = [_kana_key(family), _kana_key(given)]
    return actual == expected or actual == expected[::-1]


def _japanese_name(name: str, kana: str, romaji: str) -> NameForm:
    from sudachipy import SplitMode
    from ..reading.engine import Analyzer

    whole = NameForm(given=name, given_reading=kata_to_hira(kana).replace(" ", "")).with_defaults("ja")
    parts = name.split("・")
    if len(parts) >= 2 and all(re.fullmatch(r"[ァ-ヺー]+", p) for p in parts):
        # Foreign spelling is not Japanese romaji; only its boundaries can be checked.
        if romaji and len(romaji.split()) != len(parts):
            return whole
        family = "・".join(parts[1:])
        return NameForm(family=family, given=parts[0], family_reading=kata_to_hira(family),
                        given_reading=kata_to_hira(parts[0]), order="given_first", separator="・").with_defaults("ja")
    morphs = [m for m in Analyzer.get().tokenize(name, mode=SplitMode.A) if m.surface.strip()]
    supplied = kata_to_hira(kana).split()
    if (len(morphs) == 2 and morphs[0].pos[:3] == ("名詞", "固有名詞", "人名")
            and (morphs[0].pos[3] == "姓" or
                 (len(supplied) == 2 and romaji and _romaji_matches(romaji, *supplied)))
            and morphs[1].pos[:4] == ("名詞", "固有名詞", "人名", "名")
            and "".join(m.surface for m in morphs) == "".join(name.split())):
        a, b = morphs
        readings = [a.reading, b.reading]
        if kana:
            if len(supplied) == 2:
                readings = supplied
            elif "".join(supplied) != "".join(readings):
                return whole
        if romaji and not _romaji_matches(romaji, *readings):
            return whole
        return NameForm(a.surface, b.surface, *readings).with_defaults("ja")
    return whole


def _chinese_name(name: str, ja: NameForm, lang: str) -> NameForm:
    whole = NameForm(given=name).with_defaults(lang)
    if not ja.family:
        return whole
    if ja.order == "given_first":
        parts = re.split(r"[・·]", name)
        if len(parts) == len(ja.compose().split("・")) and all(parts):
            return NameForm(family="·".join(parts[1:]), given=parts[0], order="given_first", separator="·").with_defaults(lang)
    elif len(name) == len(ja.family) + len(ja.given) and re.fullmatch(r"[\u3400-\u9fff]+", name):
        boundary = len(ja.family)
        return NameForm(family=name[:boundary], given=name[boundary:]).with_defaults(lang)
    return whole


def build_candidates(subject, characters, details, book, languages, traditional: bool = False) -> list[dict]:
    """Pure preview: no network, book writes, aliases or automatic speaker decisions.

    Languages is the project's lane map. Only Japanese and available Chinese names
    are proposed; romanization is evidence, never an invented English translation.
    ``match`` may be edited to an existing ID or None before applying.
    """
    converter = None
    if traditional:
        from opencc import OpenCC
        converter = OpenCC("s2tw")
    codes = {normalize_language_code(code) for code in languages.values()}
    rows = []
    for character in characters:
        cid = character["id"]
        detail = details.get(cid, details.get(str(cid), {})) or {}
        info = _info(detail)
        ja_name = (info.get("日文名") or character.get("name") or detail.get("name", "") or "").strip()
        zh_name = (info.get("简体中文名") or info.get("第二中文名", "")).strip()
        if converter:
            zh_name = converter.convert(zh_name)
        romaji = info.get("罗马字", "")
        ja = _japanese_name(ja_name, info.get("假名") or info.get("纯假名", ""), romaji)
        names = {"ja": ja}
        if zh_name:
            for lang in sorted(codes):
                if lang.startswith("zh-"):
                    names[lang] = _chinese_name(zh_name, ja, lang)
        matches = [c.id for c in book.characters if ja_name and
                   (ja_name == c.full_name("ja") or ja_name in c.aliases)]
        relation = character.get("relation", "")
        rows.append({"bangumi": cid, "ja_name": ja_name, "zh_name": zh_name, "romaji": romaji,
                     "relation": relation, "cv": "、".join(a.get("name", "") for a in character.get("actors", [])),
                     "names": names, "match": matches[0] if len(matches) == 1 else None,
                     "checked": relation in ("主角", "配角") and any(f.compose().strip() for f in names.values()),
                     "missing_name": not any(f.compose().strip() for f in names.values())})
    return rows


def has_displayable_name(row):
    return any(form.compose().strip() for form in row.get("names", {}).values())


def apply_import(book: CharacterBook, rows: list[dict], chosen, *, editor=None) -> list[str]:
    """Import chosen row indexes, using editable names/match; one editor undo step.

    With no open document/editor, save once. Existing fields (including source IDs)
    are only filled when empty. No aliases/persona/credits are guessed or replaced.
    """
    selected = [rows[i] for i in dict.fromkeys(chosen)]
    if not selected:
        return []
    if any(not has_displayable_name(row) for row in selected):
        raise ValueError("这一行没有名字，请先填名字")

    def change(target):
        chars = copy.deepcopy(target.characters)
        imported = []
        for row in selected:
            cid = row.get("match")
            if cid is None:
                ch = Character(new_id("chr_"), copy.deepcopy(row["names"]))
                target.ensure_default(ch)
                chars.append(ch)
            else:
                ch = next((c for c in chars if c.id == cid), None)
                if ch is None:
                    raise ValueError(f"Unknown character: {cid}")
                for lang, proposed in row["names"].items():
                    if lang not in ch.names:
                        ch.names[lang] = copy.deepcopy(proposed)
                        continue
                    existing = ch.names[lang]
                    # A saved whole name cannot safely acquire a second copy of its surname.
                    if not existing.family and existing.given == proposed.compose() and proposed.family:
                        if not existing.given_reading:
                            existing.given_reading = "".join(proposed.parts(readings=True))
                        continue
                    for field in ("family", "given", "family_reading", "given_reading", "order", "separator"):
                        if not getattr(existing, field):
                            if field.endswith("_reading") and getattr(existing, field[:-8]) != getattr(proposed, field[:-8]):
                                continue
                            # Empty separator is an explicit valid setting; defaults are already resolved.
                            if field == "separator" and existing.separator is not None:
                                continue
                            setattr(existing, field, getattr(proposed, field))
            if not ch.extra.get("source"):
                ch.extra["source"] = {}
            source = ch.extra["source"]
            if isinstance(source, dict) and not source.get("bangumi"):
                source["bangumi"] = row["bangumi"]
            imported.append(ch.id)
        previous = target.characters
        target.characters = chars
        try:
            target.save()
        except Exception:
            target.characters = previous
            raise
        return imported

    if editor is not None:
        return editor.edit_characters(change, book=book)
    return change(book)
