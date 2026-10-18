"""Chinese annotations, using sentence context and the shared character anchors."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Mapping, Optional

from ..model.document import Annotation, Document, Member
from ..languages import get_language
from ..model.ids import new_id
from ..pinyin import line_pinyin
from ..textutil import is_han
from .readings import anchor_of, kept_positions, visible_map

INVALID_PINYIN = "拼音无效：请输入有效的拼音音节（如 hang2、háng、lv4；5 或不写数字为轻声）。"
INVALID_RULE = "拼音规则须为每个汉字填写一个音节，用空格分隔。"
MODES = ("off", "PY", "ZY", "WG")
_CONTEXT_ALIASES = {"銀行": "银行", "行長": "行长"}


@lru_cache(maxsize=1)
def _syllables() -> frozenset[str]:
    from pypinyin.pinyin_dict import pinyin_dict
    from pypinyin.contrib.tone_convert import to_normal

    return frozenset(to_normal(s) for value in pinyin_dict.values() for s in value.split(","))


def normalize_pinyin(text: str) -> str:
    """Accept one marked or numbered syllable; v represents ü, 5 is neutral."""
    from pypinyin.contrib.tone_convert import to_normal, to_tone, to_tone3

    s = unicodedata.normalize("NFC", text.strip().lower()).replace("v", "ü")
    digit = s[-1:] if s[-1:].isdigit() else ""
    if digit:
        if digit not in "12345":
            raise ValueError(INVALID_PINYIN)
        s = s[:-1]
    base = to_normal(s)
    if base not in _syllables() or not re.fullmatch(r"[a-züêāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹḿ]+", s):
        raise ValueError(INVALID_PINYIN)
    marked = s != base.replace("v", "ü")
    if digit and marked:
        raise ValueError(INVALID_PINYIN)
    result = to_tone(base + digit) if digit else (to_tone(to_tone3(s)) if marked else s)
    if marked and result != s:
        raise ValueError(INVALID_PINYIN)
    return result


def normalize_rule(surface: str, reading: str) -> tuple[str, str]:
    word = surface.strip()
    syllables = reading.split()
    if not word or not all(is_han(ch) for ch in word) or len(syllables) != len(word):
        raise ValueError(INVALID_RULE)
    return word, " ".join(normalize_pinyin(s) for s in syllables)


def suggest_pinyin(m: Member, rules: Optional[Mapping[str, str]] = None, *,
                   covered: Optional[set[int]] = None,
                   window: Optional[tuple[int, int]] = None) -> list[Annotation]:
    """Per-Han-character suggestions; longest matching words win, kept positions are skipped."""
    if get_language(m.lang).annotation_system != "pinyin":
        return []
    vt, ids = visible_map(m)
    chosen = {i: (s, alts) for i, s, alts in line_pinyin(vt.text)}
    # pypinyin 0.55 lacks these Traditional phrase entries; retain the original anchors/candidates.
    for word, alias in _CONTEXT_ALIASES.items():
        start = 0
        while (start := vt.text.find(word, start)) >= 0:
            for k, syllable, _alts in line_pinyin(alias):
                i = start + k
                if i in chosen:
                    old, alts = chosen[i]
                    chosen[i] = (syllable, list(dict.fromkeys([old] + alts)))
            start += len(word)
    overrides: dict[int, tuple[str, str]] = {}
    words = sorted((normalize_rule(w, r) for w, r in (rules or {}).items()), key=lambda x: -len(x[0]))
    for word, reading in words:
        start = 0
        while (start := vt.text.find(word, start)) >= 0:
            if not any(i in overrides for i in range(start, start + len(word))):
                overrides.update({start + k: (s, word) for k, s in enumerate(reading.split())})
            start += 1
    out = []
    for i, (syllable, alts) in chosen.items():
        if i in (covered or ()) or ids[i] is None or (window is not None and not window[0] <= i < window[1]):
            continue
        rule = overrides.get(i)
        if rule:
            syllable = rule[0]
        try:
            syllable = normalize_pinyin(syllable)
        except ValueError:
            continue                     # unknown ideographs have no pinyin
        candidates = list(dict.fromkeys([syllable] + [chosen[i][0]] + alts))
        out.append(Annotation(new_id("a_"), "pinyin", m.id, anchor_of([ids[i]]), vt.text[i], syllable,
                              status="auto", origin="rule" if rule else "analyzer",
                              rule=f"pinyin:{rule[1]}" if rule else None, candidates=candidates,
                              parts=[[0, 1, syllable]]))
    return out


def annotate_pinyin(doc: Document, m: Member, rules: Optional[Mapping[str, str]] = None) -> int:
    """Refresh only automatic pinyin, preserving every human choice."""
    if get_language(m.lang).annotation_system != "pinyin":
        return 0
    _vt, ids = visible_map(m)
    keep, covered = kept_positions(doc, m, ids, kind="pinyin")
    for ann in list(doc.annotations.values()):
        if ann.member == m.id and ann.kind == "pinyin" and ann not in keep:
            del doc.annotations[ann.id]
    fresh = suggest_pinyin(m, rules, covered=covered)
    doc.annotations.update((ann.id, ann) for ann in fresh)
    return len(fresh)


def episode_rules(doc: Document, words: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    rules = dict(words or {})
    rules.update((r["surface"], r["reading"]) for r in doc.rules if r.get("kind") == "pinyin")
    return rules
