"""Pinyin / Zhuyin / Wade–Giles over Chinese lines.

Readings come from pypinyin with sentence context (the old tool's fix for
index drift around spaces/punctuation is built in: only Han characters are
sent to pypinyin, then mapped back).  The ``@拼音`` record lists the
polyphonic characters of the line — edit it like the Japanese record::

    行（háng） 长（zhǎng/cháng） 了（le）

Display mode (config ``[pinyin] mode``): PY (tone marks), ZY (注音), WG (威妥玛).
"""

from __future__ import annotations

import re
from typing import Optional

from .textutil import is_han

try:
    from pypinyin import Style as _Style, pinyin as _pinyin
except ImportError:  # pragma: no cover
    _pinyin = None
    _Style = None

_TONE = {"ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4), "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3),
         "è": ("e", 4), "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4), "ō": ("o", 1), "ó": ("o", 2),
         "ǒ": ("o", 3), "ò": ("o", 4), "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4), "ǖ": ("ü", 1),
         "ǘ": ("ü", 2), "ǚ": ("ü", 3), "ǜ": ("ü", 4), "ń": ("n", 2), "ň": ("n", 3), "ǹ": ("n", 4)}
_MARK = {(b, t): k for k, (b, t) in _TONE.items()}

ZY_INI = {"b": "ㄅ", "p": "ㄆ", "m": "ㄇ", "f": "ㄈ", "d": "ㄉ", "t": "ㄊ", "n": "ㄋ", "l": "ㄌ", "g": "ㄍ", "k": "ㄎ",
          "h": "ㄏ", "j": "ㄐ", "q": "ㄑ", "x": "ㄒ", "zh": "ㄓ", "ch": "ㄔ", "sh": "ㄕ", "r": "ㄖ", "z": "ㄗ", "c": "ㄘ", "s": "ㄙ"}
ZY_FIN = {"": "", "a": "ㄚ", "o": "ㄛ", "e": "ㄜ", "ê": "ㄝ", "ai": "ㄞ", "ei": "ㄟ", "ao": "ㄠ", "ou": "ㄡ", "an": "ㄢ",
          "en": "ㄣ", "ang": "ㄤ", "eng": "ㄥ", "er": "ㄦ", "i": "ㄧ", "ia": "ㄧㄚ", "ie": "ㄧㄝ", "iao": "ㄧㄠ", "iu": "ㄧㄡ",
          "iou": "ㄧㄡ", "ian": "ㄧㄢ", "in": "ㄧㄣ", "iang": "ㄧㄤ", "ing": "ㄧㄥ", "iong": "ㄩㄥ", "u": "ㄨ", "ua": "ㄨㄚ",
          "uo": "ㄨㄛ", "uai": "ㄨㄞ", "ui": "ㄨㄟ", "uei": "ㄨㄟ", "uan": "ㄨㄢ", "un": "ㄨㄣ", "uen": "ㄨㄣ", "uang": "ㄨㄤ",
          "ong": "ㄨㄥ", "ü": "ㄩ", "üe": "ㄩㄝ", "üan": "ㄩㄢ", "ün": "ㄩㄣ"}
ZY_TONE = {1: "", 2: "ˊ", 3: "ˇ", 4: "ˋ", 5: "˙"}
WG_INI = {"b": "p", "p": "p'", "d": "t", "t": "t'", "g": "k", "k": "k'", "j": "ch", "q": "ch'", "x": "hs", "zh": "ch",
          "ch": "ch'", "sh": "sh", "r": "j", "z": "ts", "c": "ts'", "s": "s", "m": "m", "f": "f", "n": "n", "l": "l", "h": "h"}


def split_tone(syl: str) -> tuple[str, int]:
    base, tone = [], 5
    for ch in syl:
        if ch in _TONE:
            b, tone = _TONE[ch]
            base.append(b)
        else:
            base.append(ch)
    return "".join(base).replace("v", "ü"), tone


def _ini_fin(base: str) -> tuple[str, str]:
    s = base
    if s.startswith("y"):
        r = s[1:]
        if r.startswith("u"):
            return "", "ü" + r[1:]
        return "", r if r.startswith("i") else "i" + r
    if s.startswith("w"):
        r = s[1:]
        return "", r if r.startswith("u") else "u" + r
    for ini in ("zh", "ch", "sh"):
        if s.startswith(ini):
            return ini, s[2:]
    if s[:1] in ZY_INI:
        return s[:1], s[1:]
    return "", s


def to_zhuyin(syl: str) -> str:
    base, tone = split_tone(syl)
    ini, fin = _ini_fin(base)
    if ini in ("zh", "ch", "sh", "r", "z", "c", "s") and fin == "i":
        fin = ""
    if ini in ("j", "q", "x") and fin.startswith("u"):
        fin = "ü" + fin[1:]
    return ZY_INI.get(ini, "") + ZY_FIN.get(fin, fin) + ZY_TONE.get(tone, "")


def to_wade(syl: str) -> str:
    base, tone = split_tone(syl)
    ini, fin = _ini_fin(base)
    if ini in ("zh", "ch", "sh", "r", "z", "c", "s") and fin == "i":
        fin = "ih" if ini in ("zh", "ch", "sh", "r") else "ŭ"
    if ini in ("j", "q", "x") and fin.startswith("u"):
        fin = "ü" + fin[1:]
    fin = {"ong": "ung", "iong": "iung", "ie": "ieh", "üe": "üeh", "ian": "ien", "uo": "o", "e": "ê"}.get(fin, fin)
    return WG_INI.get(ini, ini) + fin + (str(tone) if tone < 5 else "")


def display(syl: str, mode: str) -> str:
    mode = (mode or "PY").upper()
    if mode == "ZY":
        return to_zhuyin(syl)
    if mode == "WG":
        return to_wade(syl)
    return syl


def line_pinyin(text: str) -> list[tuple[int, str, list[str]]]:
    """[(visible index, chosen syllable, alternatives)] for every Han character."""
    if _pinyin is None:
        return []
    idx = [i for i, ch in enumerate(text) if is_han(ch)]
    if not idx:
        return []
    han = "".join(text[i] for i in idx)
    chosen = _pinyin(han, style=_Style.TONE, heteronym=False, strict=False)
    out = []
    for k, i in enumerate(idx):
        c = chosen[k][0] if k < len(chosen) and chosen[k] else ""
        alts = _pinyin(text[i], style=_Style.TONE, heteronym=True, strict=False)
        alts = [a for a in (alts[0] if alts else []) if a and a != c]
        out.append((i, c, alts))
    return out


def format_pinyin_record(items: list[tuple[int, str, list[str]]], text: str, manual: dict[int, str]) -> str:
    """Record lists polyphonic characters (and every manual choice)."""
    parts = []
    for i, syl, alts in items:
        if i in manual:
            parts.append(f"{text[i]}（{manual[i]}）")
        elif alts:
            parts.append(f"{text[i]}（{'/'.join([syl] + alts[:3])}）")
    return " ".join(parts)


_REC = re.compile(r"([^\s（）()])[（(]([^（）()]*)[）)]")


def parse_pinyin_record(record: str, text: str) -> dict[int, str]:
    """Map record entries back to Han positions (in order of appearance)."""
    out: dict[int, str] = {}
    cursor = 0
    for m in _REC.finditer(record or ""):
        ch, body = m.group(1), m.group(2).strip()
        p = text.find(ch, cursor)
        if p < 0:
            continue
        syl = re.split(r"[/／|]", body)[0].strip()
        out[p] = syl
        cursor = p + 1
    return out
