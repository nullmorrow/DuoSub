"""Character classes and kana/width normalisation shared by every module."""

from __future__ import annotations

import re
import unicodedata

# Characters that behave like kanji for ruby purposes.  Iteration marks belong
# to the kanji block they repeat (日々, 神々); ヶ/ヵ are counters inside kanji
# compounds (一ヶ月); 〆 is a kanji-like abbreviation.
KANJI_LIKE_EXTRA = set("々〆〻ヶヵ")


def is_kanji(ch: str) -> bool:
    if not ch:
        return False
    if ch in KANJI_LIKE_EXTRA:
        return True
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x3134F
    )


def is_han(ch: str) -> bool:
    """Han ideograph only (no iteration marks) — used for Chinese text."""
    if not ch:
        return False
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x3134F


def is_hiragana(ch: str) -> bool:
    return bool(ch) and ("ぁ" <= ch <= "ゖ" or ch in "ゝゞゟ")


def is_katakana(ch: str) -> bool:
    return bool(ch) and ("ァ" <= ch <= "ヺ" or ch in "ーヽヾヿ" or "ㇰ" <= ch <= "ㇿ")


def is_kana(ch: str) -> bool:
    return is_hiragana(ch) or is_katakana(ch)


def is_halfwidth_katakana(ch: str) -> bool:
    return bool(ch) and "｡" <= ch <= "ﾟ"


def has_kanji(text: str) -> bool:
    return any(is_kanji(ch) for ch in text or "")


def has_kana(text: str) -> bool:
    return any(is_kana(ch) for ch in text or "")


def kata_to_hira(text: str) -> str:
    out = []
    for ch in text or "":
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        elif ch == "ヽ":
            out.append("ゝ")
        elif ch == "ヾ":
            out.append("ゞ")
        else:
            out.append(ch)
    return "".join(out)


def hira_to_kata(text: str) -> str:
    out = []
    for ch in text or "":
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        elif ch == "ゝ":
            out.append("ヽ")
        elif ch == "ゞ":
            out.append("ヾ")
        else:
            out.append(ch)
    return "".join(out)


_HALFWIDTH_RUN = re.compile(r"[｡-ﾟ]+")


def normalize_halfwidth_kana(text: str) -> str:
    """ｱｷ → アキ, ｶﾞ → ガ, ｡ → 。, ｢｣ → 「」, ･ → ・ (only the U+FF61–FF9F block)."""
    return _HALFWIDTH_RUN.sub(lambda m: unicodedata.normalize("NFKC", m.group(0)), text or "")


def fullwidth_ascii_to_half(text: str, *, digits: bool = True, letters: bool = True, space: bool = False) -> str:
    out = []
    for ch in text or "":
        code = ord(ch)
        if digits and 0xFF10 <= code <= 0xFF19:
            out.append(chr(code - 0xFEE0))
        elif letters and (0xFF21 <= code <= 0xFF3A or 0xFF41 <= code <= 0xFF5A):
            out.append(chr(code - 0xFEE0))
        elif space and ch == "　":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def is_digit_char(ch: str) -> bool:
    return bool(ch) and ("0" <= ch <= "9" or "０" <= ch <= "９")


def digits_to_ascii(text: str) -> str:
    return "".join(chr(ord(c) - 0xFEE0) if "０" <= c <= "９" else c for c in text)


def is_cjk_punct(ch: str) -> bool:
    if not ch:
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S") or ch in "　 "


def is_space(ch: str) -> bool:
    return ch in (" ", "　", "\t", " ")


def visible_width_units(text: str) -> float:
    """Rough display width in full-width units (CJK = 1, ASCII = 0.5)."""
    total = 0.0
    for ch in text:
        eaw = unicodedata.east_asian_width(ch)
        total += 1.0 if eaw in ("F", "W", "A") else 0.5
    return total


BIDI_CONTROLS = re.compile("[؜‎‏‪-‮⁦-⁩﻿]")


def strip_bidi(text: str) -> str:
    return BIDI_CONTROLS.sub("", text or "")
