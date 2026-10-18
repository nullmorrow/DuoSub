"""The per-line reading record (the ``@读音`` comment the user edits in Aegisub).

Syntax — entries separated by spaces::

    遠月（とおつき） 学園（がくえん） ?何（なに/なん） 考え（かんがえ） *幸平創真（ゆきひら そうま）

    表記（よみ）        an entry; full- or half-width brackets
    ?表記（a/b）        tool is unsure; first alternative is used — delete the wrong one or the "?"
    !表記（よみ）       force this reading here even against the dictionary
    *表記（よみ）       shown by the tool: comes from the dictionary (editing it needs "!")
    表記（-）           no ruby for this word
    表記（よみ,80）     fixed horizontal scale (\\fscx80) for this ruby
    学（がっ）校（こう） adjacent entries without space = per-kanji (mono) ruby
    A（a）+B（b）       join into one group ruby
    読音 input helpers: romaji (nani → なに), K: prefix for katakana (K:furai → フライ)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..textutil import hira_to_kata

_ENTRY = re.compile(r"([?!*？！＊]*)([^\s（）()+]+?)[（(]([^（）()]*)[）)]")


@dataclass
class RecEntry:
    surface: str
    reading: str                 # first alternative, normalised; "" when hidden
    alternatives: list[str] = field(default_factory=list)
    uncertain: bool = False
    forced: bool = False
    from_dict: bool = False
    hidden: bool = False
    scale: Optional[float] = None
    joined_with_next: bool = False
    adjacent_to_next: bool = False   # written without a space before the next entry (mono ruby)

    def key(self) -> tuple:
        return (self.surface, self.reading, self.hidden, self.forced, self.scale)


# ------------------------------------------------------------------ romaji
_ROMAJI = {
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お", "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "sa": "さ", "shi": "し", "si": "し", "su": "す", "se": "せ", "so": "そ", "ta": "た", "chi": "ち", "ti": "ち",
    "tsu": "つ", "tu": "つ", "te": "て", "to": "と", "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "fu": "ふ", "hu": "ふ", "he": "へ", "ho": "ほ", "ma": "ま", "mi": "み", "mu": "む",
    "me": "め", "mo": "も", "ya": "や", "yu": "ゆ", "yo": "よ", "ra": "ら", "ri": "り", "ru": "る", "re": "れ",
    "ro": "ろ", "wa": "わ", "wo": "を", "ga": "が", "gi": "ぎ", "gu": "ぐ", "ge": "げ", "go": "ご", "za": "ざ",
    "ji": "じ", "zi": "じ", "zu": "ず", "ze": "ぜ", "zo": "ぞ", "da": "だ", "di": "ぢ", "du": "づ", "de": "で",
    "do": "ど", "ba": "ば", "bi": "び", "bu": "ぶ", "be": "べ", "bo": "ぼ", "pa": "ぱ", "pi": "ぴ", "pu": "ぷ",
    "pe": "ぺ", "po": "ぽ", "kya": "きゃ", "kyu": "きゅ", "kyo": "きょ", "sha": "しゃ", "shu": "しゅ", "sho": "しょ",
    "sya": "しゃ", "syu": "しゅ", "syo": "しょ", "cha": "ちゃ", "chu": "ちゅ", "cho": "ちょ", "tya": "ちゃ",
    "tyu": "ちゅ", "tyo": "ちょ", "nya": "にゃ", "nyu": "にゅ", "nyo": "にょ", "hya": "ひゃ", "hyu": "ひゅ",
    "hyo": "ひょ", "mya": "みゃ", "myu": "みゅ", "myo": "みょ", "rya": "りゃ", "ryu": "りゅ", "ryo": "りょ",
    "gya": "ぎゃ", "gyu": "ぎゅ", "gyo": "ぎょ", "ja": "じゃ", "ju": "じゅ", "jo": "じょ", "jya": "じゃ",
    "jyu": "じゅ", "jyo": "じょ", "bya": "びゃ", "byu": "びゅ", "byo": "びょ", "pya": "ぴゃ", "pyu": "ぴゅ",
    "pyo": "ぴょ", "fa": "ふぁ", "fi": "ふぃ", "fe": "ふぇ", "fo": "ふぉ", "va": "ゔぁ", "vi": "ゔぃ", "vu": "ゔ",
    "ve": "ゔぇ", "vo": "ゔぉ", "xtu": "っ", "ltu": "っ", "xa": "ぁ", "xi": "ぃ", "xu": "ぅ", "xe": "ぇ", "xo": "ぉ",
    "xya": "ゃ", "xyu": "ゅ", "xyo": "ょ", "-": "ー",
}


def romaji_to_kana(text: str) -> str:
    s = text.lower()
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        if not ("a" <= ch <= "z" or ch in "-'"):
            out.append(text[i])
            i += 1
            continue
        if ch == "n":
            nxt = s[i + 1] if i + 1 < len(s) else ""
            nxt2 = s[i + 2] if i + 2 < len(s) else ""
            if nxt == "'":
                out.append("ん")
                i += 2
                continue
            if nxt == "n":
                out.append("ん")
                i += 1 if nxt2 in tuple("aiueoy") and nxt2 else 2
                continue
            if nxt not in tuple("aiueoy") or not nxt:
                out.append("ん")
                i += 1
                continue
        if i + 1 < len(s) and ch == s[i + 1] and ch not in "aiueon":
            out.append("っ")
            i += 1
            continue
        for L in (3, 2, 1):
            seg = s[i : i + L]
            if seg in _ROMAJI:
                out.append(_ROMAJI[seg])
                i += L
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def normalize_reading(raw: str) -> str:
    r = raw.strip().replace(" ", "").replace("　", "")
    katakana = False
    m = re.match(r"^[Kk][:：](.*)$", r)
    if m:
        katakana, r = True, m.group(1)
    if re.search(r"[A-Za-z]", r):
        r = romaji_to_kana(r)
    return hira_to_kata(r) if katakana else r


# ------------------------------------------------------------------ parse / format
def parse_record(text: str) -> list[RecEntry]:
    entries: list[RecEntry] = []
    matches = list(_ENTRY.finditer(text or ""))
    for k, m in enumerate(matches):
        flags, surface, body = m.group(1), m.group(2), m.group(3)
        scale = None
        mm = re.match(r"^(.*?)[,，]\s*(\d+(?:\.\d+)?)\s*$", body)
        if mm:
            body, scale = mm.group(1), float(mm.group(2))
        e = RecEntry(surface=surface, reading="", scale=scale)
        e.uncertain = any(f in "?？" for f in flags)
        e.forced = any(f in "!！" for f in flags)
        e.from_dict = any(f in "*＊" for f in flags)
        body = body.strip()
        if body in ("-", "－", "ー", ""):
            e.hidden = True
        else:
            alts = [normalize_reading(x) for x in re.split(r"[/／|｜]", body) if x.strip()]
            e.reading = alts[0] if alts else ""
            e.alternatives = alts[1:]
        if k + 1 < len(matches):
            between = text[m.end() : matches[k + 1].start()]
            e.joined_with_next = between.strip() == "+"
            e.adjacent_to_next = between == ""
        entries.append(e)
    # merge "+" joins into single entries
    merged: list[RecEntry] = []
    for e in entries:
        if merged and merged[-1].joined_with_next:
            prev = merged[-1]
            prev.surface += e.surface
            prev.reading += e.reading
            prev.joined_with_next = e.joined_with_next
            prev.adjacent_to_next = e.adjacent_to_next
            prev.scale = None
            continue
        merged.append(e)
    return merged


def format_entry(surface: str, reading: str, *, alternatives=(), uncertain=False, forced=False, from_dict=False,
                 hidden=False, scale: Optional[float] = None) -> str:
    prefix = ("!" if forced else "") + ("*" if from_dict and not forced else "") + ("?" if uncertain and not forced else "")
    body = "-" if hidden else "/".join([reading] + [a for a in alternatives if a and a != reading])
    if scale is not None and not hidden:
        body += f",{scale:g}"
    return f"{prefix}{surface}（{body}）"
