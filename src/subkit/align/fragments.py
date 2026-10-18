"""Split cleaned JP cues into alignment units (phrases).

Japanese captions mark phrase boundaries with spaces ("大体さあ  数学なんて九九が言えりゃ
それでいいのよ").  Chinese fansubs regularly cut lines exactly there, so a
phrase is the natural unit: a CN row receives a contiguous run of phrases.
Each unit gets a time span proportional to its length inside its cue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..languages import get_language
from ..sources.jp import JpCue

_SENT_END = "。！？!?…‥"
_INTERJ_CORE = set("あいうえおぁぃぅぇぉんアイウエオァィゥェォンはひふへほハヒフヘホわワやヤよヨねネむムぬヌっッーゃゅょャュョゎ")
_INTERJ_WORDS = get_language("ja").filler_words
_PUNCT_STRIP = re.compile(r"[\s　、。，,.!！?？…‥・～〜~ー－\-―「」『』（）()]+")


# Response words are usually translated (はい→是, うん→嗯, そう→是嗎) — treat as content.
_RESPONSE_WORDS = {"はい", "うん", "ええ", "いいえ", "いや", "そう", "そうか", "そうね", "そうよ", "あっそ", "ああそう", "よし", "おい", "ねえ", "ほら", "さあ"}
_FILLER_KANA = re.compile(r"^[ぁ-ゖァ-ヺーっッ゛゜]+$")
# short words that are often drawn out but still mean something (translated)
_ELONGATED_WORDS = {"やだ", "いや", "だめ", "もう", "まだ", "うそ", "ほら", "ねえ", "なに", "なん", "どこ", "だれ", "すご", "ずる",
                    "よし", "はい", "ほう", "へえ", "そう", "あれ", "これ", "それ", "どれ", "いい", "ない", "たす", "まて", "やめ", "おい"}


def is_interjection(text: str) -> bool:
    """Exclamations, laughter and sound words a Chinese translation commonly leaves out
    (あっ, えっ？, フフフ…, オーホッホッホ！, チュッ, はぁ…, へ〜え)."""
    core = _PUNCT_STRIP.sub("", text)
    if not core:
        return True
    from ..textutil import kata_to_hira

    h = kata_to_hira(core)
    elongated = re.search(r"[～〜ー]", text) is not None
    if h in _RESPONSE_WORDS:
        # うん / ええ answer (translated), but ええ～… / うん？ are exclamations
        return h in ("うん", "ええ") and (elongated or "？" in text or "?" in text)
    if h in _INTERJ_WORDS:
        return True
    if not _FILLER_KANA.match(core):
        return False
    if all(ch in _INTERJ_CORE for ch in h):
        # あっ, ええ…, はあ？, ふ〜ん, ウフフフ, オーホッホッホ
        return len(h) <= 10
    # short sound words ending in っ/ー: チュッ, ふげっ, ぎゃー (but not まさか, そんな, だが)
    stripped = h.replace("っ", "").replace("ー", "")
    if len(stripped) <= 2 and (h.endswith("っ") or h.endswith("ー")):
        return True
    # drawn-out cries: もがぁ～～！, ひゃぁ (not the words やだぁ, もう～, だめぇ)
    bare = re.sub("[ぁぃぅぇぉっー]", "", h)
    if len(bare) <= 2 and bare not in _ELONGATED_WORDS and (elongated or h[-1] in "ぁぃぅぇぉ"):
        return True
    # katakana sound effects the captions write without brackets: ジュー！ ダン！ ゴン！ ガシャン プツ ツーツー ガラガラ
    if all("ァ" <= c <= "ヺ" or c in "ーッ" for c in core):
        if len(stripped) <= 2 and (text.rstrip().endswith(("！", "!", "…")) or core.endswith(("ン", "ッ", "ー"))):
            return True
        if _is_repetition(core):
            return True  # reduplication: ガラガラ, ツーツー, コツコツコツ, ガシャンガシャン
    return False


def _is_repetition(core: str) -> bool:
    n = len(core)
    return any(n % k == 0 and n // k >= 2 and core == core[:k] * (n // k) for k in range(1, min(5, n // 2) + 1))


def _katakana_core(text: str) -> str:
    core = _PUNCT_STRIP.sub("", text)
    return core if core and all("ァ" <= c <= "ヺ" or c in "ーッ" for c in core) else ""


@dataclass
class Unit:
    idx: int
    cue: int            # cue index (1-based)
    line: int           # line index inside the cue
    start: int          # cs, original JP timeline
    end: int
    text: str
    interj: bool = False
    speaker: str | None = None
    wstart: float = 0.0  # warped to the CN timeline (seconds*100, float)
    wend: float = 0.0
    tags: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return max(1, len(_PUNCT_STRIP.sub("", self.text)))


def split_phrases(text: str) -> list[str]:
    """Split one caption line at spaces and sentence-final punctuation."""
    parts: list[str] = []
    for chunk in re.split(r"[ 　]+", text.strip()):
        if not chunk:
            continue
        buf = ""
        i = 0
        while i < len(chunk):
            buf += chunk[i]
            if chunk[i] in _SENT_END:
                # keep runs like "！？" / "……" together
                while i + 1 < len(chunk) and chunk[i + 1] in _SENT_END:
                    i += 1
                    buf += chunk[i]
                if i + 1 < len(chunk):
                    parts.append(buf)
                    buf = ""
            i += 1
        if buf:
            parts.append(buf)
    # keep repeated sound words together: コツ コツ / ガシャン ガシャン！
    joined: list[str] = []
    for p in parts:
        k = _katakana_core(p)
        if joined and k and _katakana_core(joined[-1]) and _is_repetition(_katakana_core(joined[-1]) + k):
            joined[-1] = joined[-1] + " " + p
        else:
            joined.append(p)
    parts = joined
    # glue orphaned 1-char pieces (particles split off by a stray space) to the previous piece
    merged: list[str] = []
    for p in parts:
        core = _PUNCT_STRIP.sub("", p)
        if (merged and len(core) <= 1 and not is_interjection(p) and not any(c in _SENT_END for c in p)
                and merged[-1][-1:] not in _SENT_END):
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


def build_units(cues: list[JpCue]) -> list[Unit]:
    units: list[Unit] = []
    for cue in cues:
        lines = cue.speech_lines
        if not lines:
            continue
        pieces: list[tuple[int, str, str | None]] = []
        for li, line in enumerate(lines):
            for p in split_phrases(line.text):
                pieces.append((li, p, line.speaker))
        if not pieces:
            continue
        weights = [max(1, len(_PUNCT_STRIP.sub("", p))) + 1 for _, p, _ in pieces]
        total = sum(weights)
        dur = max(10, cue.end - cue.start)
        t = float(cue.start)
        first = len(units)
        for (li, p, spk), w in zip(pieces, weights):
            seg = dur * w / total
            u = Unit(len(units), cue.idx, li, int(round(t)), int(round(t + seg)), p, is_interjection(p), spk)
            units.append(u)
            t += seg
        # A filler that shares its caption with real text stays with it (users keep
        # ~90% of those); a filler alone in its caption is usually left untranslated.
        has_content = any(not u.interj for u in units[first:])
        for u in units[first:]:
            u.tags["shares_cue"] = has_content
    return units
