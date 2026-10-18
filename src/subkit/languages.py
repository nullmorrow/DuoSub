"""Language data shared by the core and UI, without Qt dependencies."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from typing import Literal, Mapping


@dataclass(frozen=True)
class Language:
    code: str
    display_name: str
    word_spaces: bool
    sentence_punctuation: str
    phrase_punctuation: str
    filler_words: frozenset[str]
    analyzer_id: Literal["ja-cc", "en-sdh", "generic"] = "generic"
    annotation_system: Literal["ruby", "pinyin", "none"] = "none"
    name_order: Literal["family_first", "given_first"] = "given_first"
    name_separator: str = " "


_JA_FILLERS = frozenset({
    "えっ", "あっ", "うん", "ううん", "はい", "ええ", "ああ", "おお", "うう", "えー", "あー", "おー", "ん", "うっ", "おっ",
    "ふふ", "うふふ", "あはは", "へえ", "へー", "ふーん", "ふん", "はぁ", "はあ", "ほう", "ほお", "わあ", "わっ", "きゃ",
    "きゃあ", "ひっ", "ひい", "うわ", "うわあ", "げっ", "あれ", "あら", "まあ", "おや", "ちっ", "くっ", "ぐっ", "やあ",
    "よし", "ほら", "ねえ", "なあ", "おい", "こら", "さあ", "いや", "ううっ", "あ", "え", "お", "う", "は", "ほ", "へ",
})
_ZH_FILLERS = frozenset("啊嗯哦呃欸唉哎哈呀喔噢嘿")
_JA_SENTENCE = "。！？!?…‥｡"
_ZH_PHRASE = "，。！？；、!?;,"

LANGUAGES = {
    "ja": Language("ja", "日语", False, _JA_SENTENCE, _JA_SENTENCE, _JA_FILLERS,
                   "ja-cc", "ruby", "family_first", ""),
    "zh-Hans": Language("zh-Hans", "简中", False, "。！？!?", _ZH_PHRASE, _ZH_FILLERS,
                        annotation_system="pinyin", name_order="family_first", name_separator=""),
    "zh-Hant": Language("zh-Hant", "繁中", False, "。！？!?", _ZH_PHRASE, _ZH_FILLERS,
                        annotation_system="pinyin", name_order="family_first", name_separator=""),
    "yue": Language("yue", "粤语", False, "。！？!?", _ZH_PHRASE, _ZH_FILLERS,
                    name_order="family_first", name_separator=""),
    "ko": Language("ko", "韩语", True, ".!?", ".!?;,", frozenset("아 어 음 네 응 에".split()),
                   name_order="family_first", name_separator=""),
    "en": Language("en", "英语", True, ".!?", ".!?;,", frozenset("uh um oh ah hmm huh er eh mm".split()),
                   analyzer_id="en-sdh"),
    "vi": Language("vi", "越南语", True, ".!?", ".!?;,", frozenset("à ừ ờ ơ ủa hả nhé".split()),
                   name_order="family_first"),
    "fr": Language("fr", "法语", True, ".!?", ".!?;,", frozenset("euh hem hein".split())),
    "de": Language("de", "德语", True, ".!?", ".!?;,", frozenset("äh ähm hm".split())),
    "es": Language("es", "西班牙语", True, ".!?", ".!?;,", frozenset("eh ah em".split())),
    "pt": Language("pt", "葡萄牙语", True, ".!?", ".!?;,", frozenset("ah eh hum".split())),
    "it": Language("it", "意大利语", True, ".!?", ".!?;,", frozenset("eh ah mah boh".split())),
    "ru": Language("ru", "俄语", True, ".!?", ".!?;,", frozenset("э эм хм".split())),
}
FALLBACK = Language("", "", True, ".!?", ".!?;,", frozenset())
LANGUAGE_NAMES = {code: language.display_name for code, language in LANGUAGES.items()}
DEFAULT_LANGUAGES = {"original": "ja", "translation": "zh-Hant"}
JAPANESE_ONLY = "只对日文有效"


def normalize_language_code(code: str) -> str:
    """Resolve regional variants to registry keys; retain canonical unknown codes."""
    parts = code.strip().replace("_", "-").split("-")
    base = parts[0].lower()
    subtags = [p.lower() for p in parts[1:]]
    if base == "zh":
        script = next((p for p in subtags if p in ("hans", "hant")), None)
        if script is None:
            script = "hant" if any(p in ("tw", "hk", "mo") for p in subtags) else "hans"
        return "zh-" + script.title()
    if base in LANGUAGES:
        return base
    return "-".join([base] + [p.title() if len(p) == 4 and p.isalpha() else
                              p.upper() if len(p) == 2 and p.isalpha() else p for p in subtags])


def get_language(code: str, text: str = "") -> Language:
    key = normalize_language_code(code)
    if key in LANGUAGES:
        return LANGUAGES[key]
    cjk = any(unicodedata.name(c, "").startswith(("CJK ", "HIRAGANA ", "KATAKANA ",
                                                 "HANGUL ", "HALFWIDTH KATAKANA ", "HALFWIDTH HANGUL "))
              for c in text)
    return replace(FALLBACK, code=key, display_name=key, word_spaces=not cjk,
                   sentence_punctuation="。！？!?" if cjk else FALLBACK.sentence_punctuation,
                   phrase_punctuation=_ZH_PHRASE + "." if cjk else FALLBACK.phrase_punctuation)


def language_name(code: str) -> str:
    return get_language(code).display_name


def lane_label(lane: str, languages: Mapping[str, str] | None = None) -> str:
    code = (languages or {}).get(lane, DEFAULT_LANGUAGES[lane])
    name = "原声" if lane == "original" else "翻译"
    return f"{name}（{language_name(code)}）"


def has_japanese(languages: Mapping[str, str]) -> bool:
    return any(get_language(languages.get(lane, DEFAULT_LANGUAGES[lane])).annotation_system == "ruby"
               for lane in DEFAULT_LANGUAGES)
