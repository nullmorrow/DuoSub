"""Display rules: user-chosen transformations applied only when rendering (preview / export).

The document is never changed by a display rule (REQUIREMENTS 二.2: derived
representations must not pretend to be changes of the original).  The export
report says how many characters a rule hid.

Rules
-----
``hide_final_period`` — 「。」/「｡」 at the end of a sentence in an original-language line
(followed by a space, a line/caption separator or the end of the line) is not shown.
「。」 inside quotes (「…。」) and in the middle of text stay.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import re
import unicodedata

from ..ass.text import tokenize
from ..languages import get_language
from ..model.document import Member, Run
from ..textutil import is_kana, is_kanji, normalize_halfwidth_kana
from .replacements import ReplacementRule, frozen_rules, replace_ass_text

PERIODS = "。｡"
SPACES = " 　\t"
PUNCTUATION_CATEGORIES = ("comma", "dunhao", "period", "colon", "ellipsis", "exclaim",
                          "question", "quotes", "brackets", "separator", "tilde", "dash", "long_vowel")
LEGACY_FONT_CATEGORIES = ("ellipsis", "exclaim", "question", "quotes")
TILDES = "~～〜∼"
DASHES = "-–—―─━‐‒－ー"
ELLIPSIS_PATTERN = r"(?:…+|[.．。・･]{3,})+"
ZH_SCOPE_DEFAULTS = dict(dialogue=True, song=True, sign=True, gloss=True, gloss_ja=False, raw=False)

MERGE_KINDS = (("dialogue", "对白"), ("song", "歌词"), ("other", "屏幕字等"))


@dataclass(frozen=True)
class PunctuationRules:
    comma: str = "off"
    dunhao: bool = False
    period: bool = False
    spaces: bool = False
    ellipsis: str = "off"
    exclaim: str = "off"
    question: str = "off"
    quotes: str = "off"
    # Retained for callers/the old font dialog until its replacement; never serialized.
    font: str | None = field(default=None, compare=False)
    separator: bool = False
    tilde: str = "off"
    dash: str = "off"
    fonts: tuple[tuple[str, str], ...] = ()
    scope_chars: tuple[str, ...] = ("dialogue",)
    scope_fonts: tuple[str, ...] = ("dialogue", "song", "other")
    halfwidth_kana: bool = False

    def __post_init__(self) -> None:
        for name in ("scope_chars", "scope_fonts"):
            scope = getattr(self, name)
            object.__setattr__(self, name, tuple(kind for kind, _label in MERGE_KINDS if kind in scope))
        fonts = dict(self.fonts)
        if not fonts and self.font:
            fonts = {category: self.font for category in LEGACY_FONT_CATEGORIES}
        object.__setattr__(self, "fonts", tuple((category, font.strip())
            for category in PUNCTUATION_CATEGORIES
            if isinstance(font := fonts.get(category), str) and font.strip()))

    def to_settings(self) -> dict:
        """Canonical lane settings; inherited fonts are explicit null values."""
        values = {name: getattr(self, name) for name in
                  ("comma", "dunhao", "period", "spaces", "separator", "ellipsis",
                   "exclaim", "question", "quotes", "tilde", "dash", "halfwidth_kana")}
        fonts = dict(self.fonts)
        values["fonts"] = {category: fonts.get(category) for category in PUNCTUATION_CATEGORIES}
        values["scope_chars"] = list(self.scope_chars)
        values["scope_fonts"] = list(self.scope_fonts)
        return values

    @classmethod
    def from_settings(cls, settings: dict) -> "PunctuationRules":
        values = {name: bool(settings.get(name, False)) for name in ("dunhao", "period", "spaces", "separator", "halfwidth_kana")}
        comma = settings.get("comma", "off")
        values["comma"] = "spaces" if comma is True else comma if comma in ("off", "spaces", "ideographic") else "off"
        if comma is True and "dunhao" not in settings:
            values["dunhao"] = True
        for name, choices in (("ellipsis", ("off", "single", "double", "keep_count")),
                              ("exclaim", ("off", "full", "half")), ("question", ("off", "full", "half")),
                              ("tilde", ("off", "full", "half", "wave")),
                              ("dash", ("off", "half", "em", "choon", "bar", "fullwidth_hyphen"))):
            value = settings.get(name, "off")
            values[name] = value if value in choices else "off"
        quotes = settings.get("quotes", "off")
        values["quotes"] = "nested" if quotes is True else quotes if quotes in ("off", "corner", "white_corner", "nested", "nested_inverse") else "off"
        if "fonts" in settings:
            fonts = settings["fonts"]
            values["fonts"] = tuple(fonts.items()) if isinstance(fonts, dict) else ()
        else:
            font = settings.get("font")
            values["font"] = font if isinstance(font, str) and font.strip() else None
        # Settings saved before the scopes existed applied fonts to dialogue only; keep their export unchanged.
        # New (empty) settings get the display-font default of every kind.
        legacy = bool(settings) and not any(key in settings for key in ("fonts", "scope_fonts", "scope_chars", "tilde", "dash", "halfwidth_kana"))
        for name, default in (("scope_chars", ("dialogue",)),
                              ("scope_fonts", ("dialogue",) if legacy else ("dialogue", "song", "other"))):
            scope = settings.get(name, default)
            values[name] = tuple(scope) if isinstance(scope, (list, tuple)) else default
        return cls(**values)

    @property
    def active(self) -> bool:
        return (self.comma != "off" or self.dunhao or self.period or self.spaces or self.separator or self.halfwidth_kana or self.quotes != "off" or bool(self.fonts)
                or any(value != "off" for value in (self.ellipsis, self.exclaim, self.question, self.tilde, self.dash)))


@dataclass
class DisplayRules:
    hide_final_period: bool = False
    zh_traditional_enabled: bool = False
    zh_traditional_variant: str = "tw"
    zh_traditional_phrases: bool = False
    merge_lines: tuple[tuple[str, str], ...] = ()
    punctuation: tuple[tuple[str, PunctuationRules], ...] = ()
    zh_script: str | None = None  # Accept the legacy constructor flag, then normalize the direction.

    zh_scope: dict[str, bool] = field(default_factory=lambda: dict(ZH_SCOPE_DEFAULTS))
    replacements: tuple[ReplacementRule, ...] = ()

    def __post_init__(self) -> None:
        self.zh_scope = {key: bool(self.zh_scope.get(key, default)) for key, default in ZH_SCOPE_DEFAULTS.items()}
        self.replacements = frozen_rules(self.replacements)
        if self.zh_script is None:
            self.zh_script = "traditional" if self.zh_traditional_enabled else "off"
        if self.zh_script not in ("off", "simplified", "traditional"):
            self.zh_script = "off"
        self.zh_traditional_enabled = self.zh_script == "traditional"

    @classmethod
    def from_settings(cls, settings: dict) -> "DisplayRules":
        d = (settings or {}).get("display", {})
        zh = d.get("zh_traditional", {})
        variant = zh.get("variant", "tw")
        direction = d.get("zh_script", "traditional" if zh.get("enabled", False) else "off")
        direction = direction if direction in ("off", "simplified", "traditional") else "off"
        merge = d.get("merge_lines", {})
        enabled = tuple((lane, kind) for lane in ("translation", "original")
                        for kind, _label in MERGE_KINDS if merge.get(lane, {}).get(kind, False))
        return cls(bool(d.get("hide_final_period", False)), direction == "traditional",
                   variant if variant in ("tw", "hk", "std") else "tw", bool(zh.get("phrases", False)), enabled,
                   tuple((lane, rule) for lane in ("translation", "original")
                         if (rule := PunctuationRules.from_settings(d.get("punctuation", {}).get(lane, {}))).active),
                   zh_script=direction, zh_scope=d.get("zh_scope", {}), replacements=d.get("replacements", ()))

    @property
    def active(self) -> bool:
        return (self.hide_final_period or self.direction != "off" or bool(self.merge_lines)
                or bool(self.punctuation) or any(r.enabled for r in self.replacements))

    def replaced_for(self, m: Member) -> bool:
        return any(r.enabled and r.lane in (m.lane, "both") for r in self.replacements)

    @staticmethod
    def kind_for(m: Member) -> str:
        return ("dialogue" if m.role in ("dialogue", "top", "alt") else
                "song" if m.role in ("song", "song_jp", "song_cn") else "other")

    def punctuation_layers_for(self, m: Member) -> tuple[PunctuationRules, tuple[tuple[str, str], ...]]:
        """Independent character rules and display fonts active for the member's kind."""
        rule = dict(self.punctuation).get(m.lane, PunctuationRules())
        kind = self.kind_for(m)
        chars = replace(rule, font=None, fonts=()) if kind in rule.scope_chars else PunctuationRules()
        return chars, rule.fonts if kind in rule.scope_fonts else ()

    def merge_for(self, m: Member) -> bool:
        return (m.lane, self.kind_for(m)) in self.merge_lines

    @property
    def direction(self) -> str:
        return self.zh_script

    def converted_for(self, m: Member) -> bool:
        kind = {"other": "sign"}.get(self.kind_for(m), self.kind_for(m))
        return (self.direction != "off" and get_language(m.lang).code.startswith("zh")
                and self.zh_scope[kind])

    def gloss_for(self, m: Member, note: str) -> bool:
        from ..gloss import contains_kana

        language = get_language(m.lang).code
        return self.direction != "off" and (
            language.startswith("zh") and self.zh_scope["gloss"]
            or language == "ja" and self.zh_scope["gloss_ja"] and not contains_kana(note))

    def traditional_for(self, m: Member) -> bool:
        """Compatibility alias for callers using the former one-way rule."""
        return self.converted_for(m)

    @property
    def script_label(self) -> str:
        return {"off": "原样（不转换）", "simplified": "显示为简体", "traditional": "显示为繁体"}[self.direction]

    @property
    def glyph_config(self) -> str:
        return ({"tw": "tw2s", "hk": "hk2s", "std": "t2s"} if self.direction == "simplified" else
                {"tw": "s2tw", "hk": "s2hk", "std": "s2t"})[self.zh_traditional_variant]

    @property
    def traditional_label(self) -> str:
        return {"tw": "台湾用字", "hk": "香港用字", "std": "标准繁体"}[self.zh_traditional_variant]

    def convert(self, text: str) -> str:
        if self.direction == "off":
            return text
        if self.direction == "simplified":
            config = "tw2sp" if self.zh_traditional_variant == "tw" and self.zh_traditional_phrases else self.glyph_config
            return _converter(config).convert(text)
        config = "s2twp" if self.zh_traditional_phrases else {
            "tw": "s2tw", "hk": "s2hk", "std": "s2t"}[self.zh_traditional_variant]
        converted = _converter(config).convert(text)
        # Phrase vocabulary is supplied by s2twp; the chosen regional glyphs still apply.
        if self.zh_traditional_phrases and self.zh_traditional_variant != "tw":
            converted = _converter("tw2s").convert(converted)
            converted = _converter({"hk": "s2hk", "std": "s2t"}[self.zh_traditional_variant]).convert(converted)
        return converted


@lru_cache(maxsize=8)
def _converter(config: str):
    from opencc import OpenCC

    return OpenCC(config)


def display_text(text: str, rules: DisplayRules) -> tuple[str, list[int]]:
    """Convert visible tokens only, with original-visible → displayed-visible boundaries.

    Equal-length glyph changes keep exact positions. Phrase expansions/contractions
    share their displayed span between the original characters without changing anchors.
    """
    out, bounds = [], [0]
    displayed = 0
    for tok in tokenize(text):
        converted = rules.convert(tok.raw) if tok.kind == "text" and (not tok.protected or rules.zh_scope["raw"]) else tok.raw
        out.append(converted)
        if tok.kind == "text":
            if tok.protected and not rules.zh_scope["raw"]:
                bounds.extend(displayed + i for i in range(1, len(tok.raw) + 1))
                displayed += len(tok.raw)
                continue
            local = [0] * (len(tok.raw) + 1)
            # Align against the glyph-only result so unchanged phrases remain exact.
            glyphs = _converter(rules.glyph_config).convert(tok.raw) if rules.direction != "off" else tok.raw
            prefix = 0
            while prefix < min(len(glyphs), len(converted)) and glyphs[prefix] == converted[prefix]:
                prefix += 1
            suffix = 0
            while (suffix < min(len(glyphs), len(converted)) - prefix
                   and glyphs[-suffix - 1] == converted[-suffix - 1]):
                suffix += 1
            a, b, c, d = prefix, len(glyphs) - suffix, prefix, len(converted) - suffix
            for i in range(len(local)):
                local[i] = (i if i < a else i + len(converted) - len(glyphs) if i > b else
                            c + ((i - a) * (d - c) + (b - a) // 2) // max(1, b - a))
            bounds.extend(displayed + p for p in local[1:])
            displayed += len(converted)
        elif tok.kind in ("hardspace", "softbreak"):
            displayed += 1
            bounds.append(displayed)
    return "".join(out), bounds


# Retained for integrations using the former one-way name.
traditional_text = display_text


def member_display_text(m: Member, rules: DisplayRules) -> tuple[str, list[int]]:
    """Body display for review text, retaining source-boundary correspondence."""
    result = replace_ass_text(m.text, rules.replacements, lane=m.lane)
    text, bounds = result.text, result.bounds
    if rules.converted_for(m):
        text, converted = display_text(text, rules)
        bounds = [converted[i] for i in bounds]
    return text, bounds


def single_line_text(text: str) -> tuple[str, list[int], list[int]]:
    """Hard breaks become identity-free spaces; map glyph starts and ends separately.

    A break has no visible identity. The preceding glyph ends before the spaces,
    while the following glyph starts after them (including leading/trailing breaks).
    """
    out, starts, ends = [], [0], [0]
    displayed = 0
    for tok in tokenize(text):
        if tok.kind == "break" and not tok.protected:
            out.append("  ")
            displayed += 2
            starts[-1] = displayed
            continue
        out.append(tok.raw)
        count = len(tok.raw) if tok.kind == "text" else int(tok.kind in ("hardspace", "softbreak"))
        for _ in range(count):
            displayed += 1
            starts.append(displayed)
            ends.append(displayed)
    return "".join(out), starts, ends


@dataclass
class _Glyph:
    raw: str
    ch: str
    origin: int | None
    prefix: str = ""
    punctuation: bool = False
    protected: bool = False
    category: str | None = None
    covered: tuple[int, ...] = ()          # original characters sharing a composed display glyph


def punctuation_text(text: str, rules: PunctuationRules, *, hide_final_period: bool = False
                     ) -> tuple[str, list[int], set[int], int]:
    """Compatible result: text, boundaries, transformed punctuation positions, hidden count."""
    shown, bounds, changed, _categories, hidden, _ends = _punctuation_text(text, rules, hide_final_period)
    return shown, bounds, changed, hidden


def punctuation_text_categories(text: str, rules: PunctuationRules, *, hide_final_period: bool = False
                                ) -> tuple[str, list[int], dict[str, set[int]], int]:
    """Display text and positions of every category, including unchanged punctuation."""
    shown, bounds, _changed, categories, hidden, _ends = _punctuation_text(text, rules, hide_final_period)
    return shown, bounds, categories, hidden


def punctuation_layout(text: str, rules: PunctuationRules, *, hide_final_period: bool = False
                       ) -> tuple[str, list[int], list[int], dict[str, set[int]], int]:
    """Separate start/end maps let both ｶ and ﾞ refer to the displayed ガ."""
    shown, starts, _changed, categories, hidden, ends = _punctuation_text(text, rules, hide_final_period)
    return shown, starts, ends, categories, hidden


def _word(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _long_vowel_positions(glyphs: list[_Glyph]) -> set[int]:
    out: set[int] = set()
    for i, g in enumerate(glyphs):
        if g.ch in "ーｰ" and g.category == "long_vowel":
            out.add(i)
        elif g.ch in "ーｰ" and i:
            prev = glyphs[i - 1].ch
            if (prev not in "ーｰ" and (is_kana(prev) or "ｦ" <= prev <= "ﾟ")) or i - 1 in out:
                out.add(i)
    return out


def _dash_positions(glyphs: list[_Glyph]) -> set[int]:
    # Compute before replacement: a dash changed to ー must not protect the next dash.
    long_vowels = _long_vowel_positions(glyphs)
    out: set[int] = set()
    for i, g in enumerate(glyphs):
        if g.protected or g.ch not in DASHES or i in long_vowels:
            continue
        prev = glyphs[i - 1].ch if i else ""
        nxt = glyphs[i + 1].ch if i + 1 < len(glyphs) else ""
        # Also keep a trailing English prefix hyphen (Re-) as specified.
        if g.ch == "-" and _word(prev) and (_word(nxt) or not nxt or nxt.isspace()):
            continue
        out.add(i)
    return out


def _punctuation_categories(glyphs: list[_Glyph]) -> dict[str, set[int]]:
    plain = "".join("\0" if g.protected else g.ch for g in glyphs)
    ellipses = {i for match in re.finditer(ELLIPSIS_PATTERN, plain)
                for i in range(match.start(), match.end())}
    abbreviations = {i for match in re.finditer(r"(?:[A-Za-zＡ-Ｚａ-ｚ][.．]){2,}", plain)
                     for i in range(match.start(), match.end()) if plain[i] in ".．"}
    long_vowels = _long_vowel_positions(glyphs)
    dashes = _dash_positions(glyphs)
    categories = {category: set() for category in PUNCTUATION_CATEGORIES}
    visible = 0
    for i, g in enumerate(glyphs):
        if g.origin is None:
            continue
        at, visible = visible, visible + 1
        if g.protected:
            continue
        prev = glyphs[i - 1].ch if i else ""
        nxt = glyphs[i + 1].ch if i + 1 < len(glyphs) else ""
        category = g.category
        if category is None:
            if i in ellipses:
                category = "ellipsis"
            elif g.ch in "·‧•･⋅∙・" or (g.ch in "．." and _cjk(prev) and _cjk(nxt)):
                category = "separator"
            elif g.ch in "。．":
                if not (i in abbreviations or _inside_word(prev, nxt)):
                    category = "period"
            elif g.ch in "，,":
                after = "".join(x.ch for x in glyphs[i + 1:i + 5])
                if not (prev.isdigit() and len(after) >= 3 and after[:3].isdigit() and not after[3:4].isdigit()):
                    category = "comma"
            elif g.ch == "、":
                category = "dunhao"
            elif g.ch in "：:；;":
                category = "colon"
            elif g.ch in "！!":
                category = "exclaim"
            elif g.ch in "？?":
                category = "question"
            elif g.ch in "「」『』“”‘’\"'":
                if not (g.ch in "\"'" and _word(prev) and _word(nxt)):
                    category = "quotes"
            elif g.ch in "（）()【】《》〈〉［］[]":
                category = "brackets"
            elif g.ch in TILDES:
                category = "tilde"
            elif i in long_vowels:
                category = "long_vowel"
            elif i in dashes:
                category = "dash"
        if category:
            categories[category].add(at)
    return categories


def _latin(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def _inside_word(prev: str, nxt: str) -> bool:
    """A full-width dot between Latin letters or digits (３．５, Ｌｖ．５), not a sentence period."""
    def alnum(ch: str) -> bool:
        return ch.isdigit() or _latin(ch) or "Ａ" <= ch <= "Ｚ" or "ａ" <= ch <= "ｚ"
    return alnum(prev) and alnum(nxt)


def _cjk(ch: str) -> bool:
    return is_kanji(ch) or is_kana(ch) or "ｦ" <= ch <= "ﾝ"


def _fullwidth_kana(glyphs: list[_Glyph], tail: str) -> str:
    """Widen only kana; preserve base styling and keep intervening tags after it."""
    i = 0
    while i < len(glyphs):
        g = glyphs[i]
        if g.protected or not "ｦ" <= g.ch <= "ﾟ":
            i += 1
            continue
        if g.ch in "ﾞﾟ":
            mark = "\u3099" if g.ch == "ﾞ" else "\u309a"
            prev = glyphs[i-1] if i else None
            combined = unicodedata.normalize("NFC", prev.ch + mark) if prev and is_kana(prev.ch) else ""
            if prev and not prev.protected and prev.origin is not None and len(combined) == 1:
                prev.ch = prev.raw = combined
                prev.covered = (prev.covered or (prev.origin,)) + (g.origin,)
                if i+1 < len(glyphs):
                    glyphs[i+1].prefix = g.prefix + glyphs[i+1].prefix
                else:
                    tail = g.prefix + tail
                del glyphs[i]
                continue
            # Unattached/invalid voicing marks stay visible as spacing marks.
            g.raw = g.ch = "゛" if g.ch == "ﾞ" else "゜"
        else:
            if g.ch == "ｰ":
                g.category = "long_vowel"
            g.raw = g.ch = normalize_halfwidth_kana(g.ch)
        i += 1
    return tail


def _punctuation_text(text: str, rules: PunctuationRules, hide_final_period: bool):
    """Transform visible glyphs, retaining source boundaries and untouched ASS blocks."""
    glyphs: list[_Glyph] = []
    prefix, count = "", 0
    for tok in tokenize(text):
        if tok.kind == "text":
            for ch in tok.raw:
                glyphs.append(_Glyph(ch, ch, count, prefix, protected=tok.protected))
                count += 1
                prefix = ""
        elif tok.kind in ("hardspace", "softbreak"):
            glyphs.append(_Glyph(tok.raw, " " if tok.kind == "hardspace" else "\v", count, prefix, protected=tok.protected))
            count += 1
            prefix = ""
        elif tok.kind in ("break", "drawing"):
            glyphs.append(_Glyph(tok.raw, "\n" if tok.kind == "break" else "\0", None, prefix, protected=tok.protected))
            prefix = ""
        else:
            prefix += tok.raw
    tail = prefix

    def replace(a: int, b: int, value: str, *, punctuation: bool = False) -> None:
        nonlocal tail
        old = glyphs[a:b]
        tags = "".join(g.prefix for g in old)
        origins = [g.origin for g in old if g.origin is not None]
        new = [_Glyph(ch, ch, origins[k * len(origins) // len(value)],
                      tags if k == 0 else "", punctuation) for k, ch in enumerate(value)]
        if not new:
            if b < len(glyphs):
                glyphs[b].prefix = tags + glyphs[b].prefix
            else:
                tail = tags + tail
        glyphs[a:b] = new

    if rules.halfwidth_kana:
        tail = _fullwidth_kana(glyphs, tail)

    if rules.ellipsis != "off":
        plain = "".join("\0" if g.protected else g.ch for g in glyphs)
        for match in reversed(list(re.finditer(ELLIPSIS_PATTERN, plain))):
            raw = match.group()
            units = raw.count("…") + (len(raw) - raw.count("…") + 2) // 3
            size = 1 if rules.ellipsis == "single" else 2 if rules.ellipsis == "double" else units
            replace(match.start(), match.end(), "…" * size, punctuation=True)
    plain = "".join(g.ch for g in glyphs)
    abbreviations = {i for match in re.finditer(r"(?:[A-Za-zＡ-Ｚａ-ｚ][.．]){2,}", plain)
                     for i in range(match.start(), match.end()) if plain[i] in ".．"}
    for i in reversed(range(len(glyphs))):
        g = glyphs[i]
        if g.protected:
            continue
        prev = glyphs[i - 1].ch if i else ""
        nxt = glyphs[i + 1].ch if i + 1 < len(glyphs) else ""
        if rules.separator and (g.ch in "·‧•･⋅∙・" or (g.ch in "．." and _cjk(prev) and _cjk(nxt))):
            g.raw = g.ch = "・"
        elif rules.period and g.ch in "。．":
            # A half-width "." belongs to English text and is never a sentence period here.
            if i in abbreviations or _inside_word(prev, nxt):
                continue
            replace(i, i + 1, "  ")
        elif rules.comma != "off" and g.ch in "，,":
            # A thousands separator: a digit before, exactly three digits after (1,000 / ２，０００).
            after = "".join(x.ch for x in glyphs[i + 1:i + 5])
            if prev.isdigit() and after[:3].isdigit() and len(after) >= 3 and not after[3:4].isdigit():
                continue
            replace(i, i + 1, "、" if rules.comma == "ideographic" else "  ")
    if rules.dunhao:
        for i in reversed(range(len(glyphs))):
            if not glyphs[i].protected and glyphs[i].ch == "、":
                replace(i, i + 1, "  ")
    if rules.spaces:
        i = 0
        while i < len(glyphs):
            if glyphs[i].protected or glyphs[i].ch not in SPACES:
                i += 1
                continue
            j = i + 1
            while j < len(glyphs) and not glyphs[j].protected and glyphs[j].ch in SPACES:
                j += 1
            if not (j == i + 1 and glyphs[i].raw == " " and i > 0 and j < len(glyphs)
                    and _word(glyphs[i - 1].ch) and _word(glyphs[j].ch)):
                replace(i, j, "  ")
                j = i + 2
            i = j
    if rules.comma != "off" or rules.dunhao or rules.period or rules.spaces:
        plain = "".join(g.ch for g in glyphs)
        for match in reversed(list(re.finditer(r"(?:(?<=\n)|^)\s+|\s+(?=\n|$)", plain))):
            # Hard breaks and soft breaks are structural, not removable spaces.
            for i in reversed(range(match.start(), match.end())):
                if not glyphs[i].protected and glyphs[i].ch in SPACES:
                    replace(i, i + 1, "")
    if rules.tilde != "off":
        value = {"full": "～", "half": "~", "wave": "〜"}[rules.tilde]
        for g in glyphs:
            if not g.protected and g.ch in TILDES:
                g.raw = g.ch = value
    if rules.dash != "off":
        value = {"half": "-", "em": "—", "choon": "ー", "bar": "―",
                 "fullwidth_hyphen": "－"}[rules.dash]
        for i in _dash_positions(glyphs):
            glyphs[i].raw = glyphs[i].ch = value
            glyphs[i].category = "dash"
    for g in glyphs:
        if g.protected:
            continue
        for enabled, chars, full, half in ((rules.exclaim, "!！", "！", "!"),
                                           (rules.question, "?？", "？", "?")):
            if enabled != "off" and g.ch in chars:
                g.raw = g.ch = full if enabled == "full" else half
                g.punctuation = True
    if rules.quotes != "off":
        stack, pairs = [], []
        opening, closing = "“‘「『", "”’」』"
        for i, g in enumerate(glyphs):
            if g.protected:
                continue
            ch = g.ch
            if ch in "\"'" and i and i + 1 < len(glyphs) and _word(glyphs[i - 1].ch) and _word(glyphs[i + 1].ch):
                continue
            if ch in closing:
                if stack:
                    pairs.append((stack.pop(), i))
            elif ch in opening:
                stack.append(i)
            elif ch in "\"'":
                if stack and glyphs[stack[-1]].ch == ch:
                    pairs.append((stack.pop(), i))
                else:
                    stack.append(i)
        for a, b in pairs:
            depth = sum(c < a and b < d for c, d in pairs)
            white = (rules.quotes == "white_corner"
                     or rules.quotes == "nested" and depth % 2 == 1
                     or rules.quotes == "nested_inverse" and depth % 2 == 0)
            left, right = ("『", "』") if white else ("「", "」")
            for i, value in ((a, left), (b, right)):
                glyphs[i].ch = glyphs[i].raw = value
                glyphs[i].punctuation = True
    hidden = 0
    if hide_final_period:
        for i in reversed(range(len(glyphs))):
            if (not glyphs[i].protected and glyphs[i].ch in PERIODS and (i + 1 == len(glyphs)
                    or glyphs[i + 1].ch in SPACES + "\n")):
                replace(i, i + 1, "")
                hidden += 1
    counts = [0] * count
    fonts: set[int] = set()
    visible = 0
    for g in glyphs:
        if g.origin is not None:
            counts[g.origin] += 1
            if g.punctuation:
                fonts.add(visible)
            visible += 1
    bounds = [0]
    for n in counts:
        bounds.append(bounds[-1] + n)
    ends = list(bounds)
    for at, g in enumerate(g for g in glyphs if g.origin is not None):
        for origin in g.covered:
            bounds[origin] = at
            ends[origin+1] = at+1
    return "".join(g.prefix + g.raw for g in glyphs) + tail, bounds, fonts, _punctuation_categories(glyphs), hidden, ends


def punctuation_font(text: str, positions: set[int], font: str | None, style, styles: dict) -> str:
    """Scope a single font using the same state restoration as character formatting."""
    return _punctuation_fonts(text, {i: font for i in positions} if font else {}, style, styles)


def punctuation_fonts(text: str, categories: dict[str, set[int]], fonts: tuple[tuple[str, str], ...],
                      style, styles: dict) -> str:
    """Apply category fonts together so restoration never inherits another inserted font."""
    positions = {i: font for category, font in fonts for i in categories.get(category, ())}
    return _punctuation_fonts(text, positions, style, styles)


def _punctuation_fonts(text: str, positions: dict[int, str], style, styles: dict) -> str:
    if not positions:
        return text
    from ..ass.state import restore_tags
    from ..ass.text import insert_at_visible

    at, current = 0, style
    style_at, boundaries = {}, set()
    for tok in tokenize(text):
        if tok.kind == "tags":
            for tag in tok.tags:
                if tag.name in ("r", "fn"):
                    boundaries.add(at)
                if tag.name == "r":
                    current = styles.get(tag.arg, style) if tag.arg else style
        elif tok.kind in ("text", "hardspace", "softbreak"):
            size = len(tok.raw) if tok.kind == "text" else 1
            for i in range(at, at + size):
                style_at[i] = current
            at += size
    style_at[at] = current
    inserts: dict[int, str] = {}
    ordered = sorted(positions)
    k = 0
    while k < len(ordered):
        start = end = ordered[k]
        font = positions[start]
        k += 1
        while (k < len(ordered) and ordered[k] == end + 1 and positions[ordered[k]] == font
               and ordered[k] not in boundaries):
            end = ordered[k]
            k += 1
        inserts[start] = inserts.get(start, "") + "{\\fn" + font + "}"
        end += 1
        inserts[end] = "{" + restore_tags(text, end, ("fn",), style_at[end]) + "}" + inserts.get(end, "")
    return insert_at_visible(text, inserts)


def display_name(m: Member, rules: DisplayRules, chars=None, format: str = "full") -> str:
    if (m.speaker or {}).get("character") == "nobody" or (chars is not None and chars.speaker_ignored(m)):
        return ""
    sp = m.speaker or {}
    label = sp.get("label") or ""
    name = chars.display_name(sp.get("character"), m.lang, label, format) if chars is not None else label
    if not rules.converted_for(m):
        return name
    if chars is not None and format != "label":
        ch = next((c for c in chars.characters if c.id == sp.get("character")), None)
        target_lang = "zh-Hans" if rules.direction == "simplified" else "zh-Hant"
        form = ch.names.get(target_lang) if ch is not None else None
        if form is not None and form.compose(format):
            return form.compose(format)
        if ch is not None and ch.name_form(m.lang) is None:
            return name
    return rules.convert(name)


def hidden_positions(m: Member, rules: DisplayRules) -> set[tuple[int, int]]:
    """(run index, offset in run) of the characters a rule hides in this member."""
    out: set[tuple[int, int]] = set()
    if not rules.hide_final_period or m.lane != "original" or m.role not in ("dialogue", "top", "alt"):
        return out
    protected = {i for tok in tokenize(m.text) if tok.protected for i in range(tok.start, tok.end)}
    raw = 0
    flat: list[tuple[int, int, str, str]] = []          # (run, offset, char, kind)
    for ri, r in enumerate(m.runs):
        if r.kind != "tag":
            for k, ch in enumerate(r.text):
                flat.append((ri, k, ch, "raw" if raw + k in protected else r.kind))
        raw += len(r.text)
    for i, (ri, k, ch, kind) in enumerate(flat):
        if kind != "text" or ch not in PERIODS:
            continue
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        if nxt is None or nxt[3] == "sep" or nxt[2] in SPACES:
            out.add((ri, k))
    return out


def display_runs(m: Member, rules: DisplayRules) -> tuple[list[Run], int]:
    """The member's runs with hidden characters cut out (identities of the rest unchanged)."""
    hide = hidden_positions(m, rules)
    if not hide:
        return list(m.runs), 0
    out: list[Run] = []
    for ri, r in enumerate(m.runs):
        if r.kind == "tag" or not any((ri, k) in hide for k in range(len(r.text))):
            out.append(r)
            continue
        k = 0
        while k < len(r.text):
            if (ri, k) in hide:
                k += 1
                continue
            j = k
            while j < len(r.text) and (ri, j) not in hide:
                j += 1
            out.append(r.slice(k, j) if (r.src is not None or r.ins is not None) else r.copy(text=r.text[k:j]))
            k = j
    # a trailing separator left behind by a hidden final period is dropped
    while out and out[-1].kind == "sep":
        out.pop()
    return out, len(hide)
