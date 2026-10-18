"""Japanese subtitle import and closed-caption cleanup.

Japanese CC files (Netflix ``ja[cc]``, BD retimes, ...) carry things that must
not end up in a bilingual subtitle, but carry useful information too:

* ``(幸平創真)台詞`` / ``（有栖川(ありすがわ)樹璃(じゅり)）台詞`` — speaker labels.
  Removed from the text; the speaker is kept, and readings inside the label
  become *reading hints* for names.
* ``(戸の開閉音)`` on its own line — sound descriptions (dropped by default).
* ``((恵！))`` — double parentheses mark recalled / flashback speech: it is
  dialogue (the translation has it), so the brackets go and the text stays.
* ``≪`` ``≫`` off-screen markers, ``➡`` continuation arrows, ``♬～`` music lines.
* ``遠月(とおつき)学園`` — inline readings: removed from the text, kept as
  per-occurrence reading hints (highest-confidence source for that line).
* half-width katakana/punctuation (``ｱｷ`` ``｡``) → full width.

Nothing is silently lost: every removal is recorded on the line so reports can
show it and the integrity check knows what was intentional.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..ass.document import AssDocument
from ..ass.text import plain_text
from ..textutil import fullwidth_ascii_to_half, is_kana, is_kanji, normalize_halfwidth_kana, strip_bidi
from .srt import load_srt

OPEN = "(（"
CLOSE = ")）"
MUSIC_CHARS = set("♪♫♬～〜~ー－-・ 　")
_KANA_ONLY = re.compile(r"^[ぁ-ゖァ-ヺー・\s]+$")


@dataclass
class RubyHint:
    start: int     # visible index in the cleaned line text
    end: int
    surface: str
    reading: str
    source: str = "cc"


@dataclass
class JpLine:
    text: str                      # cleaned text
    raw: str
    kind: str = "speech"           # speech | sfx | music | lyric | empty
    speaker: Optional[str] = None
    speaker_raw: Optional[str] = None
    offscreen: bool = False
    flashback: bool = False        # ((…)) recalled speech
    hints: list[RubyHint] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


@dataclass
class JpCue:
    idx: int
    start: int
    end: int
    lines: list[JpLine]

    @property
    def speech_lines(self) -> list[JpLine]:
        return [ln for ln in self.lines if ln.kind == "speech" and ln.text]


@dataclass
class JpOptions:
    drop_sfx: bool = True
    strip_speaker: bool = True
    strip_period: bool = True
    halfwidth_alnum: bool = False

    @classmethod
    def from_config(cls, section: dict) -> "JpOptions":
        return cls(
            drop_sfx=bool(section.get("drop_sfx", True)),
            strip_speaker=bool(section.get("strip_speaker", True)),
            strip_period=bool(section.get("strip_period", True)),
            halfwidth_alnum=bool(section.get("halfwidth_alnum", False)),
        )


def _balanced_paren(s: str, i: int) -> int:
    """Index after the bracket that closes the one opening at s[i]; -1 if unbalanced."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] in OPEN:
            depth += 1
        elif s[j] in CLOSE:
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def extract_inline_readings(text: str) -> tuple[str, list[RubyHint]]:
    """``遠月(とおつき)学園`` → ``遠月学園`` + hint(遠月, とおつき).

    Only a kanji run directly followed by a kana-only parenthetical counts as a
    reading, so ``３人(さんにん)`` style is also accepted when the base has digits.
    """
    out: list[str] = []
    hints: list[RubyHint] = []
    annotated_upto = 0  # never let a reading reach back into an already annotated base
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in OPEN and out:
            j = _balanced_paren(text, i)
            inner = text[i + 1 : j - 1] if j > 0 else ""
            if j > 0 and inner and _KANA_ONLY.match(inner):
                # walk back over the kanji (or digit+kanji) run already emitted
                k = len(out)
                # kanji plus attached digits/letters (Ｂ子(びーこ), ３人(さんにん))
                while k > annotated_upto and (is_kanji(out[k - 1]) or out[k - 1].isalnum() and not is_kana(out[k - 1])):
                    k -= 1
                if k < len(out) and any(is_kanji(c) for c in out[k:]):
                    surface = "".join(out[k:])
                    hints.append(RubyHint(k, len(out), surface, inner.replace(" ", "").replace("　", "")))
                    annotated_upto = len(out)
                    i = j
                    continue
        out.append(ch)
        i += 1
    return "".join(out), hints


def parse_speaker_label(label: str) -> tuple[str, list[tuple[str, str]]]:
    """``有栖川(ありすがわ)樹璃(じゅり)`` → ("有栖川樹璃", [("有栖川","ありすがわ"),("樹璃","じゅり")])."""
    name, hints = extract_inline_readings(label)
    return normalize_speaker(name.strip()), [(h.surface, h.reading) for h in hints]


_SPEAKER_SUFFIX = re.compile(r"^(.+?)・(?:回想|心の声|ナレーション|独白|声|電話|.*モノマネ|.*ものまね)$")


def normalize_speaker(name: str) -> str:
    """``城一郎・回想`` / ``幸平創真・心の声`` / ``小西・萩本欽一のモノマネ`` → the speaker alone."""
    m = _SPEAKER_SUFFIX.match(name)
    return m.group(1) if m else name


_DESC_WORDS = {
    "足音", "物音", "爆音", "轟音", "雑音", "水音", "羽音", "靴音", "波音", "銃声", "歓声", "悲鳴", "拍手", "鼓動", "笑い",
    "咳払い", "いびき", "寝息", "あくび", "舌打ち", "口笛", "効果音", "BGM", "ＢＧＭ", "バイブ", "サイレン", "チャイム",
}
_DESC_TAIL = re.compile(r"(の音|の声|音声|音楽|ため息|息遣い|ざわめき|どよめき|拍手|鼓動|悲鳴|鼻歌|笑い|モード|で|ながら)$")


def looks_like_description(label: str) -> bool:
    """A bracketed note that describes a sound or the way of speaking, not who speaks:
    ``ノックの音`` ``携帯電話の振動音`` ``マナーモード`` ``棒読みで`` ``鼻歌「スパイス」``."""
    core = re.sub(r"[「『][^」』]*[」』]", "", label).strip()
    if not core or core in _DESC_WORDS or core.startswith("鼻歌") or _DESC_TAIL.search(core):
        return True
    # 振動音, 笑い声, シャッター音 (a two-kanji 和音 / 初音 may be a name)
    return len(core) >= 3 and core.endswith(("音", "声"))


_FB_OPEN = ("((", "（（", "(（", "（(")
_FB_CLOSE = ("))", "））", ")）", "）)")


def _strip_flashback(t: str, line: "JpLine") -> str:
    """``((…))`` recalled speech; a long one opens on one line and closes on the next."""
    if t.startswith(_FB_OPEN):
        t = t[2:].lstrip()
        line.flashback = True
    opens = sum(t.count(c) for c in OPEN)
    closes = sum(t.count(c) for c in CLOSE)
    if t.endswith(_FB_CLOSE) and closes - opens >= 2:
        t = t[:-2].rstrip()
        line.flashback = True
    return t


def _is_music_only(text: str) -> bool:
    t = text.strip()
    return bool(t) and all(ch in MUSIC_CHARS for ch in t) and any(ch in "♪♫♬" for ch in t)


def clean_line(raw: str, opts: JpOptions) -> JpLine:
    text = strip_bidi(raw)
    text = normalize_halfwidth_kana(text)
    if opts.halfwidth_alnum:
        text = fullwidth_ascii_to_half(text)
    line = JpLine(text="", raw=raw)
    t = text.strip()
    # off-screen markers
    while t[:1] in ("≪", "≫"):
        line.offscreen = True
        line.removed.append(t[0])
        t = t[1:].lstrip()
    if t.endswith("➡"):
        line.removed.append("➡")
        t = t[:-1].rstrip()
    t = t.replace("➡", " ")
    # [TEL] (phone voice) and similar caption markers; ≪≫ inside the line mark a new voice → phrase break
    for m in re.findall(r"\[[A-Za-z]+\]|［[Ａ-Ｚａ-ｚA-Za-z]+］", t):
        line.removed.append(m)
    t = re.sub(r"\[[A-Za-z]+\]|［[Ａ-Ｚａ-ｚA-Za-z]+］", " ", t)
    if "≪" in t or "≫" in t:
        line.removed.append("≫")
        t = re.sub(r"[≪≫]", " ", t)
    t = _strip_flashback(t.strip(), line)
    if not t:
        line.kind = "empty"
        return line
    if _is_music_only(t):
        line.kind = "music"
        line.removed.append(t)
        return line
    if t[0] in "♪♫♬" or t[-1:] in ("♪", "♫", "♬"):
        line.kind = "lyric"
        t = t.strip("♪♫♬ 　～〜")
    # whole-line parenthetical = sound / description
    if t[0] in OPEN:
        j = _balanced_paren(t, 0)
        if j == len(t):
            inner = t[1:-1]
            line.kind = "sfx"
            line.removed.append(t)
            line.speaker_raw = inner
            if not opts.drop_sfx:
                line.kind = "speech"
                line.text = t
            return line
        if j > 0 and opts.strip_speaker:
            label = t[1 : j - 1]
            name, label_hints = parse_speaker_label(label)
            if label_hints or not looks_like_description(name):
                line.speaker = name
                line.speaker_raw = label
                for surface, reading in label_hints:
                    line.hints.append(RubyHint(-1, -1, surface, reading, "label"))
            line.removed.append(t[:j])
            t = t[j:].lstrip()
            # off-screen marker may also follow the label
            while t[:1] in ("≪", "≫"):
                line.offscreen = True
                t = t[1:].lstrip()
            t = _strip_flashback(t, line)
    t, inline_hints = extract_inline_readings(t)
    # remaining parentheticals are descriptions: ガラガラ(戸の開閉音), …(笑い)
    described = False
    while True:
        m = re.search(r"[(（]([^()（）]*)[)）]", t)
        if not m:
            break
        line.removed.append(m.group(0))
        t = (t[: m.start()] + " " + t[m.end():]).strip()
        described = True
    if described and t and all(is_kana(c) or c in "ー～〜…！？!? 　っッ" for c in t) and not any("ぁ" <= c <= "ゖ" for c in t):
        # only a katakana sound word is left (ガラガラ): the whole line is a sound description
        line.kind = "sfx"
        line.removed.append(t)
        if not opts.drop_sfx:
            line.kind = "speech"
            line.text = t
        return line
    t = re.sub(r"[ 　\t]+", " ", t).strip()
    if opts.strip_period:
        t = re.sub(r"。(?=\s|$)", "", t).strip()
        t = re.sub(r"[ ]+", " ", t)
    # recompute hint offsets on the whitespace-normalised text
    fixed: list[RubyHint] = []
    cursor = 0
    for h in inline_hints:
        pos = t.find(h.surface, cursor)
        if pos >= 0:
            fixed.append(RubyHint(pos, pos + len(h.surface), h.surface, h.reading, "cc"))
            cursor = pos + len(h.surface)
    line.hints.extend(fixed)
    line.text = t
    if not t:
        line.kind = "empty"
    return line


def _label_only_to_speaker(lines: list[JpLine]) -> None:
    """``（有栖川(ありすがわ)樹璃(じゅり)）`` alone on a line, followed by speech in the
    same cue, is a speaker label for that speech — not a sound description."""
    for i, line in enumerate(lines[:-1]):
        nxt = lines[i + 1]
        if line.kind == "sfx" and nxt.kind == "speech" and nxt.speaker is None and line.speaker_raw:
            name, label_hints = parse_speaker_label(line.speaker_raw)
            if not label_hints and looks_like_description(name):
                continue  # (ノックの音) / (マナーモード) before a line is a sound, not a speaker
            nxt.speaker = name
            nxt.speaker_raw = line.speaker_raw
            for surface, reading in label_hints:
                nxt.hints.append(RubyHint(-1, -1, surface, reading, "label"))
            line.kind = "label"


def clean_cues(raw_cues: list[tuple[int, int, list[str]]], opts: JpOptions) -> list[JpCue]:
    cues: list[JpCue] = []
    for start, end, lines in raw_cues:
        cleaned = [clean_line(l, opts) for l in lines if l.strip()]
        _label_only_to_speaker(cleaned)
        cues.append(JpCue(len(cues) + 1, start, end, cleaned))
    return cues


def load_jp(path: Path, opts: JpOptions, *, ass_styles: Optional[list[str]] = None) -> list[JpCue]:
    """Load a Japanese SRT or ASS and clean it."""
    path = Path(path)
    raw: list[tuple[int, int, list[str]]] = []
    if path.suffix.lower() == ".ass":
        doc = AssDocument.load(path)
        for ev in doc.events:
            if not ev.is_dialogue:
                continue
            if ass_styles and ev.style not in ass_styles:
                continue
            txt = plain_text(ev.text, breaks="\n")
            raw.append((ev.start, ev.end, txt.split("\n")))
        raw.sort(key=lambda x: (x[0], x[1]))
    else:
        for c in load_srt(path):
            raw.append((c.start, c.end, c.lines))
    return clean_cues(raw, opts)


def speaker_reading_hints(cues: list[JpCue]) -> dict[str, str]:
    """Name readings collected from speaker labels across the episode."""
    out: dict[str, str] = {}
    for cue in cues:
        for line in cue.lines:
            for h in line.hints:
                if h.source == "label":
                    out.setdefault(h.surface, h.reading)
    # 西園寺莢一(きょういち): the label annotated only the given name; if the
    # family name is known from other labels, cut it off the surface.
    refined: dict[str, str] = {}
    for surface, reading in out.items():
        for other in sorted(out, key=len, reverse=True):
            if other != surface and surface.startswith(other) and not reading.startswith(out[other]):
                surface = surface[len(other):]
                break
        refined.setdefault(surface, reading)
    return refined
