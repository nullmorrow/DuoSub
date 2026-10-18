"""Conservative, lossless English SDH and generic subtitle analysis."""

from __future__ import annotations

import re
from typing import Callable

from ..languages import get_language
from ..model.source import Span
from . import cc

ANALYZER = {"name": "en-sdh", "version": 1}
GENERIC_ANALYZER = {"name": "generic", "version": 1}
_BLANK = " \t　" + cc._MASK
_BRACKET = re.compile(r"\[([^\[\]\n]+)\]|\(([^()\n]+)\)")
_COLON = re.compile(r"([A-Z][A-Z '\-]{0,39}):(?=\s|$)")
_NAME = re.compile(r"[A-Z][a-zA-Z'\-]*(?: [A-Z][a-zA-Z'\-]*){0,2}")
_SOUNDS = {
    "laugh", "laughs", "laughing", "laughter", "chuckles", "chuckling", "sighs", "sighing",
    "gasps", "gasping", "sobs", "sobbing", "crying", "coughs", "coughing", "screams", "screaming",
    "whispers", "whispering", "applause", "cheering", "footsteps", "silence", "gunshot", "gunshots",
    "door creaks", "door creaking", "door opens", "door opening", "door closes", "door closing",
    "door slams", "door slamming", "phone rings", "phone ringing", "telephone ringing", "bell rings",
    "bell ringing", "dog barks", "dog barking", "wind blowing", "thunder", "thunder rumbling",
}
_MUSIC = {"music", "music playing", "background music", "instrumental music", "singing"}


def analyzer_for(lang: str) -> tuple[dict, Callable]:
    return {"ja-cc": (cc.ANALYZER, cc.analyze_event),
            "en-sdh": (ANALYZER, analyze_event),
            "generic": (GENERIC_ANALYZER, analyze_generic)}[get_language(lang).analyzer_id]


def analyze_event(e: int, text: str, fmt: str = "srt") -> cc.EventAnalysis:
    return _analyze(e, text, fmt, english=True)


def analyze_generic(e: int, text: str, fmt: str = "srt") -> cc.EventAnalysis:
    return _analyze(e, text, fmt, english=False)


def _analyze(e: int, text: str, fmt: str, *, english: bool) -> cc.EventAnalysis:
    result = cc.EventAnalysis()
    chars = list(cc._masked(text) if fmt != "srt" else text)
    for a, b, raw in cc.format_tags(text) if fmt == "srt" else []:
        result.spans.append(Span(e, a, b, "format", raw, data=cc._tag_data(raw)))
        chars[a:b] = [cc._MASK] * (b - a)
    masked = "".join(chars)
    taken = [ch == cc._MASK for ch in masked]

    def span(a: int, b: int, kind: str, value: str, data=None) -> None:
        result.spans.append(Span(e, a, b, kind, value, data=data))
        taken[a:b] = [True] * (b - a)

    lines = cc.split_lines(text, fmt)
    heads = [(a, b, masked[a:b].lstrip(_BLANK)) for a, b in lines if masked[a:b].strip(_BLANK)]
    dashed = len(heads) == 2 and all(re.match(r"[-－‐‑–]\s+\S", body) for _, _, body in heads)
    pending = None
    for a, b in lines:
        info = cc.LineInfo(a, b, "empty")
        p = a + len(masked[a:b]) - len(masked[a:b].lstrip(_BLANK))
        q = a + len(masked[a:b].rstrip(_BLANK))
        if p >= q:
            result.lines.append(info)
            continue
        if dashed:
            span(p, p + 1, "marker", text[p], {"turn": True})
            p += 1
            while p < q and masked[p] in _BLANK:
                p += 1
        if english:
            label = _COLON.match(masked, p, q)
            if label:
                name = label.group(1).strip()
                span(p, label.end(), "speaker_label", name, {"raw": name, "line_only": label.end() == q})
                info.speaker = name
                if label.end() == q:
                    pending = name
                p = label.end()
            for match in _BRACKET.finditer(masked, p, q):
                value = (match.group(1) or match.group(2)).strip()
                low = value.casefold()
                if low in _SOUNDS or low in _MUSIC:
                    kind = "music" if low in _MUSIC else "sfx"
                    span(match.start(), match.end(), kind, value)
                    info.kind = kind
                elif match.start() == p and _NAME.fullmatch(value):
                    span(match.start(), match.end(), "speaker_label", value,
                         {"raw": value, "line_only": match.end() == q})
                    info.speaker = value
                    if match.end() == q:
                        pending = value
            body = "".join(masked[k] for k in range(p, q) if not taken[k]).strip(_BLANK)
            if any(c in cc.NOTE_CHARS for c in body):
                music = all(c in cc.MUSIC_CHARS for c in body)
                for k in range(p, q):
                    if not taken[k] and (masked[k] in cc.NOTE_CHARS or music and not masked[k].isspace()):
                        span(k, k + 1, "music" if music else "marker", text[k])
                info.kind = "music" if music else "lyric"
        rest = "".join(masked[k] for k in range(p, q) if not taken[k]).strip(_BLANK)
        if rest:
            if info.kind != "lyric":
                info.kind = "speech"
            info.speaker = info.speaker or pending
            pending = None
        elif info.kind == "empty" and info.speaker:
            info.kind = "label"
        result.lines.append(info)
    result.spans.sort(key=lambda s: (s.a, s.b))
    return result
