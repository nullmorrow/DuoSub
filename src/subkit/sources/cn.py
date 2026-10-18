"""Chinese subtitle import: ASS or SRT → dialogue rows + lines kept verbatim.

Every source event gets one role:

* ``dialogue``  bottom dialogue (gets a JP partner line)
* ``top``       top-positioned dialogue (also gets a JP partner)
* ``song_jp`` / ``song_cn``  lyric lines already present in the CN release
* ``keep``      signs, staff, notes, comments — copied through unchanged
* ``drop``      explicitly unwanted styles

Roles come from ``subproj.toml`` ``[cn]``; when a list is empty the role is
inferred (most frequent un-positioned style = dialogue, op*/ed*/song*/kara* =
songs, ``\\pos``/``\\move``/drawings = signs).
"""

from __future__ import annotations

import fnmatch
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..ass.document import AssDocument, Event
from ..ass.text import VisibleText, line_tag, tokenize
from ..textutil import strip_bidi
from .srt import load_srt

_SONG_HINT = re.compile(r"^(op|ed|in|ins|song|kara|lyric)", re.IGNORECASE)
# A style named only after the song part (OP, ED2, IN, Song) without a language marker: translation lyrics.
_SONG_ONLY = re.compile(r"^(op|ed|in|ins|insert|song|kara|karaoke|lyrics?)[\W_\d]*$", re.IGNORECASE)


@dataclass
class CnRow:
    event: Event            # the (copied) source event
    role: str               # dialogue | top | song_cn
    text: str               # cleaned CN text (tags kept)
    plain: str              # visible text, breaks → space

    @property
    def start(self) -> int:
        return self.event.start

    @property
    def end(self) -> int:
        return self.event.end


@dataclass
class CnSource:
    doc: AssDocument                     # source document (resampled later)
    roles: dict[int, str] = field(default_factory=dict)   # event index → role
    style_roles: dict[str, str] = field(default_factory=dict)
    lang: str = "zh-Hans"                # zh-Hans | zh-Hant (detected)
    notes: list[str] = field(default_factory=list)   # warnings for the prepare log

    def events_with_role(self, *roles: str) -> list[tuple[int, Event]]:
        return [(i, ev) for i, ev in enumerate(self.doc.events) if self.roles.get(i) in roles]


def _match_any(name: str, patterns: list[str]) -> bool:
    low = name.lower()
    return any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns)


def _is_positioned(text: str) -> bool:
    for tok in tokenize(text):
        if tok.kind == "tags":
            for t in tok.tags:
                if t.name in ("pos", "move", "org") or (t.name == "p" and (t.num() or 0) > 0):
                    return True
        if tok.kind == "drawing":
            return True
    return False


# A tiny Simplified/Traditional discriminator: pairs of very frequent characters.
_SIMP = set("这个们来时会对发还没过吗为让点样东说话问间关见听开长门车进现实经买卖头记")
_TRAD = set("這個們來時會對發還沒過嗎為讓點樣東說話問間關見聽開長門車進現實經買賣頭記")


def detect_cn_script(texts: list[str]) -> str:
    s = t = 0
    for text in texts:
        for ch in text:
            if ch in _SIMP and ch not in _TRAD:
                s += 1
            elif ch in _TRAD and ch not in _SIMP:
                t += 1
    return "zh-Hant" if t > s else "zh-Hans"


def song_role(style: str, text: str) -> str:
    """Language markers win; shared song styles use kana in each original line."""
    low = style.casefold()
    if any(marker in low for marker in ("jp", "ja")):
        return "song_jp"
    if any(marker in low for marker in ("cn", "ch", "zh", "sc", "tc")):
        return "song_cn"
    return "song_jp" if any("ぁ" <= c <= "ゖ" or "ァ" <= c <= "ヺ" for c in VisibleText(text).text) else "song_cn"


def classify(doc: AssDocument, cfg: dict) -> CnSource:
    src = CnSource(doc)
    dialogue = list(cfg.get("dialogue") or [])
    top = list(cfg.get("top") or [])
    song_jp = list(cfg.get("song_jp") or [])
    song_cn = list(cfg.get("song_cn") or [])
    drop = list(cfg.get("drop") or [])
    keep = list(cfg.get("keep") or [])

    counts: Counter[str] = Counter()
    positioned: Counter[str] = Counter()
    for ev in doc.events:
        if ev.is_dialogue:
            counts[ev.style] += 1
            if _is_positioned(ev.text):
                positioned[ev.style] += 1

    if dialogue and not any(_match_any(s, dialogue) for s in counts):
        src.notes.append(f"配置的中文对白 Style {dialogue} 在这个文件里不存在，改为自动识别（可在 subproj.toml 加 [cn.S{{季:02}}] 段）")
        dialogue = []
    if not dialogue:
        plain_styles = [s for s, n in counts.most_common() if positioned[s] < n * 0.5 and not _SONG_HINT.match(s)]
        if plain_styles:
            dialogue = [plain_styles[0]]
            if src.notes:
                src.notes[-1] += f"：{plain_styles[0]}"
    if not song_jp and not song_cn:
        for s in counts:
            if _SONG_HINT.match(s):
                low = s.lower()
                if "jp" in low or "ja" in low or "jap" in low:
                    song_jp.append(s)
                elif "cn" in low or "ch" in low or "zh" in low or "sc" in low or "tc" in low:
                    song_cn.append(s)
                elif _SONG_ONLY.match(s):
                    song_cn.append(s)
    for s in counts:
        if _match_any(s, keep):
            src.style_roles[s] = "keep"
        elif _match_any(s, drop):
            src.style_roles[s] = "drop"
        elif _match_any(s, song_jp) and _match_any(s, song_cn):
            src.style_roles[s] = "song"      # JP and CN lyrics share one style: decided per line
        elif _match_any(s, song_jp):
            src.style_roles[s] = "song_jp"
        elif _match_any(s, song_cn):
            src.style_roles[s] = "song_cn"
        elif _match_any(s, top):
            src.style_roles[s] = "top"
        elif _match_any(s, dialogue):
            src.style_roles[s] = "dialogue"
        else:
            src.style_roles[s] = "keep"

    for i, ev in enumerate(doc.events):
        override = (cfg.get("event_roles") or {}).get(i)
        if override is not None:
            src.roles[i] = override
            continue
        if not ev.is_dialogue:
            src.roles[i] = "keep"
            continue
        role = src.style_roles.get(ev.style, "keep")
        if role.startswith("song"):
            vis = VisibleText(ev.text).text
            if not vis.strip():
                role = "keep"
            elif role == "song":
                role = "song_jp" if any("ぁ" <= c <= "ゖ" or "ァ" <= c <= "ヺ" for c in vis) else "song_cn"
        if role in ("dialogue", "top"):
            vis = VisibleText(ev.text)
            if not vis.text.strip():
                role = "keep"          # empty / pure drawing
            elif _is_positioned(ev.text):
                role = "keep"          # positioned sign that happens to use the dialogue style
        src.roles[i] = role
    src.lang = detect_cn_script([VisibleText(ev.text).text for i, ev in enumerate(doc.events) if src.roles[i] in ("dialogue", "top")])
    return src


def load_cn(path: Path, cfg: dict) -> CnSource:
    path = Path(path)
    if path.suffix.lower() == ".srt":
        doc = AssDocument()
        doc.set_info("ScriptType", "v4.00+")
        doc.set_info("PlayResX", 1920)
        doc.set_info("PlayResY", 1080)
        for c in load_srt(path):
            text = "\\N".join(strip_bidi(x).strip() for x in c.lines if x.strip())
            doc.events.append(Event(kind="Dialogue", start=c.start, end=c.end, style="Default", text=text))
        cfg = dict(cfg)
        cfg["dialogue"] = ["Default"]
        return classify(doc, cfg)
    doc = AssDocument.load(path)
    return classify(doc, cfg)


_Q_OPEN = set("“‘「『〝｢")
_Q_CLOSE = set("”’」』〞〟｣")


def normalize_quotes(text: str, pair: str) -> str:
    """Unify quotation marks to ``pair`` (e.g. 『』) outside override blocks.
    Straight quotes alternate open/close.  《》〈〉 (book titles) are left alone."""
    if not pair or len(pair) != 2:
        return text
    out = []
    opened = False
    for tok in tokenize(text):
        if tok.kind != "text":
            out.append(tok.raw)
            continue
        buf = []
        for ch in tok.raw:
            if ch in _Q_OPEN:
                buf.append(pair[0])
                opened = True
            elif ch in _Q_CLOSE:
                buf.append(pair[1])
                opened = False
            elif ch in "\"＂":
                buf.append(pair[1] if opened else pair[0])
                opened = not opened
            else:
                buf.append(ch)
        out.append("".join(buf))
    return "".join(out)


def clean_cn_text(text: str, *, linebreak: str = "  ", space: str = "  ") -> str:
    """CN dialogue: \\N → configured separator, runs of spaces → configured space, keep tags."""
    out = []
    for tok in tokenize(strip_bidi(text)):
        if tok.kind in ("break", "softbreak"):
            out.append("\x00")
        elif tok.kind == "text":
            out.append(tok.raw)
        elif tok.kind == "hardspace":
            out.append(" ")
        else:
            out.append(tok.raw)
    joined = "".join(out)
    joined = re.sub(r"[ 　\t]*\x00[ 　\t]*", "\x00", joined)
    joined = re.sub(r"[ 　\t]+", space, joined)
    joined = joined.replace("\x00", linebreak)
    return joined.strip(" 　")
