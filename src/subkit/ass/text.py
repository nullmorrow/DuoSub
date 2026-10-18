"""The one and only tokenizer for ASS dialogue text.

Every feature that needs "visible character positions" (ruby, karaoke, pinyin,
glosses, private annotations) goes through :func:`tokenize` and
:class:`VisibleText`, so ``{}`` blocks, ``\\N``, ``\\n``, ``\\h`` and drawing
mode are handled identically everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional
import unicodedata

from .tags import KARAOKE_TAGS, Tag, parse_block


@dataclass
class Token:
    kind: str          # "tags" | "comment" | "text" | "break" | "softbreak" | "hardspace" | "drawing"
    raw: str
    start: int         # offset in the raw text
    end: int
    tags: list[Tag] = field(default_factory=list)
    protected: bool = False             # valid Raw content and its markers

    @property
    def body(self) -> str:
        if self.kind in ("tags", "comment"):
            return self.raw[1:-1]
        return self.raw


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    text = text or ""
    i, n = 0, len(text)
    drawing = False
    buf_start = 0

    def flush(upto: int) -> None:
        nonlocal buf_start
        if upto > buf_start:
            tokens.append(Token("drawing" if drawing else "text", text[buf_start:upto], buf_start, upto))
        buf_start = upto

    while i < n:
        ch = text[i]
        if ch == "{" and text.startswith("{@Raw", i):
            j = text.find("}", i + 1)
            if text.startswith("{@Raw=", i) and j >= 0:
                flush(i)
                start = i + len("{@Raw=")
                tokens.append(Token("comment", text[i:start], i, start, protected=True))
                for tok in tokenize(text[start:j]):
                    tok.start += start
                    tok.end += start
                    tok.protected = True
                    tokens.append(tok)
                tokens.append(Token("comment", "}", j, j + 1, protected=True))
                i = buf_start = j + 1
                continue
            if j >= 0:
                flush(i)
                tokens.append(Token("text", text[i:j + 1], i, j + 1))
                i = buf_start = j + 1
                continue
        if ch == "{":
            j = text.find("}", i + 1)
            if j < 0:
                i += 1  # unclosed brace: literal text
                continue
            flush(i)
            body = text[i + 1 : j]
            if "\\" in body:
                tags = parse_block(body)
                tokens.append(Token("tags", text[i : j + 1], i, j + 1, tags))
                for t in tags:
                    if t.name == "p":
                        v = t.num()
                        drawing = bool(v and v > 0)
            else:
                tokens.append(Token("comment", text[i : j + 1], i, j + 1))
            i = j + 1
            buf_start = i
            continue
        if ch == "\\" and i + 1 < n and text[i + 1] in "Nnh":
            flush(i)
            kind = {"N": "break", "n": "softbreak", "h": "hardspace"}[text[i + 1]]
            tokens.append(Token(kind, text[i : i + 2], i, i + 2))
            i += 2
            buf_start = i
            continue
        i += 1
    flush(n)
    return tokens


@dataclass
class VisChar:
    ch: str
    raw_start: int  # offset of the char (or of the \h escape) in the raw text
    raw_end: int
    line: int       # visual line index (hard breaks)


class VisibleText:
    """Visible characters of a dialogue text with a mapping back to raw offsets.

    ``softbreak`` (``\\n``) only breaks in WrapStyle 2; otherwise libass renders it
    as a space.  ``hardspace`` is a non-breaking space (rendered as one space).
    """

    def __init__(self, text: str, *, wrap_style: int = 0) -> None:
        self.raw = text or ""
        self.tokens = tokenize(self.raw)
        self.chars: list[VisChar] = []
        line = 0
        for tok in self.tokens:
            if tok.kind == "text":
                for k, ch in enumerate(tok.raw):
                    self.chars.append(VisChar(ch, tok.start + k, tok.start + k + 1, line))
            elif tok.kind == "hardspace":
                self.chars.append(VisChar(" ", tok.start, tok.end, line))
            elif tok.kind == "break" or (tok.kind == "softbreak" and wrap_style == 2):
                line += 1
            elif tok.kind == "softbreak":
                self.chars.append(VisChar(" ", tok.start, tok.end, line))
        self.text = "".join(c.ch for c in self.chars)
        self.line_count = line + 1

    def __len__(self) -> int:
        return len(self.text)

    def raw_offset(self, vis_index: int) -> int:
        """Raw offset where visible char ``vis_index`` starts (len(raw) for end)."""
        if vis_index >= len(self.chars):
            return len(self.raw)
        return self.chars[vis_index].raw_start

    def line_of(self, vis_index: int) -> int:
        if not self.chars:
            return 0
        return self.chars[min(vis_index, len(self.chars) - 1)].line

    def lines(self) -> list[str]:
        out: list[list[str]] = [[] for _ in range(self.line_count)]
        for c in self.chars:
            out[c.line].append(c.ch)
        return ["".join(x) for x in out]


def plain_text(text: str, *, breaks: str = "", wrap_style: int = 0) -> str:
    """Visible text.  Hard line breaks are replaced with ``breaks``."""
    vt = VisibleText(text, wrap_style=wrap_style)
    if breaks == "" or vt.line_count == 1:
        return vt.text
    return breaks.join(vt.lines())


def strip_tags(text: str) -> str:
    """Remove every ``{...}`` block but keep \\N/\\h escapes."""
    return "".join(t.raw for t in tokenize(text) if t.kind not in ("tags", "comment"))


def insert_at_visible(text: str, inserts: dict[int, str]) -> str:
    """Insert raw strings so they end up immediately before visible char i.

    Index ``len(visible)`` appends at the very end.  Override blocks that precede
    a character stay before the inserted string only if they come earlier in the
    raw text — insertion happens right before the character itself.
    """
    if not inserts:
        return text
    vt = VisibleText(text)
    by_raw: dict[int, list[str]] = {}
    for vis, s in inserts.items():
        pos = vt.raw_offset(vis) if vis < len(vt.chars) else len(text)
        by_raw.setdefault(pos, []).append(s)
    out, last = [], 0
    for pos in sorted(by_raw):
        out.append(text[last:pos])
        out.extend(by_raw[pos])
        last = pos
    out.append(text[last:])
    return "".join(out)


def private_blocks(text: str, prefix: str = "@") -> list[tuple[int, str, Token]]:
    """``{@...}`` annotation blocks with the visible index they sit at."""
    out = []
    vis = 0
    for tok in tokenize(text):
        if tok.kind == "comment" and not tok.protected and tok.body.startswith(prefix):
            out.append((vis, tok.body[len(prefix):], tok))
        elif tok.kind == "text":
            vis += len(tok.raw)
        elif tok.kind == "hardspace":
            vis += 1
        elif tok.kind == "softbreak":
            vis += 1
    return out


def remove_private(text: str, prefix: str = "@", *, keep_raw: bool = False) -> str:
    """Drop private metadata, retaining Raw content and optionally its markers."""
    return "".join(t.raw for t in tokenize(text) if not (t.kind == "comment" and
                  ((t.protected and not keep_raw) or (not t.protected and t.body.startswith(prefix)))))


def raw_mark_errors(text: str) -> list[str]:
    """Malformed Raw syntax stays literal; valid markers do not trigger brace warnings."""
    return [tok.raw for tok in tokenize(text) if not tok.protected and tok.kind == "text"
            and "{@Raw" in tok.raw]


def remove_tags_named(text: str, names: Iterable[str]) -> str:
    """Remove specific override tags; drop blocks that become empty."""
    names = set(names)
    out = []
    for tok in tokenize(text):
        if tok.kind != "tags":
            out.append(tok.raw)
            continue
        kept = [t.raw for t in tok.tags if t.name not in names]
        if kept:
            out.append("{" + "".join(kept) + "}")
    return "".join(out)


_PROLONG = frozenset("ー〜～~―—…‥")


def karaoke_sung(ch: str) -> bool:
    """Characters that take karaoke fill time; punctuation, symbols and spaces do not.  Held-note marks do
    (ー 〜 ～ ― — and the ellipsis … ‥: 分かってる… is sung out, 」 is not)."""
    if ch in _PROLONG:
        return True
    return not (ch.isspace() or unicodedata.category(ch)[0] in "PSZ")


def split_karaoke_overrides(text: str) -> str:
    """Split karaoke syllables so libass fills them as sung; preserves raw tags and each syllable's total time.

    * libass ends a syllable's fill at an override block inside it, so text after a mid-syllable block (e.g. a
      punctuation display font) was never filled: the syllable is split at every block between visible
      characters, the parts sharing the time by visible-character count — as the fill would sweep them by width
      without the block (啦{\\fn…}…」 like る…」).
    * ``\\kf``/``\\K``: punctuation and symbols take no time (user 2026-10-13: only the ellipsis and held-note
      marks count, see :func:`karaoke_sung`).  Before the first sung character (「我, 「分) they fill at once when
      the syllable starts, so an opening bracket never eats half of a beat; after the last one (る…」) they fill
      at once when it is sung.  Spaces after the last sung character stay in the syllable (く␣ as in the user's
      finished files).
    * ``\\k``/``\\ko`` change colour for the whole syllable at its start: parts get 0 … 0, full.
    """
    tokens = tokenize(text)
    marks = [(i, next(t for t in reversed(tok.tags) if t.name in KARAOKE_TAGS))
             for i, tok in enumerate(tokens) if tok.kind == "tags" and any(t.name in KARAOKE_TAGS for t in tok.tags)]
    raw = [tok.raw for tok in tokens]
    inserts: dict[int, list[tuple[int, str]]] = {}          # text token -> [(char offset, new block)]
    for n, (start, tag) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(tokens)
        if not tag.arg.strip().isdigit():
            continue
        fill = tag.name in ("kf", "K")
        chars = []                                          # (token index, char offset or None, sung)
        for i in range(start + 1, end):
            tok = tokens[i]
            if tok.kind == "text":
                chars.extend((i, k, karaoke_sung(ch) if fill else True) for k, ch in enumerate(tok.raw))
            elif tok.kind in ("hardspace", "softbreak"):
                chars.append((i, None, not fill))
        sung = [c for c, item in enumerate(chars) if item[2]]
        if not sung:
            continue
        cuts = {c for c in range(1, len(chars))
                if any(tokens[i].kind == "tags" for i in range(chars[c - 1][0] + 1, chars[c][0]))}
        lead = sung[0] if fill else 0
        trail = len(chars)
        if fill and any(not (k is None or tokens[i].raw[k].isspace()) for i, k, _s in chars[sung[-1] + 1:]):
            trail = sung[-1] + 1
        cuts |= {c for c in (lead, trail) if 0 < c < len(chars)}
        if not cuts:
            continue
        bounds = [0, *sorted(cuts), len(chars)]
        counts = [0 if z <= lead or a >= trail else z - a for a, z in zip(bounds, bounds[1:])]
        duration = int(tag.arg.strip())
        if fill:
            total = sum(counts)
            durations = [duration * c // total for c in counts]
            order = sorted((i for i in range(len(counts)) if counts[i]), key=lambda i: -(duration * counts[i] % total))
            for i in order[:duration - sum(durations)]:
                durations[i] += 1
        else:
            durations = [0] * (len(counts) - 1) + [duration]
        at = raw[start].rfind(tag.raw)
        raw[start] = raw[start][:at] + f"\\{tag.name}{durations[0]}" + raw[start][at + len(tag.raw):]
        for c, d in zip(bounds[1:-1], durations[1:]):
            i, k, _sung = chars[c]
            block = next((j for j in range(i - 1, chars[c - 1][0], -1) if tokens[j].kind == "tags"), None)
            if (k is None or k == 0) and block is not None:
                raw[block] = raw[block][:-1] + f"\\{tag.name}{d}" + "}"
            elif k:
                inserts.setdefault(i, []).append((k, f"{{\\{tag.name}{d}}}"))
            else:
                raw[i] = f"{{\\{tag.name}{d}}}" + raw[i]
    for i, places in inserts.items():
        text_raw, out, last = raw[i], [], 0
        for k, block in sorted(places):
            out.extend((text_raw[last:k], block))
            last = k
        raw[i] = "".join(out) + text_raw[last:]
    return "".join(raw)


def leading_tags(text: str) -> list[Tag]:
    """Tags of the override blocks before the first visible character."""
    tags: list[Tag] = []
    for tok in tokenize(text):
        if tok.kind == "tags":
            tags.extend(tok.tags)
        elif tok.kind in ("text", "hardspace", "drawing"):
            break
    return tags


def line_tag(text: str, name: str) -> Optional[Tag]:
    """Line-level tag (\\pos, \\an, \\move, \\org, \\fad...) — like VSFilter/libass, the first one wins."""
    for tok in tokenize(text):
        if tok.kind == "tags":
            for t in tok.tags:
                if t.name == name:
                    return t
    return None
