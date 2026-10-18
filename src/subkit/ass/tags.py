"""Override tag parsing (``{\\pos(1,2)\\fs40}`` → structured commands)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Longest names first so that \fscx is not read as \fs, \an not as \a, ...
_TAG_NAMES = sorted(
    [
        "alpha", "iclip", "xbord", "ybord", "xshad", "yshad", "fscx", "fscy", "fade", "blur", "bord", "shad",
        "clip", "move", "pos", "org", "fad", "fax", "fay", "frx", "fry", "frz", "fsp", "pbo",
        "1c", "2c", "3c", "4c", "1a", "2a", "3a", "4a", "an", "be", "fe", "fn", "fr", "fs", "kf", "ko", "kt",
        "a", "b", "c", "i", "k", "K", "p", "q", "r", "s", "t", "u",
    ],
    key=len,
    reverse=True,
)
# Tags whose argument is free text up to the next backslash (may contain spaces).
_FREE_TEXT_ARG = {"fn", "r"}


@dataclass
class Tag:
    name: str       # canonical name, e.g. "pos", "fs", "1c", "k"
    arg: str        # argument without surrounding parentheses
    raw: str        # exact source text, starting with the backslash
    paren: bool = False

    def args(self) -> list[str]:
        """Comma separated arguments for parenthesised tags (top level only)."""
        if not self.paren:
            return [self.arg] if self.arg != "" else []
        out, depth, cur = [], 0, []
        for ch in self.arg:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                out.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        out.append("".join(cur).strip())
        return out

    def float_args(self) -> Optional[list[float]]:
        try:
            return [float(x) for x in self.args()]
        except ValueError:
            return None

    def num(self) -> Optional[float]:
        m = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", self.arg or "")
        return float(m.group(1)) if m else None


def _match_name(body: str, i: int) -> str:
    for name in _TAG_NAMES:
        if body.startswith(name, i):
            # \b, \i, \s, \u, \p, \q, \a, \k, \c, \r ... only when followed by an argument-ish char,
            # otherwise a longer unknown tag like \bold would be mis-read.  ASS renderers do the
            # same prefix match, so we mirror them.
            return name
    return ""


def parse_block(body: str) -> list[Tag]:
    """Parse the inside of an override block.  Text outside tags (comments) is ignored."""
    tags: list[Tag] = []
    i, n = 0, len(body)
    while i < n:
        if body[i] != "\\":
            i += 1
            continue
        start = i
        i += 1
        name = _match_name(body, i)
        if not name:
            # unknown tag: swallow letters/digits so we can still re-emit it
            j = i
            while j < n and body[j] != "\\":
                j += 1
            tags.append(Tag("?", body[i:j], body[start:j]))
            i = j
            continue
        i += len(name)
        while i < n and body[i] == " " and name not in _FREE_TEXT_ARG:
            i += 1
        if i < n and body[i] == "(":
            depth, j = 0, i
            while j < n:
                if body[j] == "(":
                    depth += 1
                elif body[j] == ")":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            arg = body[i + 1 : j - 1] if j - 1 >= i + 1 and body[j - 1] == ")" else body[i + 1 : j]
            tags.append(Tag(name, arg, body[start:j], paren=True))
            i = j
        else:
            j = i
            while j < n and body[j] != "\\":
                j += 1
            arg = body[i:j]
            if name not in _FREE_TEXT_ARG:
                arg = arg.strip()
            tags.append(Tag(name, arg, body[start:j]))
            i = j
    return tags


def canonical(name: str) -> str:
    """Group aliases: \\K is \\kf, \\fr is \\frz, \\c is \\1c, \\fad/\\fade are fades."""
    return {"K": "kf", "fr": "frz", "c": "1c"}.get(name, name)


KARAOKE_TAGS = {"k", "K", "kf", "ko"}
POSITION_TAGS = {"pos", "move", "org", "an", "a"}
FONT_TAGS = {"fn", "fs", "fscx", "fscy", "fsp", "fe", "b", "i"}


def tags_to_block(tags: list[Tag]) -> str:
    if not tags:
        return ""
    return "{" + "".join(t.raw for t in tags) + "}"
