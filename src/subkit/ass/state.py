"""Which override tags are in effect at a visible character.

Two users: lines that follow another line (furigana and pinyin take the rotation, shear, scaling and paint of the
characters they sit on), and formatted spans, which must hand back exactly what was in effect before them.

libass semantics this relies on (measured, see tests/test_tags_raw.py):
* \\frz \\frx \\fry rotate every glyph around the line's \\org, or its anchor point when there is none;
* \\fax slants each glyph around its own top line (the glyph's position does not move);
* \\fay moves glyphs down cumulatively along the line: + fay × advance per glyph, starting at 0 on each line.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional

from .document import Style
from .tags import Tag, canonical, parse_block
from .text import tokenize

ROTATE = ("frz", "frx", "fry")
SHEAR = ("fax", "fay")
TRANSFORM = ROTATE + SHEAR
SCALE = ("fscx", "fscy")
ALPHAS = ("1a", "2a", "3a", "4a")
PAINT = ("1c", "2c", "3c", "4c", "alpha") + ALPHAS + ("blur", "be")
# a line takes these once (libass: the first one wins) — they cannot apply to part of a line
LINE_ONLY = {"pos", "move", "org", "an", "a", "clip", "iclip", "fad", "fade", "q"}
POSITION = {"pos", "move", "an", "a"}
KARAOKE = {"k", "kf", "ko", "kt"}
# inside \t these change the layout; a separate line (furigana) cannot follow such an animation
LAYOUT_ANIMATED = {"fs", "fscx", "fscy", "fsp", "fn", "b", "i", "fe"}
# names that are the same property (removing one must remove the other)
ALIASES = {"frz": {"frz", "fr"}, "1c": {"1c", "c"}, "kf": {"kf", "K"}}


def names_of(raw: str) -> set[str]:
    """Every spelling of the tags in an override string (\\frz also removes \\fr, \\1c also \\c …)."""
    out: set[str] = set()
    for t in parse_block(raw):
        c = canonical(t.name)
        out |= ALIASES.get(c, {c, t.name})
    return out


def state_at(text: str, vis: int, names: Iterable[str]) -> tuple["OrderedDict[str, Tag]", list[Tag]]:
    """The last tag of each canonical name in ``names`` before visible character ``vis`` — ordered by when it was
    last set, so re-emitting them in order reproduces the state — and every \\t before it.  \\r resets both."""
    want = {canonical(n) for n in names}
    st: OrderedDict[str, Tag] = OrderedDict()
    ts: list[Tag] = []
    n = 0
    for tok in tokenize(text):
        if tok.kind == "tags":
            for t in tok.tags:
                c = canonical(t.name)
                if c == "r":
                    st.clear()
                    ts.clear()
                elif c == "t":
                    ts.append(t)
                elif c in want:
                    st.pop(c, None)
                    st[c] = t
        elif tok.kind in ("text", "hardspace", "softbreak"):
            n += len(tok.raw) if tok.kind == "text" else 1
            if n > vis:
                break
    return st, ts


def split_t(t: Tag) -> tuple[list[str], list[Tag]]:
    """\\t(t1,t2,accel,tags) → (timing arguments, inner tags)."""
    args = t.args()
    if not args:
        return [], []
    return args[:-1], parse_block(args[-1])


def filter_t(t: Tag, allowed: set[str]) -> tuple[Optional[str], set[str]]:
    """The \\t with only the inner tags in ``allowed`` (None when nothing is left), and the names it dropped."""
    timing, inner = split_t(t)
    keep = [x.raw for x in inner if canonical(x.name) in allowed]
    dropped = {canonical(x.name) for x in inner if canonical(x.name) not in allowed}
    if not keep:
        return None, dropped
    return "\\t(" + ",".join(timing + ["".join(keep)]) + ")", dropped


# ------------------------------------------------------------------ the style's own values, as tags
def _colour(style: Style, key: str) -> tuple[str, str]:
    v = (style.get(key) or "&H00000000").upper().replace("&H", "").replace("&", "").rjust(8, "0")[-8:]
    return v[2:], v[:2]          # BBGGRR, AA


def style_tag(name: str, style: Optional[Style]) -> str:
    """The tag that sets property ``name`` to the style's value (or the ASS default)."""
    st = style if style is not None else Style.new("Default")
    n = canonical(name)
    num = lambda v: f"{float(v):g}"  # noqa: E731
    colour_keys = {"1": "PrimaryColour", "2": "SecondaryColour", "3": "OutlineColour", "4": "BackColour"}
    if n in ("1c", "2c", "3c", "4c"):
        return f"\\{n}&H{_colour(st, colour_keys[n[0]])[0]}&"
    if n in ALPHAS:
        return f"\\{n}&H{_colour(st, colour_keys[n[0]])[1]}&"
    if n == "alpha":
        return "".join(style_tag(a, st) for a in ALPHAS)
    table = {
        "fn": f"\\fn{st.fontname}", "fs": f"\\fs{num(st.fontsize)}", "b": f"\\b{1 if st.bold else 0}",
        "i": f"\\i{1 if st.italic else 0}", "u": f"\\u{1 if st.num('Underline') else 0}",
        "s": f"\\s{1 if st.num('StrikeOut') else 0}", "fscx": f"\\fscx{num(st.scale_x)}",
        "fscy": f"\\fscy{num(st.scale_y)}", "fsp": f"\\fsp{num(st.spacing)}", "frz": f"\\frz{num(st.num('Angle'))}",
        "frx": "\\frx0", "fry": "\\fry0", "fax": "\\fax0", "fay": "\\fay0",
        "bord": f"\\bord{num(st.outline)}", "xbord": f"\\xbord{num(st.outline)}", "ybord": f"\\ybord{num(st.outline)}",
        "shad": f"\\shad{num(st.shadow)}", "xshad": f"\\xshad{num(st.shadow)}", "yshad": f"\\yshad{num(st.shadow)}",
        "blur": "\\blur0", "be": "\\be0", "fe": f"\\fe{int(st.num('Encoding', 1))}", "p": "\\p0",
    }
    return table.get(n, "")


def restore_tags(text: str, vis: int, names: Iterable[str], style: Optional[Style]) -> str:
    """Tags that put each property in ``names`` back to what was in effect just before visible char ``vis``: the
    line's own last setting if it has one, else the style's value."""
    order = list(dict.fromkeys(canonical(n) for n in names))
    track = set(order) | {"alpha", "bord", "shad"} | set(ALPHAS)
    st, _ts = state_at(text, vis, track)
    rank = {k: i for i, k in enumerate(st)}
    out: list[str] = []

    def one(n: str) -> str:
        if n in ALPHAS:                                  # \alpha sets all four; whichever came last wins
            own, all_ = st.get(n), st.get("alpha")
            if all_ is not None and (own is None or rank["alpha"] > rank[n]):
                return f"\\{n}{all_.arg}"
            return own.raw if own is not None else style_tag(n, style)
        if n == "alpha":
            return "".join(one(a) for a in ALPHAS)
        if n in ("xbord", "ybord", "xshad", "yshad"):
            whole = n[1:]
            own, all_ = st.get(n), st.get(whole)
            if all_ is not None and (own is None or rank[whole] > rank[n]):
                return f"\\{n}{all_.arg}"
            return own.raw if own is not None else style_tag(n, style)
        if n in st:
            return st[n].raw
        return style_tag(n, style)

    for n in order:
        if n == "t":
            continue
        out.append(one(n))
    return "".join(out)


def strip_inside(text: str, a: int, b: int, names: Iterable[str]) -> str:
    """Remove the tags named ``names`` from the override blocks that sit between visible characters a and b-1 (they
    would change the look half-way through a formatted span).  Blocks before a and after b-1 are left alone."""
    names = set(names)
    if b - a < 2 or not names:
        return text
    out: list[str] = []
    n = 0
    for tok in tokenize(text):
        if tok.kind == "tags" and a < n < b:
            kept = [t.raw for t in tok.tags if t.name not in names]
            if kept:
                out.append("{" + "".join(kept) + "}")
            continue
        out.append(tok.raw)
        if tok.kind == "text":
            n += len(tok.raw)
        elif tok.kind in ("hardspace", "softbreak"):
            n += 1
    return "".join(out)


def animated_names(raw: str) -> set[str]:
    """Properties a raw string changes, including those inside its \\t (they need restoring after a span)."""
    out: set[str] = set()
    for t in parse_block(raw):
        c = canonical(t.name)
        if c == "t":
            out |= {canonical(x.name) for x in split_t(t)[1]}
        else:
            out.add(c)
    return out


def check_raw(raw: str, *, span: bool) -> tuple[str, str]:
    """Validate override tags typed by the user.  Returns (normalised tags, error message — '' when fine)."""
    s = (raw or "").strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if not s:
        return "", ""
    if "{" in s or "}" in s:
        return "", "只写标签本身，中间不要有 { }"
    if not s.startswith("\\"):
        return "", "标签以 \\ 开头，例如 \\frz15（旋转）或 \\fscx120（横向拉伸）"
    tags = parse_block(s)

    def bad(t: Tag, inner: bool = False) -> str:
        c = canonical(t.name)
        if t.name == "?":
            return f"不认识的标签：{t.raw}"
        if c in KARAOKE:
            return "K 轴请用 K 轴工具（这里不写 \\k）"
        if c in POSITION:
            return "位置请直接拖动字幕或用位置设置（这里不写 \\pos、\\move、\\an）"
        if c == "p":
            return "不支持绘图模式（\\p）"
        if c == "r":
            return "\\r 会把整行重置成样式，这里不能用"
        if inner and c in ("t", "org", "fad", "fade", "q"):
            return f"\\t 里不能放 {t.raw}"
        if span and not inner and c in LINE_ONLY:
            return f"{t.raw} 只能用于整行（取消选择后再写）"
        if t.paren and t.raw.count("(") != t.raw.count(")"):
            return f"括号不配对：{t.raw}"
        return ""

    for t in tags:
        msg = bad(t)
        if msg:
            return "", msg
        if canonical(t.name) == "t":
            timing, inner = split_t(t)
            if not inner:
                return "", f"\\t 里没有标签：{t.raw}"
            try:
                [float(x) for x in timing]
            except ValueError:
                return "", f"\\t 的时间写法不对：{t.raw}（例：\\t(0,500,\\frz30)）"
            for x in inner:
                msg = bad(x, inner=True)
                if msg:
                    return "", msg
    return "".join(t.raw for t in tags), ""
