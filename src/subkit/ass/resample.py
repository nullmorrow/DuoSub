"""Change a script's PlayRes the way Aegisub's "Resample Resolution" does.

Scales style metrics, margins and every coordinate-like override tag
(\\pos \\move \\org \\clip drawings \\fs \\fsp \\bord \\shad \\blur, \\p drawings).
Border/shadow handling respects ScaledBorderAndShadow semantics.
"""

from __future__ import annotations

import re
from typing import Optional

from .document import AssDocument, _fmt_num
from .tags import Tag
from .text import tokenize

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _scale_drawing(cmds: str, sx: float, sy: float) -> str:
    """Scale the numbers of a drawing command string (m/l/b/s/p/c/n)."""
    out, axis = [], 0
    pos = 0
    for m in re.finditer(r"[a-zA-Z]|[-+]?\d+(?:\.\d+)?", cmds):
        out.append(cmds[pos : m.start()])
        tok = m.group(0)
        if tok.isalpha():
            axis = 0
            out.append(tok)
        else:
            v = float(tok) * (sx if axis == 0 else sy)
            axis ^= 1
            out.append(_fmt_num(round(v, 2)))
        pos = m.end()
    out.append(cmds[pos:])
    return "".join(out)


def _scale_tag(t: Tag, sx: float, sy: float, sb: float) -> str:
    name = t.name
    if name in ("pos", "org") and t.paren:
        a = t.float_args()
        if a and len(a) >= 2:
            return f"\\{name}({_fmt_num(round(a[0] * sx, 2))},{_fmt_num(round(a[1] * sy, 2))})"
    elif name == "move" and t.paren:
        a = t.args()
        try:
            nums = [float(x) for x in a[:4]]
        except ValueError:
            return t.raw
        if len(nums) == 4:
            scaled = [nums[0] * sx, nums[1] * sy, nums[2] * sx, nums[3] * sy]
            rest = a[4:]
            return "\\move(" + ",".join([_fmt_num(round(v, 2)) for v in scaled] + rest) + ")"
    elif name in ("clip", "iclip") and t.paren:
        a = t.args()
        try:
            nums = [float(x) for x in a]
            if len(nums) == 4:
                return f"\\{name}(" + ",".join(
                    _fmt_num(round(v, 2)) for v in (nums[0] * sx, nums[1] * sy, nums[2] * sx, nums[3] * sy)
                ) + ")"
        except ValueError:
            # vector clip: optional scale factor then drawing commands
            if len(a) == 2:
                return f"\\{name}({a[0]},{_scale_drawing(a[1], sx, sy)})"
            return f"\\{name}({_scale_drawing(t.arg, sx, sy)})"
    elif name == "fs":
        v = t.num()
        if v is not None and v > 0 and not t.arg.strip().startswith(("+", "-")):
            return f"\\fs{_fmt_num(round(v * sy, 2))}"
    elif name == "fsp":
        v = t.num()
        if v is not None:
            return f"\\fsp{_fmt_num(round(v * sx, 2))}"
    elif name in ("bord", "shad", "xbord", "ybord", "xshad", "yshad", "blur"):
        v = t.num()
        if v is not None:
            return f"\\{name}{_fmt_num(round(v * sb, 2))}"
    elif name == "t" and t.paren:
        # scale nested tags inside \t(...): keep timing args, rescale the tag part
        args = t.args()
        tag_part = args[-1]
        from .tags import parse_block

        inner = "".join(_scale_tag(x, sx, sy, sb) for x in parse_block(tag_part)) if "\\" in tag_part else tag_part
        return "\\t(" + ",".join(args[:-1] + [inner]) + ")"
    return t.raw


def scale_text(text: str, sx: float, sy: float, sb: float) -> str:
    out = []
    for tok in tokenize(text):
        if tok.kind == "tags":
            out.append("{" + "".join(_scale_tag(t, sx, sy, sb) for t in tok.tags) + "}")
        elif tok.kind == "drawing":
            out.append(_scale_drawing(tok.raw, sx, sy))
        else:
            out.append(tok.raw)
    return "".join(out)


def scale_positions(text: str, sx: float) -> str:
    """Convert document-owned horizontal coordinates; keep font/drawing sizes and other raw bytes."""
    from .tags import parse_block

    def block(body: str) -> str:
        out, offset = [], 0
        for tag in parse_block(body):
            start = body.find(tag.raw, offset)
            out.append(body[offset:start])
            raw = tag.raw
            if tag.name in ("pos", "move", "org", "clip", "iclip"):
                raw = _scale_tag(tag, sx, 1.0, 1.0)
            elif tag.name == "t" and tag.paren:
                inner = block(tag.arg)
                if inner != tag.arg:
                    raw = raw.replace(tag.arg, inner, 1)
            out.append(raw)
            offset = start + len(tag.raw)
        out.append(body[offset:])
        return "".join(out)

    if sx == 1:
        return text
    return "".join("{" + block(tok.raw[1:-1]) + "}" if tok.kind == "tags" else tok.raw
                   for tok in tokenize(text))


def resample(doc: AssDocument, new_x: int, new_y: int, *, video_height: Optional[int] = None) -> float:
    """Resample in place.  Returns the vertical scale factor."""
    old_x, old_y = doc.playres
    sx = new_x / old_x
    sy = new_y / old_y
    # Border / shadow: in "ScaledBorderAndShadow: yes" they are script pixels and
    # scale with PlayRes; otherwise they are video pixels.  After resampling we
    # always switch to "yes", so convert video pixels to new script pixels.
    if doc.scaled_border:
        sb = sy
    else:
        vh = video_height or new_y
        sb = new_y / vh
    for st in doc.styles:
        for key, f in (("Fontsize", sy), ("Spacing", sx), ("MarginL", sx), ("MarginR", sx), ("MarginV", sy)):
            st.set(key, round(st.num(key) * f, 2) if key in ("Fontsize", "Spacing") else round(st.num(key) * f))
        st.set("Outline", round(st.num("Outline") * sb, 2))
        st.set("Shadow", round(st.num("Shadow") * sb, 2))
    for ev in doc.events:
        ev.margin_l = int(round(ev.margin_l * sx))
        ev.margin_r = int(round(ev.margin_r * sx))
        ev.margin_v = int(round(ev.margin_v * sy))
        if "{" in ev.text or "\\p" in ev.text:
            ev.text = scale_text(ev.text, sx, sy, sb)
    doc.set_info("PlayResX", new_x)
    doc.set_info("PlayResY", new_y)
    doc.set_info("ScaledBorderAndShadow", "yes")
    return sy
