"""Release: strip everything the tool or the editor added for work, check, write.

Removed: @-styled comments (records, notes, settings), row ids in Effect,
{@…} inline marks, other Comment lines (unless kept), Aegisub project data,
unused styles.  Optional \\blur on dialogue/ruby lines.  A final check makes sure
nothing tool-specific leaked into the release.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Callable, Optional

from .ass.document import AssDocument
from .karaoke import fx_slot
from .ass.text import remove_private, tokenize
from .project import Episode
from .sources.cn import detect_cn_script
from .ass.text import plain_text

_ROW = re.compile(r"^#\d+(?::[A-Za-z]+)?\s*")


def _add_blur(text: str, value: float) -> str:
    toks = tokenize(text)
    if any(t.kind == "tags" and any(tag.name in ("blur", "be") for tag in t.tags) for t in toks):
        return text
    tag = f"\\blur{value:g}"
    if text.startswith("{") and toks and toks[0].kind == "tags":
        return "{" + tag + text[1:]
    return "{" + tag + "}" + text


def release(ep: Episode, *, log: Callable[[str], None] = print, blur: Optional[float] = None,
            output: Optional[Path] = None) -> Path:
    cfg = ep.project.config
    doc = AssDocument.load(ep.work_ass)
    rcfg = cfg["release"]
    blur = float(rcfg.get("blur", 0)) if blur is None else blur
    keep_comments = bool(rcfg.get("keep_comments", False))
    patterns = list(rcfg.get("blur_styles", ["CN*", "JP*", "OP.*", "ED.*", "IN.*"]))
    out = []
    removed = 0
    for ev in doc.events:
        if ev.style.startswith("@") or (ev.is_comment and not keep_comments):
            removed += 1
            continue
        fx = fx_slot(ev.text)
        ev = ev.copy(effect=_ROW.sub("", ev.effect or ""), text=remove_private(ev.text))
        if fx:
            # karaoke effect slot survives the release for a later effects script
            ev.effect = f"karaoke-fx:{fx}"
        if ev.is_dialogue and not plain_text(ev.text).strip() and "\\p" not in ev.text:
            removed += 1  # empty JP/CN lines left for pasting
            continue
        if ev.style.split(".")[0] in ("JP", "CN") and ev.name and ev.style.startswith("JP"):
            pass  # speaker names in Actor are harmless; keep them for reference
        if blur > 0 and any(fnmatch.fnmatchcase(ev.style, p) for p in patterns):
            ev.text = _add_blur(ev.text, blur)
        out.append(ev)
    doc.events = out
    used = {e.style for e in doc.events}
    doc.styles = [s for s in doc.styles if s.name in used and not s.name.startswith("@")]
    doc.drop_section("[Aegisub Project Garbage]")
    doc.drop_section("[Aegisub Extradata]")
    doc.script_info = [(k, v) for k, v in doc.script_info if k is not None]
    doc.set_info("ScriptType", "v4.00+")
    # leak check
    leaks = [e for e in doc.events if "{@" in e.text or e.style.startswith("@") or _ROW.match(e.effect or "")]
    if leaks:
        raise RuntimeError(f"发布检查失败：仍有 {len(leaks)} 行工程标注")
    lang = detect_cn_script([plain_text(e.text) for e in doc.events if e.style.startswith("CN")])
    path = output or ep.release_path(lang)
    if path is None:
        raise FileNotFoundError("找不到视频，无法决定发布文件位置（用 -o 指定）")
    doc.save(path, newline="\r\n")
    log(f"  删除工程行 {removed}；保留 {len(doc.events)} 行；Style {len(doc.styles)} 个；blur={blur:g}")
    return path
