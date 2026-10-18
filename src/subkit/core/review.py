"""审议包 — a clear export of one episode for the final review (by the user or a large language model).

The project file already ties every reading and karaoke beat to exact characters, but it is built for the
editor.  The review packet is a derived, plain view in which the relationships to check are explicit:

* Japanese (original) text with every reading on its kanji block — ``修業(しゅぎょう)が足(た)りねぇ``;
  okurigana stay outside the brackets; ``(…?)`` marks a reading that is still a guess;
* the Chinese (translation) text of the same group (pinyin the same way once enabled);
* who speaks, with the character sheet (Japanese / kana / Chinese names, labels, self reference, notes),
  so the reviewer can judge whether the wording fits the character;
* the editor's open questions: unconfirmed groups, guessed readings, unpaired lines, fillers that are not shown.

Markdown for reading, JSONL (one group per line) for tools.  The reviewer answers with JSON lines in the format
written at the top of the packet; SubKit imports them as suggestions to accept or reject — nothing is applied
automatically (REQUIREMENTS: automation only suggests).
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Optional

from ..languages import get_language, lane_label, language_name
from ..model.document import Document, Group, Member
from .readings import anchor_positions, display_annotations, pieces_of, visible_map
from .display import DisplayRules, display_name, member_display_text
from .replacements import replace_ass_text

RESPONSE_FORMAT = {
    "group": "g_…（组 ID，照抄）",
    "lang": "ja | zh",
    "kind": "reading（读音/拼音） | text（错字、漏字、标点） | pairing（原声与翻译没有对应） | speaker（说话人） | persona（用词不符合人设） | other",
    "target": "要改的原文片段（照抄）",
    "old": "现在的写法/读音",
    "new": "建议的写法/读音",
    "reason": "理由（简短）",
}

INSTRUCTIONS = """\
# 审议包：{episode}

这是双语字幕（原声 = {orig_lang}，翻译 = {trans_lang}）发布前的复查材料。请逐组检查：

1. **对应**：{trans_label}是否准确翻译了同一组的{orig_label}（漏译、错译、多出来的内容）。
2. **读音**：日文汉字上的读音是否正确（`漢字(かんじ)` 只注在汉字上，送假名在括号外）；带 `?` 的是还没确认的猜测，请重点看。
3. **拼音**（如果有）：中文字后面括号里的拼音是否正确，特别是多音字。
4. **人设**：说话人的自称、对别人的称呼、口癖、语气是否符合下面人物表里的人设；翻译的语气是否符合这个人物。
5. **文字**：错字、漏字、标点（不要只因为风格不同就改）。

不要改写没有问题的句子，不要统一标点或全半角（原文保留是有意的）。

**请只用下面的格式回答**，每条意见一行 JSON（可以多行），没有问题就回答 `[]`：

```json
{response}
```
"""


def annotated(doc: Document, m: Member, kind: str = "ruby", display: Optional[DisplayRules] = None) -> tuple[str, list[dict[str, Any]]]:
    """Visible text with readings written in after each annotated block, plus the reading list."""
    vt, ids = visible_map(m)
    text = vt.text
    bounds = list(range(len(text) + 1))
    transformed = display is not None and (display.converted_for(m) or display.replaced_for(m))
    deleted = set()
    if transformed:
        raw, bounds = member_display_text(m, display)
        deleted = replace_ass_text(m.text, display.replacements, lane=m.lane).deleted
        from ..ass.text import VisibleText

        text = VisibleText(raw).text
    inserts: dict[int, str] = {}
    readings: list[dict[str, Any]] = []
    anns = sorted((x for x in doc.annotations.values() if x.member == m.id and x.kind == kind), key=lambda x: x.id)
    for a in display_annotations(doc, m, anns):
        pos = anchor_positions(a, ids)
        if pos is None:
            continue
        visible_pieces = [(pa, pb, t) for pa, pb, t in pieces_of(a)
                          if bounds[pos[0]+pb] > bounds[pos[0]+pa] and
                          not deleted.intersection(range(pos[0]+pa, pos[0]+pb))]
        if transformed and not visible_pieces:
            continue
        item = {"base": text[bounds[pos[0]]:bounds[pos[1]]] if transformed else a.base, "reading": a.text,
                "at": [bounds[pos[0]], bounds[pos[1]]], "status": a.status, "origin": a.origin}
        if transformed and deleted.intersection(range(pos[0], pos[1])):
            item["reading"] = (" " if kind == "pinyin" else "").join(t for _a, _b, t in visible_pieces)
        if a.candidates and len(a.candidates) > 1:
            item["candidates"] = a.candidates[:6]
        readings.append(item)
        if a.status == "hidden" or not a.text:
            continue
        if kind == "pinyin" and a.status != "confirmed" and len(set(a.candidates)) <= 1:
            continue
        mark = "?" if a.status == "uncertain" else ""
        for pa, pb, t in visible_pieces:
            at = bounds[pos[0] + pb]
            inserts[at] = inserts.get(at, "") + f"({t}{mark})"
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if i + 1 in inserts:
            out.append(inserts[i + 1])
    readings.sort(key=lambda r: r["at"][0])
    return "".join(out), readings


def _speaker(m: Member, chars) -> Optional[dict[str, Any]]:
    if (m.speaker or {}).get("character") == "nobody" or (chars is not None and chars.speaker_ignored(m)):
        return None
    sp = m.speaker or {}
    label = sp.get("label") or ""
    if not label and not sp.get("character"):
        return None
    out: dict[str, Any] = {"label": label}
    ch = next((c for c in chars.characters if c.id == sp.get("character")), None) if chars is not None else None
    if ch is not None:
        out["character"] = ch.id
        out["name"] = chars.display_name(ch.id, m.lang, label)
    elif sp.get("character") == "crowd":
        out["character"] = "crowd"
    return out


def _cell(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _fmt(ms: int) -> str:
    return f"{ms // 3600000}:{ms // 60000 % 60:02d}:{ms % 60000 / 1000:05.2f}"


def group_record(doc: Document, g: Group, n: int, chars=None, *, pinyin_mode: str = "off",
                 display: Optional[DisplayRules] = None) -> dict[str, Any]:
    s, e = doc.group_time(g)
    rec: dict[str, Any] = {"group": g.id, "n": n, "kind": g.kind, "time": [s, e], "time_text": f"{_fmt(s)}–{_fmt(e)}",
                           "confirmed": g.state == "confirmed"}
    orig, trans = [], []
    for m in doc.members_of(g):
        if m.role == "comment":
            continue
        shown_text = visible_map(m)[0].text
        if display is not None and (display.converted_for(m) or display.replaced_for(m)):
            from ..ass.text import VisibleText

            shown_text = VisibleText(member_display_text(m, display)[0]).text
        if get_language(m.lang).annotation_system == "pinyin" and pinyin_mode != "off":
            text, readings = annotated(doc, m, "pinyin", display)
            item = {"member": m.id, "lang": m.lang, "text": shown_text,
                    "annotated": text, "pinyin": readings}
        elif get_language(m.lang).annotation_system == "ruby":
            text, readings = annotated(doc, m, "ruby", display)
            item: dict[str, Any] = {"member": m.id, "lang": m.lang, "text": shown_text, "annotated": text,
                                    "readings": readings}
        else:
            item = {"member": m.id, "lang": m.lang, "text": shown_text}
            if m.lane == "original":
                item.update(annotated=item["text"], readings=[])
        sp = _speaker(m, chars)
        if sp:
            if display is not None and display.converted_for(m):
                sp["name"] = display_name(m, display, chars)
            item["speaker"] = sp
        (orig if m.lane == "original" else trans).append(item)
    rec["original"] = orig
    rec["translation"] = trans
    notes = []
    if orig and not trans:
        notes.append("只有原声（没有翻译）")
    if trans and not orig and g.kind == "dialogue":
        notes.append("只有翻译（没有原声）")
    hidden = [doc.comments[c].content for c in g.comments if doc.comments[c].kind in ("hidden_filler", "user_hidden")]
    if hidden:
        notes.append("未显示的语气词/文字：" + "、".join(hidden))
    if "order_exception" in g.flags:
        notes.append("用户接受的顺序例外")
    if any(r["status"] == "uncertain" for it in orig for r in it.get("readings", ())):
        notes.append("有未确认的读音（带 ?）")
    if notes:
        rec["notes"] = notes
    return rec


def build_packet(doc: Document, *, episode: str, orig_lang: str = "ja", trans_lang: str = "zh",
                 chars=None, kinds: tuple[str, ...] = ("dialogue", "song"),
                 pinyin_mode: str = "off", display: Optional[DisplayRules] = None,
                 check_cancelled: Optional[Callable[[], None]] = None) -> tuple[str, str]:
    """(markdown, jsonl) for all groups of the given kinds, in document order."""
    if check_cancelled:
        check_cancelled()
    recs = []
    for i, g in enumerate(doc.groups):
        if check_cancelled:
            check_cancelled()
        if g.kind in kinds:
            recs.append(group_record(doc, g, i + 1, chars, pinyin_mode=pinyin_mode, display=display))
    languages = {"original": orig_lang, "translation": trans_lang}
    md = [INSTRUCTIONS.format(episode=episode, orig_lang=language_name(orig_lang), trans_lang=language_name(trans_lang),
                              orig_label=lane_label("original", languages), trans_label=lane_label("translation", languages),
                              response=json.dumps({**RESPONSE_FORMAT, "lang": f"{orig_lang} | {trans_lang}"}, ensure_ascii=False))]
    if chars is not None and chars.characters:
        md.append("## 人物表\n")
        trans_name = language_name(trans_lang)
        headers = ["日语名", "假名", f"{trans_name}名", "其他名字", "字幕标签", "自称读法", "自称",
                   "对别人的称呼", "口癖", "语气 / 敬语", "备注（日语）", f"备注（{trans_name}）"]
        md.append("| " + " | ".join(headers) + " |\n|" + "---|" * len(headers))
        for ch in chars.characters:
            if check_cancelled:
                check_cancelled()
            refs = "；".join(f"{r['surface']}={r['reading']}" +
                            (f" ({json.dumps(r['scope'], ensure_ascii=False)})" if r.get("scope") else "")
                            for r in ch.self_refs)
            names = "；".join(f"{lang}: {ch.full_name(lang)}" for lang in ch.names
                             if lang not in ("ja", trans_lang))
            ja = ch.name_form("ja")
            cells = [chars.display_name(ch.id, "ja"), " ".join(ja.parts(readings=True)) if ja else "",
                     chars.display_name(ch.id, trans_lang), names, "/".join(ch.aliases), refs,
                     *(ch.persona.get(k, "") for k in ("self", "address", "catchphrases", "tone", "note_ja", "note_tr"))]
            md.append("| " + " | ".join(_cell(x) for x in cells) + " |")
        md.append("")
    md.append("## 字幕\n")
    for r in recs:
        if check_cancelled:
            check_cancelled()
        md.append(f"### #{r['n']} · {r['time_text']} · `{r['group']}`" + ("" if r["confirmed"] else " · 未确认"))
        for it in r["original"]:
            who = f"［{it['speaker'].get('name', it['speaker']['label'])}］" if it.get("speaker") else ""
            md.append(f"- {lane_label('original', languages)}{who}：{it['annotated']}")
        for it in r["translation"]:
            who = f"［{it['speaker'].get('name', it['speaker']['label'])}］" if it.get("speaker") else ""
            md.append(f"- {lane_label('translation', languages)}{who}：{it.get('annotated', it['text'])}")
        for nt in r.get("notes", ()):
            md.append(f"- ⚠ {nt}")
        md.append("")
    jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n"
    if check_cancelled:
        check_cancelled()
    return "\n".join(md), jsonl


def packet_paths(out_dir: Path, *, episode: str, stamp: str) -> tuple[Path, Path]:
    return (Path(out_dir) / f"{episode}.review-{stamp}.md",
            Path(out_dir) / f"{episode}.review-{stamp}.jsonl")


def available_packet_paths(out_dir: Path, *, episode: str, stamp: str) -> tuple[Path, Path]:
    """Choose a fresh pair without creating files or replacing an earlier packet."""
    suffix = 0
    while True:
        paths = packet_paths(out_dir, episode=episode, stamp=stamp + (f"-{suffix}" if suffix else ""))
        if not any(path.exists() for path in paths):
            return paths
        suffix += 1


def write_packet_files(packet: tuple[str, str], paths: tuple[Path, Path]) -> tuple[Path, Path]:
    """Write a built pair on the GUI thread, exclusively; remove only our new files on failure."""
    if len(packet) != 2 or len(paths) != 2:
        raise ValueError("审议包需要 Markdown 和 JSONL 两个文件")
    created = []
    try:
        with ExitStack() as stack:
            streams = []
            for path in paths:
                streams.append(stack.enter_context(path.open("x", encoding="utf-8")))
                created.append(path)
            for stream, text in zip(streams, packet):
                stream.write(text)
    except BaseException as exc:
        failures = []
        for path in created:
            try:
                path.unlink()
            except OSError as cleanup:
                failures.append(f"{path.name}：{cleanup}")
        if failures:
            raise OSError(f"{exc}；未能清理本次未完成的审议包：" + "；".join(failures)) from exc
        raise
    return paths


def write_packet(doc: Document, out_dir: Path, *, episode: str, stamp: str, **kw) -> tuple[Path, Path]:
    md, jl = build_packet(doc, episode=episode, **kw)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_md, p_jl = packet_paths(out_dir, episode=episode, stamp=stamp)
    p_md.write_text(md, encoding="utf-8")
    p_jl.write_text(jl, encoding="utf-8")
    return p_md, p_jl
