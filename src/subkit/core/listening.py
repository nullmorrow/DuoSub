"""Explicitly adopt a listening proposal as an inserted Japanese speech member."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

from ..ass.text import plain_text
from ..listening import Candidate, ListeningResult
from ..model.document import Member
from ..model.ids import new_id


def unavailable_reason(doc, gids, languages) -> str:
    if len(gids) != 1:
        return "请只选择一组只有中文的字幕"
    if languages.get("original", "ja") != "ja":
        return "AI听译目前支持日语原声"
    try:
        g = doc.group(gids[0])
    except KeyError:
        return "找不到这一组字幕"
    members = list(doc.members_of(g))
    if g.kind == "song":
        return "AI听译目前用于对白和屏幕字"
    if any(m.lane == "original" and m.role != "comment" for m in members):
        return "这一组已经有原文；不会覆盖已有原文"
    if not any(m.lane == "translation" and m.lang.startswith("zh") and m.role != "comment"
               and m.source_kind != "Comment" and plain_text(m.text).strip() for m in members):
        return "这一组没有可参考的中文译文"
    return ""


def target_signature(doc, gid: str, video: Path, audio_track, languages) -> dict:
    reason = unavailable_reason(doc, [gid], languages)
    if reason:
        raise ValueError(reason)
    stat = Path(video).stat()
    g = doc.group(gid)
    return {"group": asdict(g), "members": [asdict(m) for m in doc.members_of(g)],
            "video": str(Path(video).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "audio_track": audio_track, "languages": copy.deepcopy(languages)}


def adopt(editor, gid: str, candidate: Candidate, text: str, start: int, end: int,
          result: ListeningResult, *, languages=None, styles=None) -> str:
    """One undo transaction; retain all existing translation, sign and provenance data."""
    from .editor_text import editing_runs
    from .merge import MergeSettings
    from .readings import suggest_readings
    from ..reading.engine import ReadingEngine

    languages = languages or {"original": "ja", "translation": "zh-Hant"}
    reason = unavailable_reason(editor.doc, [gid], languages)
    if reason:
        raise ValueError(reason)
    text = text.strip()
    # A proposal is plain speech, not an ASS command entry point.
    if not text or any(ch in text for ch in ("{", "}", "\x00", "\r", "\n", "\u2028", "\u2029")) or "\\N" in text or "\\n" in text:
        raise ValueError("请填写一行日文正文；ASS 控制请补入后在编辑器设置")
    if type(start) is not int or type(end) is not int or not result.window_start_ms <= start < end <= result.window_end_ms:
        raise ValueError("日文时间须位于试听音频范围内，并且开始早于结束")
    g = editor.doc.group(gid)
    style = {**MergeSettings().styles, **(styles or {})}["original/dialogue"]
    member = Member(new_id("m_"), gid, "ja", "original", "dialogue", start, end,
                    style=style, runs=editing_runs(text, new_id("i_")))
    clean_candidate = lambda c: {k: v for k, v in asdict(c).items() if k != "audio"}
    member.extra["asr"] = {"version": 1, "adopted_by": "user", "model": copy.deepcopy(result.metadata),
                           "candidate_kind": candidate.kind,
                           "acoustically_confirmed": False,
                           "device": result.device, "window": [result.window_start_ms, result.window_end_ms],
                           "selected": clean_candidate(candidate), "adopted_text": text,
                           "adopted_time": [start, end], "candidates": [clean_candidate(c) for c in result.candidates],
                           "diagnostics": list(result.diagnostics)}
    engine = editor.reading_engine_factory() if editor.reading_engine_factory else ReadingEngine()
    annotations = suggest_readings(member, engine)

    def apply(tx):
        tx.keeps_text = True
        tx.add_member(member)
        tx.group(gid)
        g.members.insert(0, member.id)
        for ann in annotations:
            tx.add_annotation(ann)
        editor.doc.structure_changed()
        return member.id

    return editor.history.run("采用 AI听译原文", apply)
