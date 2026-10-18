"""``subkit voice S01E001`` — settle uncertain readings with the voice track.

Looks at every ``?`` entry of the ``@读音`` records (and every 私, unless the
dictionary/character table already decided it), scores the candidates against
the line's audio (see reading/voice.py) and, when the acoustic margin is
clear, writes the winner as if you had edited the record.  A ``※🔊`` note on
the row says what was decided and how sure it was.

Only changes leave a trace: a 私 the voice confirms (or cannot decide) stays
as it is without a note; an undecided ``?`` entry gets a note with the scores
because you have to decide it anyway.  Re-running is safe (scores are cached,
identical notes are not added twice).
"""

from __future__ import annotations

import time
from pathlib import Path
from .local_paths import local_value
from typing import Callable

from .ass.document import AssDocument, Event
from .ass.text import VisibleText
from .project import Episode
from .reading.records import parse_record
from .reading.voice import VoiceCache, WhisperScorer, load_audio
from .workfile import PERSIST_NOTE, STYLE_NOTE, _occurrence, collect_rows, refresh

Log = Callable[[str], None]
WATASHI = ["わたし", "わたくし", "あたし"]


def voice_check(ep: Episode, *, log: Log = print, model: str = "medium", min_margin: float = 3.0,
                watashi: bool = True) -> int:
    cfg = ep.project.config
    video = ep.video()
    if video is None:
        raise FileNotFoundError("找不到视频")
    doc = AssDocument.load(ep.work_ass)
    rows, _ = collect_rows(doc)
    state = ep.load_state()
    cache = VoiceCache(ep.project.cache_dir / "voice.sqlite3")
    scorer = None
    decided = undecided = 0
    t0 = time.time()
    new_notes: list[Event] = []
    existing = {(ev.effect, ev.text) for ev in doc.events if ev.kind == "Comment" and ev.style == STYLE_NOTE}
    for rid, row in sorted(rows.items()):
        if row.record is None or row.jp is None:
            continue
        text = VisibleText(row.jp.text).text
        rs = state["rows"].setdefault(str(rid), {})
        cursor = 0
        for e in parse_record(row.record.text):
            pos = text.find(e.surface, cursor)
            if pos < 0:
                continue
            cursor = pos + len(e.surface)
            if e.uncertain and e.alternatives:
                cands = [e.reading] + [a for a in e.alternatives if a != e.reading]
            elif watashi and e.surface == "私" and not e.forced and not e.from_dict:
                cands = WATASHI
            else:
                continue
            texts = [text[:pos] + c + text[pos + len(e.surface):] for c in cands]
            key = cache.key(str(video), row.jp.start, row.jp.end, texts, model)
            scores = cache.get(key)
            if scores is None:
                if scorer is None:
                    log(f"  载入 whisper {model}（CPU 上每行约 10–30 秒）…")
                    scorer = WhisperScorer(model, download_root=cfg.data.get("reading", {}).get("whisper_dir", local_value("SUBKIT_ASR_MODEL_DIR", Path.home() / ".cache" / "whisper")))
                audio = load_audio(video, row.jp.start / 100 - 0.3, (row.jp.end - row.jp.start) / 100 + 0.6, cfg["tools"]["ffmpeg"])
                scores = scorer.scores(audio, texts)
                cache.put(key, scores)
            order = sorted(range(len(cands)), key=lambda i: -scores[i])
            best, second = order[0], order[1]
            margin = scores[best] - scores[second]
            flagged = bool(e.uncertain and e.alternatives)
            if margin >= min_margin:
                if not flagged and cands[best] == e.reading:
                    continue  # the voice agrees with the record: nothing to report
                decided += 1
                edit = {"surface": e.surface, "nth": _occurrence(text, e.surface, pos), "reading": cands[best],
                        "hidden": False, "forced": False, "scale": e.scale, "mono": False}
                manual = [m for m in rs.get("manual", []) if not (m["surface"] == e.surface and m.get("nth", 0) == edit["nth"])]
                rs["manual"] = manual + [edit]
                msg = f"{PERSIST_NOTE}🔊按配音判定：{e.surface}→{cands[best]}（领先 {margin:.1f}）"
            elif flagged:
                undecided += 1
                msg = f"{PERSIST_NOTE}🔊配音也难以区分：{e.surface} " + " / ".join(f"{cands[i]} {scores[i]:+.1f}" for i in order)
            else:
                continue  # 私 stays as the record has it
            if (f"#{rid}", msg) not in existing:
                existing.add((f"#{rid}", msg))
                new_notes.append(Event(kind="Comment", start=row.jp.start, end=row.jp.end, style=STYLE_NOTE, effect=f"#{rid}", text=msg))
    if new_notes or decided:
        doc.events.extend(new_notes)
        doc.save(ep.work_ass)
        ep.save_state(state)
        refresh(ep, log=log)
    log(f"  配音判定：采用 {decided}，难以区分 {undecided}，用时 {time.time() - t0:.0f}s")
    return decided
