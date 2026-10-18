"""Qt-free dialogue readings, per-line CTC edges and timing suggestions.

Numerical rules match tools/eval/dialogue_{align,tpp,chain}_eval.py.
Times are absolute milliseconds; emission and RMS frames start at clip_ms.
No recognition models are imported or loaded here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Optional

import numpy as np

from ..core import karaoke
from ..core.readings import visible_map
from ..textutil import is_kanji
from . import tokens, viterbi
from .song import ModelEmission


@dataclass
class DialogueLine:
    gid: str
    mid: str
    start: int
    end: int
    kana: str
    signature: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> DialogueLine:
        return cls(**json.loads(value))


@dataclass
class DialogueEdge:
    mid: str
    speech_start: int
    speech_end: int
    conf: float
    flags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> DialogueEdge:
        return cls(**json.loads(value))


def _clean_reading(text: str) -> str:
    # ー is an alignment vowel extension, rather than punctuation.
    return "".join(c for c in tokens.kata_to_hira(text)
                   if "ぁ" <= c <= "ゖ" or c == "ー" or c.isascii() and c.isalpha())


def spoken_kana(doc, member, analyzer=None, particles=None) -> tuple[str, str]:
    """Use occurrence ruby first, Sudachi for unannotated kanji, then particles.

    Unannotated spans are tokenized together (今日 and 行く must not be read
    kanji by kanji). Kana and Latin spelling are retained; Latin only becomes
    targets in MMS. Hidden ruby is ignored by karaoke.derive. The optional
    analyzer is borrowed; the reading engine's singleton is loaded lazily.
    """
    visible, ids = visible_map(member)
    text = visible.text
    beats = karaoke.derive(doc, member)
    # Callers validating many lines pass particle offsets computed in one batch: creating a Sudachi
    # dictionary per line costs ~70 ms (tokens.particle_positions_many on many texts loads it once).
    particles = tokens.particle_positions_many([text])[0] if particles is None else particles
    positions = {identity: i for i, identity in enumerate(ids)}
    out = []
    i = 0
    while i < len(beats):
        beat = beats[i]
        if beat.ann is not None:
            out.append(beat.sung)
            # derive can attach a trailing ん/ー/small kana to a ruby beat.
            if beat.k + 1 == beat.n:
                end = max(positions[p] for p in beat.core) + 1
                out.append(text[end:beat.b])
            i += 1
            continue
        j = i + 1
        while j < len(beats) and beats[j].ann is None:
            j += 1
        block = beats[i:j]
        a, z = block[0].a, block[-1].b
        if any("no_reading" in b.flags for b in block):
            if analyzer is None:
                from ..reading.engine import Analyzer
                analyzer = Analyzer.get()
            for morph in analyzer.tokenize(text[a:z]):
                if any(is_kanji(c) for c in morph.surface):
                    out.append(morph.reading or morph.surface)
                else:
                    out.append("".join({"は": "わ", "へ": "え"}.get(c, c)
                                       if a + morph.start + k in particles else c
                                       for k, c in enumerate(morph.surface)))
        else:
            # Use core offsets: beat.a may include leading punctuation.
            for b in block:
                for part in b.parts:
                    at = positions[part.core[0]]
                    out.append(tokens.sung_for_alignment(part.sung, at in particles, False))
        i = j
    kana = _clean_reading("".join(out))
    payload = json.dumps([kana, text], ensure_ascii=False, separators=(",", ":"))
    return kana, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _align_line(line, model, rms, silent_db, clip_ms, window_ms):
    lp, frame_ms = np.asarray(model.log_probs), model.frame_ms
    if model.name == "mms":
        target = [model.vocab[c] for c in tokens.romaji(line.kana) if c in model.vocab]
    else:
        target = tokens.kana_tokens(line.kana, model.vocab)
    if not target:
        return None
    # Truncate before clipping, including the rightmost frame, as in the eval.
    a = max(0, int((line.start - window_ms - clip_ms) / frame_ms))
    z = min(len(lp), int((line.end + window_ms - clip_ms) / frame_ms) + 1)
    if z - a < len(target) + 4:
        return None
    # Cached whole episodes are float16; compute only this window in float32.
    local = lp[a:z].astype(np.float32, copy=False)
    star = model.star
    if star is None:
        others = np.delete(local, model.blank, axis=1)
        local = np.concatenate([local, others.max(axis=1, keepdims=True) - 1.0], axis=1)
        star = local.shape[1] - 1
    full = (0, len(local) - 1)
    try:
        first, last, _score = viterbi.align(local, [[star], target, [star]],
                                           [full, full, full], blank=model.blank)
    except ValueError:
        return None
    shift = model.shift_ms if model.shift_ms is not None else (-10 if model.name == "mms" else -50)
    start = clip_ms + (a + first[1]) * frame_ms + shift
    last_end = clip_ms + (a + last[1] + 1) * frame_ms
    begin = max(0, int((last_end - clip_ms) // 10))
    run, end = 0, last_end
    for j in range(begin, min(len(rms), begin + 80)):
        run = run + 1 if rms[j] < silent_db else 0
        if run >= 15:
            end = clip_ms + (j - run + 1) * 10
            break
    conf = float(np.exp(local[first[1]:last[1] + 1][:, target]).max(axis=1).mean())
    return DialogueEdge(line.mid, int(round(start)), int(round(max(end, last_end))), conf)


def fuse(a: Optional[DialogueEdge], b: Optional[DialogueEdge], tol_ms: int = 200) -> Optional[DialogueEdge]:
    """Eval's floor mean on agreement, confidence winner on disagreement."""
    if a is None and b is None:
        return None
    if a is None or b is None:
        edge = a if a is not None else b
        return DialogueEdge(edge.mid, edge.speech_start, edge.speech_end, edge.conf,
                            list(dict.fromkeys([*edge.flags, "one_model"])))
    if a.mid != b.mid:
        raise ValueError("cannot fuse different dialogue members")
    flags = list(dict.fromkeys([*a.flags, *b.flags]))
    if abs(a.speech_start - b.speech_start) <= tol_ms and abs(a.speech_end - b.speech_end) <= tol_ms:
        return DialogueEdge(a.mid, (a.speech_start + b.speech_start) // 2,
                            (a.speech_end + b.speech_end) // 2, max(a.conf, b.conf), flags)
    edge = a if a.conf >= b.conf else b
    return DialogueEdge(edge.mid, edge.speech_start, edge.speech_end, edge.conf,
                        list(dict.fromkeys([*flags, "disagree"])))


def align_dialogue(lines: list[DialogueLine], emissions: list[ModelEmission], rms_db_10ms,
                   clip_ms: int = 0, *, window_ms: int = 1000, progress=None) -> list[Optional[DialogueEdge]]:
    """Align independently in each current window; absent targets/paths abstain.

    Accept one or two emissions; an absent model is flagged on a usable edge.
    Unlike song alignment, kana separators are not merged into blank scores.
    All input arrays are borrowed and remain unchanged.
    """
    if window_ms < 0 or len(emissions) > 2:
        raise ValueError("expected nonnegative window and at most two emissions")
    rms = np.asarray(rms_db_10ms)
    if rms.ndim != 1 or np.isnan(rms).any():
        raise ValueError("RMS must be a one-dimensional dB array without NaN")
    silent_db = float(np.percentile(rms, 80) - 20) if len(rms) else -np.inf
    names = set()
    for model in emissions:
        lp = np.asarray(model.log_probs)
        name = "kana" if model.name == "hira" else model.name
        if name not in ("mms", "kana") or name in names:
            raise ValueError("expected distinct MMS/kana models")
        names.add(name)
        if (lp.ndim != 2 or not np.issubdtype(lp.dtype, np.floating)
                or not np.isfinite(model.frame_ms) or model.frame_ms <= 0
                or not 0 <= model.blank < lp.shape[1] or lp.shape[1] < 2
                or np.isnan(lp).any() or np.isposinf(lp).any()
                or model.star is not None and (not 0 <= model.star < lp.shape[1] or model.star == model.blank)):
            raise ValueError("invalid model emission")
    out = []
    for line in lines:
        edges = [_align_line(line, model, rms, silent_db, clip_ms, window_ms) for model in emissions]
        out.append(fuse(edges[0] if edges else None, edges[1] if len(edges) > 1 else None))
        if progress is not None:
            progress(len(out), len(lines))
    return out


def _snap_end(end, keyframes, before, after):
    if not (before or after):
        return end
    i = np.searchsorted(keyframes, end - before)
    best = None
    while i < len(keyframes) and keyframes[i] <= end + after:
        if best is None or abs(keyframes[i] - end) < abs(best - end):
            best = keyframes[i]
        i += 1
    return int(best) if best is not None else end


def suggest_times(edges, times, keyframes_ms, *, lead_in_ms=165, lead_out_ms=250,
                  gap_ms=40, end_before_ms=300, end_after_ms=250,
                  previous="current") -> list[Optional[tuple[int, int]]]:
    """Fixed eval post-processor, followed by previous-row safety.

    Keyframes must be sorted. Ties prefer the earlier frame; windows include
    both bounds. Starts are never snapped. Previous rows use their current
    end by default; 'suggested' uses the previous accepted suggestion's end,
    falling back to its current end. None disables this clamp for evaluation.
    Next rows use a suggested start when present.
    A collapsed span abstains, leaving the row unchanged.
    """
    if len(edges) != len(times):
        raise ValueError("edges and current times must have equal lengths")
    if previous not in ("current", "suggested", None):
        raise ValueError("previous must be 'current', 'suggested' or None")
    starts = [None if edge is None else max(0, edge.speech_start - lead_in_ms) for edge in edges]
    out = []
    for i, edge in enumerate(edges):
        if edge is None:
            out.append(None)
            continue
        start, end = starts[i], edge.speech_end + lead_out_ms
        nxt = None if i + 1 == len(times) else (starts[i + 1] if starts[i + 1] is not None else times[i + 1][0])
        if nxt is not None and nxt > start and end > nxt:
            end = nxt - gap_ms
        chained = nxt is not None and end == nxt - gap_ms
        if not chained:
            snapped = _snap_end(end, keyframes_ms, end_before_ms, end_after_ms)
            if nxt is None or snapped <= nxt - gap_ms:
                end = snapped
        if i and previous is not None:
            prev_end = out[i - 1][1] if previous == "suggested" and out[i - 1] is not None else times[i - 1][1]
            start = max(start, prev_end + gap_ms)
        out.append((int(start), int(end)) if end > start else None)
    return out


def learn_habit(edges, manual_edges, stored=None, minimum=20) -> tuple[int, int, str, int, int]:
    """Learn padding from ordered current-row records keyed by member id.

    Each record has mid, start, end; optional time_src selects only 'manual'
    boundaries. Without time_src, non-None boundaries are manual samples
    (useful for gold evaluation). Keep neighbouring rows, including those
    without an aligned edge, to exclude ends with gaps <=50 ms. next_start
    may explicitly supply the next current start when pooling episodes or
    selecting a subset; None denotes an episode's last row. stored is a
    (lead_in, lead_out) pair. Both sample counts must reach minimum.
    """
    pred = {edge.mid: edge for edge in edges if edge is not None}
    rows = list(manual_edges)
    onsets, offsets = [], []
    for i, row in enumerate(rows):
        edge = pred.get(row["mid"])
        if edge is None:
            continue
        source = row.get("time_src")
        start, end = row.get("start"), row.get("end")
        if start is not None and (source is None or source.get("start") == "manual"):
            onsets.append(edge.speech_start - start)
        nxt = row.get("next_start") if "next_start" in row else rows[i + 1].get("start") if i + 1 < len(rows) else None
        if end is not None and (source is None or source.get("end") == "manual") and (nxt is None or nxt - end > 50):
            offsets.append(end - edge.speech_end)
    counts = (len(onsets), len(offsets))
    if onsets and offsets and min(counts) >= minimum:
        return (int(np.clip(np.median(onsets), -100, 600)),
                int(np.clip(np.median(offsets), -200, 800)), "habit", *counts)
    if stored is not None:
        return int(stored[0]), int(stored[1]), "stored", *counts
    return 165, 250, "default", *counts
