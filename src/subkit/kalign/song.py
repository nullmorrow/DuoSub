"""Song-wide CTC assembly, beat fusion, vocal ends and start calibration.

Times are absolute milliseconds; ``clip_ms`` is the audio clip's start offset.
Only NumPy is imported here. Model loading belongs to the separate worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Optional

import numpy as np

from . import tokens, viterbi

# Shared voice/end thresholds; RMS bins describe the separated song vocals.
RMS_BIN_MS = 10
SILENCE_PERCENTILE = 80
SILENCE_DROP_DB = 18
SILENCE_MS = 100
VOICE_GATE_PENALTY = 6.0  # Nats: discourage targets in silence without banning them.
MIN_CONTINUATION_MS = 100
DEFAULT_CONTINUATION_RATIO = 0.5
MIN_CONTINUATION_SAMPLES = 5
CONTINUATION_RATIO_LIMITS = (0.3, 0.8)
MIN_BEAT_MS = 100  # Avoid visually squeezed karaoke beats where the span fits.
TRUST_TOLERANCE_MS = 60  # Agreement tighter than the ordinary fusion tolerance.
MAX_ALIGNMENT_ROUNDS = 3  # Bound refinement cost, including the initial pass.
ANCHOR_FRAMES = 1  # Model frame quantisation around a fixed fused beat start.
MORA_WEIGHT = 0.6  # Same mora/duration balance as translation redistribution.


def silent_start(start, silent, clip_ms):
    """True iff no observed 10 ms bin in [start, start+100 ms) is voiced."""
    if not len(silent):
        return False  # Missing RMS is unknown, preserving model-free callers.
    lo = max(0, int(np.floor((start - clip_ms) / RMS_BIN_MS)))
    hi = min(len(silent), int(np.ceil((start + SILENCE_MS - clip_ms) / RMS_BIN_MS)))
    return hi <= lo or bool(np.all(silent[lo:hi]))


def sokuon_only(reading):
    hira = tokens.kata_to_hira(reading)
    return "".join(c for c in hira if "ぁ" <= c <= "ゟ") == "っ"


@dataclass
class BeatInput:
    sung: str
    particle: bool = False
    from_ruby: bool = False
    fixed: Optional[tuple[int, int]] = None
    a: Optional[int] = None             # visible offset, when original text is supplied


@dataclass
class LineInput:
    mid: str
    start: int
    end: int
    beats: list[BeatInput]
    text: str = ""                     # visible original text, without ruby/ASS markup


@dataclass
class ModelEmission:
    name: str                          # mms | kana (prototype's hira is also accepted)
    log_probs: np.ndarray
    frame_ms: float
    vocab: dict[str, int]
    blank: int
    star: Optional[int] = None
    shift_ms: Optional[float] = None    # None selects MMS -10 / kana -50


@dataclass
class ModelBeat:
    start: int
    conf: float
    check: bool = False


@dataclass
class ModelResult:
    lines: list[list[ModelBeat]]
    flags: list[str] = field(default_factory=list)


@dataclass
class LineResult:
    mid: str
    beats: list[tuple[int, int, float, bool]]


@dataclass
class SongResult:
    lines: list[LineResult]
    offset_ms: int
    offset_source: str
    offset_count: int
    flags: list[str] = field(default_factory=list)
    continuation_ratio: Optional[float] = None
    continuation_source: str = "none"
    continuation_count: int = 0


def silence_mask(rms_db_10ms) -> np.ndarray:
    """Song silence mask for 10 ms bins, using percentile 80 minus 18 dB."""
    rms = np.asarray(rms_db_10ms, dtype=np.float64)
    if rms.ndim != 1 or np.isnan(rms).any():
        raise ValueError("RMS must be a one-dimensional dB array without NaN")
    return rms < np.percentile(rms, SILENCE_PERCENTILE) - SILENCE_DROP_DB if len(rms) else np.zeros(0, dtype=bool)


def _sung_lines(lines: list[LineInput]) -> list[list[str]]:
    # One dictionary for the entire song, shared by both alignment passes.
    texts = [line.text or "".join(b.sung for b in line.beats) for line in lines]
    positions = tokens.particle_positions_many(texts)
    out = []
    for line, particles in zip(lines, positions):
        offset, sung = 0, []
        for beat in line.beats:
            at = beat.a if beat.a is not None else offset
            sung.append(tokens.sung_for_alignment(beat.sung, beat.particle or at in particles, beat.from_ruby))
            offset += len(beat.sung)
        out.append(sung)
    return out


def _continuations(lines: list[LineInput], sung_lines: list[list[str]]) -> list[list[bool]]:
    return [tokens.continuation_morae(sung, tokens.word_starts([(b.a, b.from_ruby) for b in line.beats], line.text))
            for line, sung in zip(lines, sung_lines)]


def align_model(lines: list[LineInput], emission: ModelEmission, clip_ms: int,
                rms=None, *, sung_lines: Optional[list[list[str]]] = None) -> ModelResult:
    """Build prototype CTC targets and inclusive windows, retrying once at double width.

    The retry doubles each window around its centre, clipped to the whole clip.
    A second failure propagates; malformed emissions are rejected before retry.
    """
    lp = np.array(emission.log_probs, dtype=np.float64, copy=True)
    if lp.ndim != 2 or not len(lp) or not np.isfinite(emission.frame_ms) or emission.frame_ms <= 0:
        raise ValueError("emission needs frames and a positive frame_ms")
    T, V = lp.shape
    if not 0 <= emission.blank < V or np.isnan(lp).any() or np.isposinf(lp).any():
        raise ValueError("invalid emission blank or log probabilities")
    is_mms = emission.name == "mms"
    if not is_mms and emission.name not in ("kana", "hira"):
        raise ValueError("unknown alignment model")
    if not is_mms:
        for extra in ("|", "ー"):
            idx = emission.vocab.get(extra)
            if idx is not None and idx != emission.blank:
                lp[:, emission.blank] = np.logaddexp(lp[:, emission.blank], lp[:, idx])
                lp[:, idx] = -np.inf
    sung_lines = _sung_lines(lines) if sung_lines is None else sung_lines
    continuations = _continuations(lines, sung_lines)
    original_lp = lp.copy()
    silent = silence_mask([] if rms is None else rms)
    bins = np.floor(np.arange(T) * emission.frame_ms / RMS_BIN_MS).astype(np.int64)
    silent_frames = np.zeros(T, dtype=bool)
    valid = bins < len(silent)
    silent_frames[valid] = silent[bins[valid]]
    penalised = np.ones(V, dtype=bool)
    penalised[emission.blank] = False
    if is_mms and emission.star is not None:
        penalised[emission.star] = False
    lp[np.ix_(silent_frames, penalised)] -= VOICE_GATE_PENALTY
    targets, windows, owner = [], [], []

    def frame_window(a, z):
        # Match prototype int conversion (truncate, then clamp).
        return (max(0, min(T - 1, int((a - clip_ms) / emission.frame_ms))),
                max(0, min(T - 1, int((z - clip_ms) / emission.frame_ms))))

    def add_star():
        if is_mms and emission.star is not None:
            targets.append([emission.star])
            windows.append((0, T - 1))
            owner.append(None)

    for li, (line, sung) in enumerate(zip(lines, sung_lines)):
        add_star()
        for bi, (beat, reading) in enumerate(zip(line.beats, sung)):
            targets.append([] if continuations[li][bi] else
                           (tokens.mms_tokens if is_mms else tokens.kana_tokens)(reading, emission.vocab))
            a, z = (beat.fixed[0] - 80, beat.fixed[1] + 80) if beat.fixed else (line.start - 1000, line.end + 1000)
            windows.append(frame_window(a, z))
            owner.append((li, bi))
    add_star()
    flags = []
    def attempt(posteriors):
        try:
            return viterbi.align(posteriors, targets, windows, emission.blank, hold=3)
        except ValueError:
            widened = [(max(0, lo - (hi - lo + 2) // 2),
                        min(T - 1, hi + (hi - lo + 1) // 2)) for lo, hi in windows]
            result = viterbi.align(posteriors, targets, widened, emission.blank, hold=3)
            flags.append("windows_widened:" + emission.name)
            return result
    fallback = False
    try:
        first, last, _score = attempt(lp)
    except ValueError:
        if not silent_frames.any():
            raise
        first, last, _score = attempt(original_lp)
        flags.append("voice_gate_fallback:" + emission.name)
        fallback = True
    shift = emission.shift_ms if emission.shift_ms is not None else (-10 if is_mms else -50)
    aligned = [[None] * len(line.beats) for line in lines]
    for f, l, target, who in zip(first, last, targets, owner):
        if who is None:
            continue
        conf = float(np.exp(original_lp[f:l + 1, target]).max(axis=1).mean()) if target and l >= f else 0.0
        aligned[who[0]][who[1]] = ModelBeat(int(round(clip_ms + f * emission.frame_ms + shift)), conf, not target or fallback)
    for row, continuation in zip(aligned, continuations):
        for i, marked in enumerate(continuation):
            if marked:
                previous = row[i - 1]
                row[i] = ModelBeat(previous.start, previous.conf, previous.check)
    return ModelResult(aligned, flags)


def fuse(a: list[ModelBeat], b: list[ModelBeat], tol_ms: int = 200, *,
         minimum_ms: int = MIN_BEAT_MS, continuations=None,
         end_bound=None) -> list[ModelBeat]:
    """Agreement mean/confidence selection; minimum spacing within a bounded span."""
    if len(a) != len(b):
        raise ValueError("model beat counts differ")
    out = []
    for i, (x, y) in enumerate(zip(a, b)):
        agree = abs(x.start - y.start) <= tol_ms
        start = (x.start + y.start) // 2 if agree else (x.start if x.conf >= y.conf else y.start)
        if out:
            spacing = 0 if continuations is not None and continuations[i] else minimum_ms
            start = max(start, out[-1].start + spacing)
        out.append(ModelBeat(start, max(x.conf, y.conf), not agree or x.check or y.check))
    if end_bound is not None and out and out[-1].start > end_bound:
        # Infeasible spacing must not push a beat past the right anchor.
        span = max(0, end_bound - out[0].start)
        for i, beat in enumerate(out):
            beat.start = out[0].start + int(round(span * i / max(1, len(out) - 1)))
            beat.check = True
    return out


def ends_from_rms(starts, rms_db_10ms, clip_ms: int, line_end: int, last_end: str = "voice", *,
                  next_start=None, minimum_ms=0) -> list[int]:
    """End at the first >=100 ms silence after t0+60 ms, bounded by the next beat.

    RMS samples describe 10 ms bins starting at clip_ms. The search uses the
    prototype's integer bin conversion. The last beat searches through line end
    +1000 ms; absent silence it ends at line end, never before its own start.
    """
    if last_end not in ("voice", "line"):
        raise ValueError("last_end must be voice or line")
    silent = silence_mask(rms_db_10ms)
    ends = []
    for i, t0 in enumerate(starts):
        final = i + 1 == len(starts)
        limit = line_end + 1000 if final else starts[i + 1]
        if final and next_start is not None:
            limit = min(limit, next_start)
        end = max(t0, line_end) if final else limit
        count = 0
        for j in range(max(0, int((t0 - clip_ms) / 10) + 6), min(len(silent), int((limit - clip_ms) / 10))):
            count = count + 1 if silent[j] else 0
            if count >= 10:
                end = clip_ms + (j - count + 1) * 10
                break
        end = max(t0, line_end, end) if final and last_end == "line" else max(t0, end)
        if not final:
            end = max(end, min(t0 + minimum_ms, limit))
        if next_start is not None and final:
            end = min(end, next_start)
        ends.append(int(max(t0, end)))
    return ends


def calibration(pred, hand_beats, stored=None) -> tuple[int, str, int]:
    """Match (mid, beat index) starts; only original tap/drag beats train the habit.

    hand_beats are dictionaries with mid, index, t0, by and optional lane.
    ``pred`` maps (mid, index) to the uncalibrated predicted start.
    """
    diffs = [b["t0"] - pred[(b["mid"], b["index"])] for b in hand_beats
             if b.get("lane", "original") == "original" and b.get("by") in ("tap", "drag")
             and (b["mid"], b["index"]) in pred]
    if len(diffs) >= 20:
        return int(round(np.clip(np.median(diffs), -120, 120))), "habit", len(diffs)
    return int(stored or 0), "stored" if stored is not None else "none", len(diffs)


def voiced_span(start, limit, rms, clip_ms, line_end) -> int:
    """End of one voiced span in ms, bounded by the following recognised beat."""
    end = ends_from_rms([start], rms, clip_ms, line_end)[0]
    return max(start, min(end, limit))


def continuation_ratio(samples, stored=None) -> tuple[float, str, int]:
    """Median hand continuation fractions; at least five valid voiced spans."""
    values = [float(x) for x in samples if np.isfinite(x) and 0 < x < 1]
    if len(values) >= MIN_CONTINUATION_SAMPLES:
        return float(np.clip(np.median(values), *CONTINUATION_RATIO_LIMITS)), "habit", len(values)
    if stored is not None:
        value = float(stored)
        if not np.isfinite(value):
            raise ValueError("stored_continuation must be finite")
        return float(np.clip(value, *CONTINUATION_RATIO_LIMITS)), "stored", len(values)
    return DEFAULT_CONTINUATION_RATIO, "default", len(values)


def _vowel_cut(emission, previous, current, start, end, clip_ms, offset):
    """First >=100 ms run of a changed kana vowel, in calibrated ms."""
    old, new = tokens.last_vowel(previous), tokens.last_vowel(current)
    if not old or not new or old == new:
        return None
    vowel_kana = dict(zip("aiueo", "あいうえお"))
    a, b = emission.vocab.get(vowel_kana[old]), emission.vocab.get(vowel_kana[new])
    if a is None or b is None:
        return None
    shift = emission.shift_ms if emission.shift_ms is not None else -50
    times = clip_ms + np.arange(len(emission.log_probs)) * emission.frame_ms + shift + offset
    active = ((times >= start + MIN_CONTINUATION_MS) & (times < end) &
              (emission.log_probs[:, b] > emission.log_probs[:, a]) &
              (emission.log_probs[:, b] > emission.log_probs[:, emission.blank]))
    first = None
    for frame, marked in enumerate(active):
        if not marked:
            first = None
        elif first is None:
            first = frame
        if first is not None and (frame - first + 1) * emission.frame_ms >= SILENCE_MS:
            cut = int(round(times[first]))
            if end - cut >= MIN_CONTINUATION_MS:
                return cut
    return None


def split_continuations(starts, ends, beats, sung, mask, rms, clip_ms, line_end,
                        emission, ratio, offset):
    """Split each previous voiced span; mutate only the two involved beat edges."""
    for i, marked in enumerate(mask):
        if not marked:
            continue
        following = next((j for j in range(i + 1, len(starts)) if not mask[j]), None)
        limit = starts[following] if following is not None else line_end + 1000
        start = starts[i - 1]
        end = voiced_span(start, limit, rms, clip_ms, line_end)
        cut = _vowel_cut(emission, sung[i - 1], sung[i], start, end, clip_ms, offset)
        if cut is None:
            cut = int(round(start + ratio * (end - start)))
        if end - start >= 2 * MIN_CONTINUATION_MS:
            cut = max(start + MIN_CONTINUATION_MS, min(end - MIN_CONTINUATION_MS, cut))
        else:
            cut = (start + end) // 2
            beats[i - 1].check = beats[i].check = True
        starts[i], ends[i - 1], ends[i] = cut, cut, end
        beats[i].conf = beats[i - 1].conf


def _refine_model(line, reading, mask, emission, rms, clip_ms, offset,
                  starts, first, stop, left, right, *, anchored=False):
    """CTC for one uncertain run, bounded by fixed starts (times in ms)."""
    is_mms = emission.name == "mms"
    shift = emission.shift_ms if emission.shift_ms is not None else (-10 if is_mms else -50)
    frame_ms = emission.frame_ms
    lo = max(0, int(np.floor((left - clip_ms - shift - offset) / frame_ms)) - ANCHOR_FRAMES)
    hi = min(len(emission.log_probs), int(np.ceil((right - clip_ms - shift - offset) / frame_ms)))
    if hi <= lo:
        raise ValueError("empty anchor span")
    lp = np.array(emission.log_probs[lo:hi], dtype=np.float64, copy=True)
    if not is_mms:
        for symbol in ("|", "ー"):
            token = emission.vocab.get(symbol)
            if token is not None and token != emission.blank:
                lp[:, emission.blank] = np.logaddexp(lp[:, emission.blank], lp[:, token])
                lp[:, token] = -np.inf
    original = lp.copy()
    silent = silence_mask(rms)
    bins = np.floor(np.arange(lo, hi) * frame_ms / RMS_BIN_MS).astype(int)
    quiet = np.zeros(len(lp), dtype=bool)
    valid = bins < len(silent)
    quiet[valid] = silent[bins[valid]]
    penalised = np.ones(lp.shape[1], dtype=bool)
    penalised[emission.blank] = False
    if is_mms and emission.star is not None:
        penalised[emission.star] = False
    lp[np.ix_(quiet, penalised)] -= VOICE_GATE_PENALTY
    # Continuations share the previous target's span, then split after fusion.
    indices = [i for i in range(first, stop) if not mask[i]]
    targets = [(tokens.mms_tokens if is_mms else tokens.kana_tokens)(reading[i], emission.vocab)
               for i in indices]
    windows = [(0, len(lp) - 1)] * len(indices)
    entry = list(windows)
    if anchored and indices and indices[0] == first:
        anchor = int(round((starts[first] - clip_ms - shift - offset) / frame_ms)) - lo
        entry[0] = (anchor - ANCHOR_FRAMES, anchor + ANCHOR_FRAMES)
    minimum = []
    for k, i in enumerate(indices):
        following = indices[k + 1] if k + 1 < len(indices) else stop
        duration = MIN_BEAT_MS * (following - i)
        if sokuon_only(reading[i]):
            duration = 0
        minimum.append(int(np.ceil(duration / frame_ms)))
    first_frames, last_frames, _ = viterbi.align(lp, targets, windows, emission.blank,
                                                min_frames=minimum, start_windows=entry)
    result = {}
    for i, target, a, z in zip(indices, targets, first_frames, last_frames):
        confidence = float(np.exp(original[a:z + 1, target]).max(axis=1).mean()) if target else 0.0
        result[i] = ModelBeat(int(round(clip_ms + (lo + a) * frame_ms + shift)), confidence, not target)
    for i in range(first, stop):
        if mask[i] and i - 1 in result:
            previous = result[i - 1]
            result[i] = ModelBeat(previous.start, previous.conf, previous.check)
    return result


def _proportional(beats, reading, ends, first, stop, left, right, offset):
    """Mora/duration weighted starts in an infeasible anchor span; all flagged."""
    mora = np.array([max(1, sum(("ぁ" <= c <= "ゟ" or c == "ー") and c not in "ゃゅょ"
                               for c in tokens.kata_to_hira(s)))
                     for s in reading[first:stop]], dtype=float)
    duration = np.array([max(0, ends[i] - beats[i].start - offset) for i in range(first, stop)], dtype=float)
    share = duration / duration.sum() if duration.sum() else np.full(len(duration), 1 / len(duration))
    weights = MORA_WEIGHT * mora / mora.sum() + (1 - MORA_WEIGHT) * share
    boundaries = left + np.r_[0, np.cumsum(weights)] * max(0, right - left)
    for i, start in zip(range(first, stop), boundaries[:-1]):
        beats[i].start = int(round(start)) - offset
        beats[i].check = True


def after_breath(start, limit, silent, clip_ms, held_from=None) -> int:
    """A line's first start, moved past the last >=100 ms silence in [start, limit) that has voice on both sides,
    when the voice before it is the previous line's held note.

    ``held_from`` is the previous line's last beat start: the rule applies only when no breath separates it from
    ``start`` (the note is still sounding).  A vowel-only first beat (明日 = あす after 知るんだ) otherwise claims
    that held vowel and the real onset after the breath is lost.  A first syllable sung before a breath of its own
    has a breath (or the song start) before it and stays.  Unchanged when the move would leave less than half a
    minimum beat before ``limit``."""
    if held_from is None or held_from >= start:
        return start
    need = SILENCE_MS // RMS_BIN_MS
    lo = max(0, int(np.floor((start - clip_ms) / RMS_BIN_MS)))
    hi = min(len(silent), int(np.floor((limit - clip_ms) / RMS_BIN_MS)))
    run, voiced = 0, False
    for j in range(max(0, int(np.floor((held_from - clip_ms) / RMS_BIN_MS))), min(lo, len(silent))):
        run = run + 1 if silent[j] else 0
        voiced = voiced or not silent[j]
        if run >= need:
            return start
    onset, run = None, 0
    for j in range(lo, hi):
        if silent[j]:
            run += 1
            continue
        if run >= need and voiced:
            onset = j
        voiced, run = True, 0
    if onset is None:
        return start
    moved = clip_ms + onset * RMS_BIN_MS
    return moved if limit - moved >= MIN_BEAT_MS // 2 else start


def _render(lines, fused, sung, masks, rms, clip_ms, options, kana, ratio, offset, *, anchors=None):
    """Recompute vocal ends/continuations and bound chronologically adjacent lines."""
    rows = []
    silent = silence_mask(rms)
    for li, (line, beats, reading, mask) in enumerate(zip(lines, fused, sung, masks)):
        starts = [max(0, b.start + offset) for b in beats]
        if starts and li and fused[li - 1] and 0 not in (anchors or {}).get(li, {}):
            following = next((j for j in range(1, len(starts)) if not mask[j]), None)
            starts[0] = after_breath(starts[0], starts[following] if following is not None else line.end,
                                     silent, clip_ms, held_from=fused[li - 1][-1].start + offset)
            for j in range(1, following if following is not None else len(starts)):
                starts[j] = max(starts[j], starts[0])     # continuations of the first beat follow it
        active = [i for i in range(len(starts)) if not mask[i]]
        ends = starts.copy()
        voiced = ends_from_rms([starts[i] for i in active], rms, clip_ms, line.end,
                              options.get("last_end", "voice"), minimum_ms=MIN_BEAT_MS)
        for i, end in zip(active, voiced):
            ends[i] = end
        if ratio is not None:
            split_continuations(starts, ends, beats, reading, mask, rms, clip_ms,
                                line.end, kana, ratio, offset)
            if starts and mask[-1] and options.get("last_end", "voice") == "line":
                ends[-1] = max(ends[-1], line.end)
        for i, fixed in (anchors or {}).get(li, {}).items():
            starts[i] = fixed
            if mask[i]:
                ends[i - 1] = fixed
                ends[i] = max(fixed, ends[i])
        rows.append([starts, ends])
    order = sorted(range(len(lines)), key=lambda i: (lines[i].start, i))
    for li, ni in zip(order, order[1:]):
        if not rows[li][0] or not rows[ni][0]:
            continue
        bound = rows[ni][0][0]
        if lines[li].end <= lines[ni].start:
            bound = min(bound, lines[ni].start)
        for i in range(len(rows[li][0])):
            if rows[li][0][i] > bound:
                rows[li][0][i] = bound
                fused[li][i].check = True
            rows[li][1][i] = min(rows[li][1][i], bound)
    return rows


def _refine(lines, models, fused, sung, masks, emissions, rms, clip_ms,
            options, kana, ratio, offset):
    """Up to three rounds, leaving trusted starts exactly fixed."""
    silent = silence_mask(rms)
    rows = _render(lines, fused, sung, masks, rms, clip_ms, options, kana, ratio, offset)
    flags = []
    for _round in range(1, MAX_ALIGNMENT_ROUNDS):
        before = [[list(a), list(z)] for a, z in rows]
        anchors = {}
        for li, (line, reading, mask, beats) in enumerate(zip(lines, sung, masks, fused)):
            starts, ends = rows[li]
            trusted = [abs(a.start - b.start) <= TRUST_TOLERANCE_MS
                       and (ends[i] - starts[i] >= MIN_BEAT_MS or i == len(beats) - 1
                            or mask[i] or sokuon_only(reading[i]))
                       and not silent_start(starts[i], silent, clip_ms)
                       for i, (a, b) in enumerate(zip(models[0].lines[li], models[1].lines[li]))]
            anchors[li] = {i: starts[i] for i, marked in enumerate(trusted) if marked}
            i = 0
            while i < len(beats):
                if trusted[i]:
                    i += 1
                    continue
                stop = i + 1
                while stop < len(beats) and not trusted[stop]:
                    stop += 1
                first = i - 1 if i else i
                # A continuation belongs to the recognised beat before it.
                while first > 0 and mask[first]:
                    first -= 1
                left = starts[first] if i else max(clip_ms, line.start - 1000)
                right = starts[stop] if stop < len(beats) else max(line.end, ends[-1])
                required = sum(MIN_BEAT_MS for j in range(first, stop) if not sokuon_only(reading[j]))
                if right - left < required:
                    _proportional(beats, reading, ends, first, stop, left, right, offset)
                    flags.append(f"proportional:{line.mid}:{i}-{stop}")
                else:
                    try:
                        refined = [_refine_model(line, reading, mask, e, rms, clip_ms, offset,
                                                 starts, first, stop, left, right, anchored=bool(i)) for e in emissions]
                    except ValueError:
                        for j in range(i, stop):
                            beats[j].check = True
                        flags.append(f"refine_infeasible:{line.mid}:{i}-{stop}")
                    else:
                        combined = fuse([refined[0][j] for j in range(first, stop)],
                                        [refined[1][j] for j in range(first, stop)],
                                        continuations=mask[first:stop], end_bound=right - offset)
                        for j in range(i, stop):
                            # Protect anchors, including a trusted continuation's parent.
                            if not trusted[j]:
                                beats[j] = combined[j - first]
                                for model, result in zip(models, refined):
                                    model.lines[li][j] = result[j]
                i = stop
        rows = _render(lines, fused, sung, masks, rms, clip_ms, options, kana, ratio, offset, anchors=anchors)
        if rows == before:
            break
    return rows, list(dict.fromkeys(flags))


def run(lines: list[LineInput], emissions: list[ModelEmission], rms, clip_ms: int,
        options: Optional[dict] = None) -> SongResult:
    """Align the whole song with two emissions; apply calibration before finding ends.

    Options: last_end (voice/line), offset (habit/none/integer milliseconds),
    stored_habit, hand_beats, stored_continuation, continuation_mode (even/habit).
    """
    options = options or {}
    if len(emissions) != 2:
        raise ValueError("song alignment requires two model emissions")
    if len({line.mid for line in lines}) != len(lines):
        raise ValueError("line member ids must be unique")
    if not lines:
        return SongResult([], 0, "none", 0)
    sung = _sung_lines(lines)
    masks = _continuations(lines, sung)
    models = [align_model(lines, e, clip_ms, rms, sung_lines=sung) for e in emissions]
    # Refinement updates its own model rows, never caller-owned fixtures.
    models = [ModelResult([[ModelBeat(b.start, b.conf, b.check) for b in row]
                           for row in m.lines], list(m.flags)) for m in models]
    fused = [fuse(a, b, continuations=mask) for a, b, mask in zip(models[0].lines, models[1].lines, masks)]
    pred = {(line.mid, i): beat.start for line, beats, mask in zip(lines, fused, masks)
            for i, beat in enumerate(beats) if not mask[i]}
    mode = options.get("offset", "habit")
    if mode == "habit":
        offset, source, count = calibration(pred, options.get("hand_beats", []), options.get("stored_habit"))
    elif mode == "none":
        offset, source, count = 0, "none", 0
    elif isinstance(mode, Integral) and not isinstance(mode, bool):
        offset, source, count = int(mode), "custom", 0
    else:
        raise ValueError("offset must be habit, none or integer milliseconds")
    hand = {(b.get("mid"), b.get("index")): b["t0"] for b in options.get("hand_beats", [])
            if b.get("lane", "original") == "original" and b.get("by") in ("tap", "drag")}
    samples = []
    for line, beats, mask in zip(lines, fused, masks):
        for i, marked in enumerate(mask):
            a, b = hand.get((line.mid, i - 1)), hand.get((line.mid, i))
            if not marked or a is None or b is None:
                continue
            following = next((j for j in range(i + 1, len(beats)) if not mask[j]), None)
            limit = (hand.get((line.mid, following), beats[following].start + offset)
                     if following is not None else line.end + 1000)
            end = voiced_span(a, limit, rms, clip_ms, line.end)
            if end > a:
                samples.append((b - a) / (end - a))
    continuation_mode = options.get("continuation_mode", "even")
    if continuation_mode == "even":
        ratio, ratio_source, ratio_count = DEFAULT_CONTINUATION_RATIO, "even", 0
    elif continuation_mode == "habit":
        ratio, ratio_source, ratio_count = continuation_ratio(samples, options.get("stored_continuation"))
    else:
        raise ValueError("continuation_mode must be even or habit")
    if "continuation_ratio" in options and "continuation_mode" not in options:
        ratio, ratio_source, ratio_count = continuation_ratio([], options["continuation_ratio"])
        ratio_source = "default"  # Explicit evaluation ratio does not train a habit.
    if not any(any(mask) for mask in masks):
        ratio, ratio_source, ratio_count = None, "none", 0
    kana = next(e for e in emissions if e.name in ("kana", "hira")) if ratio is not None else None
    rows, refinement_flags = _refine(lines, models, fused, sung, masks, emissions, rms,
                                    clip_ms, options, kana, ratio, offset)
    silent = silence_mask(rms)
    results = []
    for line, beats, reading, (starts, ends) in zip(lines, fused, sung, rows):
        for i, (a, z, beat) in enumerate(zip(starts, ends, beats)):
            beat.check |= (z - a < MIN_BEAT_MS and i + 1 < len(beats) and not sokuon_only(reading[i])) or silent_start(a, silent, clip_ms)
        results.append(LineResult(line.mid, [(a, z, b.conf, b.check) for a, z, b in zip(starts, ends, beats)]))
    return SongResult(results, offset, source, count, [flag for model in models for flag in model.flags] + refinement_flags,
                      ratio, ratio_source, ratio_count)
