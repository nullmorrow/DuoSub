"""Windowed CTC Viterbi, vectorised over states with int8 back-pointers.

No torch/torchaudio dependency; any vocabulary index may be the CTC blank.
The topology and tie-breaking follow tools/experiments/kalign/ctc_align.py.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def align(log_probs: np.ndarray, beats: list[list[int]], windows: list[tuple[int, int]],
          blank: int = 0, hold: int = 0, *, path_score: bool = False,
          min_frames: list[int] | None = None,
          start_windows: list[tuple[int, int]] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Align all beats to (T, V) floating log posteriors, or raise ValueError.

    Token states self-loop; intervening blanks may be skipped only between
    different symbols. Each beat's inclusive window restricts both its tokens
    and its hold states. After its last token, exactly ``hold`` extra states
    emit max(token, blank), without self-loops. The following blank is optional.
    An empty beat has one self-looping blank-emitting timed state and its hold
    chain. Returned first/last frames cover timed states, excluding hold frames.
    Optional min_frames guarantees a minimum occupied span (tokens plus holds)
    per beat. start_windows restrict entry into its first timed state, allowing
    narrow start anchors without restricting the rest of the beat's duration.

    For compatibility, score is the prototype's maximum over the final DP row,
    which may exceed the score of the completed path used for first/last.
    ``path_score=True`` returns the completed path's log score instead — use it to
    compare transcriptions of the same audio (e.g. candidate readings).
    With S = 1 + 2 * sum(max(1, len(beat))) + hold * len(beats), time and
    storage are O(T*S), including the emission matrix and int8 back-pointers.
    Impossible paths use -inf rather than the prototype's finite sentinel.
    """
    lp = np.asarray(log_probs)
    if lp.ndim != 2 or not np.issubdtype(lp.dtype, np.floating):
        raise ValueError("log_probs must be a (T, V) floating array")
    T, V = lp.shape
    if not isinstance(blank, Integral) or not 0 <= blank < V:
        raise ValueError("blank must be a vocabulary index")
    if not isinstance(hold, Integral) or hold < 0:
        raise ValueError("hold must be a non-negative integer")
    if len(windows) != len(beats):
        raise ValueError("one frame window is required per beat")
    minimum = [0] * len(beats) if min_frames is None else min_frames
    if len(minimum) != len(beats) or any(not isinstance(x, Integral) or x < 0 for x in minimum):
        raise ValueError("min_frames requires one non-negative integer per beat")
    starts = windows if start_windows is None else start_windows
    if len(starts) != len(beats) or any(len(w) != 2 or any(not isinstance(x, Integral) for x in w)
                                       or w[0] > w[1] for w in starts):
        raise ValueError("start_windows requires one integer (lo, hi) pair per beat")
    if np.isnan(lp).any() or np.isposinf(lp).any():
        raise ValueError("log_probs cannot contain NaN or positive infinity")
    for tokens, window in zip(beats, windows):
        if len(window) != 2 or any(not isinstance(x, Integral) for x in window):
            raise ValueError("frame windows must be integer (lo, hi) pairs")
        if window[0] > window[1]:
            raise ValueError("frame window start must not exceed its end")
        if any(not isinstance(tok, Integral) or not 0 <= tok < V or tok == blank
               for tok in tokens):
            raise ValueError("beat tokens must be non-blank vocabulary indices")
    if not beats:
        score = float(lp[:, blank].sum(dtype=np.float64))
        if not np.isfinite(score):
            raise ValueError("no valid CTC path")
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), score
    if T == 0:
        raise ValueError("no valid CTC path: no emission frames")

    blank_lp = lp[:, blank]
    emissions, stay, skip, owner, timed = [], [], [], [], []
    entries = {}

    def add(col, can_stay, can_skip, who, is_timed):
        emissions.append(col)
        stay.append(can_stay)
        skip.append(can_skip)
        owner.append(who)
        timed.append(is_timed)

    add(blank_lp, True, False, -1, False)
    previous = None
    frames = np.arange(T)
    for bi, (tokens, (lo, hi)) in enumerate(zip(beats, windows)):
        if start_windows is not None:
            entries[len(emissions)] = starts[bi]
        shortest = max(1, len(tokens)) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        beat_hold = max(hold, minimum[bi] - shortest)
        mask = (frames >= lo) & (frames <= hi)
        if not tokens:
            held = np.where(mask, blank_lp, -np.inf)
            add(held, True, False, bi, True)
            for _ in range(beat_hold):
                add(held, False, False, bi, False)
            add(blank_lp, True, False, -1, False)
            previous = None
            continue
        for k, tok in enumerate(tokens):
            add(np.where(mask, lp[:, tok], -np.inf), True,
                previous is not None and previous != tok, bi, True)
            previous = tok
            if k + 1 == len(tokens) and beat_hold:
                held = np.where(mask, np.maximum(lp[:, tok], blank_lp), -np.inf)
                for _ in range(beat_hold):
                    add(held, False, False, bi, False)
                previous = None
            add(blank_lp, True, False, -1, False)

    S = len(emissions)
    E = np.stack(emissions, axis=1)
    stay = np.asarray(stay)
    skip = np.asarray(skip)
    for s in range(2, S):
        # Skip the blank immediately following a hold chain, even for repeats.
        if timed[s] and owner[s - 1] == -1 and owner[s - 2] != -1 and not stay[s - 2]:
            skip[s] = True
    score = np.full(S, -np.inf)
    score[:2] = E[0, :2]
    if 1 in entries and not entries[1][0] <= 0 <= entries[1][1]:
        score[1] = -np.inf
    back = np.zeros((T, S), dtype=np.int8)
    idx = np.arange(S)
    for t in range(1, T):
        candidates = np.stack((np.where(stay, score, -np.inf),
                               np.r_[-np.inf, score[:-1]],
                               np.where(skip, np.r_[-np.inf, -np.inf, score[:-2]], -np.inf)))
        for state, (lo, hi) in entries.items():
            if not lo <= t <= hi:
                candidates[1:, state] = -np.inf
        choice = np.argmax(candidates, axis=0)
        score = candidates[choice, idx] + E[t]
        back[t] = choice
    s = S - 1 if score[-1] >= score[-2] else S - 2
    if not np.isfinite(score[s]):
        raise ValueError("no valid CTC path within the beat windows and hold constraints")
    completed = float(score[s])
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    first = np.full(len(beats), -1, dtype=np.int64)
    last = np.full(len(beats), -1, dtype=np.int64)
    for t, state in enumerate(path):
        bi = owner[state]
        if bi >= 0 and timed[state]:
            if first[bi] < 0:
                first[bi] = t
            last[bi] = t
    if np.any(first < 0):
        raise ValueError("no valid CTC path covering every beat")
    return first, last, completed if path_score else float(score.max())
