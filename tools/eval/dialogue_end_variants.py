"""End-of-voice rule variants: re-derive ends from the last aligned token + vocal rms (cached), LOO scoring."""
from __future__ import annotations

from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from subkit.kalign import tokens, viterbi  # noqa: E402
from tools.eval import dialogue_align_eval as ev  # noqa: E402
from tools.eval import dialogue_chain_eval as ce  # noqa: E402
from tools.eval import dialogue_tpp_eval as tp  # noqa: E402
from tools.eval.dialogue_keyframe_probe import keyframes  # noqa: E402
from tools.eval.hint_rough_real import inputs, score_rows  # noqa: E402


def raw_align(model, kana, start, end):
    lp, frame_ms, vocab, blank, star = model["lp"], model["frame_ms"], model["vocab"], model["blank"], model["star"]
    toks = [vocab[c] for c in tokens.romaji(kana) if c in vocab] if model["name"] == "mms" else tokens.kana_tokens(kana, vocab)
    if not toks:
        return None
    a = max(0, int((start - ev.WINDOW_MS) / frame_ms))
    z = min(len(lp), int((end + ev.WINDOW_MS) / frame_ms) + 1)
    local = lp[a:z]
    if star is None:
        others = np.delete(local, blank, axis=1)
        local = np.concatenate([local, others.max(axis=1, keepdims=True) - 1.0], axis=1)
        star = local.shape[1] - 1
    if len(local) < len(toks) + 4:
        return None
    full = (0, len(local) - 1)
    try:
        first, last, _ = viterbi.align(local, [[star], toks, [star]], [full, full, full], blank=blank)
    except ValueError:
        return None
    return (a + first[1]) * frame_ms + ev.SHIFT[model["name"]], (a + last[1] + 1) * frame_ms


def raw_data(ep):
    path = ev.CACHE / f"E{ep:02d}" / "raw_edges.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    d = ce.episode_data(ep)
    video = inputs(ep)[0]
    vocals = ev.vocals_16k(ep, video)
    rms = ev.rms_db_10ms(vocals)
    models = {}
    for name in ("mms", "kana"):
        lp, frame_ms, vocab, blank, star = ev.emission(ep, name, vocals)
        models[name] = dict(name=name, lp=lp, frame_ms=frame_ms, vocab=vocab, blank=blank, star=star)
    raw = {}
    for s in ("A", "B"):
        texts = [r["text"] for r in d["source"]["B"]]
        out = []
        for r in d["source"][s]:
            if s == "B":
                text = r["text"]
            else:
                best, text = 0, ""
                for cue in d["source"]["B"]:
                    overlap = min(r["end"], cue["end"]) - max(r["start"], cue["start"])
                    if overlap > best:
                        best, text = overlap, cue["text"]
            k = ev.kana_of(text) if text.strip() else ""
            if not k:
                out.append(None)
                continue
            m, kk = raw_align(models["mms"], k, r["start"], r["end"]), raw_align(models["kana"], k, r["start"], r["end"])
            if m is None or kk is None:
                out.append(m or kk)
            elif abs(m[0] - kk[0]) <= 200 and abs(m[1] - kk[1]) <= 200:
                out.append(((m[0] + kk[0]) / 2, (m[1] + kk[1]) / 2))
            else:
                out.append(kk)
        raw[s] = out
    result = dict(raw=raw, rms=rms.astype(np.float32))
    path.write_bytes(pickle.dumps(result))
    return result


def end_of_voice(t_last, rms, threshold, run_frames, search_ms):
    i = int(t_last // 10)
    run = 0
    for j in range(i, min(len(rms), i + search_ms // 10)):
        run = run + 1 if rms[j] < threshold else 0
        if run >= run_frames:
            return max(t_last, (j - run + 1) * 10)
    return t_last + search_ms if search_ms <= 0 else t_last


episodes = list(range(6, 16))
data = {ep: ce.episode_data(ep) for ep in episodes}
raws = {ep: raw_data(ep) for ep in episodes}
kfs = {ep: keyframes(ep, inputs(ep)[0]) for ep in episodes}
variants = [("token only", None, 0, 0)] + [(f"rms p80-{db}dB run{run}0ms search{search}", db, run, search)
                                           for db in (15, 20, 25, 30) for run in (10, 15, 30) for search in (800, 1500)]
for s in ("A", "B"):
    best = []
    for label, db, run, search in variants:
        for ep in episodes:
            rms = raws[ep]["rms"]
            th = None if db is None else float(np.percentile(rms, 80) - db)
            edges = []
            for e in raws[ep]["raw"][s]:
                if e is None:
                    edges.append(None)
                    continue
                t1 = e[1] if th is None else end_of_voice(e[1], rms, th, run, search)
                edges.append((int(e[0]), int(t1), 0.0))
            data[ep]["variant"] = {s: edges}
        rows = []
        for ep in episodes:
            train = [data[o] for o in episodes if o != ep]
            tmp = [dict(d, edges=d["variant"]) for d in train]
            lead_in, lead_out, gap = ce.learn(tmp, s)
            p = dict(tp.DEFAULT, end_before=300, end_after=250)
            rows += tp.rows_for(data[ep], s, tp.post_process(data[ep]["source"][s], data[ep]["variant"][s], kfs[ep],
                                                             lead_in, lead_out, gap, p))
        m = score_rows(rows)
        best.append((m["end"]["adopted_within_200ms"], m["end"]["adopted_within_100ms"], m["end"]["made_worse"],
                     m["start"]["adopted_within_200ms"], label))
    for item in sorted(best, reverse=True)[:8]:
        print(s, f"end ≤200 {item[0]} ≤100 {item[1]} worse {item[2]} | start ≤200 {item[3]} | {item[4]}")
