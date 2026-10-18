"""Dialogue timing from separated vocals and forced alignment, against the user's finished Utena episodes.

Run from the project root: python tools/eval/dialogue_align_eval.py --episodes 6 7
Methods (each suggests start/end per rough row, then the C-key rules clamp them to the neighbours):
  mix      - shipped suggest_group_times on the mixed audio (as docs/eval/HINT-ROUGH-2026-10-12.md)
  vocals   - the same function on Demucs-separated vocals
  mms/kana/fused - forced alignment of the Japanese text (MMS romaji, Japanese kana CTC, both) inside
             [rough start - 1 s, rough end + 1 s]; speech edges = first token frame / end of voice after the
             last token (vocal energy), then the same padding as the C key.
Rough rows: A = fansub Chinese timing (Japanese text taken from the rough JP SRT cue overlapping it most),
B = JP BD SRT (its own text). Answers: Done/Publish Candidate. Media and subtitles are read-only; vocals and
emissions are cached under .local-cache/kalign-eval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from subkit import autotime, proc  # noqa: E402
from subkit.audio import compute_features  # noqa: E402
from subkit.kalign import models, tokens, viterbi  # noqa: E402
from subkit.sources.jp import JpOptions, clean_line  # noqa: E402
from tools.eval.hint_rough_real import DEFAULT_PADDING, inputs, pair_rows, read_rows, score_rows  # noqa: E402

CACHE = Path('./sample-media/kalign-eval')
WINDOW_MS = 1000
SHIFT = {"mms": -10, "kana": -50}


# ------------------------------------------------------------------ audio, cached per episode
def vocals_16k(ep: int, video: Path) -> np.ndarray:
    path = CACHE / f"E{ep:02d}" / "vocals16k.npy"
    if path.exists():
        return np.load(path).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-map", "0:a:0",
                    "-ac", "2", "-ar", "44100", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    mix = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    vocals = models.separate_vocals(mix)
    from scipy.signal import resample_poly
    mono = resample_poly(vocals.mean(axis=1), 160, 441).astype(np.float32)
    np.save(path, mono.astype(np.float16))
    return mono


def mix_16k(video: Path) -> np.ndarray:
    raw = proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-map", "0:a:0",
                    "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


def emission(ep: int, name: str, audio: np.ndarray):
    path = CACHE / f"E{ep:02d}" / f"{name}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        vocab = json.loads(str(z["vocab"]))
        return z["lp"].astype(np.float32), float(z["frame_ms"]), vocab, int(z["blank"]), (int(z["star"]) if z["star"] >= 0 else None)
    e = models.mms_emission(audio) if name == "mms" else models.kana_emission(audio)
    np.savez(path, lp=e.log_probs.astype(np.float16), frame_ms=e.frame_ms, vocab=json.dumps(e.vocab),
             blank=e.blank, star=-1 if e.star is None else e.star)
    return e.log_probs.astype(np.float32), e.frame_ms, e.vocab, e.blank, e.star


def rms_db_10ms(audio16k: np.ndarray) -> np.ndarray:
    hop = 160
    padded = np.pad(audio16k, (hop * 2, hop * 3))
    frames = np.lib.stride_tricks.sliding_window_view(padded, hop * 5)[::hop][:len(audio16k) // hop + 1]
    return 20 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) + 1e-9)


# ------------------------------------------------------------------ readings
_analyzer = None


def kana_of(text: str) -> str:
    """Spoken kana of a Japanese line (speaker labels/readings cleaned like the importer)."""
    global _analyzer
    from subkit.reading.engine import Analyzer
    if _analyzer is None:
        _analyzer = Analyzer.get()
    clean = " ".join(clean_line(t, JpOptions()).text for t in text.splitlines())
    out = []
    for m in _analyzer.tokenize(clean):
        if m.pos and m.pos[0] == "助詞" and m.surface in ("は", "へ"):
            out.append({"は": "わ", "へ": "え"}[m.surface])
        elif m.pos and m.pos[0] in ("補助記号", "空白"):
            continue
        else:
            out.append(m.reading or m.surface)
    return "".join(out)


# ------------------------------------------------------------------ forced alignment per line
def align_line(model, kana: str, start: int, end: int, rms, silent_db):
    lp, frame_ms, vocab, blank, star = model["lp"], model["frame_ms"], model["vocab"], model["blank"], model["star"]
    if model["name"] == "mms":
        toks = [vocab[c] for c in tokens.romaji(kana) if c in vocab]
    else:
        toks = tokens.kana_tokens(kana, vocab)
    if not toks:
        return None
    a = max(0, int((start - WINDOW_MS) / frame_ms))
    z = min(len(lp), int((end + WINDOW_MS) / frame_ms) + 1)
    local = lp[a:z]
    if star is None:
        # Kana model has no garbage token: append one that scores just below the best real token.
        others = np.delete(local, blank, axis=1)
        local = np.concatenate([local, others.max(axis=1, keepdims=True) - 1.0], axis=1)
        star_id = local.shape[1] - 1
    else:
        star_id = star
    if len(local) < len(toks) + 4:
        return None
    full = (0, len(local) - 1)
    try:
        first, last, _score = viterbi.align(local, [[star_id], toks, [star_id]], [full, full, full], blank=blank)
    except ValueError:
        return None
    t0 = (a + first[1]) * frame_ms + SHIFT[model["name"]]
    t_last = (a + last[1] + 1) * frame_ms
    # End of voice: the first >=150 ms silence after the last token (else the last token end).
    i = int(t_last // 10)
    run, t1 = 0, t_last
    for j in range(i, min(len(rms), i + 80)):
        run = run + 1 if rms[j] < silent_db else 0
        if run >= 15:
            t1 = (j - run + 1) * 10
            break
    conf = float(np.exp(local[first[1]:last[1] + 1][:, toks]).max(axis=1).mean())
    return int(round(t0)), int(round(max(t1, t_last))), conf


def fuse(a, b, tol=200):
    if a is None or b is None:
        return a or b
    if abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol:
        return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2, max(a[2], b[2]))
    return a if a[2] >= b[2] else b


# ------------------------------------------------------------------ scoring
def suggestion_rows(rough, gold, mapping, edges, padding):
    rows = []
    for i, (source, j) in enumerate(zip(rough, mapping)):
        previous = (rough[i - 1]["start"], rough[i - 1]["end"]) if i else None
        following = (rough[i + 1]["start"], rough[i + 1]["end"]) if i + 1 < len(rough) else None
        current = (source["start"], source["end"])
        e = edges[i]
        suggestion = None
        if e is not None:
            suggestion = autotime.clamp_suggested_times((max(0, e[0] - padding[0]), e[1] + padding[1]), current,
                                                        previous=previous, following=following)
        adopted = tuple(c if s is None else s for c, s in zip(current, suggestion)) if suggestion else current
        answer = gold[j] if j >= 0 else None
        rows.append(dict(source_id=source["id"], rough_ms=list(current),
                         gold_ms=[answer["start"], answer["end"]] if answer else None,
                         suggested_ms=suggestion, adopted_ms=adopted, idempotent=True, gold_suggested_ms=None))
    return rows


def hint_rows(activity, rough, gold, mapping, padding):
    rows = []
    for i, (source, j) in enumerate(zip(rough, mapping)):
        previous = (rough[i - 1]["start"], rough[i - 1]["end"]) if i else None
        following = (rough[i + 1]["start"], rough[i + 1]["end"]) if i + 1 < len(rough) else None
        current = (source["start"], source["end"])
        suggestion = autotime.suggest_group_times(activity, *current, lead_in_ms=padding[0], lead_out_ms=padding[1],
                                                  previous=previous, following=following)
        adopted = tuple(c if s is None else s for c, s in zip(current, suggestion)) if suggestion else current
        answer = gold[j] if j >= 0 else None
        rows.append(dict(source_id=source["id"], rough_ms=list(current),
                         gold_ms=[answer["start"], answer["end"]] if answer else None,
                         suggested_ms=suggestion, adopted_ms=adopted, idempotent=True, gold_suggested_ms=None))
    return rows


def padding_samples(edges, gold, mapping):
    """(gold start offsets before speech, gold end offsets after speech) for LOO padding."""
    onsets, offsets = [], []
    for e, j in zip(edges, mapping):
        if e is not None and j >= 0:
            onsets.append(e[0] - gold[j]["start"])
            offsets.append(gold[j]["end"] - e[1])
    return onsets, offsets


def summary(metrics):
    out = {}
    for edge in ("start", "end"):
        s = metrics[edge]
        out[edge] = dict(n=s["denominator"], covered=s["covered"],
                         rough200=s["baseline_within_200ms"], rough100=s["baseline_within_100ms"],
                         adopted200=s["adopted_within_200ms"], adopted100=s["adopted_within_100ms"],
                         worse=s["made_worse"], better=s["made_better"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=[6, 7])
    parser.add_argument("--output", type=Path, default=ROOT / "docs/eval/DIALOGUE-ALIGN-2026-10-13.json")
    args = parser.parse_args()
    started = time.perf_counter()
    per_episode = {}
    for ep in args.episodes:
        video, cn, jp, final = inputs(ep)
        source, gold = read_rows(cn, jp, final)
        mapping = {s: pair_rows(source[s], gold[s], language="zh" if s == "A" else "ja") for s in source}
        t = time.perf_counter()
        vocals = vocals_16k(ep, video)
        t_sep = time.perf_counter() - t
        t = time.perf_counter()
        act_vocals = autotime._snap_activity(compute_features(vocals))
        act_mix = autotime._snap_activity(compute_features(mix_16k(video)))
        rms = rms_db_10ms(vocals)
        silent_db = float(np.percentile(rms, 80) - 20)
        modelset = {}
        for name in ("mms", "kana"):
            lp, frame_ms, vocab, blank, star = emission(ep, name, vocals)
            modelset[name] = dict(name=name, lp=lp, frame_ms=frame_ms, vocab=vocab, blank=blank, star=star)
        t_models = time.perf_counter() - t
        # Japanese text for A rows: the rough JP cue overlapping most (what a merged document pairs).
        def jp_text_for(row):
            best, text = 0, ""
            for cue in source["B"]:
                overlap = min(row["end"], cue["end"]) - max(row["start"], cue["start"])
                if overlap > best:
                    best, text = overlap, cue["text"]
            return text
        t = time.perf_counter()
        edges = {}
        for s in source:
            texts = [r["text"] if s == "B" else jp_text_for(r) for r in source[s]]
            kana = [kana_of(x) if x.strip() else "" for x in texts]
            per = {}
            for name in ("mms", "kana"):
                per[name] = [align_line(modelset[name], k, r["start"], r["end"], rms, silent_db) if k else None
                             for k, r in zip(kana, source[s])]
            per["fused"] = [fuse(x, y) for x, y in zip(per["mms"], per["kana"])]
            edges[s] = per
        t_align = time.perf_counter() - t
        per_episode[ep] = dict(source=source, gold=gold, mapping=mapping, edges=edges,
                               act_vocals=act_vocals, act_mix=act_mix,
                               seconds=dict(separate=t_sep, features_models=t_models, align=t_align))
        print(f"E{ep:02d}: separate {t_sep:.0f}s, features+models {t_models:.0f}s, align {t_align:.0f}s", flush=True)
    report = dict(episodes=args.episodes, results={}, seconds={ep: d["seconds"] for ep, d in per_episode.items()})
    for s in ("A", "B"):
        for method in ("mix", "vocals", "mms", "kana", "fused"):
            for mode in ("default", "loo"):
                rows = []
                for ep, d in per_episode.items():
                    if method in ("mix", "vocals"):
                        padding = DEFAULT_PADDING
                        if mode == "loo":
                            act = d["act_mix" if method == "mix" else "act_vocals"]
                            on, off = [], []
                            for other, o in per_episode.items():
                                if other != ep:
                                    a, b = autotime._hint_padding_samples(o["act_mix" if method == "mix" else "act_vocals"],
                                                                          sorted({(r["start"], r["end"]) for r in o["gold"]["B"]}))
                                    on += a
                                    off += b
                            if len(on) >= 20 and len(off) >= 20:
                                padding = (int(np.clip(np.median(on), 0, 600)), int(np.clip(np.median(off), -200, 600)))
                        act = d["act_mix" if method == "mix" else "act_vocals"]
                        rows += hint_rows(act, d["source"][s], d["gold"][s], d["mapping"][s], padding)
                    else:
                        padding = DEFAULT_PADDING
                        if mode == "loo":
                            on, off = [], []
                            for other, o in per_episode.items():
                                if other != ep:
                                    a, b = padding_samples(o["edges"][s][method], o["gold"][s], o["mapping"][s])
                                    on += a
                                    off += b
                            if len(on) >= 20:
                                padding = (int(np.clip(np.median(on), -100, 600)), int(np.clip(np.median(off), -200, 600)))
                        rows += suggestion_rows(d["source"][s], d["gold"][s], d["mapping"][s], d["edges"][s][method], padding)
                result = summary(score_rows(rows))
                report["results"][f"{s}/{method}/{mode}"] = result
                st, en = result["start"], result["end"]
                print(f"{s} {method:6s} {mode:7s} start ≤200 {st['rough200']}→{st['adopted200']} (≤100 {st['rough100']}→{st['adopted100']}) "
                      f"worse/better {st['worse']}/{st['better']} | end ≤200 {en['rough200']}→{en['adopted200']} "
                      f"(≤100 {en['rough100']}→{en['adopted100']}) worse/better {en['worse']}/{en['better']} | n={st['n']} covered={st['covered']}",
                      flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"elapsed {report['elapsed_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
