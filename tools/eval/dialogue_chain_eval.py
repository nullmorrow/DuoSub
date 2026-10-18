"""Forced-alignment dialogue timing with the user's chaining habit, leave-one-episode-out.

Run from the project root: python tools/eval/dialogue_chain_eval.py --episodes 6 7 8 9 10 11 12 13 14 15
Uses tools/eval/dialogue_align_eval.py for audio, emissions (cached under .local-cache/kalign-eval) and line
alignment. For each held-out episode the parameters are learned on the other episodes only:
  lead-in  = median(speech start - gold start), lead-out = median(gold end - speech end) over unchained lines,
  chain    = threshold T and gap G: if the next line's suggested start minus this end is in [-inf, T], the end
             becomes next start - G (G = median of the training gold gaps within 0..50 ms),
then scored with the same counts as docs/eval/HINT-ROUGH-2026-10-12.md.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tools.eval import dialogue_align_eval as ev  # noqa: E402
from tools.eval.hint_rough_real import inputs, pair_rows, read_rows, score_rows  # noqa: E402

CHAINS = (None, 150, 250, 350, 500, 700, 1000)


def episode_data(ep):
    path = ev.CACHE / f"E{ep:02d}" / "edges.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    video, cn, jp, final = inputs(ep)
    source, gold = read_rows(cn, jp, final)
    mapping = {s: pair_rows(source[s], gold[s], language="zh" if s == "A" else "ja") for s in source}
    vocals = ev.vocals_16k(ep, video)
    rms = ev.rms_db_10ms(vocals)
    silent_db = float(np.percentile(rms, 80) - 20)
    models = {}
    for name in ("mms", "kana"):
        lp, frame_ms, vocab, blank, star = ev.emission(ep, name, vocals)
        models[name] = dict(name=name, lp=lp, frame_ms=frame_ms, vocab=vocab, blank=blank, star=star)

    def jp_text_for(row):
        best, text = 0, ""
        for cue in source["B"]:
            overlap = min(row["end"], cue["end"]) - max(row["start"], cue["start"])
            if overlap > best:
                best, text = overlap, cue["text"]
        return text

    edges = {}
    for s in source:
        out = []
        for r in source[s]:
            text = r["text"] if s == "B" else jp_text_for(r)
            k = ev.kana_of(text) if text.strip() else ""
            if not k:
                out.append(None)
                continue
            out.append(ev.fuse(ev.align_line(models["mms"], k, r["start"], r["end"], rms, silent_db),
                               ev.align_line(models["kana"], k, r["start"], r["end"], rms, silent_db)))
        edges[s] = out
    data = dict(source=source, gold=gold, mapping=mapping, edges=edges)
    path.write_bytes(pickle.dumps(data))
    return data


def learn(train, s):
    starts, ends, gaps = [], [], []
    for d in train:
        g, m, e = d["gold"][s], d["mapping"][s], d["edges"][s]
        for i, (edge, j) in enumerate(zip(e, m)):
            if edge is None or j < 0:
                continue
            starts.append(edge[0] - g[j]["start"])
            gap = g[j + 1]["start"] - g[j]["end"] if j + 1 < len(g) else 10 ** 6
            if gap > 50:
                ends.append(g[j]["end"] - edge[1])
            elif gap >= 0:
                gaps.append(gap)
    lead_in = int(np.clip(np.median(starts), -100, 600)) if starts else ev.DEFAULT_PADDING[0]
    lead_out = int(np.clip(np.median(ends), -200, 800)) if ends else ev.DEFAULT_PADDING[1]
    gap = int(np.median(gaps)) if gaps else 0
    return lead_in, lead_out, gap


def suggest(d, s, lead_in, lead_out, chain, gap):
    rough = d["source"][s]
    edges = d["edges"][s]
    starts = [None if e is None else max(0, e[0] - lead_in) for e in edges]
    ends = [None if e is None else e[1] + lead_out for e in edges]
    rows = []
    for i, r in enumerate(rough):
        start, end = starts[i], ends[i]
        nxt = next((starts[k] if starts[k] is not None else rough[k]["start"] for k in range(i + 1, len(rough))), None)
        prv_end = None
        if i:
            prv_end = ends[i - 1] if ends[i - 1] is not None else rough[i - 1]["end"]
        if end is not None and nxt is not None:
            if chain is not None and nxt - end <= chain and nxt > (start if start is not None else r["start"]):
                end = nxt - gap
            elif end > nxt:
                end = nxt - gap
        if start is not None and prv_end is not None and start < prv_end and (i and rough[i - 1]["start"] < start):
            start = prv_end + gap
        suggestion = None if start is None else (start, end)
        if suggestion is not None and suggestion[1] <= suggestion[0]:
            suggestion = None
        current = (r["start"], r["end"])
        adopted = suggestion or current
        j = d["mapping"][s][i]
        answer = d["gold"][s][j] if j >= 0 else None
        rows.append(dict(source_id=r["id"], rough_ms=list(current), gold_ms=[answer["start"], answer["end"]] if answer else None,
                         suggested_ms=suggestion, adopted_ms=adopted, idempotent=True, gold_suggested_ms=None))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(6, 16)))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/eval/DIALOGUE-ALIGN-2026-10-13.json")
    args = parser.parse_args()
    started = time.perf_counter()
    data = {}
    for ep in args.episodes:
        t = time.perf_counter()
        data[ep] = episode_data(ep)
        print(f"E{ep:02d} ready {time.perf_counter() - t:.0f}s", flush=True)
    report = dict(episodes=args.episodes, results={}, parameters={})
    for s in ("A", "B"):
        for chain_mode in ("none", "fixed", "learned"):
            rows = []
            for ep in args.episodes:
                train = [data[o] for o in args.episodes if o != ep]
                lead_in, lead_out, gap = learn(train, s)
                if chain_mode == "none":
                    chain = None
                elif chain_mode == "fixed":
                    chain = 300
                else:
                    best = None
                    for c in CHAINS:
                        r = []
                        for d in train:
                            r += suggest(d, s, lead_in, lead_out, c, gap)
                        m = score_rows(r)
                        value = m["start"]["adopted_within_200ms"] + m["end"]["adopted_within_200ms"]
                        if best is None or value > best[0]:
                            best = (value, c)
                    chain = best[1]
                report["parameters"][f"{s}/{chain_mode}/E{ep:02d}"] = dict(lead_in=lead_in, lead_out=lead_out, gap=gap, chain=chain)
                rows += suggest(data[ep], s, lead_in, lead_out, chain, gap)
            m = score_rows(rows)
            report["results"][f"{s}/{chain_mode}"] = m
            st, en = m["start"], m["end"]
            print(f"{s} chain={chain_mode:7s} start ≤200 {st['baseline_within_200ms']}→{st['adopted_within_200ms']} "
                  f"(≤100 {st['baseline_within_100ms']}→{st['adopted_within_100ms']}) worse/better {st['made_worse']}/{st['made_better']} | "
                  f"end ≤200 {en['baseline_within_200ms']}→{en['adopted_within_200ms']} (≤100 {en['baseline_within_100ms']}→"
                  f"{en['adopted_within_100ms']}) worse/better {en['made_worse']}/{en['made_better']} | n={st['denominator']} "
                  f"matched={st['matched']} covered={st['covered']}", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"elapsed {report['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
