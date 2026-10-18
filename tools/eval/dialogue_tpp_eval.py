"""Forced-alignment speech edges + an Aegisub-style timing post-processor learned from the user's finished files.

Run from the project root after dialogue_chain_eval.py has cached edges:
    python tools/eval/dialogue_tpp_eval.py --episodes 6 7 8 9 10 11 12 13 14 15
Order (as Aegisub's Timing Post-Processor): lead-in/lead-out from the speech edges, make adjacent lines
continuous when the gap is at most T, then snap starts/ends to keyframes inside [before, after] windows.
Every parameter is learned on the other episodes only (coordinate search maximising the <=200 ms counts);
the held-out episode is scored with docs/eval/HINT-ROUGH-2026-10-12.md's counts. Also reported: the same
post-processor applied to the shipped energy hint and to the rough times alone (to separate the two effects).
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
from tools.eval import dialogue_chain_eval as ce  # noqa: E402
from tools.eval.dialogue_keyframe_probe import keyframes  # noqa: E402
from tools.eval.hint_rough_real import inputs, score_rows  # noqa: E402

GRID = dict(chain=(None, 150, 250, 350, 500, 700, 1000),
            end_before=(0, 100, 200, 300), end_after=(0, 250, 500, 750, 1000),
            start_before=(0, 100, 200), start_after=(0, 100, 200),
            keep_start=(0, 100, 150, 200, 300, 500), keep_end=(0, 100, 150, 200, 300, 500))
DEFAULT = dict(chain=None, end_before=0, end_after=0, start_before=0, start_after=0, keep_start=0, keep_end=0)


def snap(t, ks, before, after):
    if not (before or after) or not len(ks):
        return t
    i = np.searchsorted(ks, t - before)
    best = None
    while i < len(ks) and ks[i] <= t + after:
        if best is None or abs(ks[i] - t) < abs(best - t):
            best = ks[i]
        i += 1
    return int(best) if best is not None else t


def post_process(rough, edges, ks, lead_in, lead_out, gap, p):
    """edges[i] = (speech start, speech end) or None; returns a suggestion per row (or None)."""
    n = len(rough)
    starts = [None if e is None else max(0, e[0] - lead_in) for e in edges]
    ends = [None if e is None else e[1] + lead_out for e in edges]
    out = []
    for i in range(n):
        s, e = starts[i], ends[i]
        if s is None:
            out.append(None)
            continue
        nxt = next((starts[k] if starts[k] is not None else rough[k]["start"] for k in range(i + 1, n)), None)
        if nxt is not None and nxt > s:
            if p["chain"] is not None and nxt - e <= p["chain"]:
                e = nxt - gap
            elif e > nxt:
                e = nxt - gap
        chained = nxt is not None and e == nxt - gap
        s = snap(s, ks, p["start_before"], p["start_after"])
        if not chained:
            e2 = snap(e, ks, p["end_before"], p["end_after"])
            if nxt is None or e2 <= nxt - gap:
                e = e2
        # An edge already close to the suggestion is probably right: keep it.
        if abs(s - rough[i]["start"]) < p["keep_start"]:
            s = rough[i]["start"]
        if abs(e - rough[i]["end"]) < p["keep_end"]:
            e = rough[i]["end"]
        out.append((s, e) if e > s else None)
    return out


def rows_for(d, s, suggestions):
    rows = []
    for i, r in enumerate(d["source"][s]):
        j = d["mapping"][s][i]
        answer = d["gold"][s][j] if j >= 0 else None
        current = (r["start"], r["end"])
        sug = suggestions[i]
        rows.append(dict(source_id=r["id"], rough_ms=list(current), gold_ms=[answer["start"], answer["end"]] if answer else None,
                         suggested_ms=sug, adopted_ms=sug or current, idempotent=True, gold_suggested_ms=None))
    return rows


def score(train, s, kfs, lead_in, lead_out, gap, p, edge_key):
    rows = []
    for ep, d in train:
        rows += rows_for(d, s, post_process(d["source"][s], d[edge_key][s], kfs[ep], lead_in, lead_out, gap, p))
    m = score_rows(rows)
    return m["start"]["adopted_within_200ms"] + m["end"]["adopted_within_200ms"]


def learn_params(train, s, kfs, lead_in, lead_out, gap, edge_key):
    p = dict(DEFAULT)
    for _round in range(2):
        for key, values in GRID.items():
            best = None
            for v in values:
                q = dict(p, **{key: v})
                value = score(train, s, kfs, lead_in, lead_out, gap, q, edge_key)
                if best is None or value > best[0]:
                    best = (value, v)
            p[key] = best[1]
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(6, 16)))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/eval/DIALOGUE-TPP-2026-10-13.json")
    args = parser.parse_args()
    started = time.perf_counter()
    data, kfs = {}, {}
    for ep in args.episodes:
        data[ep] = ce.episode_data(ep)
        video = inputs(ep)[0]
        kfs[ep] = keyframes(ep, video)
        # Pseudo edges for the comparison rows: rough times as "speech" (post-processor alone).
        data[ep]["rough_edges"] = {s: [(r["start"], r["end"]) for r in data[ep]["source"][s]] for s in ("A", "B")}
    report = dict(episodes=args.episodes, results={}, parameters={})
    for s in ("A", "B"):
        for edge_key, label in (("edges", "align+tpp"), ("rough_edges", "rough+tpp")):
            for mode in ("align-only", "tpp"):
                if edge_key == "rough_edges" and mode == "align-only":
                    continue
                rows = []
                for ep in args.episodes:
                    train = [(o, data[o]) for o in args.episodes if o != ep]
                    if edge_key == "edges":
                        lead_in, lead_out, gap = ce.learn([d for _o, d in train], s)
                    else:
                        lead_in, lead_out, gap = 0, 0, ce.learn([d for _o, d in train], s)[2]
                    p = dict(DEFAULT) if mode == "align-only" else learn_params(train, s, kfs, lead_in, lead_out, gap, edge_key)
                    report["parameters"][f"{s}/{label}/{mode}/E{ep:02d}"] = dict(lead_in=lead_in, lead_out=lead_out, gap=gap, **p)
                    rows += rows_for(data[ep], s, post_process(data[ep]["source"][s], data[ep][edge_key][s], kfs[ep],
                                                               lead_in, lead_out, gap, p))
                m = score_rows(rows)
                report["results"][f"{s}/{label}/{mode}"] = m
                st, en = m["start"], m["end"]
                print(f"{s} {label:9s} {mode:10s} start ≤200 {st['baseline_within_200ms']}→{st['adopted_within_200ms']} "
                      f"(≤100 {st['baseline_within_100ms']}→{st['adopted_within_100ms']}) worse/better {st['made_worse']}/{st['made_better']} | "
                      f"end ≤200 {en['baseline_within_200ms']}→{en['adopted_within_200ms']} (≤100 {en['baseline_within_100ms']}→"
                      f"{en['adopted_within_100ms']}) worse/better {en['made_worse']}/{en['made_better']} | matched={st['matched']} "
                      f"covered={st['covered']}", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"elapsed {report['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
