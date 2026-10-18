"""Coordinate-descent tuning of aligner weights on dev episodes; report held-out test.

python tune_align.py --backend labse --dev 1-5 --test 6-10
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from subkit.align.aligner import AlignRow, Aligner, Weights, estimate_offset, make_sim_fn, piecewise_offsets, prefetch_spans  # noqa: E402
from subkit.align.fragments import build_units  # noqa: E402
from subkit.sources.jp import JpOptions, load_jp  # noqa: E402

from run_align import embedder, parse_eps  # noqa: E402
from score_align import DATA, load_gold, score  # noqa: E402


def prepare(ep: int, backend: str):
    gold = load_gold(ep)
    units = build_units(load_jp(DATA / gold["jp_srt"], JpOptions()))
    rows = [AlignRow(i, r["start"], r["end"], r["cn"]) for i, r in enumerate(gold["rows"])]
    warp = piecewise_offsets(rows, units, estimate_offset(rows, units))
    for u in units:
        u.wstart, u.wend = warp(u.start), warp(u.end)
    emb = embedder(backend)
    sim = make_sim_fn(emb, rows, units) if emb else None
    if emb:
        prefetch_spans(emb, Aligner(rows, units, weights=Weights(), sim_fn=sim), 6)
    return gold, rows, units, sim


def evaluate(prepped, w: Weights) -> float:
    tot = acc = 0.0
    for gold, rows, units, sim in prepped:
        res = Aligner(rows, units, weights=w, sim_fn=sim).run()
        pred = ["  ".join(units[k].text for k in r) for r in res.rows]
        s = score(pred, gold["rows"])
        acc += s["exact"] * s["rows"] + 0.5 * s["lcs_f1"] * s["rows"]
        tot += s["rows"]
    return acc / tot


GRID = {
    "sem": [2.0, 3.0, 4.0, 6.0, 8.0],
    "sem_base": [0.2, 0.25, 0.3, 0.35, 0.4],
    "time": [0.75, 1.0, 1.5, 2.0, 3.0],
    "gap_scale": [60.0, 100.0, 150.0, 250.0],
    "empty_row": [0.5, 1.0, 1.5, 2.5],
    "skip_content": [1.0, 1.5, 2.5, 4.0],
    "skip_interj": [0.0, 0.05, 0.2, 0.5],
    "interj_attach": [0.0, 0.15, 0.3, 0.6],
    "swap": [0.8, 1.2, 2.0, 4.0],
    "window": [250.0, 400.0, 600.0],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="labse")
    ap.add_argument("--dev", default="1-5")
    ap.add_argument("--test", default="6-10")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()
    dev = [prepare(ep, args.backend) for ep in parse_eps(args.dev)]
    test = [prepare(ep, args.backend) for ep in parse_eps(args.test)]
    w = Weights()
    best = evaluate(dev, w)
    print(f"start dev={best:.4f} test={evaluate(test, w):.4f}", flush=True)
    for rnd in range(args.rounds):
        for name, values in GRID.items():
            for v in values:
                cand = replace(w, **{name: v})
                s = evaluate(dev, cand)
                if s > best + 1e-6:
                    best, w = s, cand
                    print(f"  r{rnd} {name}={v} dev={best:.4f}", flush=True)
    print(f"final dev={best:.4f} test={evaluate(test, w):.4f}")
    print(json.dumps(asdict(w)))


if __name__ == "__main__":
    main()
