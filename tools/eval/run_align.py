"""Evaluate the new aligner on the Utena gold set.

usage: python run_align.py [--backend labse] [--eps 1-5] [--weights '{"sem":4}'] [--show 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from subkit.align.aligner import AlignRow, Aligner, Weights, estimate_offset, make_sim_fn, piecewise_offsets, prefetch_spans, realign_detached_blocks  # noqa: E402
from subkit.align.embed import Embedder  # noqa: E402
from subkit.align.fragments import build_units  # noqa: E402
from subkit.sources.jp import JpOptions, load_jp  # noqa: E402

from score_align import DATA, load_gold, mistakes, score  # noqa: E402

_EMB: dict[str, Embedder] = {}


def embedder(backend: str) -> Embedder | None:
    if backend == "none":
        return None
    if backend not in _EMB:
        _EMB[backend] = Embedder(backend, cache_path=DATA.parent / "emb_cache.sqlite3")
    return _EMB[backend]


def align_episode(ep: int, backend: str, weights: Weights):
    gold = load_gold(ep)
    cues = load_jp(DATA / gold["jp_srt"], JpOptions())
    units = build_units(cues)
    rows = [AlignRow(i, r["start"], r["end"], r["cn"]) for i, r in enumerate(gold["rows"])]
    off = estimate_offset(rows, units)
    warp = piecewise_offsets(rows, units, off)
    for u in units:
        u.wstart, u.wend = warp(u.start), warp(u.end)
    emb = embedder(backend)
    sim = make_sim_fn(emb, rows, units) if emb else None
    al = Aligner(rows, units, weights=weights, sim_fn=sim)
    if emb:
        prefetch_spans(emb, al, weights.max_units)
    res = al.run()
    moved = realign_detached_blocks(rows, units, res, lambda r, u: Aligner(r, u, weights=weights, sim_fn=sim))
    if moved:
        print(f"   E{ep:02d} re-anchored blocks: {moved}")
    res.offset_cs = off
    pred = ["  ".join(units[k].text for k in r) for r in res.rows]
    return gold, units, res, pred


def parse_eps(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="labse")
    ap.add_argument("--eps", default="1-10")
    ap.add_argument("--weights", default="{}")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    w = Weights(**json.loads(args.weights))
    tot_rows = 0
    acc = {"exact": 0.0, "lcs_f1": 0.0}
    t0 = time.time()
    for ep in parse_eps(args.eps):
        gold, units, res, pred = align_episode(ep, args.backend, w)
        s = score(pred, gold["rows"])
        dropped = [units[k].text for k in res.skipped]
        print(f"E{ep:02d} off={res.offset_cs/100:+.2f}s rows={s['rows']} exact={s['exact']:.3f} lcs_f1={s['lcs_f1']:.3f} "
              f"skipped={len(res.skipped)} swaps={len(res.swapped_rows)//2}")
        tot_rows += s["rows"]
        for k in acc:
            acc[k] += s[k] * s["rows"]
        if args.show:
            for i, st, cn, g, p in mistakes(pred, gold["rows"], limit=args.show):
                print(f"   [{i}] {st//6000}:{st%6000/100:05.2f} CN={cn[:22]}\n        gold={g}\n        pred={p}  conf={res.confidence[i]:.2f}")
            print("   dropped:", dropped[:20])
    print(f"ALL rows={tot_rows} exact={acc['exact']/tot_rows:.3f} lcs_f1={acc['lcs_f1']/tot_rows:.3f}  ({time.time()-t0:.0f}s)  weights={asdict(w)}")


if __name__ == "__main__":
    main()
