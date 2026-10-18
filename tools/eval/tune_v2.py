"""Grid search for the dp2 cut / straddle weights on Utena E01–E05 (E06–E10 held out).

usage: python tools/eval/tune_v2.py [--backend labse] [--grid '{"cut_space":[0.3,0.6]}']
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from run_align_v2 import align_episode, evaluate  # noqa: E402
from subkit.align.dp2 import Weights2  # noqa: E402


def score(eps, w, backend, mode="keep"):
    tot = {"rows": 0, "exact": 0, "exact_nofill": 0}
    for ep in eps:
        gold, units, out = align_episode(ep, backend, w, mode)
        s = evaluate(gold, units, out)
        assert s["order"] == 0 and s["hidden"] == 0
        for k in tot:
            tot[k] += s[k]
    return tot["exact"] / tot["rows"], tot["exact_nofill"] / tot["rows"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="labse")
    ap.add_argument("--grid", default=json.dumps({"cut_space": [0.3, 0.6, 1.0], "protect_scale": [1.0, 2.0],
                                                   "straddle": [0.8, 1.5, 2.5], "mix_speaker": [0.5, 1.0]}))
    args = ap.parse_args()
    grid = json.loads(args.grid)
    keys = list(grid)
    results = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        w = Weights2(**dict(zip(keys, combo)))
        ex, nf = score(range(1, 6), w, args.backend)
        results.append((nf, ex, dict(zip(keys, combo))))
        print(f"train exact={ex:.4f} nofill={nf:.4f} {dict(zip(keys, combo))}", flush=True)
    results.sort(key=lambda x: (-x[0], -x[1]))
    for nf, ex, params in results[:3]:
        tex, tnf = score(range(6, 11), Weights2(**params), args.backend)
        print(f"BEST train nofill={nf:.4f} exact={ex:.4f} | test exact={tex:.4f} nofill={tnf:.4f} {params}", flush=True)


if __name__ == "__main__":
    main()
