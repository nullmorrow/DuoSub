"""Do the user's finished dialogue edges snap to video keyframes (scene cuts)? Utena E06-E15, read-only."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from subkit import proc  # noqa: E402
from tools.eval.hint_rough_real import inputs, read_rows  # noqa: E402

CACHE = Path('./sample-media/kalign-eval')


def keyframes(ep, video):
    path = CACHE / f"E{ep:02d}" / "keyframes.npy"
    if path.exists():
        return np.load(path)
    out = proc.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                    "-of", "csv=p=0", str(video)], capture_output=True, text=True, check=True).stdout
    ks = [float(t) * 1000 for t, _, f in (line.partition(",") for line in out.splitlines()) if "K" in f and t]
    arr = np.unique(np.round(np.asarray(ks)))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def near(times, ks, tol):
    times = np.asarray(times, float)
    i = np.clip(np.searchsorted(ks, times), 1, len(ks) - 1)
    d = np.minimum(np.abs(ks[i] - times), np.abs(ks[i - 1] - times))
    return d <= tol


def main():
    rng = np.random.default_rng(0)
    tot = {"start": [], "end": [], "rand": []}
    for ep in [int(x) for x in sys.argv[1:]] or list(range(6, 16)):
        video, cn, jp, final = inputs(ep)
        _src, gold = read_rows(cn, jp, final)
        ks = keyframes(ep, video)
        g = gold["B"]
        starts, ends = [r["start"] for r in g], [r["end"] for r in g]
        rand = rng.uniform(min(starts), max(ends), 2000)
        for name, t in (("start", starts), ("end", ends), ("rand", rand)):
            tot[name].append(near(t, ks, 50))
        print(f"E{ep:02d}: {len(ks)} keyframes; gold start near kf {near(starts, ks, 50).mean():.2f}, "
              f"end {near(ends, ks, 50).mean():.2f}, random {near(rand, ks, 50).mean():.2f}", flush=True)
    for name, v in tot.items():
        print(name, f"{np.concatenate(v).mean():.3f}")


if __name__ == "__main__":
    main()
