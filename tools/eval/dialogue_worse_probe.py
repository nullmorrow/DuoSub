"""What do the made-worse ends look like? Gold vs suggestion direction, line length, gap to the next gold line."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tools.eval import dialogue_chain_eval as ce  # noqa: E402
from tools.eval import dialogue_tpp_eval as tp  # noqa: E402
from tools.eval.dialogue_keyframe_probe import keyframes  # noqa: E402
from tools.eval.hint_rough_real import inputs  # noqa: E402


def q(v):
    v = np.asarray(v)
    return " ".join(f"p{p}={np.percentile(v, p):.0f}" for p in (10, 25, 50, 75, 90)) + f" n={len(v)}"


episodes = list(range(6, 16))
data = {ep: ce.episode_data(ep) for ep in episodes}
kfs = {ep: keyframes(ep, inputs(ep)[0]) for ep in episodes}
for s in ("A", "B"):
    rows = {"worse": [], "kept_good": []}
    for ep in episodes:
        train = [(o, data[o]) for o in episodes if o != ep]
        lead_in, lead_out, gap = ce.learn([d for _o, d in train], s)
        p = dict(tp.DEFAULT, end_before=300, end_after=250)
        d = data[ep]
        sug = tp.post_process(d["source"][s], d["edges"][s], kfs[ep], lead_in, lead_out, gap, p)
        gold = d["gold"][s]
        for i, (r, e, sg) in enumerate(zip(d["source"][s], d["edges"][s], sug)):
            j = d["mapping"][s][i]
            if j < 0 or sg is None:
                continue
            g = gold[j]
            rough, new = abs(r["end"] - g["end"]), abs(sg[1] - g["end"])
            nxt = gold[j + 1]["start"] - g["end"] if j + 1 < len(gold) else 99999
            item = (sg[1] - g["end"], g["end"] - g["start"], nxt, e[1] - e[0], len(r["text"]))
            if rough <= 200 < new:
                rows["worse"].append(item)
            elif new <= 200:
                rows["kept_good"].append(item)
    for k, v in rows.items():
        a = np.asarray(v)
        print(f"{s} {k}: suggestion-gold end {q(a[:, 0])}")
        print(f"{s} {k}: gold duration {q(a[:, 1])} | gold gap to next {q(a[:, 2])} | speech length {q(a[:, 3])}")
        print(f"{s} {k}: share suggestion earlier than gold {np.mean(a[:, 0] < 0):.2f}; gold chained (gap<=50) {np.mean(a[:, 2] <= 50):.2f}")
