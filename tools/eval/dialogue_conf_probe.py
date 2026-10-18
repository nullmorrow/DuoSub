"""Does alignment confidence separate harmful suggestions? Confidence quantiles for made-worse / made-better rows."""
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

episodes = list(range(6, 16))
data = {ep: ce.episode_data(ep) for ep in episodes}
kfs = {ep: keyframes(ep, inputs(ep)[0]) for ep in episodes}
for s in ("A", "B"):
    classes = {"start": {"worse": [], "better": [], "kept_good": [], "other": []},
               "end": {"worse": [], "better": [], "kept_good": [], "other": []}}
    for ep in episodes:
        train = [(o, data[o]) for o in episodes if o != ep]
        lead_in, lead_out, gap = ce.learn([d for _o, d in train], s)
        p = dict(tp.DEFAULT, end_before=300, end_after=250)
        d = data[ep]
        sug = tp.post_process(d["source"][s], d["edges"][s], kfs[ep], lead_in, lead_out, gap, p)
        for i, (r, e, sg) in enumerate(zip(d["source"][s], d["edges"][s], sug)):
            j = d["mapping"][s][i]
            if j < 0 or sg is None or e is None:
                continue
            g = d["gold"][s][j]
            for k, edge in enumerate(("start", "end")):
                rough = abs((r["start"], r["end"])[k] - (g["start"], g["end"])[k])
                new = abs(sg[k] - (g["start"], g["end"])[k])
                cls = "worse" if rough <= 200 < new else "better" if new <= 200 < rough else "kept_good" if new <= 200 else "other"
                classes[edge][cls].append(e[2])
    for edge, c in classes.items():
        print(s, edge, "  ".join(f"{k}: n={len(v)} p25={np.percentile(v, 25):.2f} p50={np.percentile(v, 50):.2f}"
                                for k, v in c.items() if v))
    # Share of worse/better kept if only suggestions with confidence >= t are used.
    for t in (0.2, 0.3, 0.4, 0.5):
        w = sum(sum(x >= t for x in classes[e]["worse"]) for e in classes)
        b = sum(sum(x >= t for x in classes[e]["better"]) for e in classes)
        print(f"  {s} conf>={t}: worse kept {w}, better kept {b}")
