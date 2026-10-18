"""Effect of a fixed 'keep edges closer than D ms' rule on top of the learned post-processor (LOO)."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tools.eval import dialogue_chain_eval as ce  # noqa: E402
from tools.eval import dialogue_tpp_eval as tp  # noqa: E402
from tools.eval.dialogue_keyframe_probe import keyframes  # noqa: E402
from tools.eval.hint_rough_real import inputs, score_rows  # noqa: E402

episodes = list(range(6, 16))
data = {ep: ce.episode_data(ep) for ep in episodes}
kfs = {ep: keyframes(ep, inputs(ep)[0]) for ep in episodes}
for s in ("A", "B"):
    for keep in (0, 100, 200, 300, 500):
        rows = []
        for ep in episodes:
            train = [(o, data[o]) for o in episodes if o != ep]
            lead_in, lead_out, gap = ce.learn([d for _o, d in train], s)
            p = dict(tp.DEFAULT, chain=None, end_before=300, end_after=250, keep_start=keep, keep_end=keep)
            rows += tp.rows_for(data[ep], s, tp.post_process(data[ep]["source"][s], data[ep]["edges"][s], kfs[ep],
                                                             lead_in, lead_out, gap, p))
        m = score_rows(rows)
        st, en = m["start"], m["end"]
        print(f"{s} keep<{keep:3d}: start ≤200 {st['adopted_within_200ms']} ≤100 {st['adopted_within_100ms']} worse/better "
              f"{st['made_worse']}/{st['made_better']} | end ≤200 {en['adopted_within_200ms']} ≤100 {en['adopted_within_100ms']} "
              f"worse/better {en['made_worse']}/{en['made_better']}")
