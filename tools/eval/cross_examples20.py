"""Re-check the 20 hand-reviewed alignment examples (Utena E06/E08/E10, read-only reference
少女革命ウテナ/cross_reference_20261002/visual_compare_E06_E08_E10/examples20.json) with the current aligner.

usage: python tools/eval/cross_examples20.py      (needs the LaBSE embedding cache; HF offline)
"""
import sys, json, os
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, './sample-media/eval'); sys.path.insert(0, './sample-media/src')
sys.stdout.reconfigure(encoding="utf-8")
from run_align_v2 import align_episode
from score_align import norm_jp
from subkit.align.dp2 import Weights2
ex = json.load(open('./sample-media/examples20.json', encoding="utf-8"))["examples"]
cache = {}
better = same_ok = worse = 0
for e in ex:
    ep = e["episode"]
    if ep not in cache:
        gold, units, out = align_episode(ep, "labse", Weights2(), "keep")
        rows_text = {}
        for bl in out.blocks:
            for r in bl.rows:
                rows_text[r] = ("  ".join(units[k].text for k in bl.units), bl.kind)
        by_start = gold["rows"]
        unpaired = [(units[bl.units[0]].wstart, "  ".join(units[k].text for k in bl.units)) for bl in out.blocks if bl.kind == "unpaired"]
        cache[ep] = (gold, rows_text, by_start, unpaired)
    gold, rows_text, by_start, unpaired = cache[ep]
    print(f"\n#{e['id']:02d} E{ep:02d} {e['time_label']} — {e['title']} [{e['judgement']}]")
    for r in e.get("rows", []):
        import re as _re
        key = _re.sub(r"\s+", "", r["cn"])
        cands = [k for k, g in enumerate(by_start) if _re.sub(r"\s+", "", g["cn"]) == key]
        i = min(cands, key=lambda k: abs(by_start[k]["start"] * 10 - r["start_ms"])) if cands else None
        new, kind = rows_text.get(i, ("?", "?")) if i is not None else ("(row not found)", "")
        ref = r.get("human_reference") or ""
        old = (r.get("claude") or {}).get("text", "")
        ok_new = norm_jp(new) == norm_jp(ref)
        ok_old = norm_jp(old) == norm_jp(ref)
        mark = "✓" if ok_new else "✗"
        if ok_new and not ok_old: better += 1
        elif ok_new and ok_old: same_ok += 1
        elif ok_old and not ok_new: worse += 1
        print(f"  {mark} CN{r['cn_ordinal']} {r['cn'][:18]}\n      ref={ref}\n      old={old}\n      new={new} [{kind}]")
    s0, s1 = e["start_ms"], e["end_ms"]
    near = [t for w, t in unpaired if s0 - 2000 <= w <= s1 + 2000]
    if near: print("      unpaired nearby:", near)
print(f"\nrows: new-correct-old-wrong={better}, both-correct={same_ok}, old-correct-new-wrong={worse}")
