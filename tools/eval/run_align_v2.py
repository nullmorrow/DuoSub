"""Evaluate the order-preserving aligner (dp2) on the Utena gold set.

Metrics (rows with a known user answer):
  exact        — row JP text equals the user's final JP (normalised); a joint block counts its whole JP for every row
  exact_nofill — same, after removing filler phrases from both sides (keep-mode shows fillers the user deleted)
  joint_ok     — rows of joint blocks whose joint JP equals the concatenated user JP of those rows (coarser but correct)
  order        — order violations in the produced block sequence (must be 0)
  hidden       — units not rendered (0 in keep mode)
The user's finals are not a correctness oracle (fillers deleted on purpose, old reading errors kept);
compare versions with the same metric, never call it "accuracy of the original text".

usage: python tools/eval/run_align_v2.py [--eps 1-10] [--backend labse|none] [--mode keep|compact] [--weights '{"cut_space":0.4}'] [--show 3]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from subkit.align.dp2 import Aligner2, Row, Weights2, estimate_offset, piecewise_offsets, realign_detached, time_warp  # noqa: E402
from subkit.align.embed import Embedder  # noqa: E402
from subkit.align.fragments import is_interjection  # noqa: E402
from subkit.core.importer import import_source  # noqa: E402
from subkit.core.units import build_units  # noqa: E402

from score_align import DATA, load_gold, norm_jp  # noqa: E402

_EMB: dict[str, Embedder] = {}
_TMP = HERE / "data" / "_import_tmp"


def embedder(backend: str):
    if backend == "none":
        return None
    if backend not in _EMB:
        _EMB[backend] = Embedder(backend, cache_path=DATA.parent / "emb_cache.sqlite3")
    return _EMB[backend]


def make_sim(emb, rows, units, w, mode):
    import numpy as np
    cache: dict[str, "np.ndarray"] = {}
    dry = Aligner2(rows, units, weights=w, sim_fn=lambda a, b: 0.0, filler_mode=mode)
    cn_t, jp_t = dry.collect_texts()
    texts = sorted(set([r.text for r in rows] + [u.text for u in units]) | cn_t | jp_t)
    for t, v in zip(texts, emb.embed(texts)):
        cache[t] = v

    def vec(t: str):
        v = cache.get(t)
        if v is None:
            v = emb.embed([t])[0]
            cache[t] = v
        return v

    return lambda cn, jp: float(np.dot(vec(cn), vec(jp)))


def nofill(text: str) -> str:
    return " ".join(p for p in re.split(r"[ 　]+", text) if p and not is_interjection(p))


WARP = "track"          # --warp old: the earlier global offset + ±3 s local corrections


def align_episode(ep: int, backend: str, w: Weights2, mode: str):
    gold = load_gold(ep)
    src = import_source(DATA / gold["jp_srt"], src_id="ja1", lang="ja", lane="original", episode_dir=_TMP / f"E{ep:02d}")
    units = build_units(src)
    rows = [Row(i, r["start"] * 10, r["end"] * 10, r["cn"], i) for i, r in enumerate(gold["rows"])]
    if WARP == "old":
        off = estimate_offset(rows, units)
        warp = piecewise_offsets(rows, units, off)
    else:
        warp, off, _rep = time_warp(rows, units)
    for u in units:
        u.wstart, u.wend = warp(u.start), warp(u.end)
    emb = embedder(backend)
    sim = make_sim(emb, rows, units, w, mode) if emb else None
    al = Aligner2(rows, units, weights=w, sim_fn=sim, filler_mode=mode)
    out = al.run()
    realign_detached(rows, units, out, lambda r, u: Aligner2(r, u, weights=w, sim_fn=sim, filler_mode=mode))
    out.offset = off
    return gold, units, out


def evaluate(gold, units, out):
    rows_text: dict[int, str] = {}
    joint_of: dict[int, list[int]] = {}
    for bl in out.blocks:
        if bl.kind in ("pair", "joint", "empty_row"):
            text = "  ".join(units[k].text for k in bl.units if not bl.hidden)
            for r in bl.rows:
                rows_text[r] = text
                if bl.kind == "joint":
                    joint_of[r] = bl.rows
    seq = [k for bl in out.blocks for k in bl.units]
    order = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
    hidden = sum(len(bl.units) for bl in out.blocks if bl.hidden)
    covered = len(set(seq)) == len(units) and len(seq) == len(units)
    n = ex = exf = jok = 0
    for i, g in enumerate(gold["rows"]):
        if g.get("gold_jp") is None:
            continue
        n += 1
        p = rows_text.get(i, "")
        ex += norm_jp(p) == norm_jp(g["gold_jp"])
        exf += norm_jp(nofill(p)) == norm_jp(nofill(g["gold_jp"]))
        if i in joint_of:
            rr = joint_of[i]
            if all(gold["rows"][r].get("gold_jp") is not None for r in rr):
                cat = "".join(norm_jp(nofill(gold["rows"][r]["gold_jp"])) for r in rr)
                jok += norm_jp(nofill(p)) == cat
    kinds = {}
    for bl in out.blocks:
        kinds[bl.kind] = kinds.get(bl.kind, 0) + 1
    return {"rows": n, "exact": ex, "exact_nofill": exf, "joint_ok": jok, "order": order, "hidden": hidden,
            "all_units_once": covered, "blocks": kinds, "reanchored": len(out.reanchored)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="1-10")
    ap.add_argument("--backend", default="labse")
    ap.add_argument("--mode", default="keep")
    ap.add_argument("--weights", default="{}")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--warp", default="track", choices=("track", "old"))
    args = ap.parse_args()
    global WARP
    WARP = args.warp
    eps = []
    for part in args.eps.split(","):
        a, _, b = part.partition("-")
        eps.extend(range(int(a), int(b or a) + 1))
    w = Weights2(**json.loads(args.weights))
    tot = {"rows": 0, "exact": 0, "exact_nofill": 0, "joint_ok": 0, "order": 0, "hidden": 0}
    t0 = time.time()
    for ep in eps:
        gold, units, out = align_episode(ep, args.backend, w, args.mode)
        s = evaluate(gold, units, out)
        for k in tot:
            tot[k] += s[k]
        print(f"E{ep:02d} off={out.offset/1000:+.2f}s rows={s['rows']} exact={s['exact']/s['rows']:.3f} "
              f"nofill={s['exact_nofill']/s['rows']:.3f} joint_ok={s['joint_ok']} order={s['order']} hidden={s['hidden']} "
              f"once={s['all_units_once']} blocks={s['blocks']}")
        if args.show:
            shown = 0
            rows_text = {}
            for bl in out.blocks:
                for r in bl.rows:
                    rows_text[r] = ("  ".join(units[k].text for k in bl.units), bl.kind)
            for i, g in enumerate(gold["rows"]):
                if g.get("gold_jp") is None:
                    continue
                p, kind = rows_text.get(i, ("", "?"))
                if norm_jp(nofill(p)) != norm_jp(nofill(g["gold_jp"])):
                    st = g["start"]
                    print(f"   [{i}] {st // 6000}:{st % 6000 / 100:05.2f} {kind} CN={g['cn'][:24]}\n        gold={g['gold_jp']}\n        pred={p}")
                    shown += 1
                    if shown >= args.show:
                        break
    r = tot["rows"]
    print(f"ALL rows={r} exact={tot['exact']/r:.3f} nofill={tot['exact_nofill']/r:.3f} joint_ok={tot['joint_ok']} "
          f"order={tot['order']} hidden={tot['hidden']} ({time.time()-t0:.0f}s) weights={json.dumps(asdict(w))}")


if __name__ == "__main__":
    main()
