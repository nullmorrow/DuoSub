"""Why do forced-alignment ends not help? Distribution of gold end vs speech end and vs neighbours (cached data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tools.eval import dialogue_align_eval as ev  # noqa: E402
from tools.eval.hint_rough_real import inputs, pair_rows, read_rows  # noqa: E402


def q(values):
    a = np.asarray(values)
    return " ".join(f"p{p}={np.percentile(a, p):.0f}" for p in (10, 25, 50, 75, 90)) + f" n={len(a)}"


for ep in [int(x) for x in sys.argv[1:]] or [6, 7]:
    video, cn, jp, final = inputs(ep)
    source, gold = read_rows(cn, jp, final)
    vocals = ev.vocals_16k(ep, video)
    rms = ev.rms_db_10ms(vocals)
    silent_db = float(np.percentile(rms, 80) - 20)
    models = {}
    for name in ("mms", "kana"):
        lp, frame_ms, vocab, blank, star = ev.emission(ep, name, vocals)
        models[name] = dict(name=name, lp=lp, frame_ms=frame_ms, vocab=vocab, blank=blank, star=star)
    s = "B"
    mapping = pair_rows(source[s], gold[s], language="ja")
    g = gold[s]
    d_start, d_end, gaps_next, end_to_next, last_tok = [], [], [], [], []
    for i, (row, j) in enumerate(zip(source[s], mapping)):
        if j < 0 or not row["text"].strip():
            continue
        k = ev.kana_of(row["text"])
        e = ev.fuse(ev.align_line(models["mms"], k, row["start"], row["end"], rms, silent_db),
                    ev.align_line(models["kana"], k, row["start"], row["end"], rms, silent_db))
        if e is None:
            continue
        d_start.append(e[0] - g[j]["start"])
        d_end.append(g[j]["end"] - e[1])
        if j + 1 < len(g):
            gaps_next.append(g[j + 1]["start"] - g[j]["end"])
    print(f"E{ep:02d} speech_start - gold_start: {q(d_start)}")
    print(f"E{ep:02d} gold_end - speech_end:     {q(d_end)}")
    print(f"E{ep:02d} gold gap to next line:      {q(gaps_next)}")
    gaps = np.asarray(gaps_next)
    print(f"E{ep:02d} share of gold ends chained to next line (gap 0..50 ms): {np.mean((gaps >= 0) & (gaps <= 50)):.2f}; "
          f"gap<=300: {np.mean((gaps >= 0) & (gaps <= 300)):.2f}")
    d = np.asarray(d_end)
    print(f"E{ep:02d} |gold_end - speech_end - 160| <=200: {np.mean(np.abs(d - 160) <= 200):.2f}; "
          f"gold_end later than speech_end+500: {np.mean(d > 500):.2f}; earlier than speech_end: {np.mean(d < 0):.2f}")
