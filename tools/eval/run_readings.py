"""Compare the reading engine with the user's final Utena readings.

Blocks are compared over the union of gold/predicted kanji spans so different
segmentations (学校 vs 学|校) still compare by their concatenated readings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from subkit.reading.dictionary import ReadingDict  # noqa: E402
from subkit.reading.engine import ReadingEngine  # noqa: E402
from subkit.reading.kana_align import align  # noqa: E402
from subkit.textutil import kata_to_hira  # noqa: E402

from run_align import parse_eps  # noqa: E402
from score_align import DATA  # noqa: E402

UTENA_DICTS = [Path('./sample-media/furigana_phrases.json'), Path('./sample-media/furigana_phrases.json')]


def gold_parts(jp: str, entries: list) -> list[tuple[int, int, str]]:
    parts = []
    cursor = 0
    for surface, reading in entries:
        pos = jp.find(surface, cursor)
        if pos < 0:
            pos = jp.find(surface)
            if pos < 0:
                continue
        sub = align(surface, reading)
        if sub:
            parts.extend((pos + a, pos + b, kata_to_hira(r)) for a, b, r in sub)
        else:
            ks = [i for i, ch in enumerate(surface) if not ("ぁ" <= ch <= "ゖ" or "ァ" <= ch <= "ヺ")]
            if ks:
                parts.append((pos + ks[0], pos + ks[-1] + 1, kata_to_hira(reading)))
        cursor = pos + len(surface)
    return parts


def compare(jp: str, gold: list[tuple[int, int, str]], pred: list[tuple[int, int, str]]):
    """Yield (span, gold_reading, pred_reading) over union blocks."""
    ivs = sorted([(a, b) for a, b, _ in gold] + [(a, b) for a, b, _ in pred])
    merged: list[list[int]] = []
    for a, b in ivs:
        if merged and a < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        g = "".join(r for x, y, r in sorted(gold) if a <= x and y <= b)
        p = "".join(r for x, y, r in sorted(pred) if a <= x and y <= b)
        yield (a, b), g, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="1-10")
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--examples", type=int, default=0)
    args = ap.parse_args()
    d = ReadingDict.load(*UTENA_DICTS)
    eng = ReadingEngine(d)
    total = ok = missing = extra = 0
    diffs: Counter = Counter()
    examples: dict[tuple, list[str]] = {}
    review_flagged = review_wrong = 0
    for ep in parse_eps(args.eps):
        items = json.loads((DATA / f"ruby_E{ep:02d}.json").read_text(encoding="utf-8"))
        for it in items:
            if it.get("style") == "JPOPED":
                continue
            jp = it["jp"]
            gp = gold_parts(jp, it["entries"])
            ents = eng.analyze(jp)
            pp = [(p.start, p.end, kata_to_hira(p.reading)) for e in ents for p in e.parts]
            flagged = {(p.start, p.end) for e in ents if e.needs_review for p in e.parts}
            for (a, b), g, p in compare(jp, gp, pp):
                if not g:
                    extra += 1
                    continue
                total += 1
                if not p:
                    missing += 1
                    diffs[(jp[a:b], g, "∅")] += 1
                    continue
                isflag = any(a <= x and y <= b for x, y in flagged)
                review_flagged += isflag
                if g == p:
                    ok += 1
                else:
                    review_wrong += isflag
                    key = (jp[a:b], g, p)
                    diffs[key] += 1
                    examples.setdefault(key, []).append(jp)
    print(f"blocks={total} agree={ok/total:.3f} missing={missing} extra={extra} flagged={review_flagged} flagged&disagree={review_wrong}")
    for (s, g, p), n in diffs.most_common(args.top):
        ex = examples.get((s, g, p), [""])[0] if args.examples else ""
        print(f"  {n:3d}  {s}  gold={g}  ours={p}   {ex}")


if __name__ == "__main__":
    main()
