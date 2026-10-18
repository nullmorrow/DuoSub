"""Decide ambiguous kanji readings by forced alignment against the voice; Utena E06-E10 gold readings.

Run from the project root (after dialogue_align_eval cached vocals/emissions under .local-cache/kalign-eval):
    python tools/eval/reading_voice_eval.py --episodes 6 7 8 9 10
The reading engine runs WITHOUT the Utena dictionaries (they were built from these finished files). For every
engine entry with alternative readings whose gold reading is among the candidates, each candidate is written
into the line's kana and scored by the windowed CTC Viterbi of both models on the separated vocals; the
decision is the best summed score, applied only when its margin over the runner-up reaches M.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from subkit.ass import AssDocument  # noqa: E402
from subkit.ass.text import plain_text  # noqa: E402
from subkit.kalign import tokens, viterbi  # noqa: E402
from subkit.reading.engine import ReadingEngine  # noqa: E402
from subkit.textutil import kata_to_hira  # noqa: E402
from tools.eval import dialogue_align_eval as ev  # noqa: E402
from tools.eval.build_utena_readings import parse_entries  # noqa: E402
from tools.eval.hint_rough_real import inputs  # noqa: E402

WINDOW_MS = 600
MARGINS = (0, 1, 2, 3, 5, 8)


def gold_lines(final: Path):
    doc = AssDocument.load(final)
    jp = {}
    for e in doc.events:
        if e.is_dialogue and e.style in ("JP", "JPSecondary", "JPTop"):
            jp.setdefault((e.start, e.end), plain_text(e.text, breaks=""))
    out = []
    for e in doc.events:
        if e.is_comment and e.style == "振假名" and "Action" not in e.text and (e.start, e.end) in jp:
            out.append((e.start * 10, e.end * 10, jp[(e.start, e.end)], parse_entries(e.text)))
    return out


def line_kana(text, entries, particles, override=None):
    """Spoken kana of a line: entry readings over their spans, particles は/へ as わ/え, other text as is."""
    spans = {e.start: e for e in entries}
    out, i = [], 0
    while i < len(text):
        e = spans.get(i)
        if e is not None and e.end > i:
            out.append(override[1] if override is not None and override[0] is e else e.reading)
            i = e.end
            continue
        ch = text[i]
        out.append({"は": "わ", "へ": "え"}.get(ch, ch) if i in particles else ch)
        i += 1
    return kata_to_hira("".join(out))


def score(model, kana, start, end):
    lp, frame_ms, vocab, blank, star = model["lp"], model["frame_ms"], model["vocab"], model["blank"], model["star"]
    toks = [vocab[c] for c in tokens.romaji(kana) if c in vocab] if model["name"] == "mms" else tokens.kana_tokens(kana, vocab)
    if not toks:
        return None
    a = max(0, int((start - WINDOW_MS) / frame_ms))
    z = min(len(lp), int((end + WINDOW_MS) / frame_ms) + 1)
    local = lp[a:z]
    if star is None:
        others = np.delete(local, blank, axis=1)
        local = np.concatenate([local, others.max(axis=1, keepdims=True) - 1.0], axis=1)
        star = local.shape[1] - 1
    full = (0, len(local) - 1)
    try:
        return viterbi.align(local, [[star], toks, [star]], [full, full, full], blank=blank, path_score=True)[2]
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", nargs="+", type=int, default=[6, 7, 8, 9, 10])
    parser.add_argument("--examples", type=int, default=25)
    args = parser.parse_args()
    engine = ReadingEngine()
    cases = []
    for ep in args.episodes:
        video, _cn, _jp, final = inputs(ep)
        vocals = ev.vocals_16k(ep, video)
        models = {}
        for name in ("mms", "kana"):
            lp, frame_ms, vocab, blank, star = ev.emission(ep, name, vocals)
            models[name] = dict(name=name, lp=lp, frame_ms=frame_ms, vocab=vocab, blank=blank, star=star)
        lines = gold_lines(final)
        texts = [t for _a, _b, t, _g in lines]
        particles = tokens.particle_positions_many(texts)
        for (start, end, text, gold), parts in zip(lines, particles):
            entries = engine.analyze(text)
            gold_at = {}
            cursor = 0
            for surface, reading in gold:
                pos = text.find(surface, cursor)
                if pos >= 0:
                    gold_at[(pos, pos + len(surface))] = kata_to_hira(reading)
                    cursor = pos + len(surface)
            for e in entries:
                cands = list(dict.fromkeys([e.reading] + list(e.alternatives)))
                if len(cands) < 2:
                    continue
                g = gold_at.get((e.start, e.end))
                if g is None:
                    continue
                cands = [kata_to_hira(c) for c in cands]
                row = dict(ep=ep, text=text, surface=e.surface, gold=g, engine=cands[0], cands=cands,
                           flagged=e.needs_review, scores={})
                for name, model in models.items():
                    row["scores"][name] = [score(model, line_kana(text, entries, parts, (e, c)), start, end) for c in cands]
                cases.append(row)
        print(f"E{ep:02d}: {len(lines)} lines, cumulative ambiguous cases {len(cases)}", flush=True)
    in_cands = [c for c in cases if c["gold"] in c["cands"]]
    print(f"ambiguous entries with gold: {len(cases)}; gold among candidates: {len(in_cands)}; flagged by engine: "
          f"{sum(c['flagged'] for c in in_cands)}")
    engine_ok = sum(c["engine"] == c["gold"] for c in in_cands)
    print(f"engine first choice correct: {engine_ok}/{len(in_cands)}")
    for combo in (("mms",), ("kana",), ("mms", "kana")):
        for margin in MARGINS:
            decided = right = final_ok = 0
            for c in in_cands:
                totals = []
                for i in range(len(c["cands"])):
                    vals = [c["scores"][m][i] for m in combo]
                    totals.append(None if any(v is None for v in vals) else sum(vals))
                if any(t is None for t in totals):
                    final_ok += c["engine"] == c["gold"]
                    continue
                order = sorted(range(len(totals)), key=lambda i: -totals[i])
                lead = totals[order[0]] - totals[order[1]]
                choice = c["cands"][order[0]] if lead >= margin else c["engine"]
                if lead >= margin:
                    decided += 1
                    right += choice == c["gold"]
                final_ok += choice == c["gold"]
            print(f"{'+'.join(combo):9s} margin {margin}: decided {decided}, of which right {right}; "
                  f"final correct {final_ok}/{len(in_cands)} (engine alone {engine_ok})")
    wrong = Counter((c["surface"], c["gold"], c["engine"]) for c in in_cands if c["engine"] != c["gold"])
    print("engine mistakes (surface, gold, engine):", wrong.most_common(args.examples))
    # Per word: what the voice (both models summed) prefers, with its lead, against gold and engine.
    table = Counter()
    for c in in_cands:
        totals = [None if any(c["scores"][m][i] is None for m in ("mms", "kana"))
                  else c["scores"]["mms"][i] + c["scores"]["kana"][i] for i in range(len(c["cands"]))]
        if any(t is None for t in totals):
            continue
        order = sorted(range(len(totals)), key=lambda i: -totals[i])
        lead = totals[order[0]] - totals[order[1]]
        voice = c["cands"][order[0]] if lead >= 3 else "?"
        table[(c["surface"], c["gold"], c["engine"], voice)] += 1
    for subset, name in ((lambda c: c["flagged"], "flagged by engine"), (lambda c: c["surface"] != "兄", "excluding 兄 (gold あに is an old-tool error)")):
        for margin in (3, 5, 8):
            agree_right = disagree_voice_right = disagree_engine_right = disagree_neither = undecided = 0
            n = 0
            for c in in_cands:
                if not subset(c):
                    continue
                n += 1
                totals = [None if any(c["scores"][m][i] is None for m in ("kana",)) else c["scores"]["kana"][i]
                          for i in range(len(c["cands"]))]
                if any(t is None for t in totals):
                    undecided += 1
                    continue
                order = sorted(range(len(totals)), key=lambda i: -totals[i])
                if totals[order[0]] - totals[order[1]] < margin:
                    undecided += 1
                    continue
                voice = c["cands"][order[0]]
                if voice == c["engine"]:
                    agree_right += voice == c["gold"]
                elif voice == c["gold"]:
                    disagree_voice_right += 1
                elif c["engine"] == c["gold"]:
                    disagree_engine_right += 1
                else:
                    disagree_neither += 1
            print(f"{name}, kana lead>={margin}: n={n} undecided={undecided} agree&right={agree_right} "
                  f"disagree: voice right {disagree_voice_right}, engine right {disagree_engine_right}, neither {disagree_neither}")
    print("surface | gold | engine | voice(lead>=3): count")
    for (s, g, e, v), n in sorted(table.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {s} | {g} | {e} | {v}: {n}")


if __name__ == "__main__":
    main()
