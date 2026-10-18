"""Score a CN↔JP alignment against the Utena gold set.

Metrics (over rows whose gold is known):
  exact   — share of rows whose JP text equals the user's final JP text (normalised)
  lcs_f1  — character-level F1 using per-row longest common subsequence
            (rewards partially-correct rows; robust to small edits the user made)
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from rapidfuzz.distance import LCSseq

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "utena"

_PUNCT = re.compile(r"[\s　、。，,.!！?？…・『』「」“”\"'（）()~～\-—―♪➡≪≫:：]+")


def norm_jp(text: str | None) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\([^)]*\)", "", t)  # leftover speaker labels / readings in parentheses
    return _PUNCT.sub("", t)


def load_gold(ep: int) -> dict:
    return json.loads((DATA / f"gold_E{ep:02d}.json").read_text(encoding="utf-8"))


def score(pred: list[str | None], gold_rows: list[dict]) -> dict:
    assert len(pred) == len(gold_rows), (len(pred), len(gold_rows))
    exact = total = 0
    lcs = gl = pl = 0
    for p, g in zip(pred, gold_rows):
        if g.get("gold_jp") is None:
            continue
        total += 1
        gp, pp = norm_jp(g["gold_jp"]), norm_jp(p)
        if gp == pp:
            exact += 1
        lcs += LCSseq.similarity(gp, pp)
        gl += len(gp)
        pl += len(pp)
    prec = lcs / pl if pl else 0.0
    rec = lcs / gl if gl else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"rows": total, "exact": exact / total if total else 0.0, "lcs_p": prec, "lcs_r": rec, "lcs_f1": f1}


def mistakes(pred: list[str | None], gold_rows: list[dict], limit: int = 30) -> list[tuple]:
    out = []
    for i, (p, g) in enumerate(zip(pred, gold_rows)):
        if g.get("gold_jp") is None:
            continue
        if norm_jp(g["gold_jp"]) != norm_jp(p):
            out.append((i, g["start"], g["cn"], g["gold_jp"], p))
    return out[:limit]
