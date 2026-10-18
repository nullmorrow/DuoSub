"""Build the Utena alignment gold set from the user's finished episodes.

Inputs (read-only):
  origin/<ep>-sub01.ass.bak          original CN fansub (640x480)
  eval/data/utena/jp/ENN.ja.srt       original JP BD-retime SRT
  Done/Publish Candidate[/Recheck]/<ep>.zh-Hant.ass   human-corrected bilingual result

Output: eval/data/utena/gold_ENN.json
  rows: [{start, end, cn, gold_jp}]  — one per original CN dialogue line that could
  be matched to a final CN row (by text, monotonic).  gold_jp is the JP text the
  user finally put next to that CN line.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from subkit.ass import AssDocument  # noqa: E402
from subkit.ass.text import plain_text  # noqa: E402
from subkit.sources.cn import classify  # noqa: E402

UTENA = Path('./sample-media/少女革命ウテナ')
OUT = HERE / "data" / "utena"


def norm(text: str) -> str:
    t = plain_text(text)
    return re.sub(r"[\s　、。，,.!！?？…・『』「」“”\"'（）()~～\-—]+", "", t)


def final_path(ep: int) -> Path:
    name = f"S01E{ep:03d} - [LoliHouse] Shoujo Kakumei Utena - {ep:02d} [BDRip 1440x1080 HEVC-10bit AACx2].zh-Hant.ass"
    for d in (UTENA / "Done/Publish Candidate/Recheck", UTENA / "Done/Publish Candidate"):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def origin_path(ep: int) -> Path:
    return UTENA / f"origin/S01E{ep:03d} - [LoliHouse] Shoujo Kakumei Utena - {ep:02d} [BDRip 1440x1080 HEVC-10bit AACx2]-sub01.ass.bak"


def final_rows(doc: AssDocument) -> list[dict]:
    cn_styles = {"CN", "CNSecondary", "CNTop"}
    jp_styles = {"JP", "JPSecondary", "JPTop"}
    jp_by_time: dict[tuple[int, int], list[str]] = {}
    for e in doc.events:
        if e.is_dialogue and e.style in jp_styles:
            jp_by_time.setdefault((e.start, e.end), []).append(plain_text(e.text, breaks=" "))
    rows = []
    for e in doc.events:
        if e.is_dialogue and e.style in cn_styles:
            jps = jp_by_time.get((e.start, e.end), [])
            rows.append({"start": e.start, "end": e.end, "cn": plain_text(e.text, breaks=" "), "jp": " ".join(jps)})
    return rows


def match(src: list[dict], fin: list[dict]) -> list[int]:
    """Monotonic DP matching of source CN lines to final CN rows (text + time)."""
    n, m = len(src), len(fin)
    a = [norm(r["cn"]) for r in src]
    b = [norm(r["cn"]) for r in fin]

    def sim(i: int, j: int) -> float:
        if not a[i] or not b[j]:
            return 0.0
        s = 1.0 - Levenshtein.normalized_distance(a[i], b[j])
        dt = abs(src[i]["start"] - fin[j]["start"])
        if dt > 1500:  # 15 s apart: not the same line
            return 0.0
        return s

    NEG = -1e9
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = 0.0
    for j in range(m + 1):
        dp[0][j] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best, how = dp[i - 1][j], 1
            if dp[i][j - 1] > best:
                best, how = dp[i][j - 1], 2
            s = sim(i - 1, j - 1)
            if s >= 0.6 and dp[i - 1][j - 1] + s > best:
                best, how = dp[i - 1][j - 1] + s, 3
            dp[i][j], back[i][j] = best, how
    res = [-1] * n
    i, j = n, m
    while i > 0 and j > 0:
        how = back[i][j]
        if how == 3:
            res[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif how == 1:
            i -= 1
        else:
            j -= 1
    return res


def build(ep: int) -> dict:
    src_doc = AssDocument.load(origin_path(ep))
    cn = classify(src_doc, {"dialogue": ["Default"], "song_cn": ["OPCH", "ED"]})
    src = [
        {"start": ev.start, "end": ev.end, "cn": plain_text(ev.text, breaks=" ")}
        for i, ev in enumerate(src_doc.events)
        if cn.roles.get(i) in ("dialogue", "top")
    ]
    fin = final_rows(AssDocument.load(final_path(ep)))
    mapping = match(src, fin)
    rows = []
    for r, j in zip(src, mapping):
        row = dict(r)
        row["gold_jp"] = fin[j]["jp"] if j >= 0 else None
        row["final_cn"] = fin[j]["cn"] if j >= 0 else None
        rows.append(row)
    matched = sum(1 for j in mapping if j >= 0)
    return {"episode": ep, "jp_srt": f"jp/E{ep:02d}.ja.srt", "cn_source": str(origin_path(ep)),
            "rows": rows, "matched": matched, "final_rows": len(fin)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in range(1, 11):
        data = build(ep)
        (OUT / f"gold_E{ep:02d}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"E{ep:02d}: source rows={len(data['rows'])} matched={data['matched']} final rows={data['final_rows']}")
