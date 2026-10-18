"""Run the legacy dual_sub_prep v6.4 (read-only, from the Utena folder) on the
Utena inputs and score it — the baseline the new aligner must beat.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from subkit.ass import AssDocument  # noqa: E402
from subkit.ass.text import plain_text  # noqa: E402

from score_align import DATA, load_gold, score  # noqa: E402

UTENA = Path('./sample-media/少女革命ウテナ')
LEGACY = UTENA / "dual_sub_prep_v6_4_cn_linebreak_space.py"


def run(ep: int) -> list[str | None]:
    gold = load_gold(ep)
    cache = DATA / f"v64_E{ep:02d}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / f"E{ep:02d}.ass"
        shutil.copy(gold["cn_source"], src)
        shutil.copy(UTENA / "jp_remove_terms.txt", tmp / "jp_remove_terms.txt")
        out = tmp / "out.ass"
        cmd = [sys.executable, str(LEGACY), str(src), "--jp", str(DATA / gold["jp_srt"]), "--source-style", "Default",
               "--no-prompt", "-o", str(out), "--no-style-template", "--no-scale-playres", "--candidate-terms-file", ""]
        subprocess.run(cmd, check=True, capture_output=True, cwd=tmp)
        doc = AssDocument.load(out)
    jp = {}
    for e in doc.events:
        if e.is_dialogue and e.style == "JP" and not e.effect.startswith("orphan"):
            jp[(e.start, e.end)] = plain_text(e.text, breaks=" ")
    pred = [jp.get((r["start"], r["end"])) for r in gold["rows"]]
    cache.write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    return pred


if __name__ == "__main__":
    tot = {"rows": 0, "exact": 0.0, "lcs_f1": 0.0}
    for ep in range(1, 11):
        res = score(run(ep), load_gold(ep)["rows"])
        print(f"E{ep:02d} rows={res['rows']} exact={res['exact']:.3f} lcs_f1={res['lcs_f1']:.3f}")
        tot["rows"] += res["rows"]
        tot["exact"] += res["exact"] * res["rows"]
        tot["lcs_f1"] += res["lcs_f1"] * res["rows"]
    print(f"ALL rows={tot['rows']} exact={tot['exact']/tot['rows']:.3f} lcs_f1={tot['lcs_f1']/tot['rows']:.3f}")
