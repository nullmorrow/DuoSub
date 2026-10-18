"""Disposable translation-only worker, with the same owned process lifetime as ASR."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def main(path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from subkit.listening_process import guard_worker
    guard_worker()
    from subkit.listening_progress import write_progress
    from subkit.listening_semantics import translate

    payload = json.loads(path.read_text(encoding="utf-8"))
    write_progress(path.parent, "正在加载本地翻译模型并尝试不同说法（不重新识别音频）…")
    result = translate(payload["translation"], device=payload["device"],
                       existing=tuple(payload["existing"]), round_index=payload["round_index"])
    (path.parent / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]))
    except Exception as exc:
        print(f"生成翻译建议失败：{exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
