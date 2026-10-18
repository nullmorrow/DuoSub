"""Generate another semantic batch without touching existing audio or ASR results."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Callable

from .proc import KillableRun


def generate_semantic_batch(request, output: Path, *, existing: tuple[str, ...], round_index: int,
                            device: str, runner: KillableRun, progress: Callable[[str], None]) -> dict:
    if (not isinstance(request.translation, str) or not request.translation.strip()
            or len(request.translation) > 1000):
        raise ValueError("生成翻译建议需要1000字以内的中文译文")
    if type(round_index) is not int or round_index < 1:
        raise ValueError("翻译轮次应为正整数")
    if device not in ("cuda", "cpu"):
        device = "cpu"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("生成更多翻译建议需要新的临时目录")
    runner.check_cancelled()
    payload = {"translation": request.translation, "existing": list(existing),
               "round_index": round_index, "device": device}
    source = output / "request.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    program = request.python or os.environ.get("SUBKIT_ASR_PYTHON") or sys.executable
    worker = Path(__file__).with_name("listening_semantic_worker.py")
    progress("正在生成更多翻译建议；保留原候选，不重新识别音频…")
    response = runner.run([program, "-B", "-X", "utf8", str(worker), str(source)], timeout=270)
    runner.check_cancelled()
    result_file = output / "result.json"
    if response.returncode or not result_file.is_file():
        error = response.stderr.decode("utf-8", "replace")[-2000:].strip()
        raise RuntimeError(error or f"翻译建议进程退出（{response.returncode}）")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not isinstance(result.get("texts"), list) or not result.get("model"):
        raise ValueError("本地翻译未返回有效候选")
    progress("正在合并新的翻译建议…")
    return result
