"""Local Japanese transcription; Chinese translation feeds a separate semantic model.

Audio extraction and inference are sequential, killable direct children. The inference
worker receives PCM WAVs, never starts ffmpeg, and never downloads model weights.
On Windows its Job also contains any helper a model library may create.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
import difflib
from pathlib import Path
from .local_paths import local_value
import sys
import unicodedata
from typing import Callable

from .audio import ffmpeg_audio_index
from .proc import KillableRun


@dataclass(frozen=True)
class ListeningRequest:
    video: str
    audio_track: int | None
    start_ms: int
    end_ms: int
    padding_ms: int = 1200
    separate: bool = True
    model: str = "large-v3"
    python: str = ""
    known_originals: tuple[str, ...] = ()
    hint: str = ""
    translation: str = ""
    semantic: bool = False


@dataclass
class Candidate:
    text: str
    start_ms: int
    end_ms: int
    sources: list[str]
    audio: str
    warnings: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    kind: str = "asr"


@dataclass
class ListeningResult:
    candidates: list[Candidate]
    clips: dict[str, str]
    window_start_ms: int
    window_end_ms: int
    device: str
    diagnostics: list[str]
    metadata: dict = field(default_factory=dict)


def clip_window(request: ListeningRequest) -> tuple[int, int]:
    for value in (request.start_ms, request.end_ms, request.padding_ms):
        if type(value) is not int:
            raise ValueError("听译时间必须是整数毫秒")
    if request.start_ms < 0 or not 0 < request.end_ms - request.start_ms <= 60000:
        raise ValueError("请选择 0–60 秒范围内的一条字幕")
    if not 0 <= request.padding_ms <= 5000:
        raise ValueError("向前后搜索的范围应为 0–5000 毫秒")
    a = max(0, request.start_ms - request.padding_ms)
    b = request.end_ms + request.padding_ms
    if b - a < 3000:
        extra = 3000 - (b - a)
        a = max(0, a - extra // 2)
        b = max(b + (extra - extra // 2), a + 3000)
    return a, b


def model_file(name: str) -> Path:
    if name not in ("medium", "large-v3"):
        raise ValueError("听译模型应为 medium 或 large-v3")
    roots = [Path(local_value("SUBKIT_ASR_MODEL_DIR",
                              Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"))]
    for root in roots:
        path = root / (name + ".pt")
        if path.is_file():
            return path.resolve()
    raise ValueError(f"找不到本地 Whisper {name} 模型。请设置 SUBKIT_ASR_MODEL_DIR；听译不会自动下载模型。")


def _key(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def assemble(request: ListeningRequest, data: dict, output: Path) -> ListeningResult:
    """Keep timestamped hypotheses near the selected row; expose ambiguity, never certify it."""
    a, b = clip_window(request)
    b = min(b, a + int(data.get("duration_ms", b - a)))
    clips = {str(k): str(output / v) for k, v in data["clips"].items()}
    candidates: list[Candidate] = []
    for run in data["runs"]:
        source = run["source"]
        audio = clips[run["clip"]]
        for seg in run["segments"]:
            if not all(math.isfinite(float(seg.get(k, 0))) for k in ("start", "end", "avg_logprob", "no_speech_prob")):
                continue
            t0, t1 = a + round(seg["start"] * 1000), a + round(seg["end"] * 1000)
            t0, t1 = max(a, t0), min(b, t1)
            if t1 <= t0 or t1 < request.start_ms - 1000 or t0 > request.end_ms + 1000:
                continue
            warnings = ["机器听写候选，请试听确认；可能是同一时段的前景对白。"]
            distorted = run.get("distorted", False)
            low = seg.get("avg_logprob", -10) < -1 or seg.get("no_speech_prob", 1) > .6
            if distorted:
                warnings.append("经消声处理，可能失真或幻听。")
            if low:
                warnings.append("音频识别依据较弱。")
            if run.get("short"):
                warnings.append("短窗识别，短片段更容易出现幻听。")
            if run.get("hint"):
                warnings.append("使用了人工提示词，可能复述提示内容；请核对实际配音。")
            # A common silence hallucination is not a useful candidate from damaged audio.
            text = seg.get("text", "").strip()
            if distorted and any(s in _key(text) for s in ("ご視聴ありがとうございました", "チャンネル登録")):
                continue
            choices = [(text, t0, t1, False)]
            words = [w for w in seg.get("words", []) if math.isfinite(float(w.get("start", 0)))
                     and math.isfinite(float(w.get("end", 0))) and w["end"] > w["start"]
                     and request.start_ms - 250 <= a + (w["start"] + w["end"]) * 500 <= request.end_ms + 250]
            if words:
                partial = "".join(w["word"] for w in words).strip()
                if _key(partial) and _key(partial) != _key(text):
                    choices.insert(0, (partial, max(a, a + round(words[0]["start"] * 1000)),
                                       min(b, a + round(words[-1]["end"] * 1000)), True))
            for text, start, end, partial in choices:
                if not _key(text) or end <= start:
                    continue
                notes = warnings + (["按估计的逐词时间截取，可能缺少上下文。"] if partial else [])
                recognized = _key(text)
                if any(len(known := _key(other)) >= 4 and len(recognized) >= 4
                       and (recognized in known or known in recognized
                            or difflib.SequenceMatcher(None, recognized, known).ratio() >= .8)
                       for other in request.known_originals):
                    notes.append("与附近已有日文对白相似，很可能仍是前景声。")
                if end <= request.start_ms or start >= request.end_ms:
                    notes.append("位于原字幕时间附近，可能属于相邻一句。")
                evidence = {"source": source, "avg_logprob": seg.get("avg_logprob"),
                            "no_speech_prob": seg.get("no_speech_prob"), "partial": partial,
                            "distorted": distorted, "short": bool(run.get("short")), "hint": run.get("hint", ""),
                            "start_ms": start, "end_ms": end}
                duplicate = next((c for c in candidates if _key(c.text) == _key(text)
                                  and min(c.end_ms, end) > max(c.start_ms, start)), None)
                if duplicate is not None:
                    if source not in duplicate.sources:
                        duplicate.sources.append(source)
                    duplicate.evidence.append(evidence)
                    duplicate.warnings.extend(note for note in notes if note not in duplicate.warnings)
                    continue
                candidates.append(Candidate(text, start, end, [source], audio, notes, [evidence]))
    def rank(c):
        overlap = max(0, min(c.end_ms, request.end_ms) - max(c.start_ms, request.start_ms))
        damaged = all(e["distorted"] for e in c.evidence)
        foreground = any("附近已有日文" in warning for warning in c.warnings)
        return (damaged, foreground, not bool(overlap), -len(c.sources),
                abs((c.start_ms + c.end_ms) - (request.start_ms + request.end_ms)))
    candidates.sort(key=rank)
    result = ListeningResult(candidates[:24], clips, a, b, data.get("device", "unknown"),
                             data.get("diagnostics", []), {"model": request.model, **data.get("metadata", {})})
    semantic = data.get("semantic")
    if semantic and request.semantic:
        from .listening_semantics import add_semantic_candidates

        start, end = max(a, request.start_ms), min(b, request.end_ms)
        if start < end:
            added = add_semantic_candidates(result, semantic["texts"], translation=request.translation,
                                            start=start, end=end, origin="本地中译日模型", model=semantic["model"],
                                            provenance={k: v for k, v in semantic.items() if k != "texts"})
            if not added:
                result.diagnostics.append("中译日模型未生成可用语义候选；请试听后手动输入。")
        else:
            result.diagnostics.append("所选字幕超出实际音频范围，未添加语义候选。")
    return result


def generate(request: ListeningRequest, output: Path, *, runner: KillableRun,
             progress: Callable[[str], None]) -> ListeningResult:
    a, b = clip_window(request)
    if not isinstance(request.hint, str) or len(request.hint) > 80 or any(c in request.hint for c in "\r\n\x00"):
        raise ValueError("听到的词应为 80 字以内的一行文字")
    if request.semantic and (not isinstance(request.translation, str) or not request.translation.strip()
                             or len(request.translation) > 1000):
        raise ValueError("生成语义候选需要 1000 字以内的中文译文")
    video = Path(request.video)
    if not video.is_file():
        raise ValueError("找不到这一集的视频")
    weights = model_file(request.model)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    wav = output / "original.wav"
    if any(output.iterdir()):
        raise ValueError("听译需要一个新的临时目录")
    runner.check_cancelled()
    progress("读取所选音轨，向前后搜索附近的声音…")
    runner.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                "-ss", f"{a / 1000:.3f}", "-i", str(video), "-t", f"{(b - a) / 1000:.3f}",
                "-map", f"0:a:{ffmpeg_audio_index(request.audio_track)}", "-vn", "-ac", "2", "-ar", "44100",
                "-c:a", "pcm_s16le", str(wav)], check=True, timeout=90)
    payload = {"request": asdict(request), "model_file": str(weights), "window_start_ms": a, "window_end_ms": b}
    request_file = output / "request.json"
    request_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    program = request.python or os.environ.get("SUBKIT_ASR_PYTHON") or sys.executable
    worker = Path(__file__).with_name("listening_worker.py")
    progress("在独立进程中加载本地模型，比较原音、声道与分离音频（可取消）…")
    result = runner.run([program, "-B", "-X", "utf8", str(worker), str(request_file)], timeout=900)
    runner.check_cancelled()
    result_file = output / "result.json"
    if result.returncode != 0 or not result_file.is_file():
        message = result.stderr.decode("utf-8", errors="replace")[-4000:].strip()
        raise RuntimeError(message or f"听译进程退出（{result.returncode}）")
    data = json.loads(result_file.read_text(encoding="utf-8"))
    progress("整理听写候选…")
    return assemble(request, data, output)
