"""Inference child: PCM/scipy I/O, with Windows helper lifetimes guarded by a Job."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import time


def main(path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from subkit.listening_process import guard_worker
    guard_worker()
    from subkit.listening_progress import write_progress
    output = path.parent
    write_progress(output, "正在加载本地听译运行环境…")
    from subkit.local_paths import local_value
    # Optional packages remain machine-local; never alter global installs.
    packages = local_value("SUBKIT_ASR_PACKAGES")
    if packages and Path(packages).is_dir():
        sys.path.append(packages)
    import numpy as np
    import torch
    import whisper
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    torch.set_num_threads(4)
    torch.manual_seed(0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    request = payload["request"]
    write_progress(output, "正在读取音频并检查声道…")
    rate, raw = wavfile.read(output / "original.wav")
    mix = raw.astype(np.float32) / 32768
    if rate != 44100 or mix.ndim != 2 or mix.shape[1] != 2 or len(mix) < rate // 20:
        raise ValueError("没有足够的可识别音频")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {"clips": {"original": "original.wav"}, "device": device, "runs": [], "diagnostics": [],
            "duration_ms": round(len(mix) * 1000 / rate),
            "metadata": {"torch": torch.__version__, "whisper": whisper.__version__,
                         "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "", "stages_seconds": {},
                         "hint": request.get("hint", "")}}
    variants = [("原声混音", "original", mix.mean(1), False)]
    left, right = mix.T
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    # Nearly dual-mono speech cancels in both channels; avoid presenting the
    # same foreground three times and turning cancellation noise into speech.
    if rms(left - right) > max(1e-6, rms(mix) * .20):
        variants.extend([("左声道", "left", left, False), ("右声道", "right", right, False),
                         ("侧向消声", "side", (left - right) / 2, True)])
    if request["separate"]:
        checkpoint = Path(local_value("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub/checkpoints/955717e8-8726e21a.th"
        if not checkpoint.is_file():
            data["diagnostics"].append("没有本地 Demucs htdemucs 模型，已仅识别原音和声道；没有下载。")
        else:
            try:
                write_progress(output, "正在加载人声分离模型…")
                from demucs.apply import apply_model
                from demucs.pretrained import get_model
                t = time.monotonic()
                # This legacy trusted checkpoint contains its model class. Scope the
                # compatibility switch strictly to this child and this load.
                previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
                try:
                    demucs = get_model("955717e8", repo=checkpoint.parent).eval().to(device)
                finally:
                    if previous is None:
                        os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
                    else:
                        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous
                x = torch.from_numpy(mix.T.copy())
                ref = x.mean(0)
                mean, std = ref.mean(), ref.std()
                if float(std) > 1e-7:
                    write_progress(output, "正在分离人声与伴奏…")
                    with torch.inference_mode():
                        stems = apply_model(demucs, ((x - mean) / std)[None], device=device,
                                            shifts=0, split=True, overlap=.25, progress=False)[0]
                    vocals = (stems[demucs.sources.index("vocals")].cpu() * std + mean).numpy().T
                    variants.extend([("分离人声", "vocals", vocals.mean(1), False),
                                     ("去人声残余", "residual", (mix - vocals).mean(1), True)])
                    del stems, vocals
                del demucs, x, ref, mean, std
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                data["metadata"]["stages_seconds"]["separate"] = time.monotonic() - t
            except Exception as exc:
                data["diagnostics"].append("分离未完成，继续识别原音和声道：" + str(exc))
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
    t = time.monotonic()
    write_progress(output, f"正在加载 Whisper {request['model']}（{device}）…")
    # Absolute existing file path prevents whisper.load_model's URL download branch.
    model = whisper.load_model(payload["model_file"], device=device)
    data["metadata"]["stages_seconds"]["load"] = time.monotonic() - t
    for label, key, audio, distorted in variants:
        if rms(audio) < 1e-5:
            continue
        peak = float(np.max(np.abs(audio)))
        audio = np.asarray(audio * min(8, .95 / max(peak, 1e-6)), dtype=np.float32)
        if key != "original":
            name = key + ".wav"
            wavfile.write(output / name, rate, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
            data["clips"][key] = name
        samples = resample_poly(audio, 160, 441).astype(np.float32)
        windows = [(label, samples, 0., False)]
        if key in ("original", "vocals"):
            # A broad context can mask a quiet, short utterance. Keep an explicitly
            # marked narrow alternative; playback still uses the full-window WAV.
            a = payload["window_start_ms"]
            lo = max(0, round((request["start_ms"] - a - 300) * 16))
            hi = min(len(samples), round((request["end_ms"] - a + 300) * 16))
            if hi - lo >= 4000 and hi - lo < len(samples) - 1600:
                windows.append((label + "短窗", samples[lo:hi], lo / 16000, True))
        decode_windows = [(source, clip, offset, short, "") for source, clip, offset, short in windows]
        if request.get("hint") and key in ("original", "vocals"):
            decode_windows.extend((source + "·提示词", clip, offset, short, request["hint"])
                                  for source, clip, offset, short in windows)
        for source, clip, offset, short, hint in decode_windows:
            write_progress(output, f"正在听译：{source}（{request['model']}／{device}）…")
            t = time.monotonic()
            result = model.transcribe(clip, language="ja", task="transcribe", beam_size=5,
                                      temperature=0., condition_on_previous_text=False, initial_prompt=hint or None,
                                      word_timestamps=True, fp16=device == "cuda", verbose=None)
            segments = [{k: s.get(k) for k in ("text", "start", "end", "avg_logprob", "no_speech_prob", "words")}
                        for s in result["segments"]]
            for seg in segments:
                seg["start"] += offset
                seg["end"] += offset
                for word in seg["words"]:
                    word["start"] += offset
                    word["end"] += offset
            data["runs"].append({"source": source, "clip": key, "distorted": distorted, "short": short, "hint": hint,
                                 "segments": segments, "seconds": time.monotonic() - t})
    write_progress(output, "正在保存识别结果…")
    if request.get("semantic"):
        # Translation is deliberately separate from Whisper: prompted echoes do
        # not count as acoustic confirmation. Release ASR VRAM before loading MT.
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        write_progress(output, "正在根据中文生成日语语义候选（未经音频确认）…")
        try:
            from subkit.listening_semantics import translate
            t = time.monotonic()
            data["semantic"] = translate(request["translation"], device=device)
            data["metadata"]["semantic"] = {k: v for k, v in data["semantic"].items() if k != "texts"}
            data["metadata"]["stages_seconds"]["semantic"] = time.monotonic() - t
        except Exception as exc:
            data["diagnostics"].append("日语语义候选未生成；音频识别结果仍保留：" + str(exc))
    write_progress(output, "正在保存听译与语义候选…")
    (output / "result.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]))
    except Exception as exc:
        print(f"本地听译失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
