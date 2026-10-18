"""Isolated song/dialogue alignment: python -m subkit.kalign.worker REQUEST.json RESULT.json."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np

from subkit import proc
from subkit.audio import ffmpeg_audio_index
from . import cache, dialogue, models, song

STAGES = {"extract", "download", "separate", "model1", "model2", "align", "done"}


@dataclass
class Request:
    video: str
    audio_track: int | None
    clip: tuple[int, int] | None
    cache_dir: str
    lines: list[song.LineInput] | list[dialogue.DialogueLine]
    options: dict
    kind: str = "song"


def request_from_json(value):
    kind = value.get("kind", "song")
    if kind == "dialogue":
        lines = [dialogue.DialogueLine(**row) for row in value["lines"]]
        if value.get("clip") is not None:
            raise ValueError("dialogue requests use whole-episode audio, without a clip")
        if not lines or len({line.mid for line in lines}) != len(lines):
            raise ValueError("expected nonempty lines with unique member ids")
        for line in lines:
            if (type(line.start) is not int or type(line.end) is not int
                    or line.start < 0 or line.end < line.start
                    or not isinstance(line.kana, str) or not isinstance(line.signature, str)):
                raise ValueError("invalid dialogue line")
        return Request(value["video"], value.get("audio_track"), None, value["cache_dir"], lines,
                       value.get("options", {}), kind)
    if kind != "song":
        raise ValueError("unknown alignment request kind")
    lines = []
    for row in value["lines"]:
        row = dict(row)
        beats = []
        for item in row.pop("beats"):
            item = dict(item)
            if item.get("fixed") is not None:
                item["fixed"] = tuple(item["fixed"])
            beats.append(song.BeatInput(**item))
        lines.append(song.LineInput(beats=beats, **row))
    clip = tuple(value["clip"])
    if len(clip) != 2 or any(isinstance(t, bool) or not isinstance(t, int) for t in clip):
        raise ValueError("clip must contain two integer milliseconds")
    clip = cache.rounded_clip(clip)
    if clip[0] < 0 or clip[1] <= clip[0]:
        raise ValueError("clip must be a positive range")
    if not lines or len({line.mid for line in lines}) != len(lines):
        raise ValueError("expected nonempty lines with unique member ids")
    options = value.get("options", {})
    offset = options.get("offset", "habit")
    if options.get("last_end", "voice") not in ("line", "voice") or not (
            offset in ("none", "habit") or isinstance(offset, int) and not isinstance(offset, bool)):
        raise ValueError("invalid alignment options")
    return Request(value["video"], value.get("audio_track"), clip, value["cache_dir"], lines, options)


def result_from_json(value):
    if isinstance(value, list):
        # Dialogue records include signatures and null boundaries for abstentions.
        return value
    value = dict(value)
    value["lines"] = [song.LineResult(row["mid"], [tuple(b) for b in row["beats"]]) for row in value["lines"]]
    return song.SongResult(**value)


def parse_progress(line):
    """Return a progress/error object, or None for logs/malformed JSON."""
    try:
        value = json.loads(line)
    except (ValueError, TypeError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("error"), str):
        return {"error": value["error"]}
    done, total = value.get("done"), value.get("total")
    if (value.get("stage") in STAGES and type(done) is int and type(total) is int
            and 0 <= done <= total and total > 0):
        return {"stage": value["stage"], "done": done, "total": total}
    return None


def progress(stage, done=0, total=1):
    print(json.dumps({"stage": stage, "done": done, "total": total}), flush=True)


def run_request(request: Request, notify=progress):
    if request.kind == "dialogue":
        return run_dialogue_request(request, notify)
    stdout = sys.stdout
    def download():
        with redirect_stdout(stdout):
            notify("download")

    video = Path(request.video)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    index = ffmpeg_audio_index(request.audio_track)
    directory = cache.cache_directory(Path(request.cache_dir) / "kalign", video, index, request.clip)
    vocals_path = directory / "vocals.npy"
    notify("extract")
    if vocals_path.exists():
        vocals = np.load(vocals_path, allow_pickle=False)
        notify("extract", 1, 1)
        notify("separate", 1, 1)
    else:
        a, b = request.clip
        raw = proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{a / 1000:.3f}",
                        "-i", str(video), "-t", f"{(b - a) / 1000:.3f}", "-map", f"0:a:{index}",
                        "-ac", "2", "-ar", "44100", "-f", "f32le", "-"],
                       capture_output=True, check=True, timeout=180).stdout
        mix = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
        if not len(mix):
            raise ValueError("ffmpeg returned empty audio")
        notify("extract", 1, 1)
        notify("separate")
        with redirect_stdout(sys.stderr):
            vocals = models.separate_vocals(mix, download=download)
        cache.write_array(vocals_path, vocals)
        notify("separate", 1, 1)
    rms_path = directory / "rms.npy"
    if rms_path.exists():
        rms = np.load(rms_path, allow_pickle=False)
    else:
        rms = models.rms_db_10ms(vocals)
        cache.write_array(rms_path, rms)
    emissions, mono16 = [], None
    for stage, name, infer in (("model1", "mms", models.mms_emission), ("model2", "kana", models.kana_emission)):
        notify(stage)
        emission = cache.load_emission(directory, name)
        if emission is None:
            if mono16 is None:
                from scipy.signal import resample_poly
                mono16 = resample_poly(vocals.mean(axis=1), 160, 441).astype(np.float32)
            # Third-party loader chatter belongs on stderr, preserving JSON stdout.
            # Restore stdout temporarily for our progress callback.
            stdout = sys.stdout
            def model_progress(i, n):
                with redirect_stdout(stdout):
                    notify(stage, i, n)
            with redirect_stdout(sys.stderr):
                emission = infer(mono16, progress=model_progress, download=download)
            cache.write_emission(directory, emission)
        emissions.append(emission)
        notify(stage, 1, 1)
    notify("align")
    result = song.run(request.lines, emissions, rms, request.clip[0], request.options)
    notify("align", 1, 1)
    return result


def run_dialogue_request(request: Request, notify=progress):
    """Whole-episode cache: float16 emissions + RMS, never separated vocals.

    Results are flat edge records with each input's signature. Missing edges
    have null speech boundaries (and no_text for empty readings). Every run
    realigns the supplied windows/text; cached audio cannot preserve stale rows.
    """
    video = Path(request.video)
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    index = ffmpeg_audio_index(request.audio_track)
    directory = cache.cache_directory(Path(request.cache_dir) / "kalign", video, index, None)
    emissions = [cache.load_emission(directory, name) for name in ("mms", "kana")]
    try:
        rms = np.load(directory / "rms.npy", allow_pickle=False)
    except (OSError, ValueError):
        rms = None
    stdout = sys.stdout
    def download():
        with redirect_stdout(stdout):
            notify("download")
    notify("extract")
    mono16 = None
    if rms is None or any(e is None for e in emissions):
        raw = proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                        "-map", f"0:a:{index}", "-ac", "2", "-ar", "44100", "-f", "f32le", "-"],
                       capture_output=True, check=True, timeout=180).stdout
        mix = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
        if not len(mix):
            raise ValueError("ffmpeg returned empty audio")
        notify("extract", 1, 1)
        notify("separate")
        with redirect_stdout(sys.stderr):
            vocals = models.separate_vocals(mix, download=download)
        del mix, raw
        if rms is None:
            rms = models.rms_db_10ms(vocals)
            cache.write_array(directory / "rms.npy", rms)
        if any(e is None for e in emissions):
            from scipy.signal import resample_poly
            mono16 = resample_poly(vocals.mean(axis=1), 160, 441).astype(np.float32)
        del vocals
    else:
        notify("extract", 1, 1)
    notify("separate", 1, 1)
    for i, (stage, infer) in enumerate((("model1", models.mms_emission), ("model2", models.kana_emission))):
        notify(stage)
        if emissions[i] is None:
            def model_progress(done, total):
                with redirect_stdout(stdout):
                    notify(stage, done, total)
            with redirect_stdout(sys.stderr):
                emission = infer(mono16, progress=model_progress, download=download)
            cache.write_emission(directory, emission, dtype=np.float16)
            # First and subsequent runs use the same cached numerical precision.
            emissions[i] = cache.load_emission(directory, emission.name)
            del emission
        notify(stage, 1, 1)
    del mono16
    notify("align", 0, len(request.lines))
    edges = dialogue.align_dialogue(request.lines, emissions, rms,
                                   progress=lambda done, total: notify("align", done, total))
    result = [dict(asdict(edge) if edge is not None else
                   dict(mid=line.mid, speech_start=None, speech_end=None, conf=0.0,
                        flags=["no_text"] if not line.kana else []), signature=line.signature)
              for line, edge in zip(request.lines, edges)]
    cache.write_json(directory / "dialogue.json", result)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    try:
        request = request_from_json(json.loads(args.request.read_text(encoding="utf-8")))
        result = run_request(request)
        cache.write_json(args.result, result if request.kind == "dialogue" else asdict(result))
        progress("done", 1, 1)
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
