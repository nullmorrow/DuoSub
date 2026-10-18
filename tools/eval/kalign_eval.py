"""Read-only Sword Oratoria ED evaluation; models run only in the worker process."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from subkit import proc
from subkit.audio import ffmpeg_audio_index
from subkit.core import karaoke
from subkit.core.readings import visible_map
from subkit.kalign import cache, song, tokens, worker
from subkit.kalign.request import build_lines
from subkit.model.serialize import load_document


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT / "Projects/ソード・オラトリア")
    parser.add_argument("--episode", default="S01E001")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offset", choices=("none", "habit"), default="none")
    return parser


def build_request(project, episode, cache_dir, offset="none"):
    """Read existing effective original beats; do not constrain predictions to gold."""
    project = Path(project)
    ep = project / "episodes" / episode
    doc = load_document(ep / "document.json")
    metadata = json.loads((ep / "episode.json").read_text(encoding="utf-8"))
    settings = json.loads((project / "subkit-project.json").read_text(encoding="utf-8")).get("settings", {})
    lines, hand, _signatures, references = build_lines(doc,
        [g.id for g in doc.groups if g.kind == "song"], fixed=False)
    if not lines:
        raise ValueError("episode contains no original song beats")
    video = Path(metadata["media"]["video"]["path"])
    if not video.is_absolute():
        video = project / video
    options = {"last_end": "voice", "offset": offset, "hand_beats": hand, "continuation_mode": "even"}
    stored = settings.get("karaoke_auto", {}).get("stored_habit")
    if stored is not None:
        options["stored_habit"] = stored
    request = worker.Request(str(video), metadata["media"].get("audio_track"),
                             cache.rounded_clip((max(0, min(line.start for line in lines) - 2000),
                                                 max(line.end for line in lines) + 2000)),
                             str(cache_dir), lines, options)
    return request, references


def metric_line(name, starts, references):
    """Identical start exclusions to tools/experiments/kalign/evaluate.py."""
    errors = []
    if len(starts) != len(references):
        raise ValueError("prediction/reference line counts differ")
    for pred, line in zip(starts, references):
        if len(pred) != len(line["beats"]):
            raise ValueError("prediction/reference beat counts differ")
        for i, (t0, ref) in enumerate(zip(pred, line["beats"])):
            if ref["by"] not in ("drag", "manual", "onset"):
                continue
            if i == 0 and ref["t0"] == line["start"]:
                continue
            errors.append(t0 - ref["t0"])
    if not errors:
        return f"{name}: no comparable hand-timed beats"
    a = np.abs(errors)
    return (f"{name}: n={len(a)} median={np.median(a):.0f} ms "
            + " ".join(f"<={t} ms={np.mean(a <= t) * 100:.0f}%" for t in (50, 100, 200)))



MIN_BEAT_MS = 100  # Musical anomaly threshold, also used for sustained silence.
RMS_BIN_MS = 10
SILENCE_PERCENTILE = 80
SILENCE_DROP_DB = 18


def anomalies(result, lines, rms, clip_ms):
    """Anomalies excluding sokuon-only and line-final beats; retain review details."""
    silent = song.silence_mask(rms)
    short = starts_silent = gaps = 0
    details = []
    for row, line in zip(result.lines, lines):
        for i, (a, z, _conf, _check) in enumerate(row.beats):
            if i + 1 == len(row.beats) or song.sokuon_only(line.beats[i].sung):
                continue
            too_short = z - a < MIN_BEAT_MS
            short += int(too_short)
            in_silence = song.silent_start(a, silent, clip_ms)
            starts_silent += int(in_silence)
            gap = bool(i and a - row.beats[i - 1][1] >= MIN_BEAT_MS and (in_silence or too_short))
            gaps += int(gap)
            reasons = [name for name, marked in (("short", too_short), ("silent-start-100ms", in_silence),
                                                  ("suspicious-gap", gap)) if marked]
            if reasons:
                details.append(f"review line={line.text!r} mid={line.mid} beat={i}:{line.beats[i].sung} "
                               f"times={a}-{z} ms reasons={','.join(reasons)} check={_check}")
    return short, starts_silent, gaps, details


def anomaly_line(result, lines, rms, clip_ms):
    short, starts_silent, gaps, _details = anomalies(result, lines, rms, clip_ms)
    return f"fused anomalies: short={short} silent-start-100ms={starts_silent} suspicious-gap={gaps}"


def continuation_metric(result, lines, references):
    errors = []
    details = []
    for row, line, ref in zip(result.lines, lines, references):
        mask = tokens.continuation_morae([b.sung for b in line.beats],
                                         tokens.word_starts([(b.a, b.from_ruby) for b in line.beats], line.text))
        for i, is_cont in enumerate(mask):
            if is_cont and ref["beats"][i]["by"] in ("tap", "drag", "manual", "onset"):
                start, hand = row.beats[i][0], ref["beats"][i]["t0"]
                errors.append(start - hand)
                details.append(f"continuation {line.beats[i-1].sung}|{line.beats[i].sung}: start={start} reference={hand} error={start-hand:+d} ms")
    if not errors:
        return ["fused continuation: no comparable hand-timed beats"]
    absolute = np.abs(errors)
    return [f"fused continuation: n={len(errors)} median={np.median(absolute):.0f} ms max={max(absolute):.0f} ms", *details]


def main(argv=None):
    args = argument_parser().parse_args(argv)
    started = time.perf_counter()
    scratch = Path(tempfile.mkdtemp(prefix="subkit-kalign-eval-"))
    cache_dir = args.cache_dir or scratch / "cache"
    request, references = build_request(args.project, args.episode, cache_dir, args.offset)
    request_path, result_path = scratch / "request.json", scratch / "result.json"
    cache.write_json(request_path, asdict(request))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = proc.run([sys.executable, "-B", "-X", "utf8", "-m", "subkit.kalign.worker",
                          str(request_path), str(result_path)], env=env, capture_output=True,
                         text=True, encoding="utf-8", timeout=600)
    (scratch / "worker-stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (scratch / "worker-stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"worker failed: {completed.stdout.strip()}; evidence: {scratch}")
    result = worker.result_from_json(json.loads(result_path.read_text(encoding="utf-8")))
    directory = cache_dir / "kalign" / cache.cache_key(request.video, ffmpeg_audio_index(request.audio_track), request.clip)
    rms = np.load(directory / "rms.npy", allow_pickle=False)
    for name in ("mms", "kana"):
        emission = cache.load_emission(directory, name)
        if emission is None:
            raise RuntimeError(f"missing {name} emission cache")
        aligned = song.align_model(request.lines, emission, request.clip[0], rms)
        print(metric_line(name, [[max(0, b.start + result.offset_ms) for b in line] for line in aligned.lines], references))
    print(metric_line("fused", [[b[0] for b in line.beats] for line in result.lines], references))
    print(anomaly_line(result, request.lines, rms, request.clip[0]))
    for text in anomalies(result, request.lines, rms, request.clip[0])[3]:
        print(text)
    for text in continuation_metric(result, request.lines, references):
        print(text)
    print(f"evidence: {scratch}")
    print(f"total elapsed seconds: {time.perf_counter() - started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
