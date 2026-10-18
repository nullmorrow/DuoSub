"""Read-only P1 validation; stdout is one JSON summary, no media-side writes."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from subkit import audio, audio_match as am, proc


def ass_time(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    whole, centiseconds = seconds.split(".")
    return (int(hours) * 3600 + int(minutes) * 60 + int(whole)) * 1000 + int(centiseconds) * 10


def lines(path: Path, style: str) -> list[tuple[int, int, str]]:
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("Dialogue:"):
            fields = line.split(":", 1)[1].lstrip().split(",", 9)
            if len(fields) == 10 and fields[3] == style:
                result.append((ass_time(fields[1]), ass_time(fields[2]), re.sub(r"\{[^}]*\}", "", fields[9])))
    return sorted(result)


def clusters(items: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    result = []
    for start, end, _ in items:
        if not result or start - result[-1][1] > 60000:
            result.append((start, end))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], end))
    return result


def lyric_key(text: str) -> str:
    # Fansubs use different spaces and ASS line breaks for the same lyric.
    return re.sub(r"\s+", "", text.replace(r"\N", "").replace(r"\n", "").replace(r"\h", ""))


def anchor(source: Path, target: Path, span: tuple[int, int], style: str) -> tuple[int, dict]:
    first = next(line for line in lines(source, "JPOPED") if span[0] <= line[0] < span[1])
    target_lines = lines(target, style)
    for source_style, lyric in [("JPOPED", first)] + [("CNOPED", line) for line in lines(source, "CNOPED") if line[0] == first[0]]:
        same = [line for line in target_lines if lyric_key(line[2]) == lyric_key(lyric[2])]
        if same:
            found = same[0]
            offset = lyric[0] - span[0]
            return found[0] - offset, {"source_style": source_style, "source_start_ms": lyric[0],
                                       "source_offset_ms": offset, "source_text": lyric[2],
                                       "target_style": style, "target_start_ms": found[0],
                                       "target_text": found[2], "comparison": "tags stripped; whitespace and ASS line breaks ignored"}
    raise ValueError(f"no identical lyric anchor for {style}: {first[2]}")


def media(folder: str, prefix: str) -> Path:
    matches = sorted(Path(folder).glob(prefix + "*.mkv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {prefix} media in {folder}, got {len(matches)}")
    return matches[0]


def decode(path: Path) -> tuple[audio.AudioFeatures, float, float]:
    begin = time.perf_counter()
    raw = proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-map", "0:a:0", "-ac", "1", "-ar", str(audio.RATE), "-f", "s16le", "-"],
                   capture_output=True, check=True).stdout
    decoded = time.perf_counter() - begin
    begin = time.perf_counter()
    features = audio.compute_features(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768)
    return features, decoded, time.perf_counter() - begin


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paths = {"utena16": media('./sample-media/少女革命ウテナ', "S02E003"),
             "utena17": media('./sample-media/少女革命ウテナ', "S02E004"),
             "sword": media('./sample-media/ソード・オラトリア', "S01E001"),
             "gundam": media('./sample-media/機動戦士ガンダム', "S01E001")}
    ass16, ass17 = paths["utena16"].with_suffix(".zh-Hant.ass"), paths["utena17"].with_suffix(".zh-Hant.ass")
    songs = clusters(lines(ass16, "JPOPED"))
    inserts = clusters(lines(ass17, "IN"))
    op_expected, op_anchor = anchor(ass16, ass17, songs[0], "OPCH")
    ed_expected, ed_anchor = anchor(ass16, ass17, songs[-1], "ED")
    summary = {"environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                               "ffmpeg": proc.run(["ffmpeg", "-version"], capture_output=True, check=True).stdout.decode("utf-8").splitlines()[0],
                               "audio": "0:a:0 / 16000 Hz mono s16le; audio-v3"},
               "thresholds": {"segment": am.SEGMENT_THRESHOLD, "single": am.SINGLE_THRESHOLD,
                              "consensus_ms": am.CONSENSUS_MS, "max_scale_delta": am.MAX_SCALE_DELTA},
               "template_ranges_ms": {"op": songs[0], "ed": songs[-1], "in": inserts[0]},
               "anchors": {"op": op_anchor, "ed": ed_anchor},
               "media": {}, "searches": []}
    features = {}
    for name in ("utena16", "utena17"):
        features[name], decoded, computed = decode(paths[name])
        summary["media"][name] = {"decode_seconds": decoded, "feature_seconds": computed}
    op = am.template_from(features["utena16"], *songs[0])
    ed = am.template_from(features["utena16"], *songs[-1])
    ins = am.template_from(features["utena17"], *inserts[0])
    cases = [("op_in_utena17", "utena17", op, op_expected),
             ("ed_in_utena17", "utena17", ed, ed_expected),
             ("in_in_utena16", "utena16", ins, None),
             ("op_in_sword", "sword", op, None), ("op_in_gundam", "gundam", op, None)]
    indices = {}
    for name, target, template, expected in cases:
        if target not in features:
            features[target], decoded, computed = decode(paths[target])
            summary["media"][target] = {"decode_seconds": decoded, "feature_seconds": computed}
        if target not in indices:
            begin = time.perf_counter()
            indices[target] = am.EpisodeIndex(features[target].spectrogram)
            summary["media"][target]["index_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        result = indices[target].find(template)
        elapsed = time.perf_counter() - begin
        found = [{"start_ms": m.start_ms, "score": m.score, "covered": m.covered,
                  "drift_ms": m.drift_ms, "scale": m.scale} for m in result.matches]
        difference = min((m.start_ms - expected for m in result.matches), key=abs, default=None) if expected is not None else None
        passed = (difference is not None and abs(difference) <= 500 and
                  any(abs(m.start_ms - expected) <= 500 and m.score >= am.SEGMENT_THRESHOLD for m in result.matches)) if expected is not None else (not result.matches if name.startswith("op_in_") else None)
        previous = {"op_in_utena17": (8220, 0.99994), "ed_in_utena17": (1307210, 0.99991)}
        equivalent = None
        if name in previous:
            old_start, old_score = previous[name]
            closest = min(result.matches, key=lambda m: abs(m.start_ms - old_start), default=None)
            equivalent = {"start_difference_ms": closest.start_ms - old_start if closest else None,
                          "score_difference": closest.score - old_score if closest else None}
            passed = passed and closest is not None and abs(closest.start_ms - old_start) <= 10 and abs(closest.score - old_score) <= 0.02
        summary["searches"].append({"name": name, "matches": found, "expected_start_ms": expected,
                                    "difference_ms": difference, "best_score": result.best_score,
                                    "decode_seconds": summary["media"][target]["decode_seconds"],
                                    "search_seconds": elapsed, "previous_run_difference": equivalent, "accepted": passed})
    summary["accepted"] = all(s["accepted"] for s in summary["searches"] if s["accepted"] is not None)
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        raise SystemExit(2)
