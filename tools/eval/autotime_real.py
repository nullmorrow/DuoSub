"""Read-only A1/A2 evaluation. One JSON on stdout; caches/evidence only in TEMP."""

from __future__ import annotations

from dataclasses import asdict
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from subkit import audio, autotime as at, proc
from subkit.encoding import decode_bytes, read_text
from subkit.core.srt import parse_srt


def ass_time(value: str) -> int:
    h, m, s = value.split(":")
    return round((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)


def ass_lines(path: Path, styles: tuple[str, ...]) -> list[tuple[int, int, str]]:
    result = []
    for line in read_text(path).text.splitlines():
        if line.startswith("Dialogue:"):
            fields = line.split(":", 1)[1].lstrip().split(",", 9)
            if len(fields) == 10 and fields[3] in styles:
                text = re.sub(r"\{[^}]*\}", "", fields[9]).replace(r"\N", " ")
                if text.strip():
                    result.append((ass_time(fields[1]), ass_time(fields[2]), text))
    return result


def srt_lines(data: bytes) -> list[tuple[int, int, str]]:
    return [(b.start, b.end, b.text) for b in parse_srt(decode_bytes(data).text) if b.text.strip()]


def one(folder: Path, pattern: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {folder}: {len(matches)}")
    return matches[0]


def features(path: Path, scratch: Path) -> tuple[audio.AudioFeatures, dict]:
    stat = path.stat()
    key = hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{audio.CACHE_VERSION}".encode("utf-8")).hexdigest()[:16]
    cache = scratch / (key + ".npz")
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            f = audio.AudioFeatures(*(data[k].copy() for k in ("mins", "maxs", "spectrogram", "onset")))
        return f, {"cached": True, "cache": str(cache), "duration_ms": len(f.onset) * 10}
    begin = time.perf_counter()
    raw = proc.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                    "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                   capture_output=True, check=True, timeout=120).stdout
    decoded = time.perf_counter() - begin
    begin = time.perf_counter()
    f = audio.compute_features(np.frombuffer(raw, "<i2").astype(np.float32) / 32768)
    computed = time.perf_counter() - begin
    with cache.open("wb") as stream:
        np.savez_compressed(stream, mins=f.mins, maxs=f.maxs, spectrogram=f.spectrogram, onset=f.onset)
    return f, {"cached": False, "cache": str(cache), "duration_ms": len(f.onset) * 10,
               "decode_seconds": round(decoded, 6), "feature_seconds": round(computed, 6)}


def intervals(lines):
    return [(s, e) for s, e, _ in lines if e > s]


def metrics(pairs, mapping, features):
    def values(lines):
        a = np.array([[abs(s[0] - t[0]), abs(s[1] - t[1])] for s, (_, t) in zip(lines, pairs)])
        return {"median_abs_start_ms": float(np.median(a[:, 0])),
                "median_abs_end_ms": float(np.median(a[:, 1])),
                "start_within_200ms": float(np.mean(a[:, 0] <= 200)),
                "end_within_200ms": float(np.mean(a[:, 1] <= 200))}
    if not pairs:
        return {"pairs": 0, "error": "no text matches"}
    original = [(s[0], s[1]) for s, _ in pairs]
    mapped = [(mapping.map(s), mapping.map(e)) for s, e in original]
    b, a = values(original), values(mapped)
    snapped = {}
    for mode in ("start", "both", "keep_duration"):
        begin = time.perf_counter()
        result = at.snap_lines(features, mapped, mode=mode)
        elapsed = time.perf_counter() - begin
        snapped[mode] = {**values(result), "changed_lines": sum(x != y for x, y in zip(mapped, result)),
                         "seconds": round(elapsed, 6)}
    return {"pairs": len(pairs), "before": b, "map_only": a, "map_and_snap": snapped,
            "median_not_worse": a["median_abs_start_ms"] <= b["median_abs_start_ms"]
            and a["median_abs_end_ms"] <= b["median_abs_end_ms"]}


def lead_measurement(features, lines):
    curve = at._snap_activity(features)
    starts, ends = [], []
    for start, end in sorted(set(intervals(lines))):
        onset, offset = at._speech_edges(curve, start, end, 600, 150, 300)
        if onset is not None:
            starts.append(start - onset)
        if offset is not None:
            ends.append(end - offset)
    def distribution(values):
        if not values:
            return {"clear_edges": 0}
        q1, median, q3 = np.percentile(values, (25, 50, 75))
        return {"clear_edges": len(values), "median_ms": float(median), "q1_ms": float(q1),
                "q3_ms": float(q3), "iqr_ms": float(q3 - q1)}
    return {"speech_groups": len(set(intervals(lines))), "start_minus_onset": distribution(starts),
            "end_minus_offset": distribution(ends), "search_ms": 600,
            "defaults_ms": {"lead_in": 150, "lead_out": 300},
            "proposal_for_claude_ms": {"lead_in": max(0, round(-float(np.median(starts)))) if starts else None,
                                       "lead_out": max(0, round(float(np.median(ends)))) if ends else None},
            "proposal_note": "defaults unchanged; heuristic medians disagree, Claude decides; negative measured end padding clamped to zero for proposal",
            "method": "local 250..3500Hz contrast runs; same heuristic as snapping; only clear nearby edges; not annotated speech truth"}


def pair_text(source, target):
    def key(text):
        return re.sub(r"[\W_]+", "", re.sub(r"\{[^}]*\}|<[^>]*>", "", text), flags=re.UNICODE)
    left, right = [key(s[2]) for s in source], [key(t[2]) for t in target]
    # Ordered identical-text blocks exclude ambiguous many-to-one matches.
    pairs = []
    for block in SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks():
        pairs.extend((source[block.a + i], target[block.b + i]) for i in range(block.size)
                     if left[block.a + i])
    return pairs


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    scratch = Path(os.environ["TEMP"]) / "autotime-real"
    scratch.mkdir(exist_ok=True)
    utena = Path('./sample-media/少女革命ウテナ')
    gank = Path('./sample-media/巌窟王 (2004)')
    paths = {"utena16": one(utena, "S02E003*.mkv"), "utena17": one(utena, "S02E004*.mkv"),
             "gank01": one(gank, "S01E001*.mp4"),
             "gank02": one(gank / "Subwork", "S01E002*.mp4"),
             "gank03": one(gank / "Working", "S01E003*.mp4")}
    lanes = {}
    with zipfile.ZipFile(utena / "Shoujo Kakumei Utena.zip") as archive:
        for episode in (16, 17):
            candidates = []
            for info in archive.infolist():
                name = info.filename
                if not info.flag_bits & 0x800:
                    try:
                        name = name.encode("cp437").decode("gbk")
                    except (UnicodeError, LookupError):
                        pass
                if f"S01E{episode}" in name and "netflix" in name.lower() and name.lower().endswith(".srt"):
                    candidates.append(info)
            if len(candidates) != 1:
                raise ValueError(f"ambiguous Netflix E{episode}: {[i.filename for i in candidates]}")
            lanes[f"utena{episode}"] = {"jp": srt_lines(archive.read(candidates[0])),
                                         "cn": ass_lines(paths[f"utena{episode}"].with_suffix(".zh-Hant.ass"), ("Default",))}
    for episode in (1, 2, 3):
        name = f"gank{episode:02d}"
        cn = paths[name].with_suffix(".zh-Hant.ass" + (".bak" if episode > 1 else ""))
        jp = gank / "Subwork/ep1.srt" if episode == 1 else paths[name].with_suffix(".srt" if episode == 2 else ".ja.srt")
        lanes[name] = {"cn": ass_lines(cn, ("Default",)), "jp": srt_lines(jp.read_bytes())}
    finished = one(gank / "Subwork/Done", "S01E001*.zh-Hant.ass")
    final_jp = ass_lines(finished, ("JP",))
    speech_spans = {(s, e) for s, e, text in final_jp if text.strip()}
    final_cn = [line for line in ass_lines(finished, ("CN",)) if line[:2] in speech_spans]
    summary = {"environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                "ffmpeg": proc.run(["ffmpeg", "-version"], capture_output=True, check=True).stdout.decode("utf-8").splitlines()[0],
                "audio": "0:a:0 / 16000 Hz mono s16le / audio-v3; host; no Qt"},
               "bias_ms": at.CORRELATION_BIAS_MS, "media": {}, "maps": {}, "e01_comparison": {}}
    times = []
    for name, path in paths.items():
        f, measurement = features(path, scratch)
        summary["media"][name] = measurement
        for lane, lines in lanes[name].items():
            runs = []
            for _ in range(3):
                begin = time.perf_counter()
                mapping = at.estimate(f, intervals(lines))
                runs.append(time.perf_counter() - begin)
            times.extend(runs)
            key = name + "_" + lane
            windows = mapping.windows
            detail = {"segments": [asdict(s) for s in mapping.segments], "lines": len(lines),
                      "raw_offsets_at_segment_start_ms": [s.offset_ms + at.CORRELATION_BIAS_MS for s in mapping.segments],
                      "window_counts": {c: sum(w.confidence == c for w in windows) for c in ("high", "medium", "low")},
                      "median_estimate_seconds": round(float(np.median(runs)), 6)}
            if name.startswith("utena") and lane == "jp":
                # Gap reference is source-time midpoint of the largest eyecatch gap.
                ordered = sorted(intervals(lines))
                gaps = [(b[0] - a[1], (a[1] + b[0]) / 2) for a, b in zip(ordered, ordered[1:])
                        if 550000 < (a[1] + b[0]) / 2 < 750000]
                gap = max(gaps)[1] if gaps else None
                first, last = mapping.segments[0], mapping.segments[-1]
                detail["acceptance"] = {"eyecatch_gap_ms": gap,
                    "targets_corrected_ms": {"first": -600, "second": 15420},
                    "targets_raw_ms": {"first": -440, "second": 15580},
                    "tolerances_ms": {"first": 150, "second": 200},
                    "first_offset_pass": abs(first.offset_ms + 600) <= 150,
                    "second_offset_pass": len(mapping.segments) >= 2 and abs(last.offset_ms - 15420) <= 200,
                    "break_pass": len(mapping.segments) == 2 and gap is not None and abs(last.start_ms - gap) <= 5000}
            summary["maps"][key] = detail
            (scratch / (key + "-windows.json")).write_text(mapping.to_json(), encoding="utf-8")
            if name == "gank01":
                target = final_cn if lane == "cn" else final_jp
                summary["e01_comparison"][lane] = metrics(pair_text(lines, target), mapping, f)
        if name == "gank01":
            calibration = {}
            for lane, lines in (("cn", final_cn), ("jp", final_jp)):
                curve = at.speech_activity(f)
                windows = at.window_lags(curve, intervals(lines))
                confident = [w.lag_ms for w in windows if w.confidence != "low"]
                calibration[lane] = {"speech_lines": len(lines), "reliable_windows": len(confident),
                    "median_raw_lag_ms": float(np.median(confident)) if confident else None,
                    "lag_iqr_ms": float(np.percentile(confident, 75) - np.percentile(confident, 25)) if confident else None}
            summary["calibration"] = calibration
            summary["lead_in_out"] = lead_measurement(f, final_jp)
    summary["performance"] = {"median_seconds": round(float(np.median(times)), 6), "target_seconds": 2.0,
                               "pass": float(np.median(times)) <= 2.0, "runs": len(times),
                               "excludes": "decode, features, cache I/O"}
    # A fixed 24-minute benchmark adds silence to the real E03 features;
    # decoding and padding are outside the measured estimate call.
    padded = audio.AudioFeatures(np.pad(f.mins, (0, max(0, 144000 - len(f.mins))))[:144000],
                                np.pad(f.maxs, (0, max(0, 144000 - len(f.maxs))))[:144000],
                                np.pad(f.spectrogram, ((0, max(0, 144000 - len(f.onset))), (0, 0)))[:144000],
                                np.pad(f.onset, (0, max(0, 144000 - len(f.onset))))[:144000])
    benchmark = []
    for _ in range(3):
        begin = time.perf_counter()
        at.estimate(padded, intervals(lanes["gank03"]["jp"]))
        benchmark.append(time.perf_counter() - begin)
    summary["performance"]["24min_padded_e03_median_seconds"] = round(float(np.median(benchmark)), 6)
    summary["performance"]["24min_pass"] = float(np.median(benchmark)) <= 2.0
    summary["calibration_note"] = "160ms correction retained: user's E01 finals define well timed. Utena corrected targets -600/+15420ms equal raw -440/+15580ms. Raw exploratory +15.6s was approximate; both scales reported."
    summary["rate_prior"] = {"minimum_reliable_span_ms": at.RATE_MIN_SPAN_MS,
                             "family_relative_tolerance": at.RATE_TOLERANCE,
                             "families": at.RATE_FAMILIES, "bounded_loss_penalty": at.RATE_PENALTY}
    summary["limitations"] = "Exact normalized ordered text matches; unmatched lines excluded. Final CN/JP speech groups require equal spans, non-empty JP, styles CN/JP only (no signs/songs). E01 final is imperfect ground truth. Speech edges are an energy heuristic, with music ambiguity and search censoring. >24min, >60s cuts untested. GUI/full suite owned by Claude, not run in A2."
    summary["evidence_dir"] = str(scratch)
    report = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    (scratch / "summary.json").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
