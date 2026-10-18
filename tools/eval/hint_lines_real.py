"""Read-only W3b evaluation; JSON and all generated output stay in this run's TEMP.

Decode each selected episode once, using its stored mpv aid and the app's PCM
normalization. References are current working dialogue groups, not source cues.
Experimental detectors run only in this process and never change product code.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from subkit import autotime, proc, vad
from subkit.audio import BAND_EDGES_HZ, HOP_MS, RATE, compute_features, ffmpeg_audio_index
from subkit.model.serialize import load_document


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    padded = np.pad(values, ((width // 2, width // 2),) + ((0, 0),) * (values.ndim - 1), mode="edge")
    sums = np.concatenate((np.zeros_like(padded[:1]), np.cumsum(padded, axis=0, dtype=np.float64)))
    return ((sums[width:] - sums[:-width]) / width).astype(np.float32)


def activity_variants(features):
    baseline = autotime._snap_activity(features)
    yield "baseline", baseline, {}
    spectrum = features.spectrogram.astype(np.float32) / 255
    left, right = BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]
    voice = (left >= 250) & (right <= 3500)
    neighbours = ((left >= 60) & (right <= 250)) | ((left >= 3500) & (right <= 7000))
    outside = spectrum[:, neighbours].mean(axis=1)
    for weight in (0.25, 0.5, 0.75):
        yield f"band_contrast_{weight:g}", np.maximum(baseline - weight * outside, 0), {"neighbour_weight": weight}
    for low, high in ((350, 2500), (500, 3000), (800, 3500)):
        curve = spectrum[:, (left >= low) & (right <= high)].mean(axis=1)
        yield f"voice_band_{low}_{high}", curve, {"band_hz": [low, high]}
    for width in (101, 201):
        residual = np.maximum(spectrum[:, voice] - smooth(spectrum[:, voice], width), 0).mean(axis=1)
        for weight in (0.5, 1.0):
            yield f"steady_residual_{width}_{weight:g}", baseline + weight * residual, {"window_ms": width * HOP_MS, "residual_weight": weight}
    # A 3..8 Hz envelope component is only weak evidence; preserve energy edges.
    frequencies = np.fft.rfftfreq(len(baseline), HOP_MS / 1000)
    transform = np.fft.rfft(baseline)
    transform[(frequencies < 3) | (frequencies > 8)] = 0
    modulation = np.sqrt(smooth(np.fft.irfft(transform, n=len(baseline)).astype(np.float32) ** 2, 21))
    for weight in (0.5, 1.0, 2.0):
        yield f"syllabic_3_8hz_{weight:g}", baseline + weight * modulation, {"modulation_hz": [3, 8], "weight": weight}
    # Suppress narrow persistent tones relative to neighbouring frequency bands.
    frequency_floor = smooth(spectrum.T, 9).T
    tonal = np.maximum(spectrum - frequency_floor, 0)[:, voice].mean(axis=1)
    for weight in (0.5, 1.0):
        yield f"tonal_suppression_{weight:g}", np.maximum(baseline - weight * tonal, 0), {"frequency_window_bands": 9, "weight": weight}
    onset = smooth(features.onset, 11)
    scale = float(np.percentile(onset, 95))
    if scale > 0:
        for weight in (0.02, 0.05):
            yield f"onset_support_{weight:g}", baseline + weight * np.clip(onset / scale, 0, 2), {"onset_weight": weight}


def experimental_segments(activity, start, end, *, threshold=0.3, gap=12):
    """Baseline segmentation with explicit experimental threshold/gap settings."""
    lo = max(0, start // HOP_MS)
    hi = min(len(activity), (end + HOP_MS - 1) // HOP_MS)
    curve = activity[lo:hi]
    if len(curve) < 10:
        return []
    floor, peak = np.percentile(curve, (20, 90))
    if peak < 0.035 or peak - floor < 0.025:
        return []
    active = curve > floor + max(0.018, (peak - floor) * threshold)
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    runs = []
    for a, b in zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()):
        if runs and a - runs[-1][1] <= gap:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return [((a + lo) * HOP_MS if a > 0 else None,
             (b + lo) * HOP_MS if b < len(curve) else None)
            for a, b in runs if b - a >= 10]


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def evaluate(activity, doc, *, vad_segments=None):
    dialogues = [g for g in doc.groups if g.kind == "dialogue"]
    calibration = autotime.calibrate_hint_padding(activity, (doc.group_time(g) for g in dialogues),
                                                 vad_segments=vad_segments)
    lead_in, lead_out = calibration.lead_in_ms, calibration.lead_out_ms
    neighbours = sorted((g for g in doc.groups if g.kind in ("dialogue", "song") and g.members), key=doc.group_time)
    positions = {g.id: i for i, g in enumerate(neighbours)}
    result = {"calibration": asdict(calibration)}
    for pipeline in ("speech_edges", "ui_hints"):
        rows = []
        for group in dialogues:
            current = start, end = doc.group_time(group)
            i = positions[group.id]
            previous = doc.group_time(neighbours[i - 1]) if i else None
            following = doc.group_time(neighbours[i + 1]) if i + 1 < len(neighbours) else None
            if pipeline == "speech_edges":
                a, b = autotime._speech_edges(activity, start, end, 600, lead_in, lead_out,
                                              vad_segments=vad_segments)
            else:
                radius = autotime.HINT_SEARCH_MS
                runs = autotime.speech_segments(activity, start - radius - 200, end + radius + 200,
                                                vad_segments=vad_segments)
                a, b = autotime.hint_speech_edges(activity, start, end, runs=runs,
                                                previous=previous, following=following)
            suggested = None
            if a is not None or b is not None:
                raw = (max(0, a - lead_in) if a is not None else None,
                       b + lead_out if b is not None else None)
                suggested = autotime.clamp_suggested_times(raw, current, previous=previous, following=following)
            effective = tuple(c if s is None else s for c, s in zip(current, suggested)) if suggested else None
            collisions = sum(effective is not None and neighbour is not None
                             and overlaps(effective, neighbour) and not overlaps(current, neighbour)
                             for neighbour in (previous, following))
            rows.append({"group": group.id, "current_ms": current, "suggested_ms": suggested,
                         "start_error_ms": suggested[0] - start if suggested and suggested[0] is not None else None,
                         "end_error_ms": suggested[1] - end if suggested and suggested[1] is not None else None,
                         "new_collisions": collisions})
        covered = [r for r in rows if r["suggested_ms"] is not None]
        scores = score_rows(rows)
        start_good = sum(r["start_error_ms"] is not None and abs(r["start_error_ms"]) <= 200 for r in covered)
        end_good = sum(r["end_error_ms"] is not None and abs(r["end_error_ms"]) <= 200 for r in covered)
        result[pipeline] = {**scores,
                            "start_within_200ms": start_good, "end_within_200ms": end_good,
                            "start_rate": scores["start_within_200ms"],
                            "end_rate": scores["end_within_200ms"],
                            "new_collisions": sum(r["new_collisions"] for r in rows), "rows": rows}
    return result


def episodes():
    soma = ROOT / "Projects" / "食戟のソーマ" / "episodes"
    yield "soma_e01", soma / "S01E001"
    yield "soma_e02", soma / "S01E002"
    candidates = []
    for path in (ROOT / "Projects" / "少女革命ウテナ" / "episodes").glob("*/episode.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        media_name = Path(data["media"]["video"]["path"]).name
        if re.search(r"(?:\b17\b|E0?17\b)", media_name):
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise ValueError(f"expected one Utena E17, found {candidates}")
    yield "utena_e17", candidates[0]


def eligible(report, variant, pipeline):
    for episode in report["episodes"]:
        baseline = episode["variants"]["baseline"][pipeline]
        candidate = episode["variants"][variant][pipeline]
        strict = episode["name"].startswith("soma")
        for metric in ("coverage", "start_rate", "end_rate"):
            worse = candidate[metric] <= baseline[metric] if strict else candidate[metric] < baseline[metric]
            if worse:
                return False
        if candidate["new_collisions"] > baseline["new_collisions"]:
            return False
    return True


def eligible_vad(report, variant):
    """X2 requires improvement on all three episodes, in both timing paths."""
    if len(report["episodes"]) != 3:
        return False
    for episode in report["episodes"]:
        for pipeline in ("speech_edges", "ui_hints"):
            before = episode["variants"]["baseline"][pipeline]
            after = episode["variants"][variant][pipeline]
            if (any(after[key] <= before[key] for key in ("coverage", "start_rate", "end_rate"))
                    or after["new_collisions"] > before["new_collisions"]):
                return False
    return True


def comparison_groups(doc, features, runs):
    """Six reproducible amplitude/duration/gap proxies, requiring human listening."""
    groups = sorted((g for g in doc.groups if g.kind == "dialogue" and g.members), key=doc.group_time)
    envelope = np.maximum(np.abs(features.mins), np.abs(features.maxs))
    data = []
    for i, group in enumerate(groups):
        start, end = doc.group_time(group)
        if end <= start:
            continue
        lo, hi = max(0, start // HOP_MS), min(len(envelope), (end + HOP_MS - 1) // HOP_MS)
        curve = envelope[lo:hi]
        loudness = float(np.sqrt(np.mean(curve ** 2))) if len(curve) else 0
        around = envelope[max(0, lo - 50):min(len(envelope), hi + 50)]
        background = float(np.percentile(around, 20)) if len(around) else 0
        previous_end = doc.group_time(groups[i - 1])[1] if i else -100000
        gap = abs(start - previous_end)
        data.append((group, dict(loud_bgm=background, tight_handoff=-gap, quiet=-loudness,
                                 shouting=loudness, long=end - start, short=start - end)))
    selected, used = [], set()
    for category in ("loud_bgm", "tight_handoff", "quiet", "shouting", "long", "short"):
        candidates = [(g, values) for g, values in data if g.id not in used]
        if not candidates:
            raise ValueError("need six distinct dialogue groups for comparisons")
        group, values = max(candidates, key=lambda item: item[1][category])
        used.add(group.id)
        selected.append((category, group, values[category]))
    return selected


def save_comparisons(folder, name, doc, features, before, after, runs, *,
                     selected=None, after_label="VAD 建议", neighbours=None):
    """Offscreen annotated plots of actual decoded waveform and display spectrum."""
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    # Windows' offscreen plugin has no system fonts. Register a local font in
    # this process only, so evidence labels remain readable without installing.
    font_family = "Sans Serif"
    if not QFontDatabase.families():
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            raise RuntimeError("no readable font for annotated comparison plots")
        font_family = families[0]
    else:
        font_family = "Microsoft YaHei" if "Microsoft YaHei" in QFontDatabase.families() else QFontDatabase.families()[0]
    old_rows = {r["group"]: r for r in before["ui_hints"]["rows"]}
    new_rows = {r["group"]: r for r in after["ui_hints"]["rows"]}
    folder.mkdir(parents=True, exist_ok=True)
    output = []
    for category, group, proxy in (comparison_groups(doc, features, runs) if selected is None else selected):
        original = doc.group_time(group)
        old, new = old_rows[group.id]["suggested_ms"], new_rows[group.id]["suggested_ms"]
        neighbour = neighbours.get(group.id) if neighbours else None
        next_pairs = (doc.group_time(neighbour), old_rows[neighbour.id]["suggested_ms"],
                      new_rows[neighbour.id]["suggested_ms"]) if neighbour else (None,) * 3
        edges = [t for pair in (original, old, new, *next_pairs) for t in (pair or ()) if t is not None]
        left, right = max(0, min(edges) - 1000), max(edges) + 1000
        width, height, x0, x1 = 1400, 750, 70, 1350
        def x(t):
            return x0 + (t - left) * (x1 - x0) / (right - left)
        image = QImage(width, height, QImage.Format.Format_RGB32)
        image.fill(QColor("#111820"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont(font_family, 10))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(35, 30, f"{name} / {category} / {group.id}   （按指标选例，类别需试听确认）")
        text = " | ".join(doc.members[m].text for m in group.members)
        painter.drawText(QRectF(35, 40, 1320, 55), Qt.TextFlag.TextWordWrap, text[:240])
        lo, hi = max(0, left // HOP_MS), min(len(features.mins), (right + HOP_MS - 1) // HOP_MS)
        peak = max(.05, float(np.max(np.maximum(np.abs(features.mins[lo:hi]), np.abs(features.maxs[lo:hi])))))
        painter.setPen(QPen(QColor("#66c896"), 1))
        for pixel in range(x1 - x0):
            a = lo + (hi - lo) * pixel // (x1 - x0)
            b = lo + (hi - lo) * (pixel + 1) // (x1 - x0)
            b = min(hi, max(a + 1, b))
            if b > a:
                painter.drawLine(x0 + pixel, int(215 - features.maxs[a:b].max() / peak * 100),
                                 x0 + pixel, int(215 - features.mins[a:b].min() / peak * 100))
        spectrum = features.spectrogram[lo:hi].T[::-1]
        rgb = np.ascontiguousarray(np.stack((spectrum, spectrum, np.minimum(spectrum.astype(int) * 2, 255)), axis=2).astype(np.uint8))
        spectral_image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        painter.drawImage(QRectF(x0, 335, x1 - x0, 220), spectral_image)
        painter.setPen(QColor("#aaaaaa"))
        painter.drawText(10, 215, "PCM")
        painter.drawText(8, 345, "8 kHz")
        painter.drawText(8, 555, "0 Hz")
        for i, (label, pair, color) in enumerate((("原时间", original, "#58a6ff"),
                                                ("旧建议", old, "#ff9966"),
                                                (after_label, new, "#ffd33d"))):
            painter.setPen(QPen(QColor(color), 2))
            y = 590 + 42 * i
            painter.drawText(35, y, f"{label}: {pair if pair is not None else '无建议'} ms")
            if pair:
                for edge in pair:
                    if edge is None:
                        continue
                    painter.drawLine(int(x(edge)), 105, int(x(edge)), 565)
                if all(t is not None for t in pair):
                    painter.drawLine(int(x(pair[0])), y + 10, int(x(pair[1])), y + 10)
            if next_pairs[i] is not None:
                painter.drawText(700, y, f"下一句: {next_pairs[i]} ms")
                painter.setPen(QPen(QColor(color), 2, Qt.PenStyle.DashLine))
                for edge in next_pairs[i]:
                    if edge is not None:
                        painter.drawLine(int(x(edge)), 105, int(x(edge)), 565)
        painter.setPen(QColor("#cccccc"))
        for tick in np.linspace(left, right, 9):
            painter.drawText(int(x(tick)) - 25, 735, f"{tick / 1000:.2f}s")
        painter.end()
        path = folder / f"{name}-{category}.png"
        if not image.save(str(path), "PNG"):
            raise OSError(f"could not save {path}")
        output.append(dict(category=category, group=group.id, proxy_value=proxy, path=str(path)))
    return output


def distribution(values):
    if not len(values):
        return {"count": 0}
    quantiles = np.percentile(values, (5, 25, 50, 75, 95))
    return {"count": len(values), **dict(zip(("p05_ms", "p25_ms", "median_ms", "p75_ms", "p95_ms"),
                                            map(float, quantiles)))}


def gold_document(path):
    from tools.eval.autotime_real import ass_lines
    from subkit.model.document import Document, Group, Member, Run

    jp, cn = ass_lines(path, ("JP",)), ass_lines(path, ("CN",))
    doc = Document(episode=path.name)
    for i, (start, end, text) in enumerate(sorted(jp)):
        if end <= start:
            continue
        gid, mid = f"g{i}", f"m{i}"
        doc.groups.append(Group(id=gid, kind="dialogue", members=[mid]))
        doc.members[mid] = Member(id=mid, group=gid, lang="ja", lane="original", role="dialogue",
                                  start=start, end=end, runs=[Run(text=text)])
    spans = {doc.group_time(g) for g in doc.groups}
    return doc, {"jp_lines": len(jp), "cn_lines": len(cn),
                 "cn_equal_jp_span": sum((s, e) in spans for s, e, _ in cn),
                 "scored": "nonempty JP, positive duration; CN counted separately, not double-weighted"}


def measured_edges(activity, times, runs=None, *, search_ms=600, run_on=False):
    result = []
    for i, (start, end) in enumerate(times):
        local = autotime.speech_segments(activity, start - search_ms - 200, end + search_ms + 200,
                                        vad_segments=runs)
        result.append(autotime.hint_speech_edges(activity, start, end, search_ms=search_ms, runs=local,
                      previous=times[i - 1] if i else None,
                      following=times[i + 1] if i + 1 < len(times) else None, run_on=run_on))
    return result


def timing_style(times, edges):
    gaps = [b[0] - a[1] for a, b in zip(times, times[1:])]
    lead_in = [a - s for (s, _), (a, _) in zip(times, edges) if a is not None]
    lead_out = [e - b for (_, e), (_, b) in zip(times, edges) if b is not None]
    # Chained ends are intentionally extended, so do not use them as voice padding.
    unlinked_out = [e - b for i, ((_, e), (_, b)) in enumerate(zip(times, edges))
                    if b is not None and (i == len(times) - 1 or times[i + 1][0] != e)]
    return {"lead_in": distribution(lead_in), "lead_out": distribution(lead_out),
            "unlinked_lead_out": distribution(unlinked_out), "gaps": distribution(gaps),
            "touching": sum(g == 0 for g in gaps), "overlapping": sum(g < 0 for g in gaps),
            "gap_bins": {str(limit): sum(0 < g <= limit for g in gaps) for limit in (50, 100, 200, 500, 1000)},
            "lead_in_samples": lead_in, "lead_out_samples": unlinked_out}


def fit_linking(samples):
    """Conservative threshold; exact touching is the independent target label."""
    candidates = []
    for threshold in (0, 50, 100, 150, 200, 250, 300, 400, 500):
        chosen = [touching for gap, touching in samples if 0 <= gap <= threshold]
        precision = sum(chosen) / len(chosen) if chosen else 0
        if len(chosen) >= 20 and precision >= .8:
            candidates.append((sum(chosen), -threshold, precision, len(chosen)))
    if not candidates:
        return {"threshold_ms": None, "supported": False, "samples": len(samples)}
    hits, neg_threshold, precision, count = max(candidates)
    positives = sum(touching for _, touching in samples)
    return {"threshold_ms": -neg_threshold, "supported": True, "precision": precision,
            "recall": hits / positives if positives else 0, "chosen": count, "samples": len(samples)}


def cross_episode_style(episodes, excluded, variant):
    training = [e for e in episodes if e["name"] != excluded]
    onsets = [v for e in training for v in e["styles"][variant]["lead_in_samples"]]
    offsets = [v for e in training for v in e["styles"][variant]["lead_out_samples"]]
    lead_in = int(round(np.clip(np.median(onsets), 0, 600))) if onsets else autotime.SNAP_LEAD_IN_MS
    lead_out = int(round(np.clip(np.median(offsets), -200, 600))) if offsets else autotime.SNAP_LEAD_OUT_MS
    links = []
    for e in training:
        for i, (a, b) in enumerate(zip(e["times"], e["times"][1:])):
            onset, offset = e["edges"][variant][i + 1][0], e["edges"][variant][i][1]
            if a[1] <= b[0] and onset is not None and offset is not None:
                links.append((onset - lead_in - offset - lead_out, a[1] == b[0]))
    return {"lead_in_ms": lead_in, "lead_out_ms": lead_out, "linking": fit_linking(links),
            "training_episodes": [e["name"] for e in training], "excluded_episode": excluded}


def score_rows(rows):
    covered = [r for r in rows if r["suggested_ms"] is not None]
    result = {"groups": len(rows), "suggestions": len(covered),
              "coverage": len(covered) / len(rows) if rows else 0,
              "new_collisions": sum(r["new_collisions"] for r in rows), "rows": rows}
    for side in ("start", "end"):
        errors = [abs(r[side + "_error_ms"]) for r in covered if r[side + "_error_ms"] is not None]
        result[side + "_suggestions"] = len(errors)
        result[side + "_coverage"] = len(errors) / len(rows) if rows else 0
        result["median_abs_" + side + "_ms"] = float(np.median(errors)) if errors else None
        for tolerance in (100, 200):
            good = sum(e <= tolerance for e in errors)
            result[f"{side}_within_{tolerance}ms"] = good / len(errors) if errors else 0
            result[f"{side}_all_within_{tolerance}ms"] = good / len(rows) if rows else 0
    return result


def evaluate_gold(times, ids, edges, style, *, linking=False, clamp=True, one_sided=False):
    proposed = [(max(0, a - style["lead_in_ms"]) if a is not None else None,
                 b + style["lead_out_ms"] if b is not None else None)
                if (a is not None or b is not None) and (one_sided or a is not None and b is not None)
                else None for a, b in edges]
    threshold = style["linking"]["threshold_ms"] if linking else None
    rows = []
    for i, (gid, current, suggested) in enumerate(zip(ids, times, proposed)):
        previous, following = times[i - 1] if i else None, times[i + 1] if i + 1 < len(times) else None
        if (suggested is not None and threshold is not None and following is not None
                and suggested[1] is not None and proposed[i + 1] is not None
                and proposed[i + 1][0] is not None and not overlaps(current, following)):
            gap = proposed[i + 1][0] - suggested[1]
            if 0 <= gap <= threshold:
                suggested = (suggested[0], following[0] if clamp else proposed[i + 1][0])
        if suggested is not None:
            if ((current[1] if suggested[1] is None else suggested[1]) <=
                    (current[0] if suggested[0] is None else suggested[0])):
                suggested = None
            elif clamp:
                suggested = autotime.clamp_suggested_times(suggested, current, previous=previous, following=following)
        effective = tuple(c if s is None else s for c, s in zip(current, suggested)) if suggested else None
        collisions = sum(effective is not None and neighbour is not None and overlaps(effective, neighbour)
                         and not overlaps(current, neighbour) for neighbour in (previous, following))
        rows.append({"group": gid, "current_ms": current, "suggested_ms": suggested,
                     "start_error_ms": suggested[0] - current[0] if suggested and suggested[0] is not None else None,
                     "end_error_ms": suggested[1] - current[1] if suggested and suggested[1] is not None else None,
                     "new_collisions": collisions})
    return score_rows(rows)


def gold_gate(episodes, variant):
    """Coverage-aware gate, independent of the detector used during import."""
    for episode in episodes:
        before, after = episode["variants"]["energy"]["ui_hints"], episode["variants"][variant]["ui_hints"]
        if after["new_collisions"] > before["new_collisions"] or after["coverage"] < before["coverage"] - .02:
            return False
        for side in ("start", "end"):
            if (after[f"{side}_all_within_200ms"] < before[f"{side}_all_within_200ms"] - .02
                    or after[f"median_abs_{side}_ms"] is None
                    or after[f"median_abs_{side}_ms"] > before[f"median_abs_{side}_ms"]):
                return False
    pooled = {v: score_rows([r for e in episodes for r in e["variants"][v]["ui_hints"]["rows"]])
              for v in ("energy", variant)}
    return all(pooled[variant][f"{side}_all_within_200ms"] >= pooled["energy"][f"{side}_all_within_200ms"] + .05
               for side in ("start", "end"))


def habit_gold_inputs():
    utena = Path('./sample-media/少女革命ウテナ')
    paths = sorted((utena / "Done/Publish Candidate").glob("*.zh-Hant.ass"))
    paths = [p for p in paths if re.match(r"S01E0(?:06|07|08|09|10|11|12|13)|S02E00[12]", p.name)]
    overlord = Path('./sample-media/オーバーロード')
    other = sorted((overlord / "S01Done/Publish").glob("*.zh-Hant.ass"))[:6]
    if min(len(paths), len(other)) < 6:
        raise ValueError("need at least six hand-timed episodes per show")
    for show, selected, video_dir in (("utena", paths, utena / "origin"),
                                      ("overlord", other, overlord / "Season 01 [BDRip][VCB-Studio]")):
        for path in selected:
            suffix = path.name.removesuffix(".zh-Hant.ass") + ".mkv"
            matches = [v for v in video_dir.iterdir() if v.name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"expected one video ending {suffix}: {matches}")
            yield show, path, matches[0]


def pooled_habit(episodes):
    onsets = [v for e in episodes for v in e["styles"]["energy"]["lead_in_samples"]]
    offsets = [v for e in episodes for v in e["styles"]["energy"]["lead_out_samples"]]
    return dict(lead_in_ms=int(round(np.clip(np.median(onsets), 0, 600))),
                lead_out_ms=int(round(np.clip(np.median(offsets), -200, 600))),
                onset_samples=len(onsets), offset_samples=len(offsets),
                linking={"threshold_ms": None}, training_episodes=[e["name"] for e in episodes])


def run_habit_gold(output, scratch, *, x10=False):
    """X2b energy-only, two-show evaluation; never writes beside the inputs."""
    from tools.eval.autotime_real import features as read_features
    started = time.perf_counter()
    report = {"environment": {"python": sys.version.split()[0], "numpy": np.__version__,
              "ffmpeg": proc.run(["ffmpeg", "-version"], capture_output=True, check=True).stdout.decode("utf-8").splitlines()[0],
              "audio": "0:a:0, 16kHz mono /32768; Windows host CPU; energy detector"},
              "reference": "hand-timed Utena + Overlord; exact nonempty JP dialogue style; CN counted separately",
              "limitations": "gold times seed +/-600ms searches and neighbour clamp; detector-relative habit, not annotated speech edges; same-episode calibration and initial-habit scores are fitted; cross-show scores train on the other show only; no UI/listening/size-limit validation",
              "episodes": []}
    for show, path, video in habit_gold_inputs():
        ass_hash, media_stat = hashlib.sha256(path.read_bytes()).hexdigest(), video.stat()
        streams = json.loads(proc.run([proc.tool("ffprobe"), "-v", "error", "-select_streams", "a",
                                      "-show_streams", "-of", "json", str(video)],
                                      capture_output=True, check=True, timeout=30).stdout)["streams"]
        doc, counts = gold_document(path)
        features, decode = read_features(video, scratch)
        activity = autotime._snap_activity(features)
        times = [doc.group_time(g) for g in doc.groups]
        edges = measured_edges(activity, times)
        calibration = autotime.calibrate_hint_padding(activity, times)
        entry = dict(show=show, name=show + "_" + path.stem, ass=str(path), video=str(video),
                     input_counts=counts, audio_stream="0:a:0", audio_streams=[dict(index=s["index"],
                     codec=s["codec_name"], channels=s.get("channels")) for s in streams],
                     decode=decode, times=times, edges={"energy": edges}, styles={"energy": timing_style(times, edges)},
                     episode_calibration=asdict(calibration), ass_sha256=ass_hash,
                     ass_unchanged=ass_hash == hashlib.sha256(path.read_bytes()).hexdigest(),
                     media_metadata_unchanged=(media_stat.st_size, media_stat.st_mtime_ns) ==
                                             (video.stat().st_size, video.stat().st_mtime_ns), variants={})
        if x10:
            entry["edges"]["run_on_600"] = measured_edges(activity, times, run_on=True)
            entry["edges"]["run_on_800"] = measured_edges(activity, times, search_ms=800, run_on=True)
        report["episodes"].append(entry)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{show}: {path.name}: {len(times)} JP; streams={entry['audio_streams']}; {decode}", flush=True)
    report["per_show_habit"] = {show: pooled_habit([e for e in report["episodes"] if e["show"] == show])
                                for show in ("utena", "overlord")}
    report["initial_habit"] = pooled_habit(report["episodes"])
    for episode in report["episodes"]:
        calibration = episode["episode_calibration"]
        modes = {"cross_show_habit": pooled_habit([e for e in report["episodes"] if e["show"] != episode["show"]]),
                 "cross_episode_habit": cross_episode_style(report["episodes"], episode["name"], "energy"),
                 "initial_habit": report["initial_habit"],
                 "per_episode": dict(lead_in_ms=calibration["lead_in_ms"], lead_out_ms=calibration["lead_out_ms"], linking={"threshold_ms": None}),
                 "old_270_0": dict(lead_in_ms=270, lead_out_ms=0, linking={"threshold_ms": None})}
        for mode, style in modes.items():
            episode["variants"][mode] = {"calibration": style, "ui_hints": evaluate_gold(
                episode["times"], [str(i) for i in range(len(episode["times"]))], episode["edges"]["energy"], style)}
        if x10:
            style = modes["cross_episode_habit"]
            ids = [f"g{i}" for i in range(len(episode["times"]))]
            for mode, edge_key, partial in (("before_600", "energy", False),
                                            ("one_sided_600", "energy", True),
                                            ("run_on_600", "run_on_600", True),
                                            ("run_on_800", "run_on_800", True)):
                episode["variants"][mode] = {"calibration": style, "ui_hints": evaluate_gold(
                    episode["times"], ids, episode["edges"][edge_key], style, one_sided=partial)}
            modes.update({mode: style for mode in ("before_600", "one_sided_600", "run_on_600", "run_on_800")})
    report["pooled"] = {show: {mode: {k: v for k, v in score_rows([r for e in report["episodes"]
                         if show == "all" or e["show"] == show
                         for r in e["variants"][mode]["ui_hints"]["rows"]]).items() if k != "rows"}
                         for mode in modes} for show in ("utena", "overlord", "all")}
    if x10:
        report["x10_calibration"] = "all candidates share leave-one-episode-out medians of observed 600ms edges; synthesized run-on valleys never train padding"
        report["x10_gate"] = "pooled independent per-side coverage gain >=2pp; conditional 100/200ms accuracy loss <=2pp; zero new collisions; 800 retained only if more accurate hits than 600"
        before = report["pooled"]["all"]["before_600"]
        report["x10_eligible"] = [mode for mode in ("one_sided_600", "run_on_600", "run_on_800")
            if x10_gate(before, report["pooled"]["all"][mode])]
        scores = report["pooled"]["all"]
        better_800 = ("run_on_800" in report["x10_eligible"]
                      and all(scores["run_on_800"][f"{side}_all_within_200ms"] >
                              scores["run_on_600"][f"{side}_all_within_200ms"]
                              and scores["run_on_800"][f"median_abs_{side}_ms"] <=
                              scores["run_on_600"][f"median_abs_{side}_ms"] for side in ("start", "end")))
        report["x10_selected"] = "run_on_800" if better_800 else next(
            (v for v in ("run_on_600", "one_sided_600") if v in report["x10_eligible"]), "before_600")
        report["x10_search_decision"] = "800ms" if better_800 else "retain 600ms; 800ms does not improve both sides' accurate coverage and medians"
        plots = []
        for episode in report["episodes"]:
            if len(plots) >= 6:
                break
            changed = [i for i in range(len(episode["times"]) - 1)
                       if episode["times"][i][1] <= episode["times"][i + 1][0]
                       and (episode["edges"]["energy"][i][1] != episode["edges"]["run_on_600"][i][1]
                            or episode["edges"]["energy"][i + 1][0] != episode["edges"]["run_on_600"][i + 1][0])]
            if not changed:
                continue
            doc, _ = gold_document(Path(episode["ass"]))
            features, _ = read_features(Path(episode["video"]), scratch)
            selected = [(f"run-on-{i}", doc.groups[i], 0) for i in changed[:6 - len(plots)]]
            neighbours = {doc.groups[i].id: doc.groups[i + 1] for i in changed}
            plots.extend(save_comparisons(scratch / "x10-run-on-comparisons", episode["name"], doc, features,
                         episode["variants"]["before_600"], episode["variants"]["run_on_600"], None,
                         selected=selected, neighbours=neighbours, after_label="X10 建议"))
        report["x10_comparisons"] = plots
    report["elapsed_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "per_show_habit": report["per_show_habit"],
                      "initial_habit": report["initial_habit"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


def x10_gate(before, after):
    return after["new_collisions"] <= before["new_collisions"] and all(
        after[f"{side}_coverage"] >= before[f"{side}_coverage"] + .02
        and all(after[f"{side}_within_{ms}ms"] >= before[f"{side}_within_{ms}ms"] - .02
                for ms in (100, 200)) for side in ("start", "end"))


def run_gold(output, scratch, fresh=False):
    from subkit.audio import AudioFeatures
    from importlib import metadata as packages

    started = time.perf_counter()
    folder = Path('./sample-media/少女革命ウテナ')
    paths = sorted((folder / "Done/Publish Candidate").glob("*.zh-Hant.ass"))
    paths = [p for p in paths if re.match(r"S01E0(?:06|07|08|09|10|11|12|13)|S02E00[12]", p.name)]
    if len(paths) < 6:
        raise ValueError("need at least six independently hand-timed episodes")
    settings = [(f"silero_{threshold:g}_{join}_{'refined' if refined else 'raw'}",
                 dict(start_threshold=threshold, end_threshold=threshold - .15, join_ms=join), refined)
                for threshold in (.35, .5, .65) for join in (96, 128, 192) for refined in (False, True)]
    report = {"environment": {"python": sys.version.split()[0], "numpy": np.__version__,
              "silero_vad": packages.version("silero-vad"), "onnxruntime": packages.version("onnxruntime"),
              "ffmpeg": proc.run(["ffmpeg", "-version"], capture_output=True, check=True).stdout.decode("utf-8").splitlines()[0],
              "audio": "0:a:0 (first AAC track), 16kHz mono /32768; Windows host CPU; Qt offscreen plots"},
              "reference": "user hand-timed Publish Candidate ASS; JP only, exact styles; CN separately counted",
              "calibration": "leave-one-episode-out pooled medians; chained ends excluded from lead-out training",
              "linking_fit": "predicted subtitle gap <= threshold; >=20 training pairs, >=80% exact-touch precision",
              "gate": "each episode: coverage/overall <=200ms no >2pp loss, both medians no worse, no collision increase; pooled start AND end overall <=200ms gain >=5pp",
              "limitations": "nearby gold times seed +/-600ms edge search; clamped metrics use gold neighbour times; unclamped control reported; detector-relative padding is not annotated speech truth; proxy categories need listening; no GUI, >24min or size-limit validation",
              "episodes": [], "secondary": []}
    work = []
    model, version = vad.packaged_model()
    report["model"] = {"path": str(model), "version": version, "provider": "CPUExecutionProvider"}
    for path in paths:
        name = path.name.split(" ", 1)[0]
        video = folder / "origin" / (path.name.removesuffix(".zh-Hant.ass") + ".mkv")
        before_hash, before_stat = hashlib.sha256(path.read_bytes()).hexdigest(), video.stat()
        doc, counts = gold_document(path)
        times = [doc.group_time(g) for g in doc.groups]
        command = [proc.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video),
                   "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"]
        begin = time.perf_counter()
        decoded = proc.run(command, capture_output=True, check=True, timeout=120)
        samples = np.frombuffer(decoded.stdout, "<i2").astype(np.float32) / 32768
        features = compute_features(samples)
        decode_seconds = time.perf_counter() - begin
        activity = autotime._snap_activity(features)
        cache = vad.cache_file(video, 1, scratch / "gold-vad-cache", version)
        begin = time.perf_counter()
        voice = vad.analyze(samples, cache_path=None if fresh else cache)
        vad_seconds = time.perf_counter() - begin
        edges, runs = {"energy": measured_edges(activity, times)}, {}
        for variant, parameters, refined in settings:
            raw = vad.segments(voice, **parameters)
            runs[variant] = vad.refine_edges(raw, features) if refined else raw
            edges[variant] = measured_edges(activity, times, runs[variant])
        feature_path = scratch / (name + "-gold-features.npz")
        with feature_path.open("wb") as stream:
            np.savez_compressed(stream, mins=features.mins, maxs=features.maxs,
                                spectrogram=features.spectrogram, onset=features.onset)
        entry = {"name": name, "ass": str(path), "video": str(video), "audio_stream": "0:a:0",
                 "ass_sha256": before_hash, "input_counts": counts, "duration_seconds": len(samples) / RATE,
                 "command": command, "ffmpeg_stderr": decoded.stderr.decode("utf-8", errors="replace"),
                 "decode_features_seconds": decode_seconds, "vad_seconds": vad_seconds,
                 "vad_under_30s": vad_seconds < 30, "styles": {v: timing_style(times, pairs) for v, pairs in edges.items()},
                 "times": times, "edges": edges, "variants": {},
                 "ass_unchanged": before_hash == hashlib.sha256(path.read_bytes()).hexdigest(),
                 "media_metadata_unchanged": (before_stat.st_size, before_stat.st_mtime_ns) == (video.stat().st_size, video.stat().st_mtime_ns)}
        entry["current_energy_same_episode"] = evaluate(activity, doc)
        report["episodes"].append(entry)
        work.append((doc, feature_path, runs))
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(times)} JP lines; VAD {vad_seconds:.3f}s; decode/features {decode_seconds:.3f}s", flush=True)
    variants = ["energy"] + [s[0] for s in settings]
    for episode in report["episodes"]:
        ids = [g.id for g in work[report["episodes"].index(episode)][0].groups]
        for variant in variants:
            style = cross_episode_style(report["episodes"], episode["name"], variant)
            result = {"calibration": style}
            for link in (False, True):
                result["linked" if link else "ui_hints"] = evaluate_gold(episode["times"], ids, episode["edges"][variant], style, linking=link)
                result["unclamped_linked" if link else "unclamped"] = evaluate_gold(episode["times"], ids, episode["edges"][variant], style, linking=link, clamp=False)
            episode["variants"][variant] = result
    report["pooled"] = {v: {pipeline: {k: value for k, value in score_rows(
                          [r for e in report["episodes"] for r in e["variants"][v][pipeline]["rows"]]).items() if k != "rows"}
                          for pipeline in ("ui_hints", "linked", "unclamped", "unclamped_linked")} for v in variants}
    report["eligible"] = [v for v in variants[1:] if gold_gate(report["episodes"], v)]
    def rank(v):
        m = report["pooled"][v]["ui_hints"]
        return (min(m["start_all_within_200ms"], m["end_all_within_200ms"]),
                m["start_all_within_200ms"] + m["end_all_within_200ms"])
    best = max(variants[1:], key=rank)
    report["best_vad"] = best
    report["selected"] = max(report["eligible"], key=rank) if report["eligible"] else "energy"
    for episode, (doc, feature_path, runs) in zip(report["episodes"], work):
        with np.load(feature_path, allow_pickle=False) as values:
            features = AudioFeatures(*(values[k].copy() for k in ("mins", "maxs", "spectrogram", "onset")))
        episode["screenshots"] = save_comparisons(scratch / "gold-comparisons", episode["name"], doc, features,
                                                   episode["variants"]["energy"], episode["variants"][best], runs[best])
    # Soma's imported times remain a secondary diagnostic, excluded from selection.
    for name, folder in list(episodes())[:2]:
        metadata = json.loads((folder / "episode.json").read_text(encoding="utf-8"))
        document_path = folder / "document.json"
        original_hash = hashlib.sha256(document_path.read_bytes()).hexdigest()
        doc = load_document(document_path)
        video = Path(metadata["media"]["video"]["path"])
        decoded = proc.run([proc.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video),
                            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"],
                           capture_output=True, check=True, timeout=120)
        samples = np.frombuffer(decoded.stdout, "<i2").astype(np.float32) / 32768
        features = compute_features(samples)
        activity = autotime._snap_activity(features)
        begin = time.perf_counter()
        voice = vad.analyze(samples, cache_path=vad.cache_file(video, 1, scratch / "gold-vad-cache", version))
        params, refined = next((p, r) for v, p, r in settings if v == best)
        runs = vad.segments(voice, **params)
        runs = vad.refine_edges(runs, features) if refined else runs
        report["secondary"].append({"name": name, "reference": "old imported current times; excluded from gate",
                                    "audio_stream": "0:a:0", "vad_seconds": time.perf_counter() - begin,
                                    "energy": evaluate(activity, doc), "vad": evaluate(activity, doc, vad_segments=runs),
                                    "document_unchanged": original_hash == hashlib.sha256(document_path.read_bytes()).hexdigest()})
        print(f"{name}: secondary sanity check complete", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "best_vad": best, "eligible": report["eligible"],
                      "selected": report["selected"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="JSON path inside this run's TEMP")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--silero", action="store_true", help="X2: cached CPU VAD comparison and six annotated PNGs per episode")
    parser.add_argument("--fresh-vad", action="store_true", help="measure ONNX inference, ignoring the VAD cache")
    parser.add_argument("--habit", action="store_true", help="X2b: --gold energy-only Utena + six Overlord episodes")
    parser.add_argument("--x10", action="store_true", help="with --gold --habit: one-sided/run-on 600/800ms comparison")
    parser.add_argument("--gold", action="store_true", help="independent hand-timed Utena JP reference; cross-episode calibration")
    parser.add_argument("--baseline-join-ms", type=int, choices=(80, 120),
                        help="explicit historical baseline; omit to evaluate the installed detector")
    args = parser.parse_args()
    scratch = Path(os.environ["TEMP"]).resolve()
    output = (args.output or scratch / "hint-lines-real.json").resolve()
    if not output.is_relative_to(scratch) or output.is_relative_to(ROOT / "Projects"):
        parser.error("output must stay inside this run's TEMP and outside Projects")
    if args.habit and not args.gold:
        parser.error("--habit requires --gold")
    if args.x10 and not (args.gold and args.habit):
        parser.error("--x10 requires --gold --habit")
    if args.gold:
        if args.habit:
            run_habit_gold(output, scratch, x10=args.x10)
            return
        run_gold(output, scratch, args.fresh_vad)
        return
    report = {"environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                                "ffmpeg": proc.run(["ffmpeg", "-version"], capture_output=True, check=True).stdout.decode("utf-8").splitlines()[0]},
              "reference": "current dialogue group times; neighbours include dialogue and song",
              "rate_denominator": "groups with a nonempty suggestion; <=200 ms is inclusive",
              "collisions": "suggestion against adjacent groups' current times, excluding pre-existing overlaps",
              "limitations": "same subtitles calibrate and score; heuristic agreement, not listening or speech recognition; no size-limit/UI validation",
              "baseline_join_ms": args.baseline_join_ms, "episodes": []}
    started = time.perf_counter()
    for name, folder in episodes():
        tracked = [folder / "episode.json", folder / "document.json"]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
        metadata = json.loads(tracked[0].read_text(encoding="utf-8"))
        doc = load_document(tracked[1])
        media = Path(metadata["media"]["video"]["path"])
        media_before = media.stat()
        index = ffmpeg_audio_index(metadata["media"].get("audio_track"))
        command = [proc.tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(media),
                   "-map", f"0:a:{index}", "-vn", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"]
        decode_started = time.perf_counter()
        decoded = proc.run(command, capture_output=True, check=True, timeout=120)
        pcm = np.frombuffer(decoded.stdout, dtype="<i2")
        features = compute_features(pcm.astype(np.float32) / 32768.0)
        episode = {"name": name, "folder": str(folder), "media": str(media), "stored_aid": metadata["media"].get("audio_track"),
                   "ffmpeg_index": index, "command": command, "ffmpeg_stderr": decoded.stderr.decode("utf-8", errors="replace"),
                   "duration_seconds": len(pcm) / RATE, "decode_features_seconds": time.perf_counter() - decode_started,
                   "input_sha256": hashes, "variants": {}}
        episode["installed_detector"] = evaluate(autotime._snap_activity(features), doc)
        for variant, activity, parameters in activity_variants(features):
            if (args.baseline_only or args.silero) and variant != "baseline":
                break
            if args.baseline_join_ms is None:
                evaluated = evaluate(activity, doc)
            else:
                def baseline_segmenter(a, s, e, **_):
                    return experimental_segments(a, s, e, gap=args.baseline_join_ms // HOP_MS)
                with patch.object(autotime, "speech_segments", baseline_segmenter):
                    evaluated = evaluate(activity, doc)
            evaluated["parameters"] = parameters
            episode["variants"][variant] = evaluated
        if args.silero:
            baseline = autotime._snap_activity(features)
            model_path, model_version = vad.packaged_model()
            cache_path = vad.cache_file(media, metadata["media"].get("audio_track"), scratch / "vad-cache", model_version)
            cache_hit = not args.fresh_vad and vad.load_cache(cache_path, model_version) is not None
            vad_started = time.perf_counter()
            voice = vad.analyze(pcm.astype(np.float32) / 32768.0,
                                cache_path=None if args.fresh_vad else cache_path)
            episode["vad_seconds"] = time.perf_counter() - vad_started
            episode["vad_cache_hit"] = cache_hit
            episode["vad_model"] = dict(path=str(model_path), version=model_version, provider="CPUExecutionProvider")
            candidate_runs = {}
            for join_ms in (96, 128):
                raw = vad.segments(voice, join_ms=join_ms)
                for refined in (False, True):
                    variant = f"silero_gap{join_ms}" + ("_refined" if refined else "_raw")
                    candidate_runs[variant] = vad.refine_edges(raw, features) if refined else raw
                    evaluated = evaluate(baseline, doc, vad_segments=candidate_runs[variant])
                    evaluated["parameters"] = dict(start_threshold=.5, end_threshold=.35, join_ms=join_ms,
                                                    min_speech_ms=100, refinement_ms=150 if refined else 0)
                    episode["variants"][variant] = evaluated
            compared = "silero_gap128_refined"
            episode["screenshot_variant"] = compared
            episode["screenshots"] = save_comparisons(scratch / "vad-comparisons", name, doc, features,
                                                      episode["variants"]["baseline"], episode["variants"][compared],
                                                      candidate_runs[compared])
            episode["selection_limitations"] = "BGM/quiet/shouting use amplitude proxies, hand-off uses current time gap; human listening must confirm categories"
            print(f"{name}: CPU VAD {episode['vad_seconds']:.3f}s, cache_hit={cache_hit}", flush=True)
        elif not args.baseline_only:
            baseline = autotime._snap_activity(features)
            for threshold, gap in ((0.2, 12), (0.25, 12), (0.35, 12), (0.4, 12), (0.3, 8), (0.3, 16)):
                def segmenter(a, s, e, threshold=threshold, gap=gap, **_):
                    return experimental_segments(a, s, e, threshold=threshold, gap=gap)
                with patch.object(autotime, "speech_segments", segmenter):
                    evaluated = evaluate(baseline, doc)
                evaluated["parameters"] = {"threshold_fraction": threshold, "join_gap_ms": gap * HOP_MS}
                episode["variants"][f"segments_{threshold:g}_{gap}"] = evaluated
        episode["inputs_unchanged"] = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in tracked)
        media_after = media.stat()
        episode["media_metadata_unchanged"] = (media_before.st_size, media_before.st_mtime_ns) == (media_after.st_size, media_after.st_mtime_ns)
        report["episodes"].append(episode)
        # Retain partial measurements if a later episode cannot be evaluated.
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{name}: {len(episode['variants'])} variants, decode/features {episode['decode_features_seconds']:.3f}s", flush=True)
    variants = list(report["episodes"][0]["variants"])[1:]
    report["eligible_speech_edges"] = [v for v in variants if eligible(report, v, "speech_edges")]
    report["eligible_ui_hints"] = [v for v in variants if eligible(report, v, "ui_hints")]
    report["eligible_both_pipelines"] = sorted(set(report["eligible_speech_edges"]) & set(report["eligible_ui_hints"]))
    if args.silero:
        report["x2_gate"] = "strictly higher coverage/start_rate/end_rate on ALL three episodes and both paths; no new collision increase"
        report["x2_eligible"] = [v for v in variants if eligible_vad(report, v)]
    report["elapsed_seconds"] = time.perf_counter() - started
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "eligible": report["eligible_both_pipelines"], "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stderr.decode("utf-8", errors="replace") if error.stderr else str(error), file=sys.stderr)
        raise
