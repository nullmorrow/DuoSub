"""A5 real merged-project evaluation; all generated data stays under TEMP.

Run from any directory with Python 3.12. Outputs one JSON summary. Media and
user projects are read-only inputs; temporary projects are retained as evidence.
No GUI, models, network, or product-code changes are involved.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from subkit import autotime, proc
from subkit.ass import AssDocument
from subkit.core.autotime_doc import Settings, plan
from subkit.core.ops import Editor
from subkit.core.workspace import Project
from subkit.model.serialize import document_to_json
from subkit.model.validate import validate
from subkit.sources.cn import classify

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autotime_real import ass_lines, features, one, pair_text


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def netflix17(folder: Path, scratch: Path) -> tuple[Path, str]:
    with zipfile.ZipFile(folder / "Shoujo Kakumei Utena.zip") as archive:
        candidates = []
        for info in archive.infolist():
            name = info.filename
            if not info.flag_bits & 0x800:
                try:
                    name = name.encode("cp437").decode("gbk")
                except UnicodeError:
                    pass
            if "S01E17" in name and "netflix" in name.lower() and name.lower().endswith(".srt"):
                candidates.append((info, name))
        if len(candidates) != 1:
            raise ValueError(f"expected one Netflix E17: {[n for _, n in candidates]}")
        info, name = candidates[0]
        target = scratch / "netflix-e17.srt"
        target.write_bytes(archive.read(info))
    return target, name


def style_mapping(path: Path) -> tuple[dict, dict]:
    doc = AssDocument.load(path)
    inferred = classify(doc, {"dialogue": ["Default"]})
    cfg = {"dialogue": ["Default"], "top": [], "song_jp": [], "song_cn": []}
    # Same role inference as AddEpisodeDialog; unrecognized ED/IN remain keep.
    for style, role in inferred.style_roles.items():
        if role in ("song_jp", "song_cn"):
            cfg[role].append(style)
    return cfg, inferred.style_roles


def visible_members(doc, group, lane: str) -> list:
    return [m for m in doc.members_of(group, lane=lane)
            if m.role != "comment" and any(r.kind == "text" and r.text.strip() for r in m.runs)]


def paired(doc, groups) -> dict:
    groups = list(groups)
    count = sum(bool(visible_members(doc, g, "original"))
                and bool(visible_members(doc, g, "translation")) for g in groups)
    return {"groups": len(groups), "paired": count,
            "paired_share": count / len(groups) if groups else None}


def validation(doc, sources) -> dict:
    report = validate(doc, sources)
    return {"ok": report.ok, "errors": [asdict(i) for i in report.errors],
            "warnings": sum(i.level == "warning" for i in report.issues)}


def break_pairing(doc, source) -> dict:
    events = sorted((e for e in source.events if e.text.strip()), key=lambda e: e.start)
    gaps = [(b.start - a.end, a.end, b.start) for a, b in zip(events, events[1:])
            if 600000 <= (a.end + b.start) / 2 <= 660000]
    if not gaps:
        return {"error": "no original-caption gap between 600 and 660 seconds"}
    gap, end, start = max(gaps)
    boundary = (end + start) / 2
    records = []
    for g in doc.groups:
        if g.kind != "dialogue":
            continue
        jp = visible_members(doc, g, "original")
        cn = visible_members(doc, g, "translation")
        refs = {r.e for m in jp for r in m.runs if r.src == source.id and r.e is not None}
        if refs:
            original = min(source.events[i].start for i in refs)
            records.append((original, g, cn))
    before = [g for t, g, _ in records if t < boundary]
    after = [g for t, g, _ in records if t >= boundary]
    samples = sorted((r for r in records if r[2]), key=lambda r: r[0])
    left = [r for r in samples if r[0] < boundary][-3:]
    right = [r for r in samples if r[0] >= boundary][:3]
    return {"original_gap_ms": [end, start], "gap_duration_ms": gap,
            "source_boundary_ms": boundary, "before": paired(doc, before), "after": paired(doc, after),
            "local_before_120s": paired(doc, [g for t, g, _ in records if boundary - 120000 <= t < boundary]),
            "local_after_120s": paired(doc, [g for t, g, _ in records if boundary <= t < boundary + 120000]),
            "nearby_pairs": [{"jp_source_start_ms": t, "cn_start_ms": min(m.start for m in cn),
                              "cn_minus_jp_ms": min(m.start for m in cn) - t} for t, _, cn in left + right],
            "denominator": "dialogue groups with visible original source references; source-time split"}


def finished_pairs(doc, finished: Path) -> tuple[dict, dict]:
    final_jp = ass_lines(finished, ("JP",))
    spans = {(s, e) for s, e, text in final_jp if text.strip()}
    targets = {"original": final_jp,
               "translation": [x for x in ass_lines(finished, ("CN",)) if x[:2] in spans]}
    matches, coverage = {}, {}
    for lane, target in targets.items():
        rows, ids = [], []
        for g in doc.groups:
            if g.kind != "dialogue" or not visible_members(doc, g, "original"):
                continue
            for m in visible_members(doc, g, lane):
                if m.role not in ("dialogue", "top", "alt"):
                    continue
                rows.append((m.start, m.end, m.text))
                ids.append(m.id)
        # Attach member IDs without changing the established ordered text matcher.
        keyed = {id(row): mid for row, mid in zip(rows, ids)}
        matches[lane] = [(keyed[id(row)], target_row) for row, target_row in pair_text(rows, target)]
        coverage[lane] = {"eligible_merged_members": len(rows), "eligible_final_lines": len(target),
                          "matched": len(matches[lane]), "unmatched_merged": len(rows) - len(matches[lane]),
                          "unmatched_final": len(target) - len(matches[lane])}
    return matches, coverage


def errors(doc, matches) -> dict:
    result = {}
    for lane, pairs in matches.items():
        if not pairs:
            result[lane] = {"error": "no ordered text matches"}
            continue
        values = np.array([[abs(doc.members[mid].start - target[0]), abs(doc.members[mid].end - target[1])]
                           for mid, target in pairs])
        result[lane] = {"median_abs_start_ms": float(np.median(values[:, 0])),
                        "median_abs_end_ms": float(np.median(values[:, 1])),
                        "start_within_200ms": float(np.mean(values[:, 0] <= 200)),
                        "end_within_200ms": float(np.mean(values[:, 1] <= 200)),
                        "both_within_200ms": float(np.mean(np.all(values <= 200, axis=1)))}
    return result


def evaluate(name: str, video: Path, cn: Path, jp: Path, scratch: Path, finished: Path | None) -> dict:
    begin = time.perf_counter()
    result = {"video": str(video), "translation": str(cn), "original": str(jp)}
    stage = "create/import/merge"
    try:
        cfg, roles = style_mapping(cn)
        project = Project.create(scratch / name, name)
        episode = project.add_episode("S01E001")
        episode.set_video(video)
        episode.add_source(cn, lang="zh-Hant", lane="translation", cn_cfg=cfg)
        source = episode.add_source(jp, lang="ja", lane="original")
        merge_begin = time.perf_counter()
        doc = episode.merge()
        result["project"] = str(project.root)
        result["merge"] = {**paired(doc, doc.groups), "dialogue": paired(doc, [g for g in doc.groups if g.kind == "dialogue"]),
                           "kinds": dict(Counter(g.kind for g in doc.groups)), "style_roles": roles,
                           "settings": episode.settings()["merge"], "seconds": round(time.perf_counter() - merge_begin, 6)}
        result["validation_before"] = validation(doc, episode.sources)
        if name == "utena17":
            result["break_pairing"] = break_pairing(doc, source)
        if finished:
            matches, coverage = finished_pairs(doc, finished)
            result["comparison"] = {"finished": str(finished), "coverage": coverage, "before": errors(doc, matches)}
        before = copy.deepcopy(document_to_json(doc))
        sources_before = copy.deepcopy(episode.sources)
        stage = "decode/features"
        f, measurement = features(video, scratch)
        result["audio"] = measurement
        stage = "plan"
        plan_begin = time.perf_counter()
        preview = plan(doc, episode.sources, f, Settings())
        result["autotime"] = {"settings": asdict(preview.settings), "segments": [asdict(s) for s in preview.segments],
                              "quality": asdict(preview.quality), "windows": dict(Counter(w.confidence for w in preview.mapping.windows)),
                              "changed_groups": preview.group_count, "changed_members": preview.member_count,
                              "protected": preview.protected_count, "low_confidence": preview.low_confidence_count,
                              "plan_seconds": round(time.perf_counter() - plan_begin, 6),
                              "plan_document_unchanged": document_to_json(doc) == before}
        stage = "apply/validate/undo"
        editor = Editor(doc, episode.sources)
        apply_begin = time.perf_counter()
        applied = editor.apply_autotime(preview)
        result["apply"] = {"groups": applied[0], "members": applied[1],
                           "history_steps": len(editor.history.done), "seconds": round(time.perf_counter() - apply_begin, 6)}
        result["validation_after"] = validation(doc, episode.sources)
        if finished:
            comparison = result["comparison"]
            comparison["after"] = errors(doc, matches)
            comparison["median_not_worse"] = {lane: all(comparison["after"][lane][key] <= comparison["before"][lane][key]
                                                        for key in ("median_abs_start_ms", "median_abs_end_ms"))
                                               for lane in matches if matches[lane]}
        undone = editor.history.undo()
        restored = document_to_json(doc)
        result["undo"] = {"returned_transaction": undone is not None, "serialized_exact": restored == before,
                          "different_top_fields": [k for k in before if restored[k] != before[k]],
                          "revision_before": before["revision"], "revision_after_undo": restored["revision"],
                          "content_exact_except_revision": {k: v for k, v in restored.items() if k != "revision"}
                          == {k: v for k, v in before.items() if k != "revision"}}
        result["validation_undo"] = validation(doc, episode.sources)
        result["sources_unchanged"] = episode.sources == sources_before and not episode.verify_sources()
        failures = []
        for label in ("validation_before", "validation_after", "validation_undo"):
            if not result[label]["ok"]:
                failures.append(label)
        if not result["undo"]["serialized_exact"]:
            failures.append("undo_serialized_exact")
        if not result["sources_unchanged"]:
            failures.append("sources_unchanged")
        if finished and not all(result["comparison"]["median_not_worse"].values()):
            failures.append("e01_median_not_worse")
        if applied != (preview.group_count, preview.member_count) or len(editor.history.undone) != bool(preview.changes):
            failures.append("apply_count_or_single_transaction")
        result["status"] = "failed" if failures else "completed"
        result["failed_checks"] = failures
    except (PermissionError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        result.update(status="environment_blocked", stage=stage, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        result.update(status="failed", stage=stage, error=f"{type(exc).__name__}: {exc}")
    result["elapsed_seconds"] = round(time.perf_counter() - begin, 6)
    return result


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    begin = time.perf_counter()
    scratch = Path(tempfile.mkdtemp(prefix="autotime-project-real-", dir=os.environ["TEMP"]))
    utena = Path('./sample-media/少女革命ウテナ')
    gank = Path('./sample-media/巌窟王 (2004)')
    jp, member = netflix17(utena, scratch)
    videos = {"utena17": one(utena, "S02E004*.mkv"), "gank01": one(gank, "S01E001*.mp4"),
              "gank03": one(gank / "Working", "S01E003*.mp4")}
    finished = one(gank / "Subwork/Done", "S01E001*.zh-Hant.ass")
    inputs = {"utena17": (videos["utena17"].with_suffix(".zh-Hant.ass"), jp),
              "gank01": (videos["gank01"].with_suffix(".zh-Hant.ass"), gank / "Subwork/ep1.srt"),
              "gank03": (videos["gank03"].with_suffix(".zh-Hant.ass.bak"), videos["gank03"].with_suffix(".ja.srt"))}
    paths = {finished, utena / "Shoujo Kakumei Utena.zip", *(p for pair in inputs.values() for p in pair)}
    hashes = {p: sha256(p) for p in paths}
    media_stats = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in videos.values()}
    summary = {"environment": {"python": sys.version.split()[0], "numpy": np.__version__, "mode": "host / core APIs / no Qt or test doubles",
                               "ffmpeg": proc.run(["ffmpeg", "-version"], capture_output=True, check=True).stdout.decode("utf-8").splitlines()[0],
                               "audio": "0:a:0 / 16000Hz mono s16le / compute_features; same as PeakJob"},
               "correlation_bias_ms": autotime.CORRELATION_BIAS_MS,
               "evidence_dir": str(scratch), "netflix_zip_member": member, "episodes": {}}
    for name, video in videos.items():
        print(f"Evaluating {name}", file=sys.stderr, flush=True)
        cn, original = inputs[name]
        summary["episodes"][name] = evaluate(name, video, cn, original, scratch, finished if name == "gank01" else None)
    summary["external_subtitle_bytes_unchanged"] = all(sha256(p) == h for p, h in hashes.items())
    summary["media_size_mtime_unchanged"] = all((p.stat().st_size, p.stat().st_mtime_ns) == value for p, value in media_stats.items())
    summary["elapsed_seconds"] = round(time.perf_counter() - begin, 6)
    summary["limitations"] = ("Ordered normalized exact-text matches only; unmatched and many-to-one lines excluded. Final JP style only; CN requires identical span to nonempty JP; name/namec/note/notec/op excluded. Pairing shares measure co-presence, not semantic correctness. Merged JP already follows CN times; a Netflix source jump need not appear in the merged timing map. Finished E01 is imperfect truth. Full suite owned by Claude; human GUI/IME/fullscreen checks not run; >24min and >60s cuts untested. Temporary read-only source copies retained; no cleanup attempted.")
    statuses = {x["status"] for x in summary["episodes"].values()}
    summary["status"] = "failed" if "failed" in statuses else "environment_blocked" if "environment_blocked" in statuses else "completed"
    if not summary["external_subtitle_bytes_unchanged"] or not summary["media_size_mtime_unchanged"]:
        summary["status"] = "failed"
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False), flush=True)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
