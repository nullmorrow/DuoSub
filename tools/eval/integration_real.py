"""Read-only real integration evaluation; generated files only in this run's TEMP.

Run with Python 3.12. Prints JSON to stdout and progress to stderr. The new
project is removed (including its read-only source copies); exports and frames
remain in TEMP as evidence. Never edits media, user projects or product code.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from subkit import proc
from subkit.ass import AssDocument
from subkit.core import bangumi, karaoke, self_check
from subkit.core.autotime_doc import Settings, plan
from subkit.core.display import DisplayRules
from subkit.core.name_finder import find_names
from subkit.core.ops import Editor
from subkit.core.readings import annotate_member
from subkit.core.review import write_packet
from subkit.core.workspace import Project
from subkit.model.serialize import document_to_json
from subkit.sources.cn import classify
from autotime_real import features

MEDIA = Path('./sample-media/S01E002 - [YYDM-11FANS][GANKUTSUOU][02][BDRIP][720P][X264-10bit_AAC][C6128355].mp4')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate():
    started = time.perf_counter()
    temp = Path(os.environ["TEMP"]).resolve()
    scratch = Path(tempfile.mkdtemp(prefix="integration-real-", dir=temp)).resolve()
    assert scratch.is_relative_to(temp)
    result = {"evidence_dir": str(scratch), "steps": []}

    def step(name, action):
        begin = time.perf_counter()
        print(name, file=sys.stderr, flush=True)
        row = {"step": name, "status": "ok"}
        try:
            row["numbers"] = action()
        except Exception as exc:
            blocked = isinstance(exc, (PermissionError, FileNotFoundError, subprocess.TimeoutExpired))
            if isinstance(exc, bangumi.BangumiError):
                blocked = True  # retain the original live API error for review
            row.update(status="environment_blocked" if blocked else "failed",
                       error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        row["seconds"] = round(time.perf_counter() - begin, 3)
        result["steps"].append(row)
        print(json.dumps(row, ensure_ascii=False), file=sys.stderr, flush=True)
        return row

    project = episode = editor = None
    original = MEDIA.with_suffix(".srt")
    translation = MEDIA.with_suffix(".zh-Hant.ass.bak")
    try:
        hashes = {str(p): digest(p) for p in (original, translation)}
        media_stat = (MEDIA.stat().st_size, MEDIA.stat().st_mtime_ns)
        result["environment"] = {"python": sys.version, "mode": "Windows host; core APIs; no Qt/test doubles",
            "ffmpeg": proc.run(["ffmpeg", "-version"], capture_output=True, check=True).stdout.decode("utf-8").splitlines()[0]}

        def create():
            nonlocal project, episode
            copied = scratch / "translation.ass"
            shutil.copyfile(translation, copied)
            # Explicitly requested OP role; signs use the add-episode keep default.
            cfg = {"dialogue": ["Default"], "top": [], "song_jp": [], "song_cn": ["op"]}
            inferred = classify(AssDocument.load(copied), cfg)
            project = Project.create(scratch / "project", "Gankutsuou E02 integration")
            episode = project.add_episode("S01E002", number=2)
            episode.set_video(MEDIA)
            tr = episode.add_source(copied, lang="zh-Hant", lane="translation", cn_cfg=cfg)
            jp = episode.add_source(original, lang="ja", lane="original")
            return {"source_events": {"translation": len(tr.events), "original": len(jp.events)},
                    "style_roles": inferred.style_roles, "notes": inferred.notes,
                    "source_role_counts": dict(Counter(tr.roles.values())), "project": str(project.root)}
        if step("create/import", create)["status"] != "ok":
            return result

        def merge():
            nonlocal editor
            doc = episode.merge()
            editor = Editor(doc, episode.sources, character_book=project.character_book(),
                            reading_engine_factory=episode.reading_engine,
                            pinyin_words_factory=lambda lang: project.pinyin_words(lang) if project.pinyin_mode != "off" else None)
            return {"groups": len(doc.groups), "members": len(doc.members), "kinds": dict(Counter(g.kind for g in doc.groups)),
                    "paired": sum(bool(doc.members_of(g, lane="original")) and bool(doc.members_of(g, lane="translation")) for g in doc.groups),
                    "annotations": len(doc.annotations), "settings": episode.settings()["merge"], "log": episode.data["merge"]["log"]}
        if step("merge defaults", merge)["status"] != "ok":
            return result
        doc = episode.document

        def source_style(m):
            return next((episode.sources[r.src].events[r.e].style or "" for r in m.runs
                         if r.src in episode.sources and r.e is not None), "")

        def export_report(rep):
            return {k: v for k, v in asdict(rep).items() if k != "owners"}

        def timing():
            f, measurement = features(MEDIA, scratch)
            before = document_to_json(doc)
            begin = time.perf_counter()
            preview = plan(doc, episode.sources, f, Settings())
            planned = time.perf_counter() - begin
            unchanged = before == document_to_json(doc)
            begin = time.perf_counter()
            applied = editor.apply_autotime(preview)
            return {"audio": measurement, "settings": asdict(preview.settings), "segments": [asdict(s) for s in preview.segments],
                    "quality": asdict(preview.quality), "groups": preview.group_count, "members": preview.member_count,
                    "protected": preview.protected_count, "low_confidence": preview.low_confidence_count,
                    "window_confidence": dict(Counter(w.confidence for w in preview.mapping.windows)),
                    "plan_seconds": round(planned, 3), "apply_seconds": round(time.perf_counter() - begin, 3),
                    "plan_document_unchanged": unchanged, "applied": applied}
        step("autotime defaults plan/apply", timing)

        def ruby():
            engine = episode.reading_engine(doc)
            # Same core function and source selection as EpisodeSession.annotate.
            generated = 0
            for m in doc.members.values():
                source = next((episode.sources[r.src] for r in m.runs if r.src in episode.sources), None)
                generated += annotate_member(doc, m, engine, source=source)
            return {"generated": generated, "ruby": sum(a.kind == "ruby" for a in doc.annotations.values())}
        step("furigana session core", ruby)

        def people():
            subject, characters, details = bangumi.fetch_subject("2903")
            book = project.character_book()
            rows = bangumi.build_candidates(subject, characters, details, book, project.languages, traditional=True)
            chosen = [i for i, row in enumerate(rows) if row["checked"]]
            imported = bangumi.apply_import(book, rows, chosen, editor=editor)
            linked = editor.relink_speakers(book)
            refreshed = editor.reannotate(list(doc.members), episode.reading_engine(doc), episode.sources)
            return {"subject": subject, "candidates": len(rows), "selected": len(chosen), "imported": len(imported),
                    "relinked": linked, "linked_members": sum(bool((m.speaker or {}).get("character")) for m in doc.members.values()),
                    "pending": book.pending, "ruby_refreshed": refreshed,
                    "candidates_detail": [{k: r.get(k) for k in ("ja_name", "zh_name", "romaji", "checked", "relation")} for r in rows]}
        step("Bangumi 2903 live defaults/traditional/relink/reannotate", people)
        step("self-reference check", lambda: {k: len(v) for k, v in self_check.check(doc, project.character_book(), episode.id, "S01").items()})
        step("name finder", lambda: {"candidates": (len(rows := find_names(doc, episode.sources, project.character_book(), project.languages))), "rows": rows})
        step("pinyin PY on", lambda: {"generated": editor.set_pinyin_mode(project, "PY"),
            "mode": project.pinyin_mode, "pinyin": sum(a.kind == "pinyin" for a in doc.annotations.values()),
            "languages": dict(Counter(doc.members[a.member].lang for a in doc.annotations.values() if a.kind == "pinyin"))})
        step("pinyin off", lambda: {"generated": editor.set_pinyin_mode(project, "off"), "mode": project.pinyin_mode})

        def traditional(enabled):
            project.data.setdefault("settings", {}).setdefault("display", {}).setdefault("zh_traditional", {})["enabled"] = enabled
            project.save()
            before = document_to_json(doc)
            _ass, rep = episode.build_export()
            assert rep.traditional_lines == 0, rep.traditional_lines
            assert before == document_to_json(doc)
            return {"enabled": enabled, "report": export_report(rep), "document_unchanged": True}
        step("Simplified->Traditional on (expect 0)", lambda: traditional(True))
        step("Simplified->Traditional off", lambda: traditional(False))
        step("review packet export", lambda: {"paths": list(map(str, write_packet(doc, scratch, episode=episode.id, stamp="integration",
            orig_lang="ja", trans_lang="zh-Hant", chars=project.character_book(), pinyin_mode=project.pinyin_mode,
            display=DisplayRules.from_settings(project.settings()))))})

        def song():
            op = [m for m in doc.members.values() if source_style(m).lower() == "op"]
            assert op, "No op members found"
            gids = list(dict.fromkeys(m.group for m in op))
            convert = [g.id for g in doc.groups if g.id in gids and g.kind != "song"]
            if convert:
                editor.convert_to_song(convert, "OP")
            members = [m for g in doc.groups if g.id in gids for m in doc.members_of(g) if m.role == "song" and m.source_kind != "Comment"]
            beats = 0
            for m in members:
                derived = karaoke.derive(doc, m)
                for i, b in enumerate(derived):
                    b.t0 = round(m.start + (m.end - m.start) * i / len(derived))
                    b.t1 = round(m.start + (m.end - m.start) * (i + 1) / len(derived))
                    b.by = "manual"
                editor.set_karaoke(m.id, derived)
                beats += len(derived)
            editor.apply_karaoke_effect(members[0].id, "glow", scope="song")
            initial_glow = sum(t.get("effect", {}).get("preset") == "glow" for t in doc.karaoke.values())
            applications = 1
            # Documented whole-song scope is contiguous song groups. Credits in
            # this real OP split it; use the same API once per remaining block.
            for m in members:
                if not any(t.get("effect", {}).get("preset") == "glow" for t in karaoke.tracks_of(doc, m.id)):
                    editor.apply_karaoke_effect(m.id, "glow", scope="song")
                    applications += 1
            glow = [t for t in doc.karaoke.values() if t.get("effect", {}).get("preset") == "glow"]
            assert len(glow) == len(members), (len(glow), len(members))
            return {"op_members": len(op), "groups": len(gids), "converted_groups": len(convert), "song_members": len(members),
                    "beats": beats, "glow_tracks": len(glow), "first_whole_song_tracks": initial_glow,
                    "whole_song_applications": applications,
                    "anomalies": ["Credits split OP into contiguous song blocks; whole-song scope must be applied per block"] if applications > 1 else []}
        step("OP song/even beats/glow whole song", song)
        step("release export defaults", lambda: {"path": str((out := episode.export(scratch / "release.ass"))[0]), "report": export_report(out[1])})
        step("validate", lambda: validation(episode))

        def render():
            dialogue = next(m for m in doc.members.values() if m.role == "dialogue" and m.lang == "ja"
                            and m.start > 0 and m.end - m.start > 500 and doc.annotations_of(m.id)
                            and doc.members_of(doc.group(m.group), lane="translation"))
            op = next(m for m in doc.members.values() if m.style.startswith("OP.") and m.end - m.start > 500)
            sign = next(m for m in doc.members.values() if source_style(m).lower() in ("name", "namec", "note", "notec") and m.start > 0 and m.end > m.start)
            points = [("dialogue", (dialogue.start + dialogue.end) / 2000), ("op-glow-a", (op.start + op.end) / 2000),
                      ("op-glow-b", (op.start * .25 + op.end * .75) / 1000), ("sign", (sign.start + sign.end) / 2000)]
            frames = []
            for name, seconds in points:
                target = scratch / (name + ".png")
                output = proc.run(["ffmpeg", "-hide_banner", "-y", "-ss", f"{seconds:.6f}", "-i", str(MEDIA),
                    "-vf", f"setpts=PTS+{seconds:.6f}/TB,ass=release.ass", "-fps_mode", "passthrough", "-frames:v", "1", "-update", "1", str(target)],
                    cwd=scratch, capture_output=True, check=True, timeout=45)
                frames.append({"path": str(target), "seconds": seconds, "bytes": target.stat().st_size,
                               "libass_log": output.stderr.decode("utf-8", errors="replace")})
            return {"frames": frames}
        step("render 4 frames real FFmpeg/libass/video", render)

        def undo():
            labels = []
            for _ in range(5):
                tx = editor.history.undo()
                assert tx is not None, "Less than five undo steps"
                labels.append(tx.label)
            return {"undone": labels, **validation(episode)}
        step("undo last 5 and validate", undo)
        step("external inputs unchanged", lambda: {"subtitle_bytes": hashes == {str(p): digest(p) for p in (original, translation)},
            "media_size_mtime": media_stat == (MEDIA.stat().st_size, MEDIA.stat().st_mtime_ns), "source_copy_issues": episode.verify_sources()})
    finally:
        def cleanup():
            target = (scratch / "project").resolve()
            assert target.is_relative_to(temp) and target.parent == scratch
            if target.exists():
                for path in target.rglob("*"):
                    if path.is_file():
                        path.chmod(stat.S_IREAD | stat.S_IWRITE)
                shutil.rmtree(target)
            return {"project_deleted": not target.exists(), "path": str(target)}
        step("cleanup TEMP project", cleanup)
        result["seconds"] = round(time.perf_counter() - started, 3)
        statuses = {row["status"] for row in result["steps"]}
        result["status"] = "failed" if "failed" in statuses else "environment_blocked" if "environment_blocked" in statuses else "completed"
    return result


def validation(episode):
    rep = episode.validate()
    assert rep.ok, rep.errors
    return {"ok": rep.ok, "issues": [asdict(i) for i in rep.issues]}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    report = evaluate()
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    raise SystemExit(0 if report["status"] == "completed" else 1)
