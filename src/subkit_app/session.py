"""One open episode: document, editor (undo), autosave, and the background preview pipeline.

The preview is the release generator run on the current document in a worker
thread.  A result is applied only if the document revision it was built from is
still current — a slow job never overwrites a newer edit (REQUIREMENTS 九.6).
"""

from __future__ import annotations

import os
import tempfile
import time
import threading
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from subkit.core.ops import Editor, Transaction
from subkit.core.readings import annotate_member
from subkit.core.workspace import Episode, Project
from subkit.model.document import Group
from subkit.proc import CancelledError, KillableRun

from . import pool as job_pool


class _JobSignals(QObject):
    done = Signal(int, str, object, object)       # revision, path, AssDocument, report (with owners)
    failed = Signal(int, str)


class PreviewJob(QRunnable):
    def __init__(self, session: "EpisodeSession", revision: int, path: Path) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.runner = KillableRun()
        self.cancelled = False
        self.finished = threading.Event()
        self.session = session
        self.revision = revision
        self.path = path
        self.signals = _JobSignals()

    def cancel(self) -> None:
        self.cancelled = True
        self.runner.cancel()

    def run(self) -> None:  # noqa: D401 — worker thread
        try:
            self.runner.check_cancelled()
            import pickle

            from subkit.core.display import DisplayRules
            from subkit.core.export import ExportOptions, build_ass
            from subkit.templates import scheme_from_dict

            ep = self.session.episode
            with self.session.lock:
                doc = pickle.loads(pickle.dumps(ep.document, protocol=pickle.HIGHEST_PROTOCOL))
                # The snapshot, not the queueing time, decides which revision this preview shows.
                self.revision = ep.document.revision
            st = ep.settings()
            fonts, _ = scheme_from_dict(ep.project.data.get("template", {}).get("scheme", {}))
            opts = ExportOptions(mode="preview", include_actor=False, ruby_gap=float(st["ruby"]["gap"]),
                                 ruby_min_scale=float(st["ruby"]["min_scale"]), ruby_overhang=float(st["ruby"]["overhang"]),
                                 ruby_max_width=st["ruby"]["max_width"],
                                 playres=ep.effective_playres(), display=DisplayRules.from_settings(st),
                                 pinyin_mode=st["pinyin"]["mode"], ruby_offset=fonts.ruby_offset,
                                 pinyin_offset=fonts.pinyin_offset, template_playres=tuple(st["playres"]))
            t0 = time.perf_counter()
            ass, rep = build_ass(doc, ep.sources, self.session.template, ep.dir, opts,
                                 measure_cache=ep.project.cache_dir / "render.sqlite3", video_height=1080,
                                 cache=self.session.export_cache, runner=self.runner)
            self.runner.check_cancelled()
            tmp = self.path.with_name(self.path.name + ".tmp")
            ass.save(tmp)
            os.replace(tmp, self.path)
            rep.elapsed = time.perf_counter() - t0
            if not self.cancelled:
                self.signals.done.emit(self.revision, str(self.path), ass, rep)
        except CancelledError:
            pass
        except Exception:
            if not self.cancelled:
                self.signals.failed.emit(self.revision, traceback.format_exc())
        finally:
            self.finished.set()


class EpisodeSession(QObject):
    changed = Signal(object)                 # set of group ids touched (empty set = structure changed)
    previewReady = Signal(str, object, object)   # path, AssDocument, report (owners → what was clicked)
    status = Signal(str)
    busy = Signal(str, bool)                 # task name, running
    saved = Signal()
    saveStateChanged = Signal()

    def __init__(self, project: Project, episode: Episode) -> None:
        super().__init__()
        import threading

        self.project = project
        self.episode = episode
        self.lock = threading.RLock()
        self.editor = Editor(episode.document, episode.sources, character_book=project.character_book(),
                             reading_engine_factory=episode.reading_engine,
                             pinyin_words_factory=lambda lang: project.pinyin_words(lang) if project.pinyin_mode != "off" else None)
        self.editor.history.listeners.append(self._on_tx)
        self.template = project.template()
        from subkit.core.export import ExportCache

        self.export_cache = ExportCache()          # used only by the (single) preview job
        self.engine = None
        self._dirty = getattr(episode, "times_changed", False) or getattr(episode, "recovered", False)
        try:
            self.last_saved_at = (episode.dir / "document.json").stat().st_mtime
        except FileNotFoundError:
            self.last_saved_at = None
        self.recovery_saved_at = None
        self._closed = False
        self.pool = QThreadPool.globalInstance()
        self._preview_dir = Path(tempfile.gettempdir()) / "subkit_preview"
        self._preview_dir.mkdir(exist_ok=True)
        self.preview_path = self._preview_dir / f"{project.data['id']}_{episode.id}.ass"
        self._preview_running = False
        self._preview_wanted = False
        self._preview_job = None
        self._applied_revision = -1
        self._quiet_revisions: set[int] = set()   # revisions that changed speakers only (layout unchanged)
        self.people_change = False
        self.preview_change = True
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._start_preview)
        self._autosaved_revision: Optional[int] = None
        self._autosave = QTimer(self)
        self._autosave.setInterval(30_000)
        self._autosave.timeout.connect(self.autosave)
        self._autosave.start()

    # ------------------------------------------------------------ document
    @property
    def doc(self):
        return self.episode.document

    def _on_tx(self, tx: Transaction, how: str) -> None:
        if self._closed:
            return
        self.dirty = True
        touched = set(tx.pre_groups) | {m.group for m in (self.doc.members.get(k) for k in tx.pre_members) if m}
        if tx.people_changed or getattr(tx, "reading_rules_changed", False):
            self.engine = None
        # members whose text changed need fresh reading suggestions (confirmed readings are kept).  Positions,
        # formats, times, confirmation and dialogue ↔ song keep the text and every reading, on undo/redo too: the
        # snapshots before and after the step tell (a member created or deleted counts as changed).
        post = (tx.post or {}).get("members", {})
        for mid in (() if tx.keeps_text else tx.pre_members):
            if getattr(tx.pre_members[mid], "runs", None) == getattr(post.get(mid), "runs", None):
                continue
            m = self.doc.members.get(mid)
            if m is not None and m.lang.lower().split("-")[0] == "ja" and m.role in ("dialogue", "top", "alt", "song"):
                self._reannotate(m)
            if (m is not None and m.lang.lower().startswith("zh")
                    and self.project.data.get("settings", {}).get("pinyin", {}).get("mode", "off") != "off"):
                from subkit.core.pinyin_notes import annotate_pinyin, episode_rules

                with self.lock:
                    annotate_pinyin(self.doc, m, episode_rules(self.doc, self.project.pinyin_words(m.lang)))
        self.people_change = tx.people_changed
        self.preview_change = True
        if tx.people_changed:
            from subkit.core.export import _member_sig

            post = tx.post or {}
            touched = {getattr(m, "group", None) for mid, m in tx.pre_members.items()
                       if m != post.get("members", {}).get(mid)} - {None}
            for side in (tx.pre_annotations, post.get("annotations", {})):
                touched.update(self.doc.members[a.member].group for a in side.values()
                               if hasattr(a, "member") and a.member in self.doc.members)
            self.preview_change = bool(tx.pre_groups or tx.pre_order is not None or tx.pre_rules is not None
                                       or tx.pre_karaoke or tx.pre_annotations)
            if not self.preview_change:
                self.preview_change = any(
                    not hasattr(before, "runs") or not hasattr(post.get("members", {}).get(mid), "runs")
                    or _member_sig(before, [], None, False) != _member_sig(post["members"][mid], [], None, False)
                    for mid, before in tx.pre_members.items())
        if not self.preview_change:
            self._quiet_revisions.add(self.doc.revision)
        self.changed.emit(touched)
        if self.preview_change:
            self.request_preview()

    def _reannotate(self, m) -> None:
        if self.engine is None:
            self.engine = self.episode.reading_engine()
        src = next((self.episode.sources[r.src] for r in m.runs if r.src in self.episode.sources), None)
        with self.lock:
            annotate_member(self.doc, m, self.engine, source=src)

    @property
    def dirty(self) -> bool:
        return self._dirty

    @dirty.setter
    def dirty(self, value: bool) -> None:
        value = bool(value)
        if value != self._dirty:
            self._dirty = value
            self.saveStateChanged.emit()

    def save(self) -> None:
        with self.lock:
            self.episode.save()
        self.last_saved_at = time.time()
        self.dirty = False
        self.saveStateChanged.emit()
        self.saved.emit()
        self.status.emit(f"已保存 {time.strftime('%H:%M:%S')}")

    @Slot()
    def autosave(self) -> None:
        if self._closed or not self.dirty or self._autosaved_revision == self.doc.revision:
            return
        from subkit.model.serialize import document_to_json, write_json_atomic

        with self.lock:
            write_json_atomic(self.episode.dir / "journal" / "autosave.json", document_to_json(self.doc))
            self._autosaved_revision = self.doc.revision
        self.recovery_saved_at = time.time()
        self.saveStateChanged.emit()
        self.saved.emit()
        self.status.emit(f"自动保存（恢复用）{time.strftime('%H:%M:%S')}")

    # ------------------------------------------------------------ preview
    def close(self) -> None:
        """Retire this episode; running jobs may finish but cannot publish results."""
        self._closed = True
        self._debounce.stop()
        self._autosave.stop()
        self._preview_wanted = False
        job, self._preview_job = self._preview_job, None
        if job is not None:
            job.cancel()
            if job_pool.try_take(job, self.pool):
                job.finished.set()
            job.finished.wait(2.0)

    def stop_preview(self) -> bool:
        """Release the layout database before explicitly clearing its cache."""
        self._debounce.stop()
        job = self._preview_job
        if job is not None:
            job.cancel()
            if job_pool.try_take(job, self.pool):
                job.finished.set()
            if not job.finished.wait(2.0):
                return False
            job.signals.done.disconnect(self._preview_done)
            job.signals.failed.disconnect(self._preview_failed)
        self._preview_job = None
        self._preview_running = self._preview_wanted = False
        self.busy.emit("预览排版", False)
        return True

    def layout_current(self, revision: int) -> bool:
        """A preview built at `revision` still shows the current layout: every later step changed speakers only."""
        return revision >= 0 and all(r in self._quiet_revisions for r in range(revision + 1, self.doc.revision + 1))

    def request_preview(self) -> None:
        if not self._closed:
            self._debounce.start()

    @Slot()
    def _start_preview(self) -> None:
        if self._closed:
            return
        if self._preview_running:
            self._preview_wanted = True
            return
        self._preview_running = True
        self.busy.emit("预览排版", True)
        job = PreviewJob(self, self.doc.revision, self.preview_path)
        self._preview_job = job
        job.signals.done.connect(self._preview_done)
        job.signals.failed.connect(self._preview_failed)
        job_pool.start(job, pool=self.pool)

    @Slot(int, str, object, object)
    def _preview_done(self, revision: int, path: str, ass, rep) -> None:
        sender = self.sender()
        if sender is not None and (self._preview_job is None or sender is not self._preview_job.signals):
            return
        self._preview_running = False
        if self._closed:
            return
        self.busy.emit("预览排版", False)
        from subkit.core.display import DisplayRules
        current_rules = DisplayRules.from_settings(self.episode.settings()).replacements
        settings_current = getattr(rep, "replacement_key", None) in (None, current_rules)
        if revision >= self._applied_revision and settings_current:
            self._applied_revision = revision
            self.previewReady.emit(path, ass, rep)
        if self._preview_wanted or not self.layout_current(revision) or not settings_current:
            self._preview_wanted = False
            self._start_preview()

    @Slot(int, str)
    def _preview_failed(self, revision: int, err: str) -> None:
        sender = self.sender()
        if sender is not None and (self._preview_job is None or sender is not self._preview_job.signals):
            return
        self._preview_running = False
        if self._closed:
            return
        self.busy.emit("预览排版", False)
        self.status.emit("预览生成失败：" + err.strip().splitlines()[-1])
        print(err)

    # ------------------------------------------------------------ helpers for views
    def dialogue_groups(self) -> list[Group]:
        return [g for g in self.doc.groups if g.kind == "dialogue"]
