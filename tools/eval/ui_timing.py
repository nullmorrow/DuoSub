"""Offscreen X4 timings on private copies; never opens a writable user project."""
from __future__ import annotations

import argparse
import json
import os
import math
import re
import statistics
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest.mock import Mock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("UI_TIMING_SOURCE_DIR", str(ROOT / "src")))
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEvent, QCoreApplication, QSettings, Qt, qVersion
from PySide6.QtTest import QTest
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget
from subkit_app.player.mpv import MpvWidget
from subkit.core.workspace import Project
from subkit.core.people import NameForm, needs_speaker
from subkit.core.layout_index import LayoutIndex
from subkit_app.mainwindow import MainWindow
from subkit_app.session import EpisodeSession, PreviewJob
from subkit_app.watch import GeometryJob, WatchController, OV_SPEAKER
from subkit_app.views.group_table import GroupTableModel, GroupFilter
from subkit_app.views.group_editor import GroupEditor
from subkit_app.views.source_panel import SourcePanel
from subkit_app.views.people import PeoplePanel
from subkit_app.views.waveform import WaveformView


class PlayerDouble(MpvWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.time, self.duration, self.paused = 0.0, 0.0, True
        self.mpv, self.osd = None, {}
        self._hide_timer, self._autohide = Mock(), False
        for name in ("message", "seek", "overlay", "clear_overlay", "shutdown", "load_subtitle"):
            setattr(self, name, Mock())

    def showEvent(self, event):
        QWidget.showEvent(self, event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scratch = Path(tempfile.mkdtemp(prefix="ui-timing-", dir=os.environ["TEMP"]))
    app = QApplication.instance() or QApplication([])
    # Windows offscreen has no native font database; load installed font files
    # into this process so QFontMetrics matches the normal Windows GUI.
    loaded_fonts = 0
    for folder in (Path(os.environ["WINDIR"]) / "Fonts",
                   Path(os.environ["LOCALAPPDATA"]) / "Microsoft/Windows/Fonts"):
        for font in folder.glob("*"):
            if font.suffix.lower() in (".ttf", ".otf", ".ttc"):
                loaded_fonts += QFontDatabase.addApplicationFont(str(font)) >= 0
    report = {"environment": f"Python {sys.version.split()[0]}, Qt {qVersion()}, offscreen, player double; real libass geometry/export",
              "scratch": str(scratch), "cases": [], "limitations": "No real mpv presentation, audio decode, human focus/IME or maximum-size acceptance. Normal full-episode background warm-up enabled; copied render probe cache, cold in-memory line geometry on temporary copies. Offscreen installed fonts loaded into this process."}
    report["offscreen_font_files_loaded"] = loaded_fonts
    steps = {}

    def instrument(cls, name):
        original = getattr(cls, name)
        def measured(self, *a, **kw):
            start = time.perf_counter()
            try:
                return original(self, *a, **kw)
            finally:
                steps.setdefault(f"{cls.__name__}.{name}", []).append((time.perf_counter() - start) * 1000)
        return patch.object(cls, name, measured)

    def settle(window, timeout=60):
        deadline = time.perf_counter() + timeout
        while True:
            app.processEvents()
            busy = (window._pending_group is not None or window.watch._job_running
                    or window.people._refresh_timer.isActive() or window.session._debounce.isActive()
                    or window.session._preview_running)
            if not busy:
                app.processEvents()
                return
            if time.perf_counter() > deadline:
                raise TimeoutError("UI did not settle within 60 s")
            QTest.qWait(1)

    patches = [instrument(cls, name) for cls, names in (
        (MainWindow, ("_show_group", "_on_time", "_flush_group_selection", "_refresh_k", "_fit_editor_docks")),
        (GroupEditor, ("set_group", "_show_fmt")), (SourcePanel, ("show_group",)),
        (EpisodeSession, ("_on_tx",)), (PreviewJob, ("run",)),
        (LayoutIndex, ("geometry", "precise_geometry")),
        (GeometryJob, ("run",)), (WatchController, ("request_geometry", "_draw_speakers")),
        (GroupTableModel, ("refresh_groups", "reset")), (PeoplePanel, ("refresh",)),
        (WaveformView, ("paintEvent",))) for name in names]
    for p in patches:
        p.start()
    # Also time the real name finder, including the initial uncached scan.
    import subkit_app.views.people as people
    finder = people.find_names
    def find(*a, **kw):
        start = time.perf_counter()
        try:
            return finder(*a, **kw)
        finally:
            steps.setdefault("name_finder.find_names", []).append((time.perf_counter() - start) * 1000)
    patches.append(patch.object(people, "find_names", find))
    patches[-1].start()
    try:
        for source in sorted((ROOT / "Projects").iterdir()):
            if source.name not in ("少女革命ウテナ", "食戟のソーマ"):
                continue
            data = json.loads((source / "subkit-project.json").read_text(encoding="utf-8"))
            episodes = data["episodes"][:1]
            target = scratch / source.name
            target.mkdir()
            shutil.copy2(source / "subkit-project.json", target)
            for sub in ("dict", "templates"):
                if (source / sub).exists():
                    shutil.copytree(source / sub, target / sub)
            # Match the normal evaluation's copied probe cache and warm-up.
            # The visible line geometry is cleared for each cold draw.
            (target / "cache").mkdir()
            for cache in (source / "cache").glob("render.sqlite3*"):
                shutil.copy2(cache, target / "cache" / cache.name)
            for entry in episodes:
                shutil.copytree(source / entry["path"], target / entry["path"])
            for entry in episodes:
                steps.clear()
                project = Project.open(target)
                episode = project.episode(entry["id"])
                settings = QSettings(str(scratch / "settings.ini"), QSettings.Format.IniFormat)
                with patch("subkit_app.mainwindow.MpvWidget", PlayerDouble), patch("subkit_app.mainwindow.QSettings", return_value=settings), patch.object(MainWindow, "_open_last"):
                    w = MainWindow()
                session = EpisodeSession(project, episode)
                w.project, w.session = project, session
                w.player.pause = Mock(side_effect=lambda: setattr(w.player, "paused", True))
                w.player.load_subtitle = Mock()
                w.player.osd = dict(w=1280, h=720)
                w.model = GroupTableModel(session.doc, languages=project.languages, character_book=session.editor.character_book)
                w.proxy = GroupFilter()
                w.proxy.setSourceModel(w.model)
                w.table.setModel(w.proxy)
                w.table.setup_columns()
                w.wave.set_document(session.doc)
                w.source.set_data(session.doc, episode.sources)
                session.changed.connect(w._on_changed)
                session.previewReady.connect(w._on_preview)
                session.busy.connect(w._on_busy)
                w.people.set_context(project, session)
                w.show()
                app.processEvents()
                case = {"project": source.name, "episode": episode.id, "groups": len(session.doc.groups)}
                report["cases"].append(case)
                try:
                    start = time.perf_counter()
                    session._start_preview()
                    settle(w)
                    case["initial_preview_ms"] = (time.perf_counter() - start) * 1000
                    case["normal_warm_up_started"] = w.watch._warm is not None
                    visible = {o.get("group") for e, o in zip(w.watch.ass.events, w.watch.owners)
                               if e.kind == "Dialogue" and e.end > e.start and o.get("kind") == "base"}
                    eligible = [g for g in session.doc.groups if g.id in visible and any(needs_speaker(g, m) for m in session.doc.members_of(g))]
                    named = next((g for g in eligible if not w.watch._speaker_text(g.id)[0].startswith("?" ) and w.watch._speaker_text(g.id)[0]), eligible[0])
                    unknown = next((g for g in eligible if w.watch._speaker_text(g.id)[0].startswith("?")), eligible[-1])
                    if w.watch._speaker_text(named.id)[0].startswith("?"):
                        book = session.editor.character_book
                        cid = book.characters[0].id if book.characters else book.add_character({"ja": NameForm(given="計測")}).id
                        for m in session.doc.members_of(named):
                            if needs_speaker(named, m):
                                m.speaker = {**(m.speaker or {}), "character": cid, "by": "user"}
                        case["name_prepared_on_copy"] = True
                        unknown = next(g for g in eligible if w.watch._speaker_text(g.id)[0].startswith("?"))
                    w.set_watch_mode(True, pinned=True)
                    app.processEvents()
                    w.act_watch_speaker.setChecked(True)
                    w.player.paused = False
                    for label, group in (("name", named), ("question", unknown)):
                        w.watch.geo.clear()
                        w.watch._speaker_geoms.clear()
                        w.watch._speaker_estimates.clear()
                        w.watch.invalidate_speakers()
                        w.player.overlay.reset_mock()
                        drawn_at, placements = [], []
                        def draw(oid, text):
                            if oid == OV_SPEAKER and text:
                                drawn_at.append(time.perf_counter())
                                placements.append(re.findall(r"\\pos\(([-.0-9]+),([-.0-9]+)\)", text))
                        w.player.overlay.side_effect = draw
                        ev = next(e for e, o in zip(w.watch.ass.events, w.watch.owners)
                                  if o.get("group") == group.id and o.get("kind") == "base" and e.kind == "Dialogue")
                        t = (ev.start + ev.end) / 200
                        w.player.time = t
                        start = time.perf_counter()
                        w._on_time(t)
                        settle(w)
                        case[label + "_settled_ms"] = (time.perf_counter() - start) * 1000
                        case[label + "_draw_ms"] = (drawn_at[0] - start) * 1000 if drawn_at else None
                        case[label + "_placements"] = placements
                        case[label + "_drawn"] = any(c.args[0] == OV_SPEAKER and c.args[1] for c in w.player.overlay.call_args_list)
                        case[label + "_debug"] = {"text": w.watch._speaker_text(group.id)[0], "active": w.watch.active(), "geometry": len(w.watch.geo), "enabled": w.watch._speakers_enabled(), "mapper": w.watch._mapper().ok}
                        start = time.perf_counter()
                        drawn_at.clear()
                        w._on_time(t)
                        settle(w)
                        case[label + "_cached_draw_ms"] = (drawn_at[0] - start) * 1000 if drawn_at else None
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    jumps = []
                    sample_groups = eligible[:4]
                    for group in sample_groups:
                        ev = next(e for e, o in zip(w.watch.ass.events, w.watch.owners)
                                  if o.get("group") == group.id and o.get("kind") == "base" and e.kind == "Dialogue")
                        w.player.time = (ev.start + ev.end) / 200
                        w.watch.geo.clear()
                        w.watch._speaker_geoms.clear()
                        w.watch._speaker_estimates.clear()
                        w.watch.invalidate_speakers()
                        samples = []
                        def capture(oid, text):
                            if oid == OV_SPEAKER and text:
                                samples.append(re.findall(r"\\pos\(([-.0-9]+),([-.0-9]+)\)", text))
                        w.player.overlay.side_effect = capture
                        w.watch.on_time(w.player.time)
                        settle(w)
                        if samples and samples[0] and len(samples[0]) == len(samples[-1]):
                            for before, after in zip(samples[0], samples[-1]):
                                distance = math.hypot(float(after[0]) - float(before[0]), float(after[1]) - float(before[1]))
                                jumps.append({"group": group.id, "distance_osd_px": distance,
                                              "estimate": before, "measured": after})
                    distances = sorted(j["distance_osd_px"] for j in jumps)
                    case["estimate_jump_samples"] = jumps
                    case["estimate_jump_osd_px"] = {"count": len(distances),
                        "median": statistics.median(distances) if distances else None,
                        "p95": distances[min(len(distances)-1, math.ceil(.95*len(distances))-1)] if distances else None,
                        "max": max(distances) if distances else None}
                    w.player.overlay.side_effect = None
                    w.set_watch_mode(False)
                    app.processEvents()
                    w.player.paused = True
                    w.table.setFocus()
                    w.select_group(eligible[0].id, seek=False)
                    settle(w)
                    focus, t0 = app.focusWidget(), w.wave.t0
                    member = next(m for m in session.doc.members_of(unknown) if needs_speaker(unknown, m))
                    book = session.editor.character_book
                    cid = book.characters[0].id if book.characters else book.add_character({"ja": NameForm(given="計測")}).id
                    start = time.perf_counter()
                    w._set_speaker(member.id, cid, True, True, False)
                    settle(w)
                    case["assign_ms"] = (time.perf_counter() - start) * 1000
                    case["assign_groups"] = len(session.editor.history.done[-1].pre_members)
                    case["focus_preserved"] = app.focusWidget() is focus
                    case["wave_t0_preserved"] = w.wave.t0 == t0
                    w.player.paused = False
                    start = time.perf_counter()
                    w._on_time(w.player.time + .04)
                    settle(w)
                    case["following_ms"] = (time.perf_counter() - start) * 1000
                    w.player.paused = True
                    clicks, settled_clicks = [], []
                    current_markers = []
                    click_steps = []
                    for row in range(1, 11):
                        idx = w.proxy.index(row, 4)
                        w.table.scrollTo(idx)
                        app.processEvents()
                        step_counts = {k: len(v) for k, v in steps.items()}
                        start = time.perf_counter()
                        QTest.mouseClick(w.table.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, w.table.visualRect(idx).center())
                        app.processEvents()
                        clicks.append((time.perf_counter() - start) * 1000)
                        click_steps.append({k: {"count": len(v[step_counts.get(k, 0):]),
                                                "total_ms": sum(v[step_counts.get(k, 0):])}
                                            for k, v in steps.items() if len(v) > step_counts.get(k, 0)})
                        current_markers.append([w.current_gid, w.table.current_gid(), w.wave.current_gid, w.watch.selected])
                        settle(w)
                        settled_clicks.append((time.perf_counter() - start) * 1000)
                    case["ctrl_click_ms"] = clicks
                    case["ctrl_click_steps"] = click_steps
                    case["ctrl_click_settled_ms"] = settled_clicks
                    case["ctrl_current_markers_match"] = all(len(set(markers)) == 1 for markers in current_markers)
                    case["click_metric"] = "Click to current-row/selection/wave/watch marker repaint; settled also waits 150 ms editor debounce and all geometry prefetch. Old ui-before ctrl_click_ms used settled."
                    case["selected_count"] = len(w.table.selected_gids())
                    case["steps"] = {k: {"count": len(v), "total_ms": sum(v), "max_ms": max(v)} for k, v in steps.items()}
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(json.dumps({k: v for k, v in case.items() if k not in ("steps", "ctrl_click_steps")}, ensure_ascii=False), flush=True)
                finally:
                    session.dirty = False
                    w.close()
                    session.close()
                    w.deleteLater()
                    session.deleteLater()
                    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                    app.processEvents()
    finally:
        for p in reversed(patches):
            p.stop()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
