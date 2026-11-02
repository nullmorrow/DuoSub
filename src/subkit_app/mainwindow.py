"""SubKit main window: video + waveform + group editor + group list + original-text panel."""

from __future__ import annotations

import copy
import json
import re
import time
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QSettings, QStandardPaths, Qt, QTimer, QThreadPool, Slot, QUrl
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QKeySequence, QDesktopServices
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QTextEdit, QLineEdit, QInputDialog, QDialog, QProgressBar, QSplitter, QTabWidget, QToolBar, QVBoxLayout, QWidget)

from .views.fonts import FontCheckDialog
from subkit.core.fontcheck import DEFAULT_FONT_LIBRARY, FontLibrary, check_fonts, report_lines
from subkit.core.convert import K_HINT, dialogue_text, export_stripped_text, plan_conversion, status_text
from subkit.core.stepping import fully_visible_time
from subkit.core.merge import MergeSettings
from subkit.audio_match import EpisodeIndex, SearchResult, template_from
from subkit.core import karaoke as kara
from subkit.core.autotime_doc import Settings as AutoTimeSettings, plan as plan_autotime
from subkit.core.passages import PassageStore, already_applied, snapshot
from subkit.core.readings import visible_map
from subkit.core.position import PositionTarget, preset_side, restore, set_alignment, set_xy
from subkit.core.workspace import Episode, Project
from subkit.core.punctuation_profiles import (PunctuationProfileStore, apply as apply_punctuation_profile,
                                               matches as punctuation_profile_matches)
from subkit.model.document import Group
from subkit.reading.kanji_split import split_kanji_reading

from subkit.languages import DEFAULT_LANGUAGES, JAPANESE_ONLY, has_japanese, lane_label, language_name

from .player.mpv import MpvWidget
from . import APP_VERSION
from . import help as app_help
from .actions import ActionRegistry
from .action_hub import ActionHub
from .session import EpisodeSession
from .episode_job import EpisodeJob, EpisodeProgress
from .export_job import ExportJob, ExportProgress
from .views.dialogs import AddEpisodeDialog, ConvertToSongDialog, ExportDialog, MergeDialog, NewProjectDialog
from .views.shift_groups import ShiftGroupsDialog
from .views.group_times import GroupTimesDialog
from .views.check_result import CheckResultDialog, reanalyze_text
from .views.group_editor import GroupEditor
from .views.group_table import GroupFilter, GroupTable, GroupTableModel, conversion_reasons, set_action_reason, sign_conversion_reason
from .views.autotime import AutoTimeResultDialog, EpisodeAutoProcessDialog, busy as autotime_busy
from .views.passages import (CATEGORY_NAMES, PassageLibraryDialog, PassagesDialog, RecordPassageDialog,
                             duplicate_choice, passage_name)
from .views.reading_popup import PinyinSettingsDialog, ReadingPopup
from .views.ruby_settings import RubySettingsDialog
from .views.punctuation import PunctuationDialog
from .views.seek_strip import SeekStrip
from .views.source_panel import SourcePanel
from .views.dock_title import DockTitleBar
from .views.people import CharacterDetailsDialog, PeoplePanel, SelfCheckDialog, SpeakerPicker
from .views.waveform import WaveformPanel, WaveformView, fmt_ms
from .kalign_runner import KalignRunner
from .karaoke_ui import KaraokeTapper
from .watch import InlineTextEditor, WatchController

APP_NAME = "双语字幕工坊"


class _ToolbarTips(QObject):
    """A tool button copies its action's tooltip on every change; put 「名字（快捷键）」 back afterwards."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.ActionChanged:
            action_id = next((key for key, (button, _text) in self.window._toolbar_buttons.items() if button is obj), None)
            if action_id is not None:
                obj.event(event)                      # let the button re-read its action first
                self.window._sync_toolbar_button(action_id)
                return True
        return False


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1600, 960)
        import os

        # settings keep the pre-rename keys so window layout and recent projects survive; tests use their own
        self.settings = QSettings("SubKit", os.environ.get("SUBKIT_SETTINGS_APP", "SubKit-Opus"))
        registry = ActionRegistry()
        registry.load(self.settings)  # Read overrides without changing stored settings.
        self.action_hub = ActionHub(registry, self)
        # Native Windows QSettings lives in the registry; file-backed settings use their own folder.
        settings_path = Path(self.settings.fileName())
        settings_dir = (settings_path.parent if settings_path.is_absolute() else
                        Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)))
        self.punctuation_profiles = PunctuationProfileStore(settings_dir / "punctuation-profiles")
        self.kalign_runner_factory = KalignRunner
        self.listening_dialog_factory = None
        self._listening_context = None
        self._kalign_runner = self._kalign_context = self._kalign_dialog = None
        self._auto_suggestion = None
        self._kalign_results = None
        self._dialign_runner = self._dialign_context = self._dialign_dialog = None
        self._dialign_results = None
        self._episode_job = None
        self._export_job = None
        self._export_progress = None
        self._export_close_pending = False
        self._episode_progress = None
        self._asking_close = False          # the "cancel and close?" question is open
        self._pending_episode_result = None  # a job that finished while that question was open
        self._closing = False
        self._episode_actions = []
        self.project: Optional[Project] = None
        self.session: Optional[EpisodeSession] = None
        self._karaoke_fx_panel = None
        self._karaoke_fx_root = None
        self._passage_features = None
        self._passage_index = None
        self._passage_audio_session = None
        self._auto_process_asked: set[tuple[str, str]] = set()
        self._auto_process_pending = None
        self._auto_process_timer = QTimer(self)
        self._auto_process_timer.setInterval(250)
        self._auto_process_timer.timeout.connect(self._try_auto_process)
        self.current_gid: Optional[str] = None
        self._pending_position_seek: tuple[EpisodeSession, int] | None = None
        self._restored_position_time: tuple[EpisodeSession, int, float] | None = None
        self._position_media_ready = False
        self._pending_group = None
        self._editor_pending = self._source_pending = False
        self._group_nav_keys = set()
        self._group_nav_last = 0.0         # time of the last navigation key event (held keys auto-repeat)
        self._group_select_timer = QTimer(self)
        self._group_select_timer.setSingleShot(True)
        self._group_select_timer.setInterval(150)
        self._group_select_timer.timeout.connect(self._settle_group_selection)
        self._stop_at: Optional[float] = None
        self._step_select_gid: Optional[str] = None
        self._group_playback_gid: Optional[str] = None
        self._follow = True
        self.watch_mode = False
        self._saved_layout: dict = {}
        self._karaoke_previous_layout = self.settings.value("karaoke/previous_layout", "video")
        self._karaoke_filter = "全部"
        self._speeds = {kind: self.settings.value("playback/" + kind + "_speed", 1.0, type=float)
                        for kind in ("song", "dialogue")}
        self._speeds = {kind: value if value in (1.0, 0.75, 0.5) else 1.0 for kind, value in self._speeds.items()}
        self._inline = None
        self._position_dirty = False
        self._position_cache = None

        # ---- widgets
        self.player = MpvWidget()
        self.wave = WaveformView()
        self.wave_panel = WaveformPanel(self.wave)
        self.wave_panel.set_settings(self.settings)
        self.editor = GroupEditor()
        self.editor.set_settings(self.settings)
        self.editor.whole_speaker.setChecked(self.settings.value("speaker_whole_group", True, type=bool))
        self.editor.whole_speaker.toggled.connect(lambda value: self.settings.setValue("speaker_whole_group", value))
        self.editor.speakerRequested.connect(self._set_speaker)
        self._recent_speakers = []
        self.editor.position.set_settings(self.settings)
        self.table = GroupTable()
        self.source = SourcePanel()
        self.video_box = QWidget()          # video (+ the waveform below it in 观剧模式, resizable)
        vb = QVBoxLayout(self.video_box)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(0)
        self.video_split = QSplitter(Qt.Orientation.Vertical)
        self.video_split.setChildrenCollapsible(False)
        self.player_box = QWidget()                   # the video + a seek strip under it (outside the picture)
        self.player_box.setMinimumHeight(220)
        pb = QVBoxLayout(self.player_box)
        pb.setContentsMargins(0, 0, 0, 0)
        pb.setSpacing(0)
        pb.addWidget(self.player, 1)                  # the native video window keeps this parent for good
        self.seek_strip = SeekStrip()
        pb.addWidget(self.seek_strip)
        self.video_split.addWidget(self.player_box)
        vb.addWidget(self.video_split, 1)
        self._watch_wave_h = 0
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.wave_panel)
        right.addWidget(self.editor)
        right.setSizes([240, 400])
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.video_box)
        self.right_host = QMainWindow()
        self.right_host.setCentralWidget(right)
        top.addWidget(self.right_host)
        top.setSizes([720, 880])
        main = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(top)
        main.addWidget(self.table)
        main.setSizes([600, 360])
        self.setCentralWidget(main)
        self.right_split, self.top_split, self.main_split = right, top, main
        for split in (main, top, right):
            split.setChildrenCollapsible(False)
        self.video_box.setMinimumSize(480, 260)
        self.editor.setMinimumWidth(380)
        self.right_host.setMinimumWidth(380)
        self.table.setMinimumHeight(200)
        self.wave_panel.setMinimumHeight(170)
        self.edit_layout = None
        self._pending_video_fit = False
        self.watch = WatchController(self.player, self)
        self.watch.speakerRequested.connect(self._open_speaker_picker)
        self._speaker_picker = None
        self.tapper = KaraokeTapper(self.player, self._k_line_at, self, on_idle_k=lambda: self._k_action("start"),
                                    on_selected_k=self._k_retap_selected, registry=self.action_hub.registry,
                                    accepts_input=lambda: not self._typing() and self._export_job is None)
        self._k_unit_pref = "syllable"
        dock = QDockWidget("原文", self)
        dock.setObjectName("dock_source")
        dock.setWidget(self.source)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.dock_source = dock
        self.people = PeoplePanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.people)
        self.people.hide()
        for panel in (self.dock_source, self.people):
            panel.setTitleBarWidget(DockTitleBar(panel, self._dock_back_right))
            panel.topLevelChanged.connect(self._queue_dock_screen_check)
            panel.visibilityChanged.connect(self._fit_editor_docks)

        # ---- status bar
        self.lbl_time = QLabel("0:00:00.00")
        self.lbl_counts = QLabel("")
        self.lbl_task = QLabel("")
        self.lbl_speed = QLabel("")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(120)
        self.progress.setRange(0, 0)
        self.progress.hide()
        sb = self.statusBar()
        sb.addPermanentWidget(self.lbl_counts)
        sb.addPermanentWidget(self.lbl_task)
        sb.addPermanentWidget(self.lbl_speed)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.lbl_time)
        self.lbl_save_state = QLabel("")
        self.lbl_save_state.help_anchor = "save"
        sb.addPermanentWidget(self.lbl_save_state)

        self.private_enabled = self.settings.value("private/enabled", True, type=bool)
        self._private_actions = []
        self._batch_dialog = None
        self._build_menus()
        self._build_shortcuts()
        QApplication.instance().installEventFilter(self)

        # ---- connections
        self.editor.syntaxCopied.connect(lambda text: self.statusBar().showMessage(f"已复制：{text}"))
        self.editor.syntaxInserted.connect(lambda text: self.statusBar().showMessage(
            f"已写入：{text}（" + ("在「=」后写注释，" if "{@注:" in text else "") + "按 Enter 生效，Esc 放弃）"))
        self.editor.contentHeightChanged.connect(self._fit_editor_docks)
        self.table.groupActivated.connect(self._on_table_group)
        self.table.groupSelectionChanged.connect(self._update_position_actions)
        self.table.confirmNext.connect(self.confirm_next)
        self.table.confirmOnly.connect(lambda: self._confirm(move=False))
        self.table.deleteGroupsRequested.connect(self._delete_groups)
        self.table.deleteLaneRequested.connect(self._delete_lane)
        self.table.confirmGroupsRequested.connect(self._confirm_groups)
        self.table.hiddenGroupsRequested.connect(lambda gids, value: self._review_flag(gids, "hidden", value))
        self.table.markedGroupsRequested.connect(lambda gids, value: self._review_flag(gids, "marked", value))
        self.table.markNoteRequested.connect(self._mark_note_dialog)
        self.table.convertRequested.connect(self._convert_groups)
        self.table.karaokeEffectRequested.connect(self._k_context_effect)
        self.table.karaokeConfirmRequested.connect(self._k_context_confirm)
        self.table.karaoke_effect_choices = self._k_context_effect_choices
        self.table.context_action_lookup = self.action_hub.action
        self.table.context_action_spec = self.action_hub.registry.lookup
        self.table.context_action_keys = self.action_hub.registry.effective_keys
        self.wave_panel.more_menu.aboutToShow.connect(self._refresh_redistribute_menu)
        for menu in self.wave_panel.scope_menus.values():
            menu.aboutToShow.connect(self._refresh_redistribute_menu)
        self.table.addLanguageRequested.connect(self._add_language_dialog)
        self.table.listeningRequested.connect(self._start_listening)
        self.table.recordPassageRequested.connect(self._record_passage)
        self.table.shiftGroupsRequested.connect(self._shift_groups_dialog)
        self.table.duplicateGroupsRequested.connect(self._duplicate_groups)
        self.table.groupTimesRequested.connect(self._group_times_dialog)
        self.wave.audioStatus.connect(lambda message: self.statusBar().showMessage(message, 6000))
        self.wave.featuresLoaded.connect(self._queue_auto_process)
        self.table.playRequested.connect(lambda gid: (self.select_group(gid), self.play_group()))
        self.player.timeChanged.connect(self._on_time)
        self.player.fileLoaded.connect(self._apply_speed)
        self.player.fileLoaded.connect(self._on_position_media_loaded)
        self.player.timeChanged.connect(self.seek_strip.set_time)
        self.player.durationChanged.connect(self.seek_strip.set_duration)
        self.seek_strip.seekRequested.connect(lambda t, exact: self.player.seek(t, exact=exact))
        self.player.pauseChanged.connect(self._on_pause_changed)
        self.wave.seekRequested.connect(lambda ms: self.player.seek(ms / 1000.0))
        self.wave.groupClicked.connect(self._on_wave_group)
        self.wave.timeClicked.connect(self._on_wave_time)
        self.wave.memberTimeEdited.connect(self._on_member_time)
        self.wave.groupShifted.connect(lambda gid, d: self._edit(lambda: self.session.editor.shift_group(gid, d)))
        self.watch.groupPicked.connect(self._on_video_pick)
        self.watch.deselected.connect(self._video_deselect)
        self.watch.resumeRequested.connect(self._video_resume)
        self.watch.readingRequested.connect(self._on_reading)
        self.watch.newReadingRequested.connect(self._on_new_reading)
        self.watch.pinyinRequested.connect(self._on_pinyin)
        self.watch.moveRequested.connect(self._on_move)
        self.watch.transformRequested.connect(self._on_transform)
        self.watch.textEditRequested.connect(self._inline_edit)
        self.watch.doubleClickedEmpty.connect(self.toggle_watch_mode)
        self.watch.note.connect(lambda t: self.player.message(t))
        self.watch.warming.connect(self._on_warming)
        self.tapper.finished.connect(self._k_tapped)
        self.tapper.activeBeat.connect(self._k_active)
        self.tapper.status.connect(lambda t: (self.player.message(t, 3000), self.statusBar().showMessage(t, 6000)))
        self.wave.beatsEdited.connect(self._k_dragged)
        self.wave.beatSplit.connect(self._k_split)
        self.wave.beatMerge.connect(self._k_merge)
        self.wave.playRange.connect(self._play_range)
        self.wave_panel.kAction.connect(self._k_action)
        self.wave_panel.kAutoOption.connect(self._auto_option)
        self.wave_panel.kauto_menu.aboutToShow.connect(self._refresh_auto_menu)
        self.wave_panel.adoptTiming.connect(self._adopt_timing)
        self.wave_panel.kEffect.connect(self._k_effect)
        self.wave_panel.kParameters.connect(self._k_parameters)
        self.wave_panel.kUnit.connect(self._k_unit)
        self.wave_panel.speedChanged.connect(self._set_speed)
        # view.karaoke owns layout changes for both the menu and K bar.
        self.wave_panel.btn_pin.toggled.connect(self._on_pin)
        self.editor.textCommitted.connect(self._on_text)
        self.editor.missingTextCommitted.connect(self._add_language_text)
        self.watch.addLanguageRequested.connect(self._add_language_dialog)
        self.editor.readingRequested.connect(self._on_reading)
        self.editor.newReadingRequested.connect(self._on_new_reading)
        self.editor.pinyinRequested.connect(self._on_pinyin)
        self.editor.restoreRequested.connect(lambda cid: self._edit(lambda: self.session.editor.restore_comment(cid)))
        self.editor.action.connect(self._on_action)
        from .views.quick_colours import QuickColourModel

        self.quick_colours = QuickColourModel(self)
        self.editor.quick_colours.set_model(self.quick_colours)
        self.editor.formatRequested.connect(self._on_format)
        self.editor.groupFormatRequested.connect(self._on_group_format)
        self.editor.clearFormatRequested.connect(lambda mid: self._edit(lambda: self.session.editor.clear_format(mid)))
        self.editor.style_lookup = self._style_of
        self.editor.positionTargetChanged.connect(self._refresh_position)
        self.editor.position.xyRequested.connect(self._position_xy)
        self.editor.position.alignmentRequested.connect(self._position_alignment)
        self.editor.position.actionRequested.connect(self._position_row_action)
        self.source.groupRequested.connect(lambda gid: self.select_group(gid, seek=True))
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        migrated = self.settings.value("layouts/version", 0, type=int) == 1
        layout = self.settings.value("layouts/current", "video") if migrated else "video"
        self.set_edit_layout(layout, initial=True, restore=migrated)
        self.settings.setValue("layouts/version", 1)
        self._apply_speed()
        if self.settings.contains("shortcuts/version"):
            self._refresh_shortcuts()
        QTimer.singleShot(0, self._open_last)

    # ================================================================ menus
    def _build_menus(self) -> None:
        mb = self.menuBar()
        self._menubar = mb
        m = self._menu(mb, "文件(&F)", "project")
        self._act(m, "file.new", self.new_project, "Ctrl+Shift+N")
        self._act(m, "file.open", self.open_project_dialog, "Ctrl+O")
        m.addSeparator()
        self._act(m, "file.save", self.save, "Ctrl+S")
        self.act_ask_save = self._act(m, "file.ask_save", lambda: None)
        self.act_ask_save.setCheckable(True)
        self.act_ask_save.setChecked(self.settings.value("save/ask_on_close", True, type=bool))
        self.act_ask_save.setProperty("help_anchor", "save")
        self.act_ask_save.toggled.connect(lambda on: self.settings.setValue("save/ask_on_close", on))
        m.addSeparator()
        export_menu = self._menu(m, "导出", "export")
        self._act(export_menu, "file.export", self.export)
        self._act(export_menu, "file.export_season", self.export_season)
        self._act(export_menu, "file.export_review_list", self._export_review_list)
        m.addSeparator()
        self._act(m, "project.template", self.change_template)
        m.addSeparator()
        self._act(m, "file.quit", self.close)
        m = self._menu(mb, "集(&P)", "project")
        self.ep_menu = m
        self._act(m, "episode.add", self.add_episode)
        from .views.batch import MODE_LABELS
        self.batch_actions = {}
        for mode in MODE_LABELS:
            action_id = "episode.batch" if mode == "5.1" else "episode.batch_bilingual"
            action = self._act(m, action_id, lambda checked=False, mode=mode: self._batch(mode))
            action.setProperty("help_anchor", "batch")
            action.setVisible(self.private_enabled)
            action.setEnabled(self.project is not None)
            self._episode_actions.append(action)
            self.batch_actions[mode] = action
        m.aboutToShow.connect(self._update_batch_actions)
        m.addSeparator()
        self._act(m, "episode.video_links", self._video_links)
        self._act(m, "episode.source_copies", self._source_copies)
        self._act(m, "episode.reanalyse", self.reanalyze)
        self._act(m, "episode.merge", self.merge)
        m.addSeparator()
        self.act_remove_episode = self._act(m, "episode.remove", self.remove_episode)
        self.act_remove_episode.setProperty("help_anchor", "project")
        m.aboutToShow.connect(self._update_remove_episode_action)
        m = self._menu(mb, "编辑(&E)", "review")
        self.edit_menu = m
        self.act_undo = self._act(m, "edit.undo", self.undo)
        self.act_redo = self._act(m, "edit.redo", self.redo)
        m.addSeparator()
        self._act(m, "group.duplicate", lambda: self._duplicate_groups(self.table.selected_gids()))
        m.aboutToShow.connect(self._update_group_menu_actions)
        m.addSeparator()
        self._act(m, "edit.delete_member", self.delete_member)
        self._act(m, "group.delete", self.delete_selected)
        m = self._menu(mb, "字幕(&S)", "review")
        self.subtitle_menu = m
        self._act(m, "group.confirm_next", self.confirm_next)
        self._act(m, "group.confirm")
        self._act(m, "group.unconfirm", lambda: self._confirm_groups(self.table.selected_gids(), False))
        m.addSeparator()
        self._act(m, "group.merge", lambda: self._on_action("merge"))
        self._act(m, "group.split", lambda: self._on_action("split"))
        self._act(m, "group.unlink", lambda: self._on_action("unlink"))
        boundary = self._menu(m, "分界", "review")
        self._act(boundary, "group.head_to_prev", lambda: self._move("head_to_prev"))
        self._act(boundary, "group.tail_to_next", lambda: self._move("tail_to_next"))
        self._act(boundary, "group.take_from_prev", lambda: self._move("take_from_prev"))
        self._act(boundary, "group.take_from_next", lambda: self._move("take_from_next"))
        boundary.addSeparator()
        self._act(boundary, "group.cursor_to_prev")
        self._act(boundary, "group.cursor_to_next")
        m.addSeparator()
        for lane in ("original", "translation"):
            self._act(m, "group.add_" + lane,
                      lambda _=False, lane=lane: self._add_language_dialog(self.table.selected_gids(), lane))
        self._act(m, "group.ai_listen", lambda: self._listen_selected())
        m.addSeparator()
        convert = self._menu(m, "转为", "review")
        self.act_convert_song = self._act(convert, "group.to_song",
                                         lambda: self._convert_groups(self.table.selected_gids(), "song"))
        self.act_convert_dialogue = self._act(convert, "group.to_dialogue",
                                             lambda: self._convert_groups(self.table.selected_gids(), "dialogue"))
        self.act_convert_sign = self._act(convert, "group.to_sign",
                                         lambda: self._convert_groups(self.table.selected_gids(), "sign"))
        self.act_convert_sign.setProperty("help_anchor", "review")
        self.act_convert_dialogue.setProperty("help_anchor", "review")
        m.aboutToShow.connect(self._update_convert_actions)
        self._update_convert_actions()
        self.position_menu = self._menu(m, "位置", "position")
        self.position_actions = {}
        for action in ("left", "right", "top", "bottom", "alt", "restore"):
            if action == "restore":
                self.position_menu.addSeparator()
            self.position_actions[action] = self._act(
                self.position_menu, "position." + action, lambda _=False, a=action: self._position_action(a, self.table.selected_gids()))
        m.aboutToShow.connect(self._update_position_actions)
        self.position_menu.aboutToShow.connect(self._update_position_actions)
        self._update_position_actions()
        self.act_transform = self._act(m, "watch.transform", self._start_transform)
        self.addAction(self.act_transform)
        self.act_transform.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        self.act_transform.setShortcuts([QKeySequence(key) for key in
                                        self.action_hub.registry.effective_keys("watch.transform", "watch")])
        m.aboutToShow.connect(self._update_transform_action)
        self._update_transform_action()
        m.addSeparator()
        for action_id, flag in (("group.hide", "hidden"), ("group.mark", "marked")):
            action = self._act(m, action_id, lambda _=False, flag=flag: self._toggle_review_flag(flag))
            action.setCheckable(True)
        self._act(m, "group.mark_note", lambda: self._mark_note_dialog(self.table.selected_gids()))
        m.aboutToShow.connect(self._update_group_menu_actions)
        m = self._menu(mb, "时间轴(&T)", "timing")
        self.time_menu = m
        self._act(m, "time.set", lambda: self._group_times_dialog(self.table.selected_gids()))
        self.act_shift_groups = self._act(m, "time.shift", lambda: self._shift_groups_dialog(self.table.selected_gids()))
        self.act_shift_groups.setProperty("help_anchor", "timing")
        m.aboutToShow.connect(self._update_shift_action)
        self._update_shift_action()
        for action_id in ("time.adopt", "time.start_minus", "time.start_plus", "time.end_minus", "time.end_plus"):
            self._act(m, action_id)
        m.addSeparator()
        unify = self._menu(m, "统一两种语言的时间", "unify-times")
        for lane in ("original", "translation"):
            self._act(unify, "time.unify_" + lane, lambda _=False, lane=lane: self._unify_lane_times(lane)).setProperty(
                "help_anchor", "unify-times")
        self._act(m, "time.autotime", self._autotime)
        self.act_dialogue_align = self._act(m, "time.dialogue_align", self._start_dialogue_align)
        self.act_dialogue_align.setProperty("help_anchor", "dialogue-align")
        m.aboutToShow.connect(self._refresh_dialogue_menu)
        self._refresh_dialogue_menu()
        m.addSeparator()
        self._build_k_menu(m)
        self.timing_menu = m.addMenu("打轴方式")
        self.timing_menu.help_anchor = "timing"
        timing_group = QActionGroup(self)
        self.timing_actions = {}
        mode = self.settings.value("wave/timing_mode", "aegisub")
        if mode not in ("aegisub", "select"):
            mode = "aegisub"
        for key, label in (("aegisub", "Aegisub 式：左键开始、右键结束（推荐）"),
                           ("select", "点块选中：左键点字幕块选中，Shift+左键开始、Shift+右键结束")):
            action = QAction(label, self, checkable=True)
            action.setProperty("help_anchor", "timing")
            action.setChecked(key == mode)
            timing_group.addAction(action)
            self.timing_menu.addAction(action)
            action.triggered.connect(lambda _=False, value=key: self._set_timing_mode(value))
            self.timing_actions[key] = action
        self.wave_panel.set_timing_mode(mode)
        m.aboutToShow.connect(self._update_group_menu_actions)
        m = self._menu(mb, "注音(&R)", "readings")
        for action_id in ("ruby.edit", "ruby.regen", "ruby.regen_all"):
            self._act(m, action_id)
        m.addSeparator()
        self.act_project_readings = self._act(m, "project.words", self._project_readings)
        self.act_project_readings.setEnabled(self.project is not None)
        m.aboutToShow.connect(lambda: self.act_project_readings.setEnabled(
            self.project is not None and self._episode_job is None))
        self._act(m, "project.ruby_settings", self._ruby_settings)
        self._act(m, "project.pinyin_settings", self._pinyin_settings)
        m = self._menu(mb, "播放(&V)", "playback")
        self._act(m, "play.toggle", self.toggle_play)
        self.act_step_playback = self._action("play.step_mode")
        self.act_step_playback.setCheckable(True)
        self.act_step_playback.setProperty("help_anchor", "playback")
        self.act_step_playback.setChecked(self.settings.value("playback/step", False, type=bool))
        self.act_step_playback.toggled.connect(self._set_step_playback)
        m.addAction(self.act_step_playback)
        self.addAction(self.act_step_playback)
        self.wave_panel.btn_step.setChecked(self.act_step_playback.isChecked())
        self.wave_panel.btn_step.toggled.connect(self.act_step_playback.setChecked)
        m.addSeparator()
        self._act(m, "play.group", self.play_group).setToolTip("提前 0.5 秒开始，组尾前一帧停")
        self._act(m, "play.s", lambda: self._play_key("S"))
        self._act(m, "play.q", lambda: self._play_key("Q"))
        self._act(m, "play.w", lambda: self._play_key("W"))
        self._act(m, "play.d", lambda: self._play_key("D"))
        self._act(m, "play.e", lambda: self._play_key("E"))
        m.addSeparator()
        for action_id in ("nav.prev", "nav.next", "play.prev_group", "play.next_group", "play.frame_back", "play.frame_forward"):
            self._act(m, action_id)
        m.addSeparator()
        follow = self._action("play.follow_list")
        follow.setCheckable(True)
        follow.setChecked(True)
        follow.toggled.connect(lambda v: setattr(self, "_follow", v))
        m.addAction(follow)
        m = self._menu(mb, "视图(&W)", "review")
        self.act_watch2 = self._act(m, "view.watch_timing", lambda: self._watch_key(True), "F10")
        self.act_watch2.setCheckable(True)
        self.act_watch = self._act(m, "view.watch_full", lambda: self._watch_key(False), "F11")
        self.act_watch.setCheckable(True)
        self.addAction(self.act_watch)
        self.addAction(self.act_watch2)
        self.act_karaoke = self._act(m, "view.karaoke", self.toggle_karaoke_mode)
        self.act_karaoke.setCheckable(True)
        self.addAction(self.act_karaoke)
        self.wave_panel.btn_karaoke.setDefaultAction(self.act_karaoke)
        m.addSeparator()
        self.layout_menu = m.addMenu("布局")
        self.layout_menu.help_anchor = "review"
        layout_group = QActionGroup(self)
        self.layout_actions = {}
        for key, label in (("video", "视频优先（推荐）"), ("list", "列表优先"), ("columns", "左右分栏")):
            action = QAction(label, self, checkable=True)
            action.setProperty("help_anchor", "karaoke" if key == "karaoke" else "review")
            layout_group.addAction(action)
            self.layout_menu.addAction(action)
            action.triggered.connect(lambda _=False, value=key: self.set_edit_layout(value))
            self.layout_actions[key] = action
        m.addAction(self._action("view.source_panel", factory=self.dock_source.toggleViewAction))
        m.addAction(self._action("view.people_panel", factory=self.people.toggleViewAction))
        self.act_docks_right = self._act(m, "view.dock_reset", self._dock_panels_right)
        self.act_docks_right.setProperty("help_anchor", "review")
        m.addSeparator()
        m = self._menu(m, "观剧模式选项", "watch")
        self.act_watch_speaker = self._action("watch.speakers")
        self.act_watch_speaker.setCheckable(True)
        self.act_watch_speaker.setChecked(self.settings.value("watch/speakers", False, type=bool))
        self.act_watch_speaker.toggled.connect(self._speaker_display_changed)
        m.addAction(self.act_watch_speaker)
        speaker_menu = self._menu(m, "说话人按", "watch")
        speaker_group = QActionGroup(self)
        self.speaker_lane_actions = {}
        for label, lane in (("原声", "original"), ("翻译", "translation")):
            action = QAction(label, self, checkable=True)
            speaker_group.addAction(action)
            speaker_menu.addAction(action)
            action.setChecked(lane == self.settings.value("watch/speaker_lane", "original"))
            action.triggered.connect(lambda _=False, value=lane: self._speaker_lane_changed(value))
            self.speaker_lane_actions[lane] = action
        self.act_watch_frames = self._action("watch.frames")
        self.act_watch_frames.setCheckable(True)
        self.act_watch_frames.setChecked(self.settings.value("watch/frames", True, type=bool))
        self.act_watch_frames.toggled.connect(self._frame_display_changed)
        m.addAction(self.act_watch_frames)
        self.act_watch_frame_color = self._action("watch.frame_color")
        self.act_watch_frame_color.triggered.connect(self._choose_frame_color)
        m.addAction(self.act_watch_frame_color)
        from .watch import FRAME_DEFAULT, ass_rgb

        self.watch.frames_visible = self.act_watch_frames.isChecked()
        saved_colour = str(self.settings.value("watch/frame_color", FRAME_DEFAULT))
        self.watch.frame_color = ass_rgb(saved_colour if QColor(saved_colour).isValid() else FRAME_DEFAULT)
        m = self._menu(mb, "显示规则(&D)", "display-rules")
        self.display_menu = m
        self.act_hide_period = self._action("display.hide_period")
        self.act_hide_period.setCheckable(True)
        self.act_hide_period.toggled.connect(self._toggle_hide_period)
        m.addAction(self.act_hide_period)
        self.zh_script_menu = m.addMenu("中文显示为")
        self.zh_script_menu.help_anchor = "readings"
        script_group = QActionGroup(self)
        self.zh_script_actions = {}
        for key, label in (("off", "原样（不转换）"), ("simplified", "简体"), ("traditional", "繁体")):
            action = QAction(label, self, checkable=True)
            action.setProperty("help_anchor", "readings")
            action.setChecked(key == "off")
            script_group.addAction(action)
            self.zh_script_menu.addAction(action)
            action.toggled.connect(lambda on, value=key: self._set_traditional("script", value) if on else None)
            self.zh_script_actions[key] = action
        self.zh_script_menu.addSeparator()
        self.zh_scope_actions = {}
        from subkit.core.display import ZH_SCOPE_DEFAULTS

        for key, label, tip in (
                ("dialogue", "对白", "中文对白行（含顶部、底部第二人）"),
                ("song", "歌词", "中文歌词行"),
                ("sign", "屏幕字等", "中文的屏幕字、注释类行"),
                ("gloss", "旁注（中文行）", "中文行里 {@注:目标=注释} 的注释文字（目标仍按正文找）"),
                ("gloss_ja", "日文行上不含假名的旁注", "日文行里的旁注，注释文字没有平假名、片假名时才转（含假名的视为日语，从不转）"),
                ("raw", "{@Raw=…} 里的文字", "中文行里 Raw 包住的文字；Raw 本来的用途是保持原样，所以默认不转")):
            action = QAction(label, self, checkable=True)
            action.setProperty("help_anchor", "readings")
            action.setToolTip(tip)
            action.setStatusTip(tip)
            action.setChecked(ZH_SCOPE_DEFAULTS[key])
            action.toggled.connect(lambda on, scope=key: self._set_zh_scope(scope, on))
            self.zh_script_menu.addAction(action)
            self.zh_scope_actions[key] = action
        self.act_traditional = self.zh_script_actions["traditional"]
        self.traditional_menu = m.addMenu("繁体用字")
        self.traditional_menu.help_anchor = "readings"
        variant_group = QActionGroup(self)
        self.traditional_actions = {}
        for key, label in (("tw", "台湾（默认）"), ("hk", "香港"), ("std", "标准繁体")):
            action = QAction(label, self, checkable=True)
            action.setChecked(key == "tw")
            variant_group.addAction(action)
            self.traditional_menu.addAction(action)
            action.triggered.connect(lambda _=False, value=key: self._set_traditional("variant", value))
            self.traditional_actions[key] = action
        self.act_traditional_phrases = QAction("同时转换用语（例：软件↔軟體）", self, checkable=True)
        self.act_traditional_phrases.toggled.connect(lambda on: self._set_traditional("phrases", on))
        self.traditional_menu.addAction(self.act_traditional_phrases)
        from subkit.core.display import MERGE_KINDS

        self.merge_menu = m.addMenu("两行合成一行（中间两个空格）")
        self.merge_menu.help_anchor = "readings"
        self.merge_actions, self.merge_lane_menus = {}, {}
        for lane in ("translation", "original"):
            menu = self.merge_menu.addMenu(lane_label(lane))
            menu.help_anchor = "readings"
            self.merge_lane_menus[lane] = menu
            for kind, label in MERGE_KINDS:
                action = QAction(label, self, checkable=True)
                action.setProperty("help_anchor", "readings")
                action.toggled.connect(lambda on, lane=lane, kind=kind: self._set_merge_lines(lane, kind, on))
                menu.addAction(action)
                self.merge_actions[lane, kind] = action
        m.addSeparator()
        self.punctuation_profiles_menu = m.addMenu("标点方案")
        self.punctuation_profiles_menu.help_anchor = "display-rules"
        self.punctuation_profiles_menu.aboutToShow.connect(self._refresh_punctuation_profiles)
        self.punctuation_profile_actions = {}
        self._punctuation_profile_group = None
        self.act_punctuation = self._act(self.punctuation_profiles_menu, "display.punctuation", self._open_punctuation_menu)
        self.act_replacements = self._act(m, "display.replacements", self._open_replacements)
        self.act_replacements.setProperty("help_anchor", "display-replacements")
        self.act_replacements.setEnabled(self.project is not None)
        m.aboutToShow.connect(lambda: self.act_replacements.setEnabled(
            self.project is not None and self._episode_job is None))
        m = self._menu(mb, "工具(&O)", "review")
        self.tools_menu = m
        check_menu = self._menu(m, "检查", "review")
        self._act(check_menu, "tools.check", self.check)
        self._act(check_menu, "tools.check_fonts", self.check_episode_fonts)
        self._act(check_menu, "tools.check_selfref", self.check_self_references)
        review_menu = self._menu(m, "审议", "review-packet")
        self._act(review_menu, "tools.export_review", self.export_review)
        self._act(review_menu, "tools.import_review", self.import_review_answers)
        self.passages_menu = self._menu(m, "重复段落", "passages")
        self._act(self.passages_menu, "tools.passage_library", self._passage_library)
        self._act(self.passages_menu, "tools.record_passage", lambda: self._record_passage(self.table.selected_gids()))
        self.act_find_passage = self._act(self.passages_menu, "tools.find_passage", self._find_passages)
        correction = self._menu(m, "原字幕修正", "add-episode")
        self.act_brackets = self._act(correction, "tools.brackets_to_text", self.brackets_to_text)
        m.addSeparator()
        self._act(m, "tools.shortcuts", self._shortcuts_dialog)
        m.addSeparator()
        m = self._menu(m, "高级", "fonts")
        self.act_private = self._act(m, "tools.private", lambda: None)
        self.act_private.setCheckable(True)
        self.act_private.setChecked(self.private_enabled)
        self.act_private.toggled.connect(self._set_private_enabled)
        self.act_private.setProperty("help_anchor", "fonts")
        action = self._act(m, "tools.font_library", self._choose_font_library)
        self._private_actions.append(action)
        self._private_actions.extend(self.batch_actions.values())
        action.setVisible(self.private_enabled)
        self._act(m, "tools.clear_cache", self._clear_layout_cache)
        self.filter_box = QComboBox()
        for x in GroupFilter.MODES:
            self.filter_box.addItem(x)
        self.filter_box.currentTextChanged.connect(self._set_group_filter)
        tb = QToolBar("主工具栏")
        tb.setObjectName("toolbar_main")
        self.addToolBar(tb)
        self.toolbar = tb
        self.ep_box = QComboBox()
        self.ep_box.setMinimumWidth(140)
        self.ep_box.currentTextChanged.connect(self._switch_episode)
        tb.addWidget(QLabel(" 集："))
        tb.addWidget(self.ep_box)
        tb.addWidget(QLabel(" 列表："))
        tb.addWidget(self.filter_box)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        groups = ((("file.save", "保存"), ("edit.undo", "撤销"), ("edit.redo", "重做")),
                  (("play.toggle", "播放 / 暂停"), ("group.confirm_next", "✓ 确认并下一组")),
                  (("view.watch_timing", "看片打轴"), ("view.watch_full", "全屏观看"),
                   ("view.karaoke", "卡拉 OK")),
                  (("view.source_panel", "原文"), ("view.people_panel", "人物")))
        self._toolbar_buttons = {}
        self._toolbar_tips = _ToolbarTips(self)
        for group in groups:
            tb.addSeparator()
            for action_id, text in group:
                action = self.action_hub.action(action_id)
                # Tool buttons show iconText and menus show text, so the short name never
                # leaks into the menus and survives Qt re-reading the action on every change.
                action.setIconText(text)
                tb.addAction(action)
                button = tb.widgetForAction(action)
                button.installEventFilter(self._toolbar_tips)
                self._toolbar_buttons[action_id] = (button, text)
                self._sync_toolbar_button(action_id)
        m = self._menu(mb, "帮助(&H)", "project")
        self.help_menu = m
        self._act(m, "help.manual", self.show_help)
        self._act(m, "help.shortcuts", lambda: app_help.open_manual("keys"))
        self._act(m, "help.changes", lambda: app_help.open_manual("changes"))
        self._act(m, "help.about", self.show_about)
        self._set_menu_tips()
        self._refresh_punctuation_profiles()
        self._update_group_menu_actions()
        self.proxy: Optional[GroupFilter] = None

    def _sync_toolbar_button(self, action_id: str) -> None:
        """Tooltip 「名字（快捷键）」 with the current key; the short name itself is the action's iconText."""
        button, text = self._toolbar_buttons[action_id]
        scene = "watch" if self.watch_mode else "main"
        keys = self.action_hub.registry.effective_keys(action_id, scene)
        button.setToolTip(text + ("（" + " / ".join(keys) + "）" if keys else ""))

    def _menu(self, parent, text: str, anchor: str):
        menu = parent.addMenu(text)
        menu.help_anchor = anchor
        menu.menuAction().setProperty("help_anchor", anchor)
        return menu

    def _build_k_menu(self, parent) -> None:
        """Mirror the K bar, retaining its signals and shared lyric/effect menus."""
        panel = self.wave_panel
        menu = self.k_menu = self._menu(parent, "K 轴", "karaoke")
        self._k_menu_controls = {}

        def command(action_id, control, target=menu, existing=None, *, shown=True):
            if existing is None:
                existing = control.defaultAction()
            action = self._action(action_id, factory=lambda: existing)
            if shown:
                target.addAction(action)
            self._k_menu_controls[action_id] = control
            keys = self.action_hub.registry.effective_keys(action_id, "k")
            # These scene keys belong to the event filter/tapper, never QAction.
            action.setProperty("enabled_text", action.text())
            if keys:
                action.setText(action.text() + "\t" + QKeySequence(keys[0]).toString())
            return action

        command("k.tap", panel.btn_k)
        # Shown inside the shared 歌词 menu below; listing it here too would repeat it.
        command("k.import", panel.btn_kimport, existing=panel.menu_actions[panel.btn_kimport], shown=False)
        auto = self.k_auto_menu = self._menu(menu, "自动 K 轴", "karaoke-auto")
        for scope, action in panel.kauto_scopes.items():
            command("k.auto_" + scope, action, auto, existing=action)
        auto.addSeparator()
        command("k.suggest_onset", panel.btn_ksuggest, auto, existing=panel.kauto_onset)
        for name, button in (("adopt", panel.btn_kadopt), ("add_rest", panel.btn_kaddrest),
                             ("remove_rest", panel.btn_kdelrest), ("clear", panel.btn_kclear),
                             ("confirm", panel.btn_kconfirm)):
            command("k." + name, button)
        menu.addSeparator()
        menu.addMenu(panel.lyrics_menu)
        menu.addMenu(panel.fx_menu)
        # 「参数…」 is the last item of the shared 特效 menu.
        command("k.parameters", panel.btn_kparams, existing=panel.menu_actions[panel.btn_kparams], shown=False)
        for name, button in (("mapping", panel.btn_kmap), ("english", panel.btn_kenglish)):
            self._action("k." + name, factory=lambda button=button: panel.menu_actions[button])
        for kind, scopes in panel.scope_actions.items():
            for scope, action in scopes.items():
                self._action(f"k.{kind}_{scope}", factory=lambda action=action: action)
        menu.aboutToShow.connect(self._sync_k_menu)
        auto.aboutToShow.connect(self._sync_k_menu)
        panel.lyrics_menu.aboutToShow.connect(self._refresh_redistribute_menu)

    def _sync_k_menu(self) -> None:
        self._flush_group_selection()
        self._refresh_auto_menu()
        self._refresh_redistribute_menu()
        group = next((g for g in self.session.doc.groups if g.id == self.current_gid), None) if self.session else None
        reason = "（先选中字幕组）" if group is None else "（不是歌曲）" if group.kind != "song" else ""
        for action_id, control in self._k_menu_controls.items():
            action = self.action_hub.action(action_id)
            spec = self.action_hub.registry.lookup(action_id)
            keys = self.action_hub.registry.effective_keys(action_id, "k")
            action.setText(spec.text + ("\t" + QKeySequence(keys[0]).toString() if keys else ""))
            action.setEnabled(not reason and getattr(control, "command_enabled", control.isEnabled()))
            tip = reason or getattr(control, "command_tooltip", control.toolTip())
            action.setProperty("shortcut_tooltip_template", None)
            action.setToolTip(tip)
            # Shared auto actions get their current reason from the refreshed tooltip.
            status = getattr(control, "command_status", control.statusTip() if control is not action else "")
            action.setStatusTip(reason or status or tip or spec.text)
        for menu, control in ((self.k_auto_menu, self.wave_panel.btn_ksuggest),
                              (self.wave_panel.lyrics_menu, self.wave_panel.btn_more),
                              (self.wave_panel.fx_menu, self.wave_panel.k_fx)):
            menu.setEnabled(not reason and control.isEnabled())
            menu.menuAction().setToolTip(reason or control.toolTip())
            menu.menuAction().setStatusTip(reason or control.statusTip() or control.toolTip() or menu.title())

    def _hidden_menu_for_key(self, event):
        """Only hidden watch menus need a dispatcher; native mnemonics handle the rest."""
        if not self.watch_mode or not self.menuBar().isHidden() or event.modifiers() != Qt.KeyboardModifier.AltModifier:
            return None
        if self.watch.transform is not None or event.key() == Qt.Key.Key_T:
            return None                  # Alt+T never opens the timeline in watch mode
        registry = self.action_hub.registry
        if any(spec.id != "watch.edit_menu" and registry.matches_event(spec.id, "watch", event) for spec in registry):
            return None                  # existing scene/window bindings always win
        for action in self.menuBar().actions():
            mnemonic = re.search(r"\(&([A-Z])\)", action.text())
            if mnemonic and event.key() == ord(mnemonic[1]) and action.isEnabled():
                return action.menu()
        return None

    def _listen_selected(self) -> None:
        gids = self.table.selected_gids()
        if gids:
            self._start_listening(gids[0])

    def _toggle_review_flag(self, flag: str) -> None:
        gids = self.table.selected_gids()
        if self.session and gids:
            value = not all(flag in self.session.doc.group(gid).flags for gid in gids)
            self._review_flag(gids, flag, value)
            self._update_group_menu_actions()

    def _update_group_menu_actions(self) -> None:
        gids = self.table.selected_gids()
        groups = [self.session.doc.group(gid) for gid in gids] if self.session else []
        reason = "先打开一集" if not self.session else "" if gids else "先选中字幕组"
        for action_id in ("group.duplicate", "group.unconfirm", "group.hide", "group.mark", "group.mark_note"):
            set_action_reason(self.action_hub.action(action_id), reason)
        set_action_reason(self.action_hub.action("time.set"), reason or ("只能对一组设置" if len(gids) != 1 else ""))
        for action_id, flag in (("group.hide", "hidden"), ("group.mark", "marked")):
            self.action_hub.action(action_id).setChecked(bool(groups) and all(flag in g.flags for g in groups))
        languages = self.project.languages if self.project else {}
        for lane in ("original", "translation"):
            action = self.action_hub.action("group.add_" + lane)
            label = lane_label(lane, languages)
            spec = self.action_hub.registry.lookup("group.add_" + lane)
            action.setProperty("enabled_text", spec.text.format(语言=language_name(languages.get(lane, DEFAULT_LANGUAGES[lane]))))
            missing = any(not any(m.lane == lane and m.role != "comment" for m in self.session.doc.members_of(g))
                          for g in groups)
            set_action_reason(action, reason or ("" if missing else f"选中的组都已经有{label}"))
        from subkit.core.listening import unavailable_reason
        set_action_reason(self.action_hub.action("group.ai_listen"), reason or
                          unavailable_reason(self.session.doc, gids, languages))

    def _shortcuts_dialog(self) -> None:
        from .views.shortcuts_dialog import ShortcutsDialog

        prefixes = ShortcutsDialog.menu_names(self.menuBar())
        languages = self.project.languages if self.project else {}

        def text(spec):                   # 补上原声（{语言}）… shows the project's language, as in the menu
            lane = {"group.add_original": "original", "group.add_translation": "translation"}.get(spec.id)
            return spec.text.format(语言=language_name(languages.get(lane, DEFAULT_LANGUAGES[lane]))) if lane else spec.text

        names = {spec.id: " → ".join(filter(None, (prefixes.get(spec.id), text(spec))))
                 for spec in self.action_hub.registry}
        dialog = ShortcutsDialog(self.action_hub.registry, self, names=names)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._apply_shortcut_overrides(dialog.registry)
        finally:
            dialog.deleteLater()

    def _apply_shortcut_overrides(self, edited: ActionRegistry) -> None:
        registry = self.action_hub.registry
        registry.apply_overrides(edited)
        registry.save(self.settings)
        self.settings.sync()
        self._refresh_shortcuts()

    def _refresh_shortcuts(self) -> None:
        registry = self.action_hub.registry
        scene = "watch" if self.watch_mode else "main"
        self.action_hub.refresh(scene)
        controls = [("group." + name, button) for name, button in zip(
            ("head_to_prev", "tail_to_next", "take_from_prev", "take_from_next",
             "merge", "split", "unlink", "confirm_next"), self.editor.action_buttons)]
        controls.extend(self._k_menu_controls.items())
        controls.extend((("play.step_mode", self.wave_panel.btn_step),
                         ("time.adopt", self.wave_panel.btn_adopt_time),
                         ("view.karaoke", self.wave_panel.btn_karaoke)))
        for action_id, control in controls:
            self._shortcut_tooltip(control, action_id, scene)
        self._sync_k_menu()
        self._update_transform_action()
        for spec in registry:
            self._shortcut_tooltip(self.action_hub.action(spec.id), spec.id, scene)
        for action_id in self._toolbar_buttons:
            self._sync_toolbar_button(action_id)

    def _shortcut_tooltip(self, control, action_id: str, scene: str) -> None:
        spec = self.action_hub.registry.lookup(action_id)
        scene = "k" if "k" in spec.keys else scene
        keys = self.action_hub.registry.effective_keys(action_id, scene)
        template = control.property("shortcut_tooltip_template")
        if template is None:
            template = control.toolTip()
            control.setProperty("shortcut_tooltip_template", template)
        text = template
        defaults = spec.keys.get(scene, spec.keys.get("main", ()))
        replacements = {}
        for key in defaults:
            for variant in (key, key.replace("Up", "↑").replace("Down", "↓"),
                            "Delete" if key == "Del" else key):
                replacements[variant] = " / ".join(keys)
        if replacements:
            pattern = r"(?<![A-Za-z+])(?:" + "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True)) + r")(?![A-Za-z+]| 轴)"
            text = re.sub(pattern, lambda match: replacements[match.group()], text)
        if keys and not any(key in text for key in keys):
            text += "（" + " / ".join(keys) + "）"
        control.setToolTip(text)

    def _action(self, action_id: str, *, factory=None) -> QAction:
        action = self.action_hub.action(action_id, factory=factory)
        spec = self.action_hub.registry.lookup(action_id)
        action.setText(spec.text)
        action.setProperty("help_anchor", spec.help_anchor)
        return action

    def _act(self, menu, action_id: str, fn=None, shortcut=None) -> QAction:
        a = self._action(action_id)
        if fn is not None:
            a.triggered.connect(fn)
            a.setProperty("menu_handler", True)
        if shortcut is not None:
            self.action_hub.bind_shortcuts(action_id)
        menu.addAction(a)
        if fn in (self.new_project, self.open_project_dialog, self.add_episode, self.remove_episode, self.merge, self.reanalyze,
                  self.change_template, self._passage_library, self._pinyin_settings, self._ruby_settings,
                  self.save, self.import_review_answers, self._autotime):
            if fn in (self.add_episode, self.merge, self.reanalyze):
                a.setProperty("help_anchor", "add-episode")
            self._episode_actions.append(a)
        return a

    def _build_shortcuts(self) -> None:
        def bind(action_id, fn):
            action = self.action_hub.action(action_id)
            self.action_hub.bind_shortcuts(action_id, lambda _key: fn(),
                                           connect_triggered=not action.property("menu_handler"))

        # Menu clicks keep their existing handlers; Space/R have distinct
        # keyboard semantics, including their input-field guards.
        bind("play.toggle", lambda: self._space())
        bind("play.toggle_plain", lambda: self.toggle_play())
        bind("play.group", lambda: self._key_r())
        bind("play.group_typing", lambda: self.play_group())
        bind("group.confirm_next", lambda: self.confirm_next())
        bind("group.confirm", lambda: self._confirm(move=False))
        for name in ("head_to_prev", "tail_to_next", "take_from_prev", "take_from_next"):
            bind("group." + name, lambda name=name: self._move(name))
        bind("group.cursor_to_prev", lambda: self._move_at_cursor(True))
        bind("group.cursor_to_next", lambda: self._move_at_cursor(False))
        for name in ("merge", "split", "unlink"):
            bind("group." + name, lambda name=name: self._on_action(name))
        bind("group.delete", lambda: self.delete_selected())
        bind("edit.undo", lambda: self.undo())
        bind("edit.redo", lambda: self.redo())
        bind("time.adopt", lambda: None if self._typing() else self._adopt_timing())
        bind("nav.prev", lambda: self._step_group(-1))
        bind("nav.next", lambda: self._step_group(+1))
        for edge, direction, name in (("start", -1, "start_minus"), ("start", 1, "start_plus"),
                                      ("end", -1, "end_minus"), ("end", 1, "end_plus")):
            bind("time." + name, lambda edge=edge, direction=direction: self._nudge(edge, direction))
        for key in "SQWDE":
            bind("play." + key.lower(), lambda key=key: self._play_key(key))
        bind("ruby.edit", lambda: self._f2())
        bind("ruby.regen", lambda: self._regen_current(False))
        bind("ruby.regen_all", lambda: self._regen_current(True))
        bind("play.prev_group", lambda: self._play_neighbour(-1))
        bind("play.next_group", lambda: self._play_neighbour(+1))
        bind("play.frame_back", lambda: self._frame(False))
        bind("play.frame_forward", lambda: self._frame(True))
        # Alt+E is handled with the other hidden-menu mnemonics below.
        self.action_hub.action("watch.edit_menu").setEnabled(False)
        bind("watch.exit", lambda: self.set_watch_mode(False))
        bind("watch.escape", lambda: self._escape())
        bind("watch.volume_up", lambda: self.player.volume_step(+5))
        bind("watch.volume_down", lambda: self.player.volume_step(-5))
        bind("group.duplicate", lambda: self._duplicate_groups(self.table.selected_gids()))
        bind("time.set", lambda: self._group_times_dialog(self.table.selected_gids()))
        bind("time.shift", lambda: self._shift_groups_dialog(self.table.selected_gids()))
        bind("group.hide", lambda: self._toggle_review_flag("hidden"))
        bind("group.mark", lambda: self._toggle_review_flag("marked"))
        bind("file.export", lambda: self.export())
        for action_id in ("view.karaoke", "play.step_mode", "help.manual"):
            self.action_hub.bind_shortcuts(action_id)
        # Compatibility names for existing callers; these are shared QActions.
        # (Alt+E and the other hidden-menu mnemonics go through _hidden_menu_for_key.)
        for name, action_id in (("sc_tab", "watch.exit"), ("sc_esc", "watch.escape"), ("sc_up", "watch.volume_up"),
                                ("sc_down", "watch.volume_down")):
            setattr(self, name, self.action_hub.action(action_id))
        self.action_hub.set_scene("main")

    def _typing(self) -> bool:
        w = QApplication.focusWidget()
        return isinstance(w, (QPlainTextEdit, QTextEdit, QLineEdit))

    def _update_shift_action(self) -> None:
        reason = "先打开一集" if not self.session else "" if self.table.selected_gids() else "先选中一组"
        set_action_reason(self.act_shift_groups, reason)

    def _shift_groups_dialog(self, gids: list[str]) -> None:
        self._flush_group_selection()
        if not self.session or not gids:
            return
        dialog = ShiftGroupsDialog(self.session.doc, gids, self.settings, self)
        try:
            if dialog.exec() == dialog.DialogCode.Accepted:
                self._edit(lambda: self.session.editor.shift_groups(gids, dialog.delta()))
        finally:
            dialog.deleteLater()

    def _duplicate_groups(self, gids: list[str]) -> None:
        self._flush_group_selection()
        if not self.session or not gids:
            return
        created = []
        if self._edit(lambda: created.extend(self.session.editor.duplicate_groups(gids))) and created:
            index = self.proxy.mapFromSource(self.model.index(self.model.row_of(created[0]), 0))
            if not index.isValid():  # An unmarked/unhidden copy is absent from those filters.
                self.filter_box.setCurrentText("全部")
            self.select_group(created[0], seek=False)
            self.table.select_gids(created)
            self.statusBar().showMessage(f"已复制 {len(created)} 组（Ctrl+Z 撤销）", 6000)

    def _group_times_dialog(self, gids: list[str]) -> None:
        self._flush_group_selection()
        if not self.session or len(gids) != 1:
            return
        session, gid = self.session, gids[0]
        dialog = GroupTimesDialog(session.doc, gid, self)
        try:
            if dialog.exec() != dialog.DialogCode.Accepted or self.session is not session:
                return
            start, end, target, shift = dialog.values()
            if self._edit(lambda: session.editor.set_exact_group_time(gid, start, end, target=target, shift_beats=shift)):
                issues = [issue for m in session.doc.members_of(session.doc.group(gid))
                          if target == "group" or m.lane == target
                          for issue in kara.effective(session.doc, m).issues if issue == "有拍落在这一行的时间外"]
                self.statusBar().showMessage("已设置时间（Ctrl+Z 撤销）" + ("；" + issues[0] if issues else ""), 7000)
        finally:
            dialog.deleteLater()

    def _update_transform_action(self) -> None:
        ready = bool(self.watch_mode and self.session and self.watch.selected)
        keys = self.action_hub.registry.effective_keys("watch.transform", "watch")
        self.act_transform.setShortcuts([QKeySequence(key) for key in keys])
        self.act_transform.setProperty("enabled_text", self.action_hub.registry.lookup("watch.transform").text)
        set_action_reason(self.act_transform, "" if ready else "先在观剧模式里点选一行字幕")

    def _start_transform(self, *, only_line: bool = False) -> None:
        self._flush_group_selection()
        if self.watch_mode and self.session and self.watch.selected and not self._typing():
            self.watch.enter_transform(only_line=only_line)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            navigation = (event.key() in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown)
                          or (obj is self.table and event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down)))
            if navigation and not self._typing() and isinstance(obj, QWidget) and app_help.belongs_to(obj, self):
                if event.type() == QEvent.Type.KeyRelease:
                    if not event.isAutoRepeat():
                        self._group_nav_keys.discard(event.key())
                else:
                    self._group_nav_keys.add(event.key())
                    self._group_nav_last = time.monotonic()
                if self._pending_group is not None:
                    self._group_select_timer.start()
        if event.type() in (QEvent.Type.WindowDeactivate, QEvent.Type.FocusOut, QEvent.Type.MouseButtonPress):
            self._group_nav_keys.clear()       # a click means no navigation key is being held
        if event.type() == QEvent.Type.FocusIn and isinstance(obj, (QPlainTextEdit, QTextEdit, QLineEdit)):
            row = next((i for i, (_mid, edit) in enumerate(self.editor.cn_edits) if edit is obj), None)
            self._flush_group_selection()
            if row is not None and row < len(self.editor.cn_edits):
                self.editor.cn_edits[row][1].setFocus()
        if (event.type() == QEvent.Type.Show
                and ((obj is self.editor and self._editor_pending) or (obj is self.source and self._source_pending))
                and self.session is not None and self.current_gid):
            self._refresh_group_panels(self.session.doc.group(self.current_gid))
        if event.type() == QEvent.Type.Show and isinstance(obj, (ReadingPopup, InlineTextEditor)):
            self._flush_group_selection()
        if event.type() == QEvent.Type.KeyPress:
            self._last_key_time = time.monotonic()        # "typing" for the deferred episode prompt
        if event.type() == QEvent.Type.Show and isinstance(obj, QMessageBox):
            anchor = getattr(self, "_help_message_anchor", None) or app_help._anchor(obj.parentWidget())
            if not hasattr(obj, "_button_row"):
                obj.help_anchor = getattr(obj, "help_anchor", None) or anchor or "project"
                app_help.dialog_buttons(obj)
        if (event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
                and event.key() == Qt.Key.Key_F1 and event.modifiers() == Qt.KeyboardModifier.NoModifier):
            if (isinstance(obj, QWidget) and app_help.belongs_to(obj, self)
                    and not obj.property("shortcut_recording")):
                if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                    if getattr(obj, "copy_syntax_help", False):
                        app_help.open_manual(obj.help_anchor)
                    else:
                        self.show_help()
                event.accept()
                return True
        if (event.type() == QEvent.Type.ShortcutOverride and event.key() == Qt.Key.Key_Escape
                and self._inline is not None and isinstance(obj, QWidget)
                and app_help.belongs_to(obj, self) and obj.window() not in (self, self._inline)):
            event.accept()                  # reading tools/dropdowns close before the editor
            return True
        if (event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
                and isinstance(obj, QWidget) and app_help.belongs_to(obj, self)
                and QApplication.activeModalWidget() is None and QApplication.activePopupWidget() is None
                and obj.window() in (self, self._inline)):
            key, mods = event.key(), event.modifiers()
            hidden_menu = self._hidden_menu_for_key(event)
            if hidden_menu is not None:
                if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                    hidden_menu.popup(self.player.mapToGlobal(QPoint(8, 8)))
                event.accept()
                return True
            if key == Qt.Key.Key_Escape and self._inline is not None and self._inline.isVisible():
                if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                    self._inline.cancel()
                event.accept()               # beats the window-level Escape shortcut
                return True
            translation_key = self.action_hub.registry.matches_event("watch.edit_translation", "watch", event)
            original_key = self.action_hub.registry.matches_event("watch.edit_original", "watch", event)
            if (self.watch_mode and (translation_key or original_key)
                    and not self._typing()
                    and self.watch.transform is None and self._export_job is None):
                if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                    self._watch_inline_lane("translation" if translation_key else "original")
                event.accept()
                return True
        if self._export_job is not None and event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return super().eventFilter(obj, event)
        if event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress) and hasattr(self, "watch"):
            focus = QApplication.focusWidget()
            belongs = ((focus is not None and (focus is self or self.isAncestorOf(focus)))
                       or (isinstance(obj, QWidget) and (obj is self or self.isAncestorOf(obj))))
            if belongs and not self._typing():
                key, mods = event.key(), event.modifiers()
                add_rest_key = self.action_hub.registry.matches_event("k.add_rest", "k", event)
                rest_key = (add_rest_key
                            or (self.action_hub.registry.matches_event("k.remove_rest", "k", event)
                                and self.wave.selected_rest() is not None))
                if (rest_key and self.watch.transform is None and not self.tapper.active
                        and QApplication.activeModalWidget() is None and QApplication.activePopupWidget() is None):
                    if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                        self._k_action("add_rest" if add_rest_key else "remove_rest")
                    event.accept()
                    return True
                if (self.wave.k_suggestions is not None and self.watch.transform is None
                        and not self.tapper.active and QApplication.activeModalWidget() is None
                        and QApplication.activePopupWidget() is None
                        and (self.action_hub.registry.matches_event("k.adopt", "k", event)
                             or self.action_hub.registry.matches_event("k.cancel", "k", event))):
                    if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                        if self.action_hub.registry.matches_event("k.cancel", "k", event):
                            self.wave.clear_suggestions()
                        else:
                            self._k_action("adopt")
                    event.accept()
                    return True
                if (self.watch.transform is None and not self.tapper.active
                        and QApplication.activeModalWidget() is None and QApplication.activePopupWidget() is None
                        and self.session and self.current_gid
                        and self.session.doc.group(self.current_gid).kind == "song"):
                    mode_ids = {"k.tap", "k.add_rest", "k.remove_rest", "k.adopt", "k.cancel"}
                    for spec in self.action_hub.registry:
                        if ("k" in spec.keys and spec.id not in mode_ids
                                and self.action_hub.registry.matches_event(spec.id, "k", event)):
                            self._sync_k_menu()
                            action = self.action_hub.action(spec.id)
                            if action.isEnabled():
                                if event.type() == QEvent.Type.KeyPress and not event.isAutoRepeat():
                                    action.trigger()
                                event.accept()
                                return True
                transform_line_key = self.action_hub.registry.matches_event("watch.transform_line", "watch", event)
                entering = (self.watch_mode and self.watch.selected
                            and (self.action_hub.registry.matches_event("watch.transform", "watch", event)
                                 or transform_line_key)
                            and QApplication.activeModalWidget() is None and QApplication.activePopupWidget() is None)
                if self.watch.transform is not None or entering:
                    if event.type() == QEvent.Type.ShortcutOverride:
                        event.accept()
                        return True
                    if self.watch.transform is not None:
                        if key == Qt.Key.Key_Tab:
                            self.set_watch_mode(False)
                            return True
                        if key in (Qt.Key.Key_F10, Qt.Key.Key_F11):
                            self._watch_key(key == Qt.Key.Key_F10)
                            return True
                        return self.watch.transform_key(key, mods)
                    self._start_transform(only_line=transform_line_key)
                    return True
        return super().eventFilter(obj, event)

    def show_help(self) -> None:
        app_help.open_manual(app_help.resolve_anchor(self))

    def show_about(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("关于双语字幕工坊")
        box.setText(f"双语字幕工坊（DuoSub）{APP_VERSION}\n"
                    "原名 SubKit。原生双语字幕编辑器：配对、调轴、读音、K 轴。\n"
                    f"使用说明：{app_help.manual_path()}")
        box.help_anchor = "project"
        box.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
        app_help.dialog_buttons(box)
        box.exec()

    def _help_message(self, anchor, fn, *args, **kwargs):
        previous = getattr(self, "_help_message_anchor", None)
        self._help_message_anchor = anchor
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._help_message_anchor = previous

    def _set_menu_tips(self) -> None:
        tips = {
            "导出标记和隐藏的句子": "保存当前集，并导出工程所有集的标记和隐藏句子为 HTML 与 CSV 复查清单。",
            "只看已标记": "只显示标记待复查的组。",
            "只看已隐藏": "只显示不预览、不导出的隐藏组。",
            "文件": "新建、打开、保存工程并管理模板和导出。",
            "编辑": "编辑字幕组、文字、歌曲类别和位置。",
            "集": "管理本集的原字幕、合并、检查和重复段落。",
            "从工程中移除这一集": "将本集工作文件移到工程内的已移除文件夹，原视频和原字幕不受影响。",
            "字体库位置": "选择本机 assfonts 字体库，缺字体时显示库内文件位置。",
            "自动导入整季": "把一季字幕导入当前工程，沿用工程模板、加振假名并导出。",
            "私有功能": "显示或隐藏字体库定位等本机私有功能。",
            "清除排版缓存": "清除工程排版测量，预览会重新测量。",
            "播放": "播放、试听字幕并确认下一组。",
            "视图": "切换编辑布局、观剧模式、原文面板和显示规则。",
            "显示规则": "只影响预览和导出，原字幕与工作文档不变。",
            "批量替换": "管理当前工程的正文替换列表，只影响显示和导出；F1 查看批量替换说明。",
            "标点方案": "切换已存标点方案；F1 查看显示规则。",
            "两行合成一行": "按语言和内容选择，把硬换行显示为两个空格。",
            "原声": "设置原声行的合成一行规则。",
            "翻译": "设置翻译行的合成一行规则。",
            "对白": "对白、顶部对白和同时说话的行合成一行。",
            "歌词": "歌词行合成一行。",
            "屏幕字等": "屏幕字、说明和其他保留行合成一行。",
            "布局": "选择编辑布局；每种布局分别记住分隔条位置。",
            "打轴方式": "选择波形的打轴和选组方式，选择会记住。",
            "Aegisub 式：左键开始、右键结束": "左键设开始，右键设结束；点击上下色条选组并暂停跳转。",
            "点块选中：左键点字幕块选中，Shift+左键开始、Shift+右键结束": "左键点时间范围选组；Shift 配合左右键打轴，Alt 临时不吸附。",
            "视频优先": "视频和波形在左上，组编辑区在右上，下方是全宽列表。",
            "列表优先": "视频在左上，波形和组编辑区在右上，列表显示更多行。",
            "左右分栏": "左栏视频和列表，右栏波形、组编辑区及原文、人物标签页。",
            "帮助": "打开使用说明、快捷键和版本信息。",
            "新建工程": "选择工程文件夹、语言和样式模板，建立工程。",
            "打开工程": "打开已有工程继续编辑字幕。",
            "样式模板：字体、颜色和位置": "调整字体、颜色和位置并预览发布画面。",
            "重复段落库": "管理本工程记录的重复段落和版本。",
            "拼音设置": "选择中文行的拼音、注音符号或威妥玛显示方式。",
            "振假名设置": "分别限制台词和歌词的振假名宽度，预览与导出同步生效。",
            "工程词语与分词规则": "统一搜索和编辑本工程的特殊读音、合词与固定拆词，保留各处人工修改。",
            "保存": "保存当前工作文档和人工编辑。",
            "关闭或换集时询问是否保存": "有未保存的改动时询问保存、不保存或取消；不勾选则直接保存。",
            "导出发布用 ASS": "检查并导出发布字幕和导出报告。",
            "导出整季发布用 ASS": "选输出文件夹和语言后缀导出整季 ASS，可同时导出给大模型复查的审议包；同名 ASS 先询问。",
            "退出": "保存窗口布局并关闭程序。",
            "撤销": "撤销上一步编辑。", "重做": "重新执行刚撤销的编辑。",
            "自由变换": "在观剧画面拖动手柄调整字幕变形。",
            "整体平移": "把选中组的所有行和 K 轴一起提前或推后，一步撤销。",
            "开头一段 → 上一组": "把当前语言的开头一段移到上一组。",
            "结尾一段 → 下一组": "把当前语言的结尾一段移到下一组。",
            "拿回上一组结尾一段": "把上一组结尾一段拿回当前组。",
            "拿回下一组开头一段": "把下一组开头一段拿回当前组。",
            "与下一组合并": "把当前组和下一组合为多行组。",
            "在光标处拆分": "按当前文字光标位置拆分字幕组。",
            "解除配对": "解除当前组中原声和翻译的配对。",
            "删除这一行": "删除当前语言这一行并保留删除记录。",
            "删除选中的组": "删除列表选中的组及其两种语言。",
            "转为歌曲": "选择 OP、ED 或插曲，把选中组转为歌曲。",
            "转为普通台词": "把歌曲或屏幕字转为普通台词；屏幕字使用对白样式和位置。",
            "转为屏幕字": "把对白转为屏幕字，恢复来源样式和位置。",
            "位置": "调整选中组的左右、顶部、底部或原位置。",
            "左侧说话人": "把选中组放到模板的左侧位置。",
            "右侧说话人": "把选中组放到模板的右侧位置。",
            "移到顶部": "把适用的字幕组切换为顶部样式。",
            "移到底部": "把适用的字幕组切换为底部样式。",
            "底部第二人": "把适用的字幕组切换为第二人样式：底部居中，在普通底部字幕上方。",
            "恢复原位置": "恢复原字幕自带的位置或模板默认位置。",
            "添加一集": "选择视频和两种字幕并确认字幕角色。",
            "合并": "按来源顺序配对原声和翻译，生成工作文档。",
            "重新分析原声字幕": "按新规则重新分析原声字幕，保留原文件。",
            "原字幕副本": "检查工程里保存的原字幕副本，找不到或被改过时用一字节不差的文件放回。",
            "视频文件位置": "视频挪过位置或改过名字时重新指定，可按文件夹自动配对整季。",
            "原字幕括号里的片假名不是注音：放回正文": "把误作注音的括号片假名恢复为正文。",
            "检查": "检查字幕顺序、原文去向和待办。",
            "检查字体": "检查本机缺少的字体和无法显示的字。",
            "检查自称": "按人物表检查自称读音，试听并逐处修正或保留。",
            "导出审议包": "导出中日对应、读音和人物表供发布前复查。",
            "导回审议意见": "选文件或粘贴审议意见，逐条核对并决定是否采用。",
            "记为重复段落": "把选中段落的时间、读音和样式记入工程库。",
            "查找并套用重复段落": "按音频查找重复段落并逐项确认套用。",
            "自动调轴": "按视频音频校准本集时间，保护确认过和手动调过的行；一步撤销。",
            "统一两种语言的时间": "对白统一用某一份字幕文件里的时间；没配对的行按附近的时间差挪；一步撤销。",
            "用原声字幕的时间": "对白回到原声字幕文件里的时间，翻译跟着；只在原声字幕和视频同步时用。",
            "用翻译字幕的时间": "对白用翻译字幕文件里的时间（合并时的默认），原声跟着。",
            "精确对话轴": "按日文台词分析本集，缓存对齐结果供提示线和 C 键采用；分析不改字幕，可取消。",
            "逐段播放": "空格播到本组结尾或下一组完全显示时停下；Ctrl+Shift+Space 开关。",
            "播放 / 暂停": "切换视频的播放和暂停。",
            "重播本组，提前 0.5 秒、组尾前一帧停": "从本组开始前 0.5 秒重播，组尾前一帧停下并保持选中。",
            "播放本组": "试听当前组的完整时间范围。",
            "播放开始前 0.5 秒": "试听当前组开始前的半秒音频。",
            "播放结束后 0.5 秒": "试听当前组结束后的半秒音频。",
            "播放开头 0.5 秒": "试听当前组开头的半秒音频。",
            "播放结尾 0.5 秒": "试听当前组结尾的半秒音频。",
            "确认并播放下一组": "确认当前组并播放下一组。",
            "播放时列表跟随": "播放时自动把列表滚动到当前字幕组。",
            "观剧模式：全屏观看": "全屏观看，点选字幕后显示波形。",
            "观剧模式：看片打轴": "同时显示视频和波形进行打轴。",
            "原文": "显示或隐藏原字幕来源面板。",
            "人物": "显示或隐藏工程人物表和待确认的人名。",
            "观剧模式显示说话人": "在观剧画面中显示说话人，点击可指定人物。",
            "暂停时显示字幕外框": "观剧模式暂停时，在选中和鼠标指着的字幕周围画框。",
            "字幕外框颜色": "选择观剧模式暂停时选中字幕外框的颜色。",
            "卡拉 OK": "F9 切换卡拉 OK 布局，只看歌曲；点拍播放，再按一次返回原布局。",
            "说话人按：原声 / 翻译": "选择说话人名字使用的语言。",
            "原文、人物面板放回右侧": "把原文、人物面板打开并停靠回当前布局的右侧。",
            "原声": "说话人名字按原声语言显示。",
            "翻译": "说话人名字按翻译语言显示。",
            "隐藏日文句末「。」": "在预览和导出中隐藏日文句末句号。",
            "中文显示为": "只影响预览和导出，工程原文不变",
            "原样": "只影响预览和导出，工程原文不变",
            "简体": "只影响预览和导出，工程原文不变",
            "繁体": "只影响预览和导出，工程原文不变",
            "繁体用字": "选择繁体的地区用字；显示为简体时按该地区的原文用语转换。",
            "台湾": "使用台湾用字（默认）。",
            "香港": "使用香港用字。",
            "标准繁体": "使用标准繁体。",
            "同时转换用语": "同时转换两岸用语，例如软件↔軟體。",
            "使用说明": "打开当前操作对应的使用说明章节。",
            "快捷键一览": "打开使用说明中的完整快捷键一览。",
            "版本记录": "打开使用说明中的版本记录。",
            "关于双语字幕工坊": "查看程序版本和使用说明路径。",
        }

        for name in ("标点和空格", "逗号", "换成两个空格", "换成「、」", "顿号「、」→ 两个空格", "分隔号统一为「・」", "句号 → 两个空格", "空格统一为两个半角空格",
                     "省略号", "感叹号", "问号", "引号", "不改", "统一为「…」", "统一为「……」",
                     "保留个数，换成「…」", "全角「！」", "半角「!」", "全角「？」", "半角「?」",
                     "统一为「」", "统一为『』", "外层「」、内层『』", "外层『』、内层「」", "标点字体"):
            tips[name] = "设置标点和空格的显示方式，只影响预览和导出；F1 查看显示规则。"

        self._menu_refs = []
        self._menu_action_refs = []

        def visit(menu):
            if menu is not self.menuBar():
                menu.setToolTipsVisible(True)
            for action in menu.actions():
                self._menu_action_refs.append(action)
                if action.isSeparator():
                    continue
                name = re.sub(r"\(&.\)", "", (action.property("enabled_text") or action.text()).split("\t")[0]).rstrip("…")
                name = re.sub(r"（[^（）]*）$", "", name)
                if action not in self.zh_scope_actions.values():
                    action.setStatusTip(tips.get(name, action.property("enabled_text") or action.text()))
                submenu = action.menu()
                if not action.property("help_anchor"):
                    action.setProperty("help_anchor", getattr(submenu, "help_anchor", getattr(menu, "help_anchor", "review")))
                if submenu is not None:
                    self._menu_refs.append(submenu)
                    visit(submenu)

        visit(self.menuBar())

    # ================================================================ edit layouts
    def _set_group_filter(self, mode: str) -> None:
        if self.proxy is None:
            return
        blocked = self.table.blockSignals(True)
        try:
            self.proxy.set_mode(mode)
        finally:
            self.table.blockSignals(blocked)

    def toggle_karaoke_mode(self) -> None:
        self.set_edit_layout(self._karaoke_previous_layout if self.edit_layout == "karaoke" else "karaoke")

    def _edit_splitters(self):
        return (("main", self.main_split), ("top", self.top_split),
                ("right", self.right_split), ("video", self.video_split))

    def _remember_edit_layout(self) -> None:
        if self.edit_layout is None or self.watch_mode:
            return
        prefix = "layouts/" + self.edit_layout + "/"
        self.settings.setValue(prefix + "sizes", json.dumps({k: w.sizes() for k, w in self._edit_splitters()}))
        self.settings.setValue(prefix + "state", self.saveState())
        self.settings.setValue(prefix + "host_state", self.right_host.saveState())
        self.settings.setValue(prefix + "source", not self.dock_source.isHidden())
        self.settings.setValue(prefix + "people", not self.people.isHidden())

    def set_edit_layout(self, layout: str, *, initial: bool = False, restore: bool = True) -> None:
        if layout not in (*self.layout_actions, "karaoke"):
            layout = "video"
        if self.watch_mode:
            self.set_watch_mode(False)
        if not initial and layout == self.edit_layout:
            return
        self._remember_edit_layout()
        karaoke = layout == "karaoke"
        was_karaoke = self.edit_layout == "karaoke"
        if karaoke and not was_karaoke:
            if not initial:
                self._karaoke_previous_layout = self.edit_layout or "video"
                self.settings.setValue("karaoke/previous_layout", self._karaoke_previous_layout)
            self._karaoke_filter = (self.settings.value("karaoke/previous_filter", "全部") if initial
                                    else self.filter_box.currentText())
            self.settings.setValue("karaoke/previous_filter", self._karaoke_filter)
        self.table.blockSignals(True)
        if karaoke or was_karaoke:
            self.filter_box.setCurrentText("只看歌曲" if karaoke else self._karaoke_filter)
            if self.proxy:
                self.proxy.set_mode(self.filter_box.currentText())
        self.table.blockSignals(False)
        self.filter_box.setEnabled(not karaoke)
        self.wave_panel.set_karaoke_mode(karaoke)
        self.editor.set_karaoke_mode(karaoke)
        for dock in (self.dock_source, self.people):
            dock.toggleViewAction().setEnabled(not karaoke)
        self.act_docks_right.setEnabled(not karaoke)
        # Keep the native player's immediate parent unchanged while rearranging its containers.
        for widget in (self.video_box, self.table, self.right_host, self.top_split):
            widget.setParent(self)
        columns = layout in ("columns", "karaoke")
        right_docks = layout in ("video", "columns")
        self.main_split.setOrientation(Qt.Orientation.Horizontal if columns else Qt.Orientation.Vertical)
        self.top_split.setOrientation(Qt.Orientation.Vertical if columns else Qt.Orientation.Horizontal)
        self.top_split.addWidget(self.video_box)
        self.top_split.addWidget(self.table if columns else self.right_host)
        self.main_split.addWidget(self.top_split)
        self.main_split.addWidget(self.right_host if columns else self.table)
        (self.video_split if layout == "video" else self.right_split).insertWidget(
            1 if layout == "video" else 0, self.wave_panel)
        floating = {dock: (dock.isFloating(), dock.geometry()) for dock in (self.dock_source, self.people)}
        for dock in (self.dock_source, self.people):
            owner = self.right_host if dock.parentWidget() is self.right_host else self
            owner.removeDockWidget(dock)
            host = self.right_host if right_docks else self
            host.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea if right_docks
                               else Qt.DockWidgetArea.RightDockWidgetArea, dock)
            dock.setFloating(False)
        if right_docks:
            self.right_host.tabifyDockWidget(self.dock_source, self.people)
            self.right_host.setTabPosition(Qt.DockWidgetArea.BottomDockWidgetArea,
                                          QTabWidget.TabPosition.North)
        self.edit_layout = layout
        self.settings.setValue("layouts/current", layout)
        self.act_karaoke.setChecked(karaoke)
        for key, action in self.layout_actions.items():
            group = action.actionGroup()
            group.setExclusive(False)
            action.setChecked(key == layout)
            group.setExclusive(True)
        prefix = "layouts/" + layout + "/"
        defaults = {"video": {"main": [max(740, self.editor.minimumHeight() + 220), 230], "top": [1120, 800], "right": [700], "video": [570, 170]},
                    "list": {"main": [560, 410], "top": [640, 1000], "right": [190, 370], "video": [560]},
                    "columns": {"main": [1080, 840], "top": [650, 320], "right": [190, 420], "video": [650]},
                    "karaoke": {"main": [760, 1160], "top": [650, 320], "right": [740, 230], "video": [650]}}
        sizes = defaults[layout]
        saved_sizes = None
        if restore:
            try:
                saved_sizes = json.loads(self.settings.value(prefix + "sizes", "null"))
                sizes = saved_sizes or sizes
            except (ValueError, TypeError):
                pass
            for host, key in ((self, "state"), (self.right_host, "host_state")):
                state = self.settings.value(prefix + key)
                if state is not None:
                    host.restoreState(state)
        # A new layout keeps floating panels floating; saved per-layout states take precedence.
        dock_state = prefix + ("host_state" if right_docks else "state")
        if not restore or not self.settings.contains(dock_state):
            for dock, (was_floating, geometry) in floating.items():
                if was_floating:
                    dock.setFloating(True)
                    dock.setGeometry(geometry)
        for widget in (self.top_split, self.right_host, self.right_split, self.video_box,
                       self.table, self.editor, self.wave_panel):
            widget.show()
        self.dock_source.setVisible(self.settings.value(prefix + "source", layout != "video", type=bool)
                                    if restore else layout != "video")
        self.people.setVisible(self.settings.value(prefix + "people", columns, type=bool) if restore else columns)
        if karaoke:
            self.dock_source.hide()
            self.people.hide()
        self.dock_source.raise_()
        if right_docks:
            self._fit_editor_docks()
        self._apply_edit_sizes(sizes)
        self._pending_video_fit = layout == "video" and not saved_sizes
        if initial:
            QTimer.singleShot(0, self, lambda: self._apply_edit_sizes(sizes))
        if self.isVisible():
            QTimer.singleShot(0, self, self._fit_video_column)
        self._queue_dock_screen_check()
        if karaoke and self.session:
            g = next((g for g in self.session.doc.groups if g.id == self.current_gid and g.kind == "song"), None)
            if g is None:
                g = next((g for g in self.session.doc.groups if g.kind == "song"), None)
                if g:
                    self.select_group(g.id)
                else:
                    self.current_gid = self.wave.current_gid = None
                    self.editor.set_group(self.session.doc, None, None, None)
                    self._refresh_k()
            elif self.proxy:
                self.table.blockSignals(True)
                self.table.select_gid(g.id, scroll=False)
                self.table.blockSignals(False)
        if self.session and (karaoke or was_karaoke):
            self._refresh_k()
        self._apply_speed()

    def _fit_editor_docks(self, *_args) -> None:
        if self.watch_mode or self.edit_layout not in ("video", "columns"):
            return
        wave_height = self.wave_panel.minimumHeight() + self.right_split.handleWidth() if self.edit_layout == "columns" else 0
        remaining = max(120, self.right_host.height() - self.editor.content.minimumHeight() - wave_height - 36)
        self.right_host.resizeDocks([self.dock_source, self.people], [remaining, remaining], Qt.Orientation.Vertical)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._pending_video_fit:
            QTimer.singleShot(0, self, self._fit_video_column)

    def _fit_video_column(self) -> None:
        if not self._pending_video_fit or self.edit_layout != "video" or self.watch_mode or not self.isVisible():
            return
        self.layout().activate()
        total = sum(self.top_split.sizes())
        width = round(self.player.height() * 16 / 9) + self.video_box.width() - self.player.width()
        width = max(round(total * .40), min(round(total * .65), width))
        self.top_split.setSizes([width, total - width])
        self._pending_video_fit = False

    def _dock_back_right(self, dock) -> None:
        owner = self.right_host if dock.parentWidget() is self.right_host else self
        owner.removeDockWidget(dock)
        right_docks = self.edit_layout in ("video", "columns")
        host = self.right_host if right_docks else self
        host.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea if right_docks
                           else Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.setFloating(False)
        if right_docks:
            other = self.people if dock is self.dock_source else self.dock_source
            if other.parentWidget() is host and not other.isFloating():
                host.tabifyDockWidget(other, dock)
        dock.show()
        dock.raise_()

    def _dock_panels_right(self) -> None:
        for dock in (self.dock_source, self.people):
            self._dock_back_right(dock)

    def _queue_dock_screen_check(self, _floating=False) -> None:
        QTimer.singleShot(0, self, self._keep_floating_docks_on_screen)

    def _keep_floating_docks_on_screen(self) -> None:
        for dock in (self.dock_source, self.people):
            if not dock.isFloating():
                continue
            frame = dock.frameGeometry()
            screens = QApplication.screens()
            if any(screen.availableGeometry().contains(frame) for screen in screens):
                continue
            screen = QApplication.screenAt(frame.center()) or self.screen()
            area = screen.availableGeometry()
            x = max(area.left(), min(frame.left(), area.right() - frame.width() + 1))
            y = max(area.top(), min(frame.top(), area.bottom() - frame.height() + 1))
            dock.move(dock.pos() + QPoint(x - frame.left(), y - frame.top()))

    def _apply_edit_sizes(self, sizes) -> None:
        for key, split in self._edit_splitters():
            values = sizes.get(key)
            if values and len(values) == split.count():
                split.setSizes(values)

    # ================================================================ project
    def _open_last(self) -> None:
        last = self.settings.value("last_project")
        if last and Path(last, "subkit-project.json").exists():
            try:
                self.open_project(Path(last))
            except Exception as exc:
                self.statusBar().showMessage(f"打开上次的工程失败：{exc}")

    def new_project(self) -> None:
        if self._episode_job is not None:
            return
        try:
            last = json.loads(self.settings.value("new_project_scheme", "") or "{}")
        except ValueError:
            last = {}
        d = NewProjectDialog(self, scheme=last)
        if d.exec() != NewProjectDialog.DialogCode.Accepted:
            return
        try:
            p = Project.create(Path(d.folder.text()), d.name.text() or Path(d.folder.text()).name,
                               original=d.original.currentData(), translation=d.translation.currentData(),
                               template_text=d.editor.styles_text(), template_scheme=d.editor.scheme(),
                               dict_from=Path(d.dict_from.text()) if d.dict_from.text() else None)
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"新建工程失败：{exc}")
            return
        try:
            if self.punctuation_profiles.apply_default(p.data.setdefault("settings", {}),
                    self.settings.value("display/default_punctuation_profile", "")):
                p.save()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, APP_NAME, f"新工程已建立，但默认标点方案未能套用：{exc}")
        self.settings.setValue("new_project_scheme", json.dumps(d.editor.scheme(), ensure_ascii=False))
        self.open_project(p.root)

    def change_template(self) -> None:
        if self.project is None:
            QMessageBox.information(self, APP_NAME, "请先新建或打开一个工程。")
            return
        from .views.template_editor import TemplateDialog

        try:
            current = self.project.template()
        except Exception:
            current = None
        ep = self.session.episode if self.session else None
        video = ep.video if ep is not None and ep.video and Path(ep.video).exists() else None
        if video is None:
            for eid in self.project.episode_ids():
                candidate = self.project.episode(eid).video
                if candidate is not None and candidate.exists():
                    video = candidate
                    break
        d = TemplateDialog(self, translation_lang=self.project.languages.get("translation", "zh-Hant"),
                           original_lang=self.project.languages.get("original", "ja"),
                           scheme=(self.project.data.get("template") or {}).get("scheme"), current=current, video=video)
        if d.exec() != TemplateDialog.DialogCode.Accepted:
            return
        try:
            path = self.project.replace_template(d.editor.styles_text(), d.editor.scheme())
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"换模板失败：{exc}")
            return
        if self.session is not None:
            from subkit.core.export import ExportCache

            self.session.template = self.project.template()
            self.session.export_cache = ExportCache()
            self.session.request_preview()
        self.statusBar().showMessage(f"样式模板已换成 {path.name}（旧模板留在 templates 文件夹）", 8000)

    def open_project_dialog(self) -> None:
        if self._episode_job is not None:
            return
        d = QFileDialog.getExistingDirectory(self, "打开工程文件夹")
        if d:
            self.open_project(Path(d))

    def open_project(self, root: Path) -> None:
        if self._episode_job is not None:
            return
        project = Project.open(root)
        if not self.close_session():
            return
        self._auto_process_asked.clear()
        self.project = project
        self.quick_colours.set_project(self.project)
        self.people.set_context(self.project)
        self._apply_language_labels()
        self.settings.setValue("last_project", str(root))
        self._update_batch_actions()
        self.ep_box.blockSignals(True)
        self.ep_box.clear()
        self.ep_box.addItems(self.project.episode_ids())
        self.ep_box.blockSignals(False)
        self._update_save_state()
        self._apply_rules()
        last_ep = self.settings.value(f"last_episode/{self.project.data['id']}")
        ids = self.project.episode_ids()
        if ids:
            self._switch_episode(last_ep if last_ep in ids else ids[0])

    def _switch_episode(self, eid: str) -> None:
        if self._episode_job is not None:
            return
        if not eid or self.project is None:
            return
        if self.session and self.session.episode.id == eid:
            return
        previous = self.session.episode.id if self.session else self.ep_box.currentText()
        if not self.close_session():
            blocked = self.ep_box.blockSignals(True)
            self.ep_box.setCurrentText(previous)
            self.ep_box.blockSignals(blocked)
            return
        ep = self.project.episode(eid)
        self._offer_autosave_recovery(ep)
        self.people.set_context(self.project)
        self.settings.setValue(f"last_episode/{self.project.data['id']}", eid)
        if self.ep_box.currentText() != eid:
            self.ep_box.blockSignals(True)
            self.ep_box.setCurrentText(eid)
            self.ep_box.blockSignals(False)
        if ep.document is None:
            self._update_save_state()
            self.statusBar().showMessage(f"{eid} 还没有合并：请用“集 → 合并”。")
            self._load_media(ep)
            return
        self._start_session(ep)

    def _offer_autosave_recovery(self, ep: Episode) -> None:
        try:
            recovery = ep.autosave_recovery()
            if recovery is None:
                return
            document, stamp = recovery
            dialog = QMessageBox(self)
            dialog.help_anchor = "project"
            dialog.setWindowTitle(APP_NAME)
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamp))
            dialog.setText(f"上次没有正常关闭。找到 {when} 的自动保存，比已保存的文档新。恢复它吗？")
            recover = dialog.addButton("恢复", QMessageBox.ButtonRole.AcceptRole)
            discard = dialog.addButton("丢弃", QMessageBox.ButtonRole.RejectRole)
            dialog.setDefaultButton(recover)
            app_help.add_help_button(dialog)
            dialog.exec()
            if dialog.clickedButton() is recover:
                ep.document = document
                ep.recovered = True
            elif dialog.clickedButton() is discard:
                ep.discard_autosave(stamp)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, APP_NAME, f"自动保存恢复失败：{exc}")

    def _start_session(self, ep: Episode) -> None:
        # A moved original is fine (the project reads its own copy); a missing or changed copy is not.
        from subkit.core.source_copies import scan_episode

        try:
            problems = scan_episode(ep.id, ep.dir)
        except (OSError, ValueError):
            problems = []
        if problems:
            self._warn_source_copies(problems)
        self.session = EpisodeSession(self.project, ep)
        s = self.session
        self.people.set_context(self.project, s)
        s.changed.connect(self._on_changed)
        s.previewReady.connect(self._on_preview)
        s.status.connect(self._on_session_status)
        s.busy.connect(self._on_busy)
        s.saved.connect(self._on_session_saved)
        s.saveStateChanged.connect(self._on_save_state_changed)
        self._update_save_state()
        self.model = GroupTableModel(ep.document, languages=self.project.languages, character_book=s.editor.character_book)
        self._recent_speakers = []
        self.proxy = GroupFilter()
        self.proxy.setSourceModel(self.model)
        self.proxy.set_mode(self.filter_box.currentText())
        self.table.setModel(self.proxy)
        self.table.setup_columns()
        self.wave_panel.set_hint_context(self.project)
        self.wave.set_document(ep.document)
        self.source.set_data(ep.document, ep.sources)
        s.request_preview()
        first_kind = "song" if self.edit_layout == "karaoke" else "dialogue"
        first = next((g for g in ep.document.groups if g.kind == first_kind), None)
        if first:
            self.select_group(first.id, seek=False)
        self._restore_episode_position()
        self._load_media(ep)
        self._load_dialogue_cache()
        self._apply_position_seek()
        self._apply_speed()
        self._update_counts()

    def _warn_source_copies(self, problems) -> None:
        from .views.source_copies import STATE

        box = QMessageBox(QMessageBox.Icon.Warning, APP_NAME,
                          f"这一集工程里保存的原字幕副本有问题：\n"
                          + "\n".join(f"· {p.copy.name}：{STATE.get(p.state, p.state)}" for p in problems)
                          + "\n\n仍然可以编辑；导出会缺原字幕里的样式和特效行等。要现在放回吗？", parent=self)
        fix = box.addButton("修复…", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("以后再说", QMessageBox.ButtonRole.RejectRole)
        box.help_anchor = "source-copies"
        app_help.add_help_button(box)
        box.exec()
        if box.clickedButton() is fix:
            self._source_copies(problems)

    def _source_copies(self, problems=None) -> None:
        """集 → 原字幕副本（检查和修复）…; with ``problems`` only those (this episode's) are shown."""
        if self.project is None:
            return
        from .views.source_copies import SourceCopiesDialog

        dialog = SourceCopiesDialog(self.project, self, problems=problems)
        dialog.fixed.connect(lambda eids: self.statusBar().showMessage(f"已放回原字幕副本：{'、'.join(eids)}", 8000))
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _video_links(self, focus_eid: Optional[str] = None) -> None:
        """集 → 视频文件位置…; the open episode reloads its video after it was pointed elsewhere."""
        if self.project is None or self._episode_job is not None:
            return
        from .views.video_links import VideoLinksDialog

        dialog = VideoLinksDialog(self.project, self, focus_eid=focus_eid)
        try:
            dialog.exec()
            applied = list(dialog.applied)
        finally:
            dialog.deleteLater()
        if not applied:
            return
        self.statusBar().showMessage(f"已重新指定 {len(applied)} 集的视频", 8000)
        current = self.session.episode if self.session else None
        for eid, path in applied:
            if current is not None and current.id == eid:
                # The episode in memory saves its metadata later: give it the new video too.
                current.set_video(path, (current.data.get("media") or {}).get("audio_track"))
                self._load_media(current)

    def _warn_missing_video(self, ep: Episode) -> None:
        if self.session is None or self.session.episode is not ep or ep.video is None or ep.video.exists():
            return
        box = QMessageBox(QMessageBox.Icon.Warning, APP_NAME,
                          f"找不到这一集的视频：\n{ep.video}\n\n视频挪过位置或改过名字时可以重新指定；"
                          "工程里的字幕不受影响，只是没有画面和波形。", parent=self)
        relink = box.addButton("重新指定…", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("以后再说", QMessageBox.ButtonRole.RejectRole)
        box.help_anchor = "video-links"
        app_help.add_help_button(box)
        box.exec()
        if box.clickedButton() is relink:
            self._video_links(ep.id)

    def _remember_episode_position(self) -> None:
        if not self.session:
            return
        pending = self._pending_position_seek
        restored = self._restored_position_time
        if pending and pending[0] is self.session:
            ms = pending[1]
        elif restored and restored[0] is self.session and self.player.time == restored[2]:
            ms = restored[1]  # mpv has not reported the seek result yet.
        else:
            ms = max(0, round(self.player.time * 1000)) if self._position_media_ready else 0
        key = f"last_position/{self.session.project.data['id']}/{self.session.episode.id}"
        self.settings.setValue(key, json.dumps({"group_id": self.current_gid, "time_ms": ms}))
        self.settings.sync()

    def _on_session_saved(self) -> None:
        if self.sender() is self.session:
            self._remember_episode_position()

    def _restore_episode_position(self) -> None:
        key = f"last_position/{self.session.project.data['id']}/{self.session.episode.id}"
        try:
            position = json.loads(self.settings.value(key, "") or "{}")
            ms = position["time_ms"]
            if not isinstance(ms, int) or isinstance(ms, bool) or ms < 0:
                return
            gid = position.get("group_id")
        except (ValueError, KeyError, TypeError):
            return
        doc = self.session.doc
        group = next((g for g in doc.groups if g.id == gid), None)
        if group is None and doc.groups:
            group = min(doc.groups, key=lambda g: abs(doc.group_time(g)[0] - ms))
        if group is not None:
            index = self.proxy.mapFromSource(self.model.index(self.model.row_of(group.id), 0))
            if not index.isValid():
                self.filter_box.setCurrentText("全部")
            self.select_group(group.id, restore=True)
        self._pending_position_seek = (self.session, ms)

    def _on_position_media_loaded(self) -> None:
        self._position_media_ready = True
        self._apply_position_seek()

    def _apply_position_seek(self) -> None:
        pending = self._pending_position_seek
        if pending is None or pending[0] is not self.session or not self._position_media_ready:
            return
        self._pending_position_seek = None
        self._restored_position_time = (self.session, pending[1], self.player.time)
        self.player.pause()
        self.player.seek(pending[1] / 1000.0)

    def _load_media(self, ep: Episode) -> None:
        self._position_media_ready = False
        self._drop_auto_process()
        self._passage_audio_session = self.session if self.session and self.session.episode is ep else None
        self._passage_features = self._passage_index = None
        video = ep.video
        if video and video.exists():
            try:
                self.player.open(str(video), audio_track=(ep.data.get("media") or {}).get("audio_track"))
            except Exception as exc:
                self.statusBar().showMessage(f"播放器：{exc}")
            self.wave.load(video, (ep.data.get("media") or {}).get("audio_track"), self.project.cache_dir / "waveform")
        else:
            self.wave.load(None, None, self.project.cache_dir / "waveform")
            if video is not None:
                # After the window has shown this episode.
                QTimer.singleShot(0, lambda: self._warn_missing_video(ep))

    def _on_save_state_changed(self) -> None:
        if self.sender() is self.session:
            self._update_save_state()

    def _update_save_state(self) -> None:
        title = APP_NAME
        if self.project:
            title += f" — {getattr(self.project, 'name', self.project.data['id'])}"
            eid = self.session.episode.id if self.session else self.ep_box.currentText()
            if eid:
                title += f" — {eid}"
        s = self.session
        if s and s.dirty:
            title += " ●"
        self.setWindowTitle(title)
        if s is None:
            self.lbl_save_state.clear()
            self.lbl_save_state.setToolTip("")
            return
        stamp = getattr(s, "last_saved_at", None)
        when = time.strftime("%H:%M", time.localtime(stamp)) if stamp is not None else None
        self.lbl_save_state.setText((f"有未保存的改动（上次保存 {when}）" if s.dirty else f"已保存 {when}")
                                   if when is not None else "还没保存过")
        recovery = getattr(s, "recovery_saved_at", None)
        tooltip = (f"恢复用副本 {time.strftime('%H:%M', time.localtime(recovery))}（程序意外退出时用，不是正式保存）"
                   if recovery is not None else "")
        self.lbl_save_state.setToolTip(tooltip)

    def _ask_save(self) -> QMessageBox.StandardButton:
        dialog = QMessageBox(self)
        dialog.help_anchor = "save"
        dialog.setWindowTitle(APP_NAME)
        dialog.setText(f"这一集（{self.session.episode.id}）有未保存的改动。保存吗？")
        save = dialog.addButton("保存", QMessageBox.ButtonRole.AcceptRole)
        discard = dialog.addButton("不保存", QMessageBox.ButtonRole.DestructiveRole)
        cancel = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(save)
        dialog.setEscapeButton(cancel)
        app_help.add_help_button(dialog)
        try:
            dialog.exec()
            if dialog.clickedButton() is save:
                return QMessageBox.StandardButton.Save
            if dialog.clickedButton() is discard:
                return QMessageBox.StandardButton.Discard
            return QMessageBox.StandardButton.Cancel
        finally:
            dialog.deleteLater()

    def _confirm_session_close(self) -> bool:
        if not self.session or not self.session.dirty:
            return True
        answer = self._ask_save() if self.act_ask_save.isChecked() else QMessageBox.StandardButton.Save
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        try:
            if answer == QMessageBox.StandardButton.Discard:
                (self.session.episode.dir / "journal" / "autosave.json").unlink(missing_ok=True)
            else:
                self.watch.finish_transform(apply=False)
                self.session.save()
        except (OSError, ValueError) as exc:
            self._help_message("save", QMessageBox.warning, APP_NAME, f"关闭前处理保存失败：{exc}")
            return False
        return True

    def close_session(self) -> bool:
        if not self._confirm_session_close():
            return False
        if not self._close_listening():
            return False
        self.watch.finish_transform(apply=False)
        self._cancel_dialogue_align()
        if self._dialign_results is not None:
            self._dialign_results.close()
            self._dialign_results.deleteLater()
            self._dialign_results = None
        self.wave.set_dialogue_records({})
        self._cancel_auto_karaoke()
        if self._kalign_results is not None:
            self._kalign_results.close()
            self._kalign_results.deleteLater()
            self._kalign_results = None
        self._remember_episode_position()
        self._group_playback_gid = None
        self._stop_at = self._step_select_gid = None
        self._group_select_timer.stop()
        self._pending_group = None
        self._group_nav_keys.clear()
        self.wave.cancel_jobs()
        self.wave_panel.set_hint_context(None)
        if self._karaoke_fx_panel is not None:
            self._karaoke_fx_panel.hide()
        dialog = getattr(self, "_review_answers_dialog", None)
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
            self._review_answers_dialog = None
        if self._speaker_picker is not None:
            self._speaker_picker.close()
            self._speaker_picker = None
        self.people.set_context(None)
        self._drop_auto_process()
        self.watch.stop_warm()
        if self.session:
            self.session.close()
            self.session.disconnect(self)
        self.session = None
        self._pending_position_seek = None
        self._restored_position_time = None
        self._position_media_ready = False
        self.current_gid = None
        self._apply_speed()
        self.watch.redraw()
        self._passage_audio_session = None
        self._passage_features = self._passage_index = None
        self._position_dirty = True
        self._position_cache = None
        self.editor.position.show_anchor(None)
        self._update_position_actions()
        self._update_transform_action()
        self._update_save_state()
        return True

    def _update_remove_episode_action(self) -> None:
        ready = self.project is not None and self.ep_box.currentText() in self.project.episode_ids()
        reason = "正在后台处理这一集" if self._episode_job is not None else ("" if ready else "先打开一集")
        set_action_reason(self.act_remove_episode, reason)

    def remove_episode(self) -> None:
        if self._episode_job is not None or self.project is None:
            return
        eid = self.ep_box.currentText()
        ids = self.project.episode_ids()
        if eid not in ids:
            return
        dialog = QMessageBox(self)
        dialog.help_anchor = "project"
        dialog.setWindowTitle("从工程中移除这一集")
        dialog.setText(f"把 {eid} 从工程里移除？这一集的工作文件会移到工程文件夹里的「已移除」文件夹，需要时可以手动找回；原视频和原字幕不受影响。")
        dialog.setInformativeText("不能用 Ctrl+Z 撤销。")
        remove = dialog.addButton("移除", QMessageBox.ButtonRole.AcceptRole)
        cancel = dialog.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.setEscapeButton(cancel)
        app_help.add_help_button(dialog)
        try:
            dialog.exec()
            accepted = dialog.clickedButton() is remove
        finally:
            dialog.deleteLater()
        if not accepted or self._episode_job is not None:
            return
        index = ids.index(eid)
        try:
            if not self.close_session():
                return
            if self.tapper.active:
                self.tapper.cancel()
            destination = self.project.remove_episode(eid)
        except (OSError, ValueError) as exc:
            self._switch_episode(eid)
            QMessageBox.warning(self, APP_NAME, f"移除失败：{exc}")
            return
        self.current_gid = self.wave.current_gid = None
        self.table.setModel(None)
        self.model = self.proxy = None
        self.editor.set_group(None, None, None, None)
        self.source.set_data(None, {})
        self.source.show_group(None)
        self.wave.set_document(None)
        self.wave.load(None, None, self.project.cache_dir / "waveform")
        self.watch.ass = None
        self.watch.geo.clear()
        self.watch.set_selected(None)
        self.player.shutdown()
        ids = self.project.episode_ids()
        self.ep_box.blockSignals(True)
        self.ep_box.clear()
        self.ep_box.addItems(ids)
        self.ep_box.blockSignals(False)
        if ids:
            self._switch_episode(ids[min(index, len(ids) - 1)])
        else:
            self.settings.remove(f"last_episode/{self.project.data['id']}")
            self.people.set_context(self.project)
        self._update_remove_episode_action()
        self._update_save_state()
        self.statusBar().showMessage(f"{eid} 已移到 {destination}", 10000)

    def add_episode(self) -> None:
        if self._episode_job is not None:
            return
        if self.project is None:
            QMessageBox.information(self, APP_NAME, "请先新建或打开一个工程。")
            return
        roles = self.project.settings().get("cn_styles", {})
        n = len(self.project.episode_ids()) + 1
        d = AddEpisodeDialog(self, suggested_id=f"S01E{n:03d}", start_dir=self.settings.value("last_dir", ""),
                             default_roles=roles, languages=self.project.languages,
                             display_settings=self.project.settings().get("display", {}))
        if d.exec() != AddEpisodeDialog.DialogCode.Accepted:
            return
        self.settings.setValue("last_dir", d.start_dir)
        eid = d.eid.text().strip()
        m = re.match(r"S(\d+)E(\d+)", eid)
        try:
            ep = self.project.add_episode(eid, season=f"S{int(m.group(1)):02d}" if m else "S01", number=int(m.group(2)) if m else n)
            langs = self.project.languages
            ep.data.setdefault("settings", {})["cn_styles"] = d.style_roles()
            ep.save_meta()
            self.project.data.setdefault("settings", {}).setdefault("display", {})["merge_lines"] = d.merge_settings()
            self.project.save()
            self._apply_rules()
            if self.session:
                self.session.request_preview()

        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"添加失败：{exc}\n\n{traceback.format_exc()[-800:]}")
            return
        imports = {"video": d.video.text(), "audio": d.audio.currentData(),
                   "cn": d.cn.text(), "jp": d.jp.text(), "roles": d.style_roles()}
        self._start_episode_job(EpisodeJob(ep, "import", imports=imports))

    def merge(self) -> None:
        if self._episode_job is not None:
            return
        if self.project is None or not self.ep_box.currentText():
            return
        ep = self.session.episode if self.session else self.project.episode(self.ep_box.currentText())
        self._offer_merge(ep)

    def _offer_merge(self, ep: Episode) -> None:
        if ep.document is not None:
            r = QMessageBox.question(self, APP_NAME, "这一集已经有工作文档。整集重新合并会重新配对，并替换你改过的字、读音、隐藏、配对和手调时间（删除过的原文不会回来；说话人保留）。当前文档会先保存并备份。继续吗？")
            if r != QMessageBox.StandardButton.Yes:
                return
        from importlib.util import find_spec

        sem_ok = find_spec("sentence_transformers") is not None
        st = ep.settings()["merge"]
        d = MergeDialog(self, filler_mode=st.get("filler_mode", "keep"), semantic=st.get("semantic", False), semantic_available=sem_ok, languages=self.project.languages,
                        joint=st.get("joint", "first"))
        if d.exec() != MergeDialog.DialogCode.Accepted:
            return
        s = dict(st)
        s.update(d.settings())
        # Save first: the pre-merge backup copies document.json, and a cancelled merge must not lose edits.
        if self.session and self.session.episode is ep and self.session.dirty:
            self.session.save()
        self._start_episode_job(EpisodeJob(ep, "merge", settings=s, semantic_factory=_labse_factory))

    def _start_episode_job(self, job) -> None:
        if self._episode_job is not None:
            return
        self._episode_job = job
        self._episode_action_states = [(a, a.isEnabled()) for a in self._episode_actions]
        for action, _enabled in self._episode_action_states:
            set_action_reason(action, "正在添加一集")
        self._episode_widget_states = [(w, w.isEnabled()) for w in (self.ep_box, self.editor, self.people)]
        for widget, _enabled in self._episode_widget_states:
            widget.setEnabled(False)
        self._edit_menu_enabled = self.edit_menu.isEnabled()
        self.edit_menu.setEnabled(False)
        if self.session:
            self.session._autosave.stop()
        self._episode_progress = EpisodeProgress(self)
        self._episode_progress.cancelRequested.connect(self._cancel_episode_job)
        job.signals.progress.connect(self._episode_step)
        job.signals.done.connect(self._episode_done)
        job.signals.failed.connect(self._episode_failed)
        job.signals.cancelled.connect(self._episode_cancelled)
        self._episode_progress.show()
        self._episode_progress.activateWindow()
        QThreadPool.globalInstance().start(job)

    @Slot(str)
    def _episode_step(self, text) -> None:
        if self._episode_progress is not None:
            self._episode_progress.step.setText(text)
        self.statusBar().showMessage(text)

    @Slot()
    def _cancel_episode_job(self) -> None:
        if self._episode_job is not None:
            self._episode_job.cancel()
            self._episode_progress.cancel_button.setEnabled(False)

    def _finish_episode_job(self):
        job = self._episode_job
        self._episode_job = None
        self._episode_progress.hide()
        self._episode_progress.deleteLater()
        self._episode_progress = None
        for action, enabled in self._episode_action_states:
            set_action_reason(action, "")
            action.setEnabled(enabled)
        self._update_batch_actions()
        for widget, enabled in self._episode_widget_states:
            widget.setEnabled(enabled)
        self.edit_menu.setEnabled(self._edit_menu_enabled)
        if self.session and not self._closing:
            self.session._autosave.start()
        self.statusBar().clearMessage()
        return job

    @Slot(object, object)
    def _episode_done(self, ep, result) -> None:
        job = self._finish_episode_job()
        if self._closing:
            return
        if self._asking_close:
            # No result dialog on top of the close question; shown afterwards if the window stays open.
            self._pending_episode_result = (job, ep, result)
            return
        self._show_episode_result(job, ep, result)

    def _show_episode_result(self, job, ep, result) -> None:
        # The job worked on a copy; settings changed in the window must reach this episode's preview and export.
        ep.project = self.project
        if job.mode == "import":
            if not self.close_session():
                self.ep_box.blockSignals(True)
                self.ep_box.addItem(ep.id)
                self.ep_box.blockSignals(False)
                return
            self.ep_box.blockSignals(True)
            self.ep_box.addItem(ep.id)
            self.ep_box.setCurrentText(ep.id)
            self.ep_box.blockSignals(False)
            self._update_save_state()
            self._load_media(ep)
            self._offer_merge(ep)
        elif job.mode == "reanalyze":
            if self.session:
                self.session.episode.sources = ep.sources
            self._reanalyzed(ep, result)
        else:
            if self.session:
                self.session.dirty = False
            self.close_session()
            self._start_session(ep)
            dialog = CheckResultDialog(ep, result, self, merged=True)
            dialog.singleLanguageRequested.connect(self._show_single_language_dialogue)
            try:
                dialog.exec()
            finally:
                dialog.deleteLater()

    @Slot(str)
    def _episode_failed(self, error) -> None:
        job = self._finish_episode_job()
        if not self._closing:
            title = {"import": "添加失败", "merge": "合并失败", "reanalyze": "重新分析失败"}[job.mode]
            QMessageBox.warning(self, APP_NAME, f"{title}：{error[:200]}\n\n{error[-1200:]}")

    @Slot()
    def _episode_cancelled(self) -> None:
        self._finish_episode_job()
        if not self._closing:
            QMessageBox.information(self, APP_NAME, "已取消，这一集没有改动。")

    def reanalyze(self) -> None:
        if self._episode_job is not None:
            return
        """Apply the current analyzer to an episode imported earlier (e.g. SRT formatting tags), then offer a merge."""
        if self.project is None or not self.ep_box.currentText():
            return
        ep = self.session.episode if self.session else self.project.episode(self.ep_box.currentText())
        self._start_episode_job(EpisodeJob(ep, "reanalyze"))

    def _reanalyzed(self, ep, res) -> None:
        if not res:
            QMessageBox.information(self, APP_NAME, "这一集没有需要分析的原声字幕。")
            return
        msg = reanalyze_text(res, ep.sources, self.project.languages)
        if ep.document is not None and any(g.state == "confirmed" for g in ep.document.groups):
            QMessageBox.information(self, APP_NAME, f"{msg}\n这一集已有确认过的配对，暂时不能整集重新合并"
                                    "（局部重新对齐在第二阶段）；新分析会用于以后的合并。")
            return
        QMessageBox.information(self, APP_NAME, f"{msg}\n旧的分析结果已备份在 sources 文件夹。\n"
                                "接下来重新合并这一集，新的分析才会进入工作文档（旧文档会先备份）。")
        self._offer_merge(ep)

    def brackets_to_text(self) -> None:
        """Older episodes: readings taken from katakana in brackets (眷属(ファミリア)) — the bracket goes back into
        the line (ja-cc 3 already treats new imports this way)."""
        if not self.session:
            return
        ed = self.session.editor
        aids = ed.bracket_candidates()
        if not aids:
            QMessageBox.information(self, APP_NAME, "这一集没有取自原字幕括号的片假名注音。")
            return
        doc = self.session.doc
        lines = [f"{doc.annotations[x].base}（{doc.annotations[x].text}）" for x in aids[:12]] + (["…"] if len(aids) > 12 else [])
        if not self._ask(f"这 {len(aids)} 处振假名取自原字幕括号里的片假名：\n" + "、".join(lines)
                         + "\n\n把括号原样放回正文、去掉这些振假名、按普通读音重新建议吗？（可以撤销）"):
            return
        n = [0]
        if self._edit(lambda: n.__setitem__(0, ed.bracket_to_text(aids, engine=self._engine()))):
            self.statusBar().showMessage(f"已放回正文：{n[0]} 处（Ctrl+Z 撤销）", 6000)

    def save(self) -> None:
        if self._episode_job is not None:
            return
        if self.session:
            self.session.save()

    def export(self) -> None:
        if not self.session:
            return
        ep = self.session.episode
        rel = ep.settings()["release"]
        lang = self.project.languages.get("translation", "zh-Hant")
        base = (ep.video.with_suffix("") if ep.video else ep.dir / "exports" / ep.id)
        default = Path(str(base) + f".{lang}.ass")
        k = 1
        while default.exists():
            default = Path(str(base) + f".{lang}.v{k}.ass")
            k += 1
        d = ExportDialog(self, default_path=str(default), include_actor=rel.get("include_actor", True),
                         include_comments=rel.get("include_source_comments", False), blur=float(rel.get("blur", 0.0)),
                         name_format=self.settings.value("export/name_format", "short"),
                         has_chinese=any(m.lang.lower().startswith("zh") for m in ep.document.members.values()),
                         zh_both=self.settings.value("export/zh_both", False, type=bool))
        if d.exec() != ExportDialog.DialogCode.Accepted:
            return
        self.settings.setValue("export/name_format", d.name_format.currentData())
        from subkit.core.season_export import script_targets
        both = d.zh_both.isChecked()
        reference_enabled = d.reference.isChecked()
        reference_path = Path(d.reference_path.text().strip()) if reference_enabled and d.reference_path.text().strip() else None
        reference_options = {"reference_enabled": True, "reference_path": reference_path} if reference_enabled else {}
        if not d.zh_both.isHidden():
            self.settings.setValue("export/zh_both", both)
        out = Path(d.path.text())
        targets = [p for _, p in script_targets(out)] if both else [out]
        for i, target in enumerate(targets):
            if target.exists():
                r = self._help_message("export", QMessageBox.question, APP_NAME, f"{target.name} 已存在，覆盖吗？（选“否”将另存为带编号的新文件）")
                if r != QMessageBox.StandardButton.Yes:
                    k = 1
                    while target.exists():
                        target = target.with_name(target.stem + f".v{k}" + target.suffix)
                        k += 1
                    targets[i] = target
        rep_v = ep.validate()
        if not rep_v.ok:
            r = self._help_message("export", QMessageBox.question, APP_NAME, f"检查发现问题：{rep_v.summary()}\n仍然导出吗？（问题会写进导出报告）")
            if r != QMessageBox.StandardButton.Yes:
                return
        from subkit.core import karaoke as kara
        from subkit.core.readings import visible_map

        k_open = kara.unfinished(ep.document)
        if k_open:
            names = [f"{fmt_ms(ep.document.members[mid].start)} {visible_map(ep.document.members[mid])[0].text[:20]}：{'；'.join(iss)}"
                     for mid, iss in k_open[:12]]
            r = self._help_message("export", QMessageBox.question, APP_NAME, f"有 {len(k_open)} 行 K 轴还没打完（占位拍不算完成的 K 轴）：\n"
                                     + "\n".join(names) + "\n\n仍然导出吗？（会写进导出报告）")
            if r != QMessageBox.StandardButton.Yes:
                return
        try:
            result = self._run_export_job(include_actor=d.actor.isChecked(), include_source_comments=d.comments.isChecked(),
                                          blur=d.blur.value(), video_height=1080, name_format=d.name_format.currentData(), zh_both=both,
                                          annotation_order=d.annotation_order.currentData(), **reference_options)
            if result is None:
                return
            results = result if both else [result]
            for ass, rep, fc in results:
                if fc.ran and not fc.ok:
                    if not self._show_font_check(fc, exporting=True):
                        return
            for target in targets:
                if any(Path(s.path).resolve() == target.resolve() for s in ep.sources.values()):
                    raise PermissionError("不能覆盖原料字幕")
            if reference_enabled:
                from subkit.core.reference_report import preflight_targets
                protected = [path for _ass, rep, _fc in results
                             for path in (getattr(rep, "reference", None) or {}).get("protected_input_paths", [])]
                protected.extend(s.path for s in ep.sources.values())
                protected.extend(ep.dir / s.copy for s in ep.sources.values() if getattr(s, "copy", ""))
                if reference_path is not None:
                    protected.append(reference_path)
                preflight_targets([*targets, *(p.with_suffix(".reference.html") for p in targets),
                                   *(p.with_suffix(".export-report.txt") for p in targets)], protected)
            messages = []
            for target, (ass, rep, fc) in zip(targets, results):
                path = ep.write_export(ass, target)
                lines = [f"已导出：{path}"] + self._export_report_lines(rep, fc)
                lines += self._write_reference_result(rep, path)
                report = path.with_suffix(".export-report.txt")
                report.write_text("\n".join(lines + ["检查：" + rep_v.summary()]), encoding="utf-8")
                messages.extend(lines)
        except Exception as exc:
            self._help_message("export", QMessageBox.warning, APP_NAME, f"导出失败：{exc}")
            return
        self._help_message("export", QMessageBox.information, APP_NAME, "\n".join(messages))

    def _write_reference_result(self, report, target: Path) -> list[str]:
        data = getattr(report, "reference", None)
        if not isinstance(data, dict):
            return []
        from subkit.core.reference_report import write_reference_report
        try:
            path = write_reference_report(data, target)
            return [f"原配字幕对照报告：{path}"]
        except Exception as exc:
            return [f"原配字幕对照报告没有写入：{exc}（ASS 已导出）"]

    def _export_report_lines(self, rep, fc) -> list[str]:
        """What one episode's export report says (counts, hidden and unfinished lines, warnings, fonts)."""
        lines = [f"事件 {rep.events}，振假名 {rep.ruby_lines}"]
        order = getattr(rep, "annotation_order", "end")
        lines.append("注音行顺序：" + ("紧跟对应正文行" if order == "after_parent" else "统一放在文件末尾"))
        edge_spaces = getattr(rep, "edge_spaces_removed", 0)
        empty_events = getattr(rep, "empty_events_removed", 0)
        if edge_spaces or empty_events:
            lines.append(f"发布清理：去掉 {edge_spaces} 个首尾半角空格，跳过 {empty_events} 条空白字幕（工程原文不变）")
        lines.append(f"隐藏的组：{rep.hidden_groups} 组（没有导出）")
        if rep.not_rendered:
            lines.append(f"未渲染（简洁模式或你选择隐藏的）：{len(rep.not_rendered)} 处 — " + "、".join(rep.not_rendered[:12]))
        if rep.uncertain:
            lines.append(f"还没确认的读音：{rep.uncertain} 处（按当前候选导出）")
        if rep.hidden_by_rules:
            lines.append(f"显示规则：{rep.hidden_by_rules} 个日文句末「。」没有显示（工程里的原文不变）")
        if rep.replacement_matches:
            lines.append(f"批量替换：{rep.replacement_lines} 行、{rep.replacement_matches} 处，"
                         f"其中删除 {rep.replacement_deleted} 个字（编辑原文不变）")
        if rep.zh_script == "traditional":
            lines.append(f"中文显示为繁体：{rep.zh_lines} 行（{rep.zh_variant}）")
        elif rep.zh_script == "simplified":
            lines.append(f"中文显示为简体：{rep.zh_lines} 行")
        if rep.karaoke_lines:
            lines.append(f"K 轴：{rep.karaoke_lines} 行" + (f"，其中 {len(rep.karaoke_unfinished)} 行未完成：" + "；".join(rep.karaoke_unfinished[:6])
                                                          if rep.karaoke_unfinished else "，全部完成"))
        if rep.positions_overridden:
            lines.append(f"位置：{rep.positions_overridden} 行原字幕自带的位置标签在输出里换成了你设置的位置（原字幕不变）")
        if rep.karaoke_stripped:
            lines.append(export_stripped_text(rep.karaoke_stripped))
        if rep.not_timed:
            lines.append(f"没有显示时间的行（结束不晚于开始，或整行在片头之前）：{len(rep.not_timed)} 行，作为注释行写入、不会显示 — "
                         + "；".join(rep.not_timed[:6]) + "（要显示请调时间，不要就删掉）")
        lines.extend(rep.warnings)
        lines.extend(report_lines(fc, library=self._font_library_index() if self.private_enabled else None))
        return lines

    def export_season(self) -> None:
        """Every merged episode → <video name>.<suffix>.ass in one folder (文件 → 导出 → 整季发布用 ASS…)."""
        if self.project is None or self._episode_job is not None or self._export_job is not None or self._closing:
            return
        from .export_job import SeasonExportJob
        from .views.season_export import SeasonExportDialog, SeasonReportDialog

        dialog = SeasonExportDialog(self.project, self.settings, self.session.episode.id if self.session else None, self)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            items, skipped = dialog.chosen(), dialog.skipped()
            include_review = dialog.review_packet.isChecked()
            review_targets = dialog.review_targets()
            annotation_order = dialog.annotation_order.currentData()
            reference_enabled = dialog.reference.isChecked()
            reference_path = Path(dialog.reference_path.text().strip()) if reference_enabled and dialog.reference_path.text().strip() else None
            reference_options = {"reference_enabled": True, "reference_path": reference_path} if reference_enabled else {}
            both = dialog.zh_both.isChecked()
            folder = Path(dialog.folder.text().strip())
            suffix = (dialog.simplified_suffix.text().strip().strip(".") + "+" + dialog.traditional_suffix.text().strip().strip(".")
                      if both else dialog.suffix.currentText().strip().strip("."))
        finally:
            dialog.deleteLater()
        current = None
        if self.session is not None:
            with self.session.lock:
                current = copy.deepcopy(self.session.episode)
        job = SeasonExportJob(self.project, items, current, check_fonts, include_review=include_review, video_height=1080,
                              name_format=self.settings.value("export/name_format", "short"), annotation_order=annotation_order,
                              **reference_options)
        progress = ExportProgress(self)
        progress.setWindowTitle("正在导出整季…")
        job.signals.progress.connect(lambda i, n, eid, step: progress.label.setText(f"第 {i}/{n} 集 {eid}：{step}…"))
        if not self._wait_export_job(job, progress, "正在导出整季…"):
            self.statusBar().showMessage("已取消整季导出，没有写入文件", 6000)
            return
        if job.error is not None:
            self._help_message("export", QMessageBox.warning, APP_NAME, f"整季导出失败：{job.error}")
            return
        if reference_enabled:
            from subkit.core.reference_report import preflight_targets
            protected = [path for result in job.results
                         for path in (getattr(result.report, "reference", None) or {}).get("protected_input_paths", [])]
            for result in job.results:
                if result.episode is not None:
                    protected.extend(source.path for source in result.episode.sources.values())
                    protected.extend(result.episode.dir / source.copy for source in result.episode.sources.values()
                                     if getattr(source, "copy", ""))
            if reference_path is not None:
                protected.append(reference_path)
            planned = [result.target for result in job.results if not result.error]
            planned += [path.with_suffix(".reference.html") for path in list(planned)]
            planned += [path for pair in review_targets.values() for path in pair]
            planned.append(folder / f"{self.project.root.name}.{suffix}.export-report.txt")
            try:
                preflight_targets(planned, protected)
            except Exception as exc:
                self._help_message("export", QMessageBox.warning, APP_NAME, f"整季导出没有写入文件：{exc}")
                return
        written, problems, sections = [], [], []
        for result in job.results:
            head = f"【{result.eid}】{result.target.name}"
            if not result.error:
                try:
                    result.episode.write_export(result.ass, result.target)
                    written.append(result.eid)
                except Exception as exc:
                    result.error = f"写入失败：{exc}"
            if result.error:
                problems.append(result.eid)
                sections.append(f"{head}\n没有导出：{result.error}")
                continue
            lines = self._export_report_lines(result.report, result.fonts)
            lines += self._write_reference_result(result.report, result.target)
            if not result.validation.ok:
                lines.append("检查：" + result.validation.summary())
            if result.unfinished:
                lines.append(f"K 轴还没打完：{len(result.unfinished)} 行（占位拍不算完成）")
            sections.append(head + "\n" + "\n".join(lines))
        packets_written, packet_problems = [], []
        if include_review:
            from subkit.core.review import write_packet_files

            for eid in dict.fromkeys(written):
                error = job.review_errors.get(eid, "")
                if not error:
                    try:
                        paths = write_packet_files(job.review_packets[eid], review_targets[eid])
                        packets_written.append(eid)
                        sections.append(f"【{eid}】审议包\n" + "\n".join(str(path) for path in paths))
                    except Exception as exc:
                        error = str(exc) or type(exc).__name__
                if error:
                    packet_problems.append(eid)
                    sections.append(f"【{eid}】审议包没有导出：{error}（ASS 已导出）")
        count = (f"{len(set(written))} 集，共 {len(written)} 个文件" if both else f"{len(written)} 集")
        summary = [f"已导出 {count}到：{folder}（语言后缀 {suffix}）"]
        if include_review:
            summary.append(f"审议包已导出 {len(packets_written)} 份（每份 Markdown + JSONL）")
            if packet_problems:
                summary.append(f"审议包没有导出：{len(packet_problems)} 集（{'、'.join(packet_problems)}），原因见下面")
        if problems:
            unit = "个文件" if both else "集"
            summary.append(f"没有导出：{len(problems)} {unit}（{'、'.join(problems)}），原因见下面")
        summary += [f"跳过：{eid}（{reason}）" for eid, reason in skipped]
        text = "\n".join(summary) + "\n\n" + "\n\n".join(sections)
        report = folder / f"{self.project.root.name}.{suffix}.export-report.txt"
        try:
            report.write_text(text, encoding="utf-8")
            text = f"报告：{report}\n" + text
        except OSError as exc:
            text = f"报告没有写入（{exc}）\n" + text
        self.statusBar().showMessage(f"已导出 {count}" + (f"；审议包 {len(packets_written)} 份" if include_review else ""), 8000)
        report_dialog = SeasonReportDialog(text, folder, self)
        report_dialog.exec()
        report_dialog.deleteLater()

    def export_review(self) -> None:
        """审议包: a plain view of the episode (readings on their kanji, speakers, open questions) for a reviewer."""
        if not self.session:
            return
        import time as _time

        from subkit.core.review import write_packet
        from subkit.core.display import DisplayRules
        ep = self.session.episode
        chars = self.project.character_book()
        langs = self.project.languages
        try:
            md, jl = write_packet(self.session.doc, ep.dir / "review", episode=ep.id, stamp=_time.strftime("%Y%m%d-%H%M%S"),
                                  orig_lang=langs.get("original", "ja"), trans_lang=langs.get("translation", "zh"), chars=chars,
                                  pinyin_mode=self.project.pinyin_mode,
                                  display=DisplayRules.from_settings(self.project.settings()))
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"导出审议包失败：{exc}")
            return
        QMessageBox.information(self, APP_NAME, f"审议包已导出：\n{md}\n{jl}\n\n"
                                "把 .md 交给模型复查（文件开头写了要检查什么、怎么回答）。\n"
                                "模型的回答（每行一条 JSON）之后可以导回双语字幕工坊逐条确认；不会自动改字幕。")

    def import_review_answers(self) -> None:
        if not self.session:
            return
        from .views.review_answers import ImportAnswersDialog, ReviewAnswersDialog

        chooser = ImportAnswersDialog(self)
        try:
            if chooser.exec() != chooser.DialogCode.Accepted:
                return
            previous = getattr(self, "_review_answers_dialog", None)
            if previous is not None:
                previous.close()
                previous.deleteLater()
            dialog = ReviewAnswersDialog(self.session, chooser.items, chooser.unreadable, self)
            self._review_answers_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        finally:
            chooser.deleteLater()

    def check(self) -> None:
        if not self.session:
            return
        rep = self.session.episode.validate()
        result = CheckResultDialog(self.session.episode, rep, self)
        try:
            result.exec()
        finally:
            result.deleteLater()

    def _show_single_language_dialogue(self) -> None:
        self.filter_box.setCurrentText("只有一种语言的对白")

    def _batch(self, mode: str) -> None:
        if not self.private_enabled or self.project is None or self._episode_job is not None:
            return
        if self._batch_dialog is not None:
            self._batch_dialog.raise_()
            self._batch_dialog.activateWindow()
            return
        from .views.batch import BatchDialog

        dialog = BatchDialog(mode, self.settings, lambda: (self.wave.lead_in_ms, self.wave.lead_out_ms),
                             self.project.root, self._prepare_batch, self._reload_batch_project, self)
        self._batch_dialog = dialog
        dialog.finished.connect(self._batch_closed)
        dialog.show()

    def _batch_closed(self, _result: int) -> None:
        dialog, self._batch_dialog = self._batch_dialog, None
        if dialog is not None:
            dialog.deleteLater()

    def _update_batch_actions(self) -> None:
        for action in self.batch_actions.values():
            action.setEnabled(self.project is not None and self._episode_job is None)

    def _prepare_batch(self) -> None:
        if self.session and self.session.dirty:
            self.session.save()
        self.project.save()

    def _reload_batch_project(self) -> None:
        fresh = Project.open(self.project.root)
        self.project.data.clear()
        self.project.data.update(fresh.data)
        current = self.ep_box.currentText()
        self.ep_box.blockSignals(True)
        try:
            self.ep_box.clear()
            self.ep_box.addItems(self.project.episode_ids())
            self.ep_box.setCurrentIndex(self.ep_box.findText(current))
        finally:
            self.ep_box.blockSignals(False)
        self._update_batch_actions()

    def _set_private_enabled(self, enabled: bool) -> None:
        self.private_enabled = enabled
        self.settings.setValue("private/enabled", enabled)
        for action in self._private_actions:
            action.setVisible(enabled)
        self._update_batch_actions()

    def _show_font_check(self, check, *, exporting=False) -> bool:
        dialog = FontCheckDialog(check, self, exporting=exporting,
                                 library=self._font_library_index() if self.private_enabled else None)
        try:
            return dialog.exec() == dialog.DialogCode.Accepted
        finally:
            dialog.deleteLater()

    def _clear_layout_cache(self) -> None:
        if self.project is None:
            return
        session = self.session
        jobs = [self.watch._warm, *self.watch._geometry_jobs.values()]
        try:
            if session is not None and not session.stop_preview():
                raise RuntimeError("后台测量尚未停止，请稍后重试")
            self.watch.stop_warm()
            if any(job is not None and not job.finished.is_set() for job in jobs):
                raise RuntimeError("后台测量尚未停止，请稍后重试")
            (self.project.cache_dir / "render.sqlite3").unlink(missing_ok=True)
            from subkit.ruby.measure import forget_measurement_environment

            forget_measurement_environment()
            self.watch.geo.clear()
            self.watch.anchors.clear()
            self.watch.invalidate_speakers()
            self._position_cache = None
            self._position_dirty = True
            if session is not None:
                session.export_cache.clear()
                session.warmed = False
            self.statusBar().showMessage("排版缓存已清除，预览会重新测量。", 8000)
        except (OSError, RuntimeError) as exc:
            self._help_message("fonts", QMessageBox.warning, APP_NAME, f"清除排版缓存失败：{exc}")
        finally:
            self.watch._measuring_stopped = False
            if session is not None:
                session.request_preview()

    def _font_library_index(self) -> FontLibrary:
        path = self.settings.value("fonts/library_path", str(DEFAULT_FONT_LIBRARY), type=str)
        library = getattr(self, "_font_library", None)
        if library is None or library.path != Path(path):
            self._font_library = FontLibrary(path)
        return self._font_library

    def _choose_font_library(self) -> None:
        dialog = QFileDialog(self, "字体库位置…", str(self._font_library_index().path), "字体库 (*.json)")
        dialog.help_anchor = "fonts"
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog)
        app_help.dialog_buttons(dialog)
        try:
            if dialog.exec() and dialog.selectedFiles():
                self.settings.setValue("fonts/library_path", dialog.selectedFiles()[0])
                self._font_library = None
        finally:
            dialog.deleteLater()

    def _run_export_job(self, *, fonts_only=False, **options):
        if self._export_job is not None or self._closing:
            return None
        with self.session.lock:
            job = ExportJob(self.session.episode, check_fonts, **options)
        dialog = ExportProgress(self, fonts_only=fonts_only)
        if not fonts_only and options.get("reference_enabled"):
            dialog.label.setText("正在生成字幕与原配对照报告…")
        if not self._wait_export_job(job, dialog, "正在检查字体…" if fonts_only else "正在导出…"):
            self.statusBar().showMessage("已取消检查字体" if fonts_only else "已取消导出", 6000)
            return None
        if job.error is not None:
            raise job.error
        return job.result

    def _wait_export_job(self, job, dialog, text: str) -> bool:
        """Run a detached export job behind its modal progress window; False when cancelled or closing."""
        self._export_job, self._export_progress = job, dialog
        dialog.cancelRequested.connect(job.cancel)
        timer = QTimer(dialog)
        timer.setInterval(20)

        def poll():
            if job.finished.is_set():
                timer.stop()
                dialog.done(0 if job.cancelled or self._closing else 1)

        timer.timeout.connect(poll)
        self.statusBar().showMessage(text)
        QThreadPool.globalInstance().start(job)
        timer.start()
        try:
            dialog.exec()
        finally:
            timer.stop()
            # Only poll/closeEvent can finish this dialog, after the worker's Event.
            self._export_job = self._export_progress = None
            dialog.deleteLater()
        if self._export_close_pending:
            self._export_close_pending = False
            QTimer.singleShot(0, self.close)
        if job.cancelled or self._closing:
            return False
        self.statusBar().clearMessage()
        return True

    def check_episode_fonts(self) -> None:
        if not self.session:
            return
        try:
            result = self._run_export_job(fonts_only=True, video_height=1080)
            if result is None:
                return
            _ass, _rep, fc = result
        except Exception as exc:
            self._help_message("fonts", QMessageBox.warning, APP_NAME, f"字体检查失败：{exc}")
            return
        self._show_font_check(fc)

    # ================================================================ selection / playback
    def select_group(self, gid: str, *, seek: bool = False, preserve_selection: bool = False,
                     defer: bool = False, scroll: bool = True, restore: bool = False,
                     keep_view: bool = False) -> None:
        if not self.session:
            return
        doc = self.session.doc
        try:
            g = doc.group(gid)
        except KeyError:
            return
        if self.edit_layout == "karaoke" and g.kind != "song" and not restore:
            return
        if self._step_select_gid or (self._group_playback_gid and gid != self._group_playback_gid):
            self._group_playback_gid = None
            self._stop_at = self._step_select_gid = None
        if self.current_gid != gid:
            self.wave.clear_suggestions()
        self.current_gid = gid
        self._apply_speed()
        self.table.blockSignals(True)
        self.table.select_gid(gid, scroll=scroll, preserve_selection=preserve_selection)
        self.table.blockSignals(False)
        self._group_select_timer.stop()
        self._pending_group = None
        if defer and not self._selection_busy():
            self._pending_group = (self.session, gid, seek)
            self.wave.current_gid = gid
            self.wave.update()
            self.watch.set_selected(gid)
            if seek:
                self.player.seek(doc.group_time(g)[0] / 1000.0, exact=False)
            self._group_select_timer.start()
            return
        self._show_group(g, seek=seek, keep_view=keep_view)

    def _selection_busy(self) -> bool:
        return (self._typing() or self._inline is not None or self.watch.transform is not None
                or self.tapper.active or any(pop.isVisible() for pop in self.findChildren(ReadingPopup)))

    def _settle_group_selection(self) -> None:
        # A lost key release must not defer the selection forever: a held key keeps auto-repeating, so
        # 1.5 s without any navigation key event (longer than the slowest Windows repeat delay) means released.
        if self._group_nav_keys and time.monotonic() - self._group_nav_last > 1.5:
            self._group_nav_keys.clear()
        if self._group_nav_keys:
            self._group_select_timer.start()
            return
        self._flush_group_selection()

    def _flush_group_selection(self) -> None:
        pending, self._pending_group = self._pending_group, None
        self._group_select_timer.stop()
        if pending is None or pending[0] is not self.session or pending[1] != self.current_gid:
            return
        try:
            g = self.session.doc.group(pending[1])
        except KeyError:
            return
        self._show_group(g, seek=pending[2])

    def _refresh_group_panels(self, g: Group) -> None:
        doc = self.session.doc
        if not self.watch_mode or self.editor.isVisible():
            self._editor_pending = False
            prev, nxt = self.session.editor.neighbours(g.id)
            self.editor.pinyin_mode = self.project.data.get("settings", {}).get("pinyin", {}).get("mode", "off")
            self.editor.character_book = self.session.editor.character_book
            self.editor.recent_speakers = self._recent_speakers
            self.editor.set_group(doc, g, prev, nxt)
        else:
            self._editor_pending = True
            # Keep the selected unit current even while its hidden rows await refresh.
            self.editor.doc, self.editor.group = doc, g
        if not self.watch_mode or self.source.isVisible():
            self._source_pending = False
            self.source.show_group(g)
        else:
            self._source_pending = True

    def _show_group(self, g: Group, *, seek: bool, keep_view: bool = False) -> None:
        doc, gid = self.session.doc, g.id
        self._refresh_group_panels(g)
        self.wave.current_gid = gid
        s, e = doc.group_time(g)
        # After an edit the zoomed-in view stays put while any part of the group is still on screen.
        if not (keep_view and s < self.wave.t0 + self.wave.span and self.wave.t0 < e):
            self.wave.show_range(s, e)
        self.watch.set_selected(gid)
        self._update_transform_action()
        self._refresh_k()
        self._update_position_actions()
        if seek:
            self.player.seek(s / 1000.0)

    def _on_table_group(self, gid: str) -> None:
        self.select_group(gid, seek=True, preserve_selection=True, defer=True, scroll=False)

    def _set_timing_mode(self, mode: str) -> None:
        self.wave_panel.set_timing_mode(mode)
        mode = self.wave.timing_mode
        self.timing_actions[mode].setChecked(True)
        self.settings.setValue("wave/timing_mode", mode)

    def _on_wave_group(self, gid: str) -> None:
        if not self.session:
            return
        self._stop_at = self._step_select_gid = None
        self.player.pause()
        self.select_group(gid, seek=True)
        if self.watch_mode:
            self._show_watch_wave(True)

    def _frame_display_changed(self, on):
        self.settings.setValue("watch/frames", on)
        self.watch.frames_visible = bool(on)
        self.watch.redraw()

    def _choose_frame_color(self):
        from PySide6.QtWidgets import QColorDialog

        from .watch import FRAME_DEFAULT

        current = QColor(str(self.settings.value("watch/frame_color", FRAME_DEFAULT)))
        chosen = QColorDialog.getColor(current, self, "字幕外框颜色")
        if chosen.isValid():
            self._apply_frame_color(chosen.name().upper())

    def _apply_frame_color(self, hex_rgb: str) -> None:
        from .watch import ass_rgb

        self.settings.setValue("watch/frame_color", hex_rgb)
        self.watch.frame_color = ass_rgb(hex_rgb)
        self.watch.redraw()

    def _speaker_display_changed(self, on):
        self.settings.setValue("watch/speakers", on)
        self.watch.redraw()

    def _speaker_lane_changed(self, lane):
        self.settings.setValue("watch/speaker_lane", lane)
        self.watch.redraw()

    def _open_speaker_picker(self, mid, pos):
        if not self.session or mid not in self.session.doc.members:
            return
        self.player.pause()
        if self._speaker_picker is not None:
            self._speaker_picker.close()
        session = self.session
        member = session.doc.members[mid]
        from subkit.core.people import shown_speaker_label
        lane = self.settings.value("watch/speaker_lane", "original")
        label = shown_speaker_label(session.doc, session.doc.group(member.group), lane, session.editor.character_book)
        def affected_count(whole, same, fill):
            gids = {member.group}
            if same:
                gids.update(session.doc.members[target].group for target in session.editor.same_label_candidates(
                    mid, whole_group=whole, shown_label=label, preferred_lane=lane))
            if fill:
                gids.update(session.editor.fill_down_candidates(mid, shown_label=label, preferred_lane=lane))
            return len(gids)
        picker = SpeakerPicker(self, session.editor.character_book, member, self._recent_speakers,
                               label=label, settings=self.settings, affected_count=affected_count,
                               single_count=len(session.editor.same_label_candidates(mid, whole_group=False,
                                   shown_label=label, preferred_lane=lane)),
                               whole_count=len(session.editor.same_label_candidates(mid, whole_group=True,
                                   shown_label=label, preferred_lane=lane)))
        self._speaker_picker = picker
        picker.destroyed.connect(lambda: self._speaker_picker is picker and setattr(self, "_speaker_picker", None))
        picker.chosen.connect(lambda cid, whole, same, fill: self.session is session and
            self._set_speaker(mid, cid, whole, same, fill, shown_label=label, preferred_lane=lane))
        picker.show()
        picker.place_at(pos)
        picker.activateWindow()
        picker.search.setFocus()

    def _set_speaker(self, mid, cid, whole_group, same_label=False, fill_down=False, *,
                     shown_label=None, preferred_lane=None):
        session = self.session
        if not session or mid not in session.doc.members:
            return
        from subkit.core.people import needs_speaker, shown_speaker_label
        member = session.doc.members[mid]
        group = session.doc.group(member.group)
        lane = preferred_lane if preferred_lane is not None else self.settings.value("watch/speaker_lane", "original")
        label = shown_label if shown_label is not None else shown_speaker_label(
            session.doc, group, lane, session.editor.character_book)
        options = dict(whole_group=whole_group, same_label=same_label, fill_down=fill_down,
                       shown_label=label, preferred_lane=lane)
        fields = None
        if cid == "new":
            from subkit.core.people import Character, NameForm

            members = list(session.doc.members_of(group))
            labelled = [m for m in members if needs_speaker(group, m)
                        and not session.editor.character_book.speaker_ignored(m)
                        and (m.speaker or {}).get("label") == label and label]
            source = next((m for m in labelled if m.lane == lane), labelled[0] if labelled else member)
            character = Character("chr_new", {source.lang: NameForm(given=label)}, aliases=[label] if label else [])
            dialog = CharacterDetailsDialog(self, character, self.project.languages, label)
            try:
                if dialog.exec() != dialog.DialogCode.Accepted or self.session is not session:
                    self.select_group(member.group, seek=False)
                    return
                fields = dialog.changes()
            finally:
                dialog.deleteLater()
        def apply():
            nonlocal cid
            with session.lock:
                if fields is None:
                    session.editor.set_speaker([mid], cid, **options)
                    return
                book = session.editor.character_book
                def create_and_assign(tx):
                    nonlocal cid
                    tx.keeps_text = True
                    tx.external(book.snapshot, book.restore)
                    cid = book.add_character(**fields).id
                    session.editor._assign_speaker(tx, [mid], cid, **options)
                    book.link_document(session.doc, tx=tx)
                    session.editor._refresh_character_readings(tx, book)
                session.editor.history.run("新建人物并设置说话人", create_and_assign)
        if self._edit(apply):
            if cid not in (None, "crowd", "nobody"):
                self._recent_speakers = [cid, *(old for old in self._recent_speakers if old != cid)][:20]
            # Refresh committed choices and recent ordering without rebuilding the editor.
            for visible_mid, combo in self.editor.speaker_combos.items():
                combo.refresh_choices(session.editor.character_book, session.doc.members[visible_mid], self._recent_speakers)

    def check_self_references(self):
        if not self.session:
            return
        dialog = SelfCheckDialog(self, self.session)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _on_time(self, t: float) -> None:
        if self._restored_position_time and t != self._restored_position_time[2]:
            self._restored_position_time = None
        ms = t * 1000
        self.lbl_time.setText(f"{int(ms) // 3600000}:{int(ms) // 60000 % 60:02d}:{ms / 1000 % 60:05.2f}")
        self.wave.set_playhead(ms, follow=not self.player.paused)
        self.watch.on_time(t)
        self.tapper.on_time(t)
        if self._stop_at is not None and t >= self._stop_at:
            stop, self._stop_at = self._stop_at, None
            step_gid, self._step_select_gid = self._step_select_gid, None
            self.player.pause()
            if self._group_playback_gid:
                # A delayed time-pos event can overshoot: restore the still frame.
                self.player.seek(stop)
                if step_gid:
                    self.select_group(step_gid, seek=False)
                self.watch.set_selected(self._group_playback_gid)
            return
        # Following the playhead would rebind the text boxes and drop text still being typed.
        if (self._follow and not self.player.paused and self.session
                and self._group_playback_gid is None and self._pending_group is None
                and not self.editor.has_uncommitted()):
            g = self._group_at(ms)
            if g and g.id != self.current_gid:
                multi = len(self.table.selected_gids()) > 1
                self.select_group(g.id, seek=False, preserve_selection=multi, scroll=not multi)

    def _group_at(self, ms: float) -> Optional[Group]:
        doc = self.session.doc
        best = None
        for g in doc.groups:
            if g.kind not in ("dialogue", "song"):
                continue
            if self.edit_layout == "karaoke" and g.kind != "song":
                continue
            s, e = doc.group_time(g)
            if s <= ms <= e:
                rank = (s, g.kind == "dialogue")
                if best is None or rank > best_rank:
                    best, best_rank = g, rank
        return best

    def _space(self) -> None:
        if self._typing():
            return
        if self.act_step_playback.isChecked() and self.player.paused:
            self._step_playback()
            return
        if self.watch_mode and self.player.paused:
            self._video_resume()          # 观剧模式: Space = continue watching (deselect, fold the waveform)
            return
        self.toggle_play()

    def _key_r(self) -> None:
        if not self._typing():
            self.play_group()

    def toggle_play(self) -> None:
        self._group_playback_gid = None
        self._stop_at = self._step_select_gid = None
        self.player.toggle()

    def _set_step_playback(self, on: bool) -> None:
        self.settings.setValue("playback/step", on)
        self.wave_panel.btn_step.blockSignals(True)
        self.wave_panel.btn_step.setChecked(on)
        self.wave_panel.btn_step.blockSignals(False)

    def _frame_duration(self) -> float:
        fps = 24000 / 1001
        try:
            value = float(self.player.mpv.get("container-fps") or fps) if self.player.mpv else fps
            if 0 < value < float("inf"):
                fps = value
        except (RuntimeError, ValueError, TypeError):
            pass
        return 1 / fps

    def _step_playback(self) -> None:
        if not self.session:
            return
        self._flush_group_selection()
        doc = self.session.doc
        ms, frame_ms = self.player.time * 1000, self._frame_duration() * 1000
        groups = [g for g in doc.groups if g.kind in ("dialogue", "song")
                  and (self.edit_layout != "karaoke" or g.kind == "song")]

        # mpv reports a frame-aligned time that can lie a few ms before the stop point (the group start
        # when there is no fade-in); without this margin the same group counts as "next" again and Space
        # replays a few ms and snaps back.
        early = frame_ms + 30

        def contains(g):
            s, e = doc.group_time(g)
            return s - early <= ms < e - frame_ms - 10

        inside = [g for g in groups if contains(g)]
        current = next((g for g in inside if g.id == self.current_gid), None)
        if current is None and inside:
            current = max(inside, key=lambda g: (doc.group_time(g)[0], g.kind == "dialogue"))
        self._stop_at = self._step_select_gid = None
        self._group_playback_gid = None
        if current is not None:
            self.select_group(current.id, seek=False)
            s, e = doc.group_time(current)
            self._stop_at = max(s, e - frame_ms) / 1000
            self._group_playback_gid = current.id
        else:
            later = sorted((g for g in groups if doc.group_time(g)[0] > ms + early),
                           key=lambda g: doc.group_time(g)[0])
            if not later:
                self.statusBar().showMessage("已经是最后一组", 4000)
                return
            target = later[0]
            s, e = doc.group_time(target)
            self._stop_at = fully_visible_time(
                target.id, s, e, frame_ms,
                events=self.watch.ass.events if self.watch.ass is not None else (),
                owners=self.watch.owners, members=(doc.members[mid] for mid in target.members)) / 1000
            self._step_select_gid = self._group_playback_gid = target.id
        self.player.play()

    def play_group(self, *, show_wave: bool = True) -> None:
        if not (self.session and self.current_gid):
            return
        self._flush_group_selection()
        s, e = self.session.doc.group_time(self.session.doc.group(self.current_gid))
        frame = self._frame_duration()
        self._step_select_gid = None
        self._group_playback_gid = self.current_gid
        self._stop_at = max(0.0, s / 1000.0, e / 1000.0 - frame)
        self.watch.set_selected(self.current_gid)
        if self.watch_mode and show_wave:
            self._show_watch_wave(True)
        self.player.seek(max(0.0, s / 1000.0 - 0.5))
        self.player.play()

    def _play_neighbour(self, d: int) -> None:
        """Select the previous / next group and play it with a little lead-in (like R); in F11 (全屏观看) the
        waveform does not pop up for it (that read as falling back to F10)."""
        if self._typing() or not self.session:
            return
        doc = self.session.doc
        groups = [g for g in doc.groups if g.kind in ("dialogue", "song")]
        if not groups:
            return
        ids = [g.id for g in groups]
        if self.current_gid in ids:
            i = ids.index(self.current_gid) + d
        else:                                          # nothing selected: relative to the playhead
            t = self.player.time * 1000
            later = [k for k, g in enumerate(groups) if doc.group_time(g)[0] > t]
            i = (later[0] if later else len(ids)) + (0 if d > 0 else -1)
        if not 0 <= i < len(ids):
            self.player.message("已经是第一组" if d < 0 else "已经是最后一组")
            return
        self.select_group(ids[i], seek=False)
        self.play_group(show_wave=not (self.watch_mode and not self.wave_panel.pinned()))

    def _step_group(self, d: int) -> None:
        if not (self.session and self.current_gid):
            return
        groups = [g for g in self.session.doc.groups if g.kind in ("dialogue", "song")]
        ids = [g.id for g in groups]
        if self.current_gid not in ids:
            return
        i = max(0, min(len(ids) - 1, ids.index(self.current_gid) + d))
        self.select_group(ids[i], seek=True, defer=True)

    def _confirm(self, *, move: bool) -> None:
        if not (self.session and self.current_gid):
            return
        self.session.editor.confirm(self.current_gid)
        if move:
            doc = self.session.doc
            i = doc.group_index(self.current_gid)
            nxt = next((g for g in doc.groups[i + 1:] if g.kind == "dialogue" and g.state != "confirmed"), None)
            if nxt is not None:
                self.select_group(nxt.id, seek=False)
                self.play_group()

    def confirm_next(self) -> None:
        self._confirm(move=True)

    def _arrow(self, d: int, *, ctrl: bool) -> None:
        """Edit mode: ← / → step a frame, Ctrl+← / → play the previous / next group (like R).  F10 / F11 swap them
        (user 2026-10-14: in watch mode the group jump is what the arrows are for)."""
        if bool(self.watch_mode) != ctrl:
            self._play_neighbour(d)
        else:
            self._frame(d > 0)

    def _frame(self, forward: bool) -> None:
        if not self._typing():
            self.player.frame_step(forward)

    # ================================================================ edits
    def _autotime(self) -> None:
        session = self.session
        if session is None:
            return
        if not session.episode.video:
            self._help_message("autotime", lambda parent: QMessageBox.information(
                self, "自动调轴", "这一集没有视频，不能按音频调轴。"))
            return
        features = self.wave.audio_features()
        if features is None:
            self._help_message("autotime", lambda parent: QMessageBox.information(
                self, "自动调轴", "音频还在读取，读完后再试。"))
            return
        dialog = None
        try:
            with autotime_busy(self):
                preview = plan_autotime(session.doc, session.episode.sources, features, AutoTimeSettings())
                counts = session.editor.apply_autotime(preview)
                transaction = session.editor.history.done[-1] if counts[0] else None
            dialog = AutoTimeResultDialog(session, preview, counts, transaction, features, self)
            dialog.exec()
        except Exception as exc:
            self._help_message("autotime", lambda parent: QMessageBox.warning(self, "自动调轴", f"自动调轴未完成：{exc}"))
        finally:
            if dialog is not None:
                dialog.deleteLater()

    def _unify_lane_times(self, lane: str) -> None:
        """时间轴 → 统一两种语言的时间 ▸: dialogue goes to one subtitle file's own times (one undo step)."""
        if not self.session:
            return
        from subkit.languages import lane_label

        report = {}
        if not self._edit(lambda: report.update(self.session.editor.unify_lane_times(lane))):
            return
        label = lane_label(lane, self.project.languages)
        kept = f"确认过或手调过时间的 {report['protected']} 组没动。" if report.get("protected") else ""
        if not report.get("lines"):
            text = f"对白已经都是{label}字幕的时间。" + kept
        else:
            text = (f"已按{label}字幕的时间调整 {report['groups']} 组、{report['lines']} 行"
                    + (f"（其中 {report['shifted']} 行没有配对，按附近的时间差一起挪）" if report["shifted"] else "")
                    + "。" + kept + "Ctrl+Z 一步撤销。")
        self.statusBar().showMessage(text, 10000)
        self._help_message("unify-times", QMessageBox.information, APP_NAME, text)

    def _search_passages(self, store: PassageStore, passages: list[dict]) -> list:
        features = self.wave.audio_features()
        if features is None:
            return [(p, None) for p in passages if not store.has_audio(p["id"])]
        previous = self.statusBar().currentMessage()
        self.statusBar().showMessage("正在查找重复段落…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.statusBar().repaint()
            if self._passage_features is not features:
                index = EpisodeIndex(features.spectrogram)
                self._passage_features, self._passage_index = features, index
            return [(p, self._passage_index.find(store.template(p['id'])) if store.has_audio(p['id']) else None)
                    for p in passages]
        finally:
            QApplication.restoreOverrideCursor()
            self.statusBar().showMessage(previous)

    def _find_passages(self) -> None:
        if not self.project or not self.session:
            return
        if self.wave.audio_features() is None and all(PassageStore(self.project).has_audio(p["id"])
                                                     for p in PassageStore(self.project).list()):
            QMessageBox.information(self, "重复段落", "音频还在读取，读完后再试。")
            return
        dialog = None
        try:
            store = PassageStore(self.project)
            passages = store.list()
            if not passages:
                QMessageBox.information(self, "重复段落", "重复段落库是空的：先在做好的一集里选中 OP/ED 等几组，用“记为重复段落”。")
                return
            results = self._search_passages(store, passages)
            dialog = PassagesDialog(self.session, results, self.player, self)
            dialog.exec()
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "重复段落", str(exc))
        finally:
            if dialog is not None:
                dialog.deleteLater()

    def _auto_process_key(self) -> tuple[str, str]:
        return self.project.data['id'], self.session.episode.id

    def _auto_process_mute_key(self, key: tuple[str, str]) -> str:
        return f"auto_process/muted/{key[0]}/{key[1]}"

    def _drop_auto_process(self) -> None:
        self._auto_process_timer.stop()
        self._auto_process_pending = None

    def _queue_auto_process(self) -> None:
        session = self.session
        if session is None or session is not self._passage_audio_session or self.project is None:
            return
        key = self._auto_process_key()
        if key in self._auto_process_asked or self._auto_process_pending is not None:
            return
        if self.settings.value(self._auto_process_mute_key(key), False, type=bool):
            return
        names, offset_ms, segmented = [], None, False
        try:
            store = PassageStore(self.project)
            passages = store.list()
            results = self._search_passages(store, passages) if passages else []
            names = [passage_name(p) for p, result in results if result is not None
                     if any(not already_applied(session.doc, p['id'], m.start_ms) for m in result.matches)]
        except (OSError, ValueError, KeyError) as exc:
            self.statusBar().showMessage(str(exc))
        if self.wave.audio_features() is not None and self.session is session:
            try:
                with autotime_busy(self):
                    preview = plan_autotime(session.doc, session.episode.sources, self.wave.audio_features())
                if preview.hint.show:
                    offset_ms = preview.hint.max_offset_ms
                    segmented = len(preview.segments) > 1
            except (OSError, ValueError, KeyError) as exc:
                self.statusBar().showMessage(str(exc))
        if (names or offset_ms is not None) and self.session is session:
            self._auto_process_pending = session, key, offset_ms, segmented, names
            self._auto_process_timer.start()

    def _try_auto_process(self) -> None:
        if self._episode_job is not None:
            return
        pending = self._auto_process_pending
        if pending is None:
            self._auto_process_timer.stop()
            return
        session, key, offset_ms, segmented, names = pending
        if self.session is not session or session is not self._passage_audio_session or session._closed:
            self._drop_auto_process()
            return
        if (QApplication.activeModalWidget() is not None or QApplication.activePopupWidget() is not None
                or any(w.isVisible() and w.windowType() == Qt.WindowType.Tool for w in QApplication.topLevelWidgets())
                # A text box keeps focus while idle in the edit layout; only recent typing defers the prompt.
                or (isinstance(QApplication.focusWidget(), (QLineEdit, QPlainTextEdit, QTextEdit))
                    and time.monotonic() - getattr(self, "_last_key_time", 0.0) < 3.0)
                or self.watch.transform is not None or self.tapper.active or not self.player.paused):
            return
        self._drop_auto_process()
        if key in self._auto_process_asked or self.settings.value(self._auto_process_mute_key(key), False, type=bool):
            return
        self._auto_process_asked.add(key)
        dialog = EpisodeAutoProcessDialog(offset_ms, segmented, names, self)
        try:
            dialog.exec()
            if dialog.mute.isChecked():
                self.settings.setValue(self._auto_process_mute_key(key), True)
            action = dialog.action
        finally:
            dialog.hide()
            dialog.deleteLater()
        if self.session is not session or session._closed:
            return
        if action == "autotime":
            self._autotime()
        if names and action in ("autotime", "passages") and self.session is session and not session._closed:
            self._find_passages()
        if self.watch_mode and self.session is session:
            self.activateWindow()
            self.player.setFocus()

    def _record_passage(self, gids: list[str]) -> None:
        if not self.session or not self.project or not gids:
            QMessageBox.information(self, "记为重复段落", "先在列表里选中这一段的几组。")
            return
        features = self.wave.audio_features()
        if features is None:
            QMessageBox.information(self, "记为重复段落", "音频还在读取，读完后再试。")
            return
        dialog = None
        try:
            doc = self.session.doc
            category = next((m.style.split('.', 1)[0] for g in doc.groups if g.id in gids and g.kind == "song"
                             for m in doc.members_of(g) if m.style.split('.', 1)[0] in ("OP", "ED", "IN")), "NA")
            passage = snapshot(doc, gids, name="", category=category, version="", episode_id=self.session.episode.id)
            first = next((m["visible"] for g in passage["groups"] for m in g["members"]
                          if m["lane"] == "original"), "")
            passage["name"] = first[:20]
            template = template_from(features, passage["start"], passage["start"] + passage["duration"])
            idx = kara.KIndex(doc)
            states = []
            for g in doc.groups:
                if g.id not in gids:
                    continue
                for m in doc.members_of(g):
                    if m.role == "comment" or not visible_map(m)[0].text.strip():
                        continue
                    view = kara.effective(doc, m, idx=idx) if idx.has(m) else None
                    full = view is not None and bool(view.beats) and all(b.timed for b in view.beats)
                    states.append("无" if view is None else "还有占位拍" if not full
                                  else "有（自动，未确认）" if view.state == "suggested" else "有")
            dialog = RecordPassageDialog(passage, self, karaoke_states=states)
            if dialog.exec() != RecordPassageDialog.DialogCode.Accepted:
                return
            passage["name"], passage["category"], passage["version"] = dialog.values()
            store = PassageStore(self.project)
            existing = next((p for p in store.list() if p["name"] == passage["name"]
                             and p["version"] == passage["version"]), None)
            replace_id = None
            if existing is not None:
                choice = duplicate_choice(self, existing)
                if choice == "cancel":
                    return
                if choice == "replace":
                    replace_id = existing["id"]
            store.add(passage, template, replace_id=replace_id)
            self.statusBar().showMessage(f"已记为重复段落：{CATEGORY_NAMES[passage['category']]}《{passage['name']}》", 6000)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "记为重复段落", str(exc))
        finally:
            if dialog is not None:
                dialog.deleteLater()

    def _passage_library(self) -> None:
        if self.project is None:
            return
        dialog = None
        try:
            dialog = PassageLibraryDialog(PassageStore(self.project), self,
                                          video=self.session.episode.video if self.session else None,
                                          languages=self.project.languages,
                                          audio_track=((getattr(self.session.episode, "data", {}) or {}).get("media") or {}).get("audio_track", 0) if self.session else 0)
            dialog.exec()
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "重复段落库", str(exc))
        finally:
            if dialog is not None:
                dialog.deleteLater()

    def _update_convert_actions(self) -> None:
        gids = set(self.table.selected_gids())
        kinds = {g.kind for g in self.session.doc.groups if g.id in gids} if self.session else set()
        song_reason, dialogue_reason = conversion_reasons(kinds)
        if not self.session:
            song_reason = dialogue_reason = "先打开一集"
        set_action_reason(self.act_convert_song, song_reason)
        set_action_reason(self.act_convert_dialogue, dialogue_reason)
        set_action_reason(self.act_convert_sign, sign_conversion_reason(kinds) if self.session else "先打开一集")

    def _convert_groups(self, gids: list[str], to: str) -> None:
        if not self.session or not gids:
            return
        styles = self.session.episode.settings().get("merge", {}).get("styles") or MergeSettings().styles
        template_styles = [s.name for s in self.session.template.styles]
        plan = plan_conversion(self.session.doc, gids, to, styles=styles, template_styles=template_styles)
        if not plan.gids:
            return
        ed = self.session.editor
        if to == "song":
            d = ConvertToSongDialog(self.session.doc, gids, self, styles=styles,
                                    template_styles=template_styles, settings=self.settings)
            if d.exec() != ConvertToSongDialog.DialogCode.Accepted:
                return
            family = d.family()
            plan = d.plan
            if not self._edit(lambda: ed.convert_to_song(gids, family, styles=styles)):
                return
            self.settings.setValue("convert/song_family", family)
        elif to == "sign":
            answer = self._help_message("review", QMessageBox.question, "转为屏幕字",
                                        f"把选中的 {len(plan.gids)} 组转为屏幕字？\n使用来源样式和位置，不再提示说话人。（可以撤销）",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
            if not self._edit(lambda: ed.convert_to_sign(gids, styles=styles)):
                return
        else:
            answer = self._help_message("review" if plan.sign_groups else "karaoke", QMessageBox.question, "转为普通台词", dialogue_text(plan, styles=styles),
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
            if not self._edit(lambda: ed.convert_to_dialogue(gids, styles=styles)):
                return
        self._refresh_k()
        self.statusBar().showMessage(status_text(plan), 6000)

    def _edit(self, fn) -> bool:
        if self._episode_job is not None:
            return False
        if not self.session:
            return False
        try:
            fn()
        except Exception as exc:
            self.statusBar().showMessage(f"操作失败：{exc}", 8000)
            if self.watch_mode:
                self.player.message(f"操作失败：{exc}", 2500)
            traceback.print_exc()
            return False
        return True

    def _move(self, direction: str) -> None:
        if self.session and self.current_gid:
            self._edit(lambda: self.session.editor.move_phrase(self.current_gid, "original", direction))

    def _move_at_cursor(self, to_prev: bool) -> None:
        self._flush_group_selection()
        if not (self.session and self.current_gid):
            return
        pos = self.editor.jp_cursor()
        if pos is None:
            return
        self._edit(lambda: self.session.editor.move_phrase(self.current_gid, "original",
                                                           "head_to_prev" if to_prev else "tail_to_next", at=pos))

    def _on_action(self, name: str) -> None:
        self._flush_group_selection()
        if not (self.session and self.current_gid):
            return
        ed = self.session.editor
        gid = self.current_gid
        if name in ("hidden", "marked"):
            self._review_flag([gid], name, name not in self.session.doc.group(gid).flags)
        elif name in ("head_to_prev", "tail_to_next", "take_from_prev", "take_from_next"):
            self._move(name)
        elif name == "merge":
            if self._edit(lambda: ed.merge_with_next(gid)) and ed.merge_note:
                self.statusBar().showMessage(ed.merge_note, 6000)
        elif name == "split":
            pos = self.editor.jp_cursor()
            g = self.session.doc.group(gid)
            n_cn = sum(1 for m in g.members if self.session.doc.members[m].lane == "translation")
            new = None

            def do() -> None:
                nonlocal new
                new = ed.split(gid, jp_at=pos, cn_member_index=1 if n_cn >= 2 else None)
            self._edit(do)
        elif name == "unlink":
            self._edit(lambda: ed.unlink(gid))
        elif name == "confirm_next":
            self.confirm_next()

    def _on_text(self, mid: str, text: str) -> None:
        self._edit(lambda: self.session.editor.replace_editor_text(mid, text))

    def _add_language_text(self, gid, lane, text):
        self._apply_missing_language({gid: text}, lane)

    def _start_listening(self, gid):
        from subkit.ass.text import plain_text
        from subkit.core.listening import target_signature
        from subkit.listening import ListeningRequest
        from .views.ai_transcribe import ListeningDialog
        from .views.floating import show_activated

        if self.session is None or self.project is None:
            return
        if self._listening_context is not None:
            dialog = self._listening_context["dialog"]
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            return
        try:
            if self._kalign_context is not None or self._dialign_context is not None:
                raise ValueError("自动对轴正在使用识别模型，请完成或取消后再听译")
            session, ep = self.session, self.session.episode
            if not ep.video:
                raise ValueError("这一集没有视频，无法听译")
            track = (getattr(ep, "data", {}).get("media") or {}).get("audio_track")
            signature = target_signature(session.doc, gid, ep.video, track, self.project.languages)
            g = session.doc.group(gid)
            start, end = session.doc.group_time(g)
            known = tuple(plain_text(m.text) for m in session.doc.members.values()
                          if m.lane == "original" and m.role != "comment" and m.start < end + 5000 and m.end > start - 5000)
            cn = " ／ ".join(plain_text(m.text) for m in session.doc.members_of(g) if m.lane == "translation")
            request = ListeningRequest(str(ep.video), track, start, end, known_originals=known,
                                       translation=cn, semantic=True)
            factory = self.listening_dialog_factory or ListeningDialog
            dialog = factory(request, cn, self)
            context = self._listening_context = {"session": session, "project": self.project,
                "gid": gid, "signature": signature, "dialog": dialog, "adopting": False}
            timer = context["timer"] = QTimer(self)
            timer.setInterval(500)
            timer.timeout.connect(lambda: self._validate_listening(context))
            timer.start()
            dialog.before_adopt = lambda selected: self._adopt_listening(context, selected)
            dialog.finished.connect(lambda _code: self._finish_listening(context))
            self.player.pause()
            show_activated(dialog, dialog.text)
        except Exception as exc:
            self._help_message("ai-transcribe", QMessageBox.warning, "AI听译", str(exc))

    def _validate_listening(self, context):
        if self._listening_context is not context or context["adopting"]:
            return False
        from subkit.core.listening import target_signature
        try:
            if self.session is not context["session"] or self.project is not context["project"]:
                raise ValueError("已经切换工程或集数")
            ep = self.session.episode
            track = (getattr(ep, "data", {}).get("media") or {}).get("audio_track")
            current = target_signature(self.session.doc, context["gid"], ep.video, track, self.project.languages)
            if current != context["signature"]:
                raise ValueError("目标字幕、视频或音轨已经改变")
            return True
        except (KeyError, ValueError, OSError, TypeError) as exc:
            context["dialog"].invalidate(f"{exc}；请关闭后重新对目标句听译。")
            context["timer"].stop()
            return False

    def _adopt_listening(self, context, selected):
        if not self._validate_listening(context):
            return False
        from subkit.core.listening import adopt
        candidate, text, start, end = selected
        inserted = []
        context["adopting"] = True
        try:
            styles = self.session.episode.settings().get("merge", {}).get("styles")
            okay = self._edit(lambda: inserted.append(adopt(self.session.editor, context["gid"], candidate,
                text, start, end, context["dialog"].result, languages=self.project.languages, styles=styles)))
            if not okay or not inserted:
                if hasattr(context["dialog"], "status"):
                    context["dialog"].status.setText(self.statusBar().currentMessage() or "未能补入原文，请核对内容和时间。")
                return False
            self.select_group(context["gid"])
            self.statusBar().showMessage("已补入日语对白，可一步撤销；请核对文字、读音和时间。", 8000)
            return True
        finally:
            context["adopting"] = False

    def _finish_listening(self, context):
        context["timer"].stop()
        context["timer"].deleteLater()
        if self._listening_context is context:
            self._listening_context = None
        context["dialog"].deleteLater()

    def _close_listening(self):
        context = self._listening_context
        if context is None:
            return True
        dialog = context["dialog"]
        dialog.invalidate("正在关闭本集，听译已取消。")
        job = dialog.job
        if job is not None and not job.finished.is_set():
            job.cancel()
            # One direct child at a time; the inference worker additionally guards
            # library helpers in its Windows Job. Retain owners until run exits.
            if not job.finished.wait(2):
                self.statusBar().showMessage("正在停止听译进程，请稍后再关闭。", 6000)
                dialog.reject()
                return False
        dialog.reject()
        return True

    def _apply_missing_language(self, texts, lane, *, allow_empty=False):
        if self.session is None:
            return []
        styles = self.session.episode.settings().get("merge", {}).get("styles")
        inserted = []
        self._edit(lambda: inserted.extend(self.session.editor.add_missing_language(
            texts, lane, languages=self.project.languages, styles=styles, allow_empty=allow_empty)))
        return inserted

    def _add_language_dialog(self, gids, lane):
        if self.session is None:
            return
        from .views.missing_language import MissingLanguageDialog
        from .views.floating import show_activated

        session = self.session
        dialog = MissingLanguageDialog(session.doc, gids, lane, self.project.languages, self)
        dialog.accepted.connect(lambda: self.session is session and self._apply_missing_language(dialog.texts(), lane))
        dialog.finished.connect(dialog.deleteLater)
        if dialog.inputs:
            show_activated(dialog, dialog.inputs[0])
        else:
            dialog.deleteLater()

    def _on_member_time(self, mid: str, start: int, end: int) -> None:
        self._edit(lambda: self.session.editor.set_member_time(mid, start, end))

    def _nudge(self, edge: str, sign: int) -> None:
        if self._typing() or not (self.session and self.current_gid):
            return
        fps = 24000 / 1001
        try:
            v = self.player.mpv.get("container-fps") if self.player.mpv else None
            fps = float(v) if v else fps
        except Exception:
            pass
        step = int(round(1000 / fps)) * sign
        g = self.session.doc.group(self.current_gid)
        s, e = self.session.doc.group_time(g)
        if edge == "start":
            self._edit(lambda: self.session.editor.set_group_edge(self.current_gid, start=s + step))
        else:
            self._edit(lambda: self.session.editor.set_group_edge(self.current_gid, end=e + step))

    def _on_wave_time(self, ms: int, edge: str) -> None:
        """Aegisub-style: left click = start, right click = end of the current group (target from the toolbar)."""
        if self.edit_layout == "karaoke":
            return
        if not (self.session and self.current_gid):
            return
        gid, target = self.current_gid, self.wave_panel.timing_target()
        if edge == "start":
            self._edit(lambda: self.session.editor.set_group_time(gid, start=ms, target=target))
        else:
            self._edit(lambda: self.session.editor.set_group_time(gid, end=ms, target=target))
        self.player.message(f"{'开始' if edge == 'start' else '结束'} → {fmt_ms(ms)}")

    def _adopt_timing(self) -> None:
        if self.edit_layout == "karaoke":
            return
        self._flush_group_selection()
        self.wave_panel.sync_hint_padding()
        times = self.wave.suggested_times()
        if not (self.session and self.current_gid and times):
            return
        if self.session.doc.group(self.current_gid).kind != "dialogue":
            return
        self._edit(lambda: self.session.editor.set_group_time(
            self.current_gid, start=times[0], end=times[1], label="采用建议时间", source="adopted"))

    def _play_key(self, key: str) -> None:
        """Aegisub's play keys: S group, Q 0.5 s before, W 0.5 s after, D first 0.5 s, E last 0.5 s."""
        if self._typing() or not (self.session and self.current_gid):
            return
        s, e = self.session.doc.group_time(self.session.doc.group(self.current_gid))
        a, b = {"S": (s, e), "Q": (s - 500, s), "W": (e, e + 500), "D": (s, min(e, s + 500)),
                "E": (max(s, e - 500), e)}[key]
        if key in ("S", "D", "E"):
            # Like R: stop one frame before the group ends and keep it on screen and selected.
            b = max(a, min(b, e - self._frame_duration() * 1000))
            self._play_range(a, b, hold_gid=self.current_gid)
        else:
            self._play_range(a, b)

    def _play_range(self, a_ms: float, b_ms: float, *, hold_gid: Optional[str] = None) -> None:
        self._step_select_gid = None
        self._group_playback_gid = hold_gid
        if hold_gid:
            self.watch.set_selected(hold_gid)
        self.player.seek(max(0.0, a_ms) / 1000.0)
        self._stop_at = b_ms / 1000.0
        self.player.play()

    def delete_member(self) -> None:
        if not (self.session and self.current_gid):
            return
        g = self.session.doc.group(self.current_gid)
        jp = [m for m in g.members if self.session.doc.members[m].lane == "original"]
        if jp:
            self._edit(lambda: self.session.editor.delete_member(jp[-1]))

    def delete_group(self) -> None:
        if self.session and self.current_gid:
            self._delete_groups([self.current_gid])

    def delete_selected(self) -> None:
        """Del: the groups selected in the list (Shift / Ctrl + click for several), or the current group in
        观剧模式 / the waveform — after a prompt."""
        if not self.session or self._typing():
            return
        if self.watch_mode or not self.table.isVisible():
            gids = [self.current_gid] if self.current_gid else []
        else:
            gids = self.table.selected_gids()
        if gids:
            self._delete_groups(gids)

    def _group_line(self, gid: str) -> str:
        from subkit.core.readings import visible_map

        doc = self.session.doc
        g = doc.group(gid)
        s, _e = doc.group_time(g)
        cn = " ".join(visible_map(m)[0].text for m in doc.members_of(g) if m.lane == "translation")
        jp = " ".join(visible_map(m)[0].text for m in doc.members_of(g) if m.lane == "original")
        text = " ／ ".join(x for x in (cn, jp) if x) or "（空）"
        return f"{fmt_ms(s)}  {text[:40]}"

    def _after(self, gids: list[str]) -> Optional[str]:
        """The group to select once these are gone: the next one after them, else the one before."""
        order = [g.id for g in self.session.doc.groups]
        gone = set(gids)
        idx = [order.index(g) for g in gids if g in order]
        if not idx:
            return self.current_gid
        nxt = next((order[i] for i in range(max(idx) + 1, len(order)) if order[i] not in gone), None)
        return nxt or next((order[i] for i in range(min(idx) - 1, -1, -1) if order[i] not in gone), None)

    def _ask(self, text: str) -> bool:
        r = QMessageBox.question(self, APP_NAME, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                 QMessageBox.StandardButton.Yes)
        return r == QMessageBox.StandardButton.Yes

    def _delete_groups(self, gids: list) -> None:
        if not self.session or not gids:
            return
        lines = [self._group_line(g) for g in gids[:8]] + (["…"] if len(gids) > 8 else [])
        if not self._ask(f"删除 {len(gids)} 组（原声和翻译都删）？\n\n" + "\n".join(lines) + "\n\n可以 Ctrl+Z 撤销。"):
            return
        self.current_gid = self._after(gids)
        if self._edit(lambda: self.session.editor.delete_groups(gids)):
            self.statusBar().showMessage(f"已删除 {len(gids)} 组（Ctrl+Z 撤销）", 5000)
            self.player.message(f"已删除 {len(gids)} 组（Ctrl+Z 撤销）", 1500)

    def _delete_lane(self, gids: list, lane: str) -> None:
        if not self.session or not gids:
            return
        doc = self.session.doc
        n = sum(1 for gid in gids for m in doc.group(gid).members if doc.members[m].lane == lane)
        name = lane_label(lane, self.project.languages)
        if n == 0:
            self.statusBar().showMessage(f"选中的组里没有{name}行", 4000)
            return
        empty = [gid for gid in gids if all(doc.members[m].lane == lane for m in doc.group(gid).members)]
        extra = f"（其中 {len(empty)} 组只有{name}，整组会删掉）" if empty else ""
        if not self._ask(f"删除 {len(gids)} 组里的 {n} 行{name}{extra}？可以 Ctrl+Z 撤销。"):
            return
        if empty and self.current_gid in empty:
            self.current_gid = self._after(empty)
        if self._edit(lambda: self.session.editor.delete_lane(gids, lane)):
            self.statusBar().showMessage(f"已删除 {n} 行{name}（Ctrl+Z 撤销）", 5000)

    def _review_flag(self, gids: list[str], flag: str, value: bool) -> None:
        if self.session and gids:
            self._edit(lambda: self.session.editor.set_group_flag(gids, flag, value))

    def _mark_note_dialog(self, gids: list[str]) -> None:
        if not self.session or not gids:
            return
        dialog = QInputDialog(self)
        dialog.help_anchor = "hide-mark"
        dialog.setWindowTitle("编辑标记备注")
        dialog.setLabelText("备注（留空表示没有备注；写到所有选中的组）")
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
        notes = {self.session.doc.group(gid).extra.get("mark_note", "") for gid in gids}
        dialog.setTextValue(next(iter(notes)) if len(notes) == 1 else "")
        app_help.dialog_buttons(dialog)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._edit(lambda: self.session.editor.set_mark_note(gids, dialog.textValue()))

    def _export_review_list(self) -> None:
        if self.project is None or self._episode_job is not None:
            return
        from subkit.core.review_list import collect_review, write_review

        try:
            if self.session:
                self.session.save()
            episodes = collect_review(self.project)
            if not episodes:
                self._help_message("hide-mark", QMessageBox.information, APP_NAME, "工程里没有标记或隐藏的句子")
                return
            default = self.project.root / f"复查清单-{time.strftime('%Y%m%d-%H%M%S')}.html"
            dialog = QFileDialog(self, "导出标记和隐藏的句子", str(default), "HTML (*.html)")
            dialog.help_anchor = "hide-mark"
            dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
            dialog.setDefaultSuffix("html")
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            path = Path(dialog.selectedFiles()[0])
            write_review(episodes, path)
        except Exception as exc:
            self._help_message("hide-mark", QMessageBox.warning, APP_NAME, f"导出失败：{exc}")
            return
        if self._help_message("hide-mark", QMessageBox.question, APP_NAME, "用浏览器打开吗？") == QMessageBox.StandardButton.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _confirm_groups(self, gids: list, value: bool) -> None:
        if self.session and gids:
            self._edit(lambda: self.session.editor.confirm_many(gids, value))

    def undo(self) -> None:
        if self._episode_job is not None:
            return
        if self.session and not self._typing():
            tx = self.session.editor.history.undo()
            if tx:
                self.statusBar().showMessage(f"撤销：{tx.label}", 3000)

    def redo(self) -> None:
        if self._episode_job is not None:
            return
        if self.session and not self._typing():
            tx = self.session.editor.history.redo()
            if tx:
                self.statusBar().showMessage(f"重做：{tx.label}", 3000)

    # ================================================================ readings
    def _engine(self):
        s = self.session
        if s.engine is None:
            s.engine = s.episode.reading_engine()
        return s.engine

    def _reading_scope(self, mid: str, a: int, b: int) -> dict:
        """The member's text cut into words, for merging / splitting the range in the reading popup."""
        from subkit.core.readings import reading_spans, visible_map

        m = self.session.doc.members[mid]
        vt, _ids = visible_map(m)
        return {"text": vt.text, "a": a, "b": b, "spans": reading_spans(self.session.doc, m, self._engine())}

    def _ruby_settings(self) -> None:
        if not (self.project and self.session):
            return
        dialog = RubySettingsDialog(self.project.settings()["ruby"], self)
        try:
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            self.project.data.setdefault("settings", {}).setdefault("ruby", {})["max_width"] = dialog.values()
            self.project.save()
            self.session.request_preview()
        except Exception as exc:
            QMessageBox.warning(self, "振假名设置", f"操作失败：{exc}")
        finally:
            dialog.deleteLater()

    def _pinyin_settings(self) -> None:
        if not (self.project and self.session):
            return
        dialog = PinyinSettingsDialog(self.project.pinyin_mode, self)
        try:
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            previous = self.statusBar().currentMessage()
            self.statusBar().showMessage("正在应用拼音设置……")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.session.editor.set_pinyin_mode(self.project, dialog.mode())
            finally:
                QApplication.restoreOverrideCursor()
                self.statusBar().showMessage(previous)
        except Exception as exc:
            QMessageBox.warning(self, "拼音", f"操作失败：{exc}")
        finally:
            dialog.deleteLater()

    def _on_pinyin(self, mid: str, a: int, b: int, pos: QPoint) -> None:
        from subkit.core.readings import anchor_positions, visible_map
        from subkit.textutil import is_han

        if not (self.session and self.project) or self.project.data.get("settings", {}).get("pinyin", {}).get("mode", "off") == "off":
            return
        doc = self.session.doc
        m = doc.members.get(mid)
        if m is None or not m.lang.lower().startswith("zh"):
            return
        vt, ids = visible_map(m)
        base = vt.text[a:b]
        if not (0 <= a < b <= len(ids)) or not all(is_han(ch) for ch in base):
            return
        anns = [(anchor_positions(ann, ids), ann) for ann in doc.annotations.values()
                if ann.member == mid and ann.kind == "pinyin"]
        ann = next((ann for rng, ann in anns if rng == (a, b)), None)
        current = ann.text if ann and ann.status != "hidden" else ""
        pop = ReadingPopup(base, current, ann.candidates if ann else [], ann.origin if ann else "manual",
                           ann.status if ann else "", self, kind="pinyin")
        if b - a > 1:
            prefill = " ".join(ann.text for rng, ann in sorted(anns, key=lambda item: item[0] or (-1, -1))
                               if rng and a <= rng[0] and rng[1] <= b)
            pop.edit.setText(prefill)
            pop.edit.selectAll()
        pop.chosen.connect(lambda text, scope: self._apply_pinyin(mid, a, b, text, scope))
        pop.hideRequested.connect(lambda: self._hide_pinyin(mid, a, b))
        pop.popup_at(pos)

    def _hide_pinyin(self, mid: str, a: int, b: int) -> None:
        from subkit.core.readings import anchor_positions, visible_map

        ed = self.session.editor
        _vt, ids = visible_map(self.session.doc.members[mid])
        aids = [ann.id for ann in self.session.doc.annotations.values() if ann.member == mid and ann.kind == "pinyin"
                and (rng := anchor_positions(ann, ids)) and a <= rng[0] and rng[1] <= b]
        def apply(tx):
            tx.keeps_text = True
            for aid in aids:
                ann = tx.annotation(aid)
                ann.status, ann.origin = "hidden", "manual"
        if aids:
            self._edit(lambda: ed.history.run("不标注", apply))

    def _apply_pinyin(self, mid: str, a: int, b: int, reading: str, scope: int) -> None:
        from subkit.core.readings import visible_map

        ed = self.session.editor
        if scope:
            word = visible_map(self.session.doc.members[mid])[0].text[a:b]
            preview = ed.preview_pinyin_rule(word, reading)
            changes = [item for item in preview if item["changes"]]
            kept = [item for item in preview if item["kept"]]
            label = "整个工程（拼音规则）" if scope == 2 else "这一集（拼音规则）"
            lines = [f"「{word}」→ {reading}，范围：{label}", f"本集会更新 {len(changes)} 处自动拼音："]
            lines += [f"　第 {self.session.doc.group_index(self.session.doc.members[item['member']].group) + 1} 组：{item['text'][:30]}"
                      for item in changes[:10]]
            if kept:
                lines.append(f"另有 {len(kept)} 处是你确认过或手动输入的拼音，保持不变。")
            if scope == 2:
                lines.append("其他集：以后打开或刷新自动拼音时使用这条规则。")
            lines.append("应用吗？（可以 Ctrl+Z 撤销）")
            if self._help_message("pinyin", QMessageBox.question, "拼音", "\n".join(lines)) != QMessageBox.StandardButton.Yes:
                return
        self._edit(lambda: ed.apply_pinyin_rule(mid, a, b, reading, scope=("here", "episode", "project")[scope],
                                               project=self.project))

    def _on_reading(self, aid: str, pos: QPoint) -> None:
        from subkit.core.readings import anchor_positions, visible_map

        doc = self.session.doc
        a = doc.annotations.get(aid)
        if a is None:
            return
        mid = a.member
        _vt, ids = visible_map(doc.members[mid])
        rng = anchor_positions(a, ids)
        if a.kind == "pinyin":
            if rng:
                self._on_pinyin(mid, *rng, pos)
            return
        scope = self._reading_scope(mid, *rng) if rng else None
        candidates = self._reading_candidates(mid, rng, a.candidates)
        pop = ReadingPopup(a.base, a.text if a.status != "hidden" else "", candidates, a.origin, a.status, self,
                           scope=scope, can_split=rng is not None,
                           candidates_for=(lambda x, y: self._merged_reading_candidates(mid, scope["text"], x, y))
                           if scope and doc.members[mid].lang.lower().split("-")[0] == "ja" else None,
                           kanji_splitter=lambda text, reading: split_kanji_reading(text, reading, self._engine().analyzer))
        if doc.members[mid].lang.lower().split("-")[0] == "ja":
            pop.typedChosen.connect(self._remember_typed_reading)
        pop.chosen.connect(lambda t, sc: self._apply_reading(mid, rng, t, sc, aid=aid))
        pop.chosenRange.connect(lambda x, y, t, sc: self._apply_reading(mid, (x, y), t, sc))
        split_session = self.session
        pop.wordSplitRequested.connect(lambda x, y, sc: QTimer.singleShot(
            0, lambda: self.session is split_session and self._on_word_split(mid, x, y, sc)))
        pop.hideRequested.connect(lambda: self._edit(lambda: self.session.editor.hide_annotation(aid)))
        pop.bracketToText.connect(lambda: self._edit(lambda: self.session.editor.bracket_to_text([aid], engine=self._engine())))
        if rng:
            pop.splitRequested.connect(lambda: self._split_reading(mid, *rng))
            session = self.session
            pop.kanjiChosen.connect(lambda readings, sc: self.session is session and
                                    self._apply_kanji_readings(mid, *rng, readings, sc))
        pop.popup_at(pos)

    def _on_new_reading(self, mid: str, a: int, b: int, pos: QPoint) -> None:
        """New reading over visible characters [a, b) of a member (as rendered); the analyzer's reading is
        pre-filled (Enter accepts it)."""
        if self.session.doc.members[mid].lang.lower().split("-")[0] != "ja":
            self.statusBar().showMessage(JAPANESE_ONLY, 4000)
            return
        scope = self._reading_scope(mid, a, b)
        text = scope["text"][a:b]
        if not text.strip():
            return
        guess = "".join((s["reading"] or scope["text"][s["a"]:s["b"]]) for s in scope["spans"] if a <= s["a"] and s["b"] <= b)
        annotations = [self.session.doc.annotations[s["ann"]] for s in scope["spans"]
                       if s["a"] == a and s["b"] == b and s.get("ann")]
        candidates = [c for ann in annotations for c in ann.candidates]
        candidates += [c for entry in self._engine().analyze(scope["text"])
                       if (entry.start, entry.end) == (a, b) for c in entry.alternatives]
        candidates = self._reading_candidates(mid, (a, b), candidates)
        pop = ReadingPopup(text, "", candidates, "manual", "", self, scope=scope, prefill=guess,
                           candidates_for=lambda x, y: self._merged_reading_candidates(mid, scope["text"], x, y),
                           kanji_splitter=lambda text, reading: split_kanji_reading(text, reading, self._engine().analyzer))
        pop.typedChosen.connect(self._remember_typed_reading)
        pop.chosen.connect(lambda t, sc: self._apply_reading(mid, (a, b), t, sc))
        pop.chosenRange.connect(lambda x, y, t, sc: self._apply_reading(mid, (x, y), t, sc))
        split_session = self.session
        pop.wordSplitRequested.connect(lambda x, y, sc: QTimer.singleShot(
            0, lambda: self.session is split_session and self._on_word_split(mid, x, y, sc)))
        session = self.session
        pop.kanjiChosen.connect(lambda readings, sc: self.session is session and
                                self._apply_kanji_readings(mid, a, b, readings, sc))
        pop.popup_at(pos)

    def _on_word_split(self, mid: str, a: int, b: int, scope: int) -> None:
        from subkit.core.readings import visible_map
        from .views.word_split_dialog import WordSplitDialog

        session = self.session
        if session is None or mid not in session.doc.members:
            return
        member = session.doc.members[mid]
        text = visible_map(member)[0].text
        if member.lang.lower().split("-")[0] != "ja" or not 0 <= a < b <= len(text):
            return
        surface = text[a:b]
        engine = self._engine()
        rule = engine.dict.word_splits.get(surface)
        parts = [{"surface": word, "reading": reading} for word, reading in rule.groups] if rule else None
        dialog = WordSplitDialog(surface, parts=parts, analyzer=engine.analyzer,
                                 scope="project" if scope == 2 else "here", existing_rule=rule is not None, parent=self)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted or self.session is not session:
                return
            if mid not in session.doc.members or visible_map(session.doc.members[mid])[0].text[a:b] != surface:
                self.statusBar().showMessage("选中文字已改变，请重新选中后拆词", 6000)
                return
            if dialog.remove_rule:
                if self._edit(lambda: session.editor.remove_word_split_rule(surface, project=self.project)):
                    self.player.message("工程拆词规则已删除；人工读音保留（Ctrl+Z 撤销）", 2200)
                return
            parts, chosen_scope = dialog.values()
            if chosen_scope == "project":
                preview = session.editor.preview_word_split_rule(surface, parts, project=self.project)
                changes = [item for item in preview if item["changes"]]
                kept = [item for item in preview if item["kept"] and (item["member"], item["at"]) != (mid, [a, b])]
                lines = [f"「{surface}」→ " + " / ".join(p["surface"] for p in parts),
                         f"保存到整个工程，本集另有 {len(changes)} 处自动读音会更新。",
                         "当前选中的这一处按新拆法替换；其他人工、确认过或不标注的读音保留。"]
                if kept:
                    lines.append(f"本集保留 {len(kept)} 处人工／已确认的读音。")
                lines.extend(["其他集以后合并或刷新自动读音时使用；同表记的本集读音规则优先。",
                              "对应的工程合词规则会替换为拆词规则；词库和本集更新可以一起 Ctrl+Z 撤销。"])
                if self._help_message("readings", QMessageBox.question, "工程拆词规则", "\n".join(lines)) != QMessageBox.StandardButton.Yes:
                    return
            if self.session is not session:
                return
            if self._edit(lambda: session.editor.split_reading_words(mid, a, b, parts,
                          scope=chosen_scope, project=self.project if chosen_scope == "project" else None)):
                label = "已保存工程拆词规则" if chosen_scope == "project" else "已按指定词拆开"
                self.player.message(f"{label}：{' / '.join(p['surface'] for p in parts)}（Ctrl+Z 撤销）", 2400)
                if session.doc.group_of(mid).kind == "song" and kara.has_karaoke(session.doc, session.doc.members[mid]):
                    self.statusBar().showMessage("拆词已更新；这行歌词的拍请复查", 8000)
        finally:
            dialog.deleteLater()

    def _project_readings(self) -> None:
        if self.project is None or self._episode_job is not None:
            return
        from .views.project_readings import ProjectReadingsDialog

        self._flush_group_selection()
        session = self.session
        dialog = None
        try:
            dialog = ProjectReadingsDialog(self.project, editor=session.editor if session else None, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.saved:
                if self.session is session and session is not None:
                    session.engine = None
                message = (f"工程规则已保存：{len(dialog.changed_surfaces)} 个词，"
                           f"更新 {dialog.updated_annotations} 条自动读音")
                if session is not None:
                    message += "（Ctrl+Z 撤销）"
                self.statusBar().showMessage(message, 8000)
        except (OSError, ValueError) as exc:
            self._help_message("project-readings", QMessageBox.warning, "工程词语与分词规则", str(exc))
        finally:
            if dialog is not None:
                dialog.deleteLater()

    def _apply_kanji_readings(self, mid: str, a: int, b: int, readings: list[str], scope: int) -> None:
        if not self.session or mid not in self.session.doc.members:
            return
        if self._edit(lambda: self.session.editor.split_kanji_readings(mid, a, b, readings)):
            message = "已按单个汉字拆开（Ctrl+Z 撤销）"
            if scope:
                message += "；只改了这一处，不影响本集或整个工程的其他读音"
            if self.session.doc.group_of(mid).kind == "song" and kara.has_karaoke(self.session.doc, self.session.doc.members[mid]):
                message += "；歌词的拍请复查"
            self.statusBar().showMessage(message, 8000)

    def _reading_candidates(self, mid: str, rng, candidates: list[str]) -> list[str]:
        from subkit.core.readings import visible_map
        from subkit.reading.dictionary import ReadingDict

        if rng is None or self.session.doc.members[mid].lang.lower().split("-")[0] != "ja":
            return candidates
        text = visible_map(self.session.doc.members[mid])[0].text
        a, b = rng
        history = ReadingDict.load(self.project.dict_path("readings")).typed_readings(text[a:b])
        return list(dict.fromkeys(candidates + history + self._engine().dictionary_readings(text, a, b)))

    def _merged_reading_candidates(self, mid: str, text: str, a: int, b: int) -> list[str]:
        from subkit.textutil import kata_to_hira

        lexicon = [kata_to_hira(r) for r in self._engine().analyzer.lookup_readings(text[a:b])]
        return self._reading_candidates(mid, (a, b), lexicon)

    def _remember_typed_reading(self, surface: str, reading: str) -> None:
        from subkit.reading.dictionary import ReadingDict

        try:
            ReadingDict().add_typed(self.project.dict_path("readings"), surface, reading)
        except (OSError, ValueError) as exc:
            self.statusBar().showMessage(f"手打读音未能记入工程词库：{exc}", 8000)

    def _apply_reading(self, mid: str, rng, reading: str, scope: int, *, aid: Optional[str] = None) -> None:
        """A reading chosen in the popup: here (set / one annotation over the range), or as an episode / project
        rule after a preview of every place it changes."""
        from subkit.core.readings import visible_map

        s = self.session
        ed = s.editor
        m = s.doc.members[mid]
        src = next((s.episode.sources[r.src] for r in m.runs if r.src in s.episode.sources), None)
        if aid is not None and s.doc.annotations.get(aid) is not None:
            self._edit(lambda: ed.set_reading(aid, reading))
        elif rng is not None:
            self._edit(lambda: ed.add_annotation(mid, rng[0], rng[1], reading, engine=self._engine(), source=src))
        if scope == 0 or rng is None:
            return
        surface = visible_map(s.doc.members[mid])[0].text[rng[0]:rng[1]]
        self._apply_rule(surface, reading, project=(scope == 2))

    def _rule_preview(self, surface: str, reading: str) -> tuple[list[str], list[str], list[str]]:
        """Where a reading rule would change something in this episode: (members to refresh, lines that change,
        lines kept because you confirmed / typed the reading there)."""
        from subkit.core.readings import KEEP_ORIGIN, KEEP_STATUS, anchor_positions, visible_map

        doc = self.session.doc
        mids, changes, kept = [], [], []
        for g in doc.groups:
            for mid in g.members:
                m = doc.members[mid]
                if m.lang.lower().split("-")[0] != "ja" or m.role in ("sign", "info", "comment"):
                    continue
                vt, ids = visible_map(m)
                k = vt.text.find(surface)
                if k < 0:
                    continue
                spans = [(anchor_positions(a, ids), a) for a in doc.annotations.values() if a.member == mid and a.kind == "ruby"]
                fixed = [a for p, a in spans if p and p[0] < k + len(surface) and k < p[1]
                         and (a.status in KEEP_STATUS or a.origin in KEEP_ORIGIN)]
                where = f"第 {doc.group_index(g.id) + 1} 组：{vt.text[:30]}"
                if fixed and all(a.text == reading and a.base == surface for a in fixed):
                    continue                                  # already exactly this reading, by hand
                if fixed:
                    kept.append(where)
                else:
                    changes.append(where)
                    mids.append(mid)
        return mids, changes, kept

    def _apply_rule(self, surface: str, reading: str, *, project: bool) -> None:
        mids, changes, kept = self._rule_preview(surface, reading)
        scope = "整个工程（写进工程词库）" if project else "这一集（本集读音规则）"
        msg = [f"「{surface}」→ {reading}，范围：{scope}", f"本集会更新 {len(changes)} 处自动读音："]
        msg += ["　" + x for x in changes[:10]] + (["　……"] if len(changes) > 10 else [])
        if kept:
            msg.append(f"另有 {len(kept)} 处是你确认过或手动输入的读音，保持不变：")
            msg += ["　" + x for x in kept[:5]] + (["　……"] if len(kept) > 5 else [])
        if project:
            msg.append("其他集：以后合并或刷新自动读音时使用这条规则。")
        msg.append("\n应用吗？（本集的更新可以 Ctrl+Z 撤销" + ("；工程规则可在「注音 → 工程词语与分词规则…」修改或删除）" if project else "）"))
        r = QMessageBox.question(self, APP_NAME, "\n".join(msg))
        if r != QMessageBox.StandardButton.Yes:
            self.player.message("只改了这里，没有加规则")
            return
        s = self.session
        if project:
            from subkit.reading.dictionary import ReadingDict

            ReadingDict().add_word(self.project.dict_path("readings"), surface, reading)
        else:
            self._edit(lambda: s.editor.add_reading_rule(surface, reading))
        s.engine = s.episode.reading_engine()
        if mids:
            self._edit(lambda: s.editor.reannotate(mids, s.engine, s.episode.sources, label=f"读音规则：{surface}"))
        self.player.message(f"「{surface}」= {reading}：规则已加入，本集更新了 {len(changes)} 处")
        self.statusBar().showMessage(f"读音规则「{surface}」→ {reading}（{scope}），本集更新 {len(changes)} 处", 8000)

    def _split_reading(self, mid: str, a: int, b: int) -> None:
        """“按自动分词拆开”: the analyzer reads the range again (e.g. 迷宮都市 → 迷宮 + 都市).  A word locked by a
        rule would be merged again, so the rule can be removed first (asked)."""
        from subkit.core.readings import visible_map

        s = self.session
        m = s.doc.members[mid]
        surface = visible_map(m)[0].text[a:b]
        ep_rule = any(r.get("kind") == "reading" and r.get("surface", "").replace(" ", "") == surface for r in s.doc.rules)
        in_dict = surface in self._engine().dict.words and self._engine().dict.words[surface].origin == "project"
        if ep_rule or in_dict:
            where = "本集读音规则" if ep_rule else "工程词库"
            r = QMessageBox.question(self, APP_NAME, f"「{surface}」在{where}里是一个词，不删掉这条规则就拆不开。\n"
                                     f"删掉这条规则再拆开吗？" + ("（可以 Ctrl+Z 撤销）" if ep_rule else "（会改动工程词库文件）"))
            if r != QMessageBox.StandardButton.Yes:
                return
            if ep_rule:
                self._edit(lambda: s.editor.remove_reading_rule(surface))
            else:
                from subkit.reading.dictionary import ReadingDict

                ReadingDict.remove_word(self.project.dict_path("readings"), surface)
            s.engine = s.episode.reading_engine()
        src = next((s.episode.sources[r.src] for r in m.runs if r.src in s.episode.sources), None)
        n = 0

        def do() -> None:
            nonlocal n
            n = s.editor.regenerate_readings(mid, a, b, self._engine(), source=src, include_confirmed=True)
        self._edit(do)
        self.player.message(f"按自动分词拆开：{n} 个读音（Ctrl+Z 撤销）")

    def _f2(self) -> None:
        if not (self.session and self.editor.jp_member):
            return
        from subkit.core.readings import plain_range_to_visible

        c = self.editor.jp_edit.textCursor()
        if c.hasSelection():
            a, b = sorted((c.selectionStart(), c.selectionEnd()))
            r = plain_range_to_visible(self.editor.jp_member, a, b)
            if r is not None:
                self._on_new_reading(self.editor.jp_member.id, r[0], r[1], self.editor.jp_edit.mapToGlobal(QPoint(0, 0)))

    # ================================================================ 观剧模式 / the video
    def toggle_watch_mode(self) -> None:
        self.set_watch_mode(not self.watch_mode)

    def _watch_key(self, pinned: bool) -> None:
        """F11 = 全屏观看, F10 = 看片打轴 (waveform always shown).  The same key again returns to the edit layout;
        the other key switches between the two."""
        if self.watch_mode and self.wave_panel.pinned() == pinned:
            self.set_watch_mode(False)
        elif self.watch_mode:
            self.wave_panel.btn_pin.setChecked(pinned)
            if not self.isFullScreen():
                self.showFullScreen()
        else:
            self.set_watch_mode(True, pinned=pinned)

    def _sync_watch_actions(self) -> None:
        pinned = self.wave_panel.pinned()
        for act, want in ((self.act_watch, self.watch_mode and not pinned), (self.act_watch2, self.watch_mode and pinned)):
            act.blockSignals(True)
            act.setChecked(want)
            act.blockSignals(False)

    def _on_pin(self, pinned: bool) -> None:
        """The waveform's “钉住” button = 看片打轴 (always shown) vs. 全屏观看 (shown when a subtitle is picked)."""
        if self.watch_mode:
            if pinned:
                self._show_watch_wave(True)
            else:                       # 全屏观看: the waveform only comes with a subtitle you pick
                self.watch.set_selected(None)
                self._update_transform_action()
                self._show_watch_wave(False)
            self.player.message("看片打轴：波形常驻（F10）" if pinned else "全屏观看：点字幕时才出波形（F11）")
        self._sync_watch_actions()
        self._sync_seek_ui()

    def _sync_seek_ui(self) -> None:
        """The seek strip under the video in the edit layout and 看片打轴; the bottom-edge overlay in 全屏观看."""
        full = self.watch_mode and not self.wave_panel.pinned()
        self.seek_strip.setVisible(not full)
        self.watch.bar.enabled = full
        if not full:
            self.watch.bar.hide()

    def set_watch_mode(self, on: bool, *, pinned: Optional[bool] = None) -> None:
        """观剧模式: full screen video; click subtitles to edit.  Two variants: 全屏观看 (F11; the waveform pops up when
        a subtitle is picked) and 看片打轴 (F10; video + waveform always).  Tab / Esc return to the edit layout."""
        if pinned is not None:
            self.wave_panel.btn_pin.blockSignals(True)
            self.wave_panel.btn_pin.setChecked(pinned)
            self.wave_panel.btn_pin.blockSignals(False)
        if on == self.watch_mode:
            self._on_pin(self.wave_panel.pinned())
            return
        if on:
            self._remember_edit_layout()
        self.watch_mode = on
        self.watch.redraw()
        self.action_hub.set_scene("watch" if on else "main")
        if not on:
            self.watch.finish_transform(apply=True)     # user-initiated mode exit applies
        self._update_transform_action()
        self.player.autohide = on
        self._sync_watch_actions()
        self._sync_seek_ui()
        if on:
            self._saved_layout = {"max": self.isMaximized(), "geo": self.saveGeometry(), "top": self.top_split.sizes(),
                                  "main": self.main_split.sizes(), "right": self.right_split.sizes(),
                                  "video": self.video_split.sizes(),
                                  "dock": self.dock_source.isVisible(), "tb": self.toolbar.isVisible()}
            self._saved_layout["people"] = self.people.isVisible()
            self._saved_layout["state"] = self.saveState()
            self._saved_layout["host_state"] = self.right_host.saveState()
            for w in (self.right_host, self.table, self.dock_source, self.people, self.toolbar, self.menuBar(), self.statusBar()):
                w.hide()
            if self.wave_panel.parentWidget() is not self.video_split:
                self.video_split.addWidget(self.wave_panel)
            self.wave_panel.setVisible(False)
            self.showFullScreen()
            self.player.setFocus()
            if self.wave_panel.pinned():
                QTimer.singleShot(0, self, lambda: self._show_watch_wave(True))
            head = "看片打轴（波形常驻，F11 = 全屏观看）" if self.wave_panel.pinned() else "全屏观看（F10 = 看片打轴，波形常驻）"
            wave_hint = "左键=选中/跳转  右键=播放这一拍" if self.edit_layout == "karaoke" else "波形上左键开始、右键结束"
            QTimer.singleShot(300, self, lambda: self.player.message(
                head + "：点字幕 = 选中并暂停（" + wave_hint + "）　右键汉字 = 改读音　拖动 = 移位置\n"
                "右键空白处 / 空格 = 继续播放　双击字幕 = 改字　" + ("进度条在视频下方" if self.wave_panel.pinned() else "鼠标碰到画面最底边 = 进度条")
                + "　Del = 删除选中的组　Tab = 编辑模式", 6000))
        else:
            s = self._saved_layout
            if self.wave_panel.isVisible():
                self._watch_wave_h = self.video_split.sizes()[-1]
            if self.edit_layout != "video":
                self.right_split.insertWidget(0, self.wave_panel)
            self.wave_panel.show()
            for w in (self.right_host, self.right_split, self.table, self.menuBar(), self.statusBar()):
                w.show()
            self.toolbar.setVisible(s.get("tb", True))
            self.dock_source.setVisible(s.get("dock", True))
            self.people.setVisible(s.get("people", False))
            if s.get("max"):
                self.showMaximized()
            else:
                self.showNormal()
                if s.get("geo") is not None:
                    self.restoreGeometry(s["geo"])
            if s.get("state") is not None:
                self.restoreState(s["state"])
            if s.get("host_state") is not None:
                self.right_host.restoreState(s["host_state"])
            self._apply_edit_sizes(s)
            QTimer.singleShot(0, self, lambda: self._apply_edit_sizes(s))
            self.player.pause()
            if self.current_gid:
                self.select_group(self.current_gid, seek=False)

    def _on_video_pick(self, gid: str) -> None:
        self.select_group(gid, seek=False)
        if self.watch_mode:
            self._show_watch_wave(True)

    def _show_watch_wave(self, on: bool) -> None:
        """Pop the waveform up under the video (its height is remembered; drag the divider to change it)."""
        if on == self.wave_panel.isVisible():
            return
        if not on:
            self._watch_wave_h = self.video_split.sizes()[-1]
            self.wave_panel.setVisible(False)
            if self.watch_mode:
                self.layout().activate()
                if not self.isFullScreen():
                    self.showFullScreen()
            return
        total = sum(self.video_split.sizes()) or self.video_split.height()
        h = self._watch_wave_h or max(170, total // 5)
        self.wave_panel.setVisible(True)
        self.video_split.setSizes([max(100, total - h), h])
        if self.watch_mode:
            self.layout().activate()
            if not self.isFullScreen():
                self.showFullScreen()

    def _video_deselect(self) -> None:
        self.watch.set_selected(None)
        self._update_transform_action()
        if self.watch_mode and not self.wave_panel.pinned():
            self._show_watch_wave(False)

    def _video_resume(self) -> None:
        self._group_playback_gid = None
        self._video_deselect()
        self._stop_at = self._step_select_gid = None
        self.player.play()

    def _on_pause_changed(self, paused: bool) -> None:
        self.watch.redraw()
        if paused:
            self.watch.request_geometry()

    def _escape(self) -> None:
        if self._inline is not None and self._inline.isVisible():
            self._inline.cancel()
            return
        if self.watch.transform is not None:
            self.watch.finish_transform()
            return
        if self.tapper.active:
            self.tapper.cancel()
            return
        if self.watch.selected:
            self._video_deselect()
        else:
            self.set_watch_mode(False)

    # ================================================================ K 轴
    def _auto_options(self):
        saved = self.project.data.get("settings", {}).get("karaoke_auto", {}) if self.project else {}
        return {"translations": True, "last_end": "voice", "offset": "habit", "overwrite": False,
                "continuation_mode": "even", "translation_follow_notes": True, **saved}

    def _auto_option(self, key, value):
        if self.project is None:
            return
        if key == "offset" and value == "custom":
            current = self._auto_options()["offset"]
            dialog = QInputDialog(self)
            dialog.help_anchor = "karaoke-auto"
            dialog.setWindowTitle("拍头提前量")
            dialog.setLabelText("拍头提前量（ms）")
            dialog.setInputMode(QInputDialog.InputMode.IntInput)
            dialog.setIntRange(-150, 150)
            dialog.setIntValue(current if type(current) is int else 0)
            app_help.dialog_buttons(dialog)
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            value = dialog.intValue()
            dialog.deleteLater()
            if not accepted:
                self.wave_panel.set_auto_options(self._auto_options())
                return
        self.project.data.setdefault("settings", {}).setdefault("karaoke_auto", {})[key] = value
        self.project.save()
        self.wave_panel.set_auto_options(self._auto_options())

    def _refresh_auto_menu(self):
        from subkit.kalign import available
        from subkit.core.songs import song_span
        doc = self.session.doc if self.session else None
        gids = self.table.selected_gids() if doc else []
        groups = {g.id: g for g in doc.groups} if doc else {}
        songs = [gid for gid in gids if gid in groups and groups[gid].kind == "song"]
        current = groups.get(self.current_gid)
        span = song_span(doc, self.current_gid) if current else []
        line = bool(current and current.kind == "song" and self._k_member(current, "original"))
        self.wave_panel.set_auto_options(self._auto_options())
        self.wave_panel.set_auto_scopes(line, len(songs) >= 2, bool(span), available()[0])

    def _refresh_dialogue_menu(self):
        from subkit.kalign import available
        ready = available()[0]
        self.act_dialogue_align.setEnabled(bool(ready and self.session and self.session.episode.video
                                               and self._dialign_context is None and self._kalign_context is None))
        self.act_dialogue_align.setToolTip("" if ready else
            "需要安装 torch、torchaudio、transformers、demucs（见说明「自动 K 轴」）")

    def _load_dialogue_cache(self):
        from .dialogue_alignment import load_records
        ep = self.session.episode
        records = load_records(ep.video, (getattr(ep, "data", {}).get("media") or {}).get("audio_track"),
                               self.project.cache_dir)
        self.wave.set_dialogue_records(records)
        self.wave_panel.recalibrate_dialogue_padding()

    def _start_dialogue_align(self):
        from subkit.kalign import available
        from subkit.kalign.request import build_dialogue_request
        if not self.session or not available()[0] or self._dialign_context is not None or self._kalign_context is not None or self._listening_context is not None:
            return
        try:
            ep = self.session.episode
            if not ep.video:
                raise ValueError("这一集没有视频")
            request, meta = build_dialogue_request(self.session.doc, video=ep.video,
                audio_track=(getattr(ep, "data", {}).get("media") or {}).get("audio_track"),
                cache_dir=self.project.cache_dir)
            context = self._dialign_context = {"session": self.session, "meta": meta, "device": None}
            runner = self._dialign_runner = self.kalign_runner_factory(self)
            runner.progress.connect(lambda value: self._dialogue_progress(context, value))
            runner.finished.connect(lambda result: self._dialogue_finished(context, result))
            runner.failed.connect(lambda error: self._dialogue_failed(context, error))
            runner.cancelled.connect(lambda: self._finish_dialogue_align(context))
            dialog = self._dialign_dialog = app_help.ProgressDialog("精确对话轴：读取音频", "取消", 0, 0, self)
            dialog.help_anchor = "dialogue-align"
            dialog.setWindowTitle("精确对话轴")
            dialog.setMinimumDuration(0)
            dialog.setAutoClose(False)
            dialog.canceled.connect(self._cancel_dialogue_align)
            dialog.show()
            runner.start(request)
        except Exception as exc:
            self._dialogue_failed(self._dialign_context, str(exc))

    def _dialogue_progress(self, context, value):
        if self._dialign_context is not context:
            return
        if value.get("device"):
            context["device"] = value["device"]
        labels = {"extract": "读取音频", "download": "下载模型（约 2.5 GB，只需一次）",
                  "separate": "分离人声", "model1": "识别（1/2）", "model2": "识别（2/2）",
                  "align": f"对齐台词（{value.get('done', 0)}/{value.get('total', 0)}）", "done": "对齐台词"}
        label = "精确对话轴：" + labels.get(value.get("stage"), "读取音频")
        if context["device"] == "cpu":
            label += "\n没有可用的显卡，CPU 上需要 15–20 分钟"
        self._dialign_dialog.setLabelText(label)

    def _finish_dialogue_align(self, context):
        if context is None or self._dialign_context is not context:
            return False
        self._dialign_context = None
        runner, self._dialign_runner = self._dialign_runner, None
        if self._dialign_dialog is not None:
            self._dialign_dialog.close()
            self._dialign_dialog.deleteLater()
            self._dialign_dialog = None
        if runner is not None:
            runner.deleteLater()
        current = self.session is context["session"]
        if current:
            self._refresh_dialogue_menu()
        return current

    def _cancel_dialogue_align(self):
        context, runner = self._dialign_context, self._dialign_runner
        self._finish_dialogue_align(context)  # Invalidate callbacks before cancel processes Qt signals.
        if runner is not None:
            runner.cancel()

    def _dialogue_failed(self, context, error):
        if context is not None and not self._finish_dialogue_align(context):
            return
        text = "精确对话轴失败：" + error
        self.statusBar().showMessage(text)
        self._help_message("dialogue-align", QMessageBox.warning, "精确对话轴", text)

    def _dialogue_finished(self, context, result):
        from .dialogue_alignment import records_from_json
        if not self._finish_dialogue_align(context):
            return
        try:
            records = records_from_json(result)
        except (ValueError, TypeError) as exc:
            self._dialogue_failed(None, str(exc))
            return
        self.wave.set_dialogue_records(records)
        self.wave_panel.recalibrate_dialogue_padding()
        self._show_dialogue_results()

    def _show_dialogue_results(self):
        from .dialogue_alignment import valid_edges
        from .views.dialign_result import DialogueAlignResultDialog
        if not self.session:
            return
        aligned = fallback = disagree = stale = 0
        for group in self.session.doc.groups:
            if group.kind != "dialogue" or "hidden" in group.flags:
                continue
            edges, changed = valid_edges(self.session.doc, group, self.wave.dialogue_records)
            aligned += bool(edges)
            fallback += not bool(edges)
            disagree += any("disagree" in e.flags for e in edges)
            stale += changed
        self.wave_panel.sync_hint_padding()
        if self._dialign_results is None:
            self._dialign_results = DialogueAlignResultDialog(self)
            self._dialign_results.applyRequested.connect(self._apply_dialogue_episode)
        self._dialign_results.set_report(aligned, fallback, disagree, stale,
            self.wave.dialogue_padding, self.wave_panel.dialogue_padding_source)
        self._dialign_results.show()
        self._dialign_results.raise_()

    def _apply_dialogue_episode(self):
        """Review frozen proposals and apply checked edges as one undo step."""
        from .views.dialign_apply import DialogueAlignApplyDialog
        if not self.session:
            return
        session = self.session
        self.wave_panel.sync_hint_padding()
        dialog = DialogueAlignApplyDialog(session.doc, self.wave.dialogue_records,
                                          self.wave.keyframes, self.wave.dialogue_padding, self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted and self.session is session:
                session.editor.set_group_times(dialog.checked_changes(),
                    label="精确对话轴（整集）", source="adopted", avoid_new_overlap=True)
        finally:
            dialog.deleteLater()

    def _start_auto_karaoke(self, scope):
        from subkit.kalign import available
        from subkit.kalign.request import build_song_request
        from subkit.core.songs import song_span
        if not available()[0] or self._kalign_context is not None or self._dialign_context is not None or self._listening_context is not None or self.tapper.active:
            return
        options = self._auto_options()
        doc = self.session.doc
        gids = ([self.current_gid] if scope == "line" else song_span(doc, self.current_gid)
                if scope == "song" else [gid for gid in self.table.selected_gids() if doc.group(gid).kind == "song"])
        if scope == "selected" and len(gids) < 2:
            return
        options["requested_gids"] = gids
        options["free_requested"] = scope == "line"     # an explicit redo of this line is not anchored to its old beats
        try:
            ep = self.session.episode
            if not ep.video:
                raise ValueError("这一集没有视频")
            request, meta = build_song_request(doc, self.current_gid, video=ep.video,
                audio_track=(getattr(ep, "data", {}).get("media") or {}).get("audio_track"),
                cache_dir=self.project.cache_dir, options=options)
            context = {"session": self.session, "scope": scope, "options": options, "meta": meta}
            self._kalign_context = context
            runner = self._kalign_runner = self.kalign_runner_factory(self)
            runner.progress.connect(lambda value: self._auto_progress(context, value))
            runner.finished.connect(lambda result: self._auto_finished(context, result))
            runner.failed.connect(lambda error: self._auto_failed(context, error))
            runner.cancelled.connect(lambda: self._finish_auto(context))
            dialog = self._kalign_dialog = app_help.ProgressDialog("自动 K 轴：读取音频", "取消", 0, 0, self)
            dialog.help_anchor = "karaoke-auto"
            dialog.setWindowTitle("自动 K 轴")
            dialog.setMinimumDuration(0)
            dialog.setAutoClose(False)
            dialog.canceled.connect(self._cancel_auto_karaoke)
            dialog.show()
            runner.start(request)
        except Exception as exc:
            if self._kalign_context is not None:
                self._auto_failed(self._kalign_context, str(exc))
            else:
                self._show_auto_failure(str(exc))

    def _auto_progress(self, context, value):
        if self._kalign_context is not context:
            return
        labels = {"extract": "读取音频", "download": "下载模型（约 2.5 GB，只需一次）",
                  "separate": "分离人声", "model1": "识别（1/2）", "model2": "识别（2/2）", "align": "对齐", "done": "对齐"}
        self._kalign_dialog.setLabelText("自动 K 轴：" + labels[value["stage"]])

    def _finish_auto(self, context):
        if self._kalign_context is not context:
            return False
        self._kalign_context = None
        runner, self._kalign_runner = self._kalign_runner, None
        if self._kalign_dialog is not None:
            self._kalign_dialog.close()
            self._kalign_dialog.deleteLater()
            self._kalign_dialog = None
        if runner is not None:
            runner.deleteLater()
        return self.session is context["session"]

    def _cancel_auto_karaoke(self):
        context, runner = self._kalign_context, self._kalign_runner
        if context is not None:
            self._finish_auto(context)  # invalidate callbacks before kill/wait processes Qt signals
        if runner is not None:
            runner.cancel()
        self._auto_suggestion = None

    def _show_auto_failure(self, error):
        text = "自动 K 轴失败：" + error
        self.statusBar().showMessage(text)
        self._help_message("karaoke-auto", QMessageBox.warning, "自动 K 轴", text)

    def _auto_failed(self, context, error):
        if self._finish_auto(context):
            self._show_auto_failure(error)

    def _auto_finished(self, context, result):
        import copy
        from subkit.kalign.request import line_signature
        if not self._finish_auto(context):
            return
        doc, meta, options = self.session.doc, context["meta"], context["options"]
        requested = set(meta["requested_ids"])
        stale = {mid for mid in requested if line_signature(doc, mid) != meta["signatures"].get(mid)}
        rows = [row for row in result.lines if row.mid in requested and row.mid not in stale]
        try:
            if context["scope"] == "line":
                if not rows:
                    self.statusBar().showMessage(f"{len(stale)} 句在处理期间改过，没有写入，请重新运行")
                    return
                row = rows[0]
                view = kara.effective(doc, doc.members[row.mid])
                beats = copy.deepcopy(view.beats)
                if len(beats) != len(row.beats):
                    raise ValueError("自动 K 轴拍数与当前歌词不一致")
                for beat, (a, b, _conf, check) in zip(beats, row.beats):
                    beat.t0, beat.t1, beat.by, beat.link = a, b, "align", {}
                    beat.flags = [f for f in beat.flags if f not in ("placeholder", "reading_changed", "check")]
                    if check:
                        beat.flags.append("check")
                self.wave.set_suggestions(row.mid, beats, view.unit)
                self._auto_suggestion = (self.wave.k_suggestions, context)
                text = f"自动 K 轴：虚线仅供参考，Enter 采用，Esc 隐藏；{sum('check' in b.flags for b in beats)} 拍需要复查"
            else:
                report = self.session.editor.apply_auto_karaoke(rows, translations=options["translations"],
                    overwrite=options["overwrite"], requested_ids={row.mid for row in rows},
                    follow_notes=options["translation_follow_notes"])
                report["stale"] = sorted(stale)
                report["requested_ids"] = meta["requested_ids"]
                report["check_counts"] = {row.mid: sum(bool(b[3]) for b in row.beats) for row in result.lines}
                self._auto_apply_report(report)
                text = f"已更新 {report['updated']} 句，跳过 {len(report['skipped'])} 句（已确认或手调过），{report['lines_with_check']} 句有拍需要复查"
                if stale:
                    text += f"；{len(stale)} 句在处理期间改过，没有写入，请重新运行"
                self._refresh_k()
            saved = self.project.data.setdefault("settings", {}).setdefault("karaoke_auto", {})
            learned = False
            if result.offset_source == "habit" and result.offset_count >= 20:
                saved["stored_habit"] = result.offset_ms
                learned = True
            if result.continuation_source == "habit":
                saved["stored_continuation"] = result.continuation_ratio
                learned = True
            if learned:
                self.project.save()
            offset = self._auto_offset_text(result) + self._auto_continuation_text(result)
            if context["scope"] != "line" and self._kalign_results is not None:
                self._kalign_results.offset.setText(offset)
            self.statusBar().showMessage(text + "；" + offset)
        except Exception as exc:
            self._show_auto_failure(str(exc))

    def _auto_apply_report(self, report):
        """Show one reusable result window; jumping keeps it open."""
        from .views.kalign_result import KalignResultDialog
        from subkit.core.readings import visible_map

        self._last_auto_apply_report = report
        if self._kalign_results is None:
            self._kalign_results = KalignResultDialog(self)
            self._kalign_results.groupRequested.connect(self.select_group)
        rows = []
        for mid in report.get("requested_ids", []):
            member = self.session.doc.members.get(mid)
            status = ("处理期间改过" if mid in report.get("stale", []) else
                      report["skipped"].get(mid, "更新"))
            rows.append((member.group if member else "", visible_map(member)[0].text[:24] if member else mid,
                         status, report.get("check_counts", {}).get(mid, 0)))
        self._kalign_results.set_report(report, rows)
        self._kalign_results.show()
        self._kalign_results.raise_()

    @staticmethod
    def _auto_offset_text(result):
        if result.offset_source in ("habit", "stored"):
            return f"拍头提前量：按我调过的拍 {result.offset_ms:+d} ms（{result.offset_count} 拍）"
        return f"拍头提前量：{'不调整' if result.offset_source == 'none' else '自定义'} {result.offset_ms:+d} ms"

    @staticmethod
    def _auto_continuation_text(result):
        source = result.continuation_source
        if source == "none":
            return ""
        if source == "habit":
            return f"；长音：按我调过的拍 {round(result.continuation_ratio * 100)}%（{result.continuation_count} 个）"
        if source == "stored":
            return f"；长音：工程存过的 {round(result.continuation_ratio * 100)}%"
        if source == "default":
            return "；长音：手调不到 5 个，平分"
        return "；长音：平分"

    def _k_member(self, g, lane: str):
        from subkit.core.readings import visible_map

        doc = self.session.doc
        return next((m for m in doc.members_of(g) if m.lane == lane and m.role != "comment" and visible_map(m)[0].text.strip()), None)

    def _refresh_k(self) -> None:
        """Show the beats of the current song group on the waveform (dashed = not timed yet)."""
        from subkit.core import karaoke as kara

        self.wave.kviews = {}
        self.wave_panel.sync_hint_padding()
        view = None
        has_tags = False
        g = None
        if self.session and self.current_gid:
            doc = self.session.doc
            try:
                g = doc.group(self.current_gid)
            except KeyError:
                g = None
            if g is not None:
                kidx = kara.KIndex(doc)
                if g.kind == "song" or any(kidx.has(doc.members[m]) for m in g.members):
                    for lane in ("original", "translation"):
                        m = self._k_member(g, lane)
                        if m is not None:
                            v = kara.effective(doc, m, idx=kidx, unit=None if kara.tracks_of(doc, m.id) else self._k_unit_pref)
                            self.wave.kviews[lane] = (m.id, v)
                    first = self.wave.kviews.get("original") or self.wave.kviews.get("translation")
                    view = first[1] if first else None
                    has_tags = any(kara.has_k_tags(doc.members[m]) for m in g.members)
        self.wave.selected_beat()
        self.wave_panel.sync_rest_buttons()
        if view is not None:
            # The toolbar also reports issues on the other language's K row.
            for _mid, other in self.wave.kviews.values():
                view.issues.extend(issue for issue in other.issues if issue not in view.issues)
        self.wave_panel.show_k(view, has_tags=has_tags)
        self.wave_panel.set_english_available(g is not None and g.kind == "song" and any(
            kara.english_split_boundaries(self.session.doc, self.session.doc.members[mid], v)
            for mid, v in self.wave.kviews.values()))
        from subkit.core.lyrics_map import disabled_reason

        self.wave_panel.set_mapping_reason(disabled_reason(self.session.doc if self.session else None, g,
                                                          self.project.languages if self.project else {}))
        self._sync_k_menu()
        self._refresh_k_fx(view)
        self.wave.update()

    def _selected_song_gids(self):
        if not self.session:
            return []
        selected = set(self.table.selected_gids())
        return [g.id for g in self.session.doc.groups if g.id in selected and g.kind == "song"]

    def _refresh_redistribute_menu(self):
        from subkit.core.lyrics_map import disabled_reason
        from subkit.core.songs import song_span

        doc = self.session.doc if self.session else None
        group = next((g for g in doc.groups if g.id == self.current_gid), None) if doc else None
        both = disabled_reason(doc, group, self.project.languages if self.project else {})
        song = "" if group is not None and group.kind == "song" else "（不是歌曲）"
        for kind, reason in (("redistribute", both), ("align_times", both), ("fade", song)):
            self.wave_panel.set_scope_reasons(kind, {
                "line": reason,
                "selected": reason or ("" if len(self._selected_song_gids()) >= 2 else "（请在列表选中至少两句歌曲）"),
                "song": reason or ("" if song_span(doc, self.current_gid) else "（这一句不属于整首歌）")})

    def _k_scope_gids(self, kind, scope):
        """The song groups a lyric command applies to, or None when that menu entry is disabled."""
        from subkit.core.songs import song_span

        self._refresh_redistribute_menu()
        if not self.wave_panel.scope_actions[kind][scope].isEnabled():
            return None
        return ([self.current_gid] if scope == "line" else self._selected_song_gids()
                if scope == "selected" else song_span(self.session.doc, self.current_gid))

    def _k_align_times(self, scope):
        gids = self._k_scope_gids("align_times", scope)
        if gids is None:
            return
        changed = []
        if self._edit(lambda: changed.append(self.session.editor.align_song_translation_times(gids))):
            self.statusBar().showMessage(f"已对齐 {changed[0]} 行的时间（Ctrl+Z 撤销）" if changed and changed[0]
                                         else "这些歌词的翻译和原声时间已经一致")
        self._refresh_k()

    def _k_fade(self, scope):
        from .views.song_fade import SongFadeDialog

        gids = self._k_scope_gids("fade", scope)
        if gids is None:
            return
        current, overridden = self.session.editor.song_fade(gids)
        dialog = SongFadeDialog(len(gids), current, overridden, self)
        try:
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            fade = dialog.fade
        finally:
            dialog.deleteLater()
        changed = []
        if self._edit(lambda: changed.append(self.session.editor.set_song_fade(gids, fade))):
            what = "恢复原字幕的淡入淡出" if fade is None else f"淡入 {fade[0]} 毫秒、淡出 {fade[1]} 毫秒"
            self.statusBar().showMessage(f"{what}：{changed[0]} 行（Ctrl+Z 撤销）" if changed and changed[0]
                                         else "淡入淡出没有变化")

    def _k_redistribute(self, scope):
        gids = self._k_scope_gids("redistribute", scope)
        if gids is None:
            return
        report = {}
        if self._edit(lambda: report.update(self.session.editor.redistribute_translation(
                gids, follow_notes=self._auto_options()["translation_follow_notes"]))):
            self.statusBar().showMessage(f"已重新分配 {report.get('updated', 0)} 句的翻译拍（Ctrl+Z 撤销）")
        self._refresh_k()

    def _k_line_at(self, t_ms: float):
        """刷剧连打: the song line under the playhead → (member id, lane, beats, start, end)."""
        if not self.session:
            return None
        from subkit.core import karaoke as kara

        doc = self.session.doc
        for g in doc.groups:
            if g.kind != "song":
                continue
            m = self._k_member(g, "original") or self._k_member(g, "translation")
            if m is not None and m.start - 150 <= t_ms <= m.end:
                v = kara.effective(doc, m, unit=None if kara.tracks_of(doc, m.id) else self._k_unit_pref)
                return m.id, m.lane, v.beats, m.start, m.end
        return None

    def _k_action(self, name: str) -> None:
        self._flush_group_selection()
        if not (self.session and self.current_gid):
            return
        from subkit.core import karaoke as kara

        doc = self.session.doc
        ed = self.session.editor
        g = doc.group(self.current_gid)
        lane = "original" if "original" in self.wave.kviews else "translation"
        if name in ("add_rest", "remove_rest"):
            if self.tapper.active:
                return
            selected = self.wave.selected_beat() if name == "add_rest" else self.wave.selected_rest()
            if selected is None:
                return
            lane, i = selected
            mid, view = self.wave.kviews[lane]
            if name == "add_rest":
                self._edit(lambda: ed.add_rest(mid, view.beats[i].key))
            else:
                self._edit(lambda: ed.remove_rest(mid, view.beats[i].key, view.beats[i + 1].key))
            self._refresh_k()
            return
        if name == "english":
            self._edit(lambda: ed.split_english_syllables(g.id))
            self._refresh_k()
            return
        if name == "mapping":
            from .views.lyrics_map import LyricsMapDialog
            from subkit.core.lyrics_map import disabled_reason

            if disabled_reason(doc, g, self.project.languages):
                return
            dialog = LyricsMapDialog(doc, g.id, self.project.languages, self,
                                     follow_notes=self._auto_options()["translation_follow_notes"])
            dialog.playRequested.connect(self._play_range)
            dialog.show()
            dialog.activateWindow()
            if dialog.exec() == dialog.DialogCode.Accepted:
                self._edit(lambda: ed.apply_lyrics_mapping(dialog.mappings()))
                self._refresh_k()
            dialog.deleteLater()
            return
        if name.startswith("redistribute:"):
            self._k_redistribute(name.split(":", 1)[1])
            return
        if name.startswith("align_times:"):
            self._k_align_times(name.split(":", 1)[1])
            return
        if name.startswith("fade:"):
            self._k_fade(name.split(":", 1)[1])
            return
        if name.startswith("auto:"):
            self._start_auto_karaoke(name.split(":", 1)[1])
            return
        if name == "suggest":
            self._k_suggest()
            return
        if name == "adopt":
            proposal = self.wave.k_suggestions
            if proposal is None:
                return
            mid, beats, unit = proposal
            self.wave.clear_suggestions()
            auto = self._auto_suggestion
            self._auto_suggestion = None
            if auto is not None and auto[0] is proposal and auto[1]["session"] is self.session:
                from subkit.kalign.request import line_signature
                if line_signature(doc, mid) != auto[1]["meta"]["signatures"].get(mid):
                    self.statusBar().showMessage("1 句在处理期间改过，没有写入，请重新运行")
                    return
                self._edit(lambda: ed.adopt_auto_karaoke(mid, beats, unit, auto[1]["options"]["translations"],
                    follow_notes=auto[1]["options"]["translation_follow_notes"]))
            else:
                self._edit(lambda: ed.set_karaoke(mid, beats, state="suggested", unit=unit,
                                                 twins=True, label="采用起音建议"))
            self._refresh_k()
            return
        if name == "start":
            if lane not in self.wave.kviews:
                self.player.message(K_HINT)
                return
            selected = self.wave.selected_beat()
            from_index = 0
            if selected is not None:
                lane, from_index = selected
            mid, view = self.wave.kviews[lane]
            m = doc.members[mid]
            self._group_playback_gid = None
            self._stop_at = self._step_select_gid = None
            self.tapper.arm(mid, lane, view.beats, (m.start, m.end), from_index=from_index)
            if selected is not None and self.tapper.active:
                self.tapper.status.emit(f"从第 {from_index + 1} 拍开始重打（前面的拍不变），Esc 取消")
        elif name == "import":
            n = 0
            for mid in g.members:
                if kara.has_k_tags(doc.members[mid]):
                    self._edit(lambda mid=mid: ed.import_karaoke(mid, self._k_unit_pref))
                    n += 1
            self.player.message(f"读入了 {n} 行原字幕的 K 轴（Ctrl+Z 撤销）")
        elif name == "clear":
            mids = [m for m in g.members if kara.tracks_of(doc, m)]
            for mid in mids:
                self._edit(lambda mid=mid: ed.clear_karaoke(mid))
            self.player.message("已清除 K 轴（Ctrl+Z 撤销）")
        elif name == "confirm":
            mids = [m for m in g.members if kara.tracks_of(doc, m)]
            if not mids:
                self.player.message("还没有 K 轴")
                return
            mids += [t for m in mids for t in ed.karaoke_twins(m)]
            self._edit(lambda: ed.set_karaoke_state(mids, state="confirmed", label="确认 K 轴"))
            self.player.message("K 轴已确认")
        self._refresh_k()

    def _k_suggest(self) -> None:
        import copy
        import numpy as np
        from subkit.audio import offset_edges, suggest_beats

        self.wave.clear_suggestions()
        if self.tapper.active:
            self.statusBar().showMessage("请先完成或取消打拍，再查看起音建议", 6000)
            return
        if self.wave.onset is None:
            self.statusBar().showMessage("音频尚未加载，无法生成起音建议", 6000)
            return
        selected = self.wave.selected_beat()
        lane = selected[0] if selected else next(
            (lane for lane, (mid, _view) in self.wave.kviews.items() if mid == self.editor.position_member()),
            "original" if "original" in self.wave.kviews else "translation")
        if lane not in self.wave.kviews:
            return
        mid, view = self.wave.kviews[lane]
        m = self.session.doc.members[mid]
        times = suggest_beats(self.wave.onset, m.start, m.end, view.beats)
        if not times:
            self.statusBar().showMessage("这一行没有可建议的拍", 6000)
            return
        beats = copy.deepcopy(view.beats)
        features = self.wave.audio_features()
        offsets = offset_edges(features, np.asarray(times), m.start, m.end) if features is not None else []
        for i, b in enumerate(beats):
            b.t0, b.t1, b.by = times[i], times[i + 1] if i + 1 < len(times) else m.end, "onset"
            if i + 1 < len(times):
                rests = [int(t) for t in offsets if b.t0 + 60 <= t <= b.t1 - 120]
                if rests:
                    b.t1 = rests[0]
            b.flags = [f for f in b.flags if f not in ("placeholder", "reading_changed")]
            b.link = {}  # Proposed times are independent of original-row links.
        self.wave.set_suggestions(mid, beats, view.unit)
        self.statusBar().showMessage("起音建议：虚线仅供参考，Enter 采用，Esc 隐藏；采用后仍需确认", 6000)

    def _k_retap_selected(self) -> bool:
        self._flush_group_selection()
        if not self.session or self.wave.selected_beat() is None:
            return False
        self._k_action("start")
        return self.tapper.active

    def _k_split(self, mid: str, at: dict, ms: int) -> None:
        if self.session:
            self._edit(lambda: self.session.editor.split_beat(mid, at, ms))
            self._refresh_k()

    def _k_merge(self, mid: str, at: dict) -> None:
        if self.session:
            self._edit(lambda: self.session.editor.merge_beats(mid, at))
            self._refresh_k()

    def _k_tapped(self, mid: str, beats) -> None:
        if not self.session or mid not in self.session.doc.members:
            return
        from subkit.core import karaoke as kara

        unit = (kara.tracks_of(self.session.doc, mid) or [{}])[0].get("unit") or self._k_unit_pref
        self._edit(lambda: self.session.editor.set_karaoke(mid, beats, state="timed", unit=unit, twins=True, label="打 K 轴"))
        self._refresh_k()

    def _k_dragged(self, mid: str, beats) -> None:
        if not self.session:
            return
        from subkit.core import karaoke as kara

        track = (kara.tracks_of(self.session.doc, mid) or [{}])[0]
        unit = track.get("unit") or self._k_unit_pref
        state = "suggested" if track.get("state") == "suggested" else "timed"
        self._edit(lambda: self.session.editor.set_karaoke(mid, beats, state=state, unit=unit, twins=True, label="调整 K 轴"))
        self._refresh_k()

    def _k_active(self, lane: str, i: int) -> None:
        self.wave.k_active = (lane, i) if i >= 0 else None
        self.wave.update()

    def _k_context_effect_choices(self):
        combo = self.wave_panel.k_fx
        return [(combo.itemData(i), "不用特效" if combo.itemData(i) == "none" else combo.itemText(i))
                for i in range(combo.count())]

    def _k_context_effect(self, gids, preset):
        self._k_effect(preset, gids=gids)

    def _k_context_confirm(self, gids):
        """List right click 「确认 K 轴（n 句歌词）」: the K toolbar's 确认 for every selected song group, one undo step."""
        if not self.session:
            return
        doc, ed = self.session.doc, self.session.editor
        mids = [mid for g in doc.groups if g.id in set(gids) and g.kind == "song"
                for mid in g.members if kara.tracks_of(doc, mid)]
        if not mids:
            self.player.message("选中的歌词还没有 K 轴")
            return
        mids += [t for m in list(mids) for t in ed.karaoke_twins(m) if t not in mids]
        self._edit(lambda: ed.set_karaoke_state(mids, state="confirmed", label=f"确认 K 轴 ×{len(gids)}"))
        self.player.message(f"已确认 {len(gids)} 句歌词的 K 轴（Ctrl+Z 撤销）")
        self._refresh_k()

    def _k_effect(self, preset: str, *, gids=None) -> None:
        if not (self.session and self.current_gid):
            return
        selected = self._selected_song_gids() if gids is None else [g.id for g in self.session.doc.groups
                    if g.id in gids and g.kind == "song"]
        label = self.wave_panel.k_fx.itemText(self.wave_panel.k_fx.findData(preset))
        params = {}
        if preset.startswith("preset:"):
            entry = next(e for e in self._k_fx_store().list() if e["name"] == preset[7:])
            preset, params = entry["effect"], entry["params"]
        else:
            # Layout is independent of the chosen effect.
            first = self._k_fx_view()
            if first:
                params = {k: v for k, v in first[1].params.items() if k in ("ktv", "line_advance_ms")}
        scope = self._karaoke_fx_panel.scope.currentData() if self._karaoke_fx_panel else "line"
        if gids is not None or len(selected) >= 2:
            scope = "selection"
        self._k_apply_fx(preset, params, scope, gids=selected if scope == "selection" else None)
        if scope == "selection" and selected:
            self.statusBar().showMessage(f"已把特效「{label}」套到选中的 {len(selected)} 句（Ctrl+Z 撤销）")

    def _k_fx_store(self):
        from subkit.karaoke_fx.presets import PresetStore

        return PresetStore(self.project.root)

    def _k_fx_view(self):
        selected = self.wave.selected_beat()
        lane = selected[0] if selected else self.wave_panel.timing_target()
        return self.wave.kviews.get(lane) or self.wave.kviews.get("original") or self.wave.kviews.get("translation")

    def _refresh_k_fx(self, view):
        root = getattr(self.project, "root", None)
        if root != self._karaoke_fx_root:
            if self._karaoke_fx_panel is not None:
                self._karaoke_fx_panel.close()
                self._karaoke_fx_panel.deleteLater()
                self._karaoke_fx_panel = None
            self._karaoke_fx_root = root
            try:
                self.wave_panel.set_fx_presets(self._k_fx_store().list() if root else [])
            except (OSError, ValueError) as exc:
                self.statusBar().showMessage(str(exc))
        panel = self._karaoke_fx_panel
        if panel is not None:
            if view is None:
                panel.hide()
            else:
                first = self._k_fx_view()
                panel.set_effect(first[1].kind, first[1].params)

    def _k_parameters(self):
        if not self.session or not getattr(self.project, "root", None) or not self.wave.kviews:
            return
        from .views.karaoke_fx_panel import KaraokeFxPanel

        if self._karaoke_fx_panel is None:
            panel = self._karaoke_fx_panel = KaraokeFxPanel(self._k_fx_store(), self)
            panel.changed.connect(self._k_apply_fx)
            panel.presetsChanged.connect(self._k_presets_changed)
        first = self._k_fx_view()
        self._karaoke_fx_panel.set_effect(first[1].kind, first[1].params)
        self._karaoke_fx_panel.show()
        self._karaoke_fx_panel.activateWindow()

    def _k_presets_changed(self):
        self.wave_panel.set_fx_presets(self._k_fx_store().list())

    def _k_apply_fx(self, effect, params, scope, *, gids=None):
        if not self.session or not self.current_gid:
            return
        first = self._k_fx_view()
        gids = self.table.selected_gids() if gids is None else gids
        if scope == "selection":
            seed = next((m for g in self.session.doc.groups if g.id in gids and g.kind == "song"
                         for m in self.session.doc.members_of(g) if m.role == "song" and m.source_kind != "Comment"), None)
            first = (seed.id, None) if seed else None
        if first:
            self._edit(lambda: self.session.editor.apply_karaoke_effect(first[0], effect, params,
                                                                       scope=scope, gids=gids))
            # Model refresh selects the current row; keep a multi-selection for subsequent parameter changes.
            from PySide6.QtCore import QItemSelectionModel

            selection = self.table.selectionModel()
            if selection is not None:
                for gid in gids:
                    row = self.model.row_of(gid)
                    if row >= 0:
                        index = self.proxy.mapFromSource(self.model.index(row, 0))
                        if index.isValid():
                            selection.select(index, QItemSelectionModel.SelectionFlag.Select |
                                             QItemSelectionModel.SelectionFlag.Rows)

    def _k_unit(self, unit: str) -> None:
        self._k_unit_pref = unit
        if self.session and self.current_gid:
            from subkit.core import karaoke as kara

            doc = self.session.doc
            mids = [m for m in doc.group(self.current_gid).members if kara.tracks_of(doc, m)]
            if mids:
                self._edit(lambda: self.session.editor.set_karaoke_state(mids, unit=unit, label="K 轴分拍"))
        self._refresh_k()

    def _set_speed(self, v: float) -> None:
        kind = self._speed_kind()
        self._speeds[kind] = v
        self.settings.setValue("playback/" + kind + "_speed", v)
        self._apply_speed()
        self.player.message(f"播放速度 {v:g}×（音高不变）")

    def _speed_kind(self) -> str:
        if self.edit_layout == "karaoke":
            return "song"
        if self.session and self.current_gid:
            try:
                if self.session.doc.group(self.current_gid).kind == "song":
                    return "song"
            except KeyError:
                pass
        return "dialogue"

    def _apply_speed(self) -> None:
        kind = self._speed_kind()
        v = self._speeds[kind]
        self.wave_panel.set_speed(kind, v)
        self.lbl_speed.setText(f"{v:g}× 播放" if v != 1.0 else "")
        self.player.set_speed(v)

    def _style_of(self, name: str):
        """The style as the preview uses it (template styles and the sources' own styles)."""
        if self.watch.ass is not None and self.watch.ass.style(name) is not None:
            return self.watch.ass.style(name)
        if self.session is not None:
            return self.session.template.style(name)
        return None

    def _on_format(self, mid: str, a, b, change: dict) -> None:
        """Font / size / bold / italic / colour / ASS tags for visible [a, b) of a line, or the whole line."""
        if not self.session or mid not in self.session.doc.members:
            return
        what = "选中的字" if a is not None else "整行"
        if self._edit(lambda: self.session.editor.set_format(mid, change, a, b, label=f"格式（{what}）")):
            self.player.message(f"格式已改：{what}（Ctrl+Z 撤销）", 1200)

    def _on_transform(self, changes: dict) -> None:
        if self.session and self._edit(lambda: self.session.editor.set_overrides(changes, "自由变换")):
            self.player.message("自由变换已应用（Ctrl+Z 撤销）")

    def _on_group_format(self, mid: str, change: dict) -> None:
        """Explicit ASS tag scope: both languages of this group, with one undo."""
        if not self.session or mid not in self.session.doc.members or "raw" not in change:
            return
        from subkit.core.textformat import group_raw_changes

        changes = group_raw_changes(self.session.doc, mid, change["raw"])
        if changes and self._edit(lambda: self.session.editor.set_overrides(changes, "ASS 标签（整组）")):
            self.player.message("ASS 标签已应用到整组原声和翻译（Ctrl+Z 撤销）", 1800)

    # ------------------------------------------------------------ numeric positions
    def _position_targets(self, gids: list[str]) -> list[PositionTarget]:
        if not self.session or self.watch.ass is None or self._position_dirty:
            return []
        key = (id(self.session), self.session.doc.revision, self.watch.serial, tuple(gids))
        if self._position_cache and self._position_cache[0] == key:
            return self._position_cache[1]
        wanted = set(gids)
        index = self.watch._layout_index()
        try:
            targets = [PositionTarget.from_layout(index, i, self.session.doc.members[o["member"]],
                                                  self.session.episode.sources)
                       for i, o in enumerate(self.watch.owners)
                       if o.get("kind") == "base" and o.get("group") in wanted
                       and o.get("member") in self.session.doc.members
                       and i < len(self.watch.ass.events) and self.watch.ass.events[i].kind == "Dialogue"]
        finally:
            index.close()
        self._position_cache = (key, targets)
        return targets

    def _position_reference(self, targets: list[PositionTarget]) -> Optional[PositionTarget]:
        mid = self.editor.position_member()
        return next((t for t in targets if t.member == mid),
                    next((t for t in reversed(targets) if t.lane == "original"), next(iter(targets), None)))

    def _refresh_position(self) -> None:
        self.editor.position.show_anchor(None)
        if not self.session or not self.current_gid or self._position_dirty:
            return
        try:
            # Display needs one parsed anchor, not measured boxes for every member.
            candidates = [(i, self.session.doc.members[o["member"]])
                          for i, o in enumerate(self.watch.owners)
                          if o.get("kind") == "base" and o.get("group") == self.current_gid
                          and o.get("member") in self.session.doc.members
                          and i < len(self.watch.ass.events) and self.watch.ass.events[i].kind == "Dialogue"]
            mid = self.editor.position_member()
            ref = next((item for item in candidates if item[1].id == mid),
                       next((item for item in reversed(candidates) if item[1].lane == "original"),
                            next(iter(candidates), None)))
            if ref is not None:
                index = self.watch._layout_index()
                try:
                    self.editor.position.show_anchor(index.anchor(ref[0]))
                finally:
                    index.close()
        except Exception as exc:
            self.statusBar().showMessage(f"位置读取失败：{exc}", 8000)

    def _update_position_actions(self) -> None:
        selected = bool(self.session and self.table.selected_gids())
        for name, action in self.position_actions.items():
            reason = ("先打开一集" if not self.session else "先选中字幕组" if not selected else
                      "先等待字幕预览更新" if name not in ("top", "bottom", "alt") and
                      (self.watch.ass is None or self._position_dirty) else "")
            set_action_reason(action, reason)

    def _position_status(self, label: str, count: int, skipped: int = 0) -> None:
        text = f"{label}：{count} 组（Ctrl+Z 撤销）"
        if skipped:
            text += f"（{skipped} 组是歌曲、屏幕字或原字幕自带位置，没有改）"
        self.statusBar().showMessage(text, 6000)
        if self.watch_mode:
            self.player.message(text, 3000)

    def _apply_position(self, changes: dict, label: str) -> None:
        changes = {mid: change for mid, change in changes.items() if change}
        if not changes:
            return
        count = len({self.session.doc.members[mid].group for mid in changes})
        self.session.editor.set_overrides(changes, label)
        self._position_status(label, count)

    def _position_xy(self, x: Optional[float], y: Optional[float]) -> None:
        def apply():
            targets = self._position_targets([self.current_gid])
            ref = self._position_reference(targets)
            if ref:
                self._apply_position(set_xy(targets, ref, x, y, self.editor.position.scope()), "位置")
        self._edit(apply)

    def _position_alignment(self, an: int) -> None:
        def apply():
            targets = self._position_targets([self.current_gid])
            ref = self._position_reference(targets)
            if ref:
                self._apply_position(set_alignment(targets, an, self.editor.position.scope(), reference=ref), "对齐点")
        self._edit(apply)

    def _position_row_action(self, action: str) -> None:
        if self.current_gid:
            self._position_action(action, [self.current_gid], scope=self.editor.position.scope())

    def _position_action(self, action: str, gids: list[str], *, scope: str = "group") -> None:
        if not self.session or not gids:
            return
        def apply():
            if action in ("top", "bottom", "alt"):
                styles = {**MergeSettings().styles, **(self.session.episode.settings().get("merge", {}).get("styles") or {})}
                if action == "alt":
                    missing = [styles[k] for k in ("translation/alt", "original/alt") if self.session.template.style(styles[k]) is None]
                    if missing:
                        self.statusBar().showMessage(f"样式模板里没有「{'」「'.join(missing)}」：先在「样式模板」里加上第二人样式", 8000)
                        return
                result = self.session.editor.set_vertical(gids, action, styles=styles)
                self._position_status({"top": "移到顶部", "bottom": "移到底部", "alt": "底部第二人"}[action], len(result.gids),
                                      len({skip.group for skip in result.skipped}))
                return
            targets = self._position_targets(gids)
            if not targets:
                return
            if scope == "line":
                ref = self._position_reference(targets)
                targets = [ref] if ref else []
            if action == "restore":
                self._apply_position(restore(targets), "恢复原位置")
            else:
                self._apply_position(preset_side(targets, action, self.watch.ass.playres),
                                     "左侧说话人" if action == "left" else "右侧说话人")
        self._edit(apply)

    def _on_move(self, changes: dict) -> None:
        if not self.session:
            return
        self._edit(lambda: self.session.editor.set_overrides(changes, "移动位置"))
        self.player.message("位置已改（Ctrl+Z 撤销）")

    def _watch_inline_lane(self, lane: str) -> None:
        self._flush_group_selection()
        if not self.session:
            return
        doc = self.session.doc
        gid = self.watch.selected
        if not gid:
            shown = {self.watch.owners[i].get("group") for i in self.watch.active()
                     if self.watch.owners[i].get("kind") == "base"}
            gid = next(iter(shown)) if len(shown) == 1 else self.current_gid
        group = next((g for g in doc.groups if g.id == gid), None)
        if group is None:
            return
        mid = next((mid for mid in group.members
                    if doc.members[mid].lane == lane and doc.members[mid].role != "comment"), None)
        if mid is None:
            if group.kind != "dialogue" or not any(
                    m.lane != lane and m.role != "comment" for m in doc.members_of(group)):
                text = "这一组没有翻译" if lane == "translation" else "这一组没有原文"
                self.statusBar().showMessage(text, 4000)
                self.player.message(text)
                return
            if self._inline is not None and self._inline.isVisible():
                self._inline.close()              # commit the previous line before creating its partner
            inserted = self._apply_missing_language({group.id: ""}, lane, allow_empty=True)
            if not inserted:
                return
            mid = inserted[0]
        if self._inline is not None and self._inline.isVisible() and mid in self._inline.mids:
            self._inline.switch_member(mid)
            return
        self.player.pause()
        active = [self.watch.owners[i] for i in self.watch.active()]
        if (not any(owner.get("member") == mid for owner in active)
                and (visible_map(doc.members[mid])[0].text
                     or not any(owner.get("group") == group.id for owner in active))):
            self.player.seek(doc.group_time(group)[0] / 1000.0)
        self.watch.edit_member(mid)

    def _inline_edit(self, mid: str, rect, pixel_size: int) -> None:
        """Double click on the video: edit the group's lines in place (hotkeys in InlineTextEditor)."""
        if not self.session or mid not in self.session.doc.members:
            return
        from subkit.core.display import DisplayRules

        if self._inline is not None and self._inline.isVisible():
            self._inline.close()
        doc = self.session.doc
        g = doc.group_of(mid)
        mids = [x for x in g.members if doc.members[x].role != "comment"]
        ed = InlineTextEditor(doc, mids, mid, rect, pixel_size, DisplayRules.from_settings(self.project.settings()), self)
        ed.committed.connect(self._on_text)
        ed.confirmRequested.connect(lambda gid: self._edit(lambda: self.session.editor.confirm(gid)))
        ed.readingRequested.connect(self._on_reading)
        ed.newReadingRequested.connect(self._on_new_reading)
        ed.regenerateRequested.connect(self._regenerate)
        ed.fmt.quick_colours.set_model(self.quick_colours)
        ed.style_lookup = self._style_of
        ed.formatRequested.connect(self._on_format)
        ed.groupFormatRequested.connect(self._on_group_format)
        ed.clearFormatRequested.connect(lambda m_: self._edit(lambda: self.session.editor.clear_format(m_)))
        ed.destroyed.connect(lambda *_: setattr(self, "_inline", None) if self._inline is ed else None)
        ed.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._inline = ed
        ed.show_now()

    def _regenerate(self, mid: str, a: int, b: int, include_confirmed: bool) -> None:
        """Run the reading analysis again for visible [a, b) of a member (one undo step)."""
        if not self.session or mid not in self.session.doc.members:
            return
        s = self.session
        if s.engine is None:
            s.engine = s.episode.reading_engine()
        m = s.doc.members[mid]
        src = next((s.episode.sources[r.src] for r in m.runs if r.src in s.episode.sources), None)
        n = 0

        def do() -> None:
            nonlocal n
            n = s.editor.regenerate_readings(mid, a, b, s.engine, source=src, include_confirmed=include_confirmed)
        self._edit(do)
        self.player.message(f"重新生成了 {n} 个读音" + ("（包括确认过的；Ctrl+Z 撤销）" if include_confirmed else "（你确认过的保留；Ctrl+Z 撤销）"))
        self.statusBar().showMessage(f"重新生成了 {n} 个读音", 4000)

    def _regen_current(self, include_confirmed: bool) -> None:
        """Ctrl+G in edit mode: the selected Japanese text, or the whole Japanese line of the current group."""
        self._flush_group_selection()
        if self.watch_mode or not (self.session and self.editor.jp_member):
            return
        from subkit.core.readings import plain_range_to_visible, visible_map

        m = self.editor.jp_member
        c = self.editor.jp_edit.textCursor()
        r = None
        if c.hasSelection():
            a, b = sorted((c.selectionStart(), c.selectionEnd()))
            r = plain_range_to_visible(m, a, b)
        if r is None:
            r = (0, len(visible_map(m)[0].text))
        self._regenerate(m.id, r[0], r[1], include_confirmed)

    # ================================================================ refresh
    def _on_changed(self, touched: set) -> None:
        if self.session is None or self.sender() is not self.session:
            return
        self._update_save_state()
        if self.session.people_change:
            self.watch.invalidate_speakers()
            self.model.refresh_groups(touched if touched else {g.id for g in self.session.doc.groups})
            # Speaker metadata does not move subtitles, the playhead or keyboard focus.
            if self.session.preview_change:
                self._position_dirty = True
                self._position_cache = None
            from .views.group_editor import RubyLine

            if self.session.preview_change:
                for row in self.editor.findChildren(RubyLine):
                    if row.member is not None and row.member.group in touched:
                        row.set_member(self.session.doc, self.session.doc.members.get(row.member.id))
            if self.editor.jp_member is not None:
                self.editor.jp_member = self.session.doc.members.get(self.editor.jp_member.id)
            for mid, combo in self.editor.speaker_combos.items():
                member = self.session.doc.members.get(mid)
                if member is not None and not combo.hasFocus() and not combo.lineEdit().hasFocus():
                    combo.refresh_choices(self.session.editor.character_book, member, self._recent_speakers)
            self.watch.redraw()
            self._update_counts()
            return
        self._position_dirty = True
        self._position_cache = None
        self.editor.position.show_anchor(None)
        self._update_position_actions()
        self.model.reset()
        self._refresh_k()
        if self._inline is not None:
            try:
                self._inline.refresh()
            except RuntimeError:
                self._inline = None
        if self.current_gid:
            try:
                self.session.doc.group(self.current_gid)
            except KeyError:
                self.current_gid = None
        gid = self.current_gid or next((g.id for g in self.session.doc.groups if g.kind == "dialogue"), None)
        if gid:
            self.select_group(gid, seek=False, keep_view=gid == self.current_gid)
        self.wave.update()
        self._update_counts()

    def _on_preview(self, path: str, ass, rep) -> None:
        if self.session is None or self.sender() is not self.session:
            return
        try:
            self.player.load_subtitle(path)
        except Exception as exc:
            self.statusBar().showMessage(f"预览字幕加载失败：{exc}")
            return
        if self.session and ass is not None:
            lanes = {m.id: m.lane for m in self.session.doc.members.values()}
            self.watch.set_preview(ass, rep.owners, lanes, self.project.cache_dir / "render.sqlite3")
            revision = self.session._applied_revision
            self._position_dirty = revision >= 0 and not self.session.layout_current(revision)
            self._position_cache = None
            self._refresh_position()
            self._update_position_actions()
            if not getattr(self.session, "warmed", False):
                self.session.warmed = True           # measure every line once, so watching never waits for it
                self.watch.warm_up()

    def _on_warming(self, done: int, total: int) -> None:
        if total <= 0:
            self.lbl_task.setText("")
            self.statusBar().showMessage("字幕位置测量完成", 3000)
        else:
            self.lbl_task.setText(f"测量字幕位置 {done}/{total}…")

    def _on_session_status(self, text: str) -> None:
        if self.session is None or self.sender() is not self.session:
            return
        self.statusBar().showMessage(text, 5000)

    def _on_busy(self, name: str, running: bool) -> None:
        if self.session is None or self.sender() is not self.session:
            return
        self.lbl_task.setText(f"{name}…" if running else "")
        self.progress.setVisible(running)

    def _update_counts(self) -> None:
        if not self.session:
            self.lbl_counts.setText("")
            return
        doc = self.session.doc
        dlg = [g for g in doc.groups if g.kind == "dialogue"]
        conf = sum(1 for g in dlg if g.state == "confirmed")
        unc = sum(1 for a in doc.annotations.values() if a.status == "uncertain")
        self.lbl_counts.setText(f"已确认 {conf}/{len(dlg)}　读音待定 {unc}　")

    def _toggle_hide_period(self, on: bool) -> None:
        if self.project is None:
            return
        st = self.project.data.setdefault("settings", {})
        st.setdefault("display", {})["hide_final_period"] = bool(on)
        self.project.save()
        self._apply_rules()
        if self.session:
            self.session.request_preview()

    def _apply_language_labels(self) -> None:
        languages = self.project.languages
        japanese = has_japanese(languages)
        for action in (self.act_brackets, self.act_hide_period):
            set_action_reason(action, "" if japanese else JAPANESE_ONLY)
            action.setToolTip("" if japanese else JAPANESE_ONLY)
            action.setStatusTip(action.text() if japanese else JAPANESE_ONLY)
        self.wave_panel.k_unit.setEnabled(japanese)
        self.wave_panel.k_unit.setToolTip("" if japanese else JAPANESE_ONLY)
        self.editor.set_languages(languages)

    def _set_traditional(self, key: str, value) -> None:
        if self.project is None:
            return
        from subkit.core.display import DisplayRules

        display = self.project.data.setdefault("settings", {}).setdefault("display", {})
        display["zh_script"] = value if key == "script" else DisplayRules.from_settings(self.project.settings()).direction
        rule = display.setdefault("zh_traditional", {})
        rule.pop("enabled", None)
        if key != "script":
            rule[key] = value
        self.project.save()
        self._apply_rules()
        if self.session:
            self.session.request_preview()

    def _set_zh_scope(self, key: str, on: bool) -> None:
        if self.project is None:
            return
        display = self.project.data.setdefault("settings", {}).setdefault("display", {})
        display.setdefault("zh_scope", {})[key] = bool(on)
        self.project.save()
        self._apply_rules()
        if self.session:
            self.session.request_preview()

    def _set_merge_lines(self, lane: str, kind: str, on: bool) -> None:
        if self.project is None:
            return
        display = self.project.data.setdefault("settings", {}).setdefault("display", {})
        display.setdefault("merge_lines", {}).setdefault(lane, {})[kind] = bool(on)
        self.project.save()
        self._apply_rules()
        if self.session:
            self.session.request_preview()

    def _refresh_punctuation_profiles(self) -> None:
        menu = self.punctuation_profiles_menu
        menu.clear()
        if self._punctuation_profile_group is not None:
            self._punctuation_profile_group.deleteLater()
        self._punctuation_profile_group = QActionGroup(menu)
        self.punctuation_profile_actions = {}
        try:
            profiles = self.punctuation_profiles.list()
        except (ValueError, OSError) as exc:
            profiles = []
            self.statusBar().showMessage(f"读取标点方案失败：{exc}", 10000)
        settings = self.project.settings() if self.project else {}
        remembered = settings.get("display", {}).get("punctuation_profile")
        default = self.settings.value("display/default_punctuation_profile", "")
        equal = [profile["name"] for profile in profiles if self.project and punctuation_profile_matches(settings, profile)]
        # Identical profiles still form a radio group; prefer the remembered name.
        current = remembered if remembered in equal else next(iter(equal), None)
        for profile in profiles:
            name = profile["name"]
            label = name
            if remembered == name and name not in equal:
                label += "（已修改）"
            if default == name:
                label += "（默认）"
            action = QAction(label, menu, checkable=True)
            self._punctuation_profile_group.addAction(action)
            action.setChecked(name == current)
            action.setEnabled(self.project is not None)
            action.setProperty("help_anchor", "display-rules")
            action.setStatusTip("套用这个标点方案，只影响预览和导出；F1 查看显示规则。")
            action.triggered.connect(lambda _=False, name=name: self._select_punctuation_profile(name))
            menu.addAction(action)
            self.punctuation_profile_actions[name] = action
        if not profiles:
            action = menu.addAction("（还没有方案）")
            action.setEnabled(False)
            action.setProperty("help_anchor", "display-rules")
            action.setStatusTip("还没有保存的标点方案；F1 查看显示规则。")
        menu.addSeparator()
        action = self.act_punctuation
        menu.addAction(action)
        action.setEnabled(self.project is not None)
        action.setProperty("help_anchor", "display-rules")
        action.setStatusTip("打开标点和空格设置；F1 查看显示规则。")

    def _open_replacements(self) -> None:
        if self.project is None or self._episode_job is not None:
            return
        from subkit.ass.text import tokenize
        from .views.replacements import ReplacementsDialog

        samples = {}
        session = self.session
        if session is not None:
            group = next((g for g in session.doc.groups if g.id == self.current_gid), None)
            if group is not None:
                for mid in group.members:
                    member = session.doc.members.get(mid)
                    if member is None:
                        continue
                    plain = "".join(t.raw if t.kind == "text" else "\n" if t.kind == "break"
                                    else " " if t.kind in ("softbreak", "hardspace") else ""
                                    for t in tokenize(member.text))
                    if member.lane in samples:
                        samples[member.lane] += "\n" + plain
                    else:
                        samples[member.lane] = plain
        dialog = None
        try:
            dialog = ReplacementsDialog(self.project, self, samples=samples)
            if dialog.exec() == QDialog.DialogCode.Accepted and dialog.saved:
                self._apply_rules()
                if self.session is session and session is not None:
                    session.request_preview()
                self.statusBar().showMessage("批量替换已保存，整个工程显示和导出时生效；编辑原文不变。", 8000)
        except (OSError, ValueError, RuntimeError) as exc:
            self._help_message("display-replacements", QMessageBox.warning, "批量替换", str(exc))
        finally:
            if dialog is not None:
                dialog.deleteLater()

    def _open_punctuation_menu(self) -> None:
        if self.project is not None:
            dialog = PunctuationDialog(self.project, self.punctuation_profiles, self.settings, self,
                                       session=self.session, current_gid=self.current_gid, on_apply=self._apply_rules)
            try:
                dialog.exec()
            finally:
                dialog.deleteLater()

    def _select_punctuation_profile(self, name: str) -> None:
        if self.project is None:
            return
        previous = self.project.data.get("settings", {})
        try:
            profile = self.punctuation_profiles.load(name)
            settings = copy.deepcopy(previous)
            apply_punctuation_profile(settings, profile)
            self.project.data["settings"] = settings
            self.project.save()
        except (ValueError, OSError) as exc:
            self.project.data["settings"] = previous
            QMessageBox.warning(self, APP_NAME, f"套用标点方案失败：{exc}")
            self._refresh_punctuation_profiles()
            return
        self._apply_rules()
        if self.session:
            self.session.request_preview()

    def _set_punctuation(self, lane: str, key: str, value) -> None:
        if self.project is None:
            return
        display = self.project.data.setdefault("settings", {}).setdefault("display", {})
        from subkit.core.display import PunctuationRules

        settings = display.setdefault("punctuation", {}).setdefault(lane, {})
        if settings.get("comma") is True:
            legacy = PunctuationRules.from_settings(settings)
            settings.update(comma=legacy.comma, dunhao=legacy.dunhao)
        settings[key] = value
        self.project.save()
        self._apply_rules()
        if self.session:
            self.session.request_preview()

    def _apply_rules(self) -> None:
        from subkit.core.display import DisplayRules, PunctuationRules

        if self.project is None:
            return
        rules = DisplayRules.from_settings(self.project.settings())
        self._refresh_punctuation_profiles()
        self.editor.set_rules(rules)
        if self.people.project is self.project:
            self.people.refresh()
        for lane, menu in self.merge_lane_menus.items():
            menu.setTitle(lane_label(lane, self.project.languages))
        for pair, action in self.merge_actions.items():
            action.blockSignals(True)
            action.setChecked(pair in rules.merge_lines)
            action.blockSignals(False)
        self.act_hide_period.blockSignals(True)
        self.act_hide_period.setChecked(rules.hide_final_period)
        self.act_hide_period.blockSignals(False)
        for action, checked in [*((a, k == rules.direction) for k, a in self.zh_script_actions.items()),
                                (self.act_traditional_phrases, rules.zh_traditional_phrases),
                                *((a, rules.zh_scope[k]) for k, a in self.zh_scope_actions.items()),
                                *((a, k == rules.zh_traditional_variant) for k, a in self.traditional_actions.items())]:
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)
        if self._inline is not None:
            self._inline.ruby.rules = rules
            self._inline.refresh()

    def closeEvent(self, e) -> None:  # noqa: N802
        if self._export_job is not None:
            self._export_job.cancel()
            if not self._export_job.finished.wait(2.0):
                self._export_close_pending = True
                e.ignore()
                return
            self._closing = True
            self._export_progress.done(0)
        if self._batch_dialog is not None and self._batch_dialog._jobs:
            self._batch_dialog.reject()
            e.ignore()
            return
        if self._episode_job is not None:
            self._asking_close = True
            try:
                answer = QMessageBox.question(self, APP_NAME, "正在添加这一集，先取消再关闭吗？")
            finally:
                self._asking_close = False
            pending, self._pending_episode_result = self._pending_episode_result, None
            if answer != QMessageBox.StandardButton.Yes:
                e.ignore()
                if pending is not None:
                    self._show_episode_result(*pending)
                return
        if self._episode_job is not None:       # still running after the question: cancel it
            self._closing = True
            self._episode_job.cancel()
            self._episode_job.finished.wait(2.0)
            if self._episode_job.mode == "merge" and self.session:
                self.session.dirty = False
            self._episode_progress.hide()
        if not self.close_session():
            self._closing = False
            e.ignore()
            return
        self._closing = True
        if self.watch_mode:
            self.set_watch_mode(False)
        self.watch.stop_warm()
        self.wave.cancel_jobs()
        self._remember_edit_layout()
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("state", self.saveState())
        self.player.shutdown()
        QApplication.instance().removeEventFilter(self)
        super().closeEvent(e)


def _labse_factory(project: Project, *, cancel_check=lambda: None):
    """Semantic similarity with the locally cached LaBSE model (no download; HF offline)."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np

    from subkit.align.dp2 import Aligner2
    from subkit.align.embed import Embedder

    emb = Embedder("labse", cache_path=project.cache_dir / "embeddings.sqlite3")

    def factory(rows, units):
        cache: dict[str, np.ndarray] = {}
        cancel_check()
        dry = Aligner2(rows, units, sim_fn=lambda a, b: 0.0, cancel_check=cancel_check)
        cn_t, jp_t = dry.collect_texts()
        texts = sorted(set([r.text for r in rows] + [u.text for u in units]) | cn_t | jp_t)
        for start in range(0, len(texts), 64):
            cancel_check()
            batch = texts[start:start + 64]
            for t, v in zip(batch, emb.embed(batch)):
                cache[t] = v

        def vec(t: str):
            cancel_check()
            v = cache.get(t)
            if v is None:
                v = emb.embed([t])[0]
                cache[t] = v
            return v

        return lambda cn, jp: float(np.dot(vec(cn), vec(jp)))

    factory.close = emb._db.close if emb._db is not None else lambda: None
    return factory
