"""Modal passage recording and project library dialogs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Qt
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QCheckBox,
                               QListWidgetItem, QTreeWidget, QTreeWidgetItem, QApplication, QFileDialog, QPlainTextEdit)

from subkit.core.passages import PassageStore, plan
from subkit.audio_match import Match
from subkit import proc
from subkit.core import oped_import
from subkit.core.merge import MergeSettings

from .group_table import fmt_time

CATEGORY_NAMES = {"OP": "片头曲", "ED": "片尾曲", "IN": "插曲", "NA": "旁白"}
CATEGORY_FIELDS = {"OP": "片头曲 OP", "ED": "片尾曲 ED", "IN": "插曲", "NA": "旁白·固定台词"}


def passage_name(passage: dict) -> str:
    text = f"{CATEGORY_NAMES[passage['category']]}《{passage['name']}》"
    return text + (f"（{passage['version']}）" if passage['version'] else "")


SKIP_REASONS = {
    "target no longer exists in its planned group": "这一行在打开对照表之后改过，请重新查找",
    "target text changed since planning": "这一行在打开对照表之后改过，请重新查找",
    "insertion group no longer exists": "这一行在打开对照表之后改过，请重新查找",
    "manual karaoke changed since planning": "这一行在打开对照表之后改过，请重新查找",
    "position changed since planning": "这一行在打开对照表之后改过，请重新查找",
    "sung sequence differs after applying readings": "读音和库里不同，K 轴对不上",
    "karaoke link target is absent from the target group": "对应的原声行不在这一组，翻译拍的吸附没带过来",
    "confirmed reading changed since planning": "你确认过的读音在打开对照表之后改过，没有覆盖",
}


def _import_base_name(song) -> str:
    source = Path(song.lines[0].source_file)
    return song.family if song.family and song.family != "OPED" else source.stem or "歌词"


def _import_names(songs) -> list[str]:
    """Default names; songs sharing one style family (OPED.txt) are named by their first lyric line."""
    base = [_import_base_name(song) for song in songs]
    counts = Counter(base)
    out = []
    for song, name in zip(songs, base):
        if counts[name] > 1:
            line = song.lines[0]
            name = (line.original or line.translation).strip()[:20] or name
        out.append(name)
    return out


def _guess_category(name: str, start_ms: int) -> str:
    """OP/ED from the style or file name; otherwise from where the song starts in the episode."""
    found = re.search(r"(?<![A-Z])(OP|ED)(?![A-Z])", name.upper())
    if found:
        return found.group(1)
    if start_ms < 5 * 60000:
        return "OP"
    return "ED" if start_ms >= 20 * 60000 else "IN"


def _tree_item(parent, text: str) -> QTreeWidgetItem:
    item = QTreeWidgetItem(parent, [text])
    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
    return item


from ..help import dialog_buttons


class PassagesDialog(QDialog):
    help_anchor = "passages"
    def __init__(self, session, results: list[tuple[dict, object]], player, parent=None) -> None:
        super().__init__(parent)
        _modal(self, f"重复段落 — {session.episode.id}")
        self.resize(1120, 640)
        self.session, self.player, self.results = session, player, results
        self.styles = session.episode.settings().get("merge", {}).get("styles") or MergeSettings().styles
        self.template_styles = [s.name for s in session.template.styles]
        self.entries = []
        self.current_plan = None
        self.action_items = {}
        self.states = {}
        self.manual_matches = {}
        layout = QVBoxLayout(self)
        content = QHBoxLayout()
        self.list = QListWidget()
        content.addWidget(self.list, 1)
        right = QVBoxLayout()
        self.drift = _label("")
        self.scale = QCheckBox("按比例缩放时间（默认不缩放）")
        right.addWidget(self.drift)
        right.addWidget(self.scale)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        right.addWidget(self.tree)
        content.addLayout(right, 2)
        layout.addLayout(content)
        buttons = QHBoxLayout()
        self.audition_button = QPushButton("试听")
        self.all_button, self.none_button = QPushButton("全选"), QPushButton("全不选")
        self.manual_button = QPushButton("从当前组开始套用")
        self.manual_button.clicked.connect(self.manual_apply)
        buttons.addWidget(self.manual_button)
        self.apply_button, close = QPushButton("应用所选"), QPushButton("关闭")
        for button in (self.audition_button, self.all_button, self.none_button, self.apply_button, close):
            buttons.addWidget(button)
        dialog_buttons(self, buttons, primary=self.apply_button, auxiliary=(self.manual_button, self.audition_button, self.all_button, self.none_button))
        layout.addLayout(buttons)
        self.list.currentRowChanged.connect(self._select)
        self.scale.toggled.connect(self._replan)
        self.audition_button.clicked.connect(self.audition)
        self.all_button.clicked.connect(lambda: self.check_all(True))
        self.none_button.clicked.connect(lambda: self.check_all(False))
        self.apply_button.clicked.connect(self.apply)
        close.clicked.connect(self.accept)
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)
        self.refresh()

    def refresh(self) -> None:
        selected = self.list.currentRow()
        self._remember()
        self.list.blockSignals(True)
        self.list.clear()
        self.entries.clear()
        for passage, result in self.results:
            matches = result.matches or [None] if result is not None else [None]
            for match in matches:
                self.entries.append((passage, match))
                text = f"{CATEGORY_NAMES[passage['category']]}《{passage['name']}》 "
                if result is None:
                    text += "没有音频，只能手动套用"
                elif match is None:
                    text += f"没找到（最高相关 {result.best_score:.2f}）"
                else:
                    p = self._plan(passage, match, False)
                    text += f"{fmt_time(match.start_ms)}（相关 {match.score:.2f}）"
                    if match.covered[0] > 0 or match.covered[1] < passage['duration']:
                        text += f" · 放了 {match.covered[0] / 1000:g}–{match.covered[1] / 1000:g} 秒"
                    if p.already_applied:
                        text += " · 已套用"
                item = QListWidgetItem(text)
                if match is None:
                    item.setForeground(QColor("gray"))
                self.list.addItem(item)
        self.list.blockSignals(False)
        row = min(max(selected, 0), len(self.entries) - 1)
        self.list.setCurrentRow(row)
        if row < 0:
            self._select(row)

    def _plan(self, passage, match, scale):
        return plan(self.session.doc, passage, match, scale=scale,
                    template_styles=self.template_styles, styles=self.styles)

    def _remember(self) -> None:
        self.states.update({aid: item.checkState(0) == Qt.CheckState.Checked
                            for aid, item in self.action_items.items()})

    def _select(self, row: int) -> None:
        self._remember()
        self.scale.blockSignals(True)
        self.scale.setChecked(False)
        self.scale.blockSignals(False)
        self._replan()

    def _replan(self) -> None:
        self._remember()
        self.tree.clear()
        self.action_items = {}
        row = self.list.currentRow()
        passage, match = self.entries[row] if row >= 0 else (None, None)
        audio_less = passage is not None and any(p["id"] == passage["id"] and r is None for p, r in self.results)
        self.manual_button.setVisible(audio_less)
        anchor = getattr(self.parent(), "current_gid", None)
        self.manual_button.setEnabled(audio_less and anchor is not None)
        self.manual_button.setToolTip("" if anchor else "（先在列表里选中这首歌的第一组）")
        if audio_less:
            match = self.manual_matches.get(passage["id"])
        enabled = match is not None
        for widget in (self.audition_button, self.all_button, self.none_button, self.apply_button):
            widget.setEnabled(enabled)
        self.current_plan = self._plan(passage, match, self.scale.isChecked()) if enabled else None
        drift = match.drift_ms if enabled else 0
        self.drift.setVisible(bool(drift))
        self.scale.setVisible(bool(drift))
        self.drift.setText(f"头尾对不上：结尾比库里{'早' if drift < 0 else '晚'} {abs(drift)} 毫秒（{passage['version']}可能不同）" if drift else "")
        if not enabled:
            return
        seen = set()
        for line in self.current_plan.lines:
            text = (f"（这一集没有）{line.text}" if line.target_member_id is None
                    else f"{fmt_time(line.start)} {line.text}")
            coverage = {"direct": "直接匹配到", "inferred": "按连续性推断 · 证据弱", "unknown": "未知"}[line.coverage]
            root = _tree_item(self.tree, f"{text} · {coverage}")
            for action in line.actions:
                key = (action.kind, action.target_group_id) if action.kind == "convert" else (action.kind, action.id)
                if key in seen:
                    continue
                seen.add(key)
                item = _tree_item(root, action.label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if self.states.get(action.id, action.checked) else Qt.CheckState.Unchecked)
                self.action_items[action.id] = item
                if action.conflict:
                    note = _tree_item(item, action.conflict)
                    note.setForeground(0, QColor("red"))
                    font = note.font(0)
                    font.setPointSizeF(max(8, font.pointSizeF() - 1))
                    note.setFont(0, font)
                if action.kind == "karaoke" and line.readings_differ:
                    _tree_item(item, "读音和库里不同，要同时补读音，K 轴才对得上")
            for note in line.notes:
                _tree_item(root, note)
        for info in self.current_plan.info:
            _tree_item(self.tree, info)
        self.tree.expandAll()

    def manual_apply(self) -> None:
        row = self.list.currentRow()
        gid = getattr(self.parent(), "current_gid", None)
        if row < 0 or gid is None:
            return
        passage = self.entries[row][0]
        members = [self.session.doc.members[mid] for mid in self.session.doc.group(gid).members
                   if self.session.doc.members[mid].role != "comment"]
        if not members:
            return
        start = min(m.start for m in members)
        self.manual_matches[passage["id"]] = Match(start, 0.0, (0, passage["duration"]), [], 0, 1.0)
        self.scale.setChecked(False)
        self._replan()

    def check_all(self, checked: bool) -> None:
        for item in self.action_items.values():
            item.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def audition(self) -> None:
        if self.current_plan:
            self.parent()._stop_at = None
            self.player.seek(max(0, self.current_plan.at_ms / 1000))
            self.player.play()

    def apply(self) -> None:
        p = self.current_plan
        if p is None:
            return
        chosen = [aid for aid, item in self.action_items.items() if item.checkState(0) == Qt.CheckState.Checked]
        try:
            summary = self.session.editor.apply_passage(p, chosen, styles=self.styles)
        except ValueError as exc:
            QMessageBox.warning(self, "重复段落", f"没有套用：{exc}")
            return
        self.parent().statusBar().showMessage(
            f"已套用 {CATEGORY_NAMES[p.category]}《{p.name}》：改时间 {summary.time_lines} 行、读音 {summary.readings} 处、"
            f"K 轴 {summary.karaoke_lines} 行、补上 {summary.inserted_lines} 行（Ctrl+Z 撤销）", 10000)
        if summary.skipped:
            texts = {a.id: line.text for line in p.lines for a in line.actions}
            lines = [f"{texts.get(s['id'], '')}：{SKIP_REASONS.get(s['reason'], '没有套用：' + s['reason'])}" for s in summary.skipped]
            QMessageBox.information(self, "重复段落", f"有 {len(lines)} 项没有套用：\n" + "\n".join(lines))
        self.refresh()

    def undo(self) -> None:
        self.session.editor.history.undo()
        self.refresh()


def _modal(dialog: QDialog, title: str) -> None:
    dialog.setWindowTitle(title)
    dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.Tool)
    dialog.setModal(True)
    dialog.setMinimumWidth(620)


def _label(text: str) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


def duplicate_choice(parent, passage: dict) -> str:
    box = QMessageBox(parent)
    box.setWindowTitle("记为重复段落")
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText(f"库里已经有《{passage['name']}》（{passage['version']}）。")
    replace = box.addButton("替换", QMessageBox.ButtonRole.AcceptRole)
    new = box.addButton("另存为新的", QMessageBox.ButtonRole.ActionRole)
    cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(cancel)
    box.setEscapeButton(cancel)
    box.help_anchor = "passages"
    dialog_buttons(box)
    box.exec()
    choice = "replace" if box.clickedButton() is replace else "new" if box.clickedButton() is new else "cancel"
    box.deleteLater()
    return choice


class RecordPassageDialog(QDialog):
    help_anchor = "passages"
    def __init__(self, passage: dict, parent=None, *, karaoke_states: list[str]) -> None:
        super().__init__(parent)
        _modal(self, "记为重复段落")
        self.passage = passage
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(passage["name"])
        self.category = QComboBox()
        for key, text in CATEGORY_FIELDS.items():
            self.category.addItem(text, key)
        self.category.setCurrentIndex(self.category.findData(passage["category"]))
        self.version = QLineEdit(passage["version"])
        self.version.setPlaceholderText("例如 TV 版、完整版")
        form.addRow("名称", self.name)
        form.addRow("类别", self.category)
        form.addRow("版本", self.version)
        layout.addLayout(form)
        summary = passage["summary"]
        start, duration = passage["start"], passage["duration"]
        self.range_label = _label(f"{fmt_time(start)} – {fmt_time(start + duration)}，共 {summary['groups']} 组 "
                                  f"{summary['lines']} 行，{duration / 1000:g} 秒")
        layout.addWidget(self.range_label)
        self.lines = QListWidget()
        members = [m for g in passage["groups"] for m in g["members"]]
        for m, state in zip(members, karaoke_states):
            self.lines.addItem(f"{fmt_time(start + m['start'])} {m['visible']}　读音 {len(m['annotations'])} 处 · K 轴 {state}")
        layout.addWidget(self.lines)
        if summary["karaoke"]["partial"]:
            layout.addWidget(_label(f"有 {summary['karaoke']['partial']} 行 K 轴还没打完，占位拍不会记下。"))
        if summary["unconfirmed_lines"]:
            layout.addWidget(_label(f"有 {summary['unconfirmed_lines']} 行在没确认的组里，它们的自动读音不记下"
                                    "（别的集会按同样的文字自动注出来）；手动改过、选过的读音照样记下。"
                                    "把这些组标为已确认后，组里的读音也会记下。"))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("记下")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.name.textChanged.connect(self._check_name)
        dialog_buttons(self, self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._check_name()

    def _check_name(self) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.name.text().strip()))

    def values(self) -> tuple[str, str, str]:
        return self.name.text().strip(), self.category.currentData(), self.version.text().strip()


class _RenameDialog(QDialog):
    help_anchor = "passages"
    def __init__(self, passage: dict, parent=None) -> None:
        super().__init__(parent)
        _modal(self, "改名…")
        form = QFormLayout(self)
        self.name, self.version = QLineEdit(passage["name"]), QLineEdit(passage["version"])
        self.version.setPlaceholderText("例如 TV 版、完整版")
        form.addRow("名称", self.name)
        form.addRow("版本", self.version)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok.setText("改名…")
        ok.setEnabled(bool(self.name.text().strip()))
        self.name.textChanged.connect(lambda text: ok.setEnabled(bool(text.strip())))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        dialog_buttons(self, buttons)
        form.addRow(buttons)


class PassageLibraryDialog(QDialog):
    help_anchor = "passages"
    def __init__(self, store: PassageStore, parent=None, *, video=None, languages=None, audio_track=0) -> None:
        super().__init__(parent)
        _modal(self, "重复段落库")
        self.store = store
        self.video, self.languages, self.audio_track = video, languages, audio_track
        self.resize(940, 420)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["名称", "类别", "版本", "来自", "行数", "时长", "记录时间"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        self.rename_button, self.delete_button = QPushButton("改名…"), QPushButton("删除")
        close = QPushButton("关闭")
        self.import_button = QPushButton("导入旧版 OPED 资料…")
        self.import_button.clicked.connect(self.import_old)
        row.addWidget(self.import_button)
        self.rename_button.clicked.connect(self.rename)
        self.delete_button.clicked.connect(self.remove)
        close.clicked.connect(self.accept)
        row.addWidget(self.rename_button)
        row.addWidget(self.delete_button)
        row.addStretch()
        row.addWidget(close)
        dialog_buttons(self, row, auxiliary=(self.import_button, self.rename_button, self.delete_button))
        layout.addLayout(row)
        self.refresh()

    def import_old(self) -> None:
        dialog = OPEDImportDialog(self.store, self, video=self.video, languages=self.languages, audio_track=self.audio_track)
        try:
            dialog.exec()
            self.refresh()
        finally:
            dialog.deleteLater()

    def refresh(self, selected_id: str | None = None) -> None:
        self.passages = self.store.list()
        self.table.setRowCount(len(self.passages))
        for row, p in enumerate(self.passages):
            values = [p["name"], CATEGORY_NAMES[p["category"]], p["version"], p["episode_id"],
                      str(p["summary"]["lines"]), fmt_time(p["duration"]), p["created_at"]]
            for column, text in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(text))
            if p["id"] == selected_id:
                self.table.selectRow(row)
        self.table.resizeColumnsToContents()
        self._selection_changed()

    def selected(self) -> dict | None:
        rows = self.table.selectionModel().selectedRows()
        return self.passages[rows[0].row()] if rows else None

    def _selection_changed(self) -> None:
        enabled = self.selected() is not None
        self.rename_button.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)

    def rename(self) -> None:
        p = self.selected()
        if p is None:
            return
        dialog = _RenameDialog(p, self)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            self.store.rename(p["id"], dialog.name.text().strip(), dialog.version.text().strip())
            self.refresh(p["id"])
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "重复段落库", str(exc))
        finally:
            dialog.deleteLater()

    def remove(self) -> None:
        p = self.selected()
        if p is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("重复段落库")
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(f"从库里删除《{p['name']}》？已经套用到各集的内容不受影响。")
        delete = box.addButton("删除", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.setEscapeButton(cancel)
        box.help_anchor = "passages"
        dialog_buttons(box)
        box.exec()
        confirmed = box.clickedButton() is delete
        box.deleteLater()
        if not confirmed:
            return
        try:
            self.store.remove(p["id"])
            self.refresh()
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "重复段落库", str(exc))


class _ImportSignals(QObject):
    progress = Signal(str)
    completed = Signal(object, str)


class _ImportJob(QRunnable):
    """One cancellable pipeline; all publication goes to dialog receiver methods."""
    def __init__(self, work):
        super().__init__()
        self.work = work
        self.signals = _ImportSignals()
        self.runner = proc.KillableRun()
        self.finished = threading.Event()

    def cancel(self):
        self.runner.cancel()

    def run(self):
        result, error = None, ""
        try:
            self.runner.check_cancelled()
            result = self.work(self.runner, self.signals.progress)
            self.runner.check_cancelled()
        except proc.CancelledError:
            result = None
        except Exception as exc:
            error = str(exc)
        finally:
            self.finished.set()
            self.signals.completed.emit(result, error)


class PasteLyricsDialog(QDialog):
    help_anchor = "lyrics-block"

    def __init__(self, parent=None):
        super().__init__(parent)
        _modal(self, "粘贴歌词块")
        self.resize(900, 620)
        layout = QVBoxLayout(self)
        self.text = QPlainTextEdit()
        layout.addWidget(self.text)
        layout.addWidget(_label("写法见帮助；可以先用“复制提示词”让语言模型帮你整理"))
        buttons = QHBoxLayout()
        self.copy_button = QPushButton("复制提示词")
        self.read_button = QPushButton("读入")
        cancel = QPushButton("取消")
        for button in (self.copy_button, self.read_button, cancel):
            buttons.addWidget(button)
        self.copy_button.clicked.connect(self.copy_prompt)
        self.read_button.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        dialog_buttons(self, buttons, primary=self.read_button, auxiliary=(self.copy_button,))
        layout.addLayout(buttons)

    def copy_prompt(self):
        QApplication.clipboard().setText(oped_import.LYRICS_PROMPT)

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()
        self.text.setFocus()


class OPEDImportDialog(QDialog):
    help_anchor = "oped-import"

    def __init__(self, store, parent=None, *, video=None, languages=None, audio_track=0):
        super().__init__(parent)
        _modal(self, "导入旧版 OPED 资料")
        self.resize(1150, 650)
        self.store, self.video = store, video
        self.languages, self.audio_track = languages, audio_track or 0
        self.sources, self.rows, self.errors = [], [], []
        self._job = None
        self._phase = ""
        self._closing = False
        self.imported_names = []
        layout = QVBoxLayout(self)
        files = QHBoxLayout()
        files.addWidget(QLabel("歌词文件"))
        self.file_button = QPushButton("选择文件…")
        self.paste_button = QPushButton("粘贴歌词块…")
        self.files_label = _label("")
        files.addWidget(self.file_button)
        files.addWidget(self.paste_button)
        files.addWidget(self.files_label, 1)
        layout.addLayout(files)
        audio = QHBoxLayout()
        audio.addWidget(QLabel("音频"))
        self.audio = QComboBox()
        self.audio.addItem("同目录同名的 .wav（OPED-DB）", "sibling")
        self.audio.addItem("这一集的视频（歌词文件就是这一集的字幕时）", "episode")
        self.audio.addItem("选一个视频或音频文件…", "file")
        self.media_button = QPushButton("选择视频或音频…")
        self.media_label = _label("")
        self.media = None
        audio.addWidget(self.audio)
        audio.addWidget(self.media_button)
        audio.addWidget(self.media_label, 1)
        layout.addLayout(audio)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(["导入", "名称", "类别", "版本", "行数", "时长", "读音", "K 轴", "对应", "音频"])
        layout.addWidget(self.table)
        self.errors_button = QPushButton("有 0 行没认出来，没有导入")
        self.errors_button.setCheckable(True)
        self.error_list = QListWidget()
        self.error_list.hide()
        layout.addWidget(self.errors_button)
        layout.addWidget(self.error_list)
        self.progress = _label("")
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        self.import_button = QPushButton("导入")
        self.cancel_button = QPushButton("取消")
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.cancel_button)
        dialog_buttons(self, buttons)
        layout.addLayout(buttons)
        self.file_button.clicked.connect(self.choose_files)
        self.paste_button.clicked.connect(self.paste)
        self.audio.currentIndexChanged.connect(self.audio_changed)
        self.media_button.clicked.connect(self.choose_media)
        self.errors_button.toggled.connect(self.error_list.setVisible)
        self.import_button.clicked.connect(self.start_import)
        self.cancel_button.clicked.connect(self.reject)
        self.table.itemChanged.connect(self.update_conflicts)
        self.audio_changed()
        self._busy(False)

    def showEvent(self, event):
        super().showEvent(event)
        self.activateWindow()

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "歌词文件", "", "歌词文件 (*.ass *.txt *.song)")
        if paths:
            self.load_files(paths)

    def load_files(self, paths):
        self.sources.extend(("file", str(path)) for path in paths)
        self.reload()

    def paste(self):
        dialog = PasteLyricsDialog(self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.load_text(dialog.text.toPlainText())
        finally:
            dialog.deleteLater()

    def load_text(self, text):
        self.sources.append(("text", text))
        self.reload()

    def audio_changed(self, *_):
        self.media_button.setEnabled(self.audio.currentData() == "file")
        if self.sources:
            self.reload()

    def choose_media(self):
        path, _ = QFileDialog.getOpenFileName(self, "选一个视频或音频文件…", "", "视频或音频 (*)")
        if path:
            self.media = Path(path)
            self.media_label.setText(path)
            self.reload()

    def reload(self):
        if self._job is not None:
            return
        mode = self.audio.currentData()
        media = self.video if mode == "episode" else self.media if mode == "file" else None
        if mode != "sibling" and media is None:
            self.progress.setText("先选择视频或音频文件。" if mode == "file" else "先打开有视频的一集。")
            self.rows = []
            self.populate()
            return
        sources, track = list(self.sources), self.audio_track if mode == "episode" else 0
        def work(runner, progress):
            rows, errors = [], []
            for kind, source in sources:
                runner.check_cancelled()
                progress.emit("正在读取歌词…")
                result = oped_import.parse(source) if kind == "file" else oped_import.parse_text(source)
                errors.extend(result.unrecognised)
                for song in result.songs:
                    runner.check_cancelled()
                    progress.emit(f"正在读取音频：{song.family}…")
                    template = oped_import.build_audio_template(song, media, runner=runner, audio_track=track)
                    rows.append((song, template))
            return rows, errors
        self._start("preview", work)

    def _start(self, phase, work):
        self._phase = phase
        self._job = _ImportJob(work)
        self._job.signals.progress.connect(self._progress)
        self._job.signals.completed.connect(self._completed)
        self._busy(True)
        QThreadPool.globalInstance().start(self._job)

    def _progress(self, text):
        if not self._closing:
            self.progress.setText(text)

    def _busy(self, busy):
        for widget in (self.file_button, self.paste_button, self.audio, self.table):
            widget.setEnabled(not busy)
        self.media_button.setEnabled(not busy and self.audio.currentData() == "file")
        self.import_button.setEnabled(not busy and bool(self.rows))
        self.table.setCursor(Qt.CursorShape.WaitCursor if busy else Qt.CursorShape.ArrowCursor)
        if busy:
            self.progress.setText("正在读取歌词…" if self._phase == "preview" else "正在生成重复段落…")

    def _completed(self, result, error):
        self._job = None
        self._busy(False)
        if self._closing:
            super().reject()
            return
        if error:
            self.progress.setText(f"导入未完成：{error}")
            return
        if result is None:
            self.progress.setText("已取消。")
            return
        if self._phase == "preview":
            self.rows, self.errors = result
            self.populate()
            self.progress.setText("没有音频，不能自动查找，只能手动套用" if any(t is None for _, t in self.rows) else "读取完成。")
            self._busy(False)
        else:
            names = []
            try:
                for passage, template, conflict in result:
                    if oped_import.store_passage(self.store, passage, template, conflict=conflict) is not None:
                        names.append(passage["name"])
            except (OSError, ValueError, KeyError) as exc:
                self.progress.setText(f"导入未完成：{exc}；已导入 {len(names)} 首。")
                return
            self.imported_names = names
            message = f"已导入 {len(names)} 首：{'、'.join(names)}。打开一集后用“工具 → 重复段落 → 查找并套用…”套用。"
            window = self.parent()
            while window is not None and not hasattr(window, "statusBar"):
                window = window.parentWidget()
            if window is not None:
                window.statusBar().showMessage(message, 10000)
            self.progress.setText(message)
            self.accept()

    def populate(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.rows))
        names = _import_names([song for song, _template in self.rows])
        for row, (song, template) in enumerate(self.rows):
            name = names[row]
            self.table.setItem(row, 1, QTableWidgetItem(name))
            category = song.category
            if category == "NA":
                category = _guess_category(_import_base_name(song), song.start_ms)
            combo = QComboBox()
            for key in ("OP", "ED", "IN"):
                combo.addItem(CATEGORY_FIELDS[key], key)
            combo.setCurrentIndex(combo.findData(category))
            self.table.setCellWidget(row, 2, combo)
            self.table.setItem(row, 3, QTableWidgetItem(""))
            beats = sum(len(l.beats) + len(l.translation_beats) for l in song.lines)
            mappings = sum(bool(l.mapping) for l in song.lines)
            values = [str(len(song.lines)), fmt_time(song.end_ms - song.start_ms),
                      f"{sum(len(l.readings) for l in song.lines)} 处", f"{beats} 拍" if beats else "没有",
                      f"{mappings} 行" if mappings else "没有", "有" if template is not None else "没有"]
            notes = "\n".join(note for line in song.lines for note in line.notes)
            for column, value in enumerate(values, 4):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
            for column in (1, 3, 4, 5, 6, 7, 8, 9):
                self.table.item(row, column).setToolTip(notes)
        self.table.blockSignals(False)
        self.update_conflicts()
        self.error_list.clear()
        for error in self.errors:
            if error.block_number is not None:
                text = f"歌词块 第 {error.block_number} 块：{error.reason}"
            else:
                text = f"{error.source_file} 第 {error.line_number} 行：{error.reason}"
            self.error_list.addItem(text)
        self.errors_button.setText(f"有 {len(self.errors)} 行没认出来，没有导入")
        self.errors_button.setEnabled(bool(self.errors))
        self.files_label.setText("、".join(Path(s).name if k == "file" else "粘贴歌词块" for k, s in self.sources))
        self.table.resizeColumnsToContents()

    def update_conflicts(self, *_):
        existing = {(p["name"], p["version"]) for p in self.store.list()}
        self.table.blockSignals(True)
        for row in range(len(self.rows)):
            name_item = self.table.item(row, 1)
            if name_item is None:
                continue
            key = (name_item.text().removesuffix("（库里已有）").strip(), self.table.item(row, 3).text().strip())
            duplicate = key in existing
            name_item.setText(key[0] + ("（库里已有）" if duplicate else ""))
            current = self.table.cellWidget(row, 0)
            if duplicate and not isinstance(current, QComboBox):
                combo = QComboBox()
                for label, value in (("替换", "replace"), ("另存为新的", "new"), ("跳过", "skip")):
                    combo.addItem(label, value)
                combo.setCurrentIndex(2)
                self.table.setCellWidget(row, 0, combo)
            elif not duplicate and not isinstance(current, QCheckBox):
                check = QCheckBox()
                check.setChecked(True)
                self.table.setCellWidget(row, 0, check)
        self.table.blockSignals(False)

    def start_import(self):
        if self._job is not None:
            return
        selected = []
        for row, (song, template) in enumerate(self.rows):
            choice = self.table.cellWidget(row, 0)
            if isinstance(choice, QCheckBox) and not choice.isChecked():
                continue
            conflict = choice.currentData() if isinstance(choice, QComboBox) else "skip"
            if isinstance(choice, QComboBox) and conflict == "skip":
                continue
            name = self.table.item(row, 1).text().removesuffix("（库里已有）").strip()
            if not name:
                self.progress.setText("请填写名称。")
                return
            selected.append((song, template, name, self.table.cellWidget(row, 2).currentData(),
                             self.table.item(row, 3).text().strip(), conflict))
        languages = self.languages
        def work(runner, progress):
            result = []
            for song, template, name, category, version, conflict in selected:
                runner.check_cancelled()
                progress.emit(f"正在生成重复段落：{name}…")
                passage = oped_import.build_passage(song, name, category, version, languages=languages)
                runner.check_cancelled()
                result.append((passage, template, conflict))
            return result
        self._start("import", work)

    def reject(self):
        if self._job is not None:
            self._closing = True
            self.progress.setText("正在取消…")
            self.cancel_button.setEnabled(False)
            self._job.cancel()
        else:
            super().reject()

    def closeEvent(self, event):
        if self._job is not None:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
