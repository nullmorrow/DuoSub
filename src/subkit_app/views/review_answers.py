"""Explicit review decisions; a tool window leaves the episode available for manual edits."""

from __future__ import annotations

from dataclasses import replace
import html
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from subkit.core import review_import as review
from subkit.core.display import DisplayRules
from subkit.core.readings import visible_map
from subkit.languages import language_name

from .. import help as app_help

KIND_NAMES = {"reading": "读音", "text": "文字", "pairing": "配对", "speaker": "说话人",
              "persona": "人设", "other": "其他"}
DECISION_NAMES = {"accepted": "采用", "rejected": "不采用", "handled": "已处理"}
NOTHING_READ = "没有读到审议意见：回答里需要每条一行 JSON（审议包开头有格式说明）。"
CROSS_BREAK = "这条意见跨了换行，请跳到这一组手动改"


class ImportAnswersDialog(QDialog):
    help_anchor = "review-answers"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导回审议意见")
        self.resize(560, 360)
        layout = QVBoxLayout(self)
        self.file_button = QPushButton("选文件…")
        layout.addWidget(self.file_button)
        self.paste = QPlainTextEdit()
        self.paste.setPlaceholderText("在这里粘贴审议意见")
        layout.addWidget(self.paste)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                        QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.file_button.clicked.connect(self.choose_file)
        app_help.dialog_buttons(self, self.buttons)
        self.items, self.unreadable = [], []

    def load_file(self, path):
        self.paste.setPlainText(Path(path).read_text(encoding="utf-8-sig"))

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选文件", "", "审议意见 (*.jsonl *.json *.txt *.md);;所有文件 (*)")
        if path:
            try:
                self.load_file(path)
            except (OSError, UnicodeError) as exc:
                QMessageBox.warning(self, "导回审议意见", f"读取文件失败：{exc}")

    def accept(self):
        self.items, self.unreadable = review.parse_answers(self.paste.toPlainText())
        if not self.items:
            QMessageBox.information(self, "导回审议意见", NOTHING_READ)
            return
        super().accept()


class EditAnswerDialog(QDialog):
    """A regular input widget and explicit acceptance button preserve IME composition."""
    help_anchor = "review-answers"

    def __init__(self, text, parent=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("改一下再采用")
        self.resize(450, 180)
        layout = QVBoxLayout(self)
        self.edit = QPlainTextEdit(text)
        layout.addWidget(self.edit)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        use = self.buttons.addButton("采用", QDialogButtonBox.ButtonRole.AcceptRole)
        use.setAutoDefault(False)
        use.clicked.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        app_help.dialog_buttons(self, self.buttons)
        layout.addWidget(self.buttons)
        self.edit.setFocus()


class ReviewAnswersDialog(QDialog):
    help_anchor = "review-answers"

    def __init__(self, session, items, unreadable, parent):
        super().__init__(parent, Qt.WindowType.Tool)
        self.session, self.main = session, parent
        self.items = list({item.fingerprint: item for item in items}.values())
        self.picks = {}
        self.setWindowTitle(f"审议意见 — {session.episode.id}")
        self.resize(780, 620)
        layout = QVBoxLayout(self)
        filters = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItem("全部", None)
        for key, name in KIND_NAMES.items():
            self.kind.addItem(name, key)
        self.pending = QCheckBox("只看还没处理的")
        self.pending.setChecked(True)
        self.counts = QLabel()
        for widget in (self.kind, self.pending, self.counts):
            filters.addWidget(widget)
        layout.addLayout(filters)
        self.list = QListWidget()
        layout.addWidget(self.list)
        self.unreadable_button = QPushButton(f"没读懂的 {len(unreadable)} 行")
        self.unreadable_button.setCheckable(True)
        self.unreadable_box = QPlainTextEdit()
        self.unreadable_box.setReadOnly(True)
        self.unreadable_box.setPlainText("\n".join(unreadable))
        self.unreadable_box.setMaximumHeight(130)
        self.unreadable_box.hide()
        layout.addWidget(self.unreadable_button)
        layout.addWidget(self.unreadable_box)
        self.unreadable_button.toggled.connect(self.unreadable_box.setVisible)
        buttons = QHBoxLayout()
        self.all_button = QPushButton("采用所有读音意见")
        self.all_button.clicked.connect(self.accept_all)
        buttons.addWidget(self.all_button)
        close = QPushButton("关闭")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        app_help.dialog_buttons(self, buttons)
        layout.addLayout(buttons)
        self.kind.currentIndexChanged.connect(self.refresh)
        self.pending.toggled.connect(self.refresh)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self.refresh)
        self.session.changed.connect(self.schedule_refresh)
        self._connected = True
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)
        self.refresh()

    def schedule_refresh(self, *_):
        self.refresh_timer.start(0)

    def done(self, result):
        self.refresh_timer.stop()
        if self._connected:
            self.session.changed.disconnect(self.schedule_refresh)
            self._connected = False
        super().done(result)

    def undo(self):
        # Let a focused text widget handle its own undo.
        if isinstance(self.focusWidget(), QPlainTextEdit):
            self.focusWidget().undo()
        else:
            self.session.editor.history.undo()
            self.refresh()

    def suggestion(self, item, decisions):
        rules = DisplayRules.from_settings(self.session.project.settings())
        # Match current state independently of the persistent decision, including after undo.
        s = review.match(self.session.doc, self.session.episode.sources, rules, item)
        picked = self.picks.get(item.fingerprint)
        if s.status == "ambiguous" and picked is not None:
            loc = next((c for c in s.candidates if c == picked and c.exact), None)
            if loc is not None:
                current = review._current(self.session.doc, loc, item, rules)
                status = ("already_done" if review._equals(self.session.doc, loc, item, current, item.new)
                          else "ok" if item.kind not in ("reading", "text") or
                          review._equals(self.session.doc, loc, item, current, item.old) else "mismatch")
                s = replace(s, location=loc, current=current, status=status)
        return replace(s, decision=decisions.get(item.fingerprint, {}).get("decision"))

    def cross_break(self, s):
        if s.item.kind != "text" or s.location is None:
            return False
        loc = s.location
        vt, _ = visible_map(self.session.doc.members[loc.member])
        a, b = vt.chars[loc.a].raw_start, vt.chars[loc.b - 1].raw_end
        return any(t.kind in ("break", "drawing") and t.start >= a and t.end <= b for t in vt.tokens)

    def refresh(self, *_):
        if self.session._closed:
            self.close()
            return
        try:
            decisions = review.read_answers(self.session.episode.dir)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "审议意见", f"读取审议记录失败：{exc}")
            self.close()
            return
        scroll = self.list.verticalScrollBar().value()
        self.suggestions = [self.suggestion(item, decisions) for item in self.items]
        processed = sum(bool(s.decision or s.status == "already_done") for s in self.suggestions)
        self.counts.setText(f"{len(self.items)} 条，已处理 {processed} 条")
        self.list.clear()
        self.rows = []
        for s in self.suggestions:
            if self.kind.currentData() not in (None, s.item.kind):
                continue
            if self.pending.isChecked() and (s.decision or s.status == "already_done"):
                continue
            widget = self.make_row(s)
            row = QListWidgetItem(self.list)
            row.setSizeHint(widget.sizeHint())
            self.list.setItemWidget(row, widget)
            self.rows.append(widget)
        self.list.verticalScrollBar().setValue(scroll)
        self.all_button.setEnabled(bool(self.eligible_readings()))

    def sentence(self, loc):
        text, _ = review._shown(self.session.doc.members[loc.member],
                                 DisplayRules.from_settings(self.session.project.settings()))
        a, b = loc.displayed_a, loc.displayed_b
        return (html.escape(text[:a]) + '<span style="background-color:#ffe08a;color:#202020">' +
                html.escape(text[a:b]) + "</span>" + html.escape(text[b:]))

    def make_row(self, s):
        widget = QWidget()
        widget.suggestion = s
        layout = QVBoxLayout(widget)
        loc = s.location
        gid = loc.group if loc else s.item.group
        try:
            start, _ = self.session.doc.group_time(self.session.doc.group(gid))
            time = f"{int(start) // 60000:02}:{int(start) // 1000 % 60:02}.{int(start) % 1000:03}"
        except KeyError:
            time = "—"
        lang = self.session.doc.members[loc.member].lang if loc else s.item.lang
        layout.addWidget(QLabel(f"{time}　{language_name(lang)}　{KIND_NAMES[s.item.kind]}"))
        sentence = QLabel(self.sentence(loc) if loc else html.escape(s.item.target))
        sentence.setTextFormat(Qt.TextFormat.RichText)
        sentence.setWordWrap(True)
        layout.addWidget(sentence)
        for text in (f"{s.item.old} → {s.item.new}", f"理由：{s.item.reason}"):
            label = QLabel(text)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            layout.addWidget(label)
        cross = self.cross_break(s)
        status = {"ok": "", "found_by_text": "按文字找到（组 ID 不对）",
                  "not_found": "找不到原文（可能已经改过）", "ambiguous": "这一句里出现了几次，请选是哪一处",
                  "already_done": "已经改过了", "mismatch": f"现在的写法和意见里的“原来”不一样：{s.current or ''}"}[s.status]
        if cross:
            status = CROSS_BREAK
        if s.decision:
            status = f"{DECISION_NAMES[s.decision]}　{status}"
        widget.status = QLabel(status)
        widget.status.setTextFormat(Qt.TextFormat.PlainText)
        widget.status.setWordWrap(True)
        layout.addWidget(widget.status)
        if s.status == "ambiguous" and not s.decision:
            pick = QComboBox()
            pick.addItem("请选择是哪一处", None)
            for c in s.candidates:
                text, _ = review._shown(self.session.doc.members[c.member], s.display_rules)
                pick.addItem(f"{c.group} · {c.member} · 第 {c.displayed_a + 1} 字：{text}", c)
            pick.currentIndexChanged.connect(lambda _, p=pick, i=s.item: self.pick(i, p.currentData()))
            layout.addWidget(pick)
            widget.pick = pick
        actions = QHBoxLayout()
        widget.buttons = {}

        def button(text, callback, enabled=True):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.setEnabled(enabled)
            b.clicked.connect(callback)
            actions.addWidget(b)
            widget.buttons[text] = b

        can_apply = review._valid(self.session.editor, s) and not cross
        if s.item.kind in ("reading", "text") and not cross:
            button("采用", lambda: self.accept_one(s.item), can_apply)
            if s.item.kind == "text":
                button("改一下再采用…", lambda: self.edit_accept(s.item), can_apply)
        if s.item.kind not in ("reading", "text") or cross:
            button("标为已处理", lambda: self.decide(s.item, "handled"), not s.decision)
        button("不采用", lambda: self.decide(s.item, "rejected"), not s.decision)
        button("跳到这一组", lambda: self.jump(gid), any(g.id == gid for g in self.session.doc.groups))
        if not cross:
            button("试听", lambda: self.jump(gid, play=True), any(g.id == gid for g in self.session.doc.groups))
        layout.addLayout(actions)
        return widget

    def pick(self, item, location):
        self.picks[item.fingerprint] = location
        self.schedule_refresh()

    def jump(self, gid, *, play=False):
        self.main.select_group(gid, seek=not play)
        if play:
            self.main.play_group()
        else:
            self.main.activateWindow()

    def decide(self, item, decision):
        try:
            review.record_decision(self.session.episode.dir, item, decision)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "审议意见", f"保存审议记录失败：{exc}")
        self.refresh()

    def accept_one(self, item, new_override=None):
        try:
            s = self.suggestion(item, review.read_answers(self.session.episode.dir))
            if not self.cross_break(s) and review.apply(self.session.editor, s, new_override=new_override):
                self.decide(item, "accepted")
            else:
                self.refresh()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "审议意见", f"采用失败：{exc}")
            self.refresh()

    def edit_accept(self, item):
        dialog = EditAnswerDialog(item.new, self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.accept_one(item, dialog.edit.toPlainText())
        finally:
            dialog.deleteLater()

    def eligible_readings(self):
        chosen = []
        for s in self.suggestions:
            if s.item.kind != "reading" or s.status != "ok" or not review._valid(self.session.editor, s):
                continue
            loc = s.location
            # Conflicting opinions on the same characters still require individual decisions.
            if any(c.location.member == loc.member and c.location.a < loc.b and loc.a < c.location.b for c in chosen):
                continue
            chosen.append(s)
        return chosen

    def accept_all(self):
        self.refresh()
        eligible = self.eligible_readings()
        if not eligible:
            return
        if QMessageBox.question(self, "审议意见", f"采用 {len(eligible)} 条读音意见？",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            count = review.accept_all_readings(self.session.editor, eligible)
            if count:
                for s in eligible:
                    review.record_decision(self.session.episode.dir, s.item, "accepted")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "审议意见", f"采用失败：{exc}")
        self.refresh()
