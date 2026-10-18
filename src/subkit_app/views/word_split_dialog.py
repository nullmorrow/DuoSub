"""Explicit word boundaries, with an optional Project reading rule."""

from __future__ import annotations

import re

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                               QLabel, QLineEdit, QPushButton, QVBoxLayout)

from subkit.reading.dictionary import normalize_word_split
from subkit.reading.records import romaji_to_kana


class WordSplitDialog(QDialog):
    help_anchor = "readings"

    def __init__(self, surface: str, *, parts=None, scope="here", analyzer=None,
                 existing_rule=False, parent=None):
        super().__init__(parent)
        self.surface = surface
        self._composing = False
        self.remove_rule = False
        self._scope = "project" if scope in ("project", 2) else "here"
        self.setWindowTitle("按指定词拆开…")
        self.resize(570, 260)
        layout = QVBoxLayout(self)
        title = QLabel(f"选中文字：{surface}")
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        layout.addWidget(title)
        explanation = QLabel("用 / 分开每个词，例如 世界/征服；读音也用 / 一一对应。原文不变，每个词会有独立的振假名。")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        if parts is None and analyzer is not None:
            try:
                parts = [{"surface": word, "reading": reading}
                         for word, reading in analyzer.split_a(surface)]
                if "".join(p["surface"] for p in parts) != surface:
                    parts = None
            except Exception:
                parts = None
        form = QFormLayout()
        self.word_edit = QLineEdit("/".join(p["surface"] for p in parts) if parts else surface)
        self.reading_edit = QLineEdit("/".join(p["reading"] for p in parts) if parts else "")
        self.word_edit.setPlaceholderText("世界/征服")
        self.reading_edit.setPlaceholderText("せかい/せいふく（也可输入罗马字）")
        form.addRow("分词表记：", self.word_edit)
        form.addRow("分别读音：", self.reading_edit)
        layout.addLayout(form)
        note = QLabel("工程规则只更新自动读音，其他地方手动合过、确认过或不标注的读音保持不变。其他集在以后合并或刷新自动读音时生效。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.preview = QLabel()
        self.preview.setTextFormat(Qt.TextFormat.PlainText)
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.apply_here_button = self.buttons.addButton("应用到当前句子", QDialogButtonBox.ButtonRole.AcceptRole)
        self.apply_project_button = self.buttons.addButton("应用到整个 Project", QDialogButtonBox.ButtonRole.AcceptRole)
        self.apply_here_button.setToolTip("只拆开当前句子里选中的这一处，不写入工程词库。")
        self.apply_project_button.setToolTip("保存为工程拆词规则，预览影响后更新自动读音；其他人工读音保留。")
        self.apply_here_button.clicked.connect(lambda: self._apply_scope("here"))
        self.apply_project_button.clicked.connect(lambda: self._apply_scope("project"))
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.rejected.connect(self.reject)
        self.remove_button = QPushButton("删除此工程规则")
        self.remove_button.setAutoDefault(False)
        self.remove_button.setToolTip("删除规则后只恢复自动读音；已经手动拆开的这一处保持不变。")
        self.remove_button.clicked.connect(self._remove)
        self.buttons.addButton(self.remove_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.remove_button.setVisible(bool(existing_rule))
        from ..help import dialog_buttons

        default_button = self.apply_project_button if self._scope == "project" else self.apply_here_button
        dialog_buttons(self, self.buttons, primary=default_button)
        layout.addWidget(self.buttons)
        for edit in (self.word_edit, self.reading_edit):
            edit.installEventFilter(self)
            edit.textChanged.connect(self._validate)
        self._validate()
        self.word_edit.setFocus()
        self.word_edit.selectAll()

    def values(self) -> tuple[list[dict[str, str]], str]:
        words = [word.strip() for word in re.split(r"[/／]", self.word_edit.text())]
        readings = [reading.strip() for reading in re.split(r"[/／]", self.reading_edit.text())]
        if len(words) != len(readings):
            raise ValueError("表记与读音的分段数必须相同，用 / 一一对应")
        readings = [romaji_to_kana(reading) if reading and reading.isascii() else reading for reading in readings]
        parts = normalize_word_split(self.surface, [{"surface": word, "reading": reading}
                                                   for word, reading in zip(words, readings)])
        return parts, self._scope

    def _validate(self, *_args):
        try:
            parts, _scope = self.values()
            self.preview.setText(" → ".join(f"{p['surface']}（{p['reading']}）" for p in parts))
            valid = True
        except ValueError as exc:
            self.preview.setText(str(exc))
            valid = False
        for button in (self.apply_here_button, self.apply_project_button):
            button.setEnabled(valid and not self._composing)

    def _apply_scope(self, scope):
        self._scope = scope
        self.accept()

    def accept(self):
        if self._composing:
            return
        try:
            self.values()
        except ValueError:
            self._validate()
            return
        self.remove_rule = False
        super().accept()

    def _remove(self):
        if self._composing:
            return
        self.remove_rule = True
        super().accept()

    def eventFilter(self, obj, event):
        if obj in (self.word_edit, self.reading_edit) and event.type() == QEvent.Type.InputMethod:
            self._composing = bool(event.preeditString())
            self._validate()
        return super().eventFilter(obj, event)
