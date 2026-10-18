"""Draft-based management of this Project's word readings and explicit word splits."""
from __future__ import annotations

from collections import Counter

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from subkit.core.project_readings import KIND_NAMES, ProjectReadingsDraft, normalize_rule_input
from .. import help as app_help


class ProjectReadingRuleDialog(QDialog):
    help_anchor = "project-readings"

    def __init__(self, rule=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑工程词语" if rule else "新增工程词语")
        self.resize(600, 300)
        self._composing = False
        layout = QVBoxLayout(self)
        note = QLabel("整词读音／合词：作为一个词处理。整词分段注音：仍是一个词，分段放置振假名。固定拆词：每段是独立的词。\n分段表记与读音用 / 一一对应。同一表记的规则互斥；编辑时会替换该表记的其他类型规则。这里只改草稿，回到管理窗口点“保存全部”才写入工程。")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.kind = QComboBox()
        for key, label in KIND_NAMES.items():
            self.kind.addItem(label, key)
        self.kind.setCurrentIndex(self.kind.findData(rule.kind if rule else "word"))
        self.word_edit = QLineEdit(rule.surface_parts if rule else "")
        self.reading_edit = QLineEdit(rule.reading_parts if rule else "")
        form.addRow("类型", self.kind)
        form.addRow("表记", self.word_edit)
        form.addRow("读音", self.reading_edit)
        layout.addLayout(form)
        self.preview = QLabel()
        self.preview.setTextFormat(Qt.TextFormat.PlainText)
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("加入草稿")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        app_help.dialog_buttons(self, self.buttons)
        layout.addWidget(self.buttons)
        self.kind.currentIndexChanged.connect(self.validate)
        for edit in (self.word_edit, self.reading_edit):
            edit.installEventFilter(self)
            edit.textChanged.connect(self.validate)
        self.help_shortcut = QShortcut(QKeySequence("F1"), self)
        self.help_shortcut.activated.connect(lambda: app_help.open_manual(self.help_anchor))
        self.validate()

    def values(self):
        kind = self.kind.currentData()
        parts = normalize_rule_input(kind, self.word_edit.text(), self.reading_edit.text())
        return kind, [w for w, _r in parts], [r for _w, r in parts]

    def validate(self, *_):
        try:
            _kind, words, readings = self.values()
            self.preview.setText(" ／ ".join(f"{w}（{r}）" for w, r in zip(words, readings)))
            valid = True
        except ValueError as exc:
            self.preview.setText(str(exc))
            valid = False
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid and not self._composing)

    def accept(self):
        if not self._composing:
            try:
                self.values()
            except ValueError:
                self.validate()
                return
            super().accept()

    def eventFilter(self, obj, event):
        if obj in (self.word_edit, self.reading_edit) and event.type() == QEvent.Type.InputMethod:
            self._composing = bool(event.preeditString())
            self.validate()
        return super().eventFilter(obj, event)


class ProjectReadingsDialog(QDialog):
    help_anchor = "project-readings"

    def __init__(self, project, editor=None, parent=None):
        super().__init__(parent)
        self.draft = ProjectReadingsDraft(project)
        self.editor = editor
        self.saved, self.changed_surfaces, self.updated_annotations = False, (), 0
        self.backup = None
        self.setWindowTitle("工程词语与分词规则")
        self.resize(900, 600)
        layout = QVBoxLayout(self)
        note = QLabel("只管理当前工程的词库，不读取全局或其他工程。整词分段注音仍是一个词；固定拆词才会产生多个词。\n所有操作先留在草稿；保存一次应用全部改动。当前集只更新相关表记的自动注音，手动、确认过、不标注的内容保留；其他集以后刷新或合并时生效。")
        note.setWordWrap(True)
        layout.addWidget(note)
        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索表记或读音…")
        self.kind = QComboBox()
        self.kind.addItem("全部类型", None)
        for key, label in KIND_NAMES.items():
            self.kind.addItem(label, key)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.kind)
        layout.addLayout(filters)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["表记（/ 分段）", "类型", "读音（/ 分段）", "备注"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        actions = QHBoxLayout()
        self.add_button, self.edit_button, self.delete_button = (QPushButton(text) for text in ("新增…", "编辑…", "删除"))
        for button in (self.add_button, self.edit_button, self.delete_button):
            button.setAutoDefault(False)
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.save_button = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText("保存全部")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        app_help.dialog_buttons(self, self.buttons)
        layout.addWidget(self.buttons)
        self.add_button.clicked.connect(lambda: self.edit_rule(False))
        self.edit_button.clicked.connect(lambda: self.edit_rule(True))
        self.delete_button.clicked.connect(self.delete_rule)
        self.table.cellDoubleClicked.connect(lambda *_: self.edit_rule(True))
        self.table.itemSelectionChanged.connect(self.selection_changed)
        self.search.textChanged.connect(self.refresh)
        self.kind.currentIndexChanged.connect(self.refresh)
        self.help_shortcut = QShortcut(QKeySequence("F1"), self)
        self.help_shortcut.activated.connect(lambda: app_help.open_manual(self.help_anchor))
        self.refresh()

    def selected_rule(self):
        row = self.table.currentRow()
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def selection_changed(self):
        selected = self.selected_rule() is not None
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def refresh(self, *_):
        query, kind = self.search.text().casefold().strip(), self.kind.currentData()
        rows = self.draft.rows()
        counts = Counter(r.surface for r in rows)
        self._rows = [r for r in rows if (kind is None or r.kind == kind)
                      and (not query or query in "\n".join((r.surface_parts, r.reading_parts, r.key, r.surface,
                                                           "".join(reading for _word, reading in r.parts))).casefold())]
        self.table.setRowCount(len(self._rows))
        for i, rule in enumerate(self._rows):
            duplicate = counts[rule.surface] > 1
            remark = ("需要修正" if rule.error else "同表记有多条，编辑将统一替换" if duplicate
                      else "旧词库规则" if rule.table in ("readings", "splits") else "")
            for j, value in enumerate((rule.surface_parts, KIND_NAMES[rule.kind], rule.reading_parts, remark)):
                cell = QTableWidgetItem(value)
                cell.setToolTip((rule.error + "\n" if rule.error else "") +
                                f"工程规则：{rule.surface}。编辑会统一替换同一表记的对应旧规则；未编辑的规则保留。")
                self.table.setItem(i, j, cell)
        self.save_button.setEnabled(self.draft.dirty)
        changed = self.draft.changed_surfaces()
        self.status.setText(f"显示 {len(self._rows)} 条规则。" +
                            ("待保存，受影响表记：" + "、".join(changed) if changed else "没有待保存的修改。"))
        self.selection_changed()

    def edit_rule(self, editing=True):
        original = self.selected_rule() if editing else None
        if editing and original is None:
            return
        dialog = ProjectReadingRuleDialog(original, self)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            try:
                kind, words, readings = dialog.values()
                self.draft.set_rule(kind, words, readings, original=original)
            except ValueError as exc:
                QMessageBox.warning(self, self.windowTitle(), str(exc))
                return
        finally:
            dialog.deleteLater()
        self.refresh()

    def delete_rule(self):
        rule = self.selected_rule()
        if rule is None:
            return
        if QMessageBox.question(self, self.windowTitle(), f"从草稿删除「{rule.surface}」的{KIND_NAMES[rule.kind]}？\n点“保存全部”后才写入工程；只刷新相关的自动注音。") != QMessageBox.StandardButton.Yes:
            return
        self.draft.remove_rule(rule)
        self.refresh()

    def accept(self):
        try:
            result = self.draft.save(editor=self.editor)
        except Exception as exc:
            QMessageBox.warning(self, self.windowTitle(), f"没有保存：{exc}")
            return
        self.changed_surfaces, self.updated_annotations = result.changed_surfaces, result.updated_annotations
        self.backup = result.backup
        self.saved = bool(result.changed_surfaces)
        super().accept()
