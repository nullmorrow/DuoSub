"""Project-wide display replacements, edited as one cancellable draft."""
from __future__ import annotations

import copy

from PySide6.QtCore import QEvent, QItemSelectionModel, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QTableWidget, QVBoxLayout)

from subkit.core.replacements import apply_replacements, load_rules, normalize_rules, save_rules
from subkit.languages import lane_label
from .. import help as app_help


SAMPLES = {"original": "黒井津さん、世界征服を始めましょう。",
           "translation": "黑井津小姐，开始征服世界吧。"}


class ReplacementsDialog(QDialog):
    help_anchor = "display-replacements"

    def __init__(self, project, parent=None, *, samples=None):
        super().__init__(parent)
        self.project = project
        self.saved = False
        self._snapshot = load_rules(project)
        self._loading = False
        self._composing = set()
        self._sample_texts = dict(SAMPLES)
        self._sample_texts.update(samples or {})
        self._current_samples = set(samples or {})
        self.setWindowTitle("批量替换")
        self.resize(1000, 720)
        layout = QVBoxLayout(self)
        note = QLabel("只在当前工程显示和导出时替换正文，编辑原文不变。适合删除多余字、统一人名。\n"
                      "词 B 留空表示删除。区分大小写，按字面匹配；每个原文位置优先用列表靠上的规则，替换结果不会再次替换。\n"
                      "所有改动先留在草稿；点“保存全部”才应用到整个工程。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["启用", "词 A（原文字面量）", "词 B（留空删除）", "作用轴"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.add_button, self.delete_button, self.up_button, self.down_button = (
            QPushButton(text) for text in ("新增", "删除所选", "上移", "下移"))
        for button in (self.add_button, self.delete_button, self.up_button, self.down_button):
            button.setAutoDefault(False)
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        preview_row = QHBoxLayout()
        self.sample_lane = QComboBox()
        for lane in ("original", "translation"):
            self.sample_lane.addItem(lane_label(lane, project.languages), lane)
        self.sample_lane.setCurrentIndex(self.sample_lane.findData(
            next(iter(self._current_samples), "translation")))
        preview_row.addWidget(QLabel("纯文本对照："))
        preview_row.addWidget(self.sample_lane)
        self.sample_origin = QLabel()
        preview_row.addWidget(self.sample_origin)
        preview_row.addStretch(1)
        layout.addLayout(preview_row)
        preview_note = QLabel("只演示下方纯文本的替换结果，不含其他显示规则、样式或注音。可直接改示例文字，不会写回字幕。")
        preview_note.setWordWrap(True)
        layout.addWidget(preview_note)
        previews = QHBoxLayout()
        self.sample = QPlainTextEdit()
        self.sample.setPlaceholderText("输入要对照的纯文本…")
        self.sample.setMaximumHeight(110)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("替换后")
        self.output.setMaximumHeight(110)
        for title, widget in (("替换前（可编辑）", self.sample), ("替换后", self.output)):
            column = QVBoxLayout()
            column.addWidget(QLabel(title))
            column.addWidget(widget)
            previews.addLayout(column, 1)
        layout.addLayout(previews)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.save_button = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText("保存全部")
        self.save_button.setAutoDefault(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        app_help.dialog_buttons(self, self.buttons, default=False)
        layout.addWidget(self.buttons)
        self.add_button.clicked.connect(self.add_rule)
        self.delete_button.clicked.connect(self.delete_rule)
        self.up_button.clicked.connect(lambda: self.move_rule(-1))
        self.down_button.clicked.connect(lambda: self.move_rule(1))
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.sample.textChanged.connect(self._preview)
        self.sample_lane.currentIndexChanged.connect(self._select_sample)
        self.help_shortcut = QShortcut(QKeySequence("F1"), self)
        self.help_shortcut.activated.connect(lambda: app_help.open_manual(self.help_anchor))
        self._populate(self._snapshot)
        self._select_sample()

    def rules(self):
        return [{"enabled": self.table.cellWidget(i, 0).isChecked(),
                 "source": self.table.cellWidget(i, 1).text(),
                 "target": self.table.cellWidget(i, 2).text(),
                 "lane": self.table.cellWidget(i, 3).currentData()}
                for i in range(self.table.rowCount())]

    def _populate(self, rules, selected=-1):
        self._loading = True
        self._composing.clear()
        self.table.setRowCount(0)
        self.table.setRowCount(len(rules))
        for i, rule in enumerate(rules):
            enabled = QCheckBox()
            enabled.setChecked(rule["enabled"])
            self.table.setCellWidget(i, 0, enabled)
            enabled.toggled.connect(self._changed)
            for column, name in ((1, "source"), (2, "target")):
                edit = QLineEdit(rule[name])
                edit.setPlaceholderText("必须填写" if name == "source" else "留空表示删除")
                edit.setToolTip("保留首尾空格；不允许反斜杠、半角大括号、换行或 NUL。")
                edit.installEventFilter(self)
                edit.textChanged.connect(self._changed)
                self.table.setCellWidget(i, column, edit)
            lane = QComboBox()
            for key, label in (("original", lane_label("original", self.project.languages)),
                               ("translation", lane_label("translation", self.project.languages)),
                               ("both", "两轴")):
                lane.addItem(label, key)
            lane.setCurrentIndex(lane.findData(rule["lane"]))
            lane.currentIndexChanged.connect(self._changed)
            self.table.setCellWidget(i, 3, lane)
        self._loading = False
        if 0 <= selected < len(rules):
            self.table.selectRow(selected)
        self._changed()

    def _selection_changed(self):
        i, count = self.table.currentRow(), self.table.rowCount()
        selected = 0 <= i < count
        self.delete_button.setEnabled(selected and not self._composing)
        self.up_button.setEnabled(selected and i > 0 and not self._composing)
        self.down_button.setEnabled(selected and i + 1 < count and not self._composing)

    def add_rule(self):
        if self._composing:
            return
        rules = self.rules() + [{"enabled": True, "source": "", "target": "", "lane": "both"}]
        self._populate(rules, len(rules) - 1)
        self.table.cellWidget(len(rules) - 1, 1).setFocus()

    def delete_rule(self):
        i = self.table.currentRow()
        if self._composing or i < 0:
            return
        rules = self.rules()
        del rules[i]
        self._populate(rules, min(i, len(rules) - 1))

    def move_rule(self, step):
        i = self.table.currentRow()
        rules = self.rules()
        if self._composing or not (0 <= i < len(rules) and 0 <= i + step < len(rules)):
            return
        rules[i], rules[i + step] = rules[i + step], rules[i]
        self._populate(rules, i + step)

    def _select_sample(self, *_):
        lane = self.sample_lane.currentData()
        self.sample.setPlainText(self._sample_texts.get(lane, ""))
        self.sample_origin.setText("来自当前一句的纯文本" if lane in self._current_samples else "示例纯文本")
        self._preview()

    def _preview(self):
        if self._loading:
            return
        try:
            rules = normalize_rules(self.rules())
            self.output.setPlainText(apply_replacements(self.sample.toPlainText(), rules,
                                                      lane=self.sample_lane.currentData()))
        except ValueError:
            self.output.clear()

    def _changed(self, *_):
        if self._loading:
            return
        rules = self.rules()
        dirty = rules != self._snapshot
        try:
            normalize_rules(rules)
            valid = True
            self.status.setText((f"待保存 {len(rules)} 条规则；保存后应用到整个工程。" if dirty
                                 else f"当前工程有 {len(rules)} 条规则，没有待保存的修改。"))
        except ValueError as exc:
            valid = False
            self.status.setText(f"请修正规则：{exc}")
        self.save_button.setEnabled(valid and dirty and not self._composing)
        self.add_button.setEnabled(not self._composing)
        self._selection_changed()
        self._preview()

    def eventFilter(self, obj, event):
        if isinstance(obj, QLineEdit):
            if event.type() == QEvent.Type.InputMethod:
                if event.preeditString():
                    self._composing.add(obj)
                else:
                    self._composing.discard(obj)
                self._changed()
            elif event.type() == QEvent.Type.FocusIn:
                for i in range(self.table.rowCount()):
                    column = next((c for c in (1, 2) if obj is self.table.cellWidget(i, c)), None)
                    if column is not None:
                        # selectRow() activates column 0 and steals focus to its checkbox.
                        self.table.setCurrentCell(i, column, QItemSelectionModel.SelectionFlag.ClearAndSelect
                                                  | QItemSelectionModel.SelectionFlag.Rows)
                        break
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def accept(self):
        if self._composing:
            return
        try:
            rules = normalize_rules(self.rules())
            self.saved = save_rules(self.project, rules)
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, self.windowTitle(), f"没有保存：{exc}")
            return
        self._snapshot = copy.deepcopy(rules)
        super().accept()
