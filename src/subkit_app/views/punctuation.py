"""Live two-layer punctuation settings with a cancellable libass preview."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal
from PySide6.QtGui import QFontDatabase, QStandardItem, QStandardItemModel, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QCompleter, QDialog, QDialogButtonBox,
    QFileDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from subkit import templates
from subkit.ass.text import remove_private, VisibleText
from subkit.core.display import (MERGE_KINDS, PUNCTUATION_CATEGORIES, PunctuationRules,
                                punctuation_fonts, punctuation_text_categories)
from subkit.core.punctuation_profiles import ProfileConflictError, apply, create_profile, matches
from subkit.languages import lane_label
from subkit.proc import CancelledError, KillableRun
from .. import help as help_ui

INHERIT = "沿用这一行的字体"
SAMPLES = {
    "translation": "「你好，世界！」他说：“这……真的吗？”——是的～ 1,000・T-shirt",
    "original": "「え〜っ、本当！？」……そう――だよ。ﾈｲｶﾞｰ、ハロー、すごーい、皆無事やったかー",
}
# Spaces/kana width have no font category; long vowels have no character rule.
ROWS = (
    ("halfwidth_kana", "半宽假名", ((False, "不改"), (True, "转为全宽（含长音）"))),
    ("comma", "逗号", (("off", "不改"), ("spaces", "换成两个空格"), ("ideographic", "换成「、」"))),
    ("dunhao", "顿号", ((False, "不改"), (True, "换成两个空格"))),
    ("period", "句号", ((False, "不改"), (True, "换成两个空格"))),
    ("colon", "冒号、分号", ()),
    ("ellipsis", "省略号", (("off", "不改"), ("single", "统一为「…」"),
        ("double", "统一为「……」"), ("keep_count", "保留个数，换成「…」"))),
    ("exclaim", "感叹号", (("off", "不改"), ("full", "全角「！」"), ("half", "半角「!」"))),
    ("question", "问号", (("off", "不改"), ("full", "全角「？」"), ("half", "半角「?」"))),
    ("quotes", "引号", (("off", "不改"), ("corner", "统一为「」"), ("white_corner", "统一为『』"),
        ("nested", "外层「」、内层『』"), ("nested_inverse", "外层『』、内层「」"))),
    ("brackets", "括号", ()),
    ("separator", "分隔号", ((False, "不改"), (True, "统一为「・」"))),
    ("tilde", "波浪号", (("off", "不改"), ("full", "全角「～」"), ("half", "半角「~」"), ("wave", "波浪号「〜」"))),
    ("dash", "破折号", (("off", "不改"), ("half", "半角「-」"), ("em", "全角「—」"),
        ("choon", "长音「ー」"), ("bar", "横线「―」"), ("fullwidth_hyphen", "全角连字符「－」"))),
    ("long_vowel", "长音「ー」", ()),
    ("spaces", "空格", ((False, "不改"), (True, "统一为两个半角空格"))),
)


def preview_events(template, lane, kind, sample, rule, playres):
    """Project styles, with the same category/font transforms as export."""
    styles = template.style_map()
    base = "CN" if lane == "translation" else "JP"
    candidates = {"dialogue": (base,), "song": ("OP." + base, "ED." + base, "IN." + base),
                  "other": (base + ".Top", base)}[kind]
    name = next((n for n in candidates if n in styles), next(iter(styles)))
    chars = replace(rule, fonts=(), font=None) if kind in rule.scope_chars else PunctuationRules()
    shown, _bounds, positions, _hidden = punctuation_text_categories(sample, chars)
    if kind in rule.scope_fonts:
        shown = punctuation_fonts(shown, positions, rule.fonts, styles[name], styles)
    px, py = playres
    events = []
    # Labels are drawn by the dialog, not in the subtitle font (a Japanese font lacks 显 and would
    # report a missing glyph that the user's text does not have).
    for text, y in ((remove_private(sample), py * PREVIEW_ROWS[0]), (remove_private(shown), py * PREVIEW_ROWS[1])):
        text = text.replace("\r\n", r"\N").replace("\n", r"\N")
        events.append(f"Dialogue: 0,0:00:00.00,0:00:03.00,{name},,0,0,0,,"
                      f"{{\\an5\\pos({px / 2:.1f},{y:.1f})}}{text}")
    return events


PREVIEW_ROWS = (.36, .68)          # vertical centres of the 原文 / 显示 lines (fraction of the frame)
PREVIEW_WIDTH = 1440               # render width; the two text bands are shown at this scale


def preview_bands(image: QPixmap, labels=("原文", "显示"), half_band: float = .075) -> QPixmap:
    """Crop the two text rows out of the rendered frame and stack them with labels, at full scale."""
    from PySide6.QtGui import QColor, QPainter

    w, h = image.width(), image.height()
    band = max(40, int(h * half_band))
    gutter = 64
    out = QPixmap(w + gutter, 2 * (2 * band) + 6)
    out.fill(QColor("#1f2328"))
    painter = QPainter(out)
    painter.setPen(QColor("#d0d7de"))
    for i, (label, row) in enumerate(zip(labels, PREVIEW_ROWS)):
        top = max(0, int(h * row) - band)
        y = i * (2 * band + 6)
        painter.drawPixmap(gutter, y, image.copy(0, top, w, 2 * band))
        painter.drawText(8, y, gutter - 12, 2 * band, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label)
    painter.end()
    return out


class _Signals(QObject):
    ready = Signal(object)
    failed = Signal(str)
    finished = Signal()


class _PreviewJob(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.setAutoDelete(False)
        self.signals = _Signals()
        self.runner = KillableRun()
        self.fn = fn

    def run(self):
        try:
            self.runner.check_cancelled()
            result = self.fn(self.runner)
            self.runner.check_cancelled()
            self.signals.ready.emit(result)
        except CancelledError:
            pass
        except Exception as exc:
            self.signals.failed.emit(str(exc))
        finally:
            self.signals.finished.emit()


class PunctuationDialog(QDialog):
    help_anchor = "display-rules"

    def __init__(self, project, store, settings, parent=None, *, session=None, current_gid=None,
                 on_apply=None, renderer=None):
        super().__init__(parent)
        self.project, self.store, self.settings = project, store, settings
        self.session, self.current_gid = session, current_gid
        self.on_apply = on_apply or (lambda: None)
        self.renderer = renderer or templates.render_scene
        self._had_settings = "settings" in project.data
        self._snapshot = copy.deepcopy(project.data.get("settings", {}))
        self._loading, self._closing, self._pending_result = True, False, None
        self._job, self._revision, self._render_revision = None, 0, -1
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.scratch = Path(tempfile.mkdtemp(prefix="punctuation-preview-"))
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._render)
        self.setWindowTitle("标点和空格")
        self.resize(1120, 850)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("方案："))
        self.profile_box = QComboBox()
        row.addWidget(self.profile_box, 1)
        self.profile_buttons = {}
        for text, fn in (("另存为方案…", self.save_profile), ("删除方案", self.delete_profile),
                         ("导出…", self.export_profile), ("导入…", self.import_profile),
                         ("设为新工程的默认", self.default_profile)):
            button = self._button(text, fn)
            self.profile_buttons[text] = button
            row.addWidget(button)
        layout.addLayout(row)
        self.font_model = QStandardItemModel(self)
        item = QStandardItem(INHERIT)
        item.setData(None, Qt.ItemDataRole.UserRole)
        self.font_model.appendRow(item)
        families = list(QFontDatabase.families())
        installed = {f.casefold() for f in families}
        used = {font for lane in ("translation", "original") for _, font in self._rules(lane).fonts}
        for font in families + sorted(f for f in used if f.casefold() not in installed):
            item = QStandardItem(font if font.casefold() in installed else font + "（未安装）")
            item.setData(font, Qt.ItemDataRole.UserRole)
            self.font_model.appendRow(item)
        self.tabs = QTabWidget()
        self.tables, self.rules, self.font_boxes, self.scope_boxes = {}, {}, {}, {}
        self.all_fonts, self.samples = {}, {}
        for lane in ("translation", "original"):
            page = QWidget()
            page_layout = QVBoxLayout(page)
            for layer, title in (("scope_chars", "字符标准化作用于："), ("scope_fonts", "显示字体作用于：")):
                row = QHBoxLayout()
                row.addWidget(QLabel(title))
                for kind, label in MERGE_KINDS:
                    box = QCheckBox(label)
                    row.addWidget(box)
                    self.scope_boxes[lane, layer, kind] = box
                    box.toggled.connect(lambda _on, lane=lane: self._changed(lane))
                row.addStretch()
                page_layout.addLayout(row)
            row = QHBoxLayout()
            row.addWidget(QLabel("全部标点："))
            self.all_fonts[lane] = self._font_box()
            row.addWidget(self.all_fonts[lane], 1)
            row.addWidget(self._button("套用到全部类别", lambda _=False, lane=lane: self.apply_all(lane)))
            page_layout.addLayout(row)
            table = QTableWidget(len(ROWS), 4)
            table.setHorizontalHeaderLabels(["类别", "字符标准化", "显示字体", "例子"])
            table.verticalHeader().hide()
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
            table.setColumnWidth(3, 220)
            self.tables[lane] = table
            for index, (category, label, choices) in enumerate(ROWS):
                table.setItem(index, 0, QTableWidgetItem(label))
                if choices:
                    box = QComboBox()
                    for value, text in choices:
                        box.addItem(text, value)
                    self.rules[lane, category] = box
                    table.setCellWidget(index, 1, box)
                    box.currentIndexChanged.connect(lambda _i, lane=lane: self._changed(lane))
                else:
                    table.setItem(index, 1, QTableWidgetItem("—" if category == "long_vowel" else "不改（只设字体）"))
                if category in PUNCTUATION_CATEGORIES:
                    box = self._font_box()
                    self.font_boxes[lane, category] = box
                    table.setCellWidget(index, 2, box)
                    box.currentTextChanged.connect(lambda _text, lane=lane: self._changed(lane))
                else:
                    table.setItem(index, 2, QTableWidgetItem("—"))
                table.setItem(index, 3, QTableWidgetItem())
            page_layout.addWidget(table, 1)
            sample = QPlainTextEdit(SAMPLES[lane])
            sample.setMaximumHeight(65)
            self.samples[lane] = sample
            sample.textChanged.connect(lambda lane=lane: self._sample_changed(lane))
            page_layout.addWidget(sample)
            row = QHBoxLayout()
            row.addWidget(self._button("用当前这一句", lambda _=False, lane=lane: self.use_current(lane)))
            row.addWidget(self._button("恢复示例", lambda _=False, lane=lane: self.samples[lane].setPlainText(SAMPLES[lane])))
            row.addStretch()
            page_layout.addLayout(row)
            self.tabs.addTab(page, lane_label(lane, project.languages))
        layout.addWidget(self.tabs, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("预览样式："))
        self.kind_box = QComboBox()
        for kind, label in (("dialogue", "对白"), ("song", "歌词"), ("other", "屏幕字")):
            self.kind_box.addItem(label, kind)
        row.addWidget(self.kind_box)
        row.addStretch()
        layout.addLayout(row)
        self.preview = QLabel("原文 / 显示：预览准备中…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(180)
        layout.addWidget(self.preview, 0)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        help_ui.dialog_buttons(self, buttons)
        layout.addWidget(buttons)
        for button in self.findChildren(QPushButton):
            button.setProperty("help_anchor", self.help_anchor)
            button.setAutoDefault(False)
        self.profile_box.activated.connect(self.select_profile)
        self.tabs.currentChanged.connect(self._schedule)
        self.kind_box.currentIndexChanged.connect(self._schedule)
        self._load()
        self._schedule()

    def _button(self, text, fn):
        button = QPushButton(text)
        button.setAutoDefault(False)
        button.setProperty("help_anchor", self.help_anchor)
        button.clicked.connect(fn)
        return button

    def _font_box(self):
        box = QComboBox()
        box.setEditable(True)
        box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        box.setModel(self.font_model)
        completion = QCompleter(self.font_model, box)
        completion.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completion.setFilterMode(Qt.MatchFlag.MatchContains)
        box.setCompleter(completion)
        return box

    @staticmethod
    def _font(box):
        if box.currentText() == box.itemText(box.currentIndex()):
            return box.currentData()
        return box.currentText().strip() or None

    def _set_font(self, box, font):
        index = box.findData(font)
        if font and index < 0:
            item = QStandardItem(font + "（未安装）")
            item.setData(font, Qt.ItemDataRole.UserRole)
            self.font_model.appendRow(item)
            index = self.font_model.rowCount() - 1
        box.setCurrentIndex(max(0, index))

    def _rules(self, lane):
        # Project.settings merges legacy defaults; absent lanes are new settings,
        # while explicitly saved legacy lanes must retain their old font scope.
        data = self.project.data.get("settings", {}).get("display", {}).get("punctuation", {}).get(lane, {})
        return PunctuationRules.from_settings(data)

    def _load(self):
        self._loading = True
        for lane in self.samples:
            rule = self._rules(lane)
            for (l, key), box in self.rules.items():
                if l == lane:
                    box.setCurrentIndex(box.findData(getattr(rule, key)))
            for (l, category), box in self.font_boxes.items():
                if l == lane:
                    self._set_font(box, dict(rule.fonts).get(category))
            for (l, layer, kind), box in self.scope_boxes.items():
                if l == lane:
                    box.setChecked(kind in getattr(rule, layer))
            self._examples(lane)
        self._loading = False
        self._profiles()

    def _changed(self, lane):
        if self._loading or self._closing:
            return
        data = self._rules(lane).to_settings()
        data.update({key: box.currentData() for (l, key), box in self.rules.items() if l == lane})
        data["fonts"] = {key: self._font(box) for (l, key), box in self.font_boxes.items() if l == lane}
        for layer in ("scope_chars", "scope_fonts"):
            data[layer] = [kind for kind, _ in MERGE_KINDS if self.scope_boxes[lane, layer, kind].isChecked()]
        self.project.data.setdefault("settings", {}).setdefault("display", {}).setdefault("punctuation", {})[lane] = PunctuationRules.from_settings(data).to_settings()
        self._apply()

    def _apply(self):
        self.project.save()
        self.on_apply()
        if self.session:
            self.session.request_preview()
        self._profiles()
        self._schedule()

    def apply_all(self, lane):
        font = self._font(self.all_fonts[lane])
        self._loading = True
        for (l, _category), box in self.font_boxes.items():
            if l == lane:
                self._set_font(box, font)
        self._loading = False
        self._changed(lane)

    def _examples(self, lane):
        sample = self.samples[lane].toPlainText()
        _shown, _bounds, categories, _hidden = punctuation_text_categories(sample, PunctuationRules())
        plain = VisibleText(sample).text
        for index, (category, _label, _choices) in enumerate(ROWS):
            if category == "halfwidth_kana":
                self.tables[lane].item(index, 3).setText("ｶﾞ → ガ、ﾊﾟ → パ、ﾈｲｶﾞｰ → ネイガー")
                self.tables[lane].item(index, 3).setToolTip("只转换显示和导出；包含半宽长音，原文不变。")
                continue
            positions = categories.get(category, set())
            example = "".join(f"【{ch}】" if i in positions else ch for i, ch in enumerate(plain))
            self.tables[lane].item(index, 3).setText(example if positions else "—")
            self.tables[lane].item(index, 3).setToolTip(example)

    def _sample_changed(self, lane):
        self._examples(lane)
        self._schedule()

    def use_current(self, lane):
        if not self.session or not self.current_gid:
            return
        group = self.session.doc.group(self.current_gid)
        text = "\n".join(self.session.doc.members[mid].text for mid in group.members
                         if self.session.doc.members[mid].lane == lane)
        self.samples[lane].setPlainText(text)

    def _profiles(self):
        try:
            profiles = self.store.list()
        except (OSError, ValueError) as exc:
            self.warning.setText(str(exc))
            profiles = []
        remembered = self.project.settings().get("display", {}).get("punctuation_profile")
        default = self.settings.value("display/default_punctuation_profile", "")
        self.profile_box.blockSignals(True)
        self.profile_box.clear()
        self.profile_box.addItem("（未存为方案）", None)
        for profile in profiles:
            name = profile["name"]
            label = name + ("（已修改）" if name == remembered and not matches(self.project.settings(), profile) else "")
            label += "（默认）" if name == default else ""
            self.profile_box.addItem(label, name)
        self.profile_box.setCurrentIndex(max(0, self.profile_box.findData(remembered)))
        self.profile_box.blockSignals(False)
        for label in ("删除方案", "导出…", "设为新工程的默认"):
            self.profile_buttons[label].setEnabled(self.profile_box.currentData() is not None)

    def select_profile(self, index):
        name = self.profile_box.itemData(index)
        if name and self._profile_action(lambda: apply(self.project.data.setdefault("settings", {}), self.store.load(name))):
            self._load()
            self._apply()

    def _profile_action(self, fn):
        try:
            fn()
            return True
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "标点和空格", str(exc))
            return False

    def save_profile(self):
        name, ok = QInputDialog.getText(self, "另存为方案", "方案名称：")
        if not ok:
            return
        def save():
            profile = create_profile(name, self._rules("translation"), self._rules("original"))
            try:
                saved = self.store.save(profile)
            except ProfileConflictError:
                if QMessageBox.question(self, "另存为方案", "已有同名方案，覆盖吗？",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                    return
                saved = self.store.save(profile, overwrite=True)
            # Save the canonical lanes too: untouched new lanes are absent from
            # the project, whose merged defaults still contain legacy fields.
            apply(self.project.data.setdefault("settings", {}), saved)
            self._apply()
        self._profile_action(save)

    def delete_profile(self):
        name = self.profile_box.currentData()
        if name and QMessageBox.question(self, "删除方案", f"删除方案「{name}」？工程里已用的设置不变。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self._profile_action(lambda: self.store.delete(name))
            self._profiles()
            self.on_apply()

    def export_profile(self):
        name = self.profile_box.currentData()
        if not name:
            return
        path, _filter = QFileDialog.getSaveFileName(self, "导出标点方案", name + ".json", "JSON (*.json)")
        if path:
            self._profile_action(lambda: self.store.export_to(name, Path(path)))

    def import_profile(self):
        path, _filter = QFileDialog.getOpenFileName(self, "导入标点方案", "", "JSON (*.json)")
        if not path:
            return
        def load():
            try:
                self.store.import_from(Path(path))
            except ProfileConflictError:
                box = QMessageBox(self)
                box.setWindowTitle("导入标点方案")
                box.setText("已有同名方案，覆盖或改名？")
                overwrite = box.addButton("覆盖", QMessageBox.ButtonRole.AcceptRole)
                rename = box.addButton("改名…", QMessageBox.ButtonRole.ActionRole)
                box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
                box.help_anchor = self.help_anchor
                help_ui.dialog_buttons(box)
                box.exec()
                if box.clickedButton() is overwrite:
                    self.store.import_from(Path(path), overwrite=True)
                elif box.clickedButton() is rename:
                    name, ok = QInputDialog.getText(self, "改名", "方案名称：")
                    if ok:
                        self.store.import_from(Path(path), name=name)
            self._profiles()
            self.on_apply()
        self._profile_action(load)

    def default_profile(self):
        name = self.profile_box.currentData()
        if name:
            self.settings.setValue("display/default_punctuation_profile", name)
            self._profiles()
            self.on_apply()

    def _schedule(self, *_args):
        if not self._loading and not self._closing:
            self._revision += 1
            self.timer.start()

    def _render(self):
        if self._closing or self._job is not None:
            return
        lane = "translation" if self.tabs.currentIndex() == 0 else "original"
        template = self.project.template()
        rule, sample, kind = self._rules(lane), self.samples[lane].toPlainText(), self.kind_box.currentData()
        self._render_revision = self._revision
        revision = self._render_revision
        # Geometry probing belongs on the worker too; it can invoke ffprobe.
        def render(runner):
            playres = (self.session.episode.effective_playres() if self.session else
                       tuple(self.project.settings()["playres"]))
            events = preview_events(template, lane, kind, sample, rule, playres)
            return self.renderer(template.serialize(), self.scratch / f"{revision}.png", event_lines=events,
                                 playres=playres, guides=False, width=PREVIEW_WIDTH, runner=runner)
        self._job = _PreviewJob(render)
        self._job.signals.ready.connect(self._ready)
        self._job.signals.failed.connect(self._failed)
        self._job.signals.finished.connect(self._collect)
        self.pool.start(self._job)

    def _ready(self, result):
        if not self._closing and self._render_revision == self._revision:
            pixmap = preview_bands(QPixmap(str(result.png)))
            room = max(200, self.preview.width() - 8)
            if pixmap.width() > room:          # only shrink to fit; never enlarge past the rendered pixels
                pixmap = pixmap.scaledToWidth(room, Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pixmap)
            self.warning.setText(result.warning())

    def _failed(self, error):
        if not self._closing and self._render_revision == self._revision:
            self.warning.setText(error)

    def _collect(self):
        # A finished signal can arrive before QRunnable.run has returned.
        if not self.pool.waitForDone(0):
            QTimer.singleShot(10, self._collect)
            return
        self._job = None
        if self._pending_result is not None:
            super().done(self._pending_result)
        elif self._render_revision != self._revision:
            self.timer.start()

    def done(self, result):
        if self._closing:
            return
        self._closing = True
        self.timer.stop()
        if result == QDialog.DialogCode.Rejected:
            if self._had_settings:
                self.project.data["settings"] = copy.deepcopy(self._snapshot)
            else:
                self.project.data.pop("settings", None)
            self.project.save()
            self.on_apply()
            if self.session:
                self.session.request_preview()
        self._pending_result = result
        if self._job:
            self._job.runner.cancel()
        else:
            super().done(result)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F1:
            help_ui.open_manual(self.help_anchor)
            event.accept()
        else:
            super().keyPressEvent(event)
