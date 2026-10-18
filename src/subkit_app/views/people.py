"""Project character dock and explicit pending-label/name decisions."""

from __future__ import annotations

import copy
import re
import threading

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QDockWidget,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QTableView, QVBoxLayout, QWidget, QGroupBox,
                               QScrollArea, QPlainTextEdit, QTableWidget, QTableWidgetItem, QCheckBox,
                               QInputDialog, QApplication, QHeaderView)

from subkit.core.name_finder import find_names
from subkit.core.people import NameForm, Character, PERSONA_FIELDS, suggest_character, character_display_name
from subkit.core import bangumi
from subkit.core.self_check import check as check_self_references
from subkit.languages import LANGUAGES, get_language, language_name, normalize_language_code
from subkit.textutil import kata_to_hira

from .. import help as app_help
from . import placement


class SpeakerPicker(QDialog):
    """Focusable tool window; no Qt.Popup grabs during Windows IME composition."""
    help_anchor = "people"
    chosen = Signal(object, bool, bool, bool)

    def __init__(self, parent, book, member, recent, *, label="", single_count=0, whole_count=0,
                 settings=None, affected_count=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("说话人")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(300, 350)
        self._settings = settings if settings is not None else getattr(parent, "settings", None)
        self._affected_count = affected_count
        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索名字、别名")
        layout.addWidget(self.search)
        self.list = QListWidget()
        layout.addWidget(self.list)
        rank = {cid: i for i, cid in enumerate(recent)}
        for ch in sorted(book.characters, key=lambda ch: rank.get(ch.id, len(rank))):
            item = QListWidgetItem(character_display_name(ch, member.lang, format="short"), self.list)
            item.setData(Qt.ItemDataRole.UserRole, ch.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, " ".join(
                [item.text(), *(ch.full_name(lang) for lang in ch.names),
                 *(ch.name(lang, "short") for lang in ch.names), *ch.aliases]).casefold())
        self.whole = QCheckBox("整组同一人")
        self.whole.setChecked(self._option("speaker_whole_group", True))
        layout.addWidget(self.whole)
        self._label = label
        self._same_label_counts = single_count, whole_count
        self.same_label = QCheckBox()
        self.same_label.setChecked(self._option("speaker_same_label", True))
        layout.addWidget(self.same_label)
        self.fill_down = QCheckBox("往后顺延到下一个有说话人的行")
        self.fill_down.setChecked(self._option("speaker_fill_down", False))
        layout.addWidget(self.fill_down)
        self.count_line = QLabel()
        self.count_line.setStyleSheet("color: #8b949e;")
        for checkbox, key in ((self.whole, "speaker_whole_group"),
                              (self.same_label, "speaker_same_label"),
                              (self.fill_down, "speaker_fill_down")):
            checkbox.toggled.connect(lambda checked, name=key: self._remember(name, checked))
            checkbox.toggled.connect(self._update_same_label)
        self._update_same_label()
        for label, cid in (("新建人物…", "new"), ("群众", "crowd"),
                           ("无说话人（不是对白）", "nobody"), ("清除（回到原标签）", None)):
            button = QPushButton(label)
            button.setAutoDefault(False)
            button.clicked.connect(lambda _=False, value=cid: self.choose(value))
            layout.addWidget(button)
        layout.addWidget(self.count_line)
        buttons = QHBoxLayout()
        close = QPushButton("关闭")
        close.clicked.connect(self.reject)
        buttons.addWidget(close)
        app_help.dialog_buttons(self, buttons, default=False)
        layout.addLayout(buttons)
        self.search.textChanged.connect(self.filter)
        self.list.itemClicked.connect(lambda item: self.choose(item.data(Qt.ItemDataRole.UserRole)))
        self.list.itemActivated.connect(lambda item: self.choose(item.data(Qt.ItemDataRole.UserRole)))

    def place_at(self, anchor):
        area = placement.screen_area(anchor, self)
        self.setMaximumHeight(max(1, area.height() - 40))
        self.list.setMinimumHeight(0)
        placement.place_near(self, anchor)

    def filter(self, text):
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setHidden(text.casefold() not in item.data(Qt.ItemDataRole.UserRole + 1))

    def _option(self, key, default):
        return self._settings.value(key, default, type=bool) if self._settings is not None else default

    def _remember(self, key, value):
        if self._settings is not None:
            self._settings.setValue(key, value)

    def _update_same_label(self, _checked=None):
        count = self._same_label_counts[int(self.whole.isChecked())]
        self.same_label.setText(f"标着「{self._label}」的 {count} 行一起改")
        self.same_label.setVisible(bool(self._label and count))
        same = not self.same_label.isHidden() and self.same_label.isChecked()
        affected = self._affected_count(self.whole.isChecked(), same, self.fill_down.isChecked()) \
            if self._affected_count is not None else 1
        self.count_line.setText(f"这次会改 {affected} 组")

    def choose(self, cid):
        self.chosen.emit(cid, self.whole.isChecked(),
                         not self.same_label.isHidden() and self.same_label.isChecked(), self.fill_down.isChecked())
        self.close()


class SelfCheckDialog(QDialog):
    def __init__(self, parent, session):
        super().__init__(parent, Qt.WindowType.Tool)
        self.help_anchor = "people"
        self.setModal(True)
        self.setWindowTitle("自称检查")
        self.resize(1000, 600)
        self.session = session
        layout = QVBoxLayout(self)
        self.message = QLabel()
        layout.addWidget(self.message)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["不一致的自称", "操作"])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setColumnWidth(0, 450)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self.jump_row)
        layout.addWidget(self.table, 2)
        layout.addWidget(QLabel("说话人不明"))
        self.unknown = QListWidget()
        self.unknown.itemDoubleClicked.connect(lambda item: self.jump(item.data(Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.unknown, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        app_help.dialog_buttons(self, buttons)
        session.changed.connect(self.refresh)
        self.finished.connect(self.disconnect_session)
        self.refresh()

    def disconnect_session(self, _result):
        self.session.changed.disconnect(self.refresh)

    def describe(self, item):
        from .group_table import fmt_time
        member = self.session.doc.members[item["member"]]
        speaker = member.speaker or {}
        name = self.session.editor.character_book.display_name(speaker.get("character"), member.lang,
                                                               speaker.get("label", ""), "short")
        return f"{fmt_time(item['time'])}　{name}：{visible_text(member)}"

    @Slot(object)
    def refresh(self, _touched=None):
        with self.session.lock:
            result = check_self_references(self.session.doc, self.session.editor.character_book,
                                          self.session.episode.id,
                                          getattr(self.session.episode, "data", {}).get("season"))
        self.rows = result["mismatches"]
        self.table.setRowCount(0)
        for i, item in enumerate(self.rows):
            self.table.insertRow(i)
            text = self.describe(item) + f"　{item['surface']}({item['current'] or '无注音'}) → 应为 {item['expected']}"
            cell = QTableWidgetItem(text)
            cell.setData(Qt.ItemDataRole.UserRole, item)
            cell.setToolTip(text)
            self.table.setItem(i, 0, cell)
            actions = QWidget()
            row = QVBoxLayout(actions)
            row.setContentsMargins(4, 2, 4, 2)
            buttons = QHBoxLayout()
            row.addLayout(buttons)
            for label, callback in (("试听", lambda _=False, it=item: self.jump(it, play=True)),
                                    (f"改成 {item['expected']}", lambda _=False, it=item: self.decide(it)),
                                    ("只保留这一句", lambda _=False, it=item: self.decide(it, keep=True, line=True)),
                                    ("这一集都这样读", lambda _=False, it=item: self.decide(it, keep=True, exception=True))):
                button = QPushButton(label)
                button.clicked.connect(callback)
                buttons.addWidget(button)
            self.table.setCellWidget(i, 1, actions)
            self.table.setRowHeight(i, actions.sizeHint().height())
        self.unknown.clear()
        for item in result["unknown_speakers"]:
            row = QListWidgetItem(self.describe(item), self.unknown)
            row.setData(Qt.ItemDataRole.UserRole, item)
        self.message.setText("自称都和人物表一致。" if not self.rows and not self.unknown.count() else "")

    def jump_row(self, row, _column):
        self.jump(self.rows[row])

    def jump(self, item, *, play=False):
        if self.parent().session is not self.session:
            return
        self.parent().select_group(item["group"], seek=not play)
        if play:
            self.parent().play_group()
        else:
            self.parent().player.pause()

    def decide(self, item, *, keep=False, exception=False, line=False):
        def apply():
            with self.session.lock:
                if keep:
                    self.session.editor.keep_self_reference(item, add_exception=exception, line_exception=line)
                else:
                    self.session.editor.fix_self_reference(item)
        self.parent()._edit(apply)
        self.refresh()


def visible_text(member):
    from subkit.core.readings import visible_map
    return visible_map(member)[0].text


def candidate_names(candidate, languages):
    """Prefill a suggestion without assigning uncertain surname boundaries."""
    lang = candidate["lang"]
    text, reading = candidate["text"], kata_to_hira(candidate.get("reading") or "")
    form = NameForm(given=text, given_reading=reading).with_defaults(lang)
    if candidate.get("kind") == "family":
        form = NameForm(family=text, family_reading=reading).with_defaults(lang)
    elif lang == "ja" and candidate.get("kind") == "person":
        from sudachipy import SplitMode
        from subkit.reading.engine import Analyzer

        parts = [m for m in Analyzer.get().tokenize(text, mode=SplitMode.A) if m.surface.strip()]
        if (len(parts) == 2 and parts[0].pos[:4] == ("名詞", "固有名詞", "人名", "姓")
                and parts[1].pos[:4] == ("名詞", "固有名詞", "人名", "名")
                and "".join(p.surface for p in parts) == text.replace(" ", "")):
            form = NameForm(parts[0].surface, parts[1].surface,
                            kata_to_hira(parts[0].reading), kata_to_hira(parts[1].reading)).with_defaults(lang)
    names = {lang: form}
    translation = languages.get("translation")
    if candidate.get("suggested_translation") and translation and translation != lang:
        names[translation] = NameForm(given=candidate["suggested_translation"]).with_defaults(translation)
    return names


class NewCharacterDialog(QDialog):
    """Small creation form; the full character details editor belongs to D2b."""

    def __init__(self, parent, languages, names=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.help_anchor = "people"
        self.setModal(True)
        self.setWindowTitle("新建人物")
        self._names = copy.deepcopy(names or {})
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)
        self.fields = {}
        codes = list(dict.fromkeys([*languages.values(), *self._names]))
        for lang in codes:
            original = self._names.get(lang, NameForm().with_defaults(lang))
            name = QLineEdit(original.compose())
            reading = QLineEdit(" ".join(original.parts(readings=True)))
            form.addRow(language_name(lang), name)
            if get_language(lang).annotation_system != "none":
                form.addRow("读音", reading)
            self.fields[lang] = name, reading
            name.textChanged.connect(self._validate)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        app_help.dialog_buttons(self, self.buttons)
        self._validate()

    def _validate(self):
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            any(name.text().strip() for name, _ in self.fields.values()))

    def names(self):
        names = {}
        for lang, (name, reading) in self.fields.items():
            text, rd = name.text().strip(), reading.text().strip()
            if not text:
                continue
            original = self._names.get(lang)
            if original and text == original.compose() and rd == " ".join(original.parts(readings=True)):
                names[lang] = copy.deepcopy(original)
            else:
                names[lang] = NameForm(given=text, given_reading=rd).with_defaults(lang)
        return names


def _accept_buttons(dialog, layout, accept):
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
    buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
    buttons.accepted.connect(accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    app_help.dialog_buttons(dialog, buttons)
    return buttons


class NameEditor(QGroupBox):
    """Real name only; the export Name choice is placed in a separate group."""

    def __init__(self, lang, original):
        super().__init__(language_name(lang))
        self.lang, self.original = lang, copy.deepcopy(original)
        layout = QFormLayout(self)
        self.fields = {}
        system = get_language(lang).annotation_system
        for key, label in (("family", "姓"), ("given", "名"),
                           ("family_reading", "姓的读音"), ("given_reading", "名的读音")):
            field = QLineEdit(getattr(original, key))
            self.fields[key] = field
            if not key.endswith("_reading") or system != "none":
                layout.addRow(label, field)
                if key.endswith("_reading"):
                    field.setPlaceholderText("假名" if system == "ruby" else "拼音")
            else:
                field.setParent(self)
                field.hide()
        self.order = QComboBox()
        for label, value in (("姓在前", "family_first"), ("名在前", "given_first")):
            self.order.addItem(label, value)
        self.order.setCurrentIndex(self.order.findData(original.order))
        layout.addRow("顺序", self.order)
        self.separator = QComboBox()
        for label, value in (("无", ""), ("空格", " "), ("・", "・"), ("·", "·")):
            self.separator.addItem(label, value)
        if self.separator.findData(original.separator) < 0:
            self.separator.addItem(original.separator, original.separator)
        self.separator.setCurrentIndex(self.separator.findData(original.separator))
        layout.addRow("连接符", self.separator)
        self.preview = QLabel()
        layout.addRow("全名", self.preview)
        self.short_box = QGroupBox(language_name(lang))
        short_layout = QFormLayout(self.short_box)
        self.short = QComboBox()
        for label, value in (("全名", "full"), ("姓", "family"), ("名", "given"), ("自己写", "custom")):
            self.short.addItem(label, value)
        self.short.setCurrentIndex(self.short.findData(original.short))
        self.custom = QLineEdit(original.short_custom)
        self.short_preview = QLabel()
        short_layout.addRow("常用称呼", self.short)
        short_layout.addRow("自己写", self.custom)
        short_layout.addRow("预览", self.short_preview)
        for field in self.fields.values():
            field.textChanged.connect(self.update_preview)
        self.custom.textChanged.connect(self.update_preview)
        for combo in (self.order, self.separator, self.short):
            combo.currentIndexChanged.connect(self.update_preview)
        self.update_preview()

    def value(self, validate=False):
        form = copy.deepcopy(self.original)
        for key, field in self.fields.items():
            setattr(form, key, field.text().strip())
        form.order, form.separator = self.order.currentData(), self.separator.currentData()
        form.short, form.short_custom = self.short.currentData(), self.custom.text()
        if validate:
            system = get_language(self.lang).annotation_system
            for part in ("family", "given"):
                reading = getattr(form, part + "_reading")
                if not reading or system == "none":
                    continue
                if system == "ruby":
                    if not re.fullmatch(r"[ぁ-ゖァ-ヺー\s]+", reading):
                        raise ValueError("假名无效：请填写假名读音。")
                    if not getattr(form, part):
                        raise ValueError("请先填写读音对应的姓或名。")
                    setattr(form, part + "_reading", kata_to_hira(reading))
                else:
                    from subkit.core.pinyin_notes import normalize_rule
                    setattr(form, part + "_reading", normalize_rule(getattr(form, part), reading)[1])
        return form

    def update_preview(self, *_args):
        form = self.value()
        self.preview.setText(form.compose())
        self.short_preview.setText(form.compose("short"))
        self.custom.setEnabled(form.short == "custom")


class ScopeEditor(QWidget):
    def __init__(self, scope, hint=None):
        super().__init__()
        self.original = copy.deepcopy(scope)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if "line" in scope:
            text = (f"第 {hint['episode']} 第 {hint['number']} 句：{hint['text']}" if hint else "某一句（找不到了）")
            layout.addWidget(QLabel(text))
            return
        self.kind, self.value = QComboBox(), QLineEdit()
        for label, key in (("全部", ""), ("季", "season"), ("集", "episodes")):
            self.kind.addItem(label, key)
        key = next((k for k in ("line", "episodes", "season") if k in scope), "")
        value = scope.get(key, "")
        self.kind.setCurrentIndex(self.kind.findData(key))
        self.value.setText("、".join(map(str, value)) if isinstance(value, list) else str(value))
        self.initial = (key, self.value.text())
        self.value.setPlaceholderText("季 / 集编号（多个用逗号分隔）")
        layout.addWidget(self.kind)
        layout.addWidget(self.value)
        self.kind.currentIndexChanged.connect(self.update_enabled)
        self.update_enabled()

    def update_enabled(self, *_args):
        self.value.setEnabled(bool(self.kind.currentData()))

    def scope(self):
        if "line" in self.original:
            return copy.deepcopy(self.original)
        key, value = self.kind.currentData(), self.value.text().strip()
        if (key, self.value.text()) == self.initial:
            return copy.deepcopy(self.original)
        if not key:
            return {}
        if not value:
            raise ValueError("请填写自称的季或集范围。")
        return {key: [x.strip() for x in re.split(r"[,，、]", value) if x.strip()] if key == "episodes" else value}


class CharacterDetailsDialog(QDialog):
    def __init__(self, parent, character, languages, display=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.help_anchor = "people"
        self.setModal(True)
        self.setWindowTitle("人物：" + (display or character_display_name(character, languages.get("translation", "ja"))))
        self.character, self.languages = copy.deepcopy(character), languages
        self.resize(660, 720)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        content = QVBoxLayout(body)
        scroll.setWidget(body)
        layout.addWidget(scroll)
        self.real_group = QGroupBox("真实姓名（用于读音）")
        self.real_layout = QVBoxLayout(self.real_group)
        self.name_group = QGroupBox("字幕名字栏")
        self.name_layout = QVBoxLayout(self.name_group)
        content.addWidget(self.real_group)
        content.addWidget(self.name_group)
        self.names = {}
        for lang in dict.fromkeys([*languages.values(), *character.names]):
            self.add_language(lang, character.names.get(lang, NameForm().with_defaults(lang)))
        add = QPushButton("加一种语言")
        add.clicked.connect(self.choose_language)
        self.real_layout.addWidget(add)
        content.addWidget(QLabel("别名（一行一个）"))
        self.aliases = QPlainTextEdit("\n".join(character.aliases))
        self.aliases.setMaximumHeight(85)
        content.addWidget(self.aliases)
        content.addWidget(QLabel("自称"))
        self.refs = QTableWidget(0, 3)
        self.refs.setHorizontalHeaderLabels(["写法", "读法", "范围"])
        self.refs.horizontalHeader().setStretchLastSection(True)
        self.refs.setMinimumHeight(160)
        self.refs.setColumnWidth(0, 85)
        self.refs.setColumnWidth(1, 115)
        content.addWidget(self.refs)
        for ref in character.self_refs:
            self.add_ref(ref)
        row = QHBoxLayout()
        content.addLayout(row)
        for label, action in (("添加自称", lambda: self.add_ref({})), ("删除自称", self.remove_ref)):
            button = QPushButton(label)
            button.clicked.connect(action)
            row.addWidget(button)
        persona = QGroupBox("人设")
        form = QFormLayout(persona)
        self.persona = {}
        labels = ("自称（说明）", "对别人的称呼", "口癖", "语气 / 敬语", "备注（日语）",
                  "备注（" + language_name(languages.get("translation", "zh-Hant")) + "）")
        for key, label in zip(PERSONA_FIELDS, labels):
            field = QPlainTextEdit(character.persona.get(key, ""))
            field.setMaximumHeight(65)
            self.persona[key] = field
            form.addRow(label, field)
        content.addWidget(persona)
        self.buttons = _accept_buttons(self, layout, self.accept)

    def add_language(self, lang, form=None):
        if lang in self.names:
            return
        editor = NameEditor(lang, form or NameForm().with_defaults(lang))
        self.names[lang] = editor
        self.real_layout.addWidget(editor)
        self.name_layout.addWidget(editor.short_box)

    def choose_language(self):
        codes = [code for code in LANGUAGES if code not in self.names]
        labels = [language_name(c) for c in codes]
        if labels:
            choice, ok = QInputDialog.getItem(self, "加一种语言", "语言", labels, 0, False)
            if ok:
                self.add_language(codes[labels.index(choice)])

    def add_ref(self, ref):
        row = self.refs.rowCount()
        self.refs.insertRow(row)
        for col, key in enumerate(("surface", "reading")):
            item = QTableWidgetItem(ref.get(key, ""))
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, copy.deepcopy(ref))
            self.refs.setItem(row, col, item)
        hint = ref.get("display_hint")
        parent = self.parent()
        while parent is not None and not getattr(parent, "session", None):
            parent = parent.parent()
        session = getattr(parent, "session", None)
        if hint and session and hint.get("episode") == session.doc.episode:
            if ref.get("scope", {}).get("line") not in session.doc.members:
                hint = None
        self.refs.setCellWidget(row, 2, ScopeEditor(ref.get("scope") or {}, hint))

    def remove_ref(self):
        if self.refs.currentRow() >= 0:
            self.refs.removeRow(self.refs.currentRow())

    def changes(self):
        names = {}
        for lang, editor in self.names.items():
            form = editor.value(validate=True)
            if form.compose():
                names[lang] = form
        if not names:
            raise ValueError("请至少填写一种语言的姓名。")
        refs = []
        for row in range(self.refs.rowCount()):
            surface, reading = (self.refs.item(row, col).text().strip() for col in (0, 1))
            if not surface and not reading:
                continue
            if not surface or not re.fullmatch(r"[ぁ-ゖァ-ヺー\s]+", reading):
                raise ValueError("自称须填写写法和有效的假名读法。")
            ref = self.refs.item(row, 0).data(Qt.ItemDataRole.UserRole) or {}
            refs.append({**ref, "surface": surface, "reading": kata_to_hira(reading),
                         "scope": self.refs.cellWidget(row, 2).scope()})
        return {"names": names, "aliases": list(dict.fromkeys(x.strip() for x in self.aliases.toPlainText().splitlines() if x.strip())),
                "self_refs": refs, "persona": {**self.character.persona,
                    **{k: field.toPlainText() for k, field in self.persona.items()}}}

    def accept(self):
        try:
            self.changes()
        except ValueError as exc:
            QMessageBox.warning(self, "人物", str(exc))
            return
        super().accept()


class BangumiFetchThread(QThread):
    progress = Signal(int, int)

    def __init__(self, text, parent):
        super().__init__(parent)
        self.text, self.cancelled = text, threading.Event()
        self.result, self.error = None, None

    def run(self):
        try:
            self.result = bangumi.fetch_subject(self.text, cancelled=self.cancelled.is_set,
                                                progress=self.progress.emit)
        except bangumi.BangumiCancelled:
            pass
        except Exception as exc:
            self.error = str(exc)


class BangumiFetchDialog(QDialog):
    """Remain modal until the owned request thread has actually stopped."""

    def __init__(self, parent, text):
        super().__init__(parent, Qt.WindowType.Tool)
        self.help_anchor = "people"
        self.setModal(True)
        self.setWindowTitle("从 Bangumi 导入…")
        layout = QVBoxLayout(self)
        self.label = QLabel("正在从 Bangumi 读取…")
        layout.addWidget(self.label)
        self.cancel = QPushButton("取消")
        self.cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(self.cancel)
        app_help.dialog_buttons(self, buttons)
        layout.addLayout(buttons)
        self.worker = BangumiFetchThread(text, self)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.finish)

    def update_progress(self, current, total):
        self.label.setText(f"正在从 Bangumi 读取人物：{current + 1}/{total}")

    def reject(self):
        self.worker.cancelled.set()
        self.cancel.setEnabled(False)
        self.label.setText("正在取消，等待当前请求结束…")

    def closeEvent(self, event):
        self.reject()
        event.ignore()

    def finish(self):
        self.done(QDialog.DialogCode.Rejected if self.worker.cancelled.is_set() else QDialog.DialogCode.Accepted)

    def fetch(self):
        window = self.parentWidget().window()
        status = window.statusBar() if hasattr(window, "statusBar") else None
        old = status.currentMessage() if status else ""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if status:
                status.showMessage("正在从 Bangumi 读取…")
            self.worker.start()
            self.exec()
            self.worker.wait()
            if self.worker.cancelled.is_set():
                return None
            if self.worker.error:
                error = self.worker.error
                raise ValueError(error if error.startswith("无法从 Bangumi 读取：") else "无法从 Bangumi 读取：" + error)
            return self.worker.result
        finally:
            self.worker.cancelled.set()
            self.worker.wait()
            QApplication.restoreOverrideCursor()
            if status:
                status.showMessage(old)


class BangumiImportDialog(QDialog):
    def __init__(self, parent, data, book, languages, display):
        super().__init__(parent, Qt.WindowType.Tool)
        self.help_anchor = "people"
        self.setModal(True)
        self.data, self.book, self.languages, self.display = data, book, languages, display
        subject = data[0]
        self.setWindowTitle("从 Bangumi 导入：" + (subject.get("name_cn") or subject.get("name", "")))
        self.resize(920, 500)
        layout = QVBoxLayout(self)
        self.traditional = QCheckBox("中文名转为繁体")
        self.traditional.setChecked(normalize_language_code(languages.get("translation", "")) == "zh-Hant")
        layout.addWidget(self.traditional)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["", "日语名", "中文名", "罗马字", "角色", "声优", "对应"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        edit = QPushButton("人物详情…")
        edit.clicked.connect(self.edit_names)
        layout.addWidget(edit)
        self.buttons = _accept_buttons(self, layout, self.accept)
        self.traditional.toggled.connect(self.rebuild)
        self.rebuild()
        self.table.itemChanged.connect(self.validate_check)

    def validate_check(self, item):
        if item.column() == 0 and item.checkState() == Qt.CheckState.Checked:
            if not bangumi.has_displayable_name(self.rows[item.row()]):
                item.setCheckState(Qt.CheckState.Unchecked)
                QMessageBox.warning(self, "从 Bangumi 导入", "这一行没有名字，请先填名字")

    def flag_name(self, i):
        missing = not bangumi.has_displayable_name(self.rows[i])
        for col in (1, 2):
            self.table.item(i, col).setForeground(QColor("#d93025") if missing else self.palette().text())
        self.table.item(i, 1).setText("缺名字" if missing else self.rows[i]["ja_name"])
        if missing:
            self.table.item(i, 0).setCheckState(Qt.CheckState.Unchecked)

    def rebuild(self, *_args):
        previous = self.selections() if hasattr(self, "rows") else None
        old_rows = getattr(self, "rows", [])
        rows = bangumi.build_candidates(*self.data, self.book, self.languages, self.traditional.isChecked())
        # Retain edited spellings/splits and choices when toggling conversion.
        if previous is not None:
            for i, row in enumerate(rows):
                if old_rows[i].get("edited"):
                    row["names"] = copy.deepcopy(old_rows[i]["names"])
                    row["edited"] = True
                    for lang, form in row["names"].items():
                        if lang.startswith("zh-"):
                            from opencc import OpenCC
                            converter = OpenCC("s2tw" if self.traditional.isChecked() else "t2s")
                            form.family, form.given = converter.convert(form.family), converter.convert(form.given)
                    row["ja_name"] = row["names"]["ja"].compose()
                    row["zh_name"] = next((f.compose() for l, f in row["names"].items() if l.startswith("zh-")), "")
        self.rows = rows
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked if (previous[i][0] if previous else row["checked"]) else Qt.CheckState.Unchecked)
            self.table.setItem(i, 0, check)
            for col, key in enumerate(("ja_name", "zh_name", "romaji", "relation", "cv"), 1):
                item = QTableWidgetItem(row[key])
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, col, item)
            target = QComboBox()
            target.addItem("新建", None)
            for ch in self.book.characters:
                target.addItem("更新：" + self.display(ch), ch.id)
            target.addItem("不导入", "skip")
            target.setCurrentIndex(target.findData(previous[i][1] if previous else row["match"]))
            self.table.setCellWidget(i, 6, target)
            self.flag_name(i)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()

    def selections(self):
        return [(self.table.item(i, 0).checkState() == Qt.CheckState.Checked,
                 self.table.cellWidget(i, 6).currentData()) for i in range(len(self.rows))]

    def chosen(self):
        chosen = []
        for i, (checked, target) in enumerate(self.selections()):
            if checked and target != "skip":
                self.rows[i]["match"] = target
                chosen.append(i)
        return chosen

    def edit_names(self):
        i = self.table.currentRow()
        if i < 0:
            return
        row = self.rows[i]
        dialog = CharacterDetailsDialog(self, Character("chr_preview", row["names"]), self.languages, row["ja_name"])
        if dialog.exec() == QDialog.DialogCode.Accepted:
            row["names"] = dialog.changes()["names"]
            row["edited"] = True
            row["ja_name"] = row["names"].get("ja", NameForm()).compose()
            row["zh_name"] = next((f.compose() for l, f in row["names"].items() if l.startswith("zh-")), "")
            self.table.item(i, 1).setText(row["ja_name"])
            self.table.item(i, 2).setText(row["zh_name"])
            self.flag_name(i)


class CharacterChoiceDialog(QDialog):
    def __init__(self, parent, title, characters, display, exclude=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.help_anchor = "people"
        self.setModal(True)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        self.choice = QComboBox()
        for ch in characters:
            if ch.id != exclude:
                self.choice.addItem(display(ch), ch.id)
        layout.addWidget(self.choice)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(self.choice.count() > 0)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        app_help.dialog_buttons(self, buttons)


class PeopleModel(QAbstractTableModel):
    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.characters, self.languages = [], []

    def refresh(self):
        self.beginResetModel()
        self.characters = list(self.panel.book.characters) if self.panel.book else []
        self.languages = list(dict.fromkeys(self.panel.project.languages.values())) if self.panel.project else []
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self.characters)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self.languages) + 3

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return [*(language_name(lang) for lang in self.languages), "别名", "自称", "备注"][section]
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        ch = self.characters[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return ch.id
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return None
        col = index.column()
        if col < len(self.languages):
            lang = self.languages[col]
            form = ch.name_form(lang)
            text = character_display_name(ch, lang)
            if form and get_language(lang).annotation_system != "none":
                rd = " ".join(form.parts(readings=True))
                if rd:
                    text += f"（{rd}）"
            return text
        return ["、".join(ch.aliases), "、".join(f"{r['surface']}（{r['reading']}）" for r in ch.self_refs),
                ch.persona.get("note_ja", "")][col - len(self.languages)]

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and index.column() in (len(self.languages), len(self.languages) + 2):
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not self.flags(index) & Qt.ItemFlag.ItemIsEditable:
            return False
        ch = self.characters[index.row()]
        if index.column() == len(self.languages):
            fields = {"aliases": list(dict.fromkeys(x.strip() for x in re.split(r"[、,，;；\n]", str(value)) if x.strip()))}
        else:
            fields = {"persona": {**ch.persona, "note_ja": str(value)}}
        if not self.panel.change(lambda book: book.update_character(ch.id, **fields)):
            return False
        self.dataChanged.emit(index, index, [role, Qt.ItemDataRole.DisplayRole])
        return True


class PeoplePanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("人物", parent)
        self.help_anchor = "people"
        self.setObjectName("dock_people")
        self.project = self.session = self.book = None
        self._name_cache = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(120)
        self._refresh_timer.timeout.connect(self.refresh)
        body = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(120)
        scroll.setWidget(body)
        self.setWidget(scroll)
        layout = QVBoxLayout(body)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索名字、别名")
        layout.addWidget(self.search)
        self.model = PeopleModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed | QAbstractItemView.EditTrigger.SelectedClicked)
        self.table.doubleClicked.connect(self.details)
        layout.addWidget(self.table, 2)
        self.buttons = self._buttons(layout, [("新建人物", self.new_character), ("删除人物", self.delete_character),
                                              ("合并到…", self.merge_character), ("人物详情…", self.details)])
        self.import_button = QPushButton("从 Bangumi 导入…")
        self.import_button.clicked.connect(self.import_bangumi)
        layout.addWidget(self.import_button)
        self.pending_label = QLabel()
        layout.addWidget(self.pending_label)
        self.pending = self._candidate_table(["标签", "次数", "例句", "建议"], 2)
        layout.addWidget(self.pending, 1)
        self.pending_buttons = self._buttons(layout, [("新建人物", self.new_pending), ("归到人物…", self.assign_pending),
                                                      ("标为群众", lambda: self.mark_pending(True)),
                                                      ("不是人物", lambda: self.mark_pending(False))])
        self.names_label = QLabel()
        layout.addWidget(self.names_label)
        self.found_names = self._candidate_table(["名字", "读音", "次数", "例句"], 3)
        layout.addWidget(self.found_names, 1)
        self.name_buttons = self._buttons(layout, [("新建人物", self.new_found), ("加到人物…", self.assign_found),
                                                   ("不是人名", self.reject_found)])
        self.table.selectionModel().selectionChanged.connect(self._enable_actions)
        self.pending.itemSelectionChanged.connect(self._enable_actions)
        self.found_names.itemSelectionChanged.connect(self._enable_actions)
        for table in (self.table, self.pending, self.found_names):
            table.setMinimumHeight(table.horizontalHeader().height() + 4 * table.verticalHeader().defaultSectionSize())
        self.refresh()

    def _buttons(self, layout, actions):
        row = QHBoxLayout()
        layout.addLayout(row)
        buttons = {}
        for text, action in actions:
            button = QPushButton(text)
            button.clicked.connect(action)
            row.addWidget(button)
            buttons[text] = button
        return buttons

    def set_context(self, project, session=None):
        if self.session is not None:
            self.session.changed.disconnect(self.schedule_refresh)
        self._refresh_timer.stop()
        self.project, self.session = project, session
        self._name_cache = None
        self.book = session.editor.character_book if session else project.character_book() if project else None
        if session:
            session.changed.connect(self.schedule_refresh)
        self.refresh()

    @Slot(object)
    def schedule_refresh(self, _touched=None):
        self._refresh_timer.start()

    def refresh(self):
        self._refresh_timer.stop()
        selected = self.selected_id()
        self.model.refresh()
        for row in range(self.proxy.rowCount()):
            if self.proxy.index(row, 0).data(Qt.ItemDataRole.UserRole) == selected:
                self.table.selectRow(row)
                break
        self.pending.setRowCount(0)
        eligible = None
        if self.session and self.book:
            from subkit.core.people import needs_speaker
            eligible = {self.book.speaker_label(m) for g in self.session.doc.groups
                        for m in self.session.doc.members_of(g) if needs_speaker(g, m)}
        for item in self.book.pending if self.book else []:
            if eligible is not None and item["label"] not in eligible and item.get("episodes", []) == [self.session.doc.episode]:
                continue
            example = " / ".join(item.get("examples", []))
            row = self._candidate_row(self.pending, [item["label"], str(item["count"]), example, ""], item)
            cid = suggest_character(self.book, item["label"])
            if cid:
                name = self.display(self.book.character(cid))
                button = QPushButton(f"归到 {name}")
                button.clicked.connect(lambda _=False, label=item["label"], target=cid:
                                       self.assign_suggested(label, target))
                self.pending.setCellWidget(row, 3, button)
                self.pending.setRowHeight(row, max(self.pending.rowHeight(row), button.sizeHint().height() + 8))
        self.pending_label.setText(f"待确认的标签（{self.pending.rowCount()}）")
        self.found_names.setRowCount(0)
        if self.session:
            from subkit.core.name_finder import _excluded
            from subkit.core.people import CharacterBook
            from subkit.core.cc import normalize_speaker

            doc = self.session.doc
            key = tuple((g.id, tuple((m.id, m.lang, m.role, m.source_kind, m.text)
                                    for m in doc.members_of(g))) for g in doc.groups)
            if self._name_cache is None or self._name_cache[0] != key:
                unfiltered = CharacterBook(self.book.path)
                self._name_cache = (key, find_names(doc, self.session.episode.sources, unfiltered, self.project.languages))
            excluded = _excluded(self.book)
            candidates = [item for item in self._name_cache[1] if normalize_speaker(item["text"]) not in excluded]
            for item in sorted(candidates, key=lambda item: (self._weak_candidate(item), -item["count"])):
                reading = kata_to_hira(item.get("reading") or "")
                example = " / ".join(e["text"] for e in item["examples"])
                row = self._candidate_row(self.found_names, [item["text"], reading, str(item["count"]), example], item)
                if self._weak_candidate(item):
                    for col in range(4):
                        cell = self.found_names.item(row, col)
                        cell.setForeground(QColor("#8b949e"))
                        cell.setToolTip("只出现一次的片假名，可能不是人名" + ("\n" + example if col == 3 else ""))
        self.names_label.setText(f"台词里的人名（{self.found_names.rowCount()}）")
        self._enable_actions()

    @staticmethod
    def _weak_candidate(item):
        return item["count"] == 1 and bool(re.fullmatch(r"[ァ-ヺー]+", item["text"]))

    @staticmethod
    def _candidate_table(headers, example_column):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setWordWrap(False)
        table.setTextElideMode(Qt.TextElideMode.ElideRight)
        table.verticalHeader().hide()
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(example_column, QHeaderView.ResizeMode.Stretch)
        return table

    @staticmethod
    def _candidate_row(table, texts, data):
        row = table.rowCount()
        table.insertRow(row)
        for col, text in enumerate(texts):
            cell = QTableWidgetItem(text.replace("\n", " ").replace("\r", " ").replace("\\N", " "))
            cell.setToolTip(text)
            if col == 0:
                cell.setData(Qt.ItemDataRole.UserRole, copy.deepcopy(data))
            table.setItem(row, col, cell)
        table.setRowHeight(row, max(28, table.fontMetrics().height() + 8))
        return row

    @staticmethod
    def _selected_item(table):
        return table.item(table.currentRow(), 0) if table.selectionModel().hasSelection() else None

    def selected_id(self):
        rows = self.table.selectionModel().selectedRows()
        return rows[0].data(Qt.ItemDataRole.UserRole) if rows else None

    def display(self, ch):
        from subkit.core.display import DisplayRules

        lang = self.project.languages.get("translation", "ja")
        name = character_display_name(ch, lang)
        rules = DisplayRules.from_settings(self.project.settings())
        if rules.direction == "off" or not get_language(lang).code.startswith("zh"):
            return name
        target = "zh-Hans" if rules.direction == "simplified" else "zh-Hant"
        form = ch.names.get(target) if ch is not None else None
        if form is not None and form.compose():
            return form.compose()
        return rules.convert(name) if ch is not None and ch.name_form(lang) is not None else name

    def _enable_actions(self, *_args):
        self.import_button.setEnabled(bool(self.book))
        for text, button in self.buttons.items():
            button.setEnabled(bool(self.book) if text == "新建人物" else bool(self.selected_id()))
        for buttons, widget in ((self.pending_buttons, self.pending), (self.name_buttons, self.found_names)):
            for button in buttons.values():
                button.setEnabled(self._selected_item(widget) is not None)
        self.buttons["合并到…"].setEnabled(bool(self.selected_id() and len(self.book.characters) > 1))

    def change(self, action):
        try:
            if self.session:
                with self.session.lock:
                    self.session.editor.edit_characters(action)
            else:
                state = self.book.snapshot()
                try:
                    action(self.book)
                except Exception:
                    self.book.restore(state)
                    raise
            self.schedule_refresh()
            return True
        except Exception as exc:
            QMessageBox.warning(self, "人物", str(exc))
            return False

    def new_character(self, _checked=False, *, names=None, label=None):
        if not self.book:
            return
        dialog = NewCharacterDialog(self, self.project.languages, names)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.change(lambda book: book.add_character(dialog.names(), aliases=[label] if label else []))

    def details(self, *_args):
        cid = self.selected_id()
        if cid:
            ch = self.book.character(cid)
            dialog = CharacterDetailsDialog(self, ch, self.project.languages, self.display(ch))
            try:
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    fields = dialog.changes()
                    self.change(lambda book: book.update_character(cid, **fields))
            finally:
                dialog.deleteLater()

    def import_bangumi(self):
        if not self.book:
            return
        text, ok = QInputDialog.getText(self, "从 Bangumi 导入…", "作品链接或编号")
        if not ok:
            return
        try:
            progress = BangumiFetchDialog(self, text)
            try:
                data = progress.fetch()
            finally:
                progress.deleteLater()
            if data is None:
                return
            dialog = BangumiImportDialog(self, data, self.book, self.project.languages, self.display)
            try:
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    chosen = dialog.chosen()
                    if self.session:
                        with self.session.lock:
                            bangumi.apply_import(self.book, dialog.rows, chosen, editor=self.session.editor)
                    else:
                        bangumi.apply_import(self.book, dialog.rows, chosen)
                    self.schedule_refresh()
            finally:
                dialog.deleteLater()
        except Exception as exc:
            QMessageBox.warning(self, "从 Bangumi 导入…", str(exc))

    def delete_character(self):
        cid = self.selected_id()
        if not cid:
            return
        count = sum(bool(m.speaker and m.speaker.get("character") == cid)
                    for m in self.session.doc.members.values()) if self.session else 0
        text = f"删除人物「{self.display(self.book.character(cid))}」？{count} 行的说话人会变回原来的标签。"
        if QMessageBox.question(self, "删除人物", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            if self.session:
                self._special(lambda: self.book.delete_character(cid, editor=self.session.editor))
            else:
                self.change(lambda book: book.delete_character(cid))

    def _special(self, action):
        try:
            with self.session.lock:
                action()
            self.schedule_refresh()
        except Exception as exc:
            QMessageBox.warning(self, "人物", str(exc))

    def choose(self, title, exclude=None):
        dialog = CharacterChoiceDialog(self, title, self.book.characters, self.display, exclude)
        return dialog.choice.currentData() if dialog.exec() == QDialog.DialogCode.Accepted else None

    def merge_character(self):
        source = self.selected_id()
        if not source:
            return
        target = self.choose("合并到…", source)
        if target and QMessageBox.question(self, "合并到…",
                f"把「{self.display(self.book.character(source))}」合并到「{self.display(self.book.character(target))}」？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            if self.session:
                self._special(lambda: self.session.editor.merge_characters(source, target))
            else:
                self.change(lambda book: book.merge_characters(source, target))

    def new_pending(self):
        item = self._selected_item(self.pending)
        if item:
            label = item.data(Qt.ItemDataRole.UserRole)["label"]
            lang = self.project.languages.get("original", "ja")
            self.new_character(names={lang: NameForm(given=label).with_defaults(lang)}, label=label)

    def assign_pending(self):
        item = self._selected_item(self.pending)
        if item:
            label = item.data(Qt.ItemDataRole.UserRole)["label"]
            cid = self.choose("归到人物…")
            if cid:
                self.change(lambda book: book.resolve_pending(label, cid))

    def assign_suggested(self, label, cid):
        if self.book and suggest_character(self.book, label) == cid:
            self.change(lambda book: book.resolve_pending(label, cid))

    def mark_pending(self, crowd):
        item = self._selected_item(self.pending)
        if item:
            label = item.data(Qt.ItemDataRole.UserRole)["label"]
            self.change(lambda book: book.mark_label(label, crowd=crowd))

    def new_found(self):
        item = self._selected_item(self.found_names)
        if item:
            self.new_character(names=candidate_names(item.data(Qt.ItemDataRole.UserRole), self.project.languages))

    def assign_found(self):
        item = self._selected_item(self.found_names)
        if not item:
            return
        candidate = item.data(Qt.ItemDataRole.UserRole)
        cid = self.choose("加到人物…")
        if cid:
            def apply(book):
                ch = book.character(cid)
                lang = candidate["lang"]
                if not ch.name_form(lang):
                    names = {**ch.names, lang: candidate_names(candidate, self.project.languages)[lang]}
                    book.update_character(cid, names=names)
                else:
                    book.update_character(cid, aliases=list(dict.fromkeys([*ch.aliases, candidate["text"]])))
            self.change(apply)

    def reject_found(self):
        item = self._selected_item(self.found_names)
        if item:
            text = item.data(Qt.ItemDataRole.UserRole)["text"]
            self.change(lambda book: book.mark_not_name(text))
