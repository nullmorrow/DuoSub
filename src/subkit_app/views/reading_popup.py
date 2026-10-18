"""Small popup to choose / type a reading (no IME needed: romaji is converted to kana).

With a ``scope`` (the member's text cut into words) the range can be changed: “◀ 并入前一个词 / 并入后一个词 ▶”
(Alt+← / Alt+→) turns 迷宮 + 都市 into one word 迷宮都市 whose reading is pre-filled from the parts
(めいきゅう + とし); choosing a reading then writes one annotation over the whole range.  “按自动分词拆开” gives a
word back to the analyzer's segmentation.  The reading applies here only, to the whole episode (an episode rule)
or to the whole project (the project dictionary) — the last two are previewed before they are applied.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (QButtonGroup, QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton,
                               QScrollArea, QVBoxLayout, QWidget)

from subkit.core.readings import placement_text
from subkit.reading.records import romaji_to_kana
from subkit.textutil import is_kanji

from .floating import FLAGS, OutsideClick, show_activated
from .placement import place_near

SCOPE_HERE, SCOPE_EPISODE, SCOPE_PROJECT = 0, 1, 2

ORIGIN_LABEL = {"exception": "本处例外", "dict": "项目词库", "character": "人物规则", "episode_rule": "本集规则",
                "manual": "人工", "cc_hint": "原料自带注音", "rule": "上下文规则", "analyzer": "自动（分词器）",
                "voice": "配音建议", "legacy": "旧工程"}


class ReadingPopup(QFrame):
    help_anchor = "readings"
    chosen = Signal(str, int)               # a reading for the same characters, scope
    chosenRange = Signal(int, int, str, int)   # one reading over a changed range: visible [a, b), reading, scope
    typedChosen = Signal(str, str)          # input box only: current surface, converted reading
    hideRequested = Signal()
    splitRequested = Signal()               # give the word back to the analyzer's segmentation
    wordSplitRequested = Signal(int, int, int)  # current range, preferred scope
    kanjiChosen = Signal(list, int)          # one reading (possibly empty) per kanji, requested scope
    bracketToText = Signal()                # the source's bracket was not a reading: back into the line

    def __init__(self, base: str, current: str, candidates: list[str], origin: str, status: str, parent=None, *,
                 scope: Optional[dict] = None, prefill: str = "", can_split: bool = False,
                 kind: str = "ruby", candidates_for: Optional[Callable[[int, int], list[str]]] = None,
                 kanji_splitter: Optional[Callable] = None) -> None:
        super().__init__(parent, FLAGS)            # not Qt.Popup: an input method must be able to compose here
        self.kind = kind
        self._composing = False
        self.kanji_splitter = kanji_splitter
        self._kanji_mode = False
        self._kanji_rows = []
        self._can_split = can_split
        if kind == "pinyin":
            self.help_anchor = "pinyin"
            self.setWindowTitle(f"拼音：{base}")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame{background:#ffffff;border:1px solid #d0d7de;border-radius:6px}"
                           "QPushButton{padding:4px 10px}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        self.origin, self.status, self.current = origin, status, current
        self.base = self.base0 = base
        self.scope = scope
        self.candidates_for = candidates_for
        self.range_cands: list[str] = []
        if scope is not None:
            self.a = self.a0 = int(scope["a"])
            self.b = self.b0 = int(scope["b"])
        self.title = QLabel()
        lay.addWidget(self.title)
        # candidates (they belong to the original characters; hidden while the range is changed)
        self.cands = [c for c in dict.fromkeys(([current] if current else []) + candidates) if c]
        self.cand_box = QWidget()
        row = QHBoxLayout(self.cand_box) if kind == "pinyin" else QGridLayout(self.cand_box)
        row.setContentsMargins(0, 0, 0, 0)
        for i, c in enumerate(self.cands[:9] if kind == "pinyin" else self.cands):
            label = c if kind == "pinyin" else placement_text(base, c)
            b = QPushButton(f"{i + 1}  {label}" if i < 9 else label)
            b.clicked.connect(lambda _=False, x=c: self._choose(x))
            if kind == "pinyin":
                row.addWidget(b)
            else:
                row.addWidget(b, i // 4, i % 4)
        if kind == "pinyin":
            row.addStretch(1)
        lay.addWidget(self.cand_box)
        self.range_cand_box = QWidget()
        range_row = QGridLayout(self.range_cand_box)
        range_row.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.range_cand_box)
        # range: merge with the neighbouring words
        if scope is not None:
            self.range_box = QWidget()
            rr = QHBoxLayout(self.range_box)
            rr.setContentsMargins(0, 0, 0, 0)
            self.btn_left = QPushButton("◀ 并入前一个词")
            self.btn_right = QPushButton("并入后一个词 ▶")
            self.btn_reset = QPushButton("还原范围")
            self.btn_left.setToolTip("Alt+←")
            self.btn_right.setToolTip("Alt+→（例如 迷宮 + 都市 → 迷宮都市，整体一个读音）")
            self.btn_left.clicked.connect(lambda: self._extend(-1))
            self.btn_right.clicked.connect(lambda: self._extend(+1))
            self.btn_reset.clicked.connect(self._reset_range)
            rr.addWidget(QLabel("范围："))
            for w in (self.btn_left, self.btn_right, self.btn_reset):
                rr.addWidget(w)
            rr.addStretch(1)
            lay.addWidget(self.range_box)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("输入假名或罗马字（watakushi → わたくし），回车确定")
        if kind == "pinyin":
            self.edit.setPlaceholderText("输入拼音（hang2、háng、lv4），回车确定")
        self.edit.installEventFilter(self)
        self.preview = QLabel("")
        self.preview.setStyleSheet("color:#0550ae")
        self.edit.textChanged.connect(self._preview)
        self.edit.returnPressed.connect(self._accept_input)
        lay.addWidget(self.edit)
        lay.addWidget(self.preview)
        self.kanji_box = QWidget()
        klay = QVBoxLayout(self.kanji_box)
        klay.setContentsMargins(0, 0, 0, 0)
        self.kanji_hint = QLabel()
        self.kanji_hint.setWordWrap(True)
        klay.addWidget(self.kanji_hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(180)
        self.kanji_fields = QWidget()
        self.kanji_grid = QGridLayout(self.kanji_fields)
        self.kanji_grid.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self.kanji_fields)
        klay.addWidget(scroll)
        self.kanji_scope_note = QLabel("按字拆开只改这一处，不写入本集规则或工程词库。")
        self.kanji_scope_note.setWordWrap(True)
        klay.addWidget(self.kanji_scope_note)
        self.kanji_ok = QPushButton("确定")
        self.kanji_ok.clicked.connect(self._accept_kanji)
        klay.addWidget(self.kanji_ok)
        lay.addWidget(self.kanji_box)
        self.kanji_box.hide()
        scope_row = QHBoxLayout()
        self.scope_group = QButtonGroup(self)
        for i, (label, tip) in enumerate((("只改这里", "只改这一处"),
                                          ("这一集都这样", "本集读音规则：这一集里这个词都这样注音（先预览，确认过的读音不动）"),
                                          ("整个工程都这样", "写进工程词库：每一集都这样注音（先预览本集，确认过的读音不动）"))):
            rb = QRadioButton(label)
            rb.setToolTip(tip.replace("读音", "拼音") if kind == "pinyin" else tip)
            rb.setChecked(i == 0)
            self.scope_group.addButton(rb, i)
            scope_row.addWidget(rb)
        scope_row.addStretch(1)
        lay.addLayout(scope_row)
        btns = QHBoxLayout()
        self.btn_hide = QPushButton("不标注")
        self.btn_hide.clicked.connect(self._hide)
        self.btn_split = QPushButton("按自动分词拆开")
        self.btn_split.setToolTip("把这个词交回分词器：例如 迷宮都市 → 迷宮 + 都市，各自的读音")
        self.btn_split.clicked.connect(self._split)
        btns.addWidget(self.btn_split)
        self.btn_word_split = QPushButton("按指定词拆开…")
        self.btn_word_split.setToolTip("指定 世界/征服 等词界；可记为本工程的拆词规则")
        self.btn_word_split.clicked.connect(self._split_words)
        btns.addWidget(self.btn_word_split)
        self.btn_kanji = QPushButton("按单个汉字拆开…")
        self.btn_kanji.setToolTip("逐字填写振假名；能唯一切开时预填，某个字留空就是不标注")
        self.btn_kanji.clicked.connect(self._open_kanji)
        btns.addWidget(self.btn_kanji)
        cancel = QPushButton("取消")
        cancel.setToolTip("Esc")
        if kind == "pinyin":
            cancel.setText("取消")
            self.btn_ok = QPushButton("确定")
            self.btn_ok.clicked.connect(self._accept_input)
            btns.addWidget(self.btn_ok)
        self.cancel_button = cancel
        cancel.clicked.connect(self._cancel)
        btns.addWidget(self.btn_hide)
        if origin == "cc_hint":                 # 眷属(ファミリア): the bracket in the original subtitles
            self.btn_bracket = QPushButton("不是注音：括号放回正文")
            self.btn_bracket.setToolTip("原字幕里写在括号里的不是这个词的读音（例如 眷属(ファミリア)）：把括号原样放回这一行，"
                                        "去掉这个振假名，按普通读音重新建议。可以撤销。")
            self.btn_bracket.clicked.connect(lambda: (self.bracketToText.emit(), self.close()))
            btns.addWidget(self.btn_bracket)
        btns.addStretch(1)
        btns.addWidget(cancel)
        from ..help import dialog_buttons

        auxiliary = tuple(btns.itemAt(i).widget() for i in range(btns.count())
                          if isinstance(btns.itemAt(i).widget(), QPushButton)
                          and btns.itemAt(i).widget() not in (cancel, getattr(self, "btn_ok", None)))
        dialog_buttons(self, btns, primary=getattr(self, "btn_ok", None), auxiliary=auxiliary, default=False)
        lay.addLayout(btns)
        # only after it has a parent: shown without one it would open as a window of its own (and close the popup)
        self.btn_split.setVisible(can_split and len(base) >= 2)
        self._refresh()
        if prefill:
            self.edit.setText(prefill)
            self.edit.selectAll()
        self.edit.setFocus()
        self._outside = OutsideClick(self, self.close)      # a press anywhere else cancels, like a popup

    def scope_choice(self) -> int:
        return max(0, self.scope_group.checkedId())

    def _split(self) -> None:
        self.splitRequested.emit()
        self.close()

    def _split_words(self) -> None:
        if self.kind != "ruby" or len(self.base) < 2:
            return
        self.wordSplitRequested.emit(getattr(self, "a", 0), getattr(self, "b", len(self.base)), self.scope_choice())
        self.close()

    # ------------------------------------------------------------ per kanji
    @staticmethod
    def _kana(text: str) -> str:
        text = text.strip()
        return romaji_to_kana(text) if text and text.isascii() else text

    def _open_kanji(self) -> None:
        if self.kind != "ruby" or self._changed() or sum(is_kanji(ch) for ch in self.base) < 2:
            return
        reading = self._converted() or self.current
        parts = self.kanji_splitter(self.base, reading) if self.kanji_splitter is not None else None
        prefill = {part.start: part.reading for part in parts} if parts is not None else {}
        while self.kanji_grid.count():
            widget = self.kanji_grid.takeAt(0).widget()
            widget.hide()
            widget.deleteLater()
        self._kanji_rows = []
        for n, (pos, ch) in enumerate((i, ch) for i, ch in enumerate(self.base) if is_kanji(ch)):
            row, col = divmod(n, 6)
            label, edit, preview = QLabel(ch), QLineEdit(prefill.get(pos, "")), QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("border:0;background:transparent;")
            edit.setAccessibleName(ch + "的读音")
            edit.setPlaceholderText("不标注")
            edit.setMinimumWidth(60)
            edit.setMaximumWidth(100)
            edit.installEventFilter(self)
            preview.setStyleSheet("color:#0550ae;border:0;background:transparent;")
            edit.textChanged.connect(lambda text, hint=preview: hint.setText(self._kana(text)))
            edit.returnPressed.connect(self._accept_kanji)
            for dy, widget in enumerate((label, edit, preview)):
                self.kanji_grid.addWidget(widget, row * 3 + dy, col)
            self._kanji_rows.append((pos, edit, preview))
            preview.setText(self._kana(edit.text()))
        self.kanji_hint.setText("每个汉字各标一个振假名；某个字留空就是不标注。" if parts is not None else
            "自动分不开（可能是熟字訓），请逐字填写；某个字留空就是那个字不标注")
        self._kanji_mode = True
        self._composing = False
        for widget in (self.cand_box, self.range_cand_box, self.edit, self.preview,
                       self.btn_split, self.btn_kanji, self.btn_hide):
            widget.hide()
        if self.scope is not None:
            self.range_box.hide()
        if hasattr(self, "btn_bracket"):
            self.btn_bracket.hide()
        self.kanji_box.show()
        self.cancel_button.setText("取消")
        self.adjustSize()
        if hasattr(self, "_popup_point"):
            place_near(self, self._popup_point)
        self._kanji_rows[0][1].setFocus()
        self._kanji_rows[0][1].selectAll()

    def _accept_kanji(self) -> None:
        if self._kanji_mode and not self._composing:
            self.kanjiChosen.emit([self._kana(edit.text()) for _pos, edit, _hint in self._kanji_rows], self.scope_choice())
            self.close()

    def _cancel(self) -> None:
        if not self._kanji_mode:
            self.close()
            return
        self._kanji_mode = False
        self._composing = False
        self.kanji_box.hide()
        self.edit.show()
        self.preview.show()
        self.btn_split.setVisible(self._can_split and len(self.base0) >= 2)
        if self.scope is not None:
            self.range_box.show()
        if hasattr(self, "btn_bracket"):
            self.btn_bracket.show()
        self.cancel_button.setText("取消")
        self.cancel_button.setToolTip("Esc")
        self._refresh()
        self.edit.setFocus()
        if hasattr(self, "_popup_point"):
            place_near(self, self._popup_point)

    # ------------------------------------------------------------ range
    def _changed(self) -> bool:
        return self.scope is not None and (self.a, self.b) != (self.a0, self.b0)

    def _extend(self, direction: int) -> None:
        if self.scope is None:
            return
        spans = self.scope["spans"]
        if direction < 0:
            sp = next((s for s in spans if s["b"] == self.a), None)
            if sp is None:
                return
            self.a = sp["a"]
        else:
            sp = next((s for s in spans if s["a"] == self.b), None)
            if sp is None:
                return
            self.b = sp["b"]
        self._refresh(prefill=True)

    def _reset_range(self) -> None:
        if self.scope is not None:
            self.a, self.b = self.a0, self.b0
            self.edit.clear()
            self._refresh()

    def _prefilled(self) -> str:
        text = self.scope["text"]
        out = []
        for s in self.scope["spans"]:
            if s["b"] <= self.a or s["a"] >= self.b:
                continue
            out.append(s["reading"] if s.get("reading") else text[s["a"]:s["b"]])
        return "".join(out)

    def _refresh(self, prefill: bool = False) -> None:
        if self.scope is not None:
            self.base = self.scope["text"][self.a:self.b]
            self.btn_left.setEnabled(any(s["b"] == self.a for s in self.scope["spans"]))
            self.btn_right.setEnabled(any(s["a"] == self.b for s in self.scope["spans"]))
            self.btn_reset.setVisible(self._changed())
        changed = self._changed()
        self.btn_word_split.setVisible(self.kind == "ruby" and len(self.base) >= 2 and not self._kanji_mode)
        self.btn_kanji.setVisible(self.kind == "ruby" and self.scope is not None and not changed
                                  and sum(is_kanji(ch) for ch in self.base) >= 2)
        self.cand_box.setVisible(not changed and bool(self.cands))
        row = self.range_cand_box.layout()
        while row.count():
            button = row.takeAt(0).widget()
            button.hide()
            button.deleteLater()
        self.range_cands = (list(dict.fromkeys(c for c in self.candidates_for(self.a, self.b) if c))
                            if changed and self.candidates_for is not None else [])
        for i, c in enumerate(self.range_cands):
            label = placement_text(self.base, c)
            button = QPushButton(f"{i + 1}  {label}" if i < 9 else label)
            button.clicked.connect(lambda _=False, x=c: self._choose(x))
            row.addWidget(button, i // 4, i % 4)
        self.range_cand_box.setVisible(changed and bool(self.range_cands))
        self.btn_hide.setVisible(not changed)
        src = ORIGIN_LABEL.get(self.origin, self.origin)
        st = {"uncertain": "（待确认）", "confirmed": "（已确认）", "hidden": "（不标注）"}.get(self.status, "")
        if changed:
            head = (f"<b style='font-size:15pt'>{self.base}</b>　<span style='color:#9a6700'>整体一个读音"
                    f"（替换原来 {self.base0} 等的读音）</span>")
        else:
            place = placement_text(self.base, self.current) if self.current else self.base
            head = (f"<b style='font-size:15pt'>{self.base}</b>　当前：{self.current or '—'} {st}<br>"
                    f"<span style='color:#0550ae'>注音位置：{place}</span>　<span style='color:#6e7781'>来源：{src}</span>")
        self.title.setText(f"拼音：{self.base}　当前：{self.current or '—'} {st}" if self.kind == "pinyin" else head)
        if prefill and changed:
            self.edit.setText(self._prefilled())
            self.edit.selectAll()
        self._preview(self.edit.text())
        self.adjustSize()

    # ------------------------------------------------------------ choose
    def _converted(self) -> str:
        t = self.edit.text().strip()
        if not t:
            return ""
        if self.kind == "pinyin":
            from subkit.core.pinyin_notes import normalize_rule

            return normalize_rule(self.base, t)[1]
        if t.isascii():
            return romaji_to_kana(t)
        return t

    def _preview(self, _t: str) -> None:
        try:
            c = self._converted()
        except ValueError as exc:
            self.preview.setText(str(exc))
            return
        self.preview.setText(f"→ {c if self.kind == 'pinyin' else placement_text(self.base, c)}" if c else "")

    def _accept_input(self) -> None:
        try:
            reading = self._converted()
            if reading and self.kind == "ruby":
                self.typedChosen.emit(self.base, reading)
            self._choose(reading or (self.current if self.kind == "pinyin" else ""))
        except ValueError as exc:
            self.preview.setText(str(exc))

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.edit or any(obj is edit for _pos, edit, _hint in self._kanji_rows):
            if event.type() == QEvent.Type.InputMethod:
                self._composing = bool(event.preeditString())
            elif (obj is self.edit and event.type() == QEvent.Type.KeyPress and not self._composing
                  and (self._changed() or (self.kind == "pinyin" and not self.edit.text()))
                  and event.modifiers() == Qt.KeyboardModifier.NoModifier
                  and Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9):
                i = event.key() - Qt.Key.Key_1
                candidates = self.range_cands if self._changed() else self.cands
                if i < len(candidates):
                    self._choose(candidates[i])
                    return True
        return super().eventFilter(obj, event)

    def _choose(self, reading: str) -> None:
        if not reading:
            return
        if self._changed():
            self.chosenRange.emit(self.a, self.b, reading, self.scope_choice())
        else:
            self.chosen.emit(reading, self.scope_choice())
        self.close()

    def _hide(self) -> None:
        self.hideRequested.emit()
        self.close()

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape:
            self._cancel()
            return
        if self._kanji_mode:
            super().keyPressEvent(e)
            return
        if e.modifiers() & Qt.KeyboardModifier.AltModifier and e.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self._extend(-1 if e.key() == Qt.Key.Key_Left else +1)
            return
        if not self._composing and (self._changed() or not self.edit.text()) and Qt.Key.Key_1 <= e.key() <= Qt.Key.Key_9:
            i = e.key() - Qt.Key.Key_1
            candidates = self.range_cands if self._changed() else self.cands
            if i < len(candidates):
                self._choose(candidates[i])
                return
        super().keyPressEvent(e)

    def popup_at(self, pos: QPoint) -> None:
        self._popup_point = pos
        place_near(self, pos)            # below the point if it fits, otherwise above; always on screen
        show_activated(self, self.edit)


class PinyinSettingsDialog(QDialog):
    help_anchor = "pinyin"

    def __init__(self, mode: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("拼音")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("显示拼音："))
        self.modes = QButtonGroup(self)
        self.values = ("off", "PY", "ZY", "WG")
        for i, label in enumerate(("关", "拼音（声调符号）", "注音符号", "威妥玛")):
            button = QRadioButton(label)
            self.modes.addButton(button, i)
            button.setChecked(self.values[i] == mode)
            lay.addWidget(button)
        lay.addWidget(QLabel("只影响中文行；关掉时你改过的拼音会保留"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        from ..help import dialog_buttons

        dialog_buttons(self, buttons)
        lay.addWidget(buttons)

    def mode(self) -> str:
        return self.values[max(0, self.modes.checkedId())]
