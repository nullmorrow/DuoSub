"""Group editor: the current group with its neighbours' edges, text editing, readings."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QModelIndex, QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QKeyEvent, QPainter, QPen, QStandardItem, QStandardItemModel, QSyntaxHighlighter, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QCompleter, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QToolButton,
                               QVBoxLayout, QWidget, QLineEdit)

from subkit.ass.text import plain_text, tokenize
from subkit.core.readings import anchor_positions, display_annotations, pieces_of, visible_map
from subkit.core.editor_text import GLOSS_MARK, editing_text
from subkit.languages import JAPANESE_ONLY, lane_label
from subkit.model.document import Annotation, Document, Group, Member

JP_FONT = ["Yu Gothic UI", "Meiryo UI", "MS UI Gothic", "Microsoft YaHei UI"]
CN_FONT = ["Microsoft YaHei UI", "Microsoft JhengHei UI", "SimHei"]


def font(families: list[str], size: float, bold: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(families)
    f.setPointSizeF(size)
    f.setBold(bold)
    return f


class RubyLine(QWidget):
    """Original-language text with ruby drawn above; click a word to change its reading, or drag across several
    characters (迷宮都市) to give them one reading."""

    readingClicked = Signal(str, QPoint)          # annotation id, global position
    charClicked = Signal(int, QPoint)             # visible index without annotation
    rangeSelected = Signal(int, int, QPoint)      # visible [a, b) dragged across

    def __init__(self, parent=None, *, kind: str = "ruby", mode: str = "PY",
                 background: str = "#fbfbfd", base_colour: str = "#1f2328", ruby_colour=None) -> None:
        super().__init__(parent)
        self.background, self.base_colour, self.ruby_colour = background, base_colour, ruby_colour
        self.kind, self.mode = kind, mode
        if kind == "pinyin":
            self.help_anchor = "pinyin"
        self.doc: Optional[Document] = None
        self.member: Optional[Member] = None
        self.rules = None                                   # DisplayRules: hidden characters are drawn grey
        self.base_font = font(JP_FONT, 17)
        self.ruby_font = font(JP_FONT, 8.5)
        self._boxes: list[tuple[QRectF, str]] = []
        self._chars: list[QRectF] = []
        self._press: Optional[tuple] = None         # (point, char index)
        self._sel: Optional[tuple[int, int]] = None
        self.setMinimumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_member(self, doc: Optional[Document], m: Optional[Member]) -> None:
        self.doc, self.member = doc, m
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(600, 64)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(self.background))
        self._boxes, self._chars = [], []
        if self.doc is None or self.member is None:
            p.setPen(QColor("#8c959f"))
            p.drawText(self.rect().adjusted(10, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, "（这一组没有原声文本）")
            return
        vt, ids = visible_map(self.member)
        text = vt.text
        fm = QFontMetricsF(self.base_font)
        rfm = QFontMetricsF(self.ruby_font)
        x0, base_y = 12.0, self.height() - 16.0
        xs = [x0]
        for ch in text:
            xs.append(xs[-1] + fm.horizontalAdvance(ch))
        for i in range(len(text)):
            self._chars.append(QRectF(xs[i], base_y - fm.ascent(), xs[i + 1] - xs[i], fm.height()))
        if self._sel is not None:
            a, b = self._sel
            if 0 <= a < b <= len(text):
                p.fillRect(QRectF(xs[a], base_y - fm.ascent() - 2, xs[b] - xs[a], fm.height() + 4), QColor(56, 139, 253, 60))
        anns = display_annotations(self.doc, self.member, [a for a in self.doc.annotations.values()
                                                          if a.member == self.member.id and a.kind == self.kind])
        spans = []
        for a in anns:
            pos = anchor_positions(a, ids)
            if pos:
                spans.append((pos, a))
        # underline uncertain / confirmed
        for (s, e), a in spans:
            r = QRectF(xs[s], base_y + 3, xs[e] - xs[s], 2)
            if a.status == "uncertain":
                pen = QPen(QColor("#d4a017"), 2, Qt.PenStyle.DotLine)
                p.setPen(pen)
                p.drawLine(r.topLeft(), r.topRight())
            elif a.status == "confirmed":
                p.setPen(QPen(QColor("#2da44e"), 1))
                p.drawLine(r.topLeft(), r.topRight())
        p.setFont(self.base_font)
        hidden_vis: set[int] = set()
        if self.rules is not None and self.rules.active:
            from subkit.core.display import hidden_positions
            from subkit.core.readings import visible_run_positions

            hp = hidden_positions(self.member, self.rules)
            if hp:
                hidden_vis = {i for i, rp in enumerate(visible_run_positions(self.member)) if rp in hp}
        for i, ch in enumerate(text):
            p.setPen(QColor("#c8ccd1") if i in hidden_vis else QColor(self.base_colour))
            p.drawText(QRectF(xs[i], base_y - fm.ascent(), xs[i + 1] - xs[i] + 2, fm.height()), Qt.AlignmentFlag.AlignLeft, ch)
        p.setFont(self.ruby_font)
        for (s, e), a in spans:
            if a.status == "hidden":
                continue
            if self.kind == "pinyin":
                from subkit.pinyin import display

                pieces = [(s, e, display(a.text, self.mode))]
            else:
                pieces = [(s + pa, s + pb, t) for pa, pb, t in pieces_of(a)]
            for ps, pe, t in pieces:
                cx = (xs[ps] + xs[pe]) / 2
                w = rfm.horizontalAdvance(t)
                avail = max(1.0, xs[pe] - xs[ps] + 8)
                p.save()
                if w > avail:
                    p.translate(cx, 0)
                    p.scale(avail / w, 1.0)
                    p.translate(-cx, 0)
                color = {"uncertain": "#b35900", "confirmed": "#1a7f37"}.get(a.status, "#57606a")
                p.setPen(QColor(self.ruby_colour or color))
                p.drawText(QRectF(cx - w / 2, base_y - fm.ascent() - rfm.height() - 1, w, rfm.height()),
                           Qt.AlignmentFlag.AlignCenter, t)
                p.restore()
            self._boxes.append((QRectF(xs[s], 0, xs[e] - xs[s], self.height()), a.id))

    def _char_at(self, x: float) -> Optional[int]:
        for i, r in enumerate(self._chars):
            if r.left() <= x < r.right():
                return i
        if self._chars and x >= self._chars[-1].right():
            return len(self._chars) - 1
        if self._chars and x < self._chars[0].left():
            return 0
        return None

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self._press = (e.position(), self._char_at(e.position().x()))
        self._sel = None

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._press is None or not (e.buttons() & Qt.MouseButton.LeftButton):
            return
        p0, i0 = self._press
        if i0 is None or abs(e.position().x() - p0.x()) < 6:
            return
        i1 = self._char_at(e.position().x())
        if i1 is not None:
            self._sel = (min(i0, i1), max(i0, i1) + 1)
            self.update()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        press, self._press = self._press, None
        if press is None:
            return
        gp = e.globalPosition().toPoint()
        if self._sel is not None:                     # dragged: one reading for the selected characters
            a, b = self._sel
            self._sel = None
            self.update()
            self.rangeSelected.emit(a, b, gp)
            return
        pt = press[0]
        for r, aid in self._boxes:
            if r.contains(pt):
                self.readingClicked.emit(aid, gp)
                return
        i = self._char_at(pt.x())
        if i is not None and self._chars[i].left() <= pt.x() < self._chars[i].right():
            self.charClicked.emit(i, gp)


class GlossHighlighter(QSyntaxHighlighter):
    def highlightBlock(self, text):  # noqa: N802
        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#8c959f"))
        ranges = [(mark.start(), mark.end()) for mark in GLOSS_MARK.finditer(text)]
        ranges.extend((tok.start, tok.end) for tok in tokenize(text) if tok.kind == "comment" and tok.protected)
        for a, b in ranges:
            # Qt positions count UTF-16 code units.
            start = len(text[:a].encode("utf-16-le")) // 2
            length = len(text[a:b].encode("utf-16-le")) // 2
            self.setFormat(start, length, fmt)


class LineEdit(QPlainTextEdit):
    """Single-member text editor: Enter commits, Shift+Enter is a newline-free no-op, Esc reverts."""

    committed = Signal(str)
    focused = Signal()

    def __init__(self, parent=None, *, fnt: Optional[QFont] = None) -> None:
        super().__init__(parent)
        self.key_hook = None                  # format shortcuts (Ctrl+B …) handled by the group editor
        if fnt is not None:
            self.setFont(fnt)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setTabChangesFocus(True)
        self.setFixedHeight(int(QFontMetricsF(self.font()).height() * 2.3) + 8)
        self._original = ""
        self._gloss_highlighter = GlossHighlighter(self.document())

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(super().sizeHint().width(), self.minimumHeight())

    def set_text(self, t: str) -> None:
        self._original = t
        self.blockSignals(True)
        self.setPlainText(t)
        self.blockSignals(False)

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        if self.key_hook is not None and self.key_hook(e):
            return
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (e.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.commit()
            return
        if e.key() == Qt.Key.Key_Escape:
            self.set_text(self._original)
            return
        super().keyPressEvent(e)

    def focusOutEvent(self, e) -> None:  # noqa: N802
        self.commit()
        super().focusOutEvent(e)

    def focusInEvent(self, e) -> None:  # noqa: N802
        super().focusInEvent(e)
        self.focused.emit()

    def selection(self) -> Optional[tuple[int, int]]:
        c = self.textCursor()
        if not c.hasSelection():
            return None
        a, b = sorted((c.selectionStart(), c.selectionEnd()))
        return a, b

    def commit(self) -> None:
        t = self.toPlainText().replace("\n", "")
        if t != self._original:
            self._original = t
            self.committed.emit(t)


class InlineLineEdit(LineEdit):
    """Single-line inline editor with the same grey mark display as the group editor."""

    def __init__(self):
        super().__init__()
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def text(self):
        return self.toPlainText()

    def setText(self, text):  # noqa: N802
        self.set_text(text)

    def isModified(self):  # noqa: N802
        return self.document().isModified()

    def setModified(self, modified):  # noqa: N802
        self.document().setModified(modified)

    def hasSelectedText(self):  # noqa: N802
        return self.textCursor().hasSelection()

    def selectionStart(self):  # noqa: N802
        return self.textCursor().selectionStart()

    def selectedText(self):  # noqa: N802
        return self.textCursor().selectedText()

    def end(self, select):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End,
                            QTextCursor.MoveMode.KeepAnchor if select else QTextCursor.MoveMode.MoveAnchor)
        self.setTextCursor(cursor)

    def keyPressEvent(self, event):  # noqa: N802
        QPlainTextEdit.keyPressEvent(self, event)

    def focusOutEvent(self, event):  # noqa: N802
        # The enclosing frame owns apply/cancel; focusing its format controls keeps typed text.
        QPlainTextEdit.focusOutEvent(self, event)


class SpeakerCompleter(QCompleter):
    def pathFromIndex(self, index):  # noqa: N802
        return index.data(Qt.ItemDataRole.DisplayRole)


class SpeakerCombo(QComboBox):
    chosen = Signal(object)

    def __init__(self, book, member, recent, parent=None):
        super().__init__(parent)
        self.help_anchor = "people"
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setMinimumWidth(120)
        self.setMaximumWidth(200)
        self.setToolTip("搜索名字、别名")
        self.activated.connect(lambda index: self.chosen.emit(self.itemData(index)))
        self.refresh_choices(book, member, recent)

    def refresh_choices(self, book, member, recent):
        model = QStandardItemModel(self)
        characters = list(book.characters) if book else []
        rank = {cid: i for i, cid in enumerate(recent)}
        characters.sort(key=lambda ch: rank.get(ch.id, len(rank)))
        for ch in characters:
            name = book.display_name(ch.id, member.lang, (member.speaker or {}).get("label", ""), "short")
            item = QStandardItem(name)
            item.setData(ch.id, Qt.ItemDataRole.UserRole)
            item.setData(" ".join([name, *(ch.full_name(lang) for lang in ch.names),
                                   *(ch.name(lang, "short") for lang in ch.names), *ch.aliases]),
                         Qt.ItemDataRole.UserRole + 1)
            model.appendRow(item)
        label = book.speaker_label(member) if book else (member.speaker or {}).get("label", "")
        for text, value in ((f"（原标签：{label}）", None), ("群众…", "crowd"),
                            ("无说话人（不是对白）", "nobody"), ("新建人物…", "new")):
            item = QStandardItem(text)
            item.setData(value, Qt.ItemDataRole.UserRole)
            item.setData(text, Qt.ItemDataRole.UserRole + 1)
            model.appendRow(item)
        self.setModel(model)
        completer = SpeakerCompleter(model, self)
        completer.setCompletionRole(Qt.ItemDataRole.UserRole + 1)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.setCompleter(completer)
        # QComboBox's built-in completion handler interprets the search-role text as
        # a displayed option and also emits activated. Resolve the stable ID once.
        completer.activated[QModelIndex].disconnect()
        completer.activated[QModelIndex].connect(lambda index: self.chosen.emit(index.data(Qt.ItemDataRole.UserRole)))
        cid = (member.speaker or {}).get("character")
        index = self.findData(cid)
        self.setCurrentIndex(index if index >= 0 else self.findData(None))
        if cid == "crowd":
            self.setEditText("群众")
        if cid == "nobody":
            self.setEditText("")
        self.lineEdit().setPlaceholderText("" if cid == "nobody" else "搜索名字、别名")


class ContextLine(QLabel):
    """Keep neighbour text available without imposing its width on the editor."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("color:#8c959f;")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.fontMetrics().height() + 2)

    def setText(self, text):  # noqa: N802
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(self.contentsRect(), Qt.AlignmentFlag.AlignVCenter,
                         self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width()))


class EditorSection(QWidget):
    """Remember the user's choice independently of temporary availability."""

    changed = Signal()

    def __init__(self, title, content, key, expanded, anchor):
        super().__init__()
        self.help_anchor = anchor
        self.key, self.expanded, self.settings = key, expanded, None
        self.caption, self.reason = title, "先选中字幕组"
        self.header = QToolButton(text=title)
        self.header.help_anchor = anchor
        self.header.setCheckable(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.header, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(content)
        self.header.clicked.connect(self._toggle)
        self._refresh()

    def set_settings(self, settings):
        self.settings = settings
        self.expanded = settings.value(self.key, self.expanded, type=bool)
        self._refresh()

    def set_reason(self, reason):
        self.reason = reason
        self._refresh()

    def _toggle(self, checked):
        if not self.reason:
            self.expanded = checked
            if self.settings is not None:
                self.settings.setValue(self.key, checked)
        self._refresh()
        self.changed.emit()

    def _refresh(self):
        opened = self.expanded and not self.reason
        self.header.setText(self.caption + (f"（{self.reason}）" if self.reason else ""))
        self.header.setEnabled(not self.reason)
        self.header.setChecked(opened)
        self.header.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow)
        self.content.setVisible(opened)
        if not opened:
            self.content.resize(self.width(), self.content.layout().heightForWidth(self.width()))
            self.content.layout().activate()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.content.isHidden():
            self.content.resize(self.width(), self.content.layout().heightForWidth(self.width()))
            self.content.layout().activate()


class GroupEditor(QWidget):
    help_anchor = "review"
    """Shows prev tail / current group / next head; emits requests the main window turns into edits."""

    syntaxCopied = Signal(str)
    syntaxInserted = Signal(str)              # mark written around the selected text (not committed yet)
    contentHeightChanged = Signal()
    textCommitted = Signal(str, str)          # member id, new visible text
    missingTextCommitted = Signal(str, str, str)
    readingRequested = Signal(str, QPoint)    # annotation id
    newReadingRequested = Signal(str, int, int, QPoint)   # member id, a, b
    pinyinRequested = Signal(str, int, int, QPoint)
    restoreRequested = Signal(str)            # comment id
    action = Signal(str)                      # named action (move/merge/split/…)
    formatRequested = Signal(str, object, object, dict)   # member id, visible a / b (None = whole line), change
    groupFormatRequested = Signal(str, dict)
    clearFormatRequested = Signal(str)
    positionTargetChanged = Signal()
    speakerRequested = Signal(str, object, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.doc: Optional[Document] = None
        self.group: Optional[Group] = None
        self.character_book = None
        self.recent_speakers = []
        self.speaker_combos = {}
        self.karaoke_mode = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.content = QWidget()
        self.scroll.setWidget(self.content)
        outer.addWidget(self.scroll)
        lay = QVBoxLayout(self.content)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        self.prev_lbl = ContextLine()
        self.title = QLabel()
        self.title.setStyleSheet("color:#57606a;")
        from .format_bar import FormatBar
        from .position_bar import PositionBar

        self.position = PositionBar()
        self.fmt = FormatBar()
        from .quick_colours import QuickColours

        self.quick_colours = QuickColours(self.fmt.current_colour, height=self.fmt.btn_c.sizeHint().height())
        self.quick_colours.formatRequested.connect(self.fmt.formatRequested)
        self.copy_buttons = []
        for kind, label, syntax, anchor, tip in (
                ("gloss", "旁注", "{@注:被注的字=注释文字}", "gloss",
                 "选中文字时：给选中的字加旁注，光标停在「=」后写注释，Enter 生效。没选中时复制旁注写法。F1 查看旁注。"),
                ("gloss_full", "旁注（全部参数）", "{@注:被注的字=注释文字;位置;距离;font=字体;size=字号;color=#RRGGBB}",
                 "gloss", "选中文字时：加带位置、距离、字体、字号、颜色的旁注（填好当前值），光标停在「=」后。"
                          "没选中时复制完整写法。F1 查看旁注。"),
                ("raw", "Raw", "{@Raw=文字}", "display-rules",
                 "选中文字时：把选中的字放进 {@Raw=…}，不参与显示规则，Enter 生效。没选中时复制 Raw 写法。F1 查看显示规则。")):
            button = QToolButton(text=label)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setToolTip(tip)
            button.help_anchor = anchor
            button.copy_syntax_help = True
            button.clicked.connect(lambda _=False, kind=kind, text=syntax: self._syntax_button(kind, text))
            self.fmt.raw_row.addWidget(button)
            self.copy_buttons.append(button)
        self.fmt.formatRequested.connect(self._format)
        self.fmt.scopedFormatRequested.connect(self._format_scoped)
        self.fmt.clearRequested.connect(lambda: self._fmt_mid and self.clearFormatRequested.emit(self._fmt_mid))
        self._fmt_mid: Optional[str] = None
        self._fmt_edit = None
        self._fmt_props: Optional[dict] = None
        self.style_lookup = None              # style name → Style (set by the main window)
        self.cn_box = QVBoxLayout()
        self.cn_box.setSpacing(2)
        self.languages = {}
        self.pinyin_mode = "off"
        self.pinyin_rows: list[RubyLine] = []
        self.pinyin_box = QVBoxLayout()
        self.ruby = RubyLine()
        self.jp_edit = LineEdit(fnt=font(JP_FONT, 13))
        self.missing_edits = {}
        self.missing_box = QVBoxLayout()
        for lane in ("translation", "original"):
            edit = QLineEdit()
            edit.returnPressed.connect(lambda lane=lane: self._add_missing(lane))
            self.missing_box.addWidget(edit)
            self.missing_edits[lane] = edit
        self.original_box = QVBoxLayout()
        self.jp_row = QHBoxLayout()
        self.jp_row.addWidget(self.jp_edit, 1)
        self.whole_speaker = QCheckBox("整组同一人")
        self.whole_speaker.setChecked(True)
        self.whole_speaker.help_anchor = "people"
        self.hidden_row = QHBoxLayout()
        self.next_lbl = ContextLine()
        from .waveform import FlowLayout

        self.action_area = QWidget()
        actions = QVBoxLayout(self.action_area)
        actions.setContentsMargins(0, 0, 0, 0)
        boundary = QWidget()
        boundary_row = FlowLayout(boundary)
        boundary_row.setContentsMargins(0, 0, 0, 0)
        boundary_row.addWidget(QLabel("分界"))
        actions.addWidget(boundary)
        lower = QHBoxLayout()
        actions.addLayout(lower)
        secondary = QWidget()
        btns = FlowLayout(secondary)
        btns.setContentsMargins(0, 0, 0, 0)
        lower.addWidget(secondary, 1)
        operations, flags = QWidget(), QWidget()
        op_row, flag_row = QHBoxLayout(operations), QHBoxLayout(flags)
        for row in (op_row, flag_row):
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
        btns.addWidget(operations)
        btns.addWidget(flags)
        self.action_buttons = []
        btns.setSpacing(4)
        for label, name, tip in (
            ("开头一段 → 上一组", "head_to_prev", "开头一段 → 上一组（Alt+↑）"),
            ("结尾一段 → 下一组", "tail_to_next", "结尾一段 → 下一组（Alt+↓）"),
            ("拿回上一组结尾一段", "take_from_prev", "拿回上一组结尾一段（Alt+Shift+↑）"),
            ("拿回下一组开头一段", "take_from_next", "拿回下一组开头一段（Alt+Shift+↓）"),
            ("合并", "merge", "与下一组合并（Ctrl+J）"),
            ("拆分", "split", "在光标处拆分（Ctrl+K）"),
            ("解除配对", "unlink", "解除配对（Ctrl+U）"),
            ("✓ 确认并下一组", "confirm_next", "确认并播放下一组（Ctrl+Enter）"),
        ):
            b = QPushButton() if name == "confirm_next" else QToolButton()
            b.setText(label)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, n=name: self.action.emit(n))
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            if name == "confirm_next":
                b.setAutoDefault(False)
                b.setDefault(True)
                lower.addWidget(b, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            elif len(self.action_buttons) < 4:
                boundary_row.addWidget(b)
            else:
                op_row.addWidget(b)
            self.action_buttons.append(b)
        self.review_buttons = {}
        for flag, label in (("hidden", "隐藏"), ("marked", "标记")):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setEnabled(False)
            button.help_anchor = "hide-mark"
            button.setToolTip("不显示、不导出，原文保留" if flag == "hidden" else "标记待复查，不影响显示和导出")
            button.clicked.connect(lambda _=False, name=flag: self.action.emit(name))
            self.review_buttons[flag] = button
            flag_row.addWidget(button)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        flag_row.insertWidget(0, separator)
        header = QHBoxLayout()
        header.addWidget(self.title, 1)
        header.addWidget(self.quick_colours, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.prev_lbl)
        lay.addLayout(header)
        lay.addWidget(self.whole_speaker)
        lay.addLayout(self.cn_box)
        lay.addLayout(self.pinyin_box)
        lay.addLayout(self.original_box)
        lay.addWidget(self.ruby)
        lay.addLayout(self.jp_row)
        lay.addLayout(self.missing_box)
        lay.addLayout(self.hidden_row)
        lay.addWidget(self.action_area)
        self.sections = {}
        for key, caption, content, expanded, anchor in (
                ("format", "格式", self.fmt, True, "format"),
                ("ass", "ASS 标签与旁注", self.fmt.ass_tools, False, "format"),
                ("position", "位置", self.position, False, "position")):
            section = EditorSection(caption, content, "editor/section/" + key, expanded, anchor)
            section.changed.connect(self._update_content_height)
            self.sections[key] = section
            lay.addWidget(section)
        lay.addWidget(self.next_lbl)
        lay.addStretch(1)
        self.cn_edits: list[tuple[str, LineEdit]] = []
        self._translation_rows = []
        self.jp_member: Optional[Member] = None
        self.jp_edit.committed.connect(lambda t: self.jp_member and self.textCommitted.emit(self.jp_member.id, t))
        self.ruby.readingClicked.connect(self.readingRequested)
        self.ruby.charClicked.connect(self._char_clicked)
        self.ruby.rangeSelected.connect(lambda a, b, pos: self.jp_member and self.newReadingRequested.emit(self.jp_member.id, a, b, pos))
        self.jp_edit.focused.connect(lambda: self.jp_member and self._set_fmt_target(self.jp_member.id, self.jp_edit))
        self.jp_edit.selectionChanged.connect(lambda: self._fmt_edit is self.jp_edit and self._show_fmt())
        self.jp_edit.key_hook = self.fmt.handle_key
        self.jp_edit.hide()
        self.ruby.hide()
        for edit in self.missing_edits.values():
            edit.hide()
        self.position.installEventFilter(self)
        self.scroll.viewport().installEventFilter(self)
        self._update_content_height()

    def set_settings(self, settings) -> None:
        for section in self.sections.values():
            section.set_settings(settings)
        self._update_content_height()

    def _update_sections(self) -> None:
        reason = "" if self.group is not None and self._fmt_mid is not None else "先选中字幕组"
        for section in self.sections.values():
            section.set_reason(reason)
        # With a group selected, the position bar is off until the preview gives an anchor (same wording as the menu).
        self.sections["position"].set_reason(reason or ("先等待字幕预览更新" if not self.position.isEnabled() else ""))

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.position and event.type() == QEvent.Type.EnabledChange:
            self._update_sections()
            self._update_content_height()
        elif obj is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._update_content_height()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_content_height()

    def _update_tab_order(self) -> None:
        widgets = [self.whole_speaker]
        for layout in (self.cn_box, self.original_box, self.jp_row, self.missing_box, self.hidden_row):
            for i in range(layout.count()):
                widget = layout.itemAt(i).widget()
                if widget is not None:
                    widgets.extend([widget, *widget.findChildren(QWidget)])
        widgets.extend([*self.action_buttons[:-1], *self.review_buttons.values(), self.action_buttons[-1]])
        for section in self.sections.values():
            widgets.extend([section.header, *section.content.findChildren(QWidget)])
        ordered = []
        for widget in widgets:
            proxy = widget.focusProxy() or widget
            if proxy not in ordered and proxy.focusPolicy() & Qt.FocusPolicy.TabFocus:
                ordered.append(proxy)
        for first, second in zip(ordered, ordered[1:]):
            QWidget.setTabOrder(first, second)

    def _copy_syntax(self, text: str) -> None:
        QApplication.clipboard().setText(text)
        self.syntaxCopied.emit(text)

    def _syntax_button(self, kind: str, syntax: str) -> None:
        """Text selected in the focused text box: write the mark around it there (糧食庫 → 糧食庫{@注:糧食庫=}
        or {@Raw=糧食庫}), not committed yet; without a selection the syntax is copied as before."""
        edit = self._fmt_edit
        cursor = edit.textCursor() if edit is not None and edit.hasFocus() else None
        selected = cursor.selectedText() if cursor is not None and cursor.hasSelection() else ""
        if not selected.strip() or any(c in selected for c in "{} \n"):
            self._copy_syntax(syntax)
            return
        if kind == "raw":
            text = "{@Raw=" + selected + "}"
            caret = len(text)
        else:
            text = f"{selected}{{@注:{selected}="
            caret = len(text)
            text += (self._gloss_options() if kind == "gloss_full" else "") + "}"
        start = cursor.selectionStart()
        cursor.insertText(text)
        cursor.setPosition(start + caret)
        edit.setTextCursor(cursor)
        self.syntaxInserted.emit(text)

    def _gloss_options(self) -> str:
        """;位置;距离;font=…;size=…;color=… filled with what the gloss shows by default for this text."""
        from .format_bar import qcolour

        props = self._fmt_props or {}
        style = self.style_lookup(self.doc.members[self._fmt_mid].style) \
            if self.style_lookup and self.doc is not None and self._fmt_mid in self.doc.members else None
        colour = qcolour(props.get("c1") or (style.get("PrimaryColour") if style else "") or "")
        family = props.get("fn") or (style.fontname if style else "")
        size = float(props.get("fs") or 48.0) / 2
        return (";8;0" + (f";font={family}" if family else "") + f";size={size:g}"
                + (f";color={colour.name().upper()}" if colour is not None else ""))

    def changeEvent(self, e) -> None:  # noqa: N802
        super().changeEvent(e)
        if e.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._update_content_height()     # control heights follow the font

    def _update_content_height(self) -> None:
        self.position.layout().invalidate()
        layout = self.content.layout()
        layout.invalidate()
        needed = max(layout.minimumSize().height(), layout.heightForWidth(self.scroll.viewport().width()))
        self.content.setMinimumHeight(needed)
        # Long text and expanded tools scroll vertically rather than enlarge the window.
        minimum = min(needed, 300)
        if minimum != self.minimumHeight():
            self.setMinimumHeight(minimum)
            self.contentHeightChanged.emit()

    def set_languages(self, languages) -> None:
        self.languages = dict(languages)
        self.jp_edit.setPlaceholderText(lane_label("original", languages))
        self.jp_edit.setToolTip(lane_label("original", languages))
        self.ruby.setToolTip(JAPANESE_ONLY)

    def has_uncommitted(self) -> bool:
        """A text box holds typed text not yet committed (Enter / focus out)."""
        return any(e.toPlainText().replace("\n", "") != e._original for e in self.findChildren(LineEdit))

    def set_karaoke_mode(self, enabled: bool) -> None:
        self.karaoke_mode = enabled
        self.help_anchor = "karaoke" if enabled else "review"
        for widget in (self.prev_lbl, self.title, self.next_lbl, self.quick_colours,
                       self.whole_speaker, self.action_area, *self.sections.values()):
            widget.setVisible(not enabled)
        if self.doc is not None:
            self.set_group(self.doc, self.group, *getattr(self, "_neighbours", (None, None)))
        self._update_content_height()

    def _add_missing(self, lane):
        text = self.missing_edits[lane].text()
        if self.group is not None and text.strip():
            self.missingTextCommitted.emit(self.group.id, lane, text)

    def _speaker_row(self, member, edit):
        from subkit.core.people import needs_speaker
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        if not self.karaoke_mode and needs_speaker(self.group, member):
            layout.addWidget(self._speaker_combo(member))
        layout.addWidget(edit, 1)
        if not self.karaoke_mode:
            layout.addWidget(self._traditional_note(member))
        return row

    def _traditional_note(self, member):
        label = QLabel()
        label.setStyleSheet("color: #8b949e; font-size: 11px;")
        label.setProperty("traditional_member", member.id)
        rules = self.ruby.rules
        label.setText(rules.script_label if rules is not None else "")
        label.setVisible(rules is not None and rules.converted_for(member))
        return label

    def _speaker_combo(self, member):
        combo = SpeakerCombo(self.character_book, member, self.recent_speakers)
        combo.setProperty("member_id", member.id)
        combo.chosen.connect(lambda cid: self.speakerRequested.emit(combo.property("member_id"), cid, self.whole_speaker.isChecked()))
        self.speaker_combos[member.id] = combo
        return combo

    def _char_clicked(self, i: int, pos: QPoint) -> None:
        if self.jp_member is None:
            return
        c = self.jp_edit.textCursor()
        if c.hasSelection():     # editing-text offsets → visible characters
            from subkit.core.readings import plain_range_to_visible

            a, b = sorted((c.selectionStart(), c.selectionEnd()))
            r = plain_range_to_visible(self.jp_member, a, b)
            if r is not None:
                self.newReadingRequested.emit(self.jp_member.id, r[0], r[1], pos)
        else:
            self.newReadingRequested.emit(self.jp_member.id, i, i + 1, pos)

    # ------------------------------------------------------------ formatting (font, size, bold, colour)
    def _set_fmt_target(self, mid: str, edit) -> None:
        self._fmt_mid, self._fmt_edit = mid, edit
        self._show_fmt()
        self._update_content_height()
        self.positionTargetChanged.emit()

    def position_member(self) -> Optional[str]:
        if self.group is None:
            return None
        if self._fmt_mid in self.group.members:
            return self._fmt_mid
        return self.jp_member.id if self.jp_member else next(iter(self.group.members), None)

    def _show_fmt(self) -> None:
        from subkit.core.readings import plain_range_to_visible
        from subkit.core.textformat import shown_props

        from .format_bar import describe_target, format_selection

        if self.doc is None or self._fmt_mid not in (self.doc.members if self.doc else {}):
            return
        m = self.doc.members[self._fmt_mid]
        st = self.style_lookup(m.style) if self.style_lookup else None
        sel = self._fmt_edit.selection() if self._fmt_edit is not None else None
        vis = plain_range_to_visible(m, *sel) if sel else None
        vis = format_selection(visible_map(m)[0].text, vis)
        props = shown_props(m, st, vis)
        # Source overrides precede the first character; manual formatting wins over them.
        family = st.fontname if st else ""
        base = family
        pos = 0
        for tok in tokenize(m.text):
            if tok.kind == "tags":
                for tag in tok.tags:
                    if tag.name == "fn":
                        family = base if tag.arg.strip() in ("", "0") else tag.arg.strip()
                    elif tag.name == "r":
                        reset = self.style_lookup(tag.arg.strip()) if tag.arg.strip() and self.style_lookup else None
                        family = reset.fontname if reset else base
            elif tok.kind in ("text", "hardspace", "softbreak"):
                pos += len(tok.raw) if tok.kind == "text" else 1
                if pos > (vis[0] if vis else 0):
                    break
        from subkit.ass.tags import parse_block
        from subkit.core.textformat import char_props

        formatting = [m.overrides]
        if vis:
            cp = char_props(m)
            formatting.append((cp[vis[0]] if vis[0] < len(cp) else None) or {})
        for fmt in formatting:
            if fmt.get("fn"):
                family = fmt["fn"]
            for tag in parse_block(fmt.get("raw") or ""):
                if tag.name == "fn":
                    family = base if tag.arg.strip() in ("", "0") else tag.arg.strip()
        props["fn"] = family
        self._fmt_props = props
        text = visible_map(m)[0].text
        from subkit.core.fades import query_fade_inheritance

        self.fmt.show_inherited_fade(query_fade_inheritance(self.doc, m.id).raw)
        self.fmt.show_state(target=describe_target(text, vis), fn=props["fn"] or "", fs=float(props["fs"] or 48.0),
                            bold=bool(props["b"]), italic=bool(props["i"]), colour=props["c1"],
                            raw=props.get("raw") or "", span=bool(vis))

    def _format(self, change: dict) -> None:
        if self.doc is None or self._fmt_mid not in self.doc.members:
            return
        from subkit.core.readings import plain_range_to_visible

        if self._fmt_edit is not None:
            self._fmt_edit.commit()
        m = self.doc.members[self._fmt_mid]
        sel = self._fmt_edit.selection() if self._fmt_edit is not None else None
        vis = plain_range_to_visible(m, *sel) if sel else None
        from .format_bar import format_selection

        vis = format_selection(visible_map(m)[0].text, vis)
        self.formatRequested.emit(self._fmt_mid, vis[0] if vis else None, vis[1] if vis else None, change)

    def _format_scoped(self, change: dict, scope: str) -> None:
        if self.doc is None or self._fmt_mid not in self.doc.members:
            return
        if self._fmt_edit is not None:
            self._fmt_edit.commit()
        if scope == "group":
            self.groupFormatRequested.emit(self._fmt_mid, change)
        else:
            self.formatRequested.emit(self._fmt_mid, None, None, change)

    def jp_cursor(self) -> Optional[int]:
        from subkit.core.editor_text import ordinary_offset

        return ordinary_offset(self.jp_edit.toPlainText(), self.jp_edit.textCursor().position()) \
            if self.jp_member is not None else None

    def set_rules(self, rules) -> None:
        self.ruby.rules = rules
        self.ruby.update()
        if self.doc is not None:
            for label in self.findChildren(QLabel):
                mid = label.property("traditional_member")
                if mid and mid in self.doc.members:
                    label.setText(rules.script_label)
                    label.setVisible(rules.converted_for(self.doc.members[mid]))

    def set_group(self, doc: Document, g: Optional[Group], prev: Optional[Group], nxt: Optional[Group]) -> None:
        from subkit.core.people import needs_speaker
        self.doc, self.group = doc, g
        for flag, button in self.review_buttons.items():
            button.setEnabled(g is not None)
            button.setChecked(g is not None and flag in g.flags)
        self.whole_speaker.setVisible(not self.karaoke_mode and g is not None
                                      and any(needs_speaker(g, m) for m in doc.members_of(g)))
        self._neighbours = (prev, nxt)
        for lane, edit in self.missing_edits.items():
            missing = g is not None and not any(m.lane == lane and m.role != "comment" for m in doc.members_of(g))
            edit.setVisible(missing)
            edit.clear()
            edit.setPlaceholderText(f"这一组没有{lane_label(lane, self.languages)}，在这里输入后按 Enter 补上")
        reusable = list(self._translation_rows)
        self._translation_rows = []
        self.speaker_combos = {}
        while self.original_box.count():
            self.original_box.takeAt(0).widget().deleteLater()
        while self.jp_row.count() > 1:
            self.jp_row.takeAt(0).widget().deleteLater()
        while self.pinyin_box.count():
            w = self.pinyin_box.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.pinyin_rows = []
        while self.cn_box.count():
            w = self.cn_box.takeAt(0).widget()
            if w and not any(w is row for row, _edit, _key in reusable):
                w.deleteLater()
        self.cn_edits = []
        while self.hidden_row.count():
            w = self.hidden_row.takeAt(0).widget()
            if w:
                w.deleteLater()
        if g is None:
            for row, _edit, _key in reusable:
                row.deleteLater()
            self._fmt_mid, self._fmt_edit, self.jp_member = None, None, None
            self.position.show_anchor(None)
            self.title.setText("")
            self.ruby.set_member(None, None)
            self.jp_edit.set_text("")
            self.jp_edit.hide()
            self.prev_lbl.clear()
            self.next_lbl.clear()
            self._update_sections()
            self._update_tab_order()
            self._update_content_height()
            return

        def edge(grp: Optional[Group], tail: bool) -> str:
            if grp is None:
                return ""
            jp = [m for m in doc.members_of(grp) if m.lane == "original"]
            t = " ／ ".join(plain_text(m.text, breaks=" ") for m in jp)
            cn = " ／ ".join(plain_text(m.text, breaks=" ") for m in doc.members_of(grp) if m.lane == "translation")
            return f"{'上一组' if tail else '下一组'}：{cn}  ｜ {t}"

        self.prev_lbl.setText(edge(prev, True))
        self.next_lbl.setText(edge(nxt, False))
        st = "✓ 已确认" if g.state == "confirmed" else "未确认"
        self.title.setText(f"第 {doc.group_index(g.id) + 1} 组 · {st} · {({'dialogue': '对白', 'song': '歌曲', 'sign': '屏幕字', 'info': '注释'}).get(g.kind, g.kind)}")
        for m in doc.members_of(g):
            if m.lane == "translation" and m.lang.lower().split("-")[0] == "ja":
                row = RubyLine()
                row.set_member(doc, m)
                row.readingClicked.connect(self.readingRequested)
                row.charClicked.connect(lambda i, pos, mid=m.id: self.newReadingRequested.emit(mid, i, i + 1, pos))
                row.rangeSelected.connect(lambda a, b, pos, mid=m.id: self.newReadingRequested.emit(mid, a, b, pos))
                self.cn_box.addWidget(row)
            if self.pinyin_mode != "off" and m.lang.lower().startswith("zh"):
                row = RubyLine(kind="pinyin", mode=self.pinyin_mode)
                row.set_member(doc, m)
                row.readingClicked.connect(self.readingRequested)
                row.charClicked.connect(lambda i, pos, mid=m.id: self.pinyinRequested.emit(mid, i, i + 1, pos))
                row.rangeSelected.connect(lambda a, b, pos, mid=m.id: self.pinyinRequested.emit(mid, a, b, pos))
                (self.cn_box if m.lane == "translation" else self.pinyin_box).addWidget(row)
                self.pinyin_rows.append(row)
            if m.lane != "translation":
                continue
            key = (self.karaoke_mode, needs_speaker(g, m), m.lang)
            match = next((i for i, (_row, _edit, old_key) in enumerate(reusable) if old_key == key), None)
            if match is None:
                e = LineEdit(fnt=font(CN_FONT, 12))
                e.focused.connect(lambda ed=e: self._set_fmt_target(ed.property("member_id"), ed))
                e.selectionChanged.connect(lambda ed=e: self._fmt_edit is ed and self._show_fmt())
                e.committed.connect(lambda t, ed=e: self.textCommitted.emit(ed.property("member_id"), t))
                e.key_hook = self.fmt.handle_key
                row = self._speaker_row(m, e)
            else:
                row, e, _key = reusable.pop(match)
                combo = row.findChild(SpeakerCombo)
                if combo is not None:
                    combo.setProperty("member_id", m.id)
                    combo.refresh_choices(self.character_book, m, self.recent_speakers)
                    self.speaker_combos[m.id] = combo
                for label in row.findChildren(QLabel):
                    if label.property("traditional_member"):
                        label.setProperty("traditional_member", m.id)
                        rules = self.ruby.rules
                        label.setText(rules.script_label if rules is not None else "")
                        label.setVisible(rules is not None and rules.converted_for(m))
            previous_mid = e.property("member_id")
            e.setProperty("member_id", m.id)
            e.setPlaceholderText(lane_label("translation", self.languages))
            e.setToolTip(lane_label("translation", self.languages))
            text = editing_text(m)
            if previous_mid != m.id or e.toPlainText() != text:
                e.set_text(text)
            self.cn_box.addWidget(row)
            self._translation_rows.append((row, e, key))
            self.cn_edits.append((m.id, e))
        for row, _edit, _key in reusable:
            row.deleteLater()
        jp = [m for m in doc.members_of(g) if m.lane == "original"]
        for m in jp[:-1]:
            row = RubyLine()
            row.set_member(doc, m)
            row.setVisible(m.lang.lower().split("-")[0] == "ja")
            row.readingClicked.connect(self.readingRequested)
            row.charClicked.connect(lambda i, pos, mid=m.id: self.newReadingRequested.emit(mid, i, i + 1, pos))
            row.rangeSelected.connect(lambda a, b, pos, mid=m.id: self.newReadingRequested.emit(mid, a, b, pos))
            self.original_box.addWidget(row)
            edit = LineEdit(fnt=font(JP_FONT, 13))
            edit.set_text(editing_text(m))
            edit.committed.connect(lambda text, mid=m.id: self.textCommitted.emit(mid, text))
            edit.focused.connect(lambda mid=m.id, ed=edit: self._set_fmt_target(mid, ed))
            edit.selectionChanged.connect(lambda ed=edit: self._fmt_edit is ed and self._show_fmt())
            edit.key_hook = self.fmt.handle_key
            self.original_box.addWidget(self._speaker_row(m, edit))
        self.jp_member = jp[-1] if jp else None
        if self.jp_member and not self.karaoke_mode:
            if needs_speaker(g, self.jp_member):
                self.jp_row.insertWidget(0, self._speaker_combo(self.jp_member))
            self.jp_row.insertWidget(self.jp_row.count() - 1, self._traditional_note(self.jp_member))
        japanese = self.jp_member is not None and self.jp_member.lang.lower().split("-")[0] == "ja"
        self.ruby.setVisible(japanese)
        self.ruby.set_member(doc, self.jp_member if japanese else None)
        self.jp_edit.set_text(editing_text(self.jp_member) if self.jp_member else "")
        self.jp_edit.setVisible(self.jp_member is not None)
        for cid in (() if self.karaoke_mode else g.comments):
            c = doc.comments[cid]
            if c.kind in ("hidden_filler", "user_hidden"):
                lab = QLabel(f"未渲染：{c.content}")
                lab.setStyleSheet("color:#b35900;")
                b = QPushButton("恢复显示")
                b.clicked.connect(lambda _=False, x=cid: self.restoreRequested.emit(x))
                self.hidden_row.addWidget(lab)
                self.hidden_row.addWidget(b)
        if self.hidden_row.count():
            self.hidden_row.addStretch(1)
        mid = self.position_member()
        edit = self.jp_edit if self.jp_member and mid == self.jp_member.id else next(
            (ed for member, ed in self.cn_edits if member == mid), None)
        self._fmt_mid, self._fmt_edit = mid, edit
        self._show_fmt()
        self._update_sections()
        self._update_tab_order()
        self._update_content_height()
        self.positionTargetChanged.emit()
