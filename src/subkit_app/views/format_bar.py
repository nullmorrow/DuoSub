"""Font / size / bold / italic / colour — and any other ASS override tags — for the selected text, or the whole line
when nothing is selected."""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import (QColorDialog, QComboBox, QDoubleSpinBox, QFontComboBox, QHBoxLayout, QLabel, QLineEdit,
                               QToolButton, QVBoxLayout, QWidget)

from subkit.ass.state import check_raw
from subkit.core.fontcheck import installed_font_aliases

RAW_HELP = ("ASS 标签，回车生效，清空回车即去掉。范围可选自动（选区／整行）、整行或整组原声和翻译。\n"
            "全选文字视作整行；整组会保留各行其他手动标签，只替换同类标签。常用：\n"
            "\\frz15 旋转（\\frx \\fry 立体旋转，\\org(x,y) 旋转中心，只能整行）\n"
            "\\fscx120 \\fscy80 横 / 纵缩放　\\fax0.3 \\fay0.1 斜切　\\fsp5 字距\n"
            "\\bord5 边框　\\shad3 阴影　\\blur2 模糊　\\3c&H0000FF& 边框颜色　\\alpha&H80& 半透明\n"
            "\\t(0,500,\\frz30) 0~500 毫秒内的动画　\\fad(200,200) 淡入淡出　\\clip(x1,y1,x2,y2) 遮罩（只能整行）\n"
            "振假名会跟着旋转、斜切、缩放、移动和颜色变化。位置请直接拖动（这里不写 \\pos \\move \\an）。")


def ass_colour(c: QColor) -> str:
    return f"&H{c.blue():02X}{c.green():02X}{c.red():02X}&"


def qcolour(ass: str) -> Optional[QColor]:
    h = (ass or "").upper().replace("&H", "").replace("&", "")
    if len(h) < 6:
        return None
    h = h[-6:]
    return QColor(int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


class FormatBar(QWidget):
    help_anchor = "format"
    """Emits ``formatRequested(change)`` (keys fn fs b i c1; None removes) and ``clearRequested()``;
    the owner decides which text it applies to (selection or whole line)."""

    formatRequested = Signal(dict)
    scopedFormatRequested = Signal(dict, str)  # ASS tags only: explicit line / group scope
    clearRequested = Signal()

    def __init__(self, parent=None, *, compact: bool = False) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        from .waveform import FlowLayout

        format_tools = QWidget()
        lay = QHBoxLayout(format_tools) if compact else FlowLayout(format_tools)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        outer.addWidget(format_tools)
        self.target = QLabel("整行")
        self.target.setStyleSheet("color:#57606a;")
        self.font_box = QFontComboBox()
        self.font_box.setEditable(True)
        self.font_box.setMaximumWidth(150 if compact else 190)
        self.font_box.setToolTip("字体（选中的字；没选就是整行）")
        self.size = QDoubleSpinBox()
        self.size.setRange(6, 300)
        self.size.setDecimals(1)
        self.size.setSingleStep(2)
        self.size.setKeyboardTracking(False)
        self.size.setToolTip("字号（脚本像素，按 1080p）")
        self.btn_smaller = QToolButton(text="A−")
        self.btn_smaller.setToolTip("缩小（Ctrl+Shift+<）")
        self.btn_bigger = QToolButton(text="A+")
        self.btn_bigger.setToolTip("放大（Ctrl+Shift+>）")
        self.btn_b = QToolButton(text="B")
        f = QFont()
        f.setBold(True)
        self.btn_b.setFont(f)
        self.btn_b.setToolTip("粗体（Ctrl+B）")
        self.btn_i = QToolButton(text="I")
        f2 = QFont()
        f2.setItalic(True)
        self.btn_i.setFont(f2)
        self.btn_i.setToolTip("斜体（Ctrl+I）")
        self.btn_c = QToolButton(text="颜色")
        self.btn_c.setToolTip("字的颜色")
        self.btn_clear = QToolButton(text="清除格式")
        self.btn_clear.setToolTip("去掉这一行所有的字体/字号/粗斜体/颜色和 ASS 标签设置（位置不变）")
        for w in (self.target, self.font_box, self.size, self.btn_smaller, self.btn_bigger, self.btn_b, self.btn_i,
                  self.btn_c, self.btn_clear):
            lay.addWidget(w)
        if compact:
            from .quick_colours import QuickColours

            self.quick_colours = QuickColours(self.current_colour, height=self.btn_c.sizeHint().height())
            self.quick_colours.formatRequested.connect(self.formatRequested)
            lay.insertWidget(lay.indexOf(self.btn_c) + 1, self.quick_colours)
        self.ass_tools = QWidget()
        ass_layout = QVBoxLayout(self.ass_tools)
        ass_layout.setContentsMargins(0, 0, 0, 0)
        raw_tools = QWidget()
        row2 = self.raw_row = QHBoxLayout(raw_tools) if compact else FlowLayout(raw_tools)
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(3)
        lab = QLabel("ASS 标签")
        lab.setStyleSheet("color:#57606a;")
        lab.setToolTip(RAW_HELP)
        self.raw_scope = QComboBox()
        for title, scope in (("自动范围", "auto"), ("整行", "line"), ("整组", "group")):
            self.raw_scope.addItem(title, scope)
        self.raw_scope.setToolTip("只控制 ASS 标签：自动范围按选区（全选视作整行）；整组同时作用于原声和翻译，振假名跟随正文")
        self.raw_scope.currentIndexChanged.connect(lambda _i: self._mark_raw(""))
        self.raw = QLineEdit()
        self.raw.setPlaceholderText("例：\\frz10\\fscx120（旋转 10°、横向拉伸）— 回车生效")
        self.raw.setToolTip(RAW_HELP)
        self.raw.setClearButtonEnabled(True)
        self.raw.returnPressed.connect(self._apply_raw)
        self.raw.textEdited.connect(lambda _t: self._mark_raw(""))
        row2.addWidget(lab)
        row2.addWidget(self.raw_scope)
        if compact:
            row2.addWidget(self.raw, 1)
        else:
            row2.addWidget(self.raw)
        ass_layout.addWidget(raw_tools)
        outer.addWidget(self.ass_tools)
        self.fade_hint = QLabel()
        self.fade_hint.setStyleSheet("color:#57606a;")
        self.fade_hint.setWordWrap(True)
        self.fade_hint.setToolTip("原声没有自己的淡入淡出时，自动跟同组同时间的翻译；原声直接设置的标签优先，\\fad(0,0) 可关闭跟随。")
        self.fade_hint.hide()
        ass_layout.addWidget(self.fade_hint)
        self._span = False
        self._raw_shown = ""
        self._current = 0.0
        self.font_box.activated.connect(lambda _i: self.formatRequested.emit({"fn": self.font_box.currentFont().family()}))
        self.size.valueChanged.connect(self._size_changed)
        self.btn_bigger.clicked.connect(lambda: self.step_size(+1))
        self.btn_smaller.clicked.connect(lambda: self.step_size(-1))
        self.btn_b.clicked.connect(lambda: self.formatRequested.emit({"b": 0 if self._bold else 1}))
        self.btn_i.clicked.connect(lambda: self.formatRequested.emit({"i": 0 if self._italic else 1}))
        self.btn_c.clicked.connect(self._pick_colour)
        self.btn_clear.clicked.connect(self.clearRequested)
        self._bold = False
        self._italic = False
        self._colour: Optional[str] = None

    # ------------------------------------------------------------ state shown
    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        return QSize(hint.width(), self.layout().heightForWidth(self.width())) if self.layout().hasHeightForWidth() else hint

    def show_inherited_fade(self, raw: str) -> None:
        self.fade_hint.setText(f"淡入淡出：{raw}（自动跟随同组翻译）" if raw else "")
        self.fade_hint.setVisible(bool(raw))

    def show_state(self, *, target: str, fn: str, fs: float, bold: bool, italic: bool, colour: Optional[str],
                   raw: str = "", span: bool = False) -> None:
        self.target.setText(target)
        self._span = span
        if not self.raw.hasFocus() or raw != self._raw_shown:
            self.raw.setText(raw or "")
            self._mark_raw("")
        self._raw_shown = raw or ""
        self.font_box.blockSignals(True)
        self.font_box.setCurrentFont(QFont(fn))
        installed = (not fn or fn.lstrip("@").casefold() in installed_font_aliases()
                     or fn.lstrip("@").casefold() in {name.casefold() for name in QFontDatabase.families()})
        self.font_box.setEditText(fn if installed else fn + "（未安装）")
        self.font_box.setToolTip("字体（选中的字；没选就是整行）" if installed
                                 else "本机没有这个字体，画面会用替代字体显示")
        self.font_box.blockSignals(False)
        self.size.blockSignals(True)
        self.size.setValue(fs)
        self.size.blockSignals(False)
        self._current = fs
        self._bold, self._italic, self._colour = bold, italic, colour
        self.btn_b.setStyleSheet("background:#ddf4ff;" if bold else "")
        self.btn_i.setStyleSheet("background:#ddf4ff;" if italic else "")
        q = qcolour(colour or "")
        self.btn_c.setStyleSheet(f"color: {q.name()};" if q is not None else "")

    # ------------------------------------------------------------ actions
    def _size_changed(self, v: float) -> None:
        if abs(v - self._current) > 0.05:
            self._current = v
            self.formatRequested.emit({"fs": round(v, 1)})

    def step_size(self, direction: int) -> None:
        base = self._current or self.size.value()
        v = max(6.0, round(base * (1.15 if direction > 0 else 1 / 1.15), 1))
        self.size.setValue(v)

    def _apply_raw(self) -> None:
        scope = self.raw_scope.currentData()
        raw, err = check_raw(self.raw.text(), span=self._span and scope == "auto")
        if err:
            if "只能用于整行" in err:
                err += "；也可以在 ASS 标签旁选择「整行」或「整组」"
            self._mark_raw(err)
            return
        self._mark_raw("")
        if scope == "auto":
            self.formatRequested.emit({"raw": raw or None})
        else:
            self.scopedFormatRequested.emit({"raw": raw or None}, scope)

    def _mark_raw(self, err: str) -> None:
        self.raw.setStyleSheet("border:1px solid #cf222e; background:#fff0f0;" if err else "")
        self.raw.setToolTip(err or RAW_HELP)
        if err:
            from PySide6.QtWidgets import QToolTip

            QToolTip.showText(self.raw.mapToGlobal(self.raw.rect().bottomLeft()), err, self.raw)

    def current_colour(self) -> QColor:
        return qcolour(self._colour or "") or QColor("white")

    def _pick_colour(self) -> None:
        c = QColorDialog.getColor(qcolour(self._colour or "") or QColor("white"), self, "字的颜色")
        if c.isValid():
            self.formatRequested.emit({"c1": ass_colour(c)})

    def handle_key(self, e) -> bool:
        """Ctrl+B / Ctrl+I / Ctrl+Shift+> / Ctrl+Shift+< from a text field; True if handled."""
        mods = e.modifiers()
        if not mods & Qt.KeyboardModifier.ControlModifier:
            return False
        k = e.key()
        if k == Qt.Key.Key_B:
            self.btn_b.click()
            return True
        if k == Qt.Key.Key_I:
            self.btn_i.click()
            return True
        if mods & Qt.KeyboardModifier.ShiftModifier and k in (Qt.Key.Key_Greater, Qt.Key.Key_Period):
            self.step_size(+1)
            return True
        if mods & Qt.KeyboardModifier.ShiftModifier and k in (Qt.Key.Key_Less, Qt.Key.Key_Comma):
            self.step_size(-1)
            return True
        return False


def describe_target(text: str, sel: Optional[tuple[int, int]]) -> str:
    if sel is None or sel[1] <= sel[0]:
        return "整行"
    s = text[sel[0]:sel[1]]
    return f"选中「{s[:8]}{'…' if len(s) > 8 else ''}」"


def format_selection(text: str, sel: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
    """Selecting every visible character means the whole line, including line-only ASS tags."""
    if sel is None or sel[1] <= sel[0] or (sel[0] == 0 and sel[1] >= len(text)):
        return None
    return sel


StyleLookup = Callable[[str], Optional[object]]
