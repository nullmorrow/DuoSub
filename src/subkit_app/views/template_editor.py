"""Visual style template editor: presets to start from, every value adjustable, the frame rendered by libass.

Left: fonts, sizes, colours (presets or any colour from the picker), outlines and where every line sits (MarginV).
Right: the scene as the release will look — the translation, the original, the furigana and the pinyin at their
real positions, with guide lines at each MarginV — over a gradient, black, white, a picture or a frame of the video.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, QRunnable, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontDatabase, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from subkit import templates as T
from subkit.proc import CancelledError, KillableRun
from subkit.languages import JAPANESE_ONLY, has_japanese, lane_label
from .. import pool as job_pool

PREVIEW_W = 800
MEASURE_CACHE = Path(tempfile.gettempdir()) / "subkit_template_measure.sqlite3"
STANDARD = {"CN", "JP", "JP.Ruby", "CN.Top", "JP.Top", "JP.Top.Ruby", "CN.Alt", "JP.Alt", "JP.Alt.Ruby", "OP.CN", "OP.JP",
            "OP.JP.Ruby", "ED.CN", "ED.JP", "ED.JP.Ruby", "IN.CN", "IN.JP", "IN.JP.Ruby", "CN.Gloss", "JP.Gloss",
            "CN.Pinyin"}


# ------------------------------------------------------------------ small widgets
class ColorButton(QToolButton):
    """A colour swatch; click to pick any colour."""

    changed = Signal(str)          # RRGGBB

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self._rgb = "FFFFFF"
        self.setToolTip(f"{title}：点一下用调色板选颜色")
        self.setMinimumWidth(92)
        self.clicked.connect(self._pick)
        self._paint()

    def rgb(self) -> str:
        return self._rgb

    def set_rgb(self, rgb: str) -> None:
        self._rgb = (rgb or "FFFFFF").upper()
        self._paint()

    def _paint(self) -> None:
        c = QColor("#" + self._rgb)
        fg = "#000000" if c.lightness() > 140 else "#ffffff"
        self.setText(f"#{self._rgb}")
        self.setStyleSheet(f"QToolButton{{background:#{self._rgb};color:{fg};border:1px solid #8c959f;"
                           f"border-radius:3px;padding:2px 6px;font-family:Consolas}}")

    def _pick(self) -> None:
        c = QColorDialog.getColor(QColor("#" + self._rgb), self, self.title)
        if c.isValid():
            self.set_rgb(c.name()[1:])
            self.changed.emit(self._rgb)


def _spin(lo: float, hi: float, step: float = 1.0, decimals: int = 0, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setKeyboardTracking(False)
    s.setSuffix(suffix)
    s.setMinimumWidth(78)
    return s


def _font_box() -> QComboBox:
    b = QComboBox()
    b.setEditable(True)
    b.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    b.addItems(QFontDatabase.families())
    b.setMinimumWidth(220)
    b.setToolTip("字体名（libass 按这个名字找字体；中文名、英文名都可以）")
    return b


# ------------------------------------------------------------------ background work
class _Signals(QObject):
    done = Signal(int, object)       # ticket, result
    failed = Signal(int, str)


class _Job(QRunnable):
    def __init__(self, ticket: int, fn) -> None:
        super().__init__()
        self.ticket, self.fn = ticket, fn
        self.signals = _Signals()
        self.runner = KillableRun()
        self.cancelled = False
        self.finished = threading.Event()

    def cancel(self) -> None:
        self.cancelled = True
        self.runner.cancel()

    def run(self) -> None:  # worker thread
        try:
            self.runner.check_cancelled()
            result = self.fn(self.runner)
            if not self.cancelled:
                self.signals.done.emit(self.ticket, result)
        except CancelledError:
            pass
        except Exception as exc:
            if not self.cancelled:
                self.signals.failed.emit(self.ticket, str(exc))
        finally:
            self.finished.set()


class StyleImportDialog(QDialog):
    """Map source styles to dialogue lanes and choose which properties to read."""

    help_anchor = "template"

    def __init__(self, doc, parent=None, *, translation_lang: str, original_lang: str) -> None:
        super().__init__(parent)
        from subkit.ass.text import plain_text
        from subkit.core.batch import source_roles
        from subkit.sources.cn import classify
        from ..help import dialog_buttons

        self.setWindowTitle("从 .ass 读入样式")
        languages = {"translation": translation_lang, "original": original_lang}
        translation, original = (lane_label(lane, languages) for lane in ("translation", "original"))
        root = QVBoxLayout(self)
        tip = QLabel(f"这个文件不是 DuoSub 的模板。选择哪个样式用作{translation}、哪个用作{original}：")
        tip.setWordWrap(True)
        root.addWidget(tip)
        form = QFormLayout()
        self.translation, self.original = QComboBox(), QComboBox()
        counts = Counter(ev.style for ev in doc.events if ev.is_dialogue)
        samples = {}
        for ev in doc.dialogues():
            sample = " ".join(plain_text(ev.text).split())
            if sample:
                samples.setdefault(ev.style, sample[:48])
        for combo in (self.translation, self.original):
            combo.addItem("（不读入）", None)
            for name in doc.style_map():
                combo.addItem(f"{name}（{counts[name]} 行，例：{samples.get(name, '')}）", name)
            combo.setMinimumContentsLength(30)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        roles, jp = source_roles(doc, "5.2", strict=False)
        original_doc = doc.copy()
        original_doc.events = [ev for ev in doc.events if ev.is_dialogue and ev.style in jp]
        jp_roles = classify(original_doc, {}).style_roles
        plain_jp = [s for s in jp if jp_roles.get(s) == "dialogue"]
        default_jp = max(plain_jp or jp, key=lambda s: counts[s], default=None)
        default_cn = max(roles["dialogue"], key=lambda s: counts[s], default=None)
        for combo, name in ((self.translation, default_cn), (self.original, default_jp)):
            index = combo.findData(name)
            combo.setCurrentIndex(max(0, index))
        form.addRow(translation, self.translation)
        form.addRow(original, self.original)
        root.addLayout(form)
        self.fonts = QCheckBox("字体和字号")
        self.colours = QCheckBox("颜色和边框")
        self.margins = QCheckBox("距底边")
        self.fonts.setChecked(True)
        self.colours.setChecked(True)
        for check in (self.fonts, self.colours, self.margins):
            root.addWidget(check)
        note = QLabel("DuoSub 会按振假名自动排开中日两行，一般不用读")
        note.setWordWrap(True)
        root.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("读入")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        dialog_buttons(self, buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)


# ------------------------------------------------------------------ the editor
class TemplateEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None, *, translation_lang: str = "zh-Hant", original_lang: str = "ja", scheme: Optional[dict] = None,
                 current=None, video: Optional[Path] = None) -> None:
        """``scheme``: what project.json remembers; ``current``: the project's template (AssDocument) — read back
        when there is no scheme; ``video``: an episode video, offered as the preview background."""
        super().__init__(parent)
        self.lang = translation_lang
        self.original_lang = original_lang
        self.video = video
        self.extra_styles: list[str] = []          # styles of a loaded .ass file that are not the tool's own
        if scheme and scheme.get("fonts") is not None and not scheme.get("file"):
            self.fonts, self.pal = T.scheme_from_dict(scheme)
        elif current is not None:
            self.fonts, self.pal = T.scheme_from_styles(current)
            self.extra_styles = [s.to_line(current.style_format) for s in current.styles if s.name not in STANDARD]
        else:
            self.fonts, self.pal = T.font_set("utena_serif"), T.palette("utena")
        self._loading = False
        self._ticket = 0
        self._running = False
        self._closed = False
        self._dirty = False
        self._queue: list[tuple] = []             # jobs wait for each other (they share the measurement cache)
        self._pixmap_cache: dict[str, tuple[QPixmap, str]] = {}
        self._bg_frame: Optional[Path] = None
        self._bg_playres = (1920, 1080)
        self._build()
        if isinstance(self.window(), QDialog):
            self.window().finished.connect(self.cancel_jobs)
        self._label_bindings = []
        for widget in self.findChildren(QWidget):
            if isinstance(widget, QLabel):
                self._label_bindings.append((widget.setText, widget.text()))
            elif isinstance(widget, QGroupBox):
                self._label_bindings.append((widget.setTitle, widget.title()))
            if widget.toolTip():
                self._label_bindings.append((widget.setToolTip, widget.toolTip()))
            if isinstance(widget, ColorButton):
                self._label_bindings.append((lambda value, w=widget: setattr(w, "title", value), widget.title))
        self._label_bindings = [(setter, text) for setter, text in self._label_bindings
                                if "{original}" in text or "{translation}" in text]
        self._update_language_labels()
        self._load_controls()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(220)
        self._timer.timeout.connect(self._render)
        self._background(self.bg_box.currentIndex())

    # ------------------------------------------------------------ layout
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.pal_box = QComboBox()
        for p in T.PALETTES:
            self.pal_box.addItem(p.name, p.id)
        self.pal_box.addItem("自定义（当前的颜色）", "custom")
        self.font_box_set = QComboBox()
        for f in T.FONT_SETS:
            self.font_box_set.addItem(f.name, f.id)
        self.font_box_set.addItem("自定义（当前的字体和位置）", "custom")
        top.addWidget(QLabel("配色预设"))
        top.addWidget(self.pal_box, 2)
        top.addSpacing(8)
        top.addWidget(QLabel("字体预设"))
        top.addWidget(self.font_box_set, 3)
        self.btn_file = QPushButton("从 .ass 文件读取…")
        self.btn_file.setToolTip("读入 DuoSub 模板，或选择字幕组的对白样式，只读入勾选的字体、颜色和位置")
        top.addWidget(self.btn_file)
        root.addLayout(top)
        body = QHBoxLayout()
        root.addLayout(body, 1)
        # ---- left: values
        form_host = QWidget()
        fl = QVBoxLayout(form_host)
        fl.setContentsMargins(0, 0, 6, 0)
        self.w: dict[str, Any] = {}

        def group(title: str) -> QFormLayout:
            g = QGroupBox(title)
            f = QFormLayout(g)
            f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            fl.addWidget(g)
            return f

        def row(*widgets) -> QWidget:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            for x in widgets:
                h.addWidget(x) if not isinstance(x, str) else h.addWidget(QLabel(x))
            h.addStretch(1)
            return w

        W = self.w
        f = group("{translation}")
        W["cn_font"] = _font_box()
        W["cn_size"] = _spin(8, 200, 1, 1)
        W["cn_bold"] = QCheckBox("粗体")
        W["spacing"] = _spin(-10, 30, 0.5, 1)
        W["cn_text"] = ColorButton("{translation}文字颜色")
        W["cn_outline"] = ColorButton("{translation}边框颜色")
        W["width"] = _spin(0, 12, 0.1, 1)
        W["outline_alpha"] = QSpinBox()
        W["outline_alpha"].setRange(0, 100)
        W["outline_alpha"].setSuffix(" %")
        W["outline_alpha"].setToolTip("边框透明度（少女革命成品是 8%）")
        W["cn_mv"] = _spin(0, 1080, 1)
        f.addRow("字体", W["cn_font"])
        f.addRow("字号", row(W["cn_size"], W["cn_bold"], "字距", W["spacing"]))
        f.addRow("颜色", row("文字", W["cn_text"], "边框", W["cn_outline"]))
        f.addRow("边框", row("宽度", W["width"], "透明", W["outline_alpha"]))
        f.addRow("距底边", W["cn_mv"])
        f = group("{original}")
        W["jp_font"] = _font_box()
        W["jp_size"] = _spin(8, 200, 1, 1)
        W["jp_bold"] = QCheckBox("粗体")
        W["jp_text"] = ColorButton("{original}文字颜色")
        W["jp_outline"] = ColorButton("{original}边框颜色")
        W["jp_width"] = _spin(0, 12, 0.1, 1)
        W["jp_mv"] = _spin(0, 1080, 1)
        f.addRow("字体", W["jp_font"])
        f.addRow("字号", row(W["jp_size"], W["jp_bold"]))
        f.addRow("颜色", row("文字", W["jp_text"], "边框", W["jp_outline"]))
        f.addRow("边框宽度", W["jp_width"])
        f.addRow("距底边", W["jp_mv"])
        f = group("振假名 / 拼音")
        W["ruby_size"] = _spin(4, 120, 0.5, 1)
        W["pinyin_size"] = _spin(4, 120, 0.5, 1)
        W["pinyin_font"] = _font_box()
        for key in ("ruby_offset", "pinyin_offset"):
            W[key] = _spin(-40, 40, 0.5, 1)
            W[key].setSuffix(" px（1080p）")
            W[key].setToolTip("正数离正文更远，负数更近")
        f.addRow("振假名字号", W["ruby_size"])
        f.addRow("拼音字号", W["pinyin_size"])
        f.addRow("拼音字体", W["pinyin_font"])
        f.addRow("振假名上下微调", W["ruby_offset"])
        f.addRow("拼音上下微调", W["pinyin_offset"])
        f = group("第二个人同时说话（显示在上面）")
        W["cn_alt_mv"] = _spin(0, 1080, 1)
        W["jp_alt_mv"] = _spin(0, 1080, 1)
        W["alt_inverted"] = QCheckBox("字和边的颜色互换（更容易分出是另一个人）")
        f.addRow("{translation}距底边", W["cn_alt_mv"])
        f.addRow("{original}距底边", W["jp_alt_mv"])
        f.addRow("", W["alt_inverted"])
        f = group("屏幕顶部（底部有画面文字时）")
        W["cn_top_mv"] = _spin(0, 1080, 1)
        W["jp_top_mv"] = _spin(0, 1080, 1)
        W["top_inverted"] = QCheckBox("字和边的颜色互换")
        W["top_inverted"].setToolTip("只管屏幕顶部（CN.Top / JP.Top）；第二个人的颜色互换在上面单独设")
        f.addRow("{translation}距顶边", W["cn_top_mv"])
        f.addRow("{original}距顶边", W["jp_top_mv"])
        f.addRow("", W["top_inverted"])
        f = group("歌词（OP / ED / 插曲）")
        W["song_cn_size"] = _spin(8, 240, 1, 1)
        W["song_jp_size"] = _spin(8, 240, 1, 1)
        W["song_cn_mv"] = _spin(0, 1080, 1)
        W["song_jp_mv"] = _spin(0, 1080, 1)
        W["song_text"] = ColorButton("歌词：唱到的颜色")
        W["song_before"] = ColorButton("歌词：还没唱到的颜色")
        W["song_outline"] = ColorButton("歌词边框颜色")
        f.addRow("字号", row("{translation}", W["song_cn_size"], "{original}", W["song_jp_size"]))
        f.addRow("距底边", row("{translation}", W["song_cn_mv"], "{original}", W["song_jp_mv"]))
        f.addRow("颜色", row("唱到", W["song_text"], "未唱", W["song_before"]))
        f.addRow("边框", W["song_outline"])
        f = group("阴影")
        W["shadow"] = ColorButton("阴影颜色")
        W["shadow_depth"] = _spin(0, 20, 0.5, 1)
        W["shadow_alpha"] = QSpinBox()
        W["shadow_alpha"].setRange(0, 100)
        W["shadow_alpha"].setSuffix(" %")
        f.addRow("颜色", row(W["shadow"], "距离", W["shadow_depth"], "透明", W["shadow_alpha"]))
        tools = QHBoxLayout()
        self.btn_fit = QPushButton("按发布版的间距自动排位置")
        self.btn_fit.setToolTip("用 libass 测量后，把{translation}、第二个人、顶部的位置挪到和你发布过的版本一样的间距"
                                "（{translation}和振假名之间约 18 像素）；{original}底边不动")
        self.btn_reset = QPushButton("恢复预设")
        tools.addWidget(self.btn_fit)
        tools.addWidget(self.btn_reset)
        tools.addStretch(1)
        fl.addLayout(tools)
        fl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(form_host)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(470)
        body.addWidget(scroll, 0)
        # ---- right: preview
        right = QVBoxLayout()
        sc = QHBoxLayout()
        self.scene_group = QButtonGroup(self)
        for k, (key, label) in enumerate(T.SCENES.items()):
            b = QToolButton(text=label)
            b.setCheckable(True)
            b.setChecked(k == 0)
            b.setProperty("scene", key)
            self.scene_group.addButton(b, k)
            sc.addWidget(b)
        sc.addStretch(1)
        right.addLayout(sc)
        opts = QHBoxLayout()
        self.chk_guides = QCheckBox("参考线（每行的位置）")
        self.chk_guides.setChecked(True)
        self.chk_pinyin = QCheckBox("拼音")
        self.chk_pinyin.setToolTip("在中文上方显示拼音的位置（需要时才用）")
        self.chk_zoom = QCheckBox("放大字幕区")
        self.bg_box = QComboBox()
        for key, label in (("dark_gradient", "背景：深色渐变（接近画面）"), ("gradient", "背景：浅灰渐变"),
                           ("black", "背景：黑"), ("white", "背景：白")):
            self.bg_box.addItem(label, key)
        if self.video is not None:
            self.bg_box.addItem("背景：这一集的视频画面", "video")
            self.bg_box.setCurrentIndex(self.bg_box.findData("video"))
        self.bg_box.addItem("背景：选图片或视频…", "pick")
        for w in (self.chk_guides, self.chk_pinyin, self.chk_zoom, self.bg_box):
            opts.addWidget(w)
        opts.addStretch(1)
        right.addLayout(opts)
        self.preview = QLabel("正在生成预览…")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(PREVIEW_W, PREVIEW_W * 9 // 16)
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setStyleSheet("background:#1f2328; color:#d0d7de;")
        right.addWidget(self.preview, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#57606a;")
        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color:#9a6700;")
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#1a7f37;")
        right.addWidget(self.status)
        right.addWidget(self.info)
        right.addWidget(self.warn)
        body.addLayout(right, 1)
        # ---- wiring
        self.pal_box.activated.connect(self._pal_preset)
        self.font_box_set.activated.connect(self._font_preset)
        self.btn_file.clicked.connect(self._from_file)
        self.btn_fit.clicked.connect(self._fit)
        self.btn_reset.clicked.connect(self._reset)
        self.scene_group.idClicked.connect(lambda _i: self._schedule())
        for c in (self.chk_guides, self.chk_pinyin, self.chk_zoom):
            c.toggled.connect(lambda _v: self._schedule())
        self.bg_box.activated.connect(self._background)
        for k, w in self.w.items():
            if isinstance(w, ColorButton):
                w.changed.connect(lambda _v: self._edited())
            elif isinstance(w, QCheckBox):
                w.toggled.connect(lambda _v: self._edited())
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(lambda _t: self._edited())
            else:
                w.valueChanged.connect(lambda _v: self._edited())

    # ------------------------------------------------------------ values ↔ controls
    def _load_controls(self) -> None:
        f, p, W = self.fonts, self.pal, self.w
        self._loading = True
        try:
            W["cn_font"].setCurrentText(f.cn_font(self.lang))
            W["cn_size"].setValue(f.cn_size)
            W["cn_bold"].setChecked(f.bold if f.cn_bold is None else f.cn_bold)
            W["spacing"].setValue(f.spacing)
            W["cn_text"].set_rgb(p.text)
            W["cn_outline"].set_rgb(p.outline)
            W["width"].setValue(p.width)
            W["outline_alpha"].setValue(round(p.outline_alpha * 100 / 255))
            W["cn_mv"].setValue(f.cn_mv)
            W["jp_font"].setCurrentText(f.jp)
            W["jp_size"].setValue(f.jp_size)
            W["jp_bold"].setChecked(f.bold if f.jp_bold is None else f.jp_bold)
            W["jp_text"].set_rgb(p.jp_fill)
            W["jp_outline"].set_rgb(p.jp_edge)
            W["jp_width"].setValue(round(f.jp_outline_width if not p.width and f.jp_outline_width is not None
                                        else p.width * f.jp_outline_ratio, 1))
            W["jp_mv"].setValue(f.jp_mv)
            W["ruby_size"].setValue(round(f.jp_size * f.ruby_ratio, 1))
            W["pinyin_size"].setValue(round(f.cn_size * f.pinyin_ratio, 1))
            W["pinyin_font"].setCurrentText(f.pinyin_font)
            W["ruby_offset"].setValue(f.ruby_offset)
            W["pinyin_offset"].setValue(f.pinyin_offset)
            W["cn_alt_mv"].setValue(f.cn_alt_mv)
            W["jp_alt_mv"].setValue(f.jp_alt_mv)
            W["alt_inverted"].setChecked(p.alt_inverted)
            W["top_inverted"].setChecked(p.alt_inverted if p.top_inverted is None else p.top_inverted)
            W["cn_top_mv"].setValue(f.cn_top_mv)
            W["jp_top_mv"].setValue(f.jp_top_mv)
            W["song_cn_size"].setValue(round(f.cn_size * f.song_cn_ratio))
            W["song_jp_size"].setValue(round(f.jp_size * f.song_jp_ratio))
            W["song_cn_mv"].setValue(f.cn_mv + 20 if f.song_cn_mv is None else f.song_cn_mv)
            W["song_jp_mv"].setValue(f.jp_mv if f.song_jp_mv is None else f.song_jp_mv)
            W["song_text"].set_rgb(p.song_text or p.text)
            W["song_before"].set_rgb(p.song_before)
            W["song_outline"].set_rgb(p.song_outline or p.outline)
            W["shadow"].set_rgb(p.shadow)
            W["shadow_depth"].setValue(p.shadow_depth)
            W["shadow_alpha"].setValue(round(p.shadow_alpha * 100 / 255))
            self._sync_presets()
        finally:
            self._loading = False

    def _read_controls(self) -> None:
        W, f, p = self.w, self.fonts, self.pal
        cn_size, jp_size = W["cn_size"].value(), W["jp_size"].value()
        width = W["width"].value()
        cn_font = W["cn_font"].currentText().strip() or f.cn_font(self.lang)
        trad = T.is_traditional(self.lang)
        self.fonts = dataclasses.replace(
            f, id="custom", name="自定义", cn_hant=cn_font if trad else f.cn_hant, cn_hans=cn_font if not trad else f.cn_hans,
            jp=W["jp_font"].currentText().strip() or f.jp, cn_size=cn_size, jp_size=jp_size,
            cn_bold=W["cn_bold"].isChecked(), jp_bold=W["jp_bold"].isChecked(), spacing=W["spacing"].value(),
            jp_outline_ratio=(W["jp_width"].value() / width) if width else f.jp_outline_ratio,
            jp_outline_width=W["jp_width"].value() if not width else None,
            cn_mv=W["cn_mv"].value(), jp_mv=W["jp_mv"].value(), cn_alt_mv=W["cn_alt_mv"].value(),
            jp_alt_mv=W["jp_alt_mv"].value(), cn_top_mv=W["cn_top_mv"].value(), jp_top_mv=W["jp_top_mv"].value(),
            ruby_ratio=W["ruby_size"].value() / jp_size if jp_size else 0.5,
            pinyin_ratio=W["pinyin_size"].value() / cn_size if cn_size else 0.38,
            pinyin_font=W["pinyin_font"].currentText().strip() or f.pinyin_font,
            ruby_offset=W["ruby_offset"].value(), pinyin_offset=W["pinyin_offset"].value(),
            song_cn_ratio=W["song_cn_size"].value() / cn_size if cn_size else 1.18,
            song_jp_ratio=W["song_jp_size"].value() / jp_size if jp_size else 1.4,
            song_cn_mv=W["song_cn_mv"].value(), song_jp_mv=W["song_jp_mv"].value())
        text, outline = W["cn_text"].rgb(), W["cn_outline"].rgb()
        self.pal = dataclasses.replace(
            p, id="custom", name="自定义", text=text, outline=outline,
            jp_text=None if W["jp_text"].rgb() == text else W["jp_text"].rgb(),
            jp_outline=None if W["jp_outline"].rgb() == outline else W["jp_outline"].rgb(), width=width,
            outline_alpha=round(W["outline_alpha"].value() * 255 / 100), alt_inverted=W["alt_inverted"].isChecked(),
            top_inverted=W["top_inverted"].isChecked(),
            song_text=None if W["song_text"].rgb() == text else W["song_text"].rgb(), song_before=W["song_before"].rgb(),
            song_outline=None if W["song_outline"].rgb() == outline else W["song_outline"].rgb(),
            shadow=W["shadow"].rgb(), shadow_depth=W["shadow_depth"].value(),
            shadow_alpha=round(W["shadow_alpha"].value() * 255 / 100))

    def _sync_presets(self) -> None:
        """Show which preset the current values are (or 自定义)."""
        styles = T.build_styles(self.fonts, self.pal, translation_lang=self.lang)
        pid = next((p.id for p in T.PALETTES
                    if T.build_styles(self.fonts, p, translation_lang=self.lang) == styles), "custom")
        fid = next((x.id for x in T.FONT_SETS
                    if T.build_styles(x, self.pal, translation_lang=self.lang) == styles), "custom")
        self.pal_box.setCurrentIndex(self.pal_box.findData(pid))
        self.font_box_set.setCurrentIndex(self.font_box_set.findData(fid))

    def _edited(self) -> None:
        if self._loading:
            return
        self.info.setText("")
        self._read_controls()
        self._loading = True
        try:
            self._sync_presets()
        finally:
            self._loading = False
        self._schedule()
        self.changed.emit()

    def _pal_preset(self, _i: int) -> None:
        pid = self.pal_box.currentData()
        if pid != "custom":
            self.pal = T.palette(pid)
            self._load_controls()
            self._schedule()
            self.changed.emit()

    def _font_preset(self, _i: int) -> None:
        fid = self.font_box_set.currentData()
        if fid != "custom":
            self.fonts = dataclasses.replace(T.font_set(fid), ruby_offset=self.fonts.ruby_offset,
                                             pinyin_offset=self.fonts.pinyin_offset)
            self._load_controls()
            self._schedule()
            self.changed.emit()

    def _reset(self) -> None:
        self.fonts = dataclasses.replace(T.font_set("utena_serif"), ruby_offset=self.fonts.ruby_offset,
                                         pinyin_offset=self.fonts.pinyin_offset)
        self.pal = T.palette("utena")
        self.extra_styles = []
        self._load_controls()
        self._schedule()
        self.changed.emit()

    def _from_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "读入 .ass 样式", "", "ASS (*.ass)")
        if not f:
            return
        from subkit.ass import AssDocument

        try:
            doc = AssDocument.load(f)
        except Exception as exc:
            self.warn.setText(f"读不了这个文件：{exc}")
            return
        if doc.style("CN") is None or doc.style("JP") is None:
            dialog = StyleImportDialog(doc, self, translation_lang=self.lang, original_lang=self.original_lang)
            try:
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    return
                translation, original = dialog.translation.currentData(), dialog.original.currentData()
                parts = [check.text() for check in (dialog.fonts, dialog.colours, dialog.margins) if check.isChecked()]
                if not parts or translation is None and original is None:
                    return
                self.fonts, self.pal = T.import_dialogue_styles(
                    self.fonts, self.pal, doc, translation, original, read_fonts=dialog.fonts.isChecked(),
                    read_colours=dialog.colours.isChecked(), read_margins=dialog.margins.isChecked())
            finally:
                dialog.deleteLater()
            self._load_controls()
            self._schedule()
            self.warn.setText("")
            languages = {"translation": self.lang, "original": self.original_lang}
            choices = [f"{lane_label(lane, languages)} ← {style}" for lane, style in
                       (("translation", translation), ("original", original)) if style is not None]
            self.info.setText(f"已从 {Path(f).name} 读入：{'，'.join(choices)}（{'、'.join(parts)}）")
            self.changed.emit()
            return
        fonts, self.pal = T.scheme_from_styles(doc)
        self.fonts = dataclasses.replace(fonts, ruby_offset=self.fonts.ruby_offset, pinyin_offset=self.fonts.pinyin_offset)
        self.extra_styles = [s.to_line(doc.style_format) for s in doc.styles if s.name not in STANDARD]
        self._load_controls()
        self._schedule()
        self.info.setText(f"已读入 {Path(f).name}" + (f"（另外保留 {len(self.extra_styles)} 个你自己的样式）"
                                                   if self.extra_styles else ""))
        self.changed.emit()

    # ------------------------------------------------------------ result
    def _update_language_labels(self) -> None:
        languages = {"original": self.original_lang, "translation": self.lang}
        for setter, text in self._label_bindings:
            setter(text.replace("{original}", lane_label("original", languages))
                   .replace("{translation}", lane_label("translation", languages)))
        japanese = has_japanese(languages)
        self.w["ruby_size"].setEnabled(japanese)
        self.w["ruby_size"].setToolTip("" if japanese else JAPANESE_ONLY)
        self.w["ruby_offset"].setEnabled(japanese)

    def set_original_lang(self, lang: str) -> None:
        if lang != self.original_lang:
            self.original_lang = lang
            self._update_language_labels()
            self._schedule()

    def set_translation_lang(self, lang: str) -> None:
        if lang != self.lang:
            self.lang = lang
            self._update_language_labels()
            self._loading = True
            try:
                self.w["cn_font"].setCurrentText(self.fonts.cn_font(lang))
            finally:
                self._loading = False
            self._schedule()

    def styles_text(self) -> str:
        text = T.build_styles(self.fonts, self.pal, translation_lang=self.lang)
        if self.extra_styles:
            text += "\n".join(self.extra_styles) + "\n"
        return text

    def scheme(self) -> dict:
        return T.scheme_dict(self.fonts, self.pal)

    # ------------------------------------------------------------ preview
    def _scene(self) -> str:
        b = self.scene_group.checkedButton()
        return b.property("scene") if b is not None else "dialogue"

    def _schedule(self) -> None:
        if not self._closed:
            self._timer.start()

    def _background(self, _i: int) -> None:
        key = self.bg_box.currentData()
        if key == "pick":
            f, _ = QFileDialog.getOpenFileName(self, "背景图片或视频", "", "图片或视频 (*.png *.jpg *.jpeg *.bmp *.mkv *.mp4 *.avi)")
            if f:
                self._grab_frame(Path(f))
            else:
                self.bg_box.setCurrentIndex(0)
            return
        if key == "video" and self.video is not None and self._bg_frame is None:
            self._grab_frame(self.video)
            return
        self._schedule()

    def _grab_frame(self, path: Path) -> None:
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
            self._bg_frame = path
            self._schedule()
            return
        self.status.setText("正在取视频画面…")

        def work(runner):
            run = runner.run

            playres = T.preview_playres(path, runner)
            out = Path(tempfile.gettempdir()) / f"subkit_tplbg_{uuid.uuid4().hex[:8]}.png"
            dur = 0.0
            try:
                r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                        capture_output=True, text=True, timeout=30)
                dur = float((r.stdout or "0").strip() or 0)
            except CancelledError:
                raise
            except Exception:
                pass
            t = max(1.0, dur * 0.3) if dur else 60.0
            run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.1f}", "-i", str(path), "-frames:v", "1",
                 "-vf", f"scale={playres[0]}:{playres[1]},setsar=1", str(out)],
                capture_output=True, timeout=60)
            if not out.exists():
                raise RuntimeError("没取到画面")
            return out, playres

        self._run_job(work, self._frame_ready)

    def _frame_ready(self, result) -> None:
        png, self._bg_playres = result
        self._bg_frame = png
        self.status.setText("")
        self._schedule()

    def _bg(self) -> Any:
        key = self.bg_box.currentData()
        if key in ("video", "pick") and self._bg_frame is not None:
            return self._bg_frame
        return key if key in T.BACKGROUNDS else "dark_gradient"

    def _render(self) -> None:
        if self._closed:
            return
        if self._running:
            self._dirty = True
            return
        styles = self.styles_text()
        scene = self._scene()
        region = ("top" if scene == "top" else "bottom") if self.chk_zoom.isChecked() else "full"
        args = dict(scene=scene, width=PREVIEW_W if region == "full" else 960,
                    original_lang=self.original_lang, translation_lang=self.lang,
                    pinyin=self.chk_pinyin.isChecked(), guides=self.chk_guides.isChecked(), background=self._bg(),
                    region=region, measure_cache=MEASURE_CACHE,
                    playres=self._bg_playres if self.bg_box.currentData() == "video" else None)
        args.update(ruby_offset=self.fonts.ruby_offset, pinyin_offset=self.fonts.pinyin_offset)
        png = Path(tempfile.gettempdir()) / f"subkit_tplprev_{uuid.uuid4().hex[:10]}.png"
        self.status.setText("正在更新预览…")
        self._run_job(lambda runner: T.render_scene(styles, png, runner=runner, **args), self._preview_ready)

    def _preview_ready(self, res) -> None:
        pm = QPixmap(str(res.png))
        try:
            os.remove(res.png)
        except OSError:
            pass
        self._last_pm = pm
        self._display_preview()
        self.status.setText("")
        self.warn.setText(res.warning())

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._display_preview()

    def _display_preview(self) -> None:
        pm = getattr(self, "_last_pm", None)
        if pm is None:
            return
        canvas = QPixmap(self.preview.size())
        key = self._bg()
        canvas.fill(QColor("white" if key == "white" else "black"))
        painter = QPainter(canvas)
        if key in ("gradient", "dark_gradient"):
            gradient = QLinearGradient(0, 0, canvas.width(), canvas.height())
            colours = ("#2a3442", "#9aa6b6") if key == "gradient" else ("#18202c", "#48566a")
            gradient.setColorAt(0, QColor(colours[0]))
            gradient.setColorAt(1, QColor(colours[1]))
            painter.fillRect(canvas.rect(), gradient)
        elif isinstance(key, Path) and self.chk_zoom.isChecked():
            bg = QPixmap(str(key)).scaled(canvas.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                         Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap((canvas.width() - bg.width()) // 2, (canvas.height() - bg.height()) // 2, bg)
        fitted = pm.scaled(canvas.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        painter.drawPixmap((canvas.width() - fitted.width()) // 2, (canvas.height() - fitted.height()) // 2, fitted)
        painter.end()
        self.preview.setPixmap(canvas)

    def _fit(self) -> None:
        self.status.setText("正在测量位置…")
        self.btn_fit.setEnabled(False)
        fonts, pal, lang, py = self.fonts, self.pal, self.lang, self.chk_pinyin.isChecked()
        self._run_job(lambda runner: T.fit_layout(fonts, pal, translation_lang=lang, pinyin=py,
                                                 measure_cache=MEASURE_CACHE, runner=runner),
                      self._fitted)

    def _fitted(self, res) -> None:
        self.btn_fit.setEnabled(True)
        fonts, c = res
        self.fonts = dataclasses.replace(self.fonts, cn_mv=fonts.cn_mv, cn_alt_mv=fonts.cn_alt_mv, jp_alt_mv=fonts.jp_alt_mv,
                                         cn_top_mv=fonts.cn_top_mv, jp_top_mv=fonts.jp_top_mv)
        self._load_controls()
        translation = lane_label("translation", {"translation": self.lang})
        self.info.setText(f"已按发布版间距排好：{translation}和振假名之间 {c['cn_ruby']} 像素，第二个人和下面一组之间 {c['alt_gap']} 像素，"
                          f"顶部离边 {c['top_edge']} 像素" + ("（算上了拼音的高度）" if c.get("pinyin") else ""))
        self._schedule()
        self.changed.emit()

    def _run_job(self, fn, on_done) -> None:
        if self._closed:
            return
        if self._running:
            self._queue.append((fn, on_done))
            return
        self._ticket += 1
        self._running = True
        job = _Job(self._ticket, fn)
        job.setAutoDelete(False)
        # bound methods of this widget: the call is delivered in its thread (a lambda would be tied to the job's
        # signal object, and dropped when that goes away first)
        job.signals.done.connect(self._on_job_done)
        job.signals.failed.connect(self._on_job_failed)
        self._jobs = getattr(self, "_jobs", {})
        self._jobs[self._ticket] = (job, on_done)
        job_pool.start(job)

    def _on_job_done(self, ticket: int, res) -> None:
        if self._closed:
            return
        _job, on_done = self._jobs.pop(ticket, (None, None))
        self._job_end(lambda: on_done(res) if on_done else None)

    def _on_job_failed(self, ticket: int, err: str) -> None:
        if self._closed:
            return
        self._jobs.pop(ticket, None)
        self._job_end(lambda: self._job_failed(err))

    def _job_failed(self, err: str) -> None:
        self.btn_fit.setEnabled(True)
        self.status.setText("")
        self.warn.setText(err)

    def _job_end(self, then) -> None:
        self._running = False
        try:
            then()
        finally:
            if self._queue:
                fn, on_done = self._queue.pop(0)
                self._run_job(fn, on_done)
            elif self._dirty:
                self._dirty = False
                self._schedule()


    def cancel_jobs(self) -> None:
        self._closed = True
        self._timer.stop()
        self._queue.clear()
        jobs = [job for job, _callback in getattr(self, "_jobs", {}).values()]
        deadline = time.monotonic() + 2.0
        for job in jobs:
            job.cancel()
            if job_pool.try_take(job):
                job.finished.set()
        for job in jobs:
            job.finished.wait(max(0.0, deadline - time.monotonic()))
        self._jobs = {}

    def closeEvent(self, e) -> None:  # noqa: N802
        self.cancel_jobs()
        super().closeEvent(e)


class TemplateDialog(QDialog):
    help_anchor = "template"
    """An open project's styles (a new template version; the old one stays in templates/)."""

    def __init__(self, parent=None, *, translation_lang: str = "zh-Hant", original_lang: str = "ja", scheme: Optional[dict] = None, current=None,
                 video: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("样式模板：字体、颜色和位置")
        self.resize(1360, 820)
        v = QVBoxLayout(self)
        self.editor = TemplateEditor(self, translation_lang=translation_lang, original_lang=original_lang, scheme=scheme, current=current, video=video)
        v.addWidget(self.editor, 1)
        tip = QLabel("保存成新版本的模板（旧模板文件留在工程的 templates 文件夹）；你自己加的其他样式（比如手写体）会原样保留。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#57606a;")
        v.addWidget(tip)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("用这个模板")
        from ..help import dialog_buttons

        dialog_buttons(self, bb)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
