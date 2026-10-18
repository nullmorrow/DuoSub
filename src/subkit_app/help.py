"""The single user manual and explicit context help mappings."""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QUrl
from PySide6.QtGui import QCursor, QDesktopServices
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QHBoxLayout, QInputDialog, QMessageBox, QProgressDialog, QPushButton, QWidget

HELP_ANCHORS = frozenset({"transform", "watch", "karaoke", "karaoke-auto", "dialogue-align", "karaoke-fx", "readings", "template", "add-episode",
                          "passages", "oped-import", "lyrics-block", "export", "fonts", "timing", "format", "position", "review",
                          "gloss", "display-rules", "batch", "project", "project-readings", "keys", "changes", "autotime", "pinyin", "people", "review-answers", "hide-mark", "ai-transcribe", "save", "export-season", "video-links", "source-copies", "unify-times", "playback", "display-replacements", "review-packet"})


def manual_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "manual" / "index.html"


def manual_url(anchor: str | None = None) -> QUrl:
    url = QUrl.fromLocalFile(str(manual_path()))
    if anchor:
        url.setFragment(anchor)
    return url


def redirect_path() -> Path:
    return Path(tempfile.gettempdir()) / "duosub-help.html"


def open_manual(anchor: str | None = None) -> bool:
    if not manual_path().is_file():
        QMessageBox.warning(QApplication.activeWindow(), "双语字幕工坊",
                            f"找不到使用说明：{manual_path()}")
        return False
    if not anchor:
        return QDesktopServices.openUrl(manual_url())
    # Windows opens file URLs through the file association and drops the #fragment, so open a
    # small page without one that jumps to the section.
    target = manual_url(anchor).toString(QUrl.ComponentFormattingOption.FullyEncoded)
    page = redirect_path()
    page.write_text(f'<!doctype html><meta charset="utf-8"><title>双语字幕工坊 使用说明</title>'
                    f'<script>location.replace({json.dumps(target)})</script>'
                    f'<a href="{html.escape(target)}">打开使用说明</a>\n', encoding="utf-8")
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(page)))


def add_help_button(dialog: QWidget, container=None) -> QPushButton:
    """Help is an action, never an accept/reject operation."""
    if isinstance(container, QDialogButtonBox):
        button = container.addButton("帮助", QDialogButtonBox.ButtonRole.HelpRole)
    elif isinstance(dialog, QMessageBox):
        return dialog_buttons(dialog)
    else:
        button = QPushButton("帮助", dialog)
        (container or dialog.layout()).addWidget(button)
    button.setAutoDefault(False)
    button.clicked.connect(lambda: open_manual(dialog.help_anchor))
    return button


_STANDARD_TEXT = {"Ok": "确定", "Cancel": "取消", "Close": "关闭", "Help": "帮助",
                  "Save": "保存", "SaveAll": "全部保存", "Yes": "是", "YesToAll": "全部是",
                  "No": "否", "NoToAll": "全部否", "Abort": "中止", "Retry": "重试",
                  "Ignore": "忽略", "Apply": "应用", "Reset": "恢复默认", "Discard": "不保存"}


def _destructive(button, role=None):
    return (role == QDialogButtonBox.ButtonRole.DestructiveRole
            or any(word in button.text() for word in ("删除", "移除", "丢弃", "不保存")))


class _ButtonRow(QObject):
    """Keep the public box layout in our order when Qt relays it after show/resize.

    Buttons stay registered with their box: accepted/rejected and QMessageBox results
    retain their original meaning. Message-box help is deliberately not registered.
    """
    def __init__(self, dialog, container, help_button, primary, auxiliary, use_default):
        super().__init__(dialog)
        self.dialog, self.container, self.help_button = dialog, container, help_button
        self.primary, self.auxiliary, self.use_default = primary, auxiliary, use_default
        self.updating = False
        if isinstance(container, QWidget):
            container.installEventFilter(self)
        dialog.installEventFilter(self)
        self.arrange()

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() in (QEvent.Type.Show, QEvent.Type.Resize, QEvent.Type.LayoutRequest):
            self.arrange()
        return False

    def arrange(self):
        if self.updating:
            return
        self.updating = True
        try:
            box = self.container if isinstance(self.container, QDialogButtonBox) else None
            layout = box.layout() if box else self.container
            buttons = box.buttons() if box else [layout.itemAt(i).widget() for i in range(layout.count())
                                                 if isinstance(layout.itemAt(i).widget(), QPushButton)]
            buttons = [b for b in buttons if b is not self.help_button]
            if box:
                for name, text in _STANDARD_TEXT.items():
                    button = box.button(getattr(QDialogButtonBox.StandardButton, name))
                    if button is not None and button.text().replace('&', '').replace(' ', '').lower() in (name.lower(), text):
                        button.setText(text)
            role = lambda b: box.buttonRole(b) if box else None
            reject = [b for b in buttons if role(b) == QDialogButtonBox.ButtonRole.RejectRole
                      or b.text() in ("取消", "关闭")]
            auxiliary = [b for b in buttons if b not in reject and b is not self.primary
                         and (b in self.auxiliary or _destructive(b, role(b))
                              or (box and role(b) in (QDialogButtonBox.ButtonRole.ActionRole,
                                                     QDialogButtonBox.ButtonRole.ResetRole)))]
            auxiliary = [b for b in self.auxiliary if b in auxiliary] + [
                b for b in auxiliary if b not in self.auxiliary]
            main = [b for b in buttons if b not in reject and b not in auxiliary]
            ordered = auxiliary + [None] + main + reject + [self.help_button]
            current = [layout.itemAt(i).widget() for i in range(layout.count())]
            if current != ordered:
                while layout.count():
                    layout.takeAt(0)
                for button in ordered:
                    if button is None:
                        layout.addStretch()
                    else:
                        layout.addWidget(button)
            existing = (self.dialog.defaultButton() if isinstance(self.dialog, QMessageBox) else
                        next((b for b in buttons if b.isDefault()), None))
            if existing in auxiliary:
                existing = None
            default = self.primary or existing or (main[0] if main else reject[0] if reject else None)
            if default is not None and _destructive(default, role(default)):
                default = reject[0] if reject else None
            for button in buttons + [self.help_button]:
                button.setAutoDefault(False)
                button.setDefault(bool(self.use_default and button is default))
            if isinstance(self.dialog, QMessageBox):
                if default is not None and self.use_default:
                    self.dialog.setDefaultButton(default)
                if reject:
                    self.dialog.setEscapeButton(reject[-1])
        finally:
            self.updating = False


def dialog_buttons(dialog, container=None, *, primary=None, auxiliary=(), default=True):
    """Arrange auxiliary | stretch | primary | cancel/close | help without rewiring actions."""
    if container is None and not isinstance(dialog, QMessageBox):
        # QInputDialog builds its button box lazily; setting its caption creates it.
        if isinstance(dialog, QInputDialog):
            dialog.setOkButtonText("确定")
        dialog.ensurePolished()
        container = dialog.findChild(QDialogButtonBox)
    if isinstance(dialog, QMessageBox):
        if hasattr(dialog, "_button_row"):
            return dialog._button_row.help_button
        container = dialog.findChild(QDialogButtonBox)
        button = QPushButton("帮助", dialog)
        # Separate from QMessageBox's button registry, but inside its existing row.
        button.clicked.connect(lambda: open_manual(dialog.help_anchor))
    else:
        button = add_help_button(dialog, container)
    button.setAutoDefault(False)
    button.setDefault(False)
    dialog._button_row = _ButtonRow(dialog, container, button, primary, auxiliary, default)
    return button


def _anchor(widget) -> str | None:
    while widget is not None:
        anchor = getattr(widget, "help_anchor", None)
        if anchor:
            return anchor
        widget = widget.parentWidget()
    return None


def belongs_to(widget, main) -> bool:
    while widget is not None:
        if widget is main:
            return True
        widget = widget.parentWidget()
    return False


def resolve_anchor(main, active=None, focus=None) -> str | None:
    popup = QApplication.activePopupWidget()
    if popup is not None and hasattr(popup, "activeAction"):
        action = popup.activeAction()
        anchor = action.property("help_anchor") if action is not None else None
        if anchor or _anchor(popup):
            return anchor or _anchor(popup)
    active = active or QApplication.activeModalWidget() or QApplication.activeWindow()
    focus = focus or QApplication.focusWidget()
    if active is not None and active is not main:
        anchor = _anchor(active)
        if anchor:
            return anchor
    if main.watch.transform is not None:
        return "transform"
    song = bool(main.session and main.current_gid and main.session.doc.group(main.current_gid).kind == "song")
    if main.watch_mode:
        return "karaoke" if song else "watch"
    hovered = QApplication.widgetAt(QCursor.pos())
    if (hovered is not None and getattr(hovered, "copy_syntax_help", False)
            and belongs_to(hovered, main)):
        return hovered.help_anchor
    anchor = _anchor(focus)
    return "karaoke" if song and anchor == "timing" else anchor


def critical_message(parent, title, text):
    """Startup failures can precede the main window and its context-help filter."""
    box = QMessageBox(QMessageBox.Icon.Critical, title, text, QMessageBox.StandardButton.Ok, parent)
    box.help_anchor = "project"
    dialog_buttons(box)
    try:
        return box.exec()
    finally:
        box.deleteLater()


class ProgressDialog(QProgressDialog):
    """Qt positions progress controls itself; keep its cancel lifecycle and our row."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_row = QWidget(self)
        row = QHBoxLayout(self._progress_row)
        row.setContentsMargins(0, 0, 0, 0)
        self._cancel_button = self.findChild(QPushButton)
        row.addWidget(self._cancel_button)
        dialog_buttons(self, row)

    def _position_buttons(self):
        if not hasattr(self, "_button_row"):
            return
        row = self._progress_row
        height = row.sizeHint().height()
        row.setGeometry(11, self.height() - height - 11, self.width() - 22, height)
        row.layout().invalidate()
        row.layout().activate()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._position_buttons()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._position_buttons()

    def setLabelText(self, text):  # noqa: N802
        super().setLabelText(text)
        self._position_buttons()
