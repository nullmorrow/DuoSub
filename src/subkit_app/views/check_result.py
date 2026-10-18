"""Readable results; internal provenance stays in the model."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLayout, QPlainTextEdit, QPushButton, QToolButton, QVBoxLayout
from subkit.ass.text import plain_text
from subkit.languages import lane_label
from subkit.core.merge import group_counts
from ..help import dialog_buttons
from .group_table import fmt_time


def report_lines(report, sources, languages):
    lines = ["每个字都有着落："]
    names = (("paired", "配进组"), ("unpaired", "没配上（单独成行）"), ("comment", "备注"),
             ("metadata", "说话人标签和音效等标记"), ("deleted", "已删除"))
    for sid, counts in report.ledger.items():
        source = sources.get(sid)
        if source is None or source.lane not in ("original", "translation"):
            continue
        total = sum(not ch.isspace() for event in source.events for ch in event.text)
        parts = [f"{name} {counts.get(key, 0)}" for key, name in names if counts.get(key, 0)]
        lines.append(f"{lane_label(source.lane, languages)}字幕 {total} 字：" + "，".join(parts) + "。")
    errors = report.errors
    order = sum(issue.code == "order" for issue in errors)
    ledger = sum(issue.code.startswith("ledger.") for issue in errors)
    lines.append(f"检查：发现 {len(errors)} 个问题（顺序 {order}，原文去向 {ledger}，其他 {len(errors) - order - ledger}）。“工具 → 检查 → 顺序、原文去向、待办”可以看详情。"
                 if errors else "检查：没有问题。")
    return lines


def issue_lines(report, doc, sources):
    categories = {"order": "顺序", "ledger.missing": "没有着落", "ledger.duplicate": "重复"}
    lines = []
    for issue in report.issues:
        if issue.code == "gloss.syntax":
            lines.append(issue.message)
            continue
        if issue.level != "error":
            continue
        where = issue.where
        member = doc.members.get(where.get("member"))
        comment = doc.comments.get(where.get("comment"))
        gid = where.get("group") or (member.group if member else comment.group if comment else None)
        group = next((g for g in doc.groups if g.id == gid), None)
        source = sources.get(where.get("source"))
        event = where.get("event")
        if source is not None and isinstance(event, int) and 0 <= event < len(source.events):
            start, text = source.events[event].start, source.events[event].text
        elif group:
            members = [doc.members[mid] for mid in group.members if mid in doc.members]
            start = min((m.start for m in members), default=0)
            text = " / ".join(m.text for m in members)
        elif member:
            start, text = member.start, member.text
        else:
            start, text = 0, "无法定位台词"
        lines.append(f"{fmt_time(start)}　{categories.get(issue.code, '内部数据（请把检查结果发给开发者）')}："
                     + plain_text(text, breaks=" ")[:24])
    return lines


def reanalyze_text(result, sources, languages):
    return "\n".join(f"已重新分析{lane_label(sources[sid].lane, languages)}字幕：说话人标签和音效等标记 {a} → {b} 处。"
                     for sid, (a, b) in result.items() if sid in sources)


class _ResultText(QLabel):
    """A short result shown as plain text that the dialog sizes to (same reading API as the text box)."""

    def toPlainText(self) -> str:  # noqa: N802
        return self.text()


class CheckResultDialog(QDialog):
    help_anchor = "add-episode"
    singleLanguageRequested = Signal()

    def __init__(self, episode, report, parent=None, *, merged=False):
        super().__init__(parent)
        self.setWindowTitle("合并完成" if merged else "检查")
        if not merged:
            self.resize(850, 560)
        layout = QVBoxLayout(self)
        doc, sources, languages = episode.document, episode.sources, episode.project.languages
        lines = []
        if merged:
            summary = episode.data.get("merge", {}).get("summary", {})
            counts = group_counts(doc)
            translation, original = lane_label("translation", languages), lane_label("original", languages)
            lines.append(f"这一集共 {counts['total']} 组：")
            lines.append(f"· 对白 {counts['dialogue']} 组：两边都有 {counts['both']} 组，"
                         f"只有{translation} {counts['translation_only']} 组，只有{original} {counts['original_only']} 组")
            other = [f"{name} {counts[key]} 组" for key, name in (("song", "歌曲"), ("sign_info", "屏幕字等"))
                     if counts[key]]
            if other:
                lines.append("· " + "，".join(other) + "（这些本来就不一定配对）")
            lines.append(f"两份字幕整体差 {summary.get('offset_ms', 0) / 1000:+.2f} 秒，已按{translation}的时间对齐。")
            for at, offset in summary.get("cuts", []):
                lines.append(f"{fmt_time(round(at))} 起有一段两边差 {offset / 1000:+.2f} 秒（两个版本剪辑不同），已单独对齐。")
        ledger_lines = report_lines(report, sources, languages)
        lines.extend(ledger_lines[-1:] if merged else ledger_lines)
        if not merged:
            lines.extend(issue_lines(report, doc, sources))
            lines.extend((f"未确认的对白组：{sum(g.kind == 'dialogue' and g.state != 'confirmed' for g in doc.groups)}",
                          f"待确认读音：{sum(a.status == 'uncertain' for a in doc.annotations.values())}",
                          f"未渲染的语气词/隐藏文字：{sum(doc.hidden_count(g) for g in doc.groups)}"))
        if merged:
            self.text = _ResultText("\n".join(lines))
            self.text.setTextFormat(Qt.TextFormat.PlainText)
            self.text.setWordWrap(True)
            self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.text.setFixedWidth(600)
            layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)   # the dialog fits the short result
        else:
            self.text = QPlainTextEdit("\n".join(lines))
            self.text.setReadOnly(True)
        layout.addWidget(self.text)
        if merged:
            single = counts["translation_only"] + counts["original_only"]
            if single:
                warning = QHBoxLayout()
                self.single_language_warning = QLabel(f"有 {single} 组对白只有一种语言，可能是没配上，也可能本来就只有一边，建议看一下。")
                self.single_language_warning.setWordWrap(True)
                self.single_language_warning.setTextFormat(Qt.TextFormat.PlainText)
                warning.addWidget(self.single_language_warning, 1)
                self.single_language_button = QPushButton("只看这些组")
                self.single_language_button.clicked.connect(self.show_single_language)
                warning.addWidget(self.single_language_button)
                layout.addLayout(warning)
            self.details_toggle = QToolButton()
            self.details_toggle.setText("详情")
            self.details_toggle.setCheckable(True)
            self.details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.details_toggle.setArrowType(Qt.ArrowType.RightArrow)
            self.details_toggle.toggled.connect(self.toggle_details)
            layout.addWidget(self.details_toggle)
            self.details_text = QPlainTextEdit("\n".join(ledger_lines[:-1]))
            self.details_text.setMinimumHeight(140)
            self.details_text.setReadOnly(True)
            self.details_text.hide()
            layout.addWidget(self.details_text)
        for key, name in (("missing", "没有着落"), ("duplicate", "重复")):
            count = sum(counts.get(key, 0) for counts in report.ledger.values())
            if count:
                label = QLabel(f"有 {count} 个字{name}。请用“工具 → 检查 → 顺序、原文去向、待办”查看。")
                label.setTextFormat(Qt.TextFormat.PlainText)
                label.setStyleSheet("color:#c62828")
                layout.addWidget(label)
        buttons = QHBoxLayout()
        close = QPushButton("开始编辑" if merged else "关闭")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        dialog_buttons(self, buttons)
        layout.addLayout(buttons)

    def toggle_details(self, expanded: bool) -> None:
        self.details_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.details_text.setVisible(expanded)
        self.adjustSize()          # the short result grows only while the details are open

    def show_single_language(self) -> None:
        self.singleLanguageRequested.emit()
        self.accept()
