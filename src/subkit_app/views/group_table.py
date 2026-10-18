"""The Group list: one row per editing unit, with visible status badges."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QAbstractTableModel, QItemSelectionModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTableView

from subkit.ass.text import plain_text
from subkit.languages import lane_label
from subkit.model.document import Document, Group
from subkit.core.people import needs_speaker
from subkit.core.merge import dialogue_lanes

COLS = ["", "#", "开始", "结束", "翻译", "原声", "标签", "说话人"]
C_STATE, C_NUM, C_START, C_END, C_CN, C_JP, C_TAGS, C_SPEAKER = range(8)


def fmt_time(ms: int) -> str:
    sign, ms = ("-" if ms < 0 else ""), abs(int(ms))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{sign}{h}:{m:02d}:{s:02d}.{ms // 10:02d}"


def uncertain_reading_counts(doc: Document) -> dict[str, int]:
    cached = getattr(doc, "_uncertain_reading_index", None)
    if cached is not None and cached[0] == doc.revision:
        return cached[1]
    counts: dict[str, int] = {}
    for annotation in doc.annotations.values():
        if annotation.status == "uncertain":
            counts[annotation.member] = counts.get(annotation.member, 0) + 1
    setattr(doc, "_uncertain_reading_index", (doc.revision, counts))
    return counts


def group_badges(doc: Document, g: Group, counts: Optional[dict[str, int]] = None) -> list[tuple[str, str]]:
    """(text, kind) badges.  kind: warn | info | ok."""
    out: list[tuple[str, str]] = []
    if "hidden" in g.flags:
        out.append(("已隐藏", "info"))
    if "marked" in g.flags:
        out.append(("★ 标记", "info"))
    if g.kind == "dialogue":
        langs = {m.lane for m in doc.members_of(g) if plain_text(m.text).strip()}
        if langs == {"translation"}:
            out.append(("只有翻译", "warn"))
        elif langs == {"original"}:
            out.append(("只有原声", "warn"))
        if sum(1 for m in doc.members_of(g) if m.lane == "translation") >= 2:
            out.append(("多行组", "info"))
        doubts = (g.origin or {}).get("doubts", [])
        if "close_boundary" in doubts and g.state != "confirmed":
            out.append(("边界拿不准", "warn"))
        if "reanchored" in doubts:
            out.append(("预告重对齐", "info"))
    elif g.kind != "dialogue":
        out.append(({"song": "歌曲", "sign": "屏幕字", "info": "注释"}.get(g.kind, g.kind), "info"))
    hidden = doc.hidden_count(g)
    if hidden:
        out.append((f"含 {hidden} 处未渲染语气词", "warn"))
    if counts is None:
        counts = uncertain_reading_counts(doc)
    unc = sum(counts.get(mid, 0) for mid in set(g.members))
    if unc:
        out.append((f"读音?×{unc}", "warn"))
    if "order_exception" in g.flags:
        out.append(("顺序例外", "warn"))
    if g.members:
        s, e = doc.group_time(g)
        if e <= max(s, 0):                # nothing on screen: exported as a comment line
            out.append(("不显示：在片头之前" if e <= 0 else "不显示：时长为 0", "warn"))
    return out


class GroupTableModel(QAbstractTableModel):
    def __init__(self, doc: Document, parent=None, *, languages=None, character_book=None) -> None:
        super().__init__(parent)
        self.doc = doc
        self.languages = dict(languages or {})
        self.character_book = character_book
        self._cache: dict[str, tuple] = {}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.doc.groups)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(COLS)

    def headerData(self, section: int, orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:  # noqa: N802
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return lane_label("translation", self.languages) if section == C_CN else (
                lane_label("original", self.languages) if section == C_JP else COLS[section])
        return None

    def group_at(self, row: int) -> Optional[Group]:
        return self.doc.groups[row] if 0 <= row < len(self.doc.groups) else None

    def row_of(self, gid: str) -> int:
        try:
            return self.doc.group_index(gid)
        except KeyError:
            return -1

    def _row(self, g: Group) -> tuple:
        key = (g.id, self.doc.revision)
        hit = self._cache.get(g.id)
        if hit and hit[0] == key:
            return hit[1]
        s, e = self.doc.group_time(g)
        cn = " ／ ".join(plain_text(m.text, breaks=" ").strip() for m in self.doc.members_of(g) if m.lane == "translation")
        jp = " ／ ".join(plain_text(m.text, breaks=" ").strip() for m in self.doc.members_of(g) if m.lane == "original")
        badges = group_badges(self.doc, g)
        val = (s, e, cn, jp, badges)
        self._cache[g.id] = (key, val)
        return val

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        g = self.doc.groups[index.row()]
        s, e, cn, jp, badges = self._row(g)
        c = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if c == C_STATE:
                return "✓" if g.state == "confirmed" else ""
            if c == C_NUM:
                return index.row() + 1
            if c == C_START:
                return fmt_time(s)
            if c == C_END:
                return fmt_time(e)
            if c == C_CN:
                return cn
            if c == C_JP:
                return jp
            if c == C_TAGS:
                return "  ".join(b for b, _k in badges)
            if c == C_SPEAKER:
                names = []
                for member in self.doc.members_of(g):
                    if not needs_speaker(g, member) or (self.character_book and self.character_book.speaker_ignored(member)):
                        continue
                    speaker = member.speaker or {}
                    cid, label = speaker.get("character"), speaker.get("label", "")
                    if self.character_book and label in self.character_book.not_person:
                        continue
                    linked = self.character_book and any(ch.id == cid for ch in self.character_book.characters)
                    name = ("群众" if cid == "crowd" else
                            self.character_book.display_name(cid, member.lang, label, "short") if linked else label + "?")
                    if name not in names:
                        names.append(name)
                return " ／ ".join(names)
        elif role == Qt.ItemDataRole.ForegroundRole:
            if "hidden" in g.flags:
                return QBrush(QColor("#8c959f"))
            if c == C_STATE:
                return QBrush(QColor("#1a7f37"))
            if c == C_TAGS and any(k == "warn" for _b, k in badges):
                return QBrush(QColor("#b35900"))
            if g.kind != "dialogue":
                return QBrush(QColor("#6e7781"))
        elif role == Qt.ItemDataRole.BackgroundRole:
            if g.kind == "dialogue" and any(b in ("只有翻译", "只有原声") for b, _k in badges):
                return QBrush(QColor("#fff8e6"))
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if c in (C_STATE, C_NUM, C_START, C_END):
                return int(Qt.AlignmentFlag.AlignCenter)
        elif role == Qt.ItemDataRole.ToolTipRole:
            if c in (C_CN, C_JP, C_TAGS):
                extra = "\n（分界落在一句原声字幕的中间）" if "cut_inside_cue" in (g.origin or {}).get("doubts", []) else ""
                return f"{lane_label('translation', self.languages)}：{cn}\n{lane_label('original', self.languages)}：{jp}\n" + ("标签：" + "、".join(b for b, _ in badges) if badges else "") + extra + ("\n备注：" + g.extra["mark_note"] if "marked" in g.flags and g.extra.get("mark_note") else "")
        elif role == Qt.ItemDataRole.UserRole:
            return g.id
        return None

    def refresh_groups(self, gids: set[str]) -> None:
        # Unchanged rows survive metadata revisions; changed rows are rebuilt below.
        for gid, (key, value) in list(self._cache.items()):
            if key[1] == self.doc.revision - 1:
                self._cache[gid] = ((gid, self.doc.revision), value)
        for gid in gids:
            self._cache.pop(gid, None)
            r = self.row_of(gid)
            if r >= 0:
                self.dataChanged.emit(self.index(r, 0), self.index(r, len(COLS) - 1))

    def reset(self) -> None:
        self.beginResetModel()
        self._cache.clear()
        self.endResetModel()


class GroupFilter(QSortFilterProxyModel):
    MODES = ("全部", "只看对白", "只看歌曲", "只有一种语言的对白", "需要处理", "只看已标记", "只看已隐藏")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mode = "全部"

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.invalidateFilter()

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:  # noqa: N802
        if self.mode == "全部":
            return True
        m: GroupTableModel = self.sourceModel()
        g = m.group_at(row)
        if g is None:
            return False
        if self.mode == "只看已标记":
            return "marked" in g.flags
        if self.mode == "只看已隐藏":
            return "hidden" in g.flags
        if self.mode == "只看对白":
            return g.kind == "dialogue"
        if self.mode == "只看歌曲":
            return g.kind == "song"
        if self.mode == "只有一种语言的对白":
            return g.kind == "dialogue" and len(dialogue_lanes(m.doc, g)) == 1
        badges = m._row(g)[4]
        return g.kind == "dialogue" and (g.state != "confirmed") and any(k == "warn" for _b, k in badges)


def set_action_reason(action, reason: str = "") -> None:
    """Keep the original label when a state-dependent action is disabled and restored."""
    base = action.property("enabled_text")
    if base is None:
        base = action.text()
        action.setProperty("enabled_text", base)
    label, sep, shortcut = base.partition("\t")
    action.setText(label + (f"（{reason}）" if reason else "") + sep + shortcut)
    action.setEnabled(not reason)
    action.setToolTip(action.text() if reason else "")


def conversion_reasons(kinds: set[str]) -> tuple[str, str]:
    song = "" if "dialogue" in kinds else (
        "先选中字幕组" if not kinds else "已经是歌曲" if kinds == {"song"} else "屏幕字和说明不能转")
    return song, "" if kinds & {"song", "sign"} else (
        "先选中字幕组" if not kinds else "已经是对白" if kinds == {"dialogue"} else "说明不能转")


def sign_conversion_reason(kinds: set[str]) -> str:
    return "" if "dialogue" in kinds else (
        "先选中字幕组" if not kinds else "已经是屏幕字" if kinds == {"sign"} else
        "歌曲先转为对白" if "song" in kinds else "说明不能转")


class GroupTable(QTableView):
    help_anchor = "review"
    groupActivated = Signal(str)
    groupSelectionChanged = Signal()
    confirmNext = Signal()
    confirmOnly = Signal()
    deleteGroupsRequested = Signal(list)          # group ids (Del / right click)
    deleteLaneRequested = Signal(list, str)       # group ids, "original" | "translation"
    addLanguageRequested = Signal(list, str)
    listeningRequested = Signal(str)
    hiddenGroupsRequested = Signal(list, bool)
    markedGroupsRequested = Signal(list, bool)
    markNoteRequested = Signal(list)
    confirmGroupsRequested = Signal(list, bool)   # group ids, confirmed
    convertRequested = Signal(list, str)         # group ids, "song" | "dialogue"
    shiftGroupsRequested = Signal(list)
    duplicateGroupsRequested = Signal(list)
    groupTimesRequested = Signal(list)
    recordPassageRequested = Signal(list)
    playRequested = Signal(str)
    karaokeEffectRequested = Signal(list, str)
    karaokeConfirmRequested = Signal(list)       # song group ids

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.confirmOnly.emit()
            else:
                self.confirmNext.emit()
            return
        super().keyPressEvent(e)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)    # Shift / Ctrl + click
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(26)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    def setup_columns(self) -> None:
        h = self.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for c, w in ((C_STATE, 28), (C_NUM, 52), (C_START, 92), (C_END, 92), (C_CN, 360), (C_JP, 420), (C_TAGS, 220), (C_SPEAKER, 150)):
            self.setColumnWidth(c, w)
        h.setStretchLastSection(True)

    def selected_gids(self) -> list[str]:
        """The selected groups in list order (the current one when nothing else is selected)."""
        sm = self.selectionModel()
        rows = sorted({i.row() for i in sm.selectedRows()}) if sm is not None else []
        out = [self.model().data(self.model().index(r, 0), Qt.ItemDataRole.UserRole) for r in rows]
        out = [g for g in out if g]
        if not out and self.current_gid():
            out = [self.current_gid()]
        return out

    def mousePressEvent(self, e) -> None:  # noqa: N802
        idx = self.indexAt(e.position().toPoint())
        sm = self.selectionModel()
        if (e.button() == Qt.MouseButton.RightButton and idx.isValid() and sm is not None
                and sm.isRowSelected(idx.row(), QModelIndex())):
            sm.setCurrentIndex(idx.siblingAtColumn(C_CN), QItemSelectionModel.SelectionFlag.NoUpdate)
            e.accept()
            return
        modifiers = e.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        bars = (self.horizontalScrollBar(), self.verticalScrollBar())
        values = [bar.value() for bar in bars]
        self._keep_scroll = bool(modifiers)
        try:
            super().mousePressEvent(e)
        finally:
            self._keep_scroll = False
            if modifiers:
                for bar, value in zip(bars, values):
                    bar.setValue(value)

    def scrollTo(self, index, hint=QAbstractItemView.ScrollHint.EnsureVisible) -> None:  # noqa: N802
        if not getattr(self, "_keep_scroll", False):
            super().scrollTo(index, hint)

    def contextMenuEvent(self, e) -> None:  # noqa: N802
        idx = self.indexAt(e.pos())
        if not idx.isValid():
            return
        sm = self.selectionModel()
        if not sm.isRowSelected(idx.row(), QModelIndex()):      # right click outside the selection: that row
            self.setCurrentIndex(idx.siblingAtColumn(C_CN))
        gids = self.selected_gids()
        if not gids:
            return
        menu = self._context_menu(gids)
        try:
            chosen = menu.exec(e.globalPos())
            # Native QMenu already triggered the proxy. Menu doubles may only return it.
            if chosen is not None and not menu.property("command_triggered"):
                chosen.trigger()
        finally:
            menu.deleteLater()

    def _context_menu(self, gids: list[str]) -> QMenu:
        """Build view-local proxies; shared action labels and shortcut state stay intact."""
        src = self.model().sourceModel()
        selected = set(gids)
        groups = [g for g in src.doc.groups if g.id in selected]
        kinds = {g.kind for g in groups}
        songs = [g.id for g in groups if g.kind == "song"]
        menu = QMenu(self)
        menu.setToolTipsVisible(True)

        def add(parent, action_id, callback, *, text=None, reason="", key=None, anchor=None, status_tip=None):
            shared = self.context_action_lookup(action_id)
            spec = self.context_action_spec(action_id)
            keys = self.context_action_keys(action_id)
            label = spec.text if text is None else text
            shortcut = key if key is not None else (keys[0] if keys else "")
            action = parent.addAction(label + ("\t" + shortcut if shortcut else ""))
            action.setProperty("action_id", action_id)
            action.setProperty("help_anchor", anchor or spec.help_anchor)
            action.setStatusTip(shared.statusTip() if status_tip is None else status_tip)
            action.setToolTip(shared.toolTip())
            set_action_reason(action, reason)
            if not reason:
                action.setToolTip(shared.toolTip())
            def invoke():
                menu.setProperty("command_triggered", True)
                callback()
            action.triggered.connect(invoke)
            return action

        def submenu(label, anchor="review"):
            child = menu.addMenu(label)
            child.help_anchor = anchor
            child.menuAction().setProperty("help_anchor", anchor)
            child.setToolTipsVisible(True)
            return child

        add(menu, "play.group", lambda: self.playRequested.emit(gids[0]), text="播放这一组")
        menu.addSeparator()
        for confirmed, action_id in ((True, "group.confirm"), (False, "group.unconfirm")):
            targets = [g.id for g in groups if (g.state == "confirmed") != confirmed]
            if targets:
                add(menu, action_id, lambda targets=targets, value=confirmed:
                    self.confirmGroupsRequested.emit(targets, value))
        menu.addSeparator()
        add(menu, "group.duplicate", lambda: self.duplicateGroupsRequested.emit(gids),
            text=f"复制这 {len(gids)} 组" if len(gids) > 1 else None,
            status_tip="在原组后面复制一份内容和时间相同的组；一步撤销。")
        add(menu, "time.set", lambda: self.groupTimesRequested.emit(gids),
            reason="只能对一组设置" if len(gids) != 1 else "",
            status_tip="精确输入开始和结束时间，可选择只改一种语言和是否平移 K 轴拍。")
        add(menu, "time.shift", lambda: self.shiftGroupsRequested.emit(gids))
        additions = [lane for lane in ("original", "translation") if any(
            not any(m.lane == lane and m.role != "comment" for m in src.doc.members_of(g)) for g in groups)]
        if additions or kinds & {"dialogue", "sign"}:
            menu.addSeparator()
            for lane in additions:
                add(menu, "group.add_" + lane, lambda lane=lane: self.addLanguageRequested.emit(gids, lane),
                    text=f"补上{lane_label(lane, src.languages)}…")
            if kinds & {"dialogue", "sign"}:
                from subkit.core.listening import unavailable_reason
                add(menu, "group.ai_listen", lambda: self.listeningRequested.emit(gids[0]),
                    reason=unavailable_reason(src.doc, gids, src.languages))
        song_reason, dialogue_reason = conversion_reasons(kinds)
        conversions = (("song", song_reason), ("dialogue", dialogue_reason), ("sign", sign_conversion_reason(kinds)))
        possible = [(kind, reason) for kind, reason in conversions if not reason]
        if possible or songs:
            menu.addSeparator()
            if possible:
                convert = submenu("转为")
                convert.menuAction().setToolTip("\n".join(
                    self.context_action_spec("group.to_" + kind).text + "：" + reason
                    for kind, reason in conversions if reason))
                for kind, _ in possible:
                    add(convert, "group.to_" + kind, lambda kind=kind: self.convertRequested.emit(gids, kind))
            if songs:
                effects = submenu("K 轴特效", "karaoke-fx")
                for key, label in getattr(self, "karaoke_effect_choices", lambda: [])():
                    action = effects.addAction(label)
                    action.setProperty("help_anchor", "karaoke-fx")
                    action.triggered.connect(lambda _=False, key=key: (
                        menu.setProperty("command_triggered", True), self.karaokeEffectRequested.emit(songs, key)))
                add(menu, "k.confirm", lambda: self.karaokeConfirmRequested.emit(songs),
                    text=f"确认 K 轴（{len(songs)} 句歌词）", anchor="karaoke-auto")
        menu.addSeparator()
        for flag, action_id, signal, opposite in (
                ("hidden", "group.hide", self.hiddenGroupsRequested, "取消隐藏"),
                ("marked", "group.mark", self.markedGroupsRequested, "取消标记")):
            for value in (True, False):
                targets = [g.id for g in groups if (flag in g.flags) != value]
                if targets:
                    add(menu, action_id, lambda targets=targets, value=value, signal=signal: signal.emit(targets, value),
                        text=None if value else opposite)
        if any("marked" in g.flags for g in groups):
            add(menu, "group.mark_note", lambda: self.markNoteRequested.emit(gids))
        menu.addSeparator()
        add(menu, "tools.record_passage", lambda: self.recordPassageRequested.emit(gids))
        menu.addSeparator()
        # When the selection holds only one language, deleting that language is
        # the same as 「删除这一组」 below, so the lane deletes appear only when both do.
        present = {m.lane for g in groups for m in src.doc.members_of(g)}
        lanes = ["original", "translation"] if present >= {"original", "translation"} else []
        if lanes:
            delete = submenu("删除")
            for lane in lanes:
                add(delete, "edit.delete_member", lambda lane=lane: self.deleteLaneRequested.emit(gids, lane),
                    text="只删原声" if lane == "original" else "只删翻译", key="")
        add(menu, "group.delete", lambda: self.deleteGroupsRequested.emit(gids))
        return menu

    def current_gid(self) -> Optional[str]:
        idx = self.currentIndex()
        if not idx.isValid():
            return None
        return self.model().data(idx.siblingAtColumn(0), Qt.ItemDataRole.UserRole)

    def select_gid(self, gid: str, *, scroll: bool = True, preserve_selection: bool = False) -> None:
        proxy = self.model()
        src = proxy.sourceModel()
        r = src.row_of(gid)
        if r < 0:
            return
        pidx = proxy.mapFromSource(src.index(r, C_CN))
        if not pidx.isValid():
            return
        keep_scroll = getattr(self, "_keep_scroll", False)
        self._keep_scroll = keep_scroll or not scroll
        try:
            if preserve_selection:
                self.selectionModel().setCurrentIndex(pidx, QItemSelectionModel.SelectionFlag.NoUpdate)
            else:
                # Explicit: setCurrentIndex() would consult the keyboard state and could add to a Ctrl selection.
                self.selectionModel().setCurrentIndex(
                    pidx, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
            if scroll:
                self.scrollTo(pidx, QAbstractItemView.ScrollHint.PositionAtCenter)
        finally:
            self._keep_scroll = keep_scroll

    def select_gids(self, gids: list[str]) -> None:
        if not gids:
            return
        self.select_gid(gids[0])
        proxy, sm = self.model(), self.selectionModel()
        for gid in gids[1:]:
            row = proxy.sourceModel().row_of(gid)
            index = proxy.mapFromSource(proxy.sourceModel().index(row, C_CN))
            if index.isValid():
                sm.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)

    def currentChanged(self, current: QModelIndex, previous: QModelIndex) -> None:  # noqa: N802
        super().currentChanged(current, previous)
        if current.isValid():
            gid = self.model().data(current.siblingAtColumn(0), Qt.ItemDataRole.UserRole)
            if gid:
                self.groupActivated.emit(gid)

    def selectionChanged(self, selected, deselected) -> None:  # noqa: N802
        super().selectionChanged(selected, deselected)
        self.groupSelectionChanged.emit()
