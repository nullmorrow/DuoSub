"""Action metadata and shortcut settings, independent of window construction.

Keys are immutable tuples of QKeySequence PortableText strings. A missing
watch binding inherits main; an explicit empty watch binding suppresses it.
No QAction, settings instance or GUI is created by importing this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence


SCENES = MappingProxyType({"main": "编辑界面", "watch": "观剧模式", "k": "K 轴", "popup": "弹窗内部（固定）"})
SCENE_PRIORITY = ("popup", "k", "watch", "main")
SHORTCUTS_VERSION = 1


def _portable(key: str | QKeySequence) -> str:
    sequence = QKeySequence(key)
    if sequence.isEmpty() or any(sequence[i].key() == Qt.Key.Key_unknown for i in range(sequence.count())):
        raise ValueError(f"Invalid shortcut: {key!r}")
    return sequence.toString(QKeySequence.SequenceFormat.PortableText)


def _keys(keys: Iterable[str | QKeySequence] | str | QKeySequence) -> tuple[str, ...]:
    if isinstance(keys, (str, QKeySequence)):
        keys = (keys,) if keys else ()
    return tuple(dict.fromkeys(_portable(key) for key in keys))


RESERVED_KEYS = frozenset(_portable(key) for key in (
    "F1", "Esc", "Tab", "Shift+Tab", "Enter", "Return", "Alt+F4", "Alt",
    "Ctrl+A", "Ctrl+C", "Ctrl+V", "Ctrl+X", "Ctrl+Shift", "Alt+Shift", "Shift+Space",
))
_IME_KEYS = frozenset(getattr(Qt.Key, "Key_" + name) for name in (
    "Mode_switch", "Kanji", "Muhenkan", "Henkan", "Romaji", "Hiragana", "Katakana",
    "Hiragana_Katakana", "Zenkaku", "Hankaku", "Zenkaku_Hankaku", "Kana_Lock", "Kana_Shift",
    "Eisu_Shift", "Eisu_toggle", "Hangul", "Hangul_Start", "Hangul_End", "Hangul_Hanja",
))


def is_reserved(key: str | QKeySequence) -> bool:
    """Check every stroke, including OS/IME keys; Ctrl+Space stays available."""
    sequence = QKeySequence(_portable(key))
    for i in range(sequence.count()):
        stroke = sequence[i]
        text = QKeySequence(stroke).toString(QKeySequence.SequenceFormat.PortableText)
        if (text in RESERVED_KEYS or stroke.keyboardModifiers() & Qt.KeyboardModifier.MetaModifier
                or stroke.key() == Qt.Key.Key_Meta or stroke.key() in _IME_KEYS):
            return True
    return False


@dataclass(frozen=True)
class ActionSpec:
    id: str
    text: str
    keys: Mapping[str, tuple[str, ...]]
    help_anchor: str
    typing: bool = False
    customizable: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", self.id):
            raise ValueError(f"Invalid action id: {self.id!r}")
        if not self.keys or set(self.keys) - SCENES.keys():
            raise ValueError("Action must declare valid scenes")
        object.__setattr__(self, "keys", MappingProxyType({scene: _keys(keys) for scene, keys in self.keys.items()}))


@dataclass(frozen=True)
class ShortcutConflict:
    scene: str
    key: str
    action_ids: tuple[str, ...]


class ActionRegistry:
    """Mutable overrides over immutable specs; conflicts remain inspectable."""

    def __init__(self, specs: Iterable[ActionSpec] | None = None) -> None:
        self._specs: dict[str, ActionSpec] = {}
        self._overrides: dict[tuple[str, str], tuple[str, ...]] = {}
        for spec in DEFAULT_ACTIONS if specs is None else specs:
            self.register(spec)

    def copy(self) -> ActionRegistry:
        """Return an independent editing copy over the same immutable specs."""
        result = ActionRegistry(self)
        result._overrides = self._overrides.copy()
        return result

    def apply_overrides(self, other: ActionRegistry) -> None:
        """Apply an accepted editing copy without replacing registry references."""
        self._overrides = other._overrides.copy()

    def register(self, spec: ActionSpec) -> None:
        if spec.id in self._specs:
            raise ValueError(f"Duplicate action id: {spec.id}")
        self._specs[spec.id] = spec

    def __iter__(self):
        return iter(self._specs.values())

    def lookup(self, action_id: str) -> ActionSpec:
        return self._specs[action_id]

    def _check_scene(self, spec: ActionSpec, scene: str) -> None:
        if scene not in SCENES:
            raise ValueError(f"Unknown scene: {scene}")
        if scene not in spec.keys and not (scene == "watch" and "main" in spec.keys):
            raise ValueError(f"Action {spec.id} is unavailable in {scene}")

    def effective_keys(self, action_id: str, scene: str = "main") -> tuple[str, ...]:
        spec = self.lookup(action_id)
        if scene not in SCENES:
            raise ValueError(f"Unknown scene: {scene}")
        if (action_id, scene) in self._overrides:
            return self._overrides[action_id, scene]
        if scene in spec.keys:
            return spec.keys[scene]
        if scene == "watch":
            return self.effective_keys(action_id, "main")
        return ()

    def set_keys(self, action_id: str, scene: str, keys: Iterable[str | QKeySequence] | str | QKeySequence) -> None:
        spec = self.lookup(action_id)
        self._check_scene(spec, scene)
        if not spec.customizable:
            raise ValueError(f"Fixed action: {action_id}")
        normalized = _keys(keys)
        if action_id == "k.tap" and (len(normalized) > 1 or any(
                QKeySequence(key).count() != 1
                or QKeySequence(key)[0].keyboardModifiers() != Qt.KeyboardModifier.NoModifier
                or QKeySequence(key)[0].key() in (Qt.Key.Key_Shift, Qt.Key.Key_Control, Qt.Key.Key_Alt,
                                                Qt.Key.Key_Meta, Qt.Key.Key_AltGr,
                                                Qt.Key.Key_Super_L, Qt.Key.Key_Super_R,
                                                Qt.Key.Key_Hyper_L, Qt.Key.Key_Hyper_R)
                for key in normalized)):
            raise ValueError("K tapping requires a single key without modifiers")
        if any(is_reserved(key) for key in normalized):
            raise ValueError("Reserved shortcut")
        self._overrides[action_id, scene] = normalized

    def matches_event(self, action_id: str, scene: str, event) -> bool:
        """Match one mode-key stroke; only untouched defaults keep legacy modifier tolerance.

        Transform defaults ignore Shift/keypad flags, and suggestion defaults
        ignore Shift/keypad/GroupSwitch. Explicit overrides always match exactly.
        Hold-to-tap compares its unmodified bound key separately on press/release.
        """
        modifiers = event.modifiers()
        mask = None
        if (action_id, scene) not in self._overrides:
            if action_id in ("watch.transform", "watch.transform_line", "k.adopt", "k.cancel"):
                mask = (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
                        | Qt.KeyboardModifier.MetaModifier)
        if mask is not None:
            modifiers &= mask
        for key in self.effective_keys(action_id, scene):
            sequence = QKeySequence(key)
            if sequence.count() == 1 and sequence[0].key() == event.key():
                if sequence[0].keyboardModifiers() == modifiers:
                    return True
        return False

    def clear_keys(self, action_id: str, scene: str) -> None:
        self.set_keys(action_id, scene, ())

    def reset(self, action_id: str, scene: str) -> None:
        self._check_scene(self.lookup(action_id), scene)
        self._overrides.pop((action_id, scene), None)

    def reset_all(self) -> None:
        self._overrides.clear()

    def conflicts(self, scene: str | None = None) -> tuple[ShortcutConflict, ...]:
        """Compare effective keys, so watch swaps replace main rather than add to it."""
        if scene is not None and scene not in SCENES:
            raise ValueError(f"Unknown scene: {scene}")
        result = []
        for active in (scene,) if scene is not None else SCENES:
            bindings: dict[str, list[str]] = {}
            for spec in self:
                for key in self.effective_keys(spec.id, active):
                    bindings.setdefault(key, []).append(spec.id)
            result.extend(ShortcutConflict(active, key, tuple(ids)) for key, ids in bindings.items() if len(ids) > 1)
        return tuple(result)

    def load(self, settings) -> None:
        """Replace overrides; ignore unknown, fixed and malformed entries."""
        self.reset_all()
        if str(settings.value("shortcuts/version", SHORTCUTS_VERSION)) != str(SHORTCUTS_VERSION):
            return
        for path in settings.allKeys():
            parts = path.split("/")
            if len(parts) != 3 or parts[0] != "shortcuts" or parts[2] not in self._specs:
                continue
            try:
                self.set_keys(parts[2], parts[1], _decode_keys(str(settings.value(path))))
            except ValueError:
                continue

    def save(self, settings) -> None:
        """Write only overrides; remove reset known entries, leaving other settings alone."""
        for spec in self:
            for scene in SCENES:
                path = f"shortcuts/{scene}/{spec.id}"
                if (spec.id, scene) in self._overrides:
                    settings.setValue(path, ";".join(self._overrides[spec.id, scene]))
                else:
                    settings.remove(path)
        settings.setValue("shortcuts/version", SHORTCUTS_VERSION)


def _decode_keys(value: str) -> tuple[str, ...]:
    # A semicolon can itself be a key (e.g. Ctrl+;). Only split after a
    # complete sequence; no escaping or additional settings format is needed.
    result, token = [], ""
    for char in value:
        if char == ";" and token:
            try:
                result.append(_portable(token))
            except ValueError:
                pass
            else:
                token = ""
                continue
        token += char
    if token:
        result.append(_portable(token))
    return tuple(result)


def _spec(action_id: str, text: str, keys=(), anchor: str = "review", *, scene: str = "main",
          watch=None, typing: bool = False, customizable: bool = True) -> ActionSpec:
    bindings = {scene: _keys(keys)}
    if watch is not None:
        bindings["watch"] = _keys(watch)
    return ActionSpec(action_id, text, bindings, anchor, typing, customizable)


# Permanent IDs and names follow DESIGN.md §3, in menu order. Choice lists
# (layouts, display options, profiles, speaker lanes and effects) are dynamic.
DEFAULT_ACTIONS = (
    _spec("file.new", "新建工程…", "Ctrl+Shift+N", "project", typing=True),
    _spec("file.open", "打开工程…", "Ctrl+O", "project", typing=True),
    _spec("file.save", "保存", "Ctrl+S", "save", typing=True),
    _spec("file.ask_save", "关闭或换集时询问是否保存", anchor="save"),
    _spec("file.export", "发布用 ASS…", "Ctrl+E", "export", typing=True),
    _spec("file.export_season", "整季发布用 ASS…", anchor="export-season"),
    _spec("file.export_review_list", "标记和隐藏的句子…", anchor="hide-mark"),
    _spec("project.template", "样式模板：字体、颜色和位置…", anchor="template"),
    _spec("file.quit", "退出", anchor="project"),
    _spec("episode.add", "添加一集…", anchor="add-episode"),
    _spec("episode.batch", "自动导入整季…", anchor="batch"),
    _spec("episode.batch_bilingual", "自动导入整季（双语字幕）…", anchor="batch"),
    _spec("episode.video_links", "视频文件位置…", anchor="video-links"),
    _spec("episode.source_copies", "原字幕副本（检查和修复）…", anchor="source-copies"),
    _spec("episode.reanalyse", "重新分析原声字幕（按新规则；原文件不变）…", anchor="add-episode"),
    _spec("episode.merge", "合并（生成工作文档）…", anchor="add-episode"),
    _spec("episode.remove", "从工程中移除这一集…", anchor="project"),
    _spec("edit.undo", "撤销", "Ctrl+Z", typing=True),
    _spec("edit.redo", "重做", ("Ctrl+Y", "Ctrl+Shift+Z"), typing=True),
    _spec("group.duplicate", "复制这一组", "Ctrl+D", typing=True),
    _spec("edit.delete_member", "删除这一行（只删当前语言）"),
    _spec("group.delete", "删除这一组（原声和翻译都删）", "Del"),
    _spec("group.confirm_next", "确认并播放下一组", ("Ctrl+Enter", "Ctrl+Return"), typing=True),
    _spec("group.confirm", "确认这一组", ("Ctrl+Shift+Enter", "Ctrl+Shift+Return"), typing=True),
    _spec("group.unconfirm", "取消确认"),
    _spec("group.merge", "与下一组合并", "Ctrl+J", typing=True),
    _spec("group.split", "在光标处拆分", "Ctrl+K", typing=True),
    _spec("group.unlink", "解除配对", "Ctrl+U", typing=True),
    _spec("group.head_to_prev", "开头一段 → 上一组", "Alt+Up", typing=True),
    _spec("group.tail_to_next", "结尾一段 → 下一组", "Alt+Down", typing=True),
    _spec("group.take_from_prev", "拿回上一组结尾一段", "Alt+Shift+Up", typing=True),
    _spec("group.take_from_next", "拿回下一组开头一段", "Alt+Shift+Down", typing=True),
    _spec("group.cursor_to_prev", "光标前面移到上一组", "Ctrl+Shift+Up", typing=True),
    _spec("group.cursor_to_next", "光标后面移到下一组", "Ctrl+Shift+Down", typing=True),
    _spec("group.add_original", "补上原声（{语言}）…"),
    _spec("group.add_translation", "补上翻译（{语言}）…"),
    _spec("group.ai_listen", "AI听译…", anchor="ai-transcribe"),
    _spec("group.to_song", "歌曲（K 轴）…"),
    _spec("group.to_dialogue", "普通台词…"),
    _spec("group.to_sign", "屏幕字…"),
    _spec("position.left", "左侧说话人", anchor="position"),
    _spec("position.right", "右侧说话人", anchor="position"),
    _spec("position.top", "移到顶部", anchor="position"),
    _spec("position.bottom", "移到底部", anchor="position"),
    _spec("position.alt", "底部第二人", anchor="position"),
    _spec("position.restore", "恢复原位置", anchor="position"),
    _spec("watch.transform", "自由变换", "T", "transform", scene="watch"),
    _spec("watch.transform_line", "自由变换这一行", "Alt+T", "transform", scene="watch"),
    _spec("watch.edit_translation", "就地编辑翻译", "Z", "watch", scene="watch"),
    _spec("watch.edit_original", "就地编辑原文", "X", "watch", scene="watch"),
    _spec("group.hide", "隐藏（不显示、不导出）", "Ctrl+H", "hide-mark", typing=True),
    _spec("group.mark", "标记待复查", "Ctrl+M", "hide-mark", typing=True),
    _spec("group.mark_note", "编辑标记备注…", anchor="hide-mark"),
    _spec("time.set", "设置开始和结束时间…", "Ctrl+T", "timing", typing=True),
    _spec("time.shift", "整体平移…", "Ctrl+Shift+T", "timing", typing=True),
    _spec("time.adopt", "采用建议时间", "C", "timing"),
    _spec("time.start_minus", "开始时间 −1 帧", "[", "timing"),
    _spec("time.start_plus", "开始时间 +1 帧", "]", "timing"),
    _spec("time.end_minus", "结束时间 −1 帧", ("Shift+[", "{"), "timing"),
    _spec("time.end_plus", "结束时间 +1 帧", ("Shift+]", "}"), "timing"),
    _spec("time.unify_original", "用原声字幕的时间", anchor="unify-times"),
    _spec("time.unify_translation", "用翻译字幕的时间", anchor="unify-times"),
    _spec("time.autotime", "自动调轴（按视频音频）", anchor="autotime"),
    _spec("time.dialogue_align", "精确对话轴（按台词对齐）…", anchor="dialogue-align"),
    _spec("k.tap", "按住 K 跟唱", "K", "karaoke", scene="k"),
    _spec("k.import", "读入原字幕的 K 轴", anchor="karaoke", scene="k"),
    _spec("k.auto_line", "这一句", anchor="karaoke-auto", scene="k"),
    _spec("k.auto_selected", "选中的句", anchor="karaoke-auto", scene="k"),
    _spec("k.auto_song", "整首歌", anchor="karaoke-auto", scene="k"),
    _spec("k.suggest_onset", "按起音（旧方法，只这一句）", anchor="karaoke-auto", scene="k"),
    _spec("k.adopt", "采用建议", ("Enter", "Return"), "karaoke", scene="k"),
    _spec("k.cancel", "取消", "Esc", "karaoke", scene="k", customizable=False),
    _spec("k.add_rest", "后面加空拍", "Shift+K", "karaoke", scene="k"),
    _spec("k.remove_rest", "删掉空拍", "Del", "karaoke", scene="k"),
    _spec("k.clear", "清除 K 轴", anchor="karaoke", scene="k"),
    _spec("k.confirm", "确认 K 轴", anchor="karaoke", scene="k"),
    _spec("k.mapping", "对应…", anchor="karaoke", scene="k"),
    _spec("k.english", "英文按音节拆开", anchor="karaoke", scene="k"),
    *(_spec(f"k.{kind}_{scope}", label + dots, anchor="karaoke", scene="k")
      for kind, dots in (("redistribute", ""), ("align_times", ""), ("fade", "…"))
      for scope, label in (("line", "这一句"), ("selected", "选中的句"), ("song", "整首歌"))),
    _spec("k.parameters", "参数…", anchor="karaoke-fx", scene="k"),
    _spec("ruby.edit", "改读音", "F2", "readings", typing=True),
    _spec("ruby.regen", "重新生成振假名（保留确认过的）", "Ctrl+G", "readings", typing=True),
    _spec("ruby.regen_all", "重新生成振假名（包括确认过的）", "Ctrl+Shift+G", "readings", typing=True),
    _spec("project.words", "工程词语与分词规则…", anchor="project-readings"),
    _spec("project.ruby_settings", "振假名设置…", anchor="readings"),
    _spec("project.pinyin_settings", "拼音设置…", anchor="pinyin"),
    _spec("play.toggle", "播放 / 暂停", "Space", "playback"),
    _spec("play.toggle_plain", "播放 / 暂停（输入框里也可用）", "Ctrl+Space", "playback", typing=True),
    _spec("play.step_mode", "逐段播放", "Ctrl+Shift+Space", "playback", typing=True),
    _spec("play.group", "重播本组", "R", "playback"),
    _spec("play.group_typing", "重播本组（输入框里也可用）", "Ctrl+R", "playback", typing=True),
    *(_spec(f"play.{key.lower()}", text, key, "playback") for key, text in (
        ("S", "播放本组"), ("Q", "播放开始前 0.5 秒"), ("W", "播放结束后 0.5 秒"),
        ("D", "播放开头 0.5 秒"), ("E", "播放结尾 0.5 秒"))),
    _spec("nav.prev", "上一组", "PgUp", "playback"),
    _spec("nav.next", "下一组", "PgDown", "playback"),
    _spec("play.prev_group", "播放上一组", "Ctrl+Left", "playback", watch="Left"),
    _spec("play.next_group", "播放下一组", "Ctrl+Right", "playback", watch="Right"),
    _spec("play.frame_back", "上一帧", "Left", "playback", watch="Ctrl+Left"),
    _spec("play.frame_forward", "下一帧", "Right", "playback", watch="Ctrl+Right"),
    _spec("play.follow_list", "播放时列表跟随", anchor="playback"),
    _spec("view.watch_timing", "观剧模式：看片打轴（视频 + 波形常驻）", "F10", "watch", typing=True),
    _spec("view.watch_full", "观剧模式：全屏观看（点字幕才出波形）", "F11", "watch", typing=True),
    _spec("view.karaoke", "卡拉 OK 模式", "F9", "karaoke", typing=True),
    _spec("view.source_panel", "原文"),
    _spec("view.people_panel", "人物"),
    _spec("view.dock_reset", "原文、人物面板放回右侧"),
    _spec("watch.speakers", "显示说话人", anchor="watch"),
    _spec("watch.frames", "暂停时显示字幕外框", anchor="watch"),
    _spec("watch.frame_color", "字幕外框颜色…", anchor="watch"),
    _spec("display.hide_period", "隐藏日文句末「。」（只影响预览和导出，原文不变）", anchor="readings"),
    _spec("display.punctuation", "标点和空格…", anchor="display-rules"),
    _spec("display.replacements", "批量替换…", anchor="display-replacements"),
    _spec("tools.check", "顺序、原文去向、待办", anchor="review"),
    _spec("tools.check_fonts", "字体（缺字体、缺字）", anchor="fonts"),
    _spec("tools.check_selfref", "自称（私 的读法）…", anchor="readings"),
    _spec("tools.export_review", "导出审议包（发布前给大模型复查）…", anchor="review-packet"),
    _spec("tools.import_review", "导回审议意见…", anchor="review-answers"),
    _spec("tools.passage_library", "重复段落库…", anchor="passages"),
    _spec("tools.record_passage", "记为重复段落…", anchor="passages"),
    _spec("tools.find_passage", "查找并套用…", anchor="passages"),
    _spec("tools.brackets_to_text", "原字幕括号里的片假名不是注音：放回正文…", anchor="add-episode"),
    _spec("tools.shortcuts", "快捷键…", anchor="keys"),
    _spec("tools.private", "私有功能", anchor="fonts"),
    _spec("tools.font_library", "字体库位置…", anchor="fonts"),
    _spec("tools.clear_cache", "清除排版缓存", anchor="fonts"),
    _spec("help.manual", "使用说明", "F1", "project", typing=True, customizable=False),
    _spec("help.shortcuts", "快捷键一览", anchor="keys"),
    _spec("help.changes", "版本记录", anchor="changes"),
    _spec("help.about", "关于双语字幕工坊", anchor="project"),
    _spec("watch.edit_menu", "编辑菜单", "Alt+E", "watch", scene="watch", customizable=False),
    _spec("watch.exit", "退出观剧模式", "Tab", "watch", scene="watch", customizable=False),
    _spec("watch.escape", "取消", "Esc", "watch", scene="watch", customizable=False),
    _spec("watch.volume_up", "增大音量", "Up", "watch", scene="watch"),
    _spec("watch.volume_down", "减小音量", "Down", "watch", scene="watch"),
    _spec("popup.accept", "确定", ("Enter", "Return"), "keys", scene="popup", customizable=False),
    _spec("popup.cancel", "取消", "Esc", "keys", scene="popup", customizable=False),
    _spec("popup.focus", "切换输入框", ("Tab", "Shift+Tab"), "keys", scene="popup", customizable=False),
    _spec("popup.choice", "选择候选（自由变换中 3 切换 3D 旋转）", tuple("123456789"),
          "keys", scene="popup", customizable=False),
    _spec("popup.navigate", "方向键", ("Left", "Right", "Up", "Down"), "keys", scene="popup", customizable=False),
    _spec("popup.word", "切换词语", ("Alt+Left", "Alt+Right"), "readings", scene="popup", customizable=False),
    _spec("popup.clear", "清除变形", "Backspace", "transform", scene="popup", customizable=False),
)
