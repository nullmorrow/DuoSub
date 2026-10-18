"""S4a: independent r2 menu contract and selection-aware command wiring."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import unittest
from unittest.mock import Mock, patch

from tests import test_convert_ui as fixtures
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest
from subkit.core.listening import unavailable_reason


def A(action_id, text, key="", anchor="review", kind="plain"):
    return (text, action_id, key, anchor, kind, None)


def D(text, anchor="readings", kind="check"):
    return A(None, text, anchor=anchor, kind=kind)


def M(text, anchor, *children):
    return (text, None, "", anchor, "menu", children)


MENU_TREE = (
    M("文件(&F)", "project",
      A("file.new", "新建工程…", "Ctrl+Shift+N", "project"), A("file.open", "打开工程…", "Ctrl+O", "project"), None,
      A("file.save", "保存", "Ctrl+S", "save"), A("file.ask_save", "关闭或换集时询问是否保存", anchor="save", kind="check"), None,
      M("导出", "export", A("file.export", "发布用 ASS…", "Ctrl+E", "export"),
        A("file.export_season", "整季发布用 ASS…", anchor="export-season"),
        A("file.export_review_list", "标记和隐藏的句子…", anchor="hide-mark")), None,
      A("project.template", "样式模板：字体、颜色和位置…", anchor="template"), None, A("file.quit", "退出", anchor="project")),
    M("集(&P)", "project", A("episode.add", "添加一集…", anchor="add-episode"),
      A("episode.batch", "自动导入整季…", anchor="batch"), A("episode.batch_bilingual", "自动导入整季（双语字幕）…", anchor="batch"), None,
      A("episode.video_links", "视频文件位置…", anchor="video-links"), A("episode.source_copies", "原字幕副本（检查和修复）…", anchor="source-copies"),
      A("episode.reanalyse", "重新分析原声字幕（按新规则；原文件不变）…", anchor="add-episode"),
      A("episode.merge", "合并（生成工作文档）…", anchor="add-episode"), None, A("episode.remove", "从工程中移除这一集…", anchor="project")),
    M("编辑(&E)", "review", A("edit.undo", "撤销", "Ctrl+Z"), A("edit.redo", "重做", "Ctrl+Y"), None,
      A("group.duplicate", "复制这一组", "Ctrl+D"), None, A("edit.delete_member", "删除这一行（只删当前语言）"),
      A("group.delete", "删除这一组（原声和翻译都删）", "Del")),
    M("字幕(&S)", "review", A("group.confirm_next", "确认并播放下一组", "Ctrl+Enter"), A("group.confirm", "确认这一组", "Ctrl+Shift+Enter"),
      A("group.unconfirm", "取消确认"), None, A("group.merge", "与下一组合并", "Ctrl+J"), A("group.split", "在光标处拆分", "Ctrl+K"),
      A("group.unlink", "解除配对", "Ctrl+U"), M("分界", "review",
        A("group.head_to_prev", "开头一段 → 上一组", "Alt+Up"), A("group.tail_to_next", "结尾一段 → 下一组", "Alt+Down"),
        A("group.take_from_prev", "拿回上一组结尾一段", "Alt+Shift+Up"), A("group.take_from_next", "拿回下一组开头一段", "Alt+Shift+Down"), None,
        A("group.cursor_to_prev", "光标前面移到上一组", "Ctrl+Shift+Up"), A("group.cursor_to_next", "光标后面移到下一组", "Ctrl+Shift+Down")), None,
      A("group.add_original", "补上原声（日语）…"), A("group.add_translation", "补上翻译（繁中）…"), A("group.ai_listen", "AI听译…", anchor="ai-transcribe"), None,
      M("转为", "review", A("group.to_song", "歌曲（K 轴）…"), A("group.to_dialogue", "普通台词…"), A("group.to_sign", "屏幕字…")),
      M("位置", "position", *(A("position." + key, text, anchor="position") for key, text in
        (("left", "左侧说话人"), ("right", "右侧说话人"), ("top", "移到顶部"), ("bottom", "移到底部"), ("alt", "底部第二人"))), None,
        A("position.restore", "恢复原位置", anchor="position")),
      A("watch.transform", "自由变换", "T", "transform"), None,
      A("group.hide", "隐藏（不显示、不导出）", "Ctrl+H", "hide-mark", "check"), A("group.mark", "标记待复查", "Ctrl+M", "hide-mark", "check"),
      A("group.mark_note", "编辑标记备注…", anchor="hide-mark")),
    M("时间轴(&T)", "timing", A("time.set", "设置开始和结束时间…", "Ctrl+T", "timing"), A("time.shift", "整体平移…", "Ctrl+Shift+T", "timing"),
      A("time.adopt", "采用建议时间", "C", "timing"), A("time.start_minus", "开始时间 −1 帧", "[", "timing"),
      A("time.start_plus", "开始时间 +1 帧", "]", "timing"), A("time.end_minus", "结束时间 −1 帧", "Shift+[", "timing"),
      A("time.end_plus", "结束时间 +1 帧", "Shift+]", "timing"), None,
      M("统一两种语言的时间", "unify-times", A("time.unify_original", "用原声字幕的时间", anchor="unify-times"),
        A("time.unify_translation", "用翻译字幕的时间", anchor="unify-times")),
      A("time.autotime", "自动调轴（按视频音频）", anchor="autotime"), A("time.dialogue_align", "精确对话轴（按台词对齐）…", anchor="dialogue-align"), None,
      M("K 轴", "karaoke", A("k.tap", "按住 K 跟唱", "K", "karaoke"),
        M("自动 K 轴", "karaoke-auto", *(A("k.auto_" + scope, label, anchor="karaoke-auto") for scope, label in
          (("line", "这一句"), ("selected", "选中的句"), ("song", "整首歌"))), None,
          A("k.suggest_onset", "按起音（旧方法，只这一句）", anchor="karaoke-auto")),
        A("k.adopt", "采用建议", "Enter", "karaoke"), A("k.add_rest", "后面加空拍", "Shift+K", "karaoke"),
        A("k.remove_rest", "删掉空拍", "Del", "karaoke"), A("k.clear", "清除 K 轴", anchor="karaoke"),
        A("k.confirm", "确认 K 轴", anchor="karaoke"), None,
        M("歌词", "karaoke", A("k.mapping", "对应…", anchor="karaoke"), A("k.english", "英文按音节拆开", anchor="karaoke"),
          A("k.import", "读入原字幕的 K 轴", anchor="karaoke"),
          *(M(label, "karaoke", *(A(f"k.{kind}_{scope}", name + dots, anchor="karaoke") for scope, name in
            (("line", "这一句"), ("selected", "选中的句"), ("song", "整首歌")))) for kind, label, dots in
            (("redistribute", "重新分配翻译拍", ""), ("align_times", "翻译时间对齐原声", ""), ("fade", "淡入淡出", "…")))),
        M("特效", "karaoke-fx", *(D(label, "karaoke-fx", "radio") for label in
          ("渐变扫过", "瞬间变色", "描边先出", "无", "放大弹出", "发光", "变色 / 彩虹", "逐字淡入淡出", "擦除", "跳动", "抖动", "星光 / 粒子", "飞入飞出", "字形变化", "惊恐颤抖")), None,
          A("k.parameters", "参数…", anchor="karaoke-fx"))),
      M("打轴方式", "timing", D("Aegisub 式：左键开始、右键结束（推荐）", "timing", "radio"),
        D("点块选中：左键点字幕块选中，Shift+左键开始、Shift+右键结束", "timing", "radio"))),
    M("注音(&R)", "readings", A("ruby.edit", "改读音", "F2", "readings"), A("ruby.regen", "重新生成振假名（保留确认过的）", "Ctrl+G", "readings"),
      A("ruby.regen_all", "重新生成振假名（包括确认过的）", "Ctrl+Shift+G", "readings"), None,
      A("project.words", "工程词语与分词规则…", anchor="project-readings"), A("project.ruby_settings", "振假名设置…", anchor="readings"),
      A("project.pinyin_settings", "拼音设置…", anchor="pinyin")),
    M("播放(&V)", "playback", A("play.toggle", "播放 / 暂停", "Space", "playback"), A("play.step_mode", "逐段播放", "Ctrl+Shift+Space", "playback", "check"), None,
      A("play.group", "重播本组", "R", "playback"), *(A("play." + key.lower(), text, key, "playback") for key, text in
        (("S", "播放本组"), ("Q", "播放开始前 0.5 秒"), ("W", "播放结束后 0.5 秒"), ("D", "播放开头 0.5 秒"), ("E", "播放结尾 0.5 秒"))), None,
      A("nav.prev", "上一组", "PgUp", "playback"), A("nav.next", "下一组", "PgDown", "playback"),
      A("play.prev_group", "播放上一组", "Ctrl+Left", "playback"), A("play.next_group", "播放下一组", "Ctrl+Right", "playback"),
      A("play.frame_back", "上一帧", "Left", "playback"), A("play.frame_forward", "下一帧", "Right", "playback"), None,
      A("play.follow_list", "播放时列表跟随", anchor="playback", kind="check")),
    M("视图(&W)", "review", A("view.watch_timing", "观剧模式：看片打轴（视频 + 波形常驻）", "F10", "watch", "check"),
      A("view.watch_full", "观剧模式：全屏观看（点字幕才出波形）", "F11", "watch", "check"), A("view.karaoke", "卡拉 OK 模式", "F9", "karaoke", "check"), None,
      M("布局", "review", D("视频优先（推荐）", "review", "radio"), D("列表优先", "review", "radio"), D("左右分栏", "review", "radio")),
      A("view.source_panel", "原文", kind="check"), A("view.people_panel", "人物", kind="check"), A("view.dock_reset", "原文、人物面板放回右侧"), None,
      M("观剧模式选项", "watch", A("watch.speakers", "显示说话人", anchor="watch", kind="check"),
        M("说话人按", "watch", D("原声", "watch", "radio"), D("翻译", "watch", "radio")),
        A("watch.frames", "暂停时显示字幕外框", anchor="watch", kind="check"), A("watch.frame_color", "字幕外框颜色…", anchor="watch"))),
    M("显示规则(&D)", "display-rules", A("display.hide_period", "隐藏日文句末「。」（只影响预览和导出，原文不变）", anchor="readings", kind="check"),
      M("中文显示为", "readings", *(D(t, kind="radio") for t in ("原样（不转换）", "简体", "繁体")), None,
        *(D(t) for t in ("对白", "歌词", "屏幕字等", "旁注（中文行）", "日文行上不含假名的旁注", "{@Raw=…} 里的文字"))),
      M("繁体用字", "readings", *(D(t, kind="radio") for t in ("台湾（默认）", "香港", "标准繁体")), D("同时转换用语（例：软件↔軟體）")),
      M("两行合成一行（中间两个空格）", "readings", *(M(lane, "readings", *(D(t) for t in ("对白", "歌词", "屏幕字等"))) for lane in ("翻译（繁中）", "原声（日语）"))), None,
      M("标点方案", "display-rules", D("（还没有方案）", "display-rules", "plain"), None, A("display.punctuation", "标点和空格…", anchor="display-rules")),
      A("display.replacements", "批量替换…", anchor="display-replacements")),
    M("工具(&O)", "review", M("检查", "review", A("tools.check", "顺序、原文去向、待办"), A("tools.check_fonts", "字体（缺字体、缺字）", anchor="fonts"),
      A("tools.check_selfref", "自称（私 的读法）…", anchor="readings")),
      M("审议", "review-packet", A("tools.export_review", "导出审议包（发布前给大模型复查）…", anchor="review-packet"), A("tools.import_review", "导回审议意见…", anchor="review-answers")),
      M("重复段落", "passages", A("tools.passage_library", "重复段落库…", anchor="passages"), A("tools.record_passage", "记为重复段落…", anchor="passages"),
        A("tools.find_passage", "查找并套用…", anchor="passages")),
      M("原字幕修正", "add-episode", A("tools.brackets_to_text", "原字幕括号里的片假名不是注音：放回正文…", anchor="add-episode")), None,
      A("tools.shortcuts", "快捷键…", anchor="keys"), None,
      M("高级", "fonts", A("tools.private", "私有功能", anchor="fonts", kind="check"), A("tools.font_library", "字体库位置…", anchor="fonts"), A("tools.clear_cache", "清除排版缓存", anchor="fonts"))),
    M("帮助(&H)", "project", A("help.manual", "使用说明", "F1", "project"), A("help.shortcuts", "快捷键一览", anchor="keys"),
      A("help.changes", "版本记录", anchor="changes"), A("help.about", "关于双语字幕工坊", anchor="project")),
)


def entries(tree=MENU_TREE):
    for item in tree:
        if item is not None:
            yield item
            if item[-1] is not None:
                yield from entries(item[-1])


MENU_IDS = {item[1] for item in entries() if item[1]}
NEW_MENU_IDS = set('''group.duplicate group.confirm group.unconfirm group.cursor_to_prev group.cursor_to_next
    group.add_original group.add_translation group.ai_listen group.hide group.mark group.mark_note
    time.set time.adopt time.start_minus time.start_plus time.end_minus time.end_plus ruby.edit ruby.regen ruby.regen_all
    nav.prev nav.next play.prev_group play.next_group play.frame_back play.frame_forward tools.shortcuts'''.split())


class MenuTreeTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose

    def test_whole_r2_tree_text_order_separators_kinds_primary_keys_and_anchors(self):
        w = self.window
        w.project.languages = {"original": "ja", "translation": "zh-Hant"}
        w._update_group_menu_actions()
        def compare(menu, expected):
            actual = menu.actions()
            self.assertEqual(len(actual), len(expected), menu.objectName() or str(menu))
            for action, item in zip(actual, expected):
                if item is None:
                    self.assertTrue(action.isSeparator())
                    continue
                text, action_id, key, anchor, kind, children = item
                with self.subTest(text=text):
                    self.assertFalse(action.isSeparator())
                    self.assertEqual(action.property("enabled_text") or action.text(), text)
                    self.assertEqual(action.property("action_id"), action_id)
                    self.assertEqual(action.property("help_anchor"), anchor)
                    if action_id and action_id.startswith("k."):
                        self.assertEqual(action.shortcuts(), [])
                        self.assertEqual(action.text().split("\t")[1:] if key else [], [key] if key else [])
                    else:
                        self.assertEqual(action.shortcut(), QKeySequence(key))
                    self.assertEqual(action.isCheckable(), kind in ("check", "radio"))
                    group = action.actionGroup()
                    self.assertEqual(bool(group and group.isExclusive()), kind == "radio")
                    self.assertEqual(action.menu() is not None, children is not None)
                    if not (action_id and action_id.startswith("k.")):
                        self.assertNotIn("\t", action.text())
                    if children is not None:
                        compare(action.menu(), children)
        compare(w.menuBar(), MENU_TREE)

    def test_karaoke_separate_check_and_saved_layout_round_trip(self):
        w = self.window
        self.assertEqual(set(w.layout_actions), {"video", "list", "columns"})
        for layout in ("list", "karaoke", "columns", "karaoke", "video"):
            w.set_edit_layout(layout)
            self.assertEqual(self.settings.value("layouts/current"), layout)
            self.assertEqual(w.act_karaoke.isChecked(), layout == "karaoke")
            self.assertEqual([k for k, a in w.layout_actions.items() if a.isChecked()], [] if layout == "karaoke" else [layout])
        w.act_karaoke.trigger()
        self.assertEqual(w.edit_layout, "karaoke")
        w.act_karaoke.trigger()
        self.assertEqual(w.edit_layout, "video")
        self.settings.setValue("layouts/current", "karaoke")
        from subkit_app import mainwindow as mw
        other = mw.MainWindow()
        try:
            self.assertEqual(other.edit_layout, "karaoke")
            self.assertTrue(other.act_karaoke.isChecked())
            self.assertFalse(any(a.isChecked() for a in other.layout_actions.values()))
        finally:
            other.close()
            other.deleteLater()


class NewMenuItemsTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select

    def test_new_menu_commands_reach_original_handlers_and_current_selection_once(self):
        w = self.window
        self.group("a")
        self.group("b")
        self.select("a")
        routes = {
            "group.duplicate": ("_duplicate_groups", (["a"],), {}),
            "group.confirm": ("_confirm", (), {"move": False}), "group.unconfirm": ("_confirm_groups", (["a"], False), {}),
            "group.cursor_to_prev": ("_move_at_cursor", (True,), {}), "group.cursor_to_next": ("_move_at_cursor", (False,), {}),
            "group.add_original": ("_add_language_dialog", (["a"], "original"), {}),
            "group.add_translation": ("_add_language_dialog", (["a"], "translation"), {}),
            "group.ai_listen": ("_start_listening", ("a",), {}), "group.mark_note": ("_mark_note_dialog", (["a"],), {}),
            "time.set": ("_group_times_dialog", (["a"],), {}), "time.adopt": ("_adopt_timing", (), {}),
            "ruby.edit": ("_f2", (), {}), "ruby.regen": ("_regen_current", (False,), {}), "ruby.regen_all": ("_regen_current", (True,), {}),
            "nav.prev": ("_step_group", (-1,), {}), "nav.next": ("_step_group", (1,), {}),
            "play.prev_group": ("_play_neighbour", (-1,), {}), "play.next_group": ("_play_neighbour", (1,), {}),
            "play.frame_back": ("_frame", (False,), {}), "play.frame_forward": ("_frame", (True,), {}),
        }
        for name, edge, delta in (("start_minus", "start", -1), ("start_plus", "start", 1), ("end_minus", "end", -1), ("end_plus", "end", 1)):
            routes["time." + name] = ("_nudge", (edge, delta), {})
        self.assertEqual(set(routes) | {"group.hide", "group.mark", "tools.shortcuts"}, NEW_MENU_IDS)
        for selected in (["a"], ["a", "b"]):
            self.select(*selected)
            for action_id, (name, args, kwargs) in routes.items():
                args = tuple(selected if arg == ["a"] else arg for arg in args)
                with self.subTest(action=action_id, selected=selected), patch.object(w, name) as called, patch.object(w, "_typing", return_value=False):
                    action = w.action_hub.action(action_id)
                    action.setEnabled(True)
                    action.trigger()
                    called.assert_called_once_with(*args, **kwargs)

    def test_six_new_default_keys_use_current_selection_once_and_menu_destination(self):
        w = self.window
        self.group("a")
        self.group("b")
        w.show()
        w.activateWindow()
        w.player.setFocus()
        self.app.processEvents()
        routes = {"Ctrl+D": ("_duplicate_groups", None), "Ctrl+T": ("_group_times_dialog", None),
                  "Ctrl+Shift+T": ("_shift_groups_dialog", None), "Ctrl+H": ("_review_flag", "hidden"),
                  "Ctrl+M": ("_review_flag", "marked"), "Ctrl+E": ("export", None)}
        for scene in ("main", "watch"):
            w.action_hub.set_scene(scene)
            for selected in (["a"], ["a", "b"]):
                self.select(*selected)
                w.player.setFocus()
                for key, (name, flag) in routes.items():
                    with self.subTest(scene=scene, key=key, selected=selected), patch.object(w, name) as called:
                        stroke = QKeySequence(key)[0]
                        QTest.keyClick(w.player, Qt.Key(stroke.key()), stroke.keyboardModifiers())
                        self.app.processEvents()
                        if flag:
                            called.assert_called_once_with(selected, flag, True)
                        elif name == "export":
                            called.assert_called_once_with()
                        else:
                            called.assert_called_once_with(selected)

    def test_flags_single_and_mixed_selection_menu_and_keys_toggle_all(self):
        w = self.window
        self.group("a")
        self.group("b")
        for flag, action_id in (("hidden", "group.hide"), ("marked", "group.mark")):
            for selected in (["a"], ["a", "b"]):
                self.select(*selected)
                for existing in (set(), {"a"}, {"a", "b"}):
                    self.select(*selected)
                    for g in self.doc.groups:
                        g.flags = [flag] if g.id in existing else []
                    w._update_group_menu_actions()
                    action = w.action_hub.action(action_id)
                    all_flagged = all(gid in existing for gid in selected)
                    self.assertEqual(action.isChecked(), all_flagged)
                    with patch.object(w, "_review_flag") as called:
                        action.trigger()
                        called.assert_called_once_with(selected, flag, not all_flagged)
                    # Exercise the real undoable flag update as well.
                    action.trigger()
                    self.assertEqual([flag in self.doc.group(gid).flags for gid in selected], [not all_flagged] * len(selected))
                    self.assertEqual(action.isChecked(), not all_flagged)
                    self.select(*selected)
                    action.trigger()
                    self.assertEqual([flag in self.doc.group(gid).flags for gid in selected], [all_flagged] * len(selected))

    def test_add_language_listening_time_availability_and_reason_suffix_restore(self):
        w = self.window
        self.group("a")
        self.group("b", kind="song")
        for gids in (["a"], ["b"], ["a", "b"]):
            self.select(*gids)
            w._update_group_menu_actions()
            listen = w.action_hub.action("group.ai_listen")
            reason = unavailable_reason(self.doc, gids, self.project.languages)
            self.assertEqual(listen.isEnabled(), not reason)
            self.assertEqual(listen.text(), "AI听译…" + (f"（{reason}）" if reason else ""))
            self.assertEqual(w.action_hub.action("time.set").isEnabled(), len(gids) == 1)
            self.assertTrue(w.action_hub.action("group.add_original").isEnabled())
            self.assertFalse(w.action_hub.action("group.add_translation").isEnabled())
        self.select("a")
        m = copy.deepcopy(self.doc.members["m_a"])
        m.id, m.lane, m.lang = "jp", "original", "ja"
        self.doc.members[m.id] = m
        self.doc.group("a").members.append(m.id)
        w._update_group_menu_actions()
        self.assertIn("已经有原文", w.action_hub.action("group.ai_listen").text())
        self.assertFalse(w.action_hub.action("group.add_original").isEnabled())
        self.project.languages["original"] = "en"
        w._update_group_menu_actions()
        self.assertEqual(w.action_hub.action("group.add_original").property("enabled_text"), "补上原声（英语）…")
        w.session = None
        w._update_group_menu_actions()
        for action_id in ("group.duplicate", "group.hide", "group.mark", "group.mark_note", "time.set", "group.ai_listen"):
            self.assertFalse(w.action_hub.action(action_id).isEnabled())
            self.assertIn("先打开一集", w.action_hub.action(action_id).text())

    def test_edit_and_time_menu_refresh_current_selection_before_show(self):
        w = self.window
        self.group("a")
        self.select("a")
        w.edit_menu.aboutToShow.emit()
        self.assertTrue(w.action_hub.action("group.duplicate").isEnabled())
        w.time_menu.aboutToShow.emit()
        self.assertTrue(w.action_hub.action("time.set").isEnabled())
        w.session = None
        w.edit_menu.aboutToShow.emit()
        self.assertFalse(w.action_hub.action("group.duplicate").isEnabled())

    def test_flag_keys_single_and_mixed_selection_choose_all_or_clear(self):
        w = self.window
        self.group("a")
        self.group("b")
        w.show()
        w.activateWindow()
        for action_id, flag, key in (("group.hide", "hidden", "Ctrl+H"), ("group.mark", "marked", "Ctrl+M")):
            for selected in (["a"], ["a", "b"]):
                for existing in (set(), {"a"}, {"a", "b"}):
                    self.select(*selected)
                    for g in self.doc.groups:
                        g.flags = [flag] if g.id in existing else []
                    w._update_group_menu_actions()
                    w.player.setFocus()
                    self.app.processEvents()
                    with patch.object(w, "_review_flag") as called:
                        stroke = QKeySequence(key)[0]
                        QTest.keyClick(w.player, Qt.Key(stroke.key()), stroke.keyboardModifiers())
                        self.app.processEvents()
                        called.assert_called_once_with(selected, flag, not all(gid in existing for gid in selected))


class KMenuTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select

    def test_shared_menus_actions_and_no_auto_options_in_menu_bar(self):
        w, p = self.window, self.window.wave_panel
        self.assertIs(w.k_menu.actions()[-2].menu(), p.lyrics_menu)
        self.assertIs(w.k_menu.actions()[-1].menu(), p.fx_menu)
        self.assertIn(p.lyrics_menu.menuAction(), p.more_menu.actions())
        self.assertIn(p.fx_menu.menuAction(), p.more_menu.actions())
        self.assertEqual([a.text() for a in w.k_auto_menu.actions() if not a.isSeparator()],
                         ["这一句", "选中的句", "整首歌", "按起音（旧方法，只这一句）"])
        for scope, action in p.kauto_scopes.items():
            self.assertIs(w.action_hub.action("k.auto_" + scope), action)
        for kind, scopes in p.scope_actions.items():
            for scope, action in scopes.items():
                self.assertIs(w.action_hub.action(f"k.{kind}_{scope}"), action)
        for name, button in (("mapping", p.btn_kmap), ("english", p.btn_kenglish),
                             ("import", p.btn_kimport), ("parameters", p.btn_kparams)):
            self.assertIs(w.action_hub.action("k." + name), p.menu_actions[button])

    def test_every_menu_command_and_bar_control_emit_identical_signal_once(self):
        w, p = self.window, self.window.wave_panel
        p.kAction.disconnect()
        called = Mock()
        p.kAction.connect(called)
        routes = {"tap": (p.btn_k, "start"), "import": (p.btn_kimport, "import"),
                  "adopt": (p.btn_kadopt, "adopt"), "add_rest": (p.btn_kaddrest, "add_rest"),
                  "remove_rest": (p.btn_kdelrest, "remove_rest"), "clear": (p.btn_kclear, "clear"),
                  "confirm": (p.btn_kconfirm, "confirm"), "mapping": (p.btn_kmap, "mapping"),
                  "english": (p.btn_kenglish, "english"), "suggest_onset": (p.kauto_onset, "suggest")}
        routes.update({"auto_" + scope: (action, "auto:" + scope) for scope, action in p.kauto_scopes.items()})
        routes.update({f"{kind}_{scope}": (action, f"{kind}:{scope}")
                       for kind, scopes in p.scope_actions.items() for scope, action in scopes.items()})
        for name, (control, signal) in routes.items():
            with self.subTest(action=name):
                control.setEnabled(True)
                action = w.action_hub.action("k." + name)
                action.setEnabled(True)
                called.reset_mock()
                (control.click if hasattr(control, "click") else control.trigger)()
                called.assert_called_once_with(signal)
                called.reset_mock()
                action.trigger()
                called.assert_called_once_with(signal)
        p.kParameters.disconnect()
        params = Mock()
        p.kParameters.connect(params)
        p.btn_kparams.click()
        params.assert_called_once_with()
        params.reset_mock()
        w.action_hub.action("k.parameters").trigger()
        params.assert_called_once_with()
        p.kEffect.disconnect()
        effect = Mock()
        p.kEffect.connect(effect)
        p.fx_menu.actions()[1].trigger()
        effect.assert_called_once_with(p.k_fx.itemData(1))
        self.assertEqual(p.k_fx.currentIndex(), 1)

    def test_song_dialogue_no_selection_and_control_reasons_refresh_before_show(self):
        w, p = self.window, self.window.wave_panel
        member = self.group("song", "song")
        original = copy.deepcopy(member)
        original.id, original.lane, original.lang = "jp_song", "original", "ja"
        self.doc.members[original.id] = original
        self.doc.group("song").members.append(original.id)
        self.doc.structure_changed()
        self.project.languages["original"] = "ja"
        available = patch("subkit.kalign.available", return_value=(True, ""))
        available.start()
        self.addCleanup(available.stop)
        self.group("dialogue")
        for gid in ("song", "dialogue", None, "song"):
            with self.subTest(group=gid):
                if gid is None:
                    w.current_gid = None
                    w.table.clearSelection()
                    w._refresh_k()
                else:
                    self.select(gid)
                w.k_menu.aboutToShow.emit()
                ready = gid == "song"
                for action_id, control in w._k_menu_controls.items():
                    action = w.action_hub.action(action_id)
                    self.assertEqual(action.isEnabled(), ready and control.isEnabled(), action_id)
                    self.assertEqual(action.toolTip(), (control.toolTip() or action.text().split("\t")[0].removesuffix("…")) if ready else
                                     "（先选中字幕组）" if gid is None else "（不是歌曲）")
                    self.assertTrue(action.statusTip())
                    if ready:
                        self.assertNotIn("不是歌曲", action.statusTip())
                        self.assertNotIn("先选中字幕组", action.statusTip())
                self.assertEqual(w.action_hub.action("k.auto_line").isEnabled(), ready)
                self.assertEqual(p.lyrics_menu.isEnabled(), ready)
                self.assertEqual(p.fx_menu.isEnabled(), ready)
        p.btn_kclear.setEnabled(False)
        p.btn_kclear.setToolTip("清除受阻原因")
        p.btn_kclear.setStatusTip("状态原因")
        w.k_menu.aboutToShow.emit()
        action = w.action_hub.action("k.clear")
        self.assertFalse(action.isEnabled())
        self.assertEqual(action.toolTip(), "清除受阻原因")
        self.assertEqual(action.statusTip(), "状态原因")

    def test_k_scene_keys_display_registry_values_and_fire_once(self):
        w = self.window
        self.group("song", "song")
        self.select("song")
        w.show()
        w.activateWindow()
        w.player.setFocus()
        self.app.processEvents()
        w._k_action = Mock()
        w.wave.selected_rest = Mock(return_value=("translation", 0))
        w.wave.k_suggestions = ("m_song", [], "syllable")
        for action_id, key, name in (("k.tap", "K", "start"), ("k.add_rest", "Shift+K", "add_rest"),
                                     ("k.remove_rest", "Del", "remove_rest"), ("k.adopt", "Enter", "adopt")):
            with self.subTest(action=action_id):
                action = w.action_hub.action(action_id)
                self.assertEqual(action.shortcuts(), [])
                self.assertEqual(action.text().split("\t")[-1], key)
                w._k_action.reset_mock()
                stroke = QKeySequence(key)[0]
                QTest.keyClick(w.player, Qt.Key(stroke.key()), stroke.keyboardModifiers())
                w._k_action.assert_called_once_with(name)
        w.action_hub.registry.set_keys("k.add_rest", "k", "Ctrl+Alt+F7")
        w.k_menu.aboutToShow.emit()
        self.assertEqual(w.action_hub.action("k.add_rest").text().split("\t")[-1], "Ctrl+Alt+F7")
        self.assertEqual(w.action_hub.action("k.add_rest").shortcuts(), [])


class HiddenMnemonicTests(unittest.TestCase):
    setUpClass = fixtures.ConversionUI.__dict__["setUpClass"]
    setUp = fixtures.ConversionUI.setUp
    _dispose = fixtures.ConversionUI._dispose
    group = fixtures.ConversionUI.group
    select = fixtures.ConversionUI.select

    def test_f10_f11_all_menus_alt_t_priority_and_no_visible_bindings(self):
        w = self.window
        self.group("dialogue")
        self.select("dialogue")
        w.show()
        w.activateWindow()
        self.app.processEvents()
        w.player._hide_timer = QTimer(w.player)
        w.player.pause = Mock()
        self.project.settings = Mock(return_value={})
        try:
            for pinned in (True, False):
                w._watch_key(pinned)
                self.app.processEvents()
                self.assertTrue(w.watch_mode)
                self.assertTrue(w.menuBar().isHidden())
                with ExitStack() as stack:
                    popups = [stack.enter_context(patch.object(a.menu(), "popup")) for a in w.menuBar().actions()]
                    transform = stack.enter_context(patch.object(w, "_start_transform"))
                    w.watch.selected = "dialogue"
                    for index, letter in enumerate("FPESTRVWDOH"):
                        for popup in popups:
                            popup.reset_mock()
                        transform.reset_mock()
                        w.player.setFocus()
                        QTest.keyClick(w.player, Qt.Key(ord(letter)), Qt.KeyboardModifier.AltModifier)
                        self.app.processEvents()
                        for other, popup in enumerate(popups):
                            if letter != "T" and other == index:
                                popup.assert_called_once_with(w.player.mapToGlobal(QPoint(8, 8)))
                            else:
                                popup.assert_not_called()
                        if letter == "T":
                            transform.assert_called_once_with(only_line=True)
                        else:
                            transform.assert_not_called()
                    # No QAction duplicates these native/dispatcher mnemonics.
                    for action in w.actions():
                        self.assertFalse({QKeySequence("Alt+" + c).toString() for c in "FPESTRVWDOH"}
                                         & {k.toString() for k in action.shortcuts()})
                w.set_watch_mode(False)
                self.app.processEvents()
                self.assertFalse(w.menuBar().isHidden())
                from PySide6.QtCore import QEvent
                from PySide6.QtGui import QKeyEvent
                for letter in "FPESTRVWDOH":
                    event = QKeyEvent(QEvent.Type.KeyPress, ord(letter), Qt.KeyboardModifier.AltModifier)
                    self.assertIsNone(w._hidden_menu_for_key(event))
        finally:
            if w.watch_mode:
                w.set_watch_mode(False)

    def test_existing_rebound_watch_key_wins_over_mnemonic(self):
        w = self.window
        w.watch_mode = True
        self.addCleanup(setattr, w, "watch_mode", False)
        w.action_hub.registry.set_keys("watch.transform_line", "watch", "Alt+F")
        w.watch.selected = "dialogue"
        w.menuBar().hide()
        w.show()
        w.activateWindow()
        w.player.setFocus()
        self.app.processEvents()
        with patch.object(w, "_start_transform") as transform, patch.object(w.menuBar().actions()[0].menu(), "popup") as popup:
            QTest.keyClick(w.player, Qt.Key.Key_F, Qt.KeyboardModifier.AltModifier)
            transform.assert_called_once_with(only_line=True)
            popup.assert_not_called()


class OldMenuPathTests(unittest.TestCase):
    def test_no_old_menu_paths_or_menu_phrasings_in_src(self):
        import re
        from pathlib import Path
        patterns = (
            r"文件\s*→\s*导出整季发布用 ASS", r"集\s*→\s*(?:统一两种语言的时间|自动调轴|检查|查找并套用重复段落)",
            r"视图\s*→\s*(?:字幕外框颜色|暂停时显示字幕外框|显示规则)",
            r"文件\s*→\s*(?:工程词语与分词规则|重复段落库|振假名设置|拼音设置)",
            r"(?:「集」|集)菜单.{0,8}(?:检查|自动调轴|重复段落|统一两种语言的时间)",
            r"(?:「文件」|文件)菜单.{0,8}(?:重复段落|工程词语|振假名设置|拼音设置)",
        )
        found = []
        for path in (Path(__file__).resolve().parents[1] / "src").rglob("*.py"):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if any(re.search(pattern, line) for pattern in patterns):
                    found.append(f"{path}:{number}: {line.strip()}")
        self.assertEqual(found, [])
