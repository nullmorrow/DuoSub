"""Dialogs: new project, add episode (sources + style roles), merge settings, export."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QRadioButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from subkit import proc
from subkit.languages import LANGUAGE_NAMES, lane_label
from subkit.ass import AssDocument
from subkit.ass.text import plain_text
from subkit.encoding import decode_bytes
from subkit.core.convert import FAMILIES, plan_conversion, song_style_text, song_text
from subkit.core.display import DisplayRules, MERGE_KINDS
from subkit.model.document import Document

from ..help import dialog_buttons

LANGS = [(code, LANGUAGE_NAMES[code]) for code in ("ja", "en", "ko", "zh-Hant", "zh-Hans")]
LANGS += [(code, name) for code, name in LANGUAGE_NAMES.items() if code not in dict(LANGS)]
ROLE_CHOICES = [("dialogue", "对白"), ("top", "顶部对白"), ("song_jp", "歌词（原文）"), ("song_cn", "歌词（翻译）"), ("keep", "屏幕字、制作名单等（原样保留）")]


def _path_row(edit: QLineEdit, button_text: str, pick) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit, 1)
    b = QPushButton(button_text)
    b.clicked.connect(pick)
    h.addWidget(b)
    return w


class NewProjectDialog(QDialog):
    help_anchor = "project"
    """Project name, folder and languages on top; the visual style template editor below (the template is set up
    before the project exists — and can be changed later from 文件 → 样式模板)."""

    def __init__(self, parent=None, *, scheme: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("新建工程")
        self.resize(1380, 900)
        v = QVBoxLayout(self)
        f = QFormLayout()
        self.name = QLineEdit()
        self.folder = QLineEdit()
        self.original = QComboBox()
        self.translation = QComboBox()
        for code, label in LANGS:
            self.original.addItem(f"{label} ({code})", code)
            self.translation.addItem(f"{label} ({code})", code)
        self.original.setCurrentIndex(0)
        self.translation.setCurrentIndex(3)
        self.dict_from = QLineEdit()
        self.dict_from.setPlaceholderText("可选：从文件夹导入 readings.json / characters.tsv")
        langs = QWidget()
        lh = QHBoxLayout(langs)
        lh.setContentsMargins(0, 0, 0, 0)
        lh.addWidget(QLabel("原声"))
        lh.addWidget(self.original)
        lh.addSpacing(16)
        lh.addWidget(QLabel("翻译"))
        lh.addWidget(self.translation)
        lh.addStretch(1)
        f.addRow("工程名称", self.name)
        f.addRow("工程文件夹", _path_row(self.folder, "选择…", self._pick_folder))
        f.addRow(QLabel("<span style='color:#6e7781'>工程数据（词库、人物、工作文档）放在这里，和程序、视频分开；原字幕只会被复制一份只读副本。</span>"))
        f.addRow("语言", langs)
        f.addRow("导入词库", _path_row(self.dict_from, "选择…", self._pick_dict))
        v.addLayout(f)
        from PySide6.QtWidgets import QGroupBox

        from .template_editor import TemplateEditor

        box = QGroupBox("样式模板：字体、颜色和位置（预览就是发布出去的样子；建好工程后也能在 文件 → 样式模板 里改）")
        bl = QVBoxLayout(box)
        # default: the scheme used last time, else the tool's long-standing default (少女革命 colours and fonts)
        self.editor = TemplateEditor(box, translation_lang=self.translation.currentData(), original_lang=self.original.currentData(), scheme=scheme or None)
        bl.addWidget(self.editor)
        v.addWidget(box, 1)
        self.translation.currentIndexChanged.connect(lambda _i: self.editor.set_translation_lang(self.translation.currentData()))
        self.original.currentIndexChanged.connect(lambda _i: self.editor.set_original_lang(self.original.currentData()))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("建立工程")
        dialog_buttons(self, bb)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择（或新建）工程文件夹")
        if d:
            self.folder.setText(d)
            if not self.name.text():
                self.name.setText(Path(d).name)

    def _pick_dict(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "词库所在文件夹")
        if d:
            self.dict_from.setText(d)


def probe_audio_tracks(video: Path) -> list[tuple[int, str]]:
    try:
        out = proc.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=index,codec_name,channels:stream_tags=language,title", "-of", "json", str(video)],
                       capture_output=True, text=True, timeout=30).stdout
        streams = json.loads(out).get("streams", [])
    except Exception:
        return []
    res = []
    for k, s in enumerate(streams):
        tags = s.get("tags", {})
        res.append((k + 1, f"音轨 {k + 1}：{tags.get('language', '?')} {s.get('codec_name', '')} {s.get('channels', '')}ch {tags.get('title', '')}".strip()))
    return res


class AddEpisodeDialog(QDialog):
    help_anchor = "add-episode"
    """Pick video, translation and original subtitles; confirm the translation file's style roles."""

    def __init__(self, parent=None, *, suggested_id: str = "S01E001", start_dir: str = "", default_roles: Optional[dict] = None,
                 languages=None, display_settings: Optional[dict] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("添加一集")
        self.setMinimumWidth(760)
        self.start_dir = start_dir
        self.default_roles = default_roles or {}
        self.languages = languages or {"original": "ja", "translation": "zh-Hant"}
        v = QVBoxLayout(self)
        f = QFormLayout()
        self.eid = QLineEdit(suggested_id)
        self.video = QLineEdit()
        self.audio = QComboBox()
        self.cn = QLineEdit()
        self.jp = QLineEdit()
        self.eid.setPlaceholderText("例：S01E017")
        self.video.setPlaceholderText("可以不选，以后再加")
        self.audio.setToolTip("视频有多条音轨时选原声那条，波形和自动调轴用它")
        self.audio.setEnabled(False)
        self.found_notes = {}
        f.addRow("集编号", self.eid)
        f.addRow("视频", _path_row(self.video, "选择…", self._pick_video))
        f.addRow("音轨", self.audio)
        for lane, edit in (("translation", self.cn), ("original", self.jp)):
            row = _path_row(edit, "选择…", lambda checked=False, edit=edit: self._pick_sub(edit))
            note = QLabel("同目录找到的，可以换")
            note.setStyleSheet("color:#6e7781; font-size:11px")
            note.hide()
            row.layout().addWidget(note)
            self.found_notes[lane] = note
            f.addRow(lane_label(lane, languages) + "字幕", row)
        v.addLayout(f)
        v.addWidget(QLabel("翻译字幕里有这几种样式（Style），每种是什么内容？已按样式名猜好，请检查："))
        self.roles = QTableWidget(0, 4)
        self.roles.setHorizontalHeaderLabels(["样式", "行数", "例句", "内容"])
        self.roles.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.roles.verticalHeader().setVisible(False)
        v.addWidget(self.roles, 1)
        merge_form = QFormLayout()
        self.merge_checks = {}
        rules = DisplayRules.from_settings({"display": display_settings or {}})
        for lane in ("translation", "original"):
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(QLabel(lane_label(lane, self.languages) + "："))
            for kind, label in MERGE_KINDS:
                check = QCheckBox(label)
                check.setChecked((lane, kind) in rules.merge_lines)
                check.setToolTip("两行合成一行（中间两个空格）；只影响预览和导出，原文不变")
                self.merge_checks[lane, kind] = check
                layout.addWidget(check)
            layout.addStretch(1)
            merge_form.addRow("合成一行" if lane == "translation" else "", row)
        v.addLayout(merge_form)
        self.cn.textChanged.connect(self._load_styles)
        self.missing_hint = QLabel()
        self.missing_hint.setStyleSheet("color:#c62828")
        v.addWidget(self.missing_hint)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok_button = bb.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("确定")
        dialog_buttons(self, bb)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        for edit in (self.eid, self.cn, self.jp):
            edit.textChanged.connect(self._validate_fields)
        self.video.textChanged.connect(self._video_changed)
        self._validate_fields()

    def _validate_fields(self) -> None:
        missing = []
        for name, valid in (("集编号", bool(self.eid.text().strip())),
                            ("翻译字幕", Path(self.cn.text()).is_file()),
                            ("原声字幕", Path(self.jp.text()).is_file())):
            if not valid:
                missing.append(name)
        self.missing_hint.setText("还缺：" + "、".join(missing) if missing else "")
        self.ok_button.setEnabled(not missing)

    def _video_changed(self) -> None:
        self.audio.clear()
        video = Path(self.video.text())
        self.audio.setEnabled(video.is_file())
        if video.is_file():
            for k, label in probe_audio_tracks(video):
                self.audio.addItem(label, k)
            self._fill_subtitles(video)

    def _fill_subtitles(self, video: Path) -> None:
        aliases = {"jp": "ja", "jpn": "ja", "eng": "en", "chs": "zh-Hans", "cht": "zh-Hant",
                   "zh-cn": "zh-Hans", "zh-tw": "zh-Hant", "zh-hans": "zh-Hans", "zh-hant": "zh-Hant"}
        cands = sorted(p for p in video.parent.iterdir() if p.is_file()
                       and p.suffix.lower() in (".ass", ".ssa", ".srt")
                       and p.stem.startswith(video.stem + "."))
        for lane, edit in (("translation", self.cn), ("original", self.jp)):
            self.found_notes[lane].hide()
            edit.setPlaceholderText("同目录的字幕：" + "、".join(p.name for p in cands[:4]) if cands else "选择字幕")
            if edit.text():
                continue
            matches = []
            for p in cands:
                tokens = p.stem[len(video.stem) + 1:].lower().split(".")
                codes = {aliases.get(token, token) for token in tokens}
                if self.languages[lane] in codes:
                    matches.append(p)
            # Prefer the plain name (video.zh-Hant.ass) over working copies such as video.zh-Hant.v6_4.ass.
            exact = [p for p in matches if "." not in p.stem[len(video.stem) + 1:]]
            if len(exact) == 1:
                matches = exact
            if len(matches) == 1:
                edit.setText(str(matches[0]))
                self.found_notes[lane].show()

    def _pick_video(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "视频", self.start_dir, "视频 (*.mkv *.mp4 *.avi *.m2ts *.ts);;所有文件 (*)")
        if not f:
            return
        self.video.setText(f)
        self.start_dir = str(Path(f).parent)
        stem = Path(f).stem
        m = re.search(r"S(\d+)E(\d+)", stem, re.IGNORECASE)
        if m and self.eid.text() in ("", "S01E001"):
            self.eid.setText(f"S{int(m.group(1)):02d}E{int(m.group(2)):03d}")

    def _pick_sub(self, edit: QLineEdit) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "字幕", self.start_dir, "字幕 (*.ass *.ssa *.srt);;所有文件 (*)")
        if f:
            edit.setText(f)
            self.found_notes["translation" if edit is self.cn else "original"].hide()
            self.start_dir = str(Path(f).parent)

    def _load_styles(self) -> None:
        self.roles.setRowCount(0)
        p = Path(self.cn.text())
        if not p.exists() or p.suffix.lower() not in (".ass", ".ssa"):
            return
        try:
            doc = AssDocument.parse(decode_bytes(p.read_bytes()).text)
        except Exception:
            return
        from subkit.sources.cn import classify

        c = classify(doc, dict(self.default_roles))
        counts = Counter(e.style for e in doc.events if e.kind == "Dialogue")
        examples = {}
        for e in doc.events:
            if e.kind == "Dialogue" and e.style not in examples and plain_text(e.text).strip():
                examples[e.style] = plain_text(e.text, breaks=" ")[:40]
        for st, n in counts.most_common():
            r = self.roles.rowCount()
            self.roles.insertRow(r)
            self.roles.setItem(r, 0, QTableWidgetItem(st))
            self.roles.setItem(r, 1, QTableWidgetItem(str(n)))
            self.roles.setItem(r, 2, QTableWidgetItem(examples.get(st, "")))
            box = QComboBox()
            for code, label in ROLE_CHOICES:
                box.addItem(label, code)
            box.setMinimumContentsLength(max(len(label) for _, label in ROLE_CHOICES))
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            role = c.style_roles.get(st, "keep")
            role = {"song": "song_jp", "drop": "keep"}.get(role, role)
            box.setCurrentIndex(max(0, [x for x, _ in ROLE_CHOICES].index(role) if role in [x for x, _ in ROLE_CHOICES] else 4))
            self.roles.setCellWidget(r, 3, box)
        self.roles.resizeColumnToContents(3)

    def style_roles(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"dialogue": [], "top": [], "song_jp": [], "song_cn": []}
        for r in range(self.roles.rowCount()):
            st = self.roles.item(r, 0).text()
            role = self.roles.cellWidget(r, 3).currentData()
            if role in out:
                out[role].append(st)
        return out

    def merge_settings(self) -> dict:
        return {lane: {kind: self.merge_checks[lane, kind].isChecked() for kind, _label in MERGE_KINDS}
                for lane in ("translation", "original")}

    def _ok(self) -> None:
        self._validate_fields()
        if self.ok_button.isEnabled():
            self.accept()


def glob_escape(s: str) -> str:
    return re.sub(r"([\[\]])", r"[\1]", s)


class MergeDialog(QDialog):
    help_anchor = "add-episode"
    def __init__(self, parent=None, *, filler_mode: str = "keep", semantic: bool = False, semantic_available: bool = True, languages=None,
                 joint: str = "first") -> None:
        super().__init__(parent)
        self.setWindowTitle("合并设置")
        v = QVBoxLayout(self)
        trans, orig = lane_label("translation", languages), lane_label("original", languages)
        v.addWidget(QLabel(f"接下来把{trans}字幕和{orig}字幕逐句配对，配好的放进同一组。"))
        v.addWidget(QLabel("<b>语气词（嗯、啊、えっ 这类）</b>"))
        self.keep = QRadioButton("全部显示（推荐）：不想要的再删，删过的会记住")
        self.compact = QRadioButton("简洁：没有对应翻译、单独成句的原声语气词先藏起来（放进这一组的备注，列表上有标记，可以一键恢复）")
        (self.keep if filler_mode == "keep" else self.compact).setChecked(True)
        self.filler_group = QButtonGroup(self)          # the two radio sets of this dialog are independent
        for button in (self.keep, self.compact):
            self.filler_group.addButton(button)
        v.addWidget(self.keep)
        v.addWidget(self.compact)
        v.addWidget(QLabel("<b>配对方式</b>"))
        semantic_text = "用语义模型辅助配对（LaBSE，本机已有；更准，第一次较慢；不联网）" if semantic_available else "用语义模型辅助配对（LaBSE）（本机没有安装，用不了）"
        self.semantic = QCheckBox(semantic_text)
        self.semantic.setChecked(semantic and semantic_available)
        self.semantic.setEnabled(semantic_available)
        v.addWidget(self.semantic)
        v.addWidget(QLabel(f"<b>一句{trans}拆成几行、{orig}只有一行时</b>"))
        self.joint = {
            "first": QRadioButton(f"每行{trans}各自成组，{orig}只跟第一行（推荐）：后半段用「结尾→下一组」交给下一组"),
            "cut": QRadioButton(f"每行{trans}各自成组，{orig}在空格和标点处自动切开（语序不同时会切错）"),
            "keep": QRadioButton("配成一个多行组（以前的做法）"),
        }
        self.joint_group = QButtonGroup(self)
        for key, button in self.joint.items():
            self.joint_group.addButton(button)
            button.setChecked(key == (joint if joint in self.joint else "first"))
            v.addWidget(button)
        note = QLabel(f"时间以{trans}字幕为准，{orig}字幕原来的时间保留在原文面板里。两边的句子顺序都不变；"
                      f"拆开的组标「边界拿不准」，方便在列表里找到。")
        note.setWordWrap(True)
        v.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        dialog_buttons(self, bb)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def settings(self) -> dict:
        return {"filler_mode": "keep" if self.keep.isChecked() else "compact", "semantic": self.semantic.isChecked(),
                "joint": next(key for key, button in self.joint.items() if button.isChecked())}


class ConvertToSongDialog(QDialog):
    help_anchor = "karaoke"
    def __init__(self, doc: Document, gids: Iterable[str], parent=None, *, styles: Mapping[str, str],
                 template_styles: Iterable[str], settings: QSettings) -> None:
        super().__init__(parent)
        self.setWindowTitle("转为歌曲（K 轴）")
        self.setModal(True)
        self.setMinimumWidth(620)
        self.doc, self.gids = doc, list(gids)
        self.styles, self.template_styles = styles, list(template_styles)
        self.plan = plan_conversion(doc, self.gids, "song", styles=styles, template_styles=self.template_styles)
        last = settings.value("convert/song_family", "IN")
        family = self.plan.default_family or (last if last in FAMILIES else "IN")
        v = QVBoxLayout(self)
        self.explanation = QLabel()
        self.explanation.setTextFormat(Qt.TextFormat.PlainText)
        self.explanation.setWordWrap(True)
        v.addWidget(self.explanation)
        row = QHBoxLayout()
        self.families: dict[str, QRadioButton] = {}
        for fam, text in (("OP", "片头曲 OP"), ("ED", "片尾曲 ED"), ("IN", "插曲 IN")):
            button = QRadioButton(text)
            self.families[fam] = button
            row.addWidget(button)
            button.setChecked(fam == family)
            button.toggled.connect(lambda checked: checked and self._update_plan())
        v.addLayout(row)
        self.style_label = QLabel()
        self.style_label.setTextFormat(Qt.TextFormat.PlainText)
        self.style_label.setWordWrap(True)
        v.addWidget(self.style_label)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("转换")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.plan.gids))
        dialog_buttons(self, bb)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self._update_plan()

    def family(self) -> str:
        return next(fam for fam, button in self.families.items() if button.isChecked())

    def _update_plan(self) -> None:
        self.plan = plan_conversion(self.doc, self.gids, "song", styles=self.styles,
                                    template_styles=self.template_styles, family=self.family())
        self.explanation.setText(song_text(self.plan))
        self.style_label.setText(song_style_text(self.plan))


class ExportDialog(QDialog):
    help_anchor = "export"
    def __init__(self, parent=None, *, default_path: str = "", include_actor: bool = True, include_comments: bool = False,
                 blur: float = 0.0, name_format: str = "short", has_chinese: bool = False, zh_both: bool = False,
                 reference_enabled: bool = False, reference_path: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("导出发布用 ASS")
        self.setMinimumWidth(620)
        f = QFormLayout(self)
        self.path = QLineEdit(default_path)
        f.addRow("保存到", _path_row(self.path, "选择…", self._pick))
        self.actor = QCheckBox("在 Name 栏保留说话人")
        self.actor.setChecked(include_actor)
        self.name_format = QComboBox()
        for label, value in (("常用称呼", "short"), ("全名", "full"), ("只写姓", "family"),
                             ("只写名", "given"), ("原标签", "label")):
            self.name_format.addItem(label, value)
        self.name_format.setCurrentIndex(max(0, self.name_format.findData(name_format)))
        self.name_format.setEnabled(include_actor)
        self.actor.toggled.connect(self.name_format.setEnabled)
        self.comments = QCheckBox("附带翻译字幕原有的注释行（Comment）")
        self.comments.setChecked(include_comments)
        self.blur = QDoubleSpinBox()
        self.blur.setRange(0, 10)
        self.blur.setSingleStep(0.5)
        self.blur.setValue(blur)
        self.zh_both = QCheckBox("同时导出简体和繁体两份")
        self.zh_both.setChecked(has_chinese and zh_both)
        self.zh_both.setVisible(has_chinese)
        f.addRow(self.zh_both)
        self.reference = QCheckBox("参考原配简繁字幕，附对照报告")
        self.reference.setChecked(reference_enabled)
        self.reference_path = QLineEdit(reference_path)
        self.reference_path.setPlaceholderText("留空：自动在原字幕目录查找原配简繁字幕")
        self.reference_row = _path_row(self.reference_path, "选择…", self._pick_reference)
        self.reference_row.setEnabled(reference_enabled)
        self.reference.toggled.connect(self.reference_row.setEnabled)
        f.addRow(self.reference)
        f.addRow("原配字幕文件（可选）", self.reference_row)
        note = QLabel("自动导出，不逐条确认。没有参考或匹配不唯一时回退自动简繁转换，并在每份 ASS 旁附 HTML 对照报告。单份导出沿用工程当前转换方向。")
        note.setWordWrap(True)
        f.addRow(note)
        f.addRow(self.actor)
        f.addRow("名字栏（Name）", self.name_format)
        f.addRow(self.comments)
        f.addRow("Blur（0 = 不加）", self.blur)
        self.annotation_order = QComboBox()
        self.annotation_order.addItem("统一放在文件末尾", "end")
        self.annotation_order.addItem("紧跟对应正文行", "after_parent")
        self.annotation_order.setToolTip("对白和歌曲的 Ruby / Furigana / Pinyin 注音行；歌曲有多条特效事件时，跟在对应正文的整块事件之后。")
        f.addRow("注音行顺序", self.annotation_order)
        cleanup_note = QLabel("发布 ASS 自动清理每行首尾半角空格并跳过空白字幕，不改工程原文。")
        cleanup_note.setWordWrap(True)
        f.addRow(cleanup_note)
        f.addRow(QLabel("<span style='color:#6e7781'>原字幕不会被覆盖；同名文件存在时会另存为带编号的新文件。</span>"))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        dialog_buttons(self, bb)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _pick(self) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "保存为", self.path.text(), "ASS (*.ass)")
        if f:
            self.path.setText(f)

    def _pick_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择原配字幕", self.reference_path.text(), "字幕 (*.ass *.ssa *.srt);;所有文件 (*)")
        if path:
            self.reference_path.setText(path)
