"""Project and episode persistence (FORMAT-v1 §2, §4, §5).

A project directory is chosen by the user and is separate from the program.
Everything is plain JSON written atomically; source subtitle bytes are copied
into the episode folder (read-only) and never written anywhere else.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..ass.document import AssDocument
from ..languages import get_language
from ..model.document import Document
from ..model.ids import new_id
from ..model.serialize import load_document, read_json, save_document, write_json_atomic
from ..model.source import Source
from ..model.validate import Report, validate
from .importer import import_source, verify_copy

if TYPE_CHECKING:
    from .export import ExportReport

PROJECT_FILE = "subkit-project.json"
EPISODE_FILE = "episode.json"
DOCUMENT_FILE = "document.json"
FORMAT_PROJECT = "subkit-project"
FORMAT_EPISODE = "subkit-episode"

DEFAULT_QUICK_COLORS = ("#FFE36E", "#8CD3FF", "#FF9CC2", "#B8F28C")

DEFAULT_SETTINGS: dict[str, Any] = {
    "playres": [1920, 1080],
    "merge": {"filler_mode": "keep", "semantic": False, "jp_sep": "  ", "joint": "first"},
    "ruby": {"gap": 2.0, "min_scale": 60, "overhang": 0.5, "max_width": {
        "dialogue": {"enabled": True, "percent": 150}, "song": {"enabled": True, "percent": 150}}},
    "release": {"include_actor": True, "include_source_comments": False, "blur": 0.0},
    "display": {"hide_final_period": False, "punctuation": {lane: {
        "comma": False, "period": False, "spaces": False, "ellipsis": "off",
        "exclaim": "off", "question": "off", "quotes": False, "font": None}
        for lane in ("translation", "original")}},   # user-chosen display rules (preview + export only)
    "pinyin": {"mode": "off"},
    "cn_styles": {},   # translation-file style roles: {"dialogue": [...], "top": [...], "song_jp": [...], "song_cn": [...]}
}


def _merge_dict(base: dict, extra: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class Project:
    def __init__(self, root: Path, data: dict[str, Any]) -> None:
        self.root = Path(root)
        self.data = data

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def create(cls, root: Path, name: str, *, original: str = "ja", translation: str = "zh-Hant",
               template_from: Optional[Path] = None, template_text: Optional[str] = None,
               template_scheme: Optional[dict] = None, dict_from: Optional[Path] = None) -> "Project":
        """``template_text`` (a [V4+ Styles] section, e.g. from subkit.templates) wins over ``template_from`` (a file
        copied as is); without either the default styles are used.  ``template_scheme`` records which built-in
        colours / fonts the template came from."""
        root = Path(root)
        if (root / PROJECT_FILE).exists():
            raise FileExistsError(f"{root} 已经是一个双语字幕工坊工程")
        root.mkdir(parents=True, exist_ok=True)
        data = {
            "format": FORMAT_PROJECT, "format_version": 1, "id": new_id("prj_"), "name": name,
            "writer": {"app": "SubKit-Opus", "version": "0.2.0"}, "created_at": now(),
            "media_roots": [], "languages": {"original": original, "translation": translation},
            "seasons": [], "episodes": [], "settings": {},
            "template": {"id": "tpl_main", "version": 1, "path": "templates/tpl_main/v1.ass"},
            "dictionaries": {"readings": "dict/readings.json", "pinyin": "dict/pinyin.json",
                             "characters": "dict/characters.tsv", "songs": "dict/songs.json"},
        }
        if template_scheme:
            data["template"]["scheme"] = dict(template_scheme)
        p = cls(root, data)
        tdir = root / "templates" / "tpl_main"
        tdir.mkdir(parents=True, exist_ok=True)
        if template_text:
            (tdir / "v1.ass").write_text(template_text, encoding="utf-8")
        elif template_from is not None and Path(template_from).exists():
            shutil.copyfile(template_from, tdir / "v1.ass")
        else:
            from ..styles_default import default_styles_ass
            (tdir / "v1.ass").write_text(default_styles_ass(), encoding="utf-8")
        (root / "dict").mkdir(exist_ok=True)
        if dict_from is not None:
            for name_ in ("readings.json", "characters.tsv", "characters.json"):
                if (Path(dict_from) / name_).exists():
                    shutil.copyfile(Path(dict_from) / name_, root / "dict" / name_)
        p.save()
        p.character_book()
        return p

    @classmethod
    def open(cls, root: Path) -> "Project":
        root = Path(root)
        data = read_json(root / PROJECT_FILE)
        if data.get("format") != FORMAT_PROJECT:
            raise ValueError(f"{root} 不是双语字幕工坊工程")
        if int(data.get("format_version", 0)) > 1:
            raise ValueError("工程格式版本比本程序新，请升级程序")
        project = cls(root, data)
        project.character_book()
        return project

    def save(self) -> None:
        self.data["modified_at"] = now()
        write_json_atomic(self.root / PROJECT_FILE, self.data)

    # ------------------------------------------------------------ properties
    @property
    def name(self) -> str:
        return self.data.get("name", "")

    @property
    def languages(self) -> dict[str, str]:
        return self.data.get("languages", {"original": "ja", "translation": "zh-Hant"})

    def settings(self, season: Optional[str] = None, episode: Optional["Episode"] = None) -> dict[str, Any]:
        s = _merge_dict(DEFAULT_SETTINGS, self.data.get("settings", {}))
        if season:
            for se in self.data.get("seasons", []):
                if se.get("id") == season:
                    s = _merge_dict(s, se.get("settings", {}))
        if episode is not None:
            s = _merge_dict(s, episode.data.get("settings", {}))
        # These are whole-project rules, not overridable episode defaults.
        from .replacements import load_rules
        s.setdefault("display", {})["replacements"] = load_rules(self)
        return s

    @property
    def quick_colors(self) -> list[str]:
        return list(self.data.get("settings", {}).get("quick_colors", DEFAULT_QUICK_COLORS))

    def set_quick_colors(self, colours: list[str]) -> None:
        import re

        if len(colours) > 12 or any(not isinstance(c, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", c)
                                    for c in colours):
            raise ValueError("quick_colors requires at most 12 #RRGGBB colours")
        self.data.setdefault("settings", {})["quick_colors"] = [c.upper() for c in colours]
        self.save()

    def template_path(self) -> Path:
        return self.root / self.data["template"]["path"]

    @property
    def pinyin_mode(self) -> str:
        return self.settings()["pinyin"]["mode"]

    def set_pinyin_mode(self, mode: str) -> None:
        from .pinyin_notes import MODES

        if mode not in MODES:
            raise ValueError(mode)
        self.data.setdefault("settings", {}).setdefault("pinyin", {})["mode"] = mode
        self.save()

    @property
    def pinyin_path(self) -> Path:
        return self.root / self.data.get("dictionaries", {}).get("pinyin", "dict/pinyin.json")

    def pinyin_words(self, lang: Optional[str] = None) -> dict[str, str]:
        words = self.character_book().pinyin_words(lang)
        if not self.pinyin_path.exists():
            return words
        data = read_json(self.pinyin_path)
        if data.get("format") != "subkit-pinyin" or data.get("format_version") != 1:
            raise ValueError("subkit-pinyin / format_version 1 required")
        from .pinyin_notes import normalize_rule

        words.update(normalize_rule(w, r) for w, r in data.get("words", {}).items())
        return words

    def set_pinyin_word(self, surface: str, reading: str) -> None:
        from .pinyin_notes import normalize_rule

        word, syllables = normalize_rule(surface, reading)
        data = read_json(self.pinyin_path) if self.pinyin_path.exists() else {
            "format": "subkit-pinyin", "format_version": 1, "words": {}}
        self.pinyin_words()                # validate an existing dictionary before writing it
        data.setdefault("words", {})[word] = syllables
        write_json_atomic(self.pinyin_path, data)

    def template(self) -> AssDocument:
        return AssDocument.load(self.template_path())

    def replace_template(self, styles_text: str, scheme: Optional[dict] = None) -> Path:
        """Switch to new styles as a new template version; the old file stays where it is.  Styles of the old
        template that the new text does not define (the user's own, e.g. a handwriting style) are carried over."""
        new = AssDocument.parse(styles_text)
        if not new.styles:
            raise ValueError("新模板里没有样式")
        names = {s.name for s in new.styles}
        try:
            kept = [s for s in self.template().styles if s.name not in names]
        except OSError:
            kept = []
        lines = ["[V4+ Styles]", "Format: " + ", ".join(new.style_format)]
        lines += [s.to_line(new.style_format) for s in new.styles + kept]
        t = dict(self.data.get("template") or {})
        from dataclasses import replace
        from ..templates import scheme_dict, scheme_from_dict, scheme_from_styles

        old_fonts, _ = scheme_from_dict(t.get("scheme", {}))
        if old_fonts.ruby_offset or old_fonts.pinyin_offset:
            fonts, pal = scheme_from_dict(scheme) if scheme is not None else scheme_from_styles(new)
            supplied = (scheme or {}).get("fonts", {})
            offsets = {key: getattr(old_fonts, key) for key in ("ruby_offset", "pinyin_offset")
                       if not isinstance(supplied, dict) or key not in supplied}
            scheme = scheme_dict(replace(fonts, **offsets), pal)
        tid = t.get("id", "tpl_main")
        version = int(t.get("version", 1)) + 1
        while (self.root / "templates" / tid / f"v{version}.ass").exists():
            version += 1
        rel = f"templates/{tid}/v{version}.ass"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        t.update({"id": tid, "version": version, "path": rel})
        if scheme is not None:
            t["scheme"] = dict(scheme)
        else:
            t.pop("scheme", None)
        self.data["template"] = t
        self.save()
        return path

    def dict_path(self, key: str) -> Path:
        return self.root / self.data.get("dictionaries", {}).get(key, f"dict/{key}")

    def character_book(self):
        from .people import CharacterBook

        path = self.root / self.data.get("dictionaries", {}).get("characters_json", "dict/characters.json")
        tsv = self.dict_path("characters")
        if not tsv.exists():
            tsv = self.root / "dict" / "characters.tsv"
        return CharacterBook.load(path, tsv=tsv if tsv.suffix.lower() == ".tsv" else None,
                                  translation=self.languages.get("translation", "zh-Hant"),
                                  original=self.languages.get("original", ""))

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    # ------------------------------------------------------------ episodes
    def episode_ids(self) -> list[str]:
        return [e["id"] for e in self.data.get("episodes", [])]

    def episode(self, eid: str) -> "Episode":
        entry = next((e for e in self.data.get("episodes", []) if e["id"] == eid), None)
        if entry is None:
            raise KeyError(eid)
        return Episode.open(self, self.root / entry["path"])

    def add_episode(self, eid: str, *, season: str = "S01", number: int = 1, title: str = "") -> "Episode":
        if eid in self.episode_ids():
            raise FileExistsError(f"集 {eid} 已存在")
        rel = f"episodes/{eid}"
        self.data.setdefault("episodes", []).append({"id": eid, "season": season, "number": number, "title": title, "path": rel})
        if season not in [s.get("id") for s in self.data.setdefault("seasons", [])]:
            self.data["seasons"].append({"id": season, "title": season, "settings": {}})
        ep = Episode.create(self, self.root / rel, eid, season, number)
        self.save()
        return ep

    def remove_episode(self, eid: str) -> Path:
        """Archive one episode folder; restore it if the project cannot be saved."""
        entries = self.data.get("episodes", [])
        entry = next((e for e in entries if e["id"] == eid), None)
        if entry is None:
            raise KeyError(eid)
        root = self.root.resolve()
        episode_path = root / entry["path"]
        source = episode_path.resolve()
        removed = (root / "已移除").resolve()
        if (not source.is_relative_to(root) or source == root
                or source.parent != root / "episodes" or removed != root / "已移除"
                or episode_path.is_symlink() or episode_path.is_junction()):
            raise ValueError("集目录或已移除目录不在工程内的预期位置")
        for other in entries:
            if other is entry:
                continue
            path = (root / other["path"]).resolve()
            if source.is_relative_to(path) or path.is_relative_to(source):
                raise ValueError("集目录与其他集重叠，不能移除")
        if not source.is_dir():
            raise FileNotFoundError(source)
        metadata = read_json(source / EPISODE_FILE)
        video = (metadata.get("media") or {}).get("video")
        originals = [s["path"] for s in metadata.get("sources", [])]
        if video:
            originals.append(video["path"])
        if any(Path(p).resolve().is_relative_to(source) for p in originals):
            raise ValueError("原视频或原字幕位于集目录内，请先移到工程工作文件夹外")
        removed.mkdir(exist_ok=True)
        name = f"{source.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        destination = removed / name
        suffix = 1
        while destination.exists():
            destination = removed / f"{name}-{suffix}"
            suffix += 1
        # Rename within the project; never copy/delete across volumes.
        source.rename(destination)
        previous = self.data
        self.data = {**previous, "episodes": [e for e in entries if e is not entry]}
        try:
            self.save()
        except Exception:
            self.data = previous
            destination.rename(source)
            raise
        return destination

    def reading_engine(self, *, episode=None, season=None):
        from ..reading.dictionary import ReadingDict
        from ..reading.engine import ReadingEngine

        rd = ReadingDict.load(self.root / "dict" / "_no_global.json", self.dict_path("readings"))
        rd.merge_character_book(self.character_book())
        return ReadingEngine(rd, episode=episode, season=season)


@dataclass
class Episode:
    project: Project
    dir: Path
    data: dict[str, Any]
    sources: dict[str, Source] = field(default_factory=dict)
    document: Optional[Document] = None
    times_changed: bool = False
    recovered: bool = False

    @classmethod
    def create(cls, project: Project, path: Path, eid: str, season: str, number: int) -> "Episode":
        path.mkdir(parents=True, exist_ok=True)
        data = {"format": FORMAT_EPISODE, "format_version": 1, "id": eid, "season": season, "number": number,
                "media": {"video": None, "audio_track": None}, "sources": [], "settings": {},
                "template": dict(project.data.get("template", {})), "merge": None, "created_at": now()}
        ep = cls(project, path, data)
        ep.save_meta()
        return ep

    @classmethod
    def open(cls, project: Project, path: Path) -> "Episode":
        data = read_json(path / EPISODE_FILE)
        ep = cls(project, path, data)
        for entry in data.get("sources", []):
            ev = read_json(path / "sources" / f"{entry['id']}.events.json")
            ep.sources[entry["id"]] = Source.from_json(entry, ev)
        if (path / DOCUMENT_FILE).exists():
            ep.document = load_document(path / DOCUMENT_FILE)
            for m in ep.document.members.values():
                if m.start < 0 or m.end < 0:
                    m.start, m.end = max(0, m.start), max(0, m.end)
                    ep.times_changed = True
            for track in ep.document.karaoke.values():
                for beat in track.get("beats", ()):
                    for edge in ("t0", "t1"):
                        if beat.get(edge) is not None and beat[edge] < 0:
                            beat[edge] = 0
                            ep.times_changed = True
            if ep.times_changed:
                ep.document.revision += 1
            ep._sync_coordinates()
        return ep

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def video(self) -> Optional[Path]:
        v = (self.data.get("media") or {}).get("video")
        return Path(v["path"]) if v else None

    def set_video(self, path: Optional[Path], audio_track: Optional[int] = None) -> None:
        if path is None:
            self.data["media"] = {"video": None, "audio_track": audio_track}
        else:
            st = Path(path).stat()
            self.data["media"] = {"video": {"path": str(path).replace("\\", "/"), "size": st.st_size,
                                            "mtime": int(st.st_mtime)}, "audio_track": audio_track}
        self.save_meta()
        self._sync_coordinates()

    def settings(self) -> dict[str, Any]:
        return self.project.settings(self.data.get("season"), self)

    def effective_playres(self) -> tuple[int, int]:
        """Episode script coordinates follow the video's display aspect (ffprobe geometry cached in the metadata).

        A video whose geometry cannot be read now (moved file, probe failure) keeps the coordinates the document
        was written in, so it never converts stored positions; an episode without a video uses the settings."""
        from fractions import Fraction

        px, py = (int(v) for v in self.settings()["playres"])
        video = (self.data.get("media") or {}).get("video")
        if not video:
            return px, py
        geometry = self._video_geometry(video)
        recorded = self.document.extra.get("coordinate_playres") if self.document is not None else None
        fallback = (int(recorded[0]), int(recorded[1])) if recorded else (px, py)
        if geometry is None:
            return fallback
        try:
            width, height = int(geometry["width"]), int(geometry["height"])
        except (KeyError, TypeError, ValueError):
            return fallback
        if width <= 0 or height <= 0 or py <= 0:
            return fallback
        try:
            sar = Fraction(str(geometry.get("sample_aspect_ratio") or "1:1").replace(":", "/"))
            if sar <= 0:
                sar = Fraction(1)
        except (ValueError, ZeroDivisionError):
            sar = Fraction(1)
        # Round to an integer first, then round an odd width up to the next even pixel.
        effective_x = max(2, round(py * Fraction(width, height) * sar))
        return effective_x + effective_x % 2, py

    def _video_geometry(self, video: dict) -> Optional[dict]:
        """Width, height and sample aspect of the video (cached); None when it cannot be read now."""
        import subprocess

        from ..proc import run

        geometry = video.get("geometry")
        if geometry is not None:
            return geometry
        path = self.video
        if path is None or not path.exists():
            return None
        try:
            result = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,sample_aspect_ratio", "-of", "json", str(path)],
                         capture_output=True, check=True, timeout=15)
            streams = json.loads(result.stdout).get("streams", [])
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        if not streams or not streams[0].get("width") or not streams[0].get("height"):
            return None
        geometry = {key: streams[0].get(key) for key in ("width", "height", "sample_aspect_ratio")}
        video["geometry"] = geometry
        self.save_meta()
        return geometry

    def _sync_coordinates(self) -> None:
        """Back up before converting stored positions; source runs stay in their source PlayRes."""
        from ..ass.resample import scale_positions
        from .render_origin import run_source

        doc = self.document
        if doc is None:
            return
        target = self.effective_playres()
        old = tuple(doc.extra.get("coordinate_playres") or self.settings()["playres"])
        if old == target:
            doc.extra["coordinate_playres"] = list(target)
            return
        self.backup("before-coordinates")
        sx = target[0] / old[0]
        for member in doc.members.values():
            for key, axes in (("pos", (0,)), ("move", (0, 2)), ("org", (0,)), ("clip", (0, 2)), ("iclip", (0, 2))):
                value = member.overrides.get(key)
                if isinstance(value, (list, tuple)):
                    value = list(value)
                    for axis in axes:
                        if axis < len(value):
                            value[axis] = round(value[axis] * sx, 2)
                    member.overrides[key] = value
            props = [member.overrides] + [span.get("props", {}) for span in member.format]
            for prop in props:
                if prop.get("raw"):
                    prop["raw"] = scale_positions("{" + prop["raw"] + "}", sx)[1:-1]
            for run in member.runs:
                if run_source(member, run) is None and run.kind == "tag":
                    run.text = scale_positions(run.text, sx)
        for comment in doc.comments.values():
            for run in comment.runs:
                if run_source(comment, run) is None and run.kind == "tag":
                    run.text = scale_positions(run.text, sx)
        doc.extra["coordinate_playres"] = list(target)
        save_document(doc, self.dir / DOCUMENT_FILE)

    # ------------------------------------------------------------ sources
    def add_source(self, path: Path, *, lang: str, lane: str, src_id: Optional[str] = None,
                   cn_cfg: Optional[dict] = None, ass_styles: Optional[list[str]] = None,
                   song_intervals: Optional[list[tuple[int, int]]] = None) -> Source:
        if src_id is None:
            prefix = {"original": "o", "translation": "t"}.get(lane, "r")
            k = 1
            while f"{prefix}{k}" in self.sources:
                k += 1
            src_id = f"{prefix}{k}"
        if src_id in self.sources:
            raise FileExistsError(f"来源 {src_id} 已存在")
        cfg = cn_cfg if cn_cfg is not None else self.settings().get("cn_styles", {})
        src = import_source(path, src_id=src_id, lang=lang, lane=lane, episode_dir=self.dir, cn_cfg=cfg,
                            ass_styles=ass_styles, song_intervals=song_intervals)
        self.sources[src_id] = src
        write_json_atomic(self.dir / "sources" / f"{src_id}.events.json", src.events_json())
        self.data["sources"] = [s.entry_json() for s in self.sources.values()]
        self.save_meta()
        return src

    def reading_engine(self, document=None):
        """Project dictionaries plus this episode's reading rules (episode rules win inside the episode)."""
        eng = self.project.reading_engine(episode=self.id, season=self.data.get("season"))
        doc = document if document is not None else self.document
        rules = [r for r in (doc.rules if doc is not None else []) if r.get("kind") == "reading"]
        if rules:
            eng.dict.merge_json({"words": {r["surface"]: r["reading"] for r in rules}}, "episode")
        return eng

    def reanalyze_sources(self, *, editor=None, cancel_check=lambda: None) -> dict[str, tuple[int, int]]:
        """Run the current original-language analyzer again (e.g. ja-cc 2: SRT formatting tags, dialogue dashes)
        on the stored events.  The source files and the project's copies are not touched; the previous analysis
        is kept as ``sources/<id>.events.v<old version>[.n].json``.  Returns {source id: (spans before, after)}."""
        import shutil

        from .sdh_en import analyzer_for

        out: dict[str, tuple[int, int]] = {}
        for sid, src in self.sources.items():
            cancel_check()
            if src.lane != "original":
                continue
            before = len(src.spans)
            p = self.dir / "sources" / f"{sid}.events.json"
            if p.exists():
                old_ver = (src.analyzer or {}).get("version", 0)
                keep = p.with_name(f"{sid}.events.v{old_ver}.json")
                k = 1
                while keep.exists():
                    keep = p.with_name(f"{sid}.events.v{old_ver}.{k}.json")
                    k += 1
                shutil.copy2(p, keep)
            src.spans = [span for span in src.spans
                         if src.roles.get(span.e) == "excluded"
                         and (span.data or {}).get("reason") == "style_filter"]
            src.doubts = []
            analyzer, analyze = analyzer_for(src.lang)
            src.analyzer = dict(analyzer)
            for ev in src.events:
                cancel_check()
                if src.roles.get(ev.i) not in ("dialogue", "top"):
                    continue
                an = analyze(ev.i, ev.text, src.format)
                src.spans.extend(an.spans)
                src.doubts.extend({"e": ev.i, "text": d} for d in an.doubts)
            cancel_check()
            write_json_atomic(p, src.events_json())
            out[sid] = (before, len(src.spans))
        if editor is not None:
            if editor.doc is not self.document:
                raise ValueError("The editor must own this episode's document")
            editor.relink_speakers(self.project.character_book())
        return out

    def source_of_lane(self, lane: str) -> Optional[Source]:
        return next((s for s in self.sources.values() if s.lane == lane), None)

    def verify_sources(self) -> list[str]:
        """Problems with the read-only copies or the originals (moved / changed)."""
        out = []
        for s in self.sources.values():
            if not verify_copy(s, self.dir):
                out.append(f"工程内的原料副本 {s.copy} 与记录的哈希不一致")
            p = Path(s.path)
            if not p.exists():
                out.append(f"原始文件已移动或删除：{s.path}（工程内副本仍然完整）")
            else:
                import hashlib
                if hashlib.sha256(p.read_bytes()).hexdigest() != s.sha256:
                    out.append(f"原始文件在导入后被改过：{s.path}（工程使用导入时的副本）")
        return out

    # ------------------------------------------------------------ document
    def merge(self, *, settings: Optional[dict] = None, log: Callable[[str], None] = lambda s: None,
              sim_factory=None, cancel_check=lambda: None, persist: bool = True,
              progress: Callable[[str], None] = lambda s: None) -> Document:
        from .merge import MergeSettings, merge_episode
        from .readings import annotate_document

        if self.document is not None and any(g.state == "confirmed" for g in self.document.groups):
            raise RuntimeError("这一集已有确认过的配对；整集重新合并需要先备份并明确确认（第二阶段提供局部重新对齐）")
        cn = self.source_of_lane("translation")
        ja = self.source_of_lane("original")
        if cn is None or ja is None:
            raise ValueError("需要一份翻译字幕和一份原声字幕")
        s = settings or self.settings()["merge"]
        ms = MergeSettings(filler_mode=s.get("filler_mode", "keep"), semantic=bool(s.get("semantic", False)),
                           jp_sep=s.get("jp_sep", "  "), weights=s.get("weights", {}),
                           pairing=s.get("pairing", "align"), joint=s.get("joint", "first"))
        res = merge_episode(self.id, cn, ja, ms, sim_factory=sim_factory, log=log,
                            character_book=self.project.character_book(), previous=self.document, cancel_check=cancel_check)
        progress("读音")
        annotate_document(res.document, self.sources, self.reading_engine(res.document), cancel_check=cancel_check)
        if self.project.pinyin_mode != "off":
            from .pinyin_notes import annotate_pinyin, episode_rules

            for m in res.document.members.values():
                cancel_check()
                if get_language(m.lang).annotation_system == "pinyin":
                    annotate_pinyin(res.document, m, episode_rules(res.document, self.project.pinyin_words(m.lang)))
        cancel_check()
        if persist and self.document is not None:
            self.backup("before-merge")
        self.document = res.document
        self.document.extra["coordinate_playres"] = list(self.effective_playres())
        self.data["merge"] = {"settings": s, "aligner": "subkit-align-v2", "ran_at": now(), "log": res.log,
                              "summary": res.summary}
        cancel_check()
        if persist:
            progress("正在保存…")
            cancel_check()
            self.save()
        return self.document

    def validate(self) -> Report:
        return validate(self.document, self.sources) if self.document else Report()

    def autosave_recovery(self) -> Optional[tuple[Document, float]]:
        """A newer autosave with different JSON content, without changing the saved file."""
        path = self.dir / "journal" / "autosave.json"
        saved = self.dir / DOCUMENT_FILE
        if not path.is_file() or not saved.is_file():
            return None
        stamp = path.stat().st_mtime
        if stamp <= saved.stat().st_mtime:
            return None
        data = read_json(path)
        if data == read_json(saved):
            return None
        from ..model.serialize import document_from_json

        return document_from_json(data), stamp

    def discard_autosave(self, stamp: float) -> Path:
        """Archive a declined recovery; keep the original bytes and avoid name collisions."""
        path = self.dir / "journal" / "autosave.json"
        name = "autosave-discarded-" + time.strftime("%Y%m%d-%H%M%S", time.localtime(stamp))
        destination = path.with_name(name + ".json")
        suffix = 1
        while destination.exists():
            destination = path.with_name(f"{name}-{suffix}.json")
            suffix += 1
        path.rename(destination)
        return destination

    def save(self) -> None:
        if self.document is not None:
            if "coordinate_playres" not in self.document.extra:
                self.document.extra["coordinate_playres"] = list(self.effective_playres())
            save_document(self.document, self.dir / DOCUMENT_FILE)
            self.times_changed = False
            self.recovered = False
        self.save_meta()

    def save_meta(self) -> None:
        self.data["sources"] = [s.entry_json() for s in self.sources.values()]
        write_json_atomic(self.dir / EPISODE_FILE, self.data)

    def backup(self, label: str) -> Path:
        bdir = self.dir / "journal" / "backups"
        bdir.mkdir(parents=True, exist_ok=True)
        dst = bdir / f"document-{time.strftime('%Y%m%d-%H%M%S')}-{label}.json"
        k = 1
        while dst.exists():
            dst = bdir / f"document-{time.strftime('%Y%m%d-%H%M%S')}-{label}-{k}.json"
            k += 1
        if (self.dir / DOCUMENT_FILE).exists():
            shutil.copy2(self.dir / DOCUMENT_FILE, dst)
        return dst

    # ------------------------------------------------------------ export
    def build_export(self, *, mode: str = "release", zh_direction: Optional[str] = None, **kw: Any) -> tuple[AssDocument, ExportReport]:
        from .display import DisplayRules
        from .export import ExportOptions, build_ass
        from ..templates import scheme_from_dict

        st = self.settings()
        rel = st["release"]
        display = DisplayRules.from_settings(st)
        if zh_direction is not None:
            if zh_direction not in ("simplified", "traditional"):
                raise ValueError("中文导出方向必须是 simplified 或 traditional")
            from dataclasses import replace
            display = replace(display, zh_script=zh_direction)
        fonts, _ = scheme_from_dict(self.project.data.get("template", {}).get("scheme", {}))
        opts = ExportOptions(mode=mode, include_actor=kw.get("include_actor", rel.get("include_actor", True)),
                             include_source_comments=kw.get("include_source_comments", rel.get("include_source_comments", False)),
                             blur=float(kw.get("blur", rel.get("blur", 0.0))),
                             ruby_gap=float(st["ruby"]["gap"]), ruby_min_scale=float(st["ruby"]["min_scale"]),
                             ruby_overhang=float(st["ruby"]["overhang"]), playres=self.effective_playres(),
                             ruby_max_width=st["ruby"]["max_width"],
                             display=display, pinyin_mode=st["pinyin"]["mode"],
                             name_format=kw.get("name_format", rel.get("name_format", "short")),
                             ruby_offset=fonts.ruby_offset, pinyin_offset=fonts.pinyin_offset,
                             template_playres=tuple(st["playres"]),
                             reference_enabled=bool(kw.get("reference_enabled", False)),
                             reference_path=kw.get("reference_path"),
                             annotation_order=kw.get("annotation_order", "end"))
        return build_ass(self.document, self.sources, self.project.template(), self.dir, opts,
                         measure_cache=self.project.cache_dir / "render.sqlite3", video_height=kw.get("video_height"),
                         character_book=self.project.character_book(), runner=kw.get("runner"))

    def write_export(self, ass: AssDocument, out_path: Path) -> Path:
        out_path = Path(out_path)
        for s in self.sources.values():
            if Path(s.path).resolve() == out_path.resolve():
                raise PermissionError("不能覆盖原料字幕")
        ass.save(out_path)
        return out_path

    def export(self, out_path: Path, *, mode: str = "release", **kw: Any) -> tuple[Path, Any]:
        ass, rep = self.build_export(mode=mode, **kw)
        if rep.reference is not None:
            from .reference_report import preflight_targets, render_reference_report, write_reference_report
            preflight_targets([Path(out_path), Path(out_path).with_suffix(".reference.html")],
                              rep.reference["protected_input_paths"])
            # Cancellation belongs to generation, before any output is written.
            from ..proc import CancelledError
            try:
                rep.reference["html"] = render_reference_report(rep.reference, runner=kw.get("runner"))
            except CancelledError:
                raise
            except Exception as exc:
                rep.reference["html_error"] = str(exc)
        written = self.write_export(ass, out_path)
        if rep.reference is not None:
            try:
                write_reference_report(rep.reference, written)
            except Exception as exc:
                rep.warnings.append(f"字幕已导出，对照报告写入失败：{exc}")
        return written, rep
