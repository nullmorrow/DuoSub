"""subkit command line.

    subkit init                     create subproj.toml / styles.ass / dict in the current folder
    subkit jp-extract ZIP REGEX     extract Japanese subtitles from a zip into sources/jp
    subkit prepare S01E001          build work/S01E001/work.ass (merge + readings + ruby)
    subkit refresh S01E001          apply your edits in work.ass and rebuild generated lines
    subkit edit S01E001             prepare if needed, then loop: refresh → Aegisub → refresh …
    subkit release S01E001          write the clean release ASS next to the video
    subkit status                   episodes and their state
    subkit dict add "表記（よみ）"   add a word to the project dictionary
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from . import __version__
from .config import TEMPLATE as CONFIG_TEMPLATE
from .project import Project, episode_id


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_init(args) -> int:
    root = Path.cwd()
    cfg = root / "subproj.toml"
    if not cfg.exists():
        cfg.write_text(CONFIG_TEMPLATE.format(name=args.name or root.name), encoding="utf-8")
        _log(f"写出 {cfg}")
    d = root / "dict" / "readings.json"
    if not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text('{\n  "version": 2,\n  "words": {},\n  "speakers": {},\n  "prefer": {},\n  "candidates": {}\n}\n', encoding="utf-8")
        _log(f"写出 {d}")
    st = root / "styles.ass"
    if not st.exists():
        from .styles_default import DEFAULT_STYLES

        st.write_text(DEFAULT_STYLES, encoding="utf-8-sig")
        _log(f"写出 {st}")
    return 0


def _zip_names(z: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    out = []
    for info in z.infolist():
        name = info.filename
        if not (info.flag_bits & 0x800):
            for enc in ("cp932", "gbk"):
                try:
                    name = info.filename.encode("cp437").decode(enc)
                    break
                except (UnicodeDecodeError, UnicodeEncodeError):
                    continue
        out.append((info, name))
    return out


def cmd_jp_extract(args) -> int:
    prj = Project.open()
    dest = prj.root / "sources" / "jp"
    dest.mkdir(parents=True, exist_ok=True)
    rx = re.compile(args.regex)
    n = 0
    with zipfile.ZipFile(prj.root / args.zip) as z:
        for info, name in _zip_names(z):
            m = rx.search(name)
            if not m:
                continue
            ep = int(m.group(1))
            ext = Path(name).suffix.lower()
            out = dest / f"S{args.season:02d}E{ep:02d}.ja{ext}"
            if out.exists() and not args.force:
                continue
            out.write_bytes(z.read(info))
            n += 1
            _log(f"  {name} → {out.name}")
    _log(f"提取 {n} 个文件到 {dest}")
    return 0


def cmd_prepare(args) -> int:
    from .workfile import prepare, refresh

    prj = Project.open()
    ep = prj.episode(args.episode)
    _log(f"[prepare] {ep.id}")
    prepare(ep, log=_log, embedding=args.embedding, force=args.force)
    _log(f"[refresh] {ep.id}")
    rep = refresh(ep, log=_log)
    _print_report(rep)
    _log(f"工作文件：{ep.work_ass}")
    return 0


def _print_report(rep) -> None:
    _log(f"  行={rep.rows} 读音记录={rep.records_written} 待确认读音={rep.review} 手改保留={rep.manual_kept} 用时 {rep.elapsed:.1f}s")
    for c in rep.conflicts[:20]:
        _log(f"  ⚠ {c}")
    for w in rep.warnings[:20]:
        _log(f"  ⚠ {w}")
    if rep.fonts:
        _log("  字体：" + "；".join(f.split("fontselect:")[-1].strip() for f in rep.fonts))


def cmd_refresh(args) -> int:
    from .workfile import refresh

    prj = Project.open()
    ep = prj.episode(args.episode)
    if not ep.work_ass.exists():
        _log(f"{ep.work_ass} 不存在，请先 subkit prepare {ep.id}")
        return 2
    rep = refresh(ep, log=_log)
    _print_report(rep)
    return 0


def _wait_aegisub(proc: subprocess.Popen) -> None:
    t0 = time.time()
    proc.wait()
    if time.time() - t0 < 5:
        # handed over to an already-running Aegisub: poll the process list
        while True:
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq aegisub.exe"], capture_output=True, text=True).stdout
            if "aegisub.exe" not in out.lower():
                return
            time.sleep(1.5)


def cmd_edit(args) -> int:
    from .workfile import prepare, refresh

    prj = Project.open()
    ep = prj.episode(args.episode)
    if not ep.work_ass.exists():
        _log(f"[prepare] {ep.id}")
        prepare(ep, log=_log)
    aegisub = prj.config["tools"]["aegisub"]
    video = ep.video()
    while True:
        _log(f"\n[refresh] {ep.id}  {time.strftime('%H:%M:%S')}")
        try:
            _print_report(refresh(ep, log=_log))
        except Exception as exc:
            _log(f"刷新失败：{exc}（work.ass 未改动；修正后保存并关闭 Aegisub 会再试）")
        cmd = [aegisub, str(ep.work_ass)] + ([str(video)] if video else [])
        _log("打开 Aegisub…（保存并关闭 Aegisub 后自动刷新；Ctrl+C 结束）")
        try:
            _wait_aegisub(subprocess.Popen(cmd))
        except KeyboardInterrupt:
            return 0
        except FileNotFoundError:
            _log(f"找不到 Aegisub：{aegisub}（subproj.toml [tools] aegisub）")
            return 2


def cmd_release(args) -> int:
    from .release import release

    prj = Project.open()
    ep = prj.episode(args.episode)
    out = release(ep, log=_log, blur=args.blur, output=Path(args.output) if args.output else None)
    _log(f"发布文件：{out}")
    return 0


def cmd_voice(args) -> int:
    from .voice_check import voice_check

    prj = Project.open()
    voice_check(prj.episode(args.episode), log=_log, model=args.model, min_margin=args.margin, watashi=not args.no_watashi)
    return 0


def cmd_check(args) -> int:
    from .check import check

    prj = Project.open()
    check(prj.episode(args.episode), log=_log)
    return 0


def cmd_status(args) -> int:
    prj = Project.open()
    work = prj.config.work_dir
    for d in sorted(work.glob("S*E*")) if work.exists() else []:
        ass = d / "work.ass"
        mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(ass.stat().st_mtime)) if ass.exists() else "-"
        _log(f"  {d.name}  work.ass 更新于 {mt}")
    return 0


def cmd_dict(args) -> int:
    from .reading.dictionary import ReadingDict

    prj = Project.open()
    path = prj.config.path(prj.config["project"]["dict"])
    m = re.match(r"^\s*(\S+?)[（(](.+?)[）)]\s*$", args.entry)
    if not m:
        _log("格式：subkit dict add \"表記（よみ）\"，分组用空格：\"幸平 創真（ゆきひら そうま）\"")
        return 2
    surface, reading = m.group(1), m.group(2)
    ReadingDict().add_word(path, surface.replace("_", " "), reading)
    _log(f"已加入词库：{surface}（{reading}） → {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="subkit", description=f"subkit {__version__} — 中日双语字幕工程工具")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="在当前目录创建工程文件")
    p.add_argument("--name", default="")
    p.set_defaults(fn=cmd_init)
    p = sub.add_parser("jp-extract", help="从 zip 提取日文字幕到 sources/jp")
    p.add_argument("zip")
    p.add_argument("regex", help="匹配压缩包内文件名，第 1 个分组是集数")
    p.add_argument("--season", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_jp_extract)
    for name, fn, hlp in (("prepare", cmd_prepare, "生成工作文件"), ("refresh", cmd_refresh, "刷新工作文件"),
                          ("edit", cmd_edit, "编辑循环（取代 refresh.bat）"), ("release", cmd_release, "生成发布文件"),
                          ("check", cmd_check, "发布前检查：待确认读音、未处理备注、空行")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("episode")
        if name == "prepare":
            p.add_argument("--force", action="store_true", help="覆盖已有工作文件（旧文件备份到 history/）")
            p.add_argument("--embedding", default=None, help="none / labse / bge-m3 / ollama:qwen3-embedding:8b")
        if name == "release":
            p.add_argument("--blur", type=float, default=None, help="给 CN/JP/振假名 加 \\blur（默认读配置）")
            p.add_argument("-o", "--output", default=None)
        p.set_defaults(fn=fn)
    p = sub.add_parser("voice", help="用配音判定拿不准的读音（whisper，较慢，可选）")
    p.add_argument("episode")
    p.add_argument("--model", default="medium", help="whisper 模型：medium / large-v3")
    p.add_argument("--margin", type=float, default=3.0, help="领先多少才采用（对数概率差）")
    p.add_argument("--no-watashi", action="store_true", help="不检查「私」")
    p.set_defaults(fn=cmd_voice)
    p = sub.add_parser("status", help="列出各集状态")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("dict", help="词库")
    dsub = p.add_subparsers(dest="dcmd", required=True)
    pa = dsub.add_parser("add")
    pa.add_argument("entry")
    pa.set_defaults(fn=cmd_dict)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
