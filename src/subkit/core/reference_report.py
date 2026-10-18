"""Self-contained, escaped reference-comparison reports and export input protection."""
from __future__ import annotations

from collections import Counter
import html
import os
from pathlib import Path
import tempfile


def preflight_targets(targets, protected_input_paths) -> None:
    """Reject any output that names, resolves to, or hard-links an input file."""
    protected = [Path(p).absolute().resolve() for p in protected_input_paths if p]
    for raw in targets:
        target = Path(raw).absolute().resolve()
        for source in protected:
            same = target == source
            if not same and target.exists() and source.exists():
                same = target.samefile(source)
            if same:
                raise PermissionError(f"不能用输出覆盖原字幕或参考输入：{raw}")


def _escaped(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _time(ms) -> str:
    try:
        ms = int(ms)
    except (TypeError, ValueError):
        return "—"
    return f"{ms // 3600000}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"


def render_reference_report(data: dict, *, runner=None) -> str:
    """Render the core's completed comparison without re-reading files or editing text."""
    def check():
        if runner is not None:
            runner.check_cancelled()
    check()
    direction = {"simplified": "简体", "traditional": "繁体", "off": "不转换"}.get(data.get("direction"), "不转换")
    rows, sources = data.get("rows", []), data.get("sources", [])
    strategies = {"reference": "采用原配字幕", "merged": "迁移原配差异并保留编辑",
                  "opencc": "采用自动简繁转换", "unchanged": "保持当前文字", "replacement": "应用工程批量替换"}
    counts = Counter(row.get("strategy", "unchanged") for row in rows)
    out = ["<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">",
           "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
           "<title>原配简繁字幕对照报告 — " + _escaped(data.get("episode", "")) + "</title>",
           """<style>
body{font:16px/1.55 system-ui,'Microsoft YaHei',sans-serif;background:#f6f8fa;color:#24292f;margin:0}
main{max-width:1180px;margin:auto;padding:28px}h1{font-size:26px}h2{font-size:21px}h3{font-size:17px}
article,.sources,.summary{background:white;border:1px solid #d0d7de;border-radius:8px;padding:18px;margin:18px 0}
summary{cursor:pointer;font-weight:600}summary small{font-weight:400;color:#57606a}.badge{color:#9a6700}
.muted{color:#57606a}.text-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.field{min-width:0;border:1px solid #d8dee4;padding:10px;border-radius:5px}.field strong{display:block;color:#57606a;font-size:13px}
pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere;margin:6px 0}code,.path{overflow-wrap:anywhere;font-size:13px}
.final{background:#dafbe1}.reason{background:#fff8c5;padding:10px}.source{border-top:1px solid #d8dee4;padding:12px 0}
@media(max-width:700px){main{padding:12px}.text-grid{grid-template-columns:1fr}}@media print{body{background:white}article{break-inside:avoid}}
</style><main>""",
           "<h1>原配简繁字幕对照报告</h1>",
           "<p>集：<b>" + _escaped(data.get("episode", "")) + "</b>　输出方向：" + direction + "</p>",
           "<p class=\"muted\">本报告记录自动导出的处理结果。参考缺失、匹配不唯一或不能安全迁移时会自动回退；字幕已经按下列结果生成，无需逐条确认。</p>",
           "<section class=\"summary\"><h2>本次结果</h2><p>共 " + str(len(rows)) + " 行中文。" +
           "；".join(_escaped(strategies.get(key, key)) + " " + str(value) + " 行" for key, value in counts.items()) + "</p>"]
    changed = sum(bool(row.get("changed")) for row in rows)
    preimport = sum(bool(row.get("preimport_changed")) for row in rows)
    out.append(f"<p>相对导入原文改过正文：{changed} 行；导入原文与可用原版底稿不同：{preimport} 行。时间调整不计入改字。</p>"
               "<p class=\"muted\">有正文改动或自动回退的行默认展开；其余行点击标题展开全部对照。可用浏览器查找字幕文字或时间。</p>")
    if data.get("issues"):
        out.append("<ul>" + "".join("<li>" + _escaped(issue) + "</li>" for issue in data["issues"]) + "</ul>")
    out.append("</section><section class=\"sources\"><h2>来源文件与 SHA-256</h2>")
    for source in sources:
        check()
        out.append("<div class=\"source\">")
        for label, path_key, hash_key in (("导入原字幕", "source_path", "source_sha256"),
                                        ("原版底稿", "baseline_path", "baseline_sha256"),
                                        ("目标原配字幕", "target_path", "target_sha256")):
            out.append("<p><b>" + label + "</b><br><span class=\"path\">" + _escaped(source.get(path_key) or "未找到") +
                       "</span><br><code>SHA-256：" + _escaped(source.get(hash_key) or "—") + "</code></p>")
        if source.get("issues"):
            out.append("<ul>" + "".join("<li>" + _escaped(issue) + "</li>" for issue in source["issues"]) + "</ul>")
        out.append("</div>")
    if not sources:
        out.append("<p>没有可用的参考来源，详细回退原因见本次结果与各行说明。</p>")
    out.append("</section><h2>逐行对照</h2>")
    for index, row in enumerate(rows, 1):
        check()
        expanded = row.get("changed") or row.get("preimport_changed") or row.get("strategy") in ("opencc", "replacement")
        badge = (" · 工程内正文有改动" if row.get("changed") else "") + (
            " · 导入原文与原版不同" if row.get("preimport_changed") else "")
        out.append("<article><details" + (" open" if expanded else "") + "><summary>" + str(index) + " · " +
                   _time(row.get("start")) + " — " + _time(row.get("end")) + " · " +
                   _escaped(strategies.get(row.get("strategy"), row.get("strategy", "保持当前文字"))) +
                   "<span class=\"badge\">" + badge + "</span><br><small>" + _escaped(row.get("final", "")[:70]) + "</small></summary>")
        if row.get("preimport_changed"):
            out.append("<p class=\"muted\">导入的原字幕已经与原版底稿不同；这些差异也纳入了迁移判断。</p>")
        out.append("<div class=\"text-grid\">")
        for label, key in (("导入原文", "imported"), ("当前工程文字", "current"), ("原版同字形底稿", "baseline"),
                           ("目标原配字幕", "reference"), ("纯自动转换", "automatic"),
                           ("采用文字（显示规则前）", "generated"), ("最终 ASS 可见文字", "final")):
            out.append("<div class=\"field" + (" final" if key == "final" else "") + "\"><strong>" + label +
                       "</strong><pre>" + _escaped(row.get(key) if row.get(key) is not None else "—") + "</pre></div>")
        out.append("</div><p class=\"reason\">原因：" + _escaped(row.get("reason") or "—") + "</p>")
        out.append("<p class=\"muted path\">原字幕：" + _escaped(row.get("source_path") or "—") +
                   "<br>原事件 ID：" + _escaped(row.get("source_event")) + "　组：" + _escaped(row.get("group")) +
                   "　成员：" + _escaped(row.get("member")) + "</p></details></article>")
    if not rows:
        out.append("<p>本次没有可列出的中文行。</p>")
    out.append("</main></html>\n")
    check()
    return "\n".join(out)


def write_reference_report(data: dict, target: Path) -> Path:
    """Atomically write HTML already built by the worker, preserving any earlier report on failure."""
    path = Path(target).with_suffix(".reference.html")
    preflight_targets([path], data.get("protected_input_paths", []))
    if data.get("html_error"):
        raise ValueError(data["html_error"])
    if not isinstance(data.get("html"), str):
        raise ValueError("原配字幕对照报告尚未生成")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data["html"].encode("utf-8"))
        preflight_targets([path], data.get("protected_input_paths", []))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
