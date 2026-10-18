"""Pure conversion plans and Chinese UI text for dialogue ↔ song (KARAOKE §2.7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from ..model.document import Document, Member
from . import karaoke as kara
from .merge import MergeSettings
from .readings import visible_map

DIALOGUE_ROLES = ("dialogue", "top", "alt")
FAMILIES = ("OP", "ED", "IN")
K_HINT = "这一组不是歌曲，不能打 K 轴（右键 → 转为歌曲）"


@dataclass
class ConvertPlan:
    to: str
    eligible: list[str] = field(default_factory=list)       # selected, in document order
    twins: list[str] = field(default_factory=list)          # additional groups, in document order
    gids: list[str] = field(default_factory=list)           # eligible + twins, in document order
    skipped: list[tuple[str, str]] = field(default_factory=list)
    members_changed: int = 0
    timed_members: list[str] = field(default_factory=list)
    dormant_members: list[str] = field(default_factory=list)
    source_k_members: list[str] = field(default_factory=list)
    target_styles: list[str] = field(default_factory=list)
    missing_styles: list[str] = field(default_factory=list)
    default_family: Optional[str] = None
    family: Optional[str] = None
    sign_groups: int = 0


def member_target(m: Member, to: str, styles: Mapping[str, str], family: Optional[str]) -> Optional[tuple[str, str]]:
    """Role/style after conversion; protected roles are untouched."""
    old = m.extra.get("before_convert") or {}
    if to == "song" and m.role in DIALOGUE_ROLES:
        saved = old.get("style", "")
        style = saved if old.get("role") == "song" and saved.split(".", 1)[0] == family else \
            styles[f"{m.lane}/song"].format(fam=family)
        return "song", style
    if to == "dialogue" and m.role == "song":
        if old.get("role") in DIALOGUE_ROLES:
            return old["role"], old["style"]
        return "dialogue", styles[f"{m.lane}/dialogue"]
    if to == "dialogue" and m.role == "sign":
        return "dialogue", styles[f"{m.lane}/dialogue"]
    if to == "sign" and m.role in DIALOGUE_ROLES:
        saved = m.extra.get("before_sign_convert") or {}
        return "sign", saved.get("style", m.style)
    return None


def plan_conversion(doc: Document, gids: Iterable[str], to: str, *, styles: Optional[Mapping[str, str]] = None,
                    template_styles: Optional[Iterable[str]] = None, family: Optional[str] = None) -> ConvertPlan:
    """Inspect without mutating. Only direct twins of selected eligible members are added.

    ``dormant_members`` counts members with saved dormant tracks, not the number of tracks.
    Missing styles are checked only when template names are supplied.
    """
    if to not in ("song", "dialogue", "sign"):
        raise ValueError(to)
    if family is not None and family not in FAMILIES:
        raise ValueError(family)
    styles = MergeSettings().styles if styles is None else styles
    current = {"dialogue"} if to in ("song", "sign") else {"song", "sign"}
    selected = set(gids)
    plan = ConvertPlan(to)
    reasons = {"dialogue": "不是歌曲（普通台词）", "song": "不是普通台词（已经是歌曲）",
               "sign": "屏幕字，不转换", "info": "制作信息或注释，不转换"}
    for g in doc.groups:
        if g.id not in selected:
            continue
        if g.kind in current:
            plan.eligible.append(g.id)
            plan.sign_groups += int(g.kind == "sign")
        else:
            plan.skipped.append((g.id, reasons.get(g.kind, "种类不适用，不转换")))
    live = {g.id for g in doc.groups}
    plan.skipped.extend((gid, "组不存在，不转换") for gid in sorted(selected - live))
    roles = DIALOGUE_ROLES if to in ("song", "sign") else ("song", "sign")
    eligible = set(plan.eligible)
    seeds = [doc.members[mid] for g in doc.groups if g.id in eligible
             for mid in g.members if doc.members[mid].role in roles
             and (to != "dialogue" or doc.members[mid].role == g.kind)]
    # Do not grow a chain of near matches beyond the selected lines' 20 ms window.
    seed_texts = [(m, visible_map(m)[0].text) for m in seeds
                  if to != "dialogue" or doc.group(m.group).kind == "song"]
    for g in doc.groups:
        if g.kind not in current or g.id in selected:
            continue
        if to == "sign" or (to == "dialogue" and g.kind != "song"):
            continue
        if any(o.lane == m.lane and abs(o.start - m.start) <= 20 and abs(o.end - m.end) <= 20
               and visible_map(o)[0].text == text
               for mid in g.members for o in (doc.members[mid],) for m, text in seed_texts):
            plan.twins.append(g.id)
    targets = set(plan.eligible + plan.twins)
    plan.gids = [g.id for g in doc.groups if g.id in targets]
    members = [doc.members[mid] for g in doc.groups if g.id in targets
               for mid in g.members if doc.members[mid].role in roles
               and (to != "dialogue" or doc.members[mid].role == g.kind)]
    for m in members:
        old = m.extra.get("before_convert") or {}
        fam = old.get("style", "").split(".", 1)[0]
        if old.get("role") == "song" and fam in FAMILIES:
            plan.default_family = fam
            break
    plan.family = (family or plan.default_family or "IN") if to == "song" else None
    idx = kara.KIndex(doc)
    for m in members:
        plan.members_changed += 1
        if idx.has(m) and any(b.timed for b in kara.effective(doc, m, idx=idx).beats):
            plan.timed_members.append(m.id)
        if any(t.get("dormant") for t in kara.tracks_of(doc, m.id, dormant=True)):
            plan.dormant_members.append(m.id)
        if kara.has_k_tags(m):
            plan.source_k_members.append(m.id)
    # Include both lane names in the live dialog label even for a single-language selection.
    names = [] if to == "sign" else [styles[f"{lane}/{to}"].format(fam=plan.family) for lane in ("translation", "original")]
    names.extend(member_target(m, to, styles, plan.family)[1] for m in members)
    plan.target_styles = list(dict.fromkeys(names))
    if template_styles is not None:
        available = set(template_styles)
        plan.missing_styles = [name for name in plan.target_styles if name not in available]
    return plan


def song_text(plan: ConvertPlan) -> str:
    lines = [f"把选中的 {len(plan.eligible)} 组转为歌曲。转换后选中一组按 K 就能打轴；"
             "原来的样式和位置会记住，转回普通台词时恢复。"]
    if plan.twins:
        lines.append(f"另外 {len(plan.twins)} 组是同一句的另一图层（同时间、同文字），一起转换。")
    if plan.skipped:
        lines.append(f"{len(plan.skipped)} 组不是普通台词（屏幕字、注释或已经是歌曲），不转换。")
    if plan.dormant_members:
        lines.append(f"会恢复 {len(plan.dormant_members)} 行之前打过的 K 轴。")
    if plan.missing_styles:
        lines.append(f"模板里没有 {' / '.join(plan.missing_styles)}，导出时会用默认样式。")
    return "\n".join(lines)


def song_style_text(plan: ConvertPlan) -> str:
    return "样式：" + " / ".join(plan.target_styles)


def dialogue_text(plan: ConvertPlan, *, styles: Optional[Mapping[str, str]] = None) -> str:
    styles = MergeSettings().styles if styles is None else styles
    names = " / ".join(styles[f"{lane}/dialogue"] for lane in ("translation", "original"))
    lines = [f"把选中的 {len(plan.eligible)} 组转为普通台词？", f"· 样式恢复成转换前的（没有记录的用 {names}）"]
    if plan.sign_groups:
        lines.append(f"· {plan.sign_groups} 组屏幕字使用 {names} 和普通台词位置；转回屏幕字恢复来源样式和位置")
    if plan.timed_members:
        lines.append(f"· 已打的 K 轴（{len(plan.timed_members)} 行）留在工程里但不导出，转回歌曲时恢复")
    if plan.source_k_members:
        lines.append(f"· 原字幕里自带的 \\k（{len(plan.source_k_members)} 行）不写进输出，原字幕不变")
    if plan.twins:
        lines.append(f"· 另外 {len(plan.twins)} 组是同一句的另一图层，一起转换")
    if plan.skipped:
        lines.append(f"· {len(plan.skipped)} 组不是歌曲或屏幕字，不转换")
    lines.append("（可以撤销）")
    return "\n".join(lines)


def status_text(plan: ConvertPlan) -> str:
    target = {"song": "歌曲", "dialogue": "普通台词", "sign": "屏幕字"}[plan.to]
    return f"已转为{target}：{len(plan.gids)} 组（Ctrl+Z 撤销）"


def export_stripped_text(refs: Iterable[str]) -> str:
    refs = list(refs)
    if not refs:
        return ""
    return (f"K 轴：{len(refs)} 行已转为普通台词，原字幕里的 \\k 没有写进输出（原字幕不变）"
            + "\n" + "\n".join(refs[:6]))
