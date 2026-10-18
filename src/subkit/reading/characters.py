"""Character table (dict/characters.tsv) — one row per character.

Columns (tab separated, header line starts with #):

    日文名        幸平 創真        (a space separates family/given name → separate ruby groups)
    假名          ゆきひら そうま
    中文名        幸平创真         (reference; shown in reports)
    别名          創真/幸平/ソーマ  (how the Japanese captions label the speaker)
    私的读法      わたし / わたくし / あたし   (self reference used when this character says 私)
    备注          free text

Effects:
* the names become locked dictionary words (full name and multi-character parts);
* caption speaker labels are mapped to the canonical name (written to the JP
  line's Actor field), so per-character readings apply automatically;
* labels that are not in the table are appended as new rows to fill in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..textutil import has_kanji

HEADER = "# 日文名\t假名\t中文名\t别名（字幕标签写法，/ 分隔）\t私的读法\t备注"


@dataclass
class Character:
    name: str                 # canonical, without spaces
    name_parts: list[str]
    kana_parts: list[str]
    cn: str = ""
    aliases: list[str] = field(default_factory=list)
    self_reading: str = ""
    note: str = ""

    @property
    def kana(self) -> str:
        return "".join(self.kana_parts)


class CharacterTable:
    def __init__(self) -> None:
        self.chars: list[Character] = []
        self._by_label: dict[str, Character] = {}
        self.path: Optional[Path] = None

    @classmethod
    def load(cls, path: Path) -> "CharacterTable":
        t = cls()
        t.path = path
        if not path.exists():
            return t
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            cols = raw.split("\t") + [""] * 6
            name, kana, cn, aliases, selfr, note = (c.strip() for c in cols[:6])
            if not name:
                continue
            parts = name.split()
            kparts = kana.split() if kana else []
            if len(kparts) != len(parts):
                kparts = [kana.replace(" ", "")] if kana else []
                parts = ["".join(parts)] if kparts else parts
            ch = Character("".join(name.split()), parts, kparts, cn, [a.strip() for a in aliases.replace("／", "/").split("/") if a.strip()], selfr, note)
            t.add(ch)
        return t

    def add(self, ch: Character) -> None:
        self.chars.append(ch)
        for label in [ch.name, *ch.name_parts, *ch.aliases]:
            if label and label not in self._by_label:
                self._by_label[label] = ch

    def lookup(self, label: Optional[str]) -> Optional[Character]:
        if not label:
            return None
        return self._by_label.get(label) or self._by_label.get(label.replace(" ", ""))

    def words(self) -> list[tuple[str, str]]:
        """(key, reading) dictionary words: full names with groups + multi-char parts."""
        out = []
        for ch in self.chars:
            if not ch.kana_parts or not has_kanji(ch.name):
                continue
            out.append((" ".join(ch.name_parts), " ".join(ch.kana_parts)))
            if len(ch.name_parts) > 1:
                for p, k in zip(ch.name_parts, ch.kana_parts):
                    if has_kanji(p) and len(p) >= 2:
                        out.append((p, k))
            for a in ch.aliases:
                # an alias that is exactly one name part reuses that part's reading
                if a in ch.name_parts and has_kanji(a):
                    out.append((a, ch.kana_parts[ch.name_parts.index(a)]))
        return out

    def append_unknown(self, labels: dict[str, int], hints: dict[str, str]) -> list[str]:
        """Append labels not yet in the table (skipping generic ones); returns what was added.

        A short label that is the start or end of a longer one (麗 / 川島麗, 大泉 /
        大泉柿之進) becomes that row's alias instead of a row of its own."""
        if self.path is None:
            return []
        new = [l for l, _n in sorted(labels.items(), key=lambda kv: -kv[1])
               if l.strip() and self.lookup(l) is None and not is_generic_label(l)]
        if not new:
            return []
        rows: dict[str, list[str]] = {}       # new full label → its aliases
        alias_of: dict[str, str] = {}         # short label → full label (a new row or an existing one)
        known = [ch.name for ch in self.chars]
        for label in sorted(new, key=len, reverse=True):
            longer = [m for m in [*rows, *known] if len(m) > len(label) and (m.startswith(label) or m.endswith(label))]
            if longer:
                alias_of[label] = longer[0]
                if longer[0] in rows:
                    rows[longer[0]].append(label)
            else:
                rows[label] = []
        to_existing = {s: f for s, f in alias_of.items() if f not in rows}
        if to_existing:
            self._add_aliases(to_existing)
        exists = self.path.exists() and self.path.stat().st_size > 0
        lines = [] if exists else [HEADER]
        for label in [l for l in new if l in rows]:
            kana = hints.get(label, "")
            aliases = rows[label]
            count = labels[label] + sum(labels[a] for a in aliases)
            lines.append(f"{label}\t{kana}\t\t{'/'.join(aliases)}\t\t自动加入：出现 {count} 次")
            self.add(Character(label, [label], [kana] if kana else [], aliases=list(aliases)))
        if len(lines) > (0 if exists else 1):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        return [l for l in new if l in rows] + [f"{s}（{f} 的别名）" for s, f in alias_of.items()]

    def _add_aliases(self, alias_of: dict[str, str]) -> None:
        """Add aliases to rows already in the file (only the alias column changes)."""
        if self.path is None or not self.path.exists():
            return
        out = []
        for raw in self.path.read_text(encoding="utf-8-sig").splitlines():
            cols = raw.split("\t")
            if raw.strip() and not raw.lstrip().startswith("#"):
                name = "".join(cols[0].split())
                extra = [s for s, f in alias_of.items() if f == name]
                if extra:
                    cols += [""] * (6 - len(cols))
                    have = [a.strip() for a in cols[3].replace("／", "/").split("/") if a.strip()]
                    cols[3] = "/".join(have + [a for a in extra if a not in have])
                    raw = "\t".join(cols)
            out.append(raw)
        self.path.write_text("\n".join(out) + "\n", encoding="utf-8")
        for s, f in alias_of.items():
            ch = self.lookup(f)
            if ch is not None and s not in ch.aliases:
                ch.aliases.append(s)
                self._by_label.setdefault(s, ch)


GENERIC_LABELS = {
    "一同", "全員", "学生", "学生たち", "生徒", "生徒たち", "客", "客たち", "店員", "司会", "アナウンス", "ナレーション",
    "男", "女", "少年", "少女", "男子生徒", "女子生徒", "男子学生", "女子学生", "男性", "女性", "観客", "みんな",
    "料理人たち", "料理人", "審査員", "声", "子供", "子供たち", "記者", "店主", "部員", "部員たち", "教師", "先生",
    "受験生", "護衛", "板前", "コック", "シェフ", "料理長", "スタッフ", "リポーター", "レポーター", "アナウンサー",
    "母", "父", "母親", "父親", "祖父", "祖母", "叔父", "叔母", "伯父", "伯母", "兄", "姉", "弟", "妹", "主人", "女将",
    "店長", "社長", "部長", "会長", "校長", "総帥", "部門長", "実況", "解説", "審判", "警官", "警備員", "運転手", "医者",
    "来賓", "講師", "研究員", "漁師", "農家", "仲居", "給仕", "ウェイター", "ウェイトレス", "老人", "老婆", "おばあさん",
    "おじいさん", "おばさん", "おじさん", "お客", "お客さん", "常連", "常連客", "テレビ", "ラジオ", "放送", "電話",
    "犬", "猫", "鳥", "カラス", "ペット", "生徒会", "観衆", "群衆", "選手", "相手",
}
_ROLE_SUFFIX = ("員", "係", "師", "官", "者", "客", "手")


def is_generic_label(label: str) -> bool:
    """Crowd / role labels (一同, ３人, 学生たち) and multi-speaker labels (創真・城一郎)."""
    import re

    if label in GENERIC_LABELS or "・" in label or "･" in label or ":" in label or "：" in label:
        return True
    if re.search(r"[0-9０-９A-ZＡ-Ｚa-zａ-ｚ]$", label) and re.search(r"[^0-9０-９A-ZＡ-Ｚa-zａ-ｚ]", label):
        return True  # 女子生徒１, 審査員Ｂ, 男性Ａ — numbered / lettered extras
    if len(label) >= 2 and label.endswith(_ROLE_SUFFIX):
        return True  # 研究員, サービス係, 漁師, 記者, 観光客
    if re.match(r"^[0-9０-９一二三四五六七八九十]+人", label):
        return True
    return re.search(r"(たち|ら|達)$", label) is not None and len(label) <= 6
