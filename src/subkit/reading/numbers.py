"""Arabic-numeral + counter readings (3本→ぼん, 1人→ひとり, 20日→はつか, 100倍→ばい).

Convention (from the user's finished subtitles): digits normally get no ruby,
only the counter kanji does — with the sound change the number causes.
Special whole-word readings (1人 ひとり, 2日 ふつか, 20歳 はたち) put the ruby
over digits + counter together.
"""

from __future__ import annotations

from typing import Optional

from ..textutil import digits_to_ascii

_DIG = ["ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"]


def number_reading(n: int) -> str:
    if n == 0:
        return "ぜろ"
    if n >= 10 ** 12:
        return "".join(_DIG[int(c)] for c in str(n))
    parts = []
    for unit, name in ((10 ** 8, "おく"), (10 ** 4, "まん")):
        q, n = divmod(n, unit)
        if q:
            # 1000万 = いっせんまん, 1億 = いちおく
            parts.append(_below_10000(q, big=True) + name if q != 1 or unit != 10 ** 4 else "いちまん")
    if n:
        parts.append(_below_10000(n))
    return "".join(parts)


def _below_10000(n: int, big: bool = False) -> str:
    out = ""
    th, n = divmod(n, 1000)
    if th:
        out += {1: "いっせん" if big else "せん", 3: "さんぜん", 8: "はっせん"}.get(th, _DIG[th] + "せん")
    hu, n = divmod(n, 100)
    if hu:
        out += {1: "ひゃく", 3: "さんびゃく", 6: "ろっぴゃく", 8: "はっぴゃく"}.get(hu, _DIG[hu] + "ひゃく")
    te, n = divmod(n, 10)
    if te:
        out += "じゅう" if te == 1 else _DIG[te] + "じゅう"
    if n:
        out += _DIG[n]
    return out


# Whole-word special readings: (number, counter) → reading (ruby spans digits+counter)
SPECIAL = {
    (1, "人"): "ひとり", (2, "人"): "ふたり",
    (1, "つ"): "ひとつ", (2, "つ"): "ふたつ", (3, "つ"): "みっつ", (4, "つ"): "よっつ", (5, "つ"): "いつつ",
    (6, "つ"): "むっつ", (7, "つ"): "ななつ", (8, "つ"): "やっつ", (9, "つ"): "ここのつ",
    (2, "日"): "ふつか", (3, "日"): "みっか", (4, "日"): "よっか", (5, "日"): "いつか", (6, "日"): "むいか",
    (7, "日"): "なのか", (8, "日"): "ようか", (9, "日"): "ここのか", (10, "日"): "とおか", (14, "日"): "じゅうよっか",
    (20, "日"): "はつか", (24, "日"): "にじゅうよっか", (20, "歳"): "はたち",
}

# counter → (base reading, sound-change class)
#   h: はひふへほ counters (本 匹 杯 発 分 泊 版 品 ...): 1,6,8,10 → っぽ ; 3 → ぼ
#   k: か-row counters (回 階 個 ヶ月 曲 軒 回 ...): 1,6,8,10 → っか ; 階/軒 3 → がい/げん
#   s: さ-row (冊 歳 才 隻 足): 1,8,10 → っさ ; 足 3 → ぞく
#   t: た-row (頭 通): 1,8,10 → っと
#   plain: no change
COUNTERS = {
    "本": ("ほん", "h"), "匹": ("ひき", "h"), "杯": ("はい", "h"), "発": ("はつ", "h"), "分": ("ふん", "h"),
    "泊": ("はく", "h"), "品": ("ひん", "h"), "敗": ("はい", "h"), "編": ("へん", "h"), "票": ("ひょう", "h"),
    "回": ("かい", "k"), "階": ("かい", "k3"), "個": ("こ", "k"), "曲": ("きょく", "k"), "軒": ("けん", "k3"),
    "件": ("けん", "k"), "巻": ("かん", "k"), "ヶ月": ("かげつ", "k"), "か月": ("かげつ", "k"), "カ月": ("かげつ", "k"),
    "ヵ月": ("かげつ", "k"), "勝": ("しょう", "s"), "冊": ("さつ", "s"), "歳": ("さい", "s"), "才": ("さい", "s"),
    "足": ("そく", "s3"), "皿": ("さら", "-"), "頭": ("とう", "t"), "通": ("つう", "t"), "点": ("てん", "t"),
    "人": ("にん", "-"), "年": ("ねん", "-"), "月": ("がつ", "-"), "日": ("にち", "-"), "時": ("じ", "-"),
    "秒": ("びょう", "-"), "円": ("えん", "-"), "倍": ("ばい", "-"), "度": ("ど", "-"), "番": ("ばん", "-"),
    "枚": ("まい", "-"), "台": ("だい", "-"), "位": ("い", "-"), "名": ("めい", "-"), "号": ("ごう", "-"),
    "話": ("わ", "-"), "週": ("しゅう", "s"), "組": ("くみ", "-"), "億": ("おく", "-"), "万": ("まん", "-"),
    "千": ("せん", "-"), "百": ("ひゃく", "-"), "割": ("わり", "-"), "戦": ("せん", "s"), "局": ("きょく", "k"),
    "着": ("ちゃく", "t"), "期": ("き", "k"), "学期": ("がっき", "-"), "階級": ("かいきゅう", "-"),
    "体": ("たい", "t"), "羽": ("わ", "-"), "粒": ("つぶ", "-"), "歩": ("ほ", "h"), "缶": ("かん", "k"),
}

_H_ROW = {"は": "ぱ", "ひ": "ぴ", "ふ": "ぷ", "へ": "ぺ", "ほ": "ぽ"}
_H_ROW3 = {"は": "ば", "ひ": "び", "ふ": "ぶ", "へ": "べ", "ほ": "ぼ"}


def counter_reading(n: int, counter: str) -> tuple[str, bool]:
    """Reading of the counter part after number n.  Returns (reading, is_whole_word)."""
    if (n, counter) in SPECIAL:
        return SPECIAL[(n, counter)], True
    if counter == "月":
        return {4: "がつ", 7: "がつ", 9: "がつ"}.get(n, "がつ"), False
    if counter == "時":
        return "じ", False
    if counter == "人" and n >= 3:
        return "にん", False
    base, cls = COUNTERS.get(counter, ("", "-"))
    if not base:
        return "", False
    last = n % 10
    tens_only = n % 10 == 0 and n > 0
    if cls == "h":
        if counter == "分":
            return ("ぷん" if last in (1, 3, 4, 6, 8, 0) else "ふん"), False
        if last in (1, 6, 8) or tens_only:
            return _H_ROW.get(base[0], base[0]) + base[1:], False
        if last == 3:
            three = {"本": "ぼん", "匹": "びき", "杯": "ばい", "発": "ぱつ", "泊": "ぱく", "品": "ぴん",
                     "編": "ぺん", "票": "びょう", "歩": "ぽ", "敗": "ぱい"}
            return three.get(counter, _H_ROW3.get(base[0], base[0]) + base[1:]), False
        return base, False
    if cls in ("k", "k3"):
        if cls == "k3" and last == 3:
            return ("がい" if counter == "階" else "げん"), False
        return base, False
    if cls in ("s", "s3", "t"):
        if cls == "s3" and last == 3:
            return "ぞく", False
        return base, False
    return base, False


def number_prefix_reading(digits: str, counter: str) -> str:
    """How the number itself is read before ``counter`` (for full-word readings / audio checks)."""
    n = int(digits_to_ascii(digits))
    base = number_reading(n)
    _, cls = COUNTERS.get(counter, ("", "-"))
    last = n % 10
    if cls == "h" and (last in (1, 6, 8) or (n % 10 == 0 and n)):
        base = base[:-1] + "っ" if base[-1] in "ちくう" else base
    if cls in ("k", "k3", "s", "s3", "t") and (last in (1, 8) or (n % 10 == 0 and n)):
        base = base[:-1] + "っ" if base[-1] in "ちくう" else base
    return base


_UNIT = {"十": 10, "百": 100, "千": 1000}
_BIG = {"万": 10 ** 4, "億": 10 ** 8, "兆": 10 ** 12}


def digit_kanji_number(s: str) -> Optional[int]:
    """'1000万' → 10000000, '2万5千' → 25000, '10億' → 1000000000; None if not such a number.
    Requires at least one digit and at least one kanji unit."""
    import re

    s = digits_to_ascii(s)
    if not re.fullmatch(r"(?:\d+[十百千万億兆]+)+\d*", s):
        return None
    total = section = 0
    for num, units in re.findall(r"(\d+)([十百千万億兆]*)", s):
        n = int(num)
        if not units:
            section += n
            continue
        for u in units:
            if u in _UNIT:
                n = n * _UNIT[u]
            else:
                total += (section + n) * _BIG[u]
                section, n = 0, 0
        section += n
    return total + section


def parse_int(digits: str) -> Optional[int]:
    try:
        return int(digits_to_ascii(digits))
    except ValueError:
        return None
