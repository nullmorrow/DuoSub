"""Context rules for words whose dictionary default is wrong in dialogue.

Sudachi/UniDic pick one reading per lexeme: 何→なん, 私→わたくし, 明日→あす,
日本→にっぽん, お兄様→あに, 一日中→ちゅう, その間→ま, 辛さ→つら...
Each rule sees the morpheme and its neighbours and returns
(reading, confidence, alternatives) or None (= keep the analyzer reading).
Confidence < ``REVIEW_BELOW`` makes the entry show up as ``?`` for review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

REVIEW_BELOW = 0.75

COUNTER_KANJI = set("人回度時日年歳才個本枚月週分秒番階倍杯匹台冊通点名話位着組戦勝敗件軒曲巻頭羽体粒歩缶期")
# nouns after which 中 is read じゅう ("throughout")
# (今週中/今月中/今年中 mean "within" and are read ちゅう, so they are not listed)
JUU_NOUNS = {"一日", "一年", "一晩", "一週間", "一月", "今日", "明日", "年", "世界", "日本", "学校", "学園", "町", "街", "村", "国", "体",
             "部屋", "家", "城", "店", "島", "会社", "寮", "夏休み", "冬休み", "春休み", "休み", "夜", "校内",
             "そこら", "そこいら", "この辺", "そこら中", "体育館", "家族"}
PUNCT_END = set("、。！？!?…‥」』）) 　~～")


@dataclass
class Ctx:
    text: str
    start: int
    end: int
    surface: str
    reading: str                 # analyzer reading (hiragana)
    prev_surface: str
    next_surface: str
    next_reading: str
    speaker: Optional[str]

    @property
    def after(self) -> str:
        return self.text[self.end:]

    @property
    def before(self) -> str:
        return self.text[: self.start]


Rule = Callable[[Ctx], Optional[tuple[str, float, list[str]]]]
RULES: dict[str, Rule] = {}


def rule(*surfaces: str):
    def deco(fn: Rule) -> Rule:
        for s in surfaces:
            RULES[s] = fn
        return fn
    return deco


@rule("何")
def _nani(c: Ctx):
    nxt = c.next_surface
    tail = c.after.lstrip()
    if not tail or tail[0] in PUNCT_END:
        return "なに", 0.9, ["なん"]
    if tail[0] in COUNTER_KANJI:
        return "なん", 0.95, ["なに"]
    for p in ("だった", "だって", "だか", "だい", "だろ", "だよ", "だね", "だ", "です", "でしょ", "でも", "と", "の", "て", "なの", "なんだ", "じゃ", "ぞ", "なり", "なら"):
        if tail.startswith(p):
            return "なん", 0.95, ["なに"]
    if tail.startswith("で"):
        # 何で: なんで (why) is far more common in dialogue than なにで (by what)
        return "なん", 0.8, ["なに"]
    for p in ("か", "が", "を", "も", "に", "へ", "から", "より", "まで", "や", "かしら", "して", "しに", "それ", "これ", "事", "こと", "者", "物", "もの", "様"):
        if tail.startswith(p):
            return "なに", 0.92, ["なん"]
    return "なに", 0.6, ["なん"]


_FORMAL = ("ですわ", "ますわ", "ございます", "ございません", "ましてよ", "あそばせ", "ませんこと", "わね", "かしら", "ですの", "ますの", "様")


@rule("私")
def _watashi(c: Ctx):
    # わたくし is a speaker trait — use a speakers rule in the dictionary for recurring
    # characters.  Without one, very polite / ojousama speech gets a review flag.
    if any(m in c.text for m in _FORMAL):
        return "わたし", 0.6, ["わたくし", "あたし"]
    return "わたし", 0.85, ["わたくし", "あたし"]


@rule("方")
def _kata(c: Ctx):
    prev = c.before.rstrip()
    tail = c.after.lstrip()
    after_verb = bool(prev) and prev[-1] in "たるうくぐすつぬぶむいえ" and not prev.endswith(("この", "その", "あの", "どの"))
    if c.reading == "ほう" and after_verb and not tail.startswith(("が", "へ", "に向")):
        # 〜した方は/の/です — "the person who…" (かた) or "the side that…" (ほう)
        return "ほう", 0.6, ["かた"]
    return None


@rule("品")
def _hin(c: Ctx):
    tail = c.after
    if tail.startswith(("のない", "がない", "がいい", "のいい", "のある", "がある", "の良", "が良", "のよ", "がよ")):
        return "ひん", 0.9, ["しな"]
    return None


@rule("部屋")
def _beya(c: Ctx):
    if c.prev_surface.endswith("人") or c.before.endswith(("人", "子供", "子ども", "大")):
        return "べや", 0.85, ["へや"]
    return None


@rule("深い", "深く", "深さ", "深")
def _bukai(c: Ctx):
    if c.prev_surface in ("興味", "用心", "注意", "欲", "疑い", "情け", "罪", "慎み", "思慮", "執念", "愛情", "嫉妬", "業"):
        return "ぶか" + c.reading[2:], 0.9, [c.reading]
    return None


@rule("一")
def _hito(c: Ctx):
    # 一突き, 一思い, 一休み: 一 + verb-stem noun → ひと
    nxt = c.next_surface
    if c.reading == "いち" and nxt and nxt[-1:] in "いきしちにひみりぎじびぴ" and any("一" <= ch <= "鿿" for ch in nxt):
        return "ひと", 0.75, ["いち"]
    return None


@rule("不味い", "不味", "不味かっ", "不味く", "不味さ", "不味けれ")
def _mazui(c: Ctx):
    # Sudachi sometimes picks the archaic まじい
    if c.reading.startswith("まじ"):
        return "まず" + c.reading[2:], 0.95, [c.reading]
    return None


@rule("体")
def _karada(c: Ctx):
    # 体中 / 体じゅう = からだじゅう ("all over the body")
    if c.after.startswith(("中", "じゅう")) and c.reading in ("たい", "てい"):
        return "からだ", 0.9, [c.reading]
    return None


@rule("脅かす", "脅かし", "脅かさ", "脅かそ", "脅かせ", "脅かし")
def _odokasu(c: Ctx):
    # casual speech: おどかす (scare); おびやかす (threaten) is written/formal
    if c.reading.startswith("おびやか"):
        return "おどか" + c.reading[4:], 0.7, [c.reading]
    return None


@rule("明日")
def _ashita(c: Ctx):
    return "あした", 0.85, ["あす", "みょうにち"]


@rule("日本")
def _nihon(c: Ctx):
    return "にほん", 0.8, ["にっぽん"]


@rule("兄")
def _ani(c: Ctx):
    if c.prev_surface == "お":
        return "にい", 0.95, ["あに"]
    if c.next_surface in ("さん", "ちゃん", "様", "さま"):
        return "にい", 0.8, ["あに"]
    return None


@rule("姉")
def _ane(c: Ctx):
    if c.prev_surface == "お" or c.next_surface in ("さん", "ちゃん", "様"):
        return "ねえ", 0.9 if c.prev_surface == "お" else 0.8, ["あね"]
    return None


@rule("中")
def _juu(c: Ctx):
    if c.prev_surface in JUU_NOUNS or any(c.before.endswith(n) for n in JUU_NOUNS):
        return "じゅう", 0.9, ["ちゅう"]
    return None


@rule("今日")
def _kyou(c: Ctx):
    # 今日中（きょうじゅう）; こんにち only in formal set phrases, which captions rarely use
    if c.after.startswith(("中", "じゅう")):
        return "きょう", 0.95, ["こんにち"]
    return None


@rule("間")
def _aida(c: Ctx):
    if c.reading == "ま" and (c.prev_surface in ("の", "その", "この", "あの", "どの") or c.before.endswith("の")):
        tail = c.after.lstrip()
        if not tail or tail[0] in PUNCT_END or tail[:1] in ("に", "は", "も", "で", "が", "を", "じゅう", "中"):
            return "あいだ", 0.85, ["ま", "かん"]
    return None


@rule("一日")
def _ichinichi(c: Ctx):
    if "月" in c.before[-3:]:
        return "ついたち", 0.85, ["いちにち"]
    return "いちにち", 0.8, ["ついたち"]


# Words with several common readings: candidates (first = usual default).
# The engine flags them for review unless a rule, the dictionary, a project
# preference or a resolver (LLM / audio) is confident.
HETERONYMS: dict[str, list[str]] = {
    "辛い": ["からい", "つらい"], "辛さ": ["からさ", "つらさ"], "辛く": ["からく", "つらく"],
    "辛かっ": ["からかっ", "つらかっ"], "辛": ["から", "つら"],
    "方": ["ほう", "かた"], "上手": ["じょうず", "うわて", "かみて"], "下手": ["へた", "したて", "しもて"],
    "人気": ["にんき", "ひとけ"], "最中": ["さいちゅう", "もなか"], "風車": ["ふうしゃ", "かざぐるま"],
    "市場": ["いちば", "しじょう"], "寒気": ["さむけ", "かんき"], "大家": ["おおや", "たいか"],
    "心中": ["しんじゅう", "しんちゅう"], "色紙": ["しきし", "いろがみ"], "一行": ["いっこう", "いちぎょう"],
    "目下": ["もっか", "めした"], "生物": ["せいぶつ", "なまもの"], "上下": ["じょうげ", "うえした"],
    "今日": ["きょう", "こんにち"], "昨日": ["きのう", "さくじつ"], "分": ["ぶん", "ふん"],
    "金": ["かね", "きん"], "後": ["あと", "のち", "うしろ", "ご"], "何人": ["なんにん", "なにじん"],
    "開く": ["ひらく", "あく"], "止め": ["とめ", "やめ"], "止める": ["とめる", "やめる"], "行く": ["いく", "ゆく"],
    "入る": ["はいる", "いる"], "身体": ["からだ", "しんたい"], "一緒": ["いっしょ"], "仕方": ["しかた"],
    "生": ["なま", "せい", "しょう"], "大事": ["だいじ", "おおごと"], "大人": ["おとな", "たいじん"],
    "家": ["いえ", "うち", "け", "か"], "空": ["そら", "から", "くう"], "物": ["もの", "ぶつ"],
}
