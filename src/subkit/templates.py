"""Style templates for a project: a colour scheme (配色) × a font set — presets to start from, every value adjustable
in the visual template editor (subkit_app/views/template_editor.py).

The presets come from the user's own releases: NANA (Noto Sans CJK Black, purple outline), Utena (思源宋体 Heavy +
FOT-モード明朝B, rose text) and the 方正兰亭圆 + A-OTF 新丸ゴ DB set the user supplied (green outline).  Fonts are the
user's call (with furigana many fonts misbehave in one player but not another — PotPlayer vs Plex); the tool only
reports a font that is not installed.  Every template carries the full set of style names the tool expects
(styles_default.py): CN / JP, .Top, .Alt, OP / ED / IN songs, .Ruby, .Gloss, CN.Pinyin.

Song styles: PrimaryColour is the sung colour (and the colour of a song line without karaoke), SecondaryColour the
not-yet-sung one.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .languages import get_language, lane_label, normalize_language_code


@dataclass(frozen=True)
class FontSet:
    """Fonts, sizes and the vertical layout that goes with them (MarginV of the bottom / top / second-speaker lines;
    the defaults are the user's NANA / Utena releases)."""

    id: str
    name: str
    cn_hans: str
    cn_hant: str
    jp: str
    cn_size: float = 68
    jp_size: float = 50
    bold: bool = True
    cn_bold: Optional[bool] = None      # per language (default: bold)
    jp_bold: Optional[bool] = None
    spacing: float = 0.0                # \fsp of the dialogue lines
    jp_outline_ratio: float = 6 / 7     # JP outline width / CN outline width (3 / 3.5 in the releases)
    side: int = 10                      # MarginL / MarginR
    cn_mv: float = 65                   # bottom: CN above JP and its furigana
    jp_mv: float = 2
    cn_alt_mv: float = 190              # a second speaker, above the first
    jp_alt_mv: float = 130
    cn_top_mv: float = 5                # top of the screen (when signs occupy the bottom)
    jp_top_mv: float = 85
    ruby_ratio: float = 0.5             # furigana size / JP size
    pinyin_ratio: float = 0.38          # pinyin size / CN size
    pinyin_font: str = "Noto Sans CJK SC Black"
    song_cn_ratio: float = 1.18         # song lines are larger
    song_jp_ratio: float = 1.4
    song_cn_mv: Optional[float] = None  # None: CN MarginV + 20
    song_jp_mv: Optional[float] = None  # None: JP MarginV
    note: str = ""
    ruby_offset: float = 0.0           # px at 1080p; positive moves annotations farther above the base
    pinyin_offset: float = 0.0
    jp_outline_width: Optional[float] = None  # independent width when CN has no outline

    def __post_init__(self) -> None:
        for value in (self.ruby_offset, self.pinyin_offset):
            if not math.isfinite(value) or not -40 <= value <= 40:
                raise ValueError("注音上下微调必须在 −40 到 +40 像素之间")

    def cn_font(self, translation_lang: str) -> str:
        return self.cn_hant if is_traditional(translation_lang) else self.cn_hans


@dataclass(frozen=True)
class Palette:
    """Colours as RGB hex ("FFFFFF")."""

    id: str
    name: str
    text: str                               # CN fill
    outline: str                            # CN outline
    jp_text: Optional[str] = None           # JP fill (defaults to text)
    jp_outline: Optional[str] = None        # JP outline (defaults to outline)
    width: float = 3.5                      # CN outline width; JP (font set's ratio) and furigana (×4/7) follow
    outline_alpha: int = 0                  # CN outline transparency (少女革命: 0x14)
    shadow: str = "000000"
    shadow_alpha: int = 0                   # shadow (BackColour) transparency; drawn only when shadow_depth > 0
    shadow_depth: float = 0.0
    alt_inverted: bool = False              # .Alt lines swap fill and outline (a second speaker stands out)
    top_inverted: Optional[bool] = None     # .Top lines swap them too; None (older schemes, presets) = as .Alt
    song_text: Optional[str] = None         # sung (defaults to text)
    song_before: str = "B8B8B8"             # not yet sung
    song_outline: Optional[str] = None      # defaults to outline
    note: str = ""

    @property
    def jp_fill(self) -> str:
        return self.jp_text or self.text

    @property
    def jp_edge(self) -> str:
        return self.jp_outline or self.outline


FONT_SETS = [
    FontSet("nana_black", "NANA 成品的字体（Noto Sans CJK Black）", "Noto Sans CJK SC Black", "Noto Sans CJK TC Black",
            "Noto Sans CJK JP Black", 68, 50),
    FontSet("utena_serif", "少女革命成品的字体（思源宋体 Heavy + モード明朝B）", "思源宋体 Heavy", "思源宋體 Heavy",
            "FOT-ModeMinBLarge ProN H", 68, 50),
    # the user's style lines (方正兰亭圆 66 regular + A-OTF 新丸ゴ DB 54 bold, spacing 1, side 15, JP MarginV 14); the CN /
    # second-speaker / top margins were measured with libass so the furigana gets the releases' 18 px clearance
    FontSet("lanting_round", "兰亭圆 + 新丸ゴ（方正兰亭圆_GBK_中粗 + A-OTF 新丸ゴ DB）", "方正兰亭圆_GBK_中粗",
            "方正兰亭圆_GBK_中粗", "A-OTF Shin Maru Go Pro DB", 66, 54, cn_bold=False, jp_bold=True, spacing=1,
            jp_outline_ratio=3.4 / 3.5, side=15, cn_mv=84, jp_mv=14, cn_alt_mv=231, jp_alt_mv=161, cn_top_mv=15,
            jp_top_mv=95),
]

PALETTES = [
    Palette("nana_purple", "紫边白字（NANA）", "FFFFFF", "760876", width=3.5, song_before="C8A8C8",
            note="NANA 双语成品的配色"),
    Palette("utena", "玫红字白边（少女革命）", "A81F5F", "FFFFFF", jp_outline="F0F0F0", outline_alpha=0x14,
            alt_inverted=True, song_text="FFFFFF", song_before="000000", song_outline="A81F5F",
            note="少女革命成品的配色：对白玫红字白边，歌词白字玫红边（未唱到的字是黑色）"),
    Palette("green", "翠绿边白字", "FFFFFF", "259948", outline_alpha=0x14, shadow="0978C2", song_before="A9D9B8",
            note="你给的兰亭圆 / 新丸ゴ 样式里的配色"),
    Palette("classic", "经典白字黑边", "FFFFFF", "111111", width=3.0, shadow_alpha=0x70, shadow_depth=1.5,
            song_before="9A9A9A"),
    Palette("ocean", "海蓝（白字 + 深海蓝边）", "FFFFFF", "1D4E89", jp_text="EEF5FF", song_before="8FB3DC"),
    Palette("sakura", "樱粉（白字 + 玫瑰边）", "FFFFFF", "C2477A", jp_text="FFF0F5", song_before="E8A6C2"),
    Palette("forest", "森绿（白字 + 墨绿边）", "FFFFFF", "2E5E3E", jp_text="F0F8F1", song_before="9CC7A8"),
    Palette("warm", "暖阳（米白 + 栗色边）", "FFF6E0", "6B3E1F", jp_text="FFEDC8", song_before="C9A27A"),
    Palette("night_gold", "夜金（金字 + 深夜蓝边）", "FFE9A8", "1E1E2E", jp_text="FFF4D2", shadow_alpha=0x90,
            shadow_depth=1.0, song_before="9A9AA8"),
    Palette("two_tone", "双色（中文蓝边 · 日文灰边）", "FFFFFF", "1D4E89", jp_text="F4F4F4", jp_outline="3C3C3C",
            song_before="8FB3DC", note="中日两行用不同的边色，一眼分开两种语言"),
]

FORMAT = ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
          "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
          "MarginR, MarginV, Encoding")


def ass_colour(rgb: str, alpha: int = 0) -> str:
    rgb = rgb.strip().lstrip("#").upper().rjust(6, "0")
    return f"&H{alpha:02X}{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}"


def rgb_alpha(ass: str) -> tuple[str, int]:
    """&HAABBGGRR → ("RRGGBB", alpha)."""
    v = (ass or "").upper().replace("&H", "").replace("&", "").rjust(8, "0")[-8:]
    return v[6:8] + v[4:6] + v[2:4], int(v[0:2], 16)


def is_traditional(lang: str) -> bool:
    return lang.lower() in ("zh-hant", "zh-tw", "zh-hk", "zh-mo", "cht")


def build_styles(fonts: FontSet, pal: Palette, *, translation_lang: str = "zh-Hans") -> str:
    """The [V4+ Styles] section of a template (the project's PlayRes is 1920×1080)."""
    f = fonts
    cn = f.cn_font(translation_lang)
    jp = f.jp
    cs, js = f.cn_size, f.jp_size
    rs = round(js * f.ruby_ratio, 1)
    b = pal.width
    jb = round(f.jp_outline_width if not b and f.jp_outline_width is not None else b * f.jp_outline_ratio, 1)
    rb = max(1.5, round(b * 4 / 7, 1))
    cbold = f.bold if f.cn_bold is None else f.cn_bold
    jbold = f.bold if f.jp_bold is None else f.jp_bold
    lines = ["[V4+ Styles]", FORMAT]

    def add(name: str, font: str, size: float, fill: str, before: str, edge: str, bord: float, align: int,
            mv: float, *, bold: bool, side: Optional[int] = None, edge_alpha: int = 0, fsp: float = 0.0) -> None:
        side = f.side if side is None else side
        lines.append(f"Style: {name},{font},{round(size, 1):g},{ass_colour(fill)},{ass_colour(before)},"
                     f"{ass_colour(edge, edge_alpha)},{ass_colour(pal.shadow, pal.shadow_alpha)},{-1 if bold else 0},"
                     f"0,0,0,100,100,{fsp:g},0,1,{bord:g},{pal.shadow_depth:g},{align},{side},{side},{round(mv, 1):g},1")

    sp = f.spacing
    cf, ce, jf, je, ca = pal.text, pal.outline, pal.jp_fill, pal.jp_edge, pal.outline_alpha
    def swapped(on: bool) -> tuple[str, str, str, str, int]:
        # a second speaker / top lines: the CN colours swapped, for both languages
        return (ce, cf, ce, cf, 0) if on else (cf, ce, jf, je, ca)

    acf, ace, ajf, aje, aca = swapped(pal.alt_inverted)
    tcf, tce, tjf, tje, tca = swapped(pal.alt_inverted if pal.top_inverted is None else pal.top_inverted)
    add("CN", cn, cs, cf, "FF0000", ce, b, 2, f.cn_mv, bold=cbold, edge_alpha=ca, fsp=sp)
    add("JP", jp, js, jf, "FF0000", je, jb, 2, f.jp_mv, bold=jbold, fsp=sp)
    add("JP.Ruby", jp, rs, jf, "FF0000", je, rb, 2, 0, bold=jbold, side=0)
    add("CN.Top", cn, cs, tcf, "FF0000", tce, b, 8, f.cn_top_mv, bold=cbold, edge_alpha=tca, fsp=sp)
    add("JP.Top", jp, js, tjf, "FF0000", tje, jb, 8, f.jp_top_mv, bold=jbold, fsp=sp)
    add("JP.Top.Ruby", jp, rs, tjf, "FF0000", tje, rb, 2, 0, bold=jbold, side=0)
    add("CN.Alt", cn, cs, acf, "FF0000", ace, b, 2, f.cn_alt_mv, bold=cbold, edge_alpha=aca, fsp=sp)
    add("JP.Alt", jp, js, ajf, "FF0000", aje, jb, 2, f.jp_alt_mv, bold=jbold, fsp=sp)
    add("JP.Alt.Ruby", jp, rs, ajf, "FF0000", aje, rb, 2, 0, bold=jbold, side=0)
    sf = pal.song_text or pal.text
    se = pal.song_outline or pal.outline
    song_js = round(js * f.song_jp_ratio)
    for fam in ("OP", "ED", "IN"):
        add(f"{fam}.CN", cn, round(cs * f.song_cn_ratio), sf, pal.song_before, se, b, 2,
            f.cn_mv + 20 if f.song_cn_mv is None else f.song_cn_mv, bold=cbold)
        add(f"{fam}.JP", jp, song_js, sf, pal.song_before, se, jb, 2, f.jp_mv if f.song_jp_mv is None else f.song_jp_mv,
            bold=jbold)
        add(f"{fam}.JP.Ruby", jp, round(song_js * f.ruby_ratio), sf, pal.song_before, se, jb, 2, 0, bold=jbold, side=0)
    add("CN.Gloss", cn, round(cs * 0.44), cf, "FF0000", ce, rb, 2, 0, bold=cbold, side=0, edge_alpha=ca)
    add("JP.Gloss", jp, rs, jf, "FF0000", je, rb, 2, 0, bold=jbold, side=0)
    add("CN.Pinyin", f.pinyin_font, round(cs * f.pinyin_ratio), cf, "FF0000", ce, rb, 2, 0, bold=False, side=0,
        edge_alpha=ca)
    return "\n".join(lines) + "\n"


def font_set(fid: str) -> FontSet:
    return next((f for f in FONT_SETS if f.id == fid), FONT_SETS[0])


def palette(pid: str) -> Palette:
    return next((p for p in PALETTES if p.id == pid), PALETTES[0])


# ------------------------------------------------------------------ saving the choice / reading a template back
def scheme_dict(fonts: FontSet, pal: Palette) -> dict[str, Any]:
    """Everything needed to open the editor again exactly as it was (stored in subkit-project.json)."""
    return {"fonts": dataclasses.asdict(fonts), "palette": dataclasses.asdict(pal)}


def scheme_from_dict(d: dict[str, Any]) -> tuple[FontSet, Palette]:
    """The editor's values from a stored scheme; older schemes stored only the preset ids."""
    fd, pd = d.get("fonts"), d.get("palette")
    fonts = font_set(fd) if isinstance(fd, str) else None
    pal = palette(pd) if isinstance(pd, str) else None
    if isinstance(fd, dict):
        keys = {x.name for x in dataclasses.fields(FontSet)}
        fonts = FontSet(**{k: v for k, v in fd.items() if k in keys})
    if isinstance(pd, dict):
        keys = {x.name for x in dataclasses.fields(Palette)}
        pal = Palette(**{k: v for k, v in pd.items() if k in keys})
    return fonts or FONT_SETS[0], pal or PALETTES[0]


def import_dialogue_styles(fonts: FontSet, pal: Palette, doc, translation: Optional[str], original: Optional[str], *,
                           read_fonts: bool = True, read_colours: bool = True, read_margins: bool = False,
                           playres_y: float = 1080) -> tuple[FontSet, Palette]:
    """Overlay only selected dialogue properties on the current template."""
    styles = doc.style_map()
    cn, jp = styles.get(translation), styles.get(original)
    try:
        height = float(doc.info("PlayResY", "1080"))
    except (TypeError, ValueError):
        height = 1080
    scale = playres_y / height if math.isfinite(height) and height > 0 else playres_y / 1080
    fd, pd = {}, {}
    if read_fonts and (cn is not None or jp is not None):
        jp_width = (fonts.jp_outline_width if not pal.width and fonts.jp_outline_width is not None
                    else pal.width * fonts.jp_outline_ratio)
        width = pal.width
        if cn is not None:
            fd.update(cn_hans=cn.fontname, cn_hant=cn.fontname, cn_size=cn.fontsize * scale, cn_bold=cn.bold,
                      spacing=cn.spacing * scale)
            width = cn.outline * scale
            pd["width"] = width
        if jp is not None:
            fd.update(jp=jp.fontname, jp_size=jp.fontsize * scale, jp_bold=jp.bold)
            jp_width = jp.outline * scale
            if cn is None:
                fd["spacing"] = jp.spacing * scale
        fd.update(jp_outline_ratio=jp_width / width if width else fonts.jp_outline_ratio,
                  jp_outline_width=jp_width if not width else None)
    if read_colours:
        if cn is not None:
            pd["text"], _ = rgb_alpha(cn.get("PrimaryColour"))
            pd["outline"], pd["outline_alpha"] = rgb_alpha(cn.get("OutlineColour"))
            # JP's inherited colours must not change when only translation is selected.
            pd.update(jp_text=pal.jp_fill, jp_outline=pal.jp_edge)
        if jp is not None:
            pd["jp_text"], _ = rgb_alpha(jp.get("PrimaryColour"))
            pd["jp_outline"], _ = rgb_alpha(jp.get("OutlineColour"))
        shadow = cn if cn is not None else jp
        if shadow is not None:
            pd["shadow"], pd["shadow_alpha"] = rgb_alpha(shadow.get("BackColour"))
    if read_margins:
        if cn is not None:
            fd["cn_mv"] = cn.num("MarginV") * scale
        if jp is not None:
            fd["jp_mv"] = jp.num("MarginV") * scale
    if fd:
        fonts = dataclasses.replace(fonts, id="custom", name="自定义", **fd)
    if pd:
        pal = dataclasses.replace(pal, id="custom", name="自定义", **pd)
    return fonts, pal


def scheme_from_styles(doc) -> tuple[FontSet, Palette]:
    """Editor values read from a template's styles (a project made before the editor, or from an .ass file).  Styles
    that are missing keep the preset's value."""
    base_f, base_p = font_set("utena_serif"), palette("utena")
    st = doc.style
    cn, jp = st("CN"), st("JP")
    if cn is None or jp is None:
        return base_f, base_p

    def mv(name: str, default: float) -> float:
        s = st(name)
        return s.num("MarginV", default) if s is not None else default

    ruby, alt, top, op_cn, op_jp, py = st("JP.Ruby"), st("CN.Alt"), st("CN.Top"), st("OP.CN"), st("OP.JP"), st("CN.Pinyin")
    fonts = dataclasses.replace(
        base_f, id="custom", name="自定义", cn_hans=cn.fontname, cn_hant=cn.fontname, jp=jp.fontname,
        cn_size=cn.fontsize, jp_size=jp.fontsize, bold=cn.bold, cn_bold=cn.bold, jp_bold=jp.bold,
        spacing=cn.spacing, jp_outline_ratio=(jp.outline / cn.outline) if cn.outline else 6 / 7,
        side=int(cn.num("MarginL", 10)), cn_mv=mv("CN", 65), jp_mv=mv("JP", 2), cn_alt_mv=mv("CN.Alt", 190),
        jp_alt_mv=mv("JP.Alt", 130), cn_top_mv=mv("CN.Top", 5), jp_top_mv=mv("JP.Top", 85),
        ruby_ratio=(ruby.fontsize / jp.fontsize) if ruby is not None and jp.fontsize else 0.5,
        pinyin_ratio=(py.fontsize / cn.fontsize) if py is not None and cn.fontsize else 0.38,
        pinyin_font=py.fontname if py is not None else base_f.pinyin_font,
        song_cn_ratio=(op_cn.fontsize / cn.fontsize) if op_cn is not None and cn.fontsize else 1.18,
        song_jp_ratio=(op_jp.fontsize / jp.fontsize) if op_jp is not None and jp.fontsize else 1.4,
        song_cn_mv=op_cn.num("MarginV") if op_cn is not None else None,
        song_jp_mv=op_jp.num("MarginV") if op_jp is not None else None)
    text, _a = rgb_alpha(cn.get("PrimaryColour"))
    outline, oa = rgb_alpha(cn.get("OutlineColour"))
    jtext, _ = rgb_alpha(jp.get("PrimaryColour"))
    jout, _ = rgb_alpha(jp.get("OutlineColour"))
    shadow, sa = rgb_alpha(cn.get("BackColour"))
    def inverted_style(s) -> bool:
        if s is None:
            return False
        st_text, _ = rgb_alpha(s.get("PrimaryColour"))
        st_out, _ = rgb_alpha(s.get("OutlineColour"))
        return (st_text, st_out) == (outline, text) and st_text != text

    inverted = inverted_style(alt)
    top_inverted = inverted_style(top) if top is not None else None
    song_text = song_before = song_outline = None
    if op_cn is not None:
        song_text, _ = rgb_alpha(op_cn.get("PrimaryColour"))
        song_before, _ = rgb_alpha(op_cn.get("SecondaryColour"))
        song_outline, _ = rgb_alpha(op_cn.get("OutlineColour"))
    pal = Palette("custom", "自定义", text, outline, jp_text=None if jtext == text else jtext,
                  jp_outline=None if jout == outline else jout, width=cn.outline, outline_alpha=oa, shadow=shadow,
                  shadow_alpha=sa, shadow_depth=cn.shadow, alt_inverted=inverted, top_inverted=top_inverted,
                  song_text=None if song_text in (None, text) else song_text,
                  song_before=song_before or base_p.song_before,
                  song_outline=None if song_outline in (None, outline) else song_outline)
    return fonts, pal


# ------------------------------------------------------------------ preview (rendered by libass, like the release)
SCENES = {"dialogue": "对白", "alt": "两人同时说话", "top": "屏幕顶部", "song": "歌词（K 轴唱到一半）"}
# Neutral sentences from docs/TEMPLATE_PREVIEW.md: dialogue, second speaker, top, lyric.
PREVIEW_SENTENCES = {
    'ja': ('駅(えき)の前(まえ)で待(ま)っています。', 'わかった、気(き)をつけてね。', '三日後(みっかご)', '空(そら)に向(む)かって歌(うた)おう'),
    'zh-Hans': ('我在车站前面等你。', '知道了，路上小心。', '三天后', '向着天空歌唱吧'),
    'zh-Hant': ('我在車站前面等你。', '知道了，路上小心。', '三天後', '向著天空歌唱吧'),
    'yue': ('我喺車站前面等你。', '知道喇，路上小心。', '三日後', '向住天空唱歌啦'),
    'ko': ('역 앞에서 기다릴게요.', '알겠어, 조심해.', '사흘 후', '하늘을 향해 노래하자'),
    'en': ("I'll wait for you in front of the station.", 'Got it. Take care.', 'Three days later', "Let's sing to the sky"),
    'vi': ('Mình sẽ đợi bạn trước nhà ga.', 'Biết rồi, đi cẩn thận nhé.', 'Ba ngày sau', 'Hãy hát về phía bầu trời'),
    'fr': ("Je t'attends devant la gare.", "D'accord, fais attention.", 'Trois jours plus tard', 'Chantons vers le ciel'),
    'de': ('Ich warte vor dem Bahnhof auf dich.', 'Alles klar, pass auf dich auf.', 'Drei Tage später', 'Lass uns zum Himmel singen'),
    'es': ('Te espero frente a la estación.', 'Entendido, cuídate.', 'Tres días después', 'Cantemos hacia el cielo'),
    'pt': ('Vou te esperar em frente à estação.', 'Entendi, tome cuidado.', 'Três dias depois', 'Vamos cantar para o céu'),
    'it': ('Ti aspetto davanti alla stazione.', 'Va bene, stai attento.', 'Tre giorni dopo', 'Cantiamo verso il cielo'),
    'ru': ('Я жду тебя у вокзала.', 'Понял, будь осторожен.', 'Три дня спустя', 'Давай споём небу'),
}
_SCENE_LINES = {
    "dialogue": [("CN", 0), ("JP", 0)],
    "alt": [("CN", 0), ("JP", 0), ("CN.Alt", 1), ("JP.Alt", 1)],
    "top": [("CN.Top", 2), ("JP.Top", 2), ("CN", 0), ("JP", 0)],
    "song": [("OP.CN", 3), ("OP.JP", 3)],
}


def preview_sentence(lang: str, sample: int) -> tuple[str, list[tuple[int, int, str]]]:
    """Return plain text and fixed Japanese readings, indexed in visible characters."""
    code = normalize_language_code(lang)
    text = PREVIEW_SENTENCES.get(code, PREVIEW_SENTENCES["en"])[sample]
    if code != "ja":
        return text, []
    plain, pieces, cursor = "", [], 0
    for match in re.finditer(r"([\u4e00-\u9faf\u3005]+)\(([^()]+)\)", text):
        plain += text[cursor:match.start()]
        start = len(plain)
        plain += match[1]
        pieces.append((start, len(plain), match[2]))
        cursor = match.end()
    return plain + text[cursor:], pieces


SONG_AT = 1.4               # seconds: the song lines are half sung
GUIDE = {"CN": ("中文", "FFD34E"), "JP": ("日文", "6EE7B7"), "CN.Alt": ("第二人中文", "FFD34E"),
         "JP.Alt": ("第二人日文", "6EE7B7"), "CN.Top": ("顶部中文", "FFD34E"), "JP.Top": ("顶部日文", "6EE7B7"),
         "OP.CN": ("歌词中文", "FFD34E"), "OP.JP": ("歌词日文", "6EE7B7")}
_HEAD = "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nScaledBorderAndShadow: yes\nWrapStyle: 2\n\n"
_EVH = "\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"


def _preview_head(playres: tuple[int, int]) -> str:
    return _HEAD.replace("PlayResX: 1920", f"PlayResX: {playres[0]}").replace("PlayResY: 1080", f"PlayResY: {playres[1]}")


def preview_playres(path: Path, runner=None) -> tuple[int, int]:
    """Display aspect, including pixel aspect; use the episode's rounding rule."""
    import json
    from fractions import Fraction
    from .proc import run

    result = (runner.run if runner is not None else run)(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,sample_aspect_ratio", "-of", "json", str(path)],
        capture_output=True, check=True, timeout=15)
    geometry = json.loads(result.stdout)["streams"][0]
    try:
        sar = Fraction(str(geometry.get("sample_aspect_ratio") or "1:1").replace(":", "/"))
        if sar <= 0:
            sar = Fraction(1)
    except (ValueError, ZeroDivisionError):
        sar = Fraction(1)
    width = max(2, round(1080 * Fraction(int(geometry["width"]), int(geometry["height"])) * sar))
    return width + width % 2, 1080


@dataclass
class Preview:
    png: Path
    missing_fonts: list[str] = field(default_factory=list)          # not installed: libass used a fallback font
    missing_glyphs: dict[str, str] = field(default_factory=dict)    # font -> characters it lacks (drawn by another font)

    def warning(self) -> str:
        parts = []
        if self.missing_fonts:
            parts.append("没有安装：" + "、".join(self.missing_fonts) + "（预览里换成了别的字体）")
        for f, chars in self.missing_glyphs.items():
            if f not in self.missing_fonts:
                parts.append(f"{f} 缺字：{chars}（这些字由别的字体代替）")
        return "；".join(parts)


_FONTSELECT = re.compile(r"fontselect: \((.+?), \d+, \d+\) -> ([^,\s]+)")
_NO_GLYPH = re.compile(r"Glyph 0x([0-9A-Fa-f]+) not found, selecting one more font for \((.+?), \d+, \d+\)")
_NO_FALLBACK = re.compile(r"fontselect: failed to find any fallback with glyph 0x([0-9A-Fa-f]+) for font: \((.+?), \d+, \d+\)")


@dataclass
class FontLog:
    selected: dict[str, str] = field(default_factory=dict)    # first PostScript name per requested family
    gaps: dict[str, str] = field(default_factory=dict)
    undrawable: dict[str, str] = field(default_factory=dict)


def parse_font_log(log: str) -> FontLog:
    """Font requests and missing glyphs, in libass log order (including failed fallbacks)."""
    result = FontLog()
    for m in _FONTSELECT.finditer(log):
        result.selected.setdefault(m.group(1), m.group(2))
    for pattern, target in ((_NO_GLYPH, result.gaps), (_NO_FALLBACK, result.undrawable)):
        for m in pattern.finditer(log):
            ch, family = chr(int(m.group(1), 16)), m.group(2)
            if ch not in target.get(family, ""):
                target[family] = target.get(family, "") + ch
    return result


def font_report(log: str) -> tuple[list[str], dict[str, str]]:
    """Read libass's font selection from an ffmpeg ``-loglevel verbose`` log: which requested families fell back to
    the default font (not installed), and which glyphs a family lacked."""
    parsed = parse_font_log(log)
    missing = [name for name, ps in parsed.selected.items() if ps.lower().startswith("arial") and "arial" not in name.lower()]
    return missing, parsed.gaps


def _geometry_styles(styles: str) -> str:
    """The same styles with every colour replaced: the layout (and the measurement cache) does not depend on them."""
    out = []
    for line in styles.splitlines():
        if line.startswith("Style:"):
            parts = line.split(",")
            if len(parts) >= 7:
                parts[3:7] = ["&H00FFFFFF", "&H00FFFFFF", "&H00000000", "&H00000000"]
            line = ",".join(parts)
        out.append(line)
    return "\n".join(out) + "\n"


def _preview_styles(styles: str, scene: str, original_lang: str, translation_lang: str) -> str:
    """Include derived annotation styles in both measurement and the actual rendered ASS."""
    from .ass import AssDocument
    from .ruby.generate import derive_ruby_style

    doc = AssDocument.parse(_HEAD + styles + _EVH)
    extra = []
    for style, _sample in _SCENE_LINES[scene]:
        base = doc.style(style)
        if base is None:
            continue
        lang = translation_lang if "CN" in style.split(".") else original_lang
        system = get_language(lang).annotation_system
        name = f"{style}.Ruby" if system == "ruby" else f"{style}.Pinyin"
        if system == "ruby" and doc.style(name) is None:
            jp, ruby = doc.style("JP"), doc.style("JP.Ruby")
            ratio = ruby.fontsize / jp.fontsize if jp and ruby and jp.fontsize else .5
            extra.append(derive_ruby_style(base, name, ratio).to_line(doc.style_format))
        elif system == "pinyin" and doc.style(name) is None and doc.style("CN.Pinyin") is not None:
            extra.append(doc.style("CN.Pinyin").copy(name).to_line(doc.style_format))
    return styles + "\n".join(extra) + "\n"


def scene_events(styles: str, scene: str, *, traditional: bool = False, pinyin: bool = False,
                 original_lang: str = "ja", translation_lang: Optional[str] = None,
                 measure_cache: Optional[Path] = None, runner=None, ruby_offset: float = 0.0,
                 pinyin_offset: float = 0.0, playres: tuple[int, int] = (1920, 1080)) -> list[str]:
    """Dialogue lines of a preview scene, with the furigana (and pinyin) laid out by the ruby generator."""
    from .ass import AssDocument, Event
    from .ruby.generate import RubyGenerator, RubyJob, RubyPiece
    from .ruby.measure import Measurer

    translation_lang = translation_lang or ("zh-Hant" if traditional else "zh-Hans")
    styles = _preview_styles(styles, scene, original_lang, translation_lang)
    doc = AssDocument.parse(_preview_head(playres) + _geometry_styles(styles) + _EVH)
    t0, t1 = 0, 300
    jobs = []
    events: list[Event] = []
    for style, sample in _SCENE_LINES[scene]:
        if doc.style(style) is None:
            continue
        lane = "translation" if "CN" in style.split(".") else "original"
        lang = translation_lang if lane == "translation" else original_lang
        txt, pieces = preview_sentence(lang, sample)
        if scene == "song":
            # Equal beats keep the neutral lyric half sung at SONG_AT, in every language.
            txt = "".join(f"{{\\kf{round(280 * (i + 1) / len(txt)) - round(280 * i / len(txt))}}}{ch}"
                          for i, ch in enumerate(txt))
        # the export puts translation lines on layer 1 and the original on layer 0 (libass only pushes overlapping
        # lines apart within one layer — the preview must overlap exactly where the release would)
        layer = 1 if lane == "translation" else 0
        ev = Event(kind="Dialogue", layer=layer, start=t0, end=t1, style=style, text=txt)
        events.append(ev)
        if pieces:
            rname = f"{style}.Ruby"
            jobs.append(RubyJob(ev, rname, [RubyPiece(s, e, r, word=(s, e)) for s, e, r in pieces], offset=ruby_offset, prefer_center=True))
        if pinyin and get_language(lang).annotation_system == "pinyin":
            from .ass.text import VisibleText
            from .pinyin import line_pinyin

            py = [RubyPiece(i, i + 1, syl) for i, syl, _alts in line_pinyin(VisibleText(txt).text) if syl]
            pname = f"{style}.Pinyin" if doc.style("CN.Pinyin") is not None else None
            if py and pname:
                jobs.append(RubyJob(ev, pname, py, offset=pinyin_offset))
    doc.events.extend(events)
    with Measurer(doc, cache_path=measure_cache, runner=runner) as measurer:
        gen = RubyGenerator(doc, measurer)
        gen.run(jobs)
    lines = [e.to_line(doc.event_format) for e in events]
    for j in jobs:
        lines.extend(e.to_line(doc.event_format) for e in j.out)
    return lines


def guide_events(styles: str, scene: str, *, original_lang: str = "ja", translation_lang: str = "zh-Hans",
                 playres: tuple[int, int] = (1920, 1080)) -> list[str]:
    """Dashed lines at every line's MarginV (where its bottom — or top, for top lines — is anchored), labelled."""
    from .ass import AssDocument

    doc = AssDocument.parse(_preview_head(playres) + styles + _EVH)
    out = []
    seen = set()
    for style, _sample in _SCENE_LINES[scene]:
        st = doc.style(style)
        if st is None or style in seen:
            continue
        seen.add(style)
        label, rgb = GUIDE.get(style, (style, "FFFFFF"))
        languages = {"original": original_lang, "translation": translation_lang}
        label = label.replace("中文", lane_label("translation", languages)).replace("日文", lane_label("original", languages))
        mv = st.num("MarginV", 0)
        top = st.alignment in (7, 8, 9)
        y = mv if top else playres[1] - mv
        dashes = " ".join(f"m {x} {y - 1:.1f} l {x + 14} {y - 1:.1f} l {x + 14} {y + 1:.1f} l {x} {y + 1:.1f}"
                          for x in range(0, playres[0], 24))
        col = ass_colour(rgb)
        out.append(f"Dialogue: 9,0:00:00.00,0:00:03.00,{style},,0,0,0,,{{\\an7\\pos(0,0)\\bord0\\shad0\\1c{col}\\1a&H50&\\p1}}"
                   f"{dashes}{{\\p0}}")
        ly = y + 4 if top else y - 4
        out.append(f"Dialogue: 9,0:00:00.00,0:00:03.00,{style},,0,0,0,,{{\\an{'7' if top else '1'}\\pos(14,{ly:.1f})"
                   f"\\fnMicrosoft YaHei\\fs30\\b0\\fsp0\\fscx100\\fscy100\\bord2\\shad0\\1c{col}\\3c&H000000&\\1a&H10&}}"
                   f"{label} · {'距顶边' if top else '距底边'} {mv:g}")
    return out


BACKGROUNDS = {"dark_gradient": "gradients=s=1920x1080:r=25:c0=0x18202c:c1=0x48566a:x0=0:y0=0:x1=1920:y1=1080:d=3:speed=0.00001",
               "gradient": "gradients=s=1920x1080:r=25:c0=0x2a3442:c1=0x9aa6b6:x0=0:y0=0:x1=1920:y1=1080:d=3:speed=0.00001",
               "black": "color=c=black:s=1920x1080:r=25:d=3", "white": "color=c=white:s=1920x1080:r=25:d=3"}


def render_scene(styles: str, out_png: Path, *, scene: str = "dialogue", width: int = 960, traditional: bool = False,
                 pinyin: bool = False, guides: bool = True, background: Any = "dark_gradient", region: str = "full",
                 original_lang: str = "ja", translation_lang: Optional[str] = None,
                 measure_cache: Optional[Path] = None, runner=None, ruby_offset: float = 0.0,
                 pinyin_offset: float = 0.0, playres: Optional[tuple[int, int]] = None,
                 event_lines: Optional[list[str]] = None) -> Preview:
    """A frame of a scene rendered by libass through ffmpeg (no console window), like the release will look, over a
    gradient / black / white or a picture (a path).  ``region`` "bottom" / "top" crops to the subtitle area (the
    bottom / top 330 px, the middle half of the width: twice as large).  Also reports fonts that are not installed and
    missing glyphs."""
    import tempfile

    from .proc import run

    run = runner.run if runner is not None else run

    image_background = isinstance(background, (str, Path)) and str(background) not in BACKGROUNDS and Path(background).exists()
    playres = playres or (preview_playres(Path(background), runner) if image_background else (1920, 1080))
    if playres != (1920, 1080):
        from .ass import AssDocument
        from .ass.resample import resample
        doc = AssDocument.parse(_HEAD + styles + _EVH)
        resample(doc, *playres)
        styles = "[V4+ Styles]\n" + FORMAT + "\n" + "\n".join(st.to_line(doc.style_format) for st in doc.styles) + "\n"
    if event_lines is not None:
        # A project template may include a Script Info/Events section. Keep its styles only.
        from .ass import AssDocument
        doc = AssDocument.parse(styles)
        styles = "[V4+ Styles]\nFormat: " + ", ".join(doc.style_format) + "\n"
        styles += "\n".join(st.to_line(doc.style_format) for st in doc.styles) + "\n"
        lines = list(event_lines)
    else:
        styles = _preview_styles(styles, scene, original_lang,
                                 translation_lang or ("zh-Hant" if traditional else "zh-Hans"))
        lines = scene_events(styles, scene, traditional=traditional, pinyin=pinyin, measure_cache=measure_cache, runner=runner,
                             ruby_offset=ruby_offset, pinyin_offset=pinyin_offset,
                             original_lang=original_lang, translation_lang=translation_lang, playres=playres)
    if guides and event_lines is None:
        lines += guide_events(styles, scene, original_lang=original_lang,
                              translation_lang=translation_lang or ("zh-Hant" if traditional else "zh-Hans"), playres=playres)
    text = _preview_head(playres) + styles + _EVH + "\n".join(lines) + "\n"
    t = SONG_AT if scene == "song" else 1.0
    crop = {"bottom": ",crop=960:330:480:750", "top": ",crop=960:330:480:0"}.get(region, "")     # 2× the middle
    px, py = playres
    if playres != (1920, 1080):
        crop = {"bottom": f",crop={px // 2}:330:{px // 4}:{py - 330}",
                "top": f",crop={px // 2}:330:{px // 4}:0"}.get(region, "")
    fit = "" if event_lines is not None or crop or playres == (1920, 1080) else ",scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
    with tempfile.TemporaryDirectory(prefix="subkit_tpl_") as tmp:
        Path(tmp, "t.ass").write_text(text, encoding="utf-8-sig")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        if image_background:
            src = ["-loop", "1", "-t", "3", "-i", str(background)]
            vf = (f"scale={px}:{py}:force_original_aspect_ratio=increase,crop={px}:{py},setsar=1,fps=25,ass=t.ass{crop}{fit},"
                  f"scale={width}:-2")
        else:
            src = ["-f", "lavfi", "-i", BACKGROUNDS.get(str(background), BACKGROUNDS["dark_gradient"])]
            vf = f"scale={px}:{py},ass=t.ass{crop}{fit},scale={width}:-2" if playres != (1920, 1080) else f"ass=t.ass{crop},scale={width}:-2"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "verbose", *src, "-ss", f"{t:.2f}", "-vf", vf,
               "-frames:v", "1", str(out_png)]
        r = run(cmd, cwd=tmp, capture_output=True)
        log = r.stderr.decode("utf-8", "replace")
        if r.returncode != 0 or not out_png.exists():
            errors = [ln for ln in log.splitlines() if "rror" in ln]
            raise RuntimeError("预览生成失败：" + (" / ".join(errors) or log)[-300:])
    missing, gaps = font_report(log)
    # the guide labels use Microsoft YaHei; the sample's own characters are what matters
    return Preview(out_png, missing if event_lines is not None else [m for m in missing if m != "Microsoft YaHei"], gaps)


def render_preview(styles: str, out_png: Path, *, width: int = 960, traditional: bool = False) -> Preview:
    """The dialogue scene without guides (kept for scripts)."""
    return render_scene(styles, out_png, scene="dialogue", width=width, traditional=traditional, guides=False)


# ------------------------------------------------------------------ measured layout: the releases' clearances
TARGET = {"cn_ruby": 18, "alt_gap": 18, "top_edge": 18, "top_ruby": 17}


def _ink_rows(styles: str, groups: list[list[str]], runner=None) -> list[tuple[int, int]]:
    """Ink extent (top row, bottom row) of each group of event lines, all in one ffmpeg run (one frame per group)."""
    import numpy as np

    from .proc import run

    run = runner.run if runner is not None else run

    lines = []
    for k, g in enumerate(groups):
        for ln in g:
            parts = ln.split(",", 9)
            parts[1] = f"0:00:{k:02d}.00"
            parts[2] = f"0:00:{k:02d}.99"
            lines.append(",".join(parts))
    import tempfile

    with tempfile.TemporaryDirectory(prefix="subkit_fit_") as tmp:
        Path(tmp, "f.ass").write_text(_HEAD + _geometry_styles(styles) + _EVH + "\n".join(lines) + "\n",
                                      encoding="utf-8-sig")
        n = len(groups)
        r = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 f"color=c=black:s=1920x1080:r=1:d={n}", "-vf", "ass=f.ass,format=gray", "-frames:v", str(n),
                 "-f", "rawvideo", "-pix_fmt", "gray", "-"], cwd=tmp, capture_output=True)
    data = np.frombuffer(r.stdout, dtype=np.uint8)
    frames = data[: n * 1920 * 1080].reshape(-1, 1080, 1920)
    out = []
    for fr in frames:
        ys = np.nonzero((fr > 40).any(axis=1))[0]
        out.append((int(ys.min()), int(ys.max())) if ys.size else (-1, -1))
    return out


def clearances(fonts: FontSet, pal: Palette, *, translation_lang: str = "zh-Hant", pinyin: bool = False,
               measure_cache: Optional[Path] = None, runner=None) -> dict[str, int]:
    """Measured gaps (px at 1080p): CN above the furigana, second speaker above the main lines, top line from the
    edge and above the top furigana.  With ``pinyin`` the Chinese lines count with the pinyin above them."""
    styles = build_styles(fonts, pal, translation_lang=translation_lang)
    trad = is_traditional(translation_lang)
    by_style: dict[str, list[str]] = {}
    for scene in ("alt", "top"):
        for ln in scene_events(styles, scene, traditional=trad, measure_cache=measure_cache, runner=runner,
                               ruby_offset=fonts.ruby_offset, pinyin_offset=fonts.pinyin_offset):
            style = ln.split(",", 4)[3]
            by_style.setdefault(style, [])
            if ln not in by_style[style]:
                by_style[style].append(ln)
    names = ["CN", "JP.Ruby", "JP.Alt", "CN.Alt", "JP.Alt.Ruby", "CN.Top", "JP.Top.Ruby"]
    groups = [by_style.get(n, []) for n in names]
    if pinyin:                       # how far the pinyin reaches above a Chinese line (the same for every CN line)
        groups.append([ln for ln in scene_events(styles, "dialogue", traditional=trad, pinyin=True,
                                                 measure_cache=measure_cache, runner=runner,
                                                 ruby_offset=fonts.ruby_offset, pinyin_offset=fonts.pinyin_offset)
                       if ln.split(",", 4)[3] == "CN.Pinyin"])
    rows = _ink_rows(styles, groups, runner=runner)
    r = dict(zip(names, rows))
    ext = max(0, r["CN"][0] - rows[-1][0]) if pinyin and rows[-1][0] >= 0 else 0
    return {"cn_ruby": r["JP.Ruby"][0] - r["CN"][1],
            "alt_gap": r["CN"][0] - ext - r["JP.Alt"][1],
            "alt_cn_ruby": r["JP.Alt.Ruby"][0] - r["CN.Alt"][1],
            "top_edge": r["CN.Top"][0] - ext,
            "top_ruby": r["JP.Top.Ruby"][0] - r["CN.Top"][1],
            "pinyin": ext}


def fit_layout(fonts: FontSet, pal: Palette, *, translation_lang: str = "zh-Hant", target: Optional[dict] = None,
               pinyin: bool = False, measure_cache: Optional[Path] = None, runner=None) -> tuple[FontSet, dict[str, int]]:
    """Move the CN line, the second speaker and the top lines so the furigana gets the releases' clearances (the JP
    bottom line stays where it is).  Margins move the ink one-for-one, so each needs one measurement and a check."""
    target = dict(TARGET, **(target or {}))
    f = fonts
    for _ in range(3):
        c = clearances(f, pal, translation_lang=translation_lang, pinyin=pinyin, measure_cache=measure_cache, runner=runner)
        d_cn = target["cn_ruby"] - c["cn_ruby"]                 # CN up (MarginV grows) by what is missing
        d_alt = target["alt_gap"] - c["alt_gap"] + d_cn         # the alt block sits above the moved CN line
        d_alt_inner = target["cn_ruby"] - c["alt_cn_ruby"]
        d_top = c["top_edge"] - target["top_edge"]               # top line down (MarginV grows) by the excess… or up
        d_top_ruby = target["top_ruby"] - c["top_ruby"]
        if max(abs(d_cn), abs(d_alt), abs(d_alt_inner), abs(d_top), abs(d_top_ruby)) <= 1:
            return f, c
        f = dataclasses.replace(
            f, cn_mv=f.cn_mv + d_cn, jp_alt_mv=f.jp_alt_mv + d_alt, cn_alt_mv=f.cn_alt_mv + d_alt + d_alt_inner,
            cn_top_mv=f.cn_top_mv - d_top, jp_top_mv=f.jp_top_mv - d_top + d_top_ruby)
    return f, clearances(f, pal, translation_lang=translation_lang, pinyin=pinyin, measure_cache=measure_cache, runner=runner)
