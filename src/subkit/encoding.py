"""Text file decoding with honest detection.

The legacy tools tried ``gb18030`` before ``cp932``.  GB18030 accepts almost any
byte sequence, so Shift-JIS files silently turned into mojibake.  Here every
candidate encoding that decodes strictly is *scored* by how plausible the
result is, and the best one wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .textutil import is_halfwidth_katakana, is_hiragana, is_katakana


@dataclass
class DecodedText:
    text: str
    encoding: str  # python codec name used for decoding (without BOM handling)
    bom: bytes
    newline: str


def _newline_of(text: str) -> str:
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf >= lf and crlf > 0 else "\n"


def _plausibility(text: str) -> float:
    if not text:
        return 0.0
    kana = halfkana = pua = cjk = ascii_ = 0
    for ch in text:
        code = ord(ch)
        if code < 0x80:
            ascii_ += 1
        elif is_hiragana(ch) or is_katakana(ch):
            kana += 1
        elif is_halfwidth_katakana(ch):
            halfkana += 1
        elif 0xE000 <= code <= 0xF8FF or ch == "�":
            pua += 1
        elif 0x4E00 <= code <= 0x9FFF:
            cjk += 1
    n = max(1, len(text))
    # Real Japanese text is full of kana; real Chinese text is CJK-heavy with
    # almost no kana.  Half-width katakana and private-use code points are the
    # classic signature of a wrong decode (GBK read as cp932 and vice versa).
    return (2.0 * kana + 1.0 * cjk + 0.2 * ascii_ - 3.0 * halfkana - 8.0 * pua) / n


def decode_bytes(data: bytes, prefer: str = "auto") -> DecodedText:
    if data.startswith(b"\xef\xbb\xbf"):
        text = data[3:].decode("utf-8")
        return DecodedText(text, "utf-8", b"\xef\xbb\xbf", _newline_of(text))
    if data.startswith(b"\xff\xfe"):
        text = data[2:].decode("utf-16-le")
        return DecodedText(text, "utf-16-le", b"\xff\xfe", _newline_of(text))
    if data.startswith(b"\xfe\xff"):
        text = data[2:].decode("utf-16-be")
        return DecodedText(text, "utf-16-be", b"\xfe\xff", _newline_of(text))
    if prefer != "auto":
        text = data.decode(prefer)
        return DecodedText(text, prefer, b"", _newline_of(text))
    try:
        text = data.decode("utf-8")
        return DecodedText(text, "utf-8", b"", _newline_of(text))
    except UnicodeDecodeError:
        pass
    best: tuple[float, str, str] | None = None
    for enc in ("cp932", "gb18030", "big5", "euc-jp", "utf-16"):
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        score = _plausibility(text)
        if best is None or score > best[0]:
            best = (score, enc, text)
    if best is None:
        text = data.decode("utf-8", errors="replace")
        return DecodedText(text, "utf-8", b"", _newline_of(text))
    return DecodedText(best[2], best[1], b"", _newline_of(best[2]))


def read_text(path: Path | str, prefer: str = "auto") -> DecodedText:
    return decode_bytes(Path(path).read_bytes(), prefer)


def write_text(path: Path | str, text: str, *, newline: str = "\r\n", bom: bool = True, encoding: str = "utf-8") -> None:
    """Write with explicit newlines in binary mode (no CRCRLF surprises on Windows)."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline != "\n":
        normalized = normalized.replace("\n", newline)
    data = normalized.encode(encoding)
    if bom and encoding.lower().replace("-", "") in ("utf8",):
        data = b"\xef\xbb\xbf" + data
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-subkit")
    tmp.write_bytes(data)
    tmp.replace(path)
