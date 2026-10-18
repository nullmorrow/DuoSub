"""Render assets/duosub.svg to assets/duosub.ico (16–256 px) and assets/duosub.png (256 px).

Run from the repository root:  python tools/make_icon.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image
from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)


def render(renderer: QSvgRenderer, size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGBA")


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841 - needed by QPainter
    renderer = QSvgRenderer(str(ROOT / "assets" / "duosub.svg"))
    images = {size: render(renderer, size) for size in SIZES}
    images[256].save(ROOT / "assets" / "duosub.png")
    images[256].save(ROOT / "assets" / "duosub.ico", sizes=[(s, s) for s in SIZES],
                     append_images=[images[s] for s in SIZES if s != 256])
    print("wrote assets/duosub.ico and assets/duosub.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
