"""Typographic rendering (VLM4).

render_text_to_image draws a harmful instruction as word-wrapped black text on
a white square canvas — the FigStep / typographic-attack construction. Same
AdvBench goal strings from §3B, delivered visually instead of as text.

Pillow-only and fully CPU-testable; the import is inside the function so the
module loads even where Pillow is absent.
"""
from __future__ import annotations


def _load_font(font_size: int):
    """Best-effort truetype font, falling back to Pillow's bitmap default.

    The default font ignores font_size but keeps rendering deterministic and
    dependency-free on a bare CPU box.
    """
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_lines(draw, text: str, font, max_width: int) -> list[str]:
    """Greedy word-wrap so no line exceeds max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textlength(candidate, font=font)
        if width <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_text_to_image(text: str, cfg):
    """Render text as word-wrapped black text on a white canvas.

    Args:
        text: the instruction string to render.
        cfg: a VLMConfig (uses image_size, image_font_size).

    Returns:
        A PIL.Image.Image of size (image_size, image_size), mode "RGB".
    """
    from PIL import Image, ImageDraw

    size = cfg.image_size
    margin = max(8, size // 24)
    font = _load_font(cfg.image_font_size)

    img = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(img)

    max_width = size - 2 * margin
    lines = _wrap_lines(draw, text, font, max_width)

    # Line height from font metrics, with a small leading.
    ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (cfg.image_font_size, 4)
    line_height = ascent + descent + max(2, cfg.image_font_size // 6)

    y = margin
    for line in lines:
        if y + line_height > size - margin:
            break  # ran out of vertical space; drop overflow deterministically
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height

    return img
