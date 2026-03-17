"""Generate a Go board grid-line texture as a PIL Image."""

from __future__ import annotations

from PIL import Image, ImageDraw


def generate_board_texture(
    board_size: int,
    board_spacing: float,
    board_half: float,
    tex_res: int = 512,
) -> Image.Image:
    """Return a *tex_res x tex_res* PIL image with grid lines on a white background.

    The white background will be tinted to the board colour by the MuJoCo
    material ``rgba``, so the lines are drawn in a dark colour that, when
    multiplied by the material tint ``(0.74, 0.62, 0.46)``, approximates the
    previous line colour ``(0.20, 0.16, 0.10)``.
    """
    img = Image.new("RGB", (tex_res, tex_res), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    line_color = (69, 66, 55)  # ≈ (0.20, 0.16, 0.10) / (0.74, 0.62, 0.46) * 255
    line_width = round(2 * 0.0012 / (2 * board_half) * tex_res)

    # Fraction of half-extent from edge to first grid line
    half_grid = 0.5 * (board_size - 1) * board_spacing
    margin_frac = (board_half - half_grid) / board_half  # e.g. 0.02 / board_half
    margin_px = margin_frac * 0.5 * tex_res

    # Pixel positions for each grid line
    usable = tex_res - 2 * margin_px
    for i in range(board_size):
        frac = i / (board_size - 1) if board_size > 1 else 0.5
        px = margin_px + frac * usable

        # Vertical line (column)
        draw.line([(px, margin_px), (px, tex_res - margin_px)],
                  fill=line_color, width=line_width)
        # Horizontal line (row)
        draw.line([(margin_px, px), (tex_res - margin_px, px)],
                  fill=line_color, width=line_width)

    return img
