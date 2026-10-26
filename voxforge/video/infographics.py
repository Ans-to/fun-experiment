"""Broadcast Infographic, Architectural Blueprint, and Data-Telemetry Plate Generator.

Generates crisp 1920x1080 widescreen documentary plates, chapter visuals, and
evidence comparison cards using bundled SIL-OFL Space Grotesk & Inter typography.
"""

from __future__ import annotations

import hashlib
import math
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from voxforge.config import FONTS_DIR, VIDEO_SPEC
from voxforge.models import KeyMetric, SubTopicDossier


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load bundled broadcast font with clean fallback."""
    candidates = [
        FONTS_DIR / name,
        FONTS_DIR / "Inter-SemiBold.ttf",
        FONTS_DIR / "Inter-Regular.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def apply_cinema_grade(img: Image.Image, target_w: int = 1920, target_h: int = 1080) -> Image.Image:
    """Smart center-crop to 16:9, apply broadcast contrast/color grade, and subtle edge vignette."""
    im = img.convert("RGB")
    src_w, src_h = im.size
    target_ratio = target_w / target_h
    src_ratio = src_w / max(src_h, 1)

    if src_ratio > target_ratio:
        new_w = int(round(src_h * target_ratio))
        left = (src_w - new_w) // 2
        im = im.crop((left, 0, left + new_w, src_h))
    elif src_ratio < target_ratio:
        new_h = int(round(src_w / target_ratio))
        # Bias slightly above center to preserve monument peaks / heads
        top = max(0, int((src_h - new_h) * 0.38))
        im = im.crop((0, top, src_w, top + new_h))

    if im.size != (target_w, target_h):
        im = im.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Auto-exposure guard if image is overly dark or washed out
    arr = np.asarray(im, dtype=np.float32)
    mean_lum = float(arr.mean())
    if mean_lum < 68.0:
        im = ImageEnhance.Brightness(im).enhance(min(1.45, 88.0 / max(mean_lum, 18.0)))
    elif mean_lum > 198.0:
        im = ImageEnhance.Brightness(im).enhance(0.86)

    im = ImageEnhance.Contrast(im).enhance(1.07)
    im = ImageEnhance.Color(im).enhance(1.05)

    # Broadcast 12% peripheral vignette so lower-thirds & badges pop cleanly
    arr = np.asarray(im, dtype=np.float32)
    yy, xx = np.mgrid[0:target_h, 0:target_w]
    nx = (xx - target_w * 0.5) / (target_w * 0.5)
    ny = (yy - target_h * 0.5) / (target_h * 0.5)
    rad2 = nx * nx + ny * ny
    vignette = np.clip(1.0 - 0.22 * (rad2**1.15), 0.62, 1.0)[..., None]
    graded = np.clip(arr * vignette, 0, 255).astype(np.uint8)
    return Image.fromarray(graded, mode="RGB")


def render_procedural_documentary_plate(
    subtopic: SubTopicDossier,
    output_path: Path,
    variant: int = 0,
) -> Path:
    """Render a 1920x1080 architectural blueprint / geometric documentary visual plate."""
    w, h = VIDEO_SPEC.width, VIDEO_SPEC.height
    seed = int(hashlib.sha256(f"{subtopic.title}:{variant}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    # Deep cinema gradient backdrop (#080D1A -> #132038 -> warm amber/cyan glow)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = xx / w
    ny = yy / h

    palettes = [
        ((10, 16, 30), (24, 42, 72), (229, 169, 60)),   # Midnight & Cinema Gold
        ((12, 18, 28), (18, 52, 68), (56, 189, 248)),   # Deep Slate & Teal
        ((18, 14, 26), (54, 32, 48), (245, 158, 11)),   # Obsidian & Warm Sandstone
    ]
    c_top, c_bot, c_acc = palettes[(subtopic.index + variant) % len(palettes)]

    bg = np.zeros((h, w, 3), dtype=np.float32)
    for ch in range(3):
        bg[..., ch] = c_top[ch] * (1.0 - ny) + c_bot[ch] * ny

    # Radial atmospheric key light glow
    cx = 0.68 if variant % 2 == 0 else 0.34
    cy = 0.42
    dist2 = ((nx - cx) * 1.35) ** 2 + ((ny - cy) * 1.0) ** 2
    glow = np.exp(-dist2 * 3.2)
    for ch in range(3):
        bg[..., ch] += c_acc[ch] * 0.24 * glow

    # Subtle architectural film grain (avoids H.264 banding on dark gradients!)
    grain = rng.normal(0.0, 2.2, size=(h, w, 1)).astype(np.float32)
    bg = np.clip(bg + grain, 0, 255).astype(np.uint8)
    im = Image.fromarray(bg, mode="RGB").convert("RGBA")

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Fine architectural blueprint grid + celestial/surveyor rings
    grid_col = (255, 255, 255, 16)
    for gx in range(120, w, 120):
        draw.line([(gx, 0), (gx, h)], fill=grid_col, width=1)
    for gy in range(120, h, 120):
        draw.line([(0, gy), (w, gy)], fill=grid_col, width=1)

    ring_cx, ring_cy = int(w * cx), int(h * 0.48)
    for r in (160, 260, 370, 490):
        draw.ellipse(
            [ring_cx - r, ring_cy - r, ring_cx + r, ring_cy + r],
            outline=(c_acc[0], c_acc[1], c_acc[2], 42),
            width=2 if r == 260 else 1,
        )
    # Crosshair ticks
    for angle_deg in range(0, 360, 15):
        ang = math.radians(angle_deg)
        r1, r2 = 358, 382
        draw.line(
            [
                (ring_cx + math.cos(ang) * r1, ring_cy + math.sin(ang) * r1),
                (ring_cx + math.cos(ang) * r2, ring_cy + math.sin(ang) * r2),
            ],
            fill=(c_acc[0], c_acc[1], c_acc[2], 70),
            width=2,
        )

    # Geometric monument silhouette motif inside the surveyor ring
    motif_col = (c_acc[0], c_acc[1], c_acc[2], 65)
    draw.polygon(
        [
            (ring_cx, ring_cy - 195),
            (ring_cx - 230, ring_cy + 145),
            (ring_cx + 230, ring_cy + 145),
        ],
        outline=(c_acc[0], c_acc[1], c_acc[2], 145),
        fill=(c_acc[0], c_acc[1], c_acc[2], 24),
    )
    draw.line([(ring_cx, ring_cy - 195), (ring_cx + 45, ring_cy + 145)], fill=motif_col, width=2)

    # Left-side editorial dossier typography block
    f_kicker = load_font("SpaceGrotesk-Bold.ttf", 22)
    f_title = load_font("SpaceGrotesk-Bold.ttf", 52)
    f_body = load_font("Inter-Regular.ttf", 25)
    f_metric_val = load_font("SpaceGrotesk-Bold.ttf", 34)
    f_metric_lbl = load_font("Inter-Medium.ttf", 18)

    panel_x, panel_y = 110, 150
    draw.rounded_rectangle(
        [panel_x - 28, panel_y - 24, panel_x + 930, panel_y + 700],
        radius=20,
        fill=(9, 14, 26, 196),
        outline=(255, 255, 255, 36),
        width=2,
    )

    kicker = f"ARCHIVAL DOSSIER  ·  {subtopic.era_or_date.upper()}  ·  {subtopic.location.upper()}"
    draw.text((panel_x, panel_y), kicker[:78], font=f_kicker, fill=(c_acc[0], c_acc[1], c_acc[2], 245))

    wrapped_title = textwrap.wrap(subtopic.title, width=30)[:3]
    ty = panel_y + 46
    for line in wrapped_title:
        draw.text((panel_x, ty), line, font=f_title, fill=(248, 250, 252, 255))
        ty += 62

    draw.line([(panel_x, ty + 14), (panel_x + 220, ty + 14)], fill=(c_acc[0], c_acc[1], c_acc[2], 220), width=4)

    # Select different research lens per variant
    fact_pool = (
        subtopic.engineering_and_mechanics
        if variant % 2 == 0
        else subtopic.legacy_and_modern_science + subtopic.origin_and_context
    )
    highlight_fact = fact_pool[0] if fact_pool else subtopic.summary
    ty += 40
    for line in textwrap.wrap(highlight_fact, width=56)[:4]:
        draw.text((panel_x, ty), line, font=f_body, fill=(203, 213, 225, 240))
        ty += 37

    # Metric pills at bottom of dossier card
    if subtopic.metrics:
        my = panel_y + 490
        for idx, m in enumerate(subtopic.metrics[:4]):
            mx = panel_x + (idx % 2) * 440
            row_y = my + (idx // 2) * 92
            draw.rounded_rectangle(
                [mx, row_y, mx + 415, row_y + 76],
                radius=12,
                fill=(20, 30, 52, 225),
                outline=(c_acc[0], c_acc[1], c_acc[2], 95),
                width=1,
            )
            draw.text((mx + 20, row_y + 10), m.value[:24], font=f_metric_val, fill=(255, 255, 255, 255))
            draw.text((mx + 20, row_y + 48), m.label.upper()[:34], font=f_metric_lbl, fill=(148, 163, 184, 240))

    combined = Image.alpha_composite(im, overlay).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path, quality=92)
    return output_path


def measure_image_quality(path: Path) -> tuple[int, int, float, float]:
    """Return (width, height, laplacian_sharpness_variance, mean_luminance) for QA auditing."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        gray = np.asarray(rgb.convert("L"), dtype=np.float32)
        mean_lum = float(gray.mean())
        # Discrete 2D Laplacian operator: 4*C - (N + S + E + W)
        if gray.shape[0] > 4 and gray.shape[1] > 4:
            lap = (
                4.0 * gray[1:-1, 1:-1]
                - gray[:-2, 1:-1]
                - gray[2:, 1:-1]
                - gray[1:-1, :-2]
                - gray[1:-1, 2:]
            )
            sharpness = float(np.var(lap))
        else:
            sharpness = 0.0
    return w, h, sharpness, mean_lum
