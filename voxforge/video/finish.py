"""Optical finishing primitives for pure-code cinema frames (post-raster, pre-grain).

These helpers operate on clean, grain-free RGB uint8 plates (the engine's native channel
order, matching the ``rgb24`` encoder pipe) *before* the Rec.709 print LUT and archival
grain are applied in ``StudioCharacterAnimator._finish_cinema_frame``, so grain is never
sharpened and broadcast color means stay invariant.

`edge_adaptive_sharpen`
    Halo-controlled unsharp mask gated by a luma-gradient edge map. Flat twilight skies,
    starfields and vignettes are left untouched; architectural silhouette edges, masonry
    courses, LiDAR contours and GIS coastlines recover the crispness lost when internal
    half-resolution rasterizers upscale to 1080p.
"""

from __future__ import annotations

import cv2
import numpy as np


def edge_adaptive_sharpen(
    img_rgb: np.ndarray,
    amount: float = 0.55,
    blur_sigma: float = 1.15,
    edge_ramp: float = 46.0,
    edge_smooth_sigma: float = 1.25,
    max_contribution: float = 34.0,
) -> np.ndarray:
    """Sharpen real geometry edges while protecting flat gradients from noise amplification.

    Parameters
    ----------
    img_rgb:
        RGB uint8 plate (any resolution), grain-free preferred.
    amount:
        Peak unsharp strength applied at a full-confidence edge.
    blur_sigma:
        Gaussian radius of the low-pass reference; smaller = tighter, more local crispness.
    edge_ramp:
        Gradient-magnitude value (0-255 luma scale) at which edge confidence reaches 1.0.
    edge_smooth_sigma:
        Softens the edge gate so the sharpen/unsharpen boundary itself never becomes visible.
    max_contribution:
        Per-channel clamp on the sharpening delta, suppressing halo rings around hot edges.

    Returns
    -------
    np.ndarray
        RGB uint8 plate, same shape, same broad color statistics.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        return img_rgb
    f = img_rgb.astype(np.float32)

    low = cv2.GaussianBlur(f, (0, 0), blur_sigma)
    highpass = f - low

    # Rec.709 luma straight from RGB (avoid cvtColor's BGR assumption), float32 for OpenCV.
    luma = (
        img_rgb[..., 0].astype(np.float32) * 0.2126
        + img_rgb[..., 1].astype(np.float32) * 0.7152
        + img_rgb[..., 2].astype(np.float32) * 0.0722
    )
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    edge_gate = np.clip(grad / max(1.0, edge_ramp), 0.0, 1.0)
    edge_gate = cv2.GaussianBlur(edge_gate, (0, 0), edge_smooth_sigma)[..., None]

    delta = np.clip(highpass * (amount * edge_gate), -max_contribution, max_contribution)
    return np.clip(f + delta, 0.0, 255.0).astype(np.uint8)


def upscale_plate(
    img_rgb: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """High-quality plate upscale.

    Lanczos-4 for upscaling (tighter than bicubic, minimal ringing); area averaging when
    downscaling (true low-pass, no aliasing) so supersampled internal buffers collapse
    cleanly onto the delivery raster.
    """
    h, w = img_rgb.shape[:2]
    if (w, h) == (target_w, target_h):
        return img_rgb
    if target_w < w or target_h < h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LANCZOS4
    return cv2.resize(img_rgb, (target_w, target_h), interpolation=interp)
