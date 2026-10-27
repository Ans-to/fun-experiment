"""Render A/B stills and objective metrics for video-quality tuning.

Renders identical monument frames across internal raster scales and finishing variants,
reporting sharpness (Laplacian variance), edge aliasing, color-stat drift and per-frame
wall-clock cost. Writes side-by-side contact sheets under qa_evidence/quality_stills/.

Usage: PYTHONPATH=. .venv/bin/python tools/render_quality_compare.py
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from voxforge.video.procedural_world import (
    Monument3DEngine,
    SCENE_GIZA,
    SCENE_TAJ_MAHAL,
    SCENE_COLOSSEUM,
)
from voxforge.video.finish import edge_adaptive_sharpen

OUT_DIR = Path("qa_evidence/quality_stills")
SCENES = [("giza", SCENE_GIZA), ("taj", SCENE_TAJ_MAHAL), ("colosseum", SCENE_COLOSSEUM)]
SCALES = [0.5, 0.625, 0.75, 1.0]


def laplacian_var(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def alias_score(img: np.ndarray) -> float:
    """High-frequency jaggedness along strong edges: mean |2nd derivative| on Canny edge pixels."""
    gray_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_u8, 60, 140) > 0
    d2 = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=1)
    if edges.sum() < 50:
        return 0.0
    return float(np.abs(d2[edges]).mean())


def color_means(img: np.ndarray) -> np.ndarray:
    return img.reshape(-1, 3).mean(axis=0)


def label_bar(img: np.ndarray, label: str) -> np.ndarray:
    tile = img.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 44), (14, 16, 20), -1)
    cv2.putText(tile, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 220, 170), 2, cv2.LINE_AA)
    return tile


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Warm up numba once on a tiny frame.
    warm = Monument3DEngine(internal_scale=0.5, sharpen=False)
    warm.render_frame(SCENE_GIZA, 0.5, 1.0, 96, 54)

    print(f"{'scene':10s} {'scale':>6s} {'sharpen':>8s} {'lap_var':>9s} {'alias':>7s} {'dBGR':>7s} {'ms':>8s}")
    for name, sid in SCENES:
        base_means = None
        sheet_frames: list[tuple[str, np.ndarray]] = []
        for scale in SCALES:
            for do_sharp in (False, True):
                eng = Monument3DEngine(internal_scale=scale, sharpen=do_sharp)
                # time 3 frames, report median
                ts = []
                frame = None
                for k in range(3):
                    t0 = time.perf_counter()
                    frame = eng.render_frame(sid, 0.42, 2.5, 1920, 1080)
                    ts.append((time.perf_counter() - t0) * 1000)
                ms = float(np.median(ts))
                if base_means is None:
                    base_means = color_means(frame)
                drift = float(np.abs(color_means(frame) - base_means).mean())
                tag = f"s{scale:g}_{'sharp' if do_sharp else 'raw'}"
                print(
                    f"{name:10s} {scale:6.3f} {str(do_sharp):>8s} {laplacian_var(frame):9.1f} "
                    f"{alias_score(frame):7.2f} {drift:7.3f} {ms:8.1f}"
                )
                sheet_frames.append((tag, frame))
        cv2.imwrite(
            str(OUT_DIR / f"{name}_scale_matrix.jpg"),
            np.vstack([label_bar(img, tag) for tag, img in sheet_frames]),
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )

        # 1:1 pixel crops of the most detailed band for human edge inspection
        h, w = frame.shape[:2]
        y0, y1, x0, x1 = int(h * 0.34), int(h * 0.66), int(w * 0.30), int(w * 0.70)
        crops = []
        for tag, img in sheet_frames:
            if tag in ("s0.5_raw", "s0.5_sharp", "s0.75_sharp", "s1_sharp"):
                crops.append(label_bar(img[y0:y1, x0:x1], tag))
        cv2.imwrite(str(OUT_DIR / f"{name}_detail_crops.jpg"), np.vstack(crops))

    # Sharpen strength sweep at default scale
    print("\nsharpen amount sweep (Taj, scale 0.5):")
    eng = Monument3DEngine(internal_scale=0.5, sharpen=False)
    base = eng.render_frame(SCENE_TAJ_MAHAL, 0.42, 2.5, 1920, 1080)
    sweep = [label_bar(base, "amount=0.00")]
    for amt in (0.25, 0.40, 0.55, 0.75, 1.0):
        img = edge_adaptive_sharpen(base, amount=amt)
        print(f"  amount={amt:4.2f}  lap_var={laplacian_var(img):8.1f}  alias={alias_score(img):6.2f}")
        sweep.append(label_bar(img, f"amount={amt:.2f}"))
    cv2.imwrite(str(OUT_DIR / "sharpen_sweep.jpg"), np.vstack(sweep))
    print(f"\ncontact sheets -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
