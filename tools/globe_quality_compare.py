"""A/B the orbital Earth globe: legacy 720p bicubic vs current raycast scale + Lanczos + sharpen."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from voxforge.audit.av_sync_qa import classify_frame_group
from voxforge.video.finish import edge_adaptive_sharpen, upscale_plate
from voxforge.video.procedural_world import GLOBE_RENDER_SCALE, raycast_orbital_earth_globe

OUT = Path("qa_evidence/quality_stills")


def lap_var(rgb: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())


def main() -> None:
    # Stand up the same Natural Earth atlas the animator uses.
    from voxforge.video.animator import _NaturalEarthAtlas

    atlas = _NaturalEarthAtlas(3840, 1920).render_highlighted_region(("EGY", "ITA"))
    cam = dict(
        cam_lat_rad=0.42,
        cam_lon_rad=0.55,
        cam_dist=1.95,
        sun_dx=0.45,
        sun_dy=0.35,
        sun_dz=0.82,
    )

    # Legacy path: 1280x720 raycast -> bicubic 1080p
    legacy_low, _ = raycast_orbital_earth_globe(atlas, 1280, 720, **cam)
    legacy = cv2.resize(legacy_low, (1920, 1080), interpolation=cv2.INTER_CUBIC)

    # Current path: configurable raycast scale -> Lanczos -> gentle edge sharpen
    gw, gh = int(round(1920 * GLOBE_RENDER_SCALE)), int(round(1080 * GLOBE_RENDER_SCALE))
    new_low, _ = raycast_orbital_earth_globe(atlas, gw, gh, **cam)
    new = upscale_plate(new_low, 1920, 1080)
    new = edge_adaptive_sharpen(new, amount=0.45)

    print(f"globe raycast: legacy 1280x720 vs current {gw}x{gh}")
    for tag, img in (("legacy_720p_cubic", legacy), ("current_lanczos_sharp", new)):
        bgr = img[..., ::-1]  # Stage E decodes BGR from the MP4
        print(f"  {tag:24s} lap_var={lap_var(img):8.1f}  stage_e={classify_frame_group(bgr)}")

    drift = np.abs(new.mean(axis=(0, 1)) - legacy.mean(axis=(0, 1))).mean()
    print(f"  mean RGB drift: {drift:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    bar_h = 44
    tiles = []
    for tag, img in (("legacy 720p cubic", legacy), ("current lanczos+sharpen", new)):
        t = img.copy()
        cv2.rectangle(t, (0, 0), (1920, bar_h), (14, 16, 20), -1)
        cv2.putText(t, tag, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 220, 170), 2, cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite(str(OUT / "globe_ab.jpg"), np.vstack(tiles))
    print(f"  contact sheet -> {OUT / 'globe_ab.jpg'}")


if __name__ == "__main__":
    main()
