"""Controlled x264 encode tuning on real engine frames.

Builds a representative raw rgb24 sequence spanning all four Stage E frame families
(monument twilight cinema, orbital globe, near-black HUD sim, LiDAR CAD), encodes it
with candidate rate-control settings, and reports bitrate vs PSNR/SSIM against a
high-quality reference plus Stage E family stability.

Usage: PYTHONPATH=. .venv/bin/python tools/encode_tuning.py
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from voxforge.audit.av_sync_qa import classify_frame_group
from voxforge.video.procedural_world import (
    Monument3DEngine,
    PhysicsSimulationEngine,
    ProceduralCadCache,
    SCENE_GIZA,
    SCENE_TAJ_MAHAL,
    SCENE_CLASSICAL_GENERIC,
)

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 24


def build_frames(n_per: int = 36) -> list[np.ndarray]:
    mon = Monument3DEngine()
    sim = PhysicsSimulationEngine()
    cad = ProceduralCadCache(mon)
    # JIT warmups
    mon.render_frame(SCENE_GIZA, 0.5, 1.0, 96, 54)
    sim.render_simulation_frame(SCENE_CLASSICAL_GENERIC, 0.5, 1.0, 96, 54, hint="warmup")
    cad.render_frame(SCENE_GIZA, 0.5, 96, 54)

    frames: list[np.ndarray] = []
    for i in range(n_per):
        u = 0.2 + 0.6 * i / max(1, n_per - 1)
        t = i / FPS
        frames.append(mon.render_frame(SCENE_TAJ_MAHAL, u, t, W, H))          # monument
        frames.append(cad.render_frame(SCENE_GIZA, u, W, H))                   # darkhud LiDAR
        frames.append(
            sim.render_simulation_frame(
                SCENE_CLASSICAL_GENERIC, u, t, W, H,
                hint="Trade rose 250 percent across a network of 40 cities",
            )
        )                                                                       # darkhud sim
    return frames


def encode(raw_path: Path, out_path: Path, crf: str, maxrate: str, bufsize: str, preset: str, aq: str) -> None:
    cmd = [
        FF, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", str(raw_path),
        "-c:v", "libx264", "-preset", preset, "-g", "240",
        "-crf", crf, "-maxrate", maxrate, "-bufsize", bufsize,
        "-x264-params", aq,
        "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def decode(out_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(out_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse < 1e-9 else 10.0 * np.log10((255.0 ** 2) / mse)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gb, (11, 11), 1.5)
    sa = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return float(np.clip(m, 0.0, 1.0).mean())


def main() -> None:
    frames = build_frames()
    print(f"built {len(frames)} raw frames ({W}x{H})")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = td / "seq.rgb"
        with open(raw, "wb") as fh:
            for f in frames:
                fh.write(np.ascontiguousarray(f).tobytes())

        ref_path = td / "ref.mp4"
        encode(raw, ref_path, "12", "20000k", "40000k", "medium", "aq-mode=3:aq-strength=0.8")
        ref = decode(ref_path)
        n = min(len(ref), len(frames))
        ref = ref[:n]

        candidates = [
            # tag, crf, maxrate, bufsize, preset, aq
            ("CRF21 cap4000 superfast aq3", "21", "4000k", "8000k", "superfast", "aq-mode=3:aq-strength=0.8"),
            ("CRF22 cap3000 superfast aq3", "22", "3000k", "6000k", "superfast", "aq-mode=3:aq-strength=0.8"),
            ("CRF22 cap4000 superfast aq3", "22", "4000k", "8000k", "superfast", "aq-mode=3:aq-strength=0.8"),
            ("CRF23 cap3000 superfast aq3", "23", "3000k", "6000k", "superfast", "aq-mode=3:aq-strength=0.8"),
            ("CRF21 cap3000 veryfast  aq3", "21", "3000k", "6000k", "veryfast", "aq-mode=3:aq-strength=0.8"),
            ("CRF22 cap3500 veryfast  aq3", "22", "3500k", "7000k", "veryfast", "aq-mode=3:aq-strength=0.8"),
            ("CRF23 cap2500 veryfast  aq3", "23", "2500k", "5000k", "veryfast", "aq-mode=3:aq-strength=0.8"),
            ("LEGACY CRF21 cap840 sf aq1", "21", "840k", "1680k", "superfast", "aq-mode=1:aq-strength=1.0"),
        ]
        # Raw-engine family labels: compression must never flip a frame family.
        raw_labels = {i: classify_frame_group(frames[i][..., ::-1]) for i in range(0, n, 12)}
        print("raw labels:", raw_labels)
        print(f"{'candidate':32s} {'kbps':>7s} {'PSNR':>7s} {'SSIM':>7s}  label_flips")
        for tag, crf, mr, bs, preset, aq in candidates:
            out = td / "cand.mp4"
            encode(raw, out, crf, mr, bs, preset, aq)
            dec = decode(out)[:n]
            kbps = out.stat().st_size * 8 / (n / FPS) / 1000
            ps = np.mean([psnr(ref[i], dec[i]) for i in range(0, n, 3)])
            ss = np.mean([ssim_gray(ref[i], dec[i]) for i in range(0, n, 3)])
            flips = [
                (i, raw_labels[i], classify_frame_group(dec[i][..., ::-1]))
                for i in raw_labels
                if classify_frame_group(dec[i][..., ::-1]) != raw_labels[i]
            ]
            print(f"{tag:32s} {kbps:7.0f} {ps:7.2f} {ss:7.4f}  {flips if flips else 'none'}")


if __name__ == "__main__":
    main()
