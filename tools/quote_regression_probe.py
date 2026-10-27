"""Isolate why a quote_card scene classifies as darkhud: raw frames vs encode settings."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import av
import imageio_ffmpeg
import numpy as np

from voxforge.audit.av_sync_qa import classify_frame_group
from voxforge.video.animator import StudioCharacterAnimator

FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1920, 1080, 24


def white_frac_bgr(f: np.ndarray) -> tuple[float, float]:
    f = f.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    wf = np.mean((r > 198) & (g > 198) & (b > 198))
    return float(wf), float(lum.mean())


def encode_decode(raw_path: Path, n: int, crf: str, maxrate: str, bufsize: str, aq: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "q.mp4"
        cmd = [
            FF, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", str(raw_path),
            "-c:v", "libx264", "-preset", "superfast", "-g", "240",
            "-crf", crf, "-maxrate", maxrate, "-bufsize", bufsize, "-x264-params", aq,
            "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv", str(out),
        ]
        subprocess.run(cmd, check=True)
        c = av.open(str(out)); vs = c.streams.video[0]
        frames = [fr.to_ndarray(format="bgr24") for fr in c.decode(vs)]
        return frames


def main() -> None:
    anim = StudioCharacterAnimator()
    n = 110
    quote = "The question is simple: what did these builders understand about stone, weight, and time?"
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw.rgb"
    frames_rgb = list(
        anim.iter_kinetic_quote_frames(
            tmp / "none.jpg", quote, "Herodotus", "Histories, c. 440 BCE",
            n, seg_seed=6, scene_hint="Pharos Lighthouse Alexandria",
        )
    )
    with open(raw, "wb") as fh:
        for f in frames_rgb:
            fh.write(np.ascontiguousarray(f).tobytes())

    # Raw (pre-encode) classifier inputs at 25/50/75% progress
    print("RAW frames (pre-encode):")
    for p in (0.25, 0.5, 0.75):
        i = min(n - 1, int(p * n))
        bgr = frames_rgb[i][..., ::-1]
        wf, ml = white_frac_bgr(bgr)
        print(f"  p={p:.2f} white_frac={wf:.4f} mean_lum={ml:.2f} label={classify_frame_group(bgr)}")

    configs = [
        ("OLD crf21 cap840 aq1", "21", "840k", "1680k", "aq-mode=1:aq-strength=1.0"),
        ("NEW crf22 cap3000 aq3", "22", "3000k", "6000k", "aq-mode=3:aq-strength=0.8"),
        ("NEW crf22 cap3000 aq1", "22", "3000k", "6000k", "aq-mode=1:aq-strength=1.0"),
    ]
    for tag, crf, mr, bs, aq in configs:
        dec = encode_decode(raw, n, crf, mr, bs, aq)
        print(f"\n{tag}:")
        for p in (0.25, 0.5, 0.75):
            i = min(len(dec) - 1, int(p * len(dec)))
            wf, ml = white_frac_bgr(dec[i])
            print(f"  p={p:.2f} white_frac={wf:.4f} mean_lum={ml:.2f} label={classify_frame_group(dec[i])}")


if __name__ == "__main__":
    main()
