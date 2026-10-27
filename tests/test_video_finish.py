"""Tests for the optical finishing chain: edge-adaptive sharpen, plate upscale,
configurable supersampling and Stage-E-safe color invariance."""

from __future__ import annotations

import cv2
import numpy as np

from voxforge.video.finish import edge_adaptive_sharpen, upscale_plate


def _lap_var(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def _max_edge_gradient(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(cv2.magnitude(gx, gy).max())


def test_sharpen_recovers_soft_edges_without_color_drift() -> None:
    # A mildly softened two-tone edge on an otherwise flat twilight plate (the same
    # mild softness a Lanczos upscale leaves on rasterized geometry).
    plate = np.full((240, 320, 3), (48, 58, 84), dtype=np.uint8)
    plate[:, 160:] = (158, 138, 104)
    soft = cv2.GaussianBlur(plate, (0, 0), 1.3)
    sharp = edge_adaptive_sharpen(soft)

    assert sharp.shape == soft.shape and sharp.dtype == np.uint8
    # The edge must get steeper and carry more local contrast after detail recovery.
    assert _max_edge_gradient(sharp) > _max_edge_gradient(soft) * 1.05
    assert _lap_var(sharp[:, 130:190]) > _lap_var(soft[:, 130:190]) * 1.3

    # Broad color statistics must stay put (broadcast grade + Stage E classifier safety).
    drift = np.abs(sharp.reshape(-1, 3).mean(axis=0) - soft.reshape(-1, 3).mean(axis=0)).mean()
    assert drift < 1.2

    # Flat regions carry no edge energy, so the gate leaves them bit-identical.
    assert np.array_equal(sharp[:40, :40], soft[:40, :40])
    assert np.array_equal(sharp[:40, -40:], soft[:40, -40:])


def test_sharpen_clamps_halos_and_handles_degenerate_input() -> None:
    rng = np.random.default_rng(11)
    plate = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)
    sharp = edge_adaptive_sharpen(plate, amount=1.2, max_contribution=20.0)
    assert sharp.min() >= 0 and sharp.max() <= 255
    # Halo clamp: no pixel moves more than the configured contribution plus rounding.
    assert np.abs(sharp.astype(np.int16) - plate.astype(np.int16)).max() <= 21
    # Non-3D input is returned untouched instead of crashing.
    flat = np.zeros((10, 10), dtype=np.uint8)
    assert edge_adaptive_sharpen(flat) is flat


def test_upscale_plate_picks_lanczos_area_and_passthrough() -> None:
    rng = np.random.default_rng(5)
    small = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    up = upscale_plate(small, 160, 120)
    assert up.shape == (120, 160, 3)
    down = upscale_plate(small, 40, 30)
    assert down.shape == (30, 40, 3)
    assert upscale_plate(small, 80, 60) is small  # identity passthrough, no copy


def test_encode_profiles(monkeypatch) -> None:
    from voxforge.video.renderer import _encode_rate_control

    monkeypatch.delenv("VOXFORGE_ENCODE_PROFILE", raising=False)
    assert _encode_rate_control(False) == ("22", "3000k", "6000k")
    assert _encode_rate_control(True) == ("23", "2000k", "4000k")

    monkeypatch.setenv("VOXFORGE_ENCODE_PROFILE", "compact")
    compact_std = _encode_rate_control(False)
    assert compact_std[0] == "24" and compact_std[1] == "1600k"

    monkeypatch.setenv("VOXFORGE_ENCODE_PROFILE", "BALANCED")  # case-insensitive
    assert _encode_rate_control(False)[0] == "23"

    # Unknown profiles never break a render: fall back to the tuned quality default.
    monkeypatch.setenv("VOXFORGE_ENCODE_PROFILE", "ultra-mega")
    assert _encode_rate_control(False) == ("22", "3000k", "6000k")


def test_env_float_parses_and_clamps(monkeypatch) -> None:
    from voxforge.video.procedural_world import _env_float

    monkeypatch.setenv("VX_FINISH_TEST", "0.9")
    assert _env_float("VX_FINISH_TEST", 0.5, 0.1, 1.0) == 0.9
    monkeypatch.setenv("VX_FINISH_TEST", "9.9")
    assert _env_float("VX_FINISH_TEST", 0.5, 0.1, 1.0) == 1.0
    monkeypatch.setenv("VX_FINISH_TEST", "garbage")
    assert _env_float("VX_FINISH_TEST", 0.5, 0.1, 1.0) == 0.5
    monkeypatch.delenv("VX_FINISH_TEST", raising=False)
    assert _env_float("VX_FINISH_TEST", 0.42, 0.1, 1.0) == 0.42


def test_monument_supersampling_crisper_and_stage_e_safe() -> None:
    from voxforge.audit.av_sync_qa import classify_frame_group
    from voxforge.video.procedural_world import MONUMENT_RENDER_SCALE, SCENE_TAJ_MAHAL, Monument3DEngine

    # JIT warmup on a tiny frame before timing/quality probes.
    warm = Monument3DEngine(internal_scale=0.5, sharpen=False)
    warm.render_frame(SCENE_TAJ_MAHAL, 0.5, 1.0, 96, 54)

    eng_lo = Monument3DEngine(internal_scale=0.5, sharpen=False)
    frame_lo = eng_lo.render_frame(SCENE_TAJ_MAHAL, 0.42, 2.5, 1920, 1080)
    eng_lo_sharp = Monument3DEngine(internal_scale=0.5, sharpen=True)
    frame_lo_sharp = eng_lo_sharp.render_frame(SCENE_TAJ_MAHAL, 0.42, 2.5, 1920, 1080)
    eng_hi = Monument3DEngine(internal_scale=1.0, sharpen=False)
    frame_hi = eng_hi.render_frame(SCENE_TAJ_MAHAL, 0.42, 2.5, 1920, 1080)

    for frame in (frame_lo, frame_lo_sharp, frame_hi):
        assert frame.shape == (1080, 1920, 3) and frame.dtype == np.uint8

    # Native-resolution rasterization resolves more detail than the upscaled half-res buffer,
    # and edge-adaptive sharpening adds detail back at the fixed default scale.
    assert _lap_var(frame_hi) > _lap_var(frame_lo)
    assert _lap_var(frame_lo_sharp) > _lap_var(frame_lo) * 1.3

    # Scale/finishing changes must not shift the plate's broadcast color balance materially.
    mean_drift = np.abs(frame_hi.mean(axis=(0, 1)) - frame_lo.mean(axis=(0, 1))).mean()
    assert mean_drift < 6.0

    # Engine plates are RGB; the Stage E decoder hands back BGR, so flip before classifying.
    label = classify_frame_group(frame_lo_sharp[..., ::-1])
    assert label in {"monument", "geo"}  # never collapses into darkhud/quote after finishing

    # The shipped default stays the legacy half-res geometry unless explicitly overridden.
    assert MONUMENT_RENDER_SCALE == 0.5
