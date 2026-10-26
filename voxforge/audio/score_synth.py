"""Cinema Documentary Film Score Synthesizer (Bowed Cello Ensemble, Felt Piano & Hall Reverb).

Generates warm Max Richter / Jóhann Jóhannsson / BBC Horizon-inspired documentary film
scores at 48kHz stereo using physical bowed-string harmonic series, wool-felt piano
inharmonicity & partial-dependent decay, sub-bass cinema swells, and stereo concert-hall
reverb — with ZERO raw sine-wave test tones and a clean 1–4 kHz speech-intelligibility pocket.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from voxforge.config import AUDIO_SPEC
from voxforge.models import MusicCue

# Contemplative cinema documentary voicings (D Dorian / F Lydian / A Aeolian / C Minor9)
_CHORD_PALETTES: dict[MusicCue, list[float]] = {
    "cold_open_pulse": [73.42, 110.00, 146.83, 220.00, 329.63],          # D2 A2 D3 A3 E4 (open fifths + 9th)
    "theme_swell": [73.42, 110.00, 146.83, 220.00, 261.63, 329.63],      # Dmin9 cinema swell
    "ambient_warm": [73.42, 146.83, 220.00, 261.63, 329.63],             # D2 D3 A3 C4 E4
    "wonder_pad": [87.31, 130.81, 174.61, 220.00, 261.63, 392.00],       # Fmaj9(#11) Lydian awe
    "tension_drone": [65.41, 98.00, 130.81, 196.00, 233.08, 293.66],     # Cmin9 mystery
    "finale_resolve": [73.42, 110.00, 146.83, 220.00, 293.66, 369.99],   # Dmaj(add9) horizon resolve
    "none": [],
}

_FELT_ARPEGGIO_NOTES: dict[MusicCue, list[tuple[float, float]]] = {
    "cold_open_pulse": [(0.35, 293.66), (1.85, 329.63), (3.10, 220.00)],
    "theme_swell": [(0.28, 293.66), (1.15, 440.00), (2.05, 329.63)],
    "ambient_warm": [(2.0, 293.66), (8.0, 329.63), (14.0, 261.63), (20.0, 220.00)],
    "wonder_pad": [(1.5, 261.63), (7.5, 329.63), (13.5, 392.00), (19.5, 329.63)],
    "tension_drone": [(3.0, 261.63), (11.0, 293.66), (18.5, 233.08)],
    "finale_resolve": [(0.30, 293.66), (1.20, 369.99), (2.25, 440.00)],
    "none": [],
}


def _build_hall_ir(sr: int, rt60_sec: float = 2.35) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth, dark-damped stereo concert-hall impulse response (decorrelated L/R)."""
    ir_len = int(round(rt60_sec * sr))
    t = np.arange(ir_len, dtype=np.float32) / float(sr)
    rng = np.random.default_rng(1979)

    decay = np.exp(-6.9078 * t / rt60_sec).astype(np.float32)
    # Soft pre-delay & early reflection bloom (22 ms) so notes never click
    bloom = np.clip(t / 0.028, 0.0, 1.0) ** 1.5
    ir_l = rng.normal(0.0, 1.0, ir_len).astype(np.float32) * decay * bloom
    ir_r = rng.normal(0.0, 1.0, ir_len).astype(np.float32) * decay * bloom

    # Dark concert-hall absorption @ 1,450 Hz
    sos_lp = butter(2, 1450.0 / (sr * 0.5), btype="low", output="sos")
    sos_hp = butter(2, 55.0 / (sr * 0.5), btype="high", output="sos")
    ir_l = sosfilt(sos_hp, sosfilt(sos_lp, ir_l)).astype(np.float32)
    ir_r = sosfilt(sos_hp, sosfilt(sos_lp, ir_r)).astype(np.float32)

    norm = float(np.sqrt(np.sum(ir_l * ir_l) + 1e-9))
    return ir_l / norm, ir_r / norm


def _apply_cinema_hall_reverb(left: np.ndarray, right: np.ndarray, sr: int, wet: float = 0.44) -> tuple[np.ndarray, np.ndarray]:
    ir_l, ir_r = _build_hall_ir(sr, rt60_sec=2.35)
    wet_l = fftconvolve(0.75 * left + 0.25 * right, ir_l, mode="full")[: len(left)].astype(np.float32)
    wet_r = fftconvolve(0.25 * left + 0.75 * right, ir_r, mode="full")[: len(right)].astype(np.float32)
    return (1.0 - wet) * left + wet * wet_l, (1.0 - wet) * right + wet * wet_r


def _bowed_string_section(
    t: np.ndarray,
    f0: float,
    pan: float = 0.0,
    seed: int = 0,
    exact_loop_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a warm 3-chair bowed cello/viola voice with natural harmonics & gentle vibrato."""
    n = len(t)
    sig_l = np.zeros(n, dtype=np.float32)
    sig_r = np.zeros(n, dtype=np.float32)

    for chair, cents in enumerate((-4.2, 0.0, +4.5)):
        fc = f0 * (2.0 ** (cents / 1200.0))
        if exact_loop_sec is not None:
            fc = max(1, int(round(fc * exact_loop_sec))) / exact_loop_sec
            vib_hz = max(1, int(round(4.6 * exact_loop_sec))) / exact_loop_sec
        else:
            vib_hz = 4.5 + 0.35 * chair

        vib_phase = 0.0018 * np.sin(2.0 * math.pi * vib_hz * t + chair * 1.7 + seed)
        inst_phase = 2.0 * math.pi * fc * t + vib_phase * fc * 0.45

        voice = np.zeros(n, dtype=np.float32)
        max_k = min(8, int(3600.0 // max(fc, 20.0)))
        for k in range(1, max(2, max_k + 1)):
            # Warm bowed string spectral slope (~ -8.5 dB/octave)
            harm_amp = 1.0 / (float(k) ** 1.48)
            voice += harm_amp * np.sin(k * inst_phase + (k * 0.41 + chair * 0.9))

        c_pan = float(np.clip(pan + (chair - 1) * 0.35, -0.85, 0.85))
        gl = math.sqrt(0.5 * (1.0 - c_pan))
        gr = math.sqrt(0.5 * (1.0 + c_pan))
        sig_l += gl * voice
        sig_r += gr * voice

    return sig_l * 0.33, sig_r * 0.33


def _felt_piano_note(
    t: np.ndarray,
    onset_sec: float,
    f0: float,
    pan: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a soft wool-felt documentary piano note (fast upper-partial damping, warm fundamental)."""
    mask = t >= onset_sec
    if not np.any(mask):
        z = np.zeros_like(t)
        return z, z

    dt = np.maximum(0.0, t - onset_sec)
    # Soft wool-felt hammer attack (14 ms — zero harsh click or MIDI ping)
    attack = (1.0 - np.exp(-dt / 0.014)) * mask.astype(np.float32)
    inharm_b = 0.00032
    note = np.zeros_like(t)
    for k in range(1, 6):
        fk = k * f0 * math.sqrt(1.0 + inharm_b * (k * k))
        # Upper partials decay much faster on felt piano, leaving a pure warm wooden tail
        decay_k = np.exp(-dt * (0.85 + 1.15 * (k - 1)))
        amp_k = 1.0 / (float(k) ** 2.35)
        note += amp_k * attack * decay_k * np.sin(2.0 * math.pi * fk * dt + k * 0.25)

    gl = math.sqrt(0.5 * (1.0 - pan))
    gr = math.sqrt(0.5 * (1.0 + pan))
    return note * gl, note * gr


def synthesize_chapter_bumper(
    duration_sec: float,
    cue: MusicCue = "theme_swell",
    sr: int = AUDIO_SPEC.sample_rate,
) -> np.ndarray:
    """Synthesize a warm bowed-cello + felt-piano cinematic act transition swell ([N, 2])."""
    n = max(1, int(round(duration_sec * sr)))
    t = np.arange(n, dtype=np.float32) / sr
    freqs = _CHORD_PALETTES.get(cue) or _CHORD_PALETTES["theme_swell"]

    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    for idx, f0 in enumerate(freqs):
        w = 1.0 / (1.0 + 0.42 * idx)
        pan = ((idx % 3) - 1) * 0.35
        sl, sr_ch = _bowed_string_section(t, f0, pan=pan, seed=idx)
        left += w * sl
        right += w * sr_ch

    # Sparse contemplative felt-piano notes
    for n_i, (onset, pf) in enumerate(_FELT_ARPEGGIO_NOTES.get(cue, [])[:2]):
        if onset < duration_sec - 0.5:
            pl, pr = _felt_piano_note(t, onset, pf, pan=-0.22 if n_i % 2 == 0 else 0.22)
            left += 0.42 * pl
            right += 0.42 * pr

    # Warm cello-body low-pass @ 1,080 Hz so transitions feel cinematic & deep
    sos = butter(3, 1080.0 / (sr * 0.5), btype="low", output="sos")
    left = sosfilt(sos, left).astype(np.float32)
    right = sosfilt(sos, right).astype(np.float32)
    left, right = _apply_cinema_hall_reverb(left, right, sr, wet=0.48)

    env = np.ones(n, dtype=np.float32)
    fade_in = min(int(0.75 * sr), n // 3)
    fade_out = min(int(0.95 * sr), n // 3)
    if fade_in > 1:
        env[:fade_in] = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, fade_in, dtype=np.float32)))
    if fade_out > 1:
        env[-fade_out:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, fade_out, dtype=np.float32)))

    stereo = np.stack([left * env, right * env], axis=1)
    peak = float(np.max(np.abs(stereo))) + 1e-8
    target_amp = 10.0 ** (-15.5 / 20.0)
    return (stereo / peak * target_amp).astype(np.float32)


def synthesize_underscore_loop(
    cue: MusicCue,
    loop_sec: float = 24.0,
    sr: int = AUDIO_SPEC.sample_rate,
) -> np.ndarray:
    """Synthesize a seamless 24s bowed-cello & sparse felt-piano documentary underscore bed."""
    if cue == "none":
        n = int(round(loop_sec * sr))
        return np.zeros((n, 2), dtype=np.float32)

    n = int(round(loop_sec * sr))
    t = np.arange(n, dtype=np.float32) / sr
    freqs = _CHORD_PALETTES.get(cue) or _CHORD_PALETTES["ambient_warm"]

    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)

    for idx, raw_f in enumerate(freqs):
        lfo_cycles = max(1, (idx % 3) + 1)
        lfo = 0.68 + 0.32 * np.sin(2.0 * math.pi * (lfo_cycles / loop_sec) * t + idx * 1.1)
        weight = 1.0 / (1.0 + 0.50 * idx)
        sl, sr_ch = _bowed_string_section(t, raw_f, pan=((idx % 3) - 1) * 0.3, seed=idx, exact_loop_sec=loop_sec)
        left += weight * lfo * sl
        right += weight * lfo * sr_ch

    # Sparse contemplative felt piano notes spaced gently every ~6s
    for n_i, (onset, pf) in enumerate(_FELT_ARPEGGIO_NOTES.get(cue, _FELT_ARPEGGIO_NOTES["ambient_warm"])):
        if onset < loop_sec - 2.5:
            pl, pr = _felt_piano_note(t, onset, pf, pan=-0.25 if n_i % 2 == 0 else 0.25)
            left += 0.26 * pl
            right += 0.26 * pr

    # Warm low-pass @ 740 Hz so underscore supports emotion without touching vocal clarity!
    sos = butter(3, 740.0 / (sr * 0.5), btype="low", output="sos")
    left = sosfilt(sos, left).astype(np.float32)
    right = sosfilt(sos, right).astype(np.float32)
    left, right = _apply_cinema_hall_reverb(left, right, sr, wet=0.45)

    xf = int(0.18 * sr)
    if n > xf * 4:
        alpha = np.linspace(0.0, 1.0, xf, dtype=np.float32)
        left[:xf] = left[:xf] * alpha + left[-xf:] * (1.0 - alpha)
        right[:xf] = right[:xf] * alpha + right[-xf:] * (1.0 - alpha)

    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) + 1e-8
    target_amp = 10.0 ** (AUDIO_SPEC.music_bed_dbfs / 20.0)
    return (stereo / peak * target_amp).astype(np.float32)


def synthesize_intro_bumper_theme(
    duration_sec: float = 4.2,
    sr: int = AUDIO_SPEC.sample_rate,
) -> np.ndarray:
    """Synthesize a deep Zimmer/Lemmino bowed-cello swell + sub-bass cinema drum + felt piano motif."""
    n = max(1, int(round(duration_sec * sr)))
    t = np.arange(n, dtype=np.float32) / sr
    chord = [73.42, 110.00, 146.83, 220.00, 329.63]  # D2 A2 D3 A3 E4

    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    for idx, f0 in enumerate(chord):
        w = 1.0 / (1.0 + 0.38 * idx)
        sl, sr_ch = _bowed_string_section(t, f0, pan=((idx % 3) - 1) * 0.35, seed=idx + 10)
        left += w * sl
        right += w * sr_ch

    # Deep cinema taiko / sub-bass impact at t = 0.12s (56 Hz -> 34 Hz pitch envelope)
    dt_imp = np.maximum(0.0, t - 0.10)
    imp_mask = (t >= 0.10).astype(np.float32)
    imp_phase = 2.0 * math.pi * (34.0 * dt_imp + 22.0 * (1.0 - np.exp(-dt_imp * 6.0)) / 6.0)
    sub_hit = 0.55 * imp_mask * (1.0 - np.exp(-dt_imp * 60.0)) * np.exp(-dt_imp * 1.85) * np.sin(imp_phase)
    left += sub_hit
    right += sub_hit

    # Soft wool-felt piano 2-note documentary signature (A3 -> E4, NO high sine beeps!)
    for onset, pf, pan in ((0.42, 220.00, -0.18), (1.35, 329.63, 0.18)):
        pl, pr = _felt_piano_note(t, onset, pf, pan=pan)
        left += 0.45 * pl
        right += 0.45 * pr

    sos_pad = butter(3, 1180.0 / (sr * 0.5), btype="low", output="sos")
    left = sosfilt(sos_pad, left).astype(np.float32)
    right = sosfilt(sos_pad, right).astype(np.float32)
    left, right = _apply_cinema_hall_reverb(left, right, sr, wet=0.50)

    env = np.ones(n, dtype=np.float32)
    fi = min(int(0.28 * sr), n // 4)
    fo = min(int(0.75 * sr), n // 3)
    if fi > 1:
        env[:fi] = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, fi, dtype=np.float32)))
    if fo > 1:
        env[-fo:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, fo, dtype=np.float32)))

    stereo = np.stack([left * env, right * env], axis=1)
    peak = float(np.max(np.abs(stereo))) + 1e-8
    target_amp = 10.0 ** (-14.2 / 20.0)
    return (stereo / peak * target_amp).astype(np.float32)


def synthesize_outro_end_theme(
    duration_sec: float = 5.0,
    sr: int = AUDIO_SPEC.sample_rate,
) -> np.ndarray:
    """Synthesize a warm bowed-cello & felt-piano resolving documentary end-credits cadence."""
    n = max(1, int(round(duration_sec * sr)))
    t = np.arange(n, dtype=np.float32) / sr
    chord = [73.42, 110.00, 146.83, 220.00, 293.66, 369.99]  # Warm Dmaj(add9)

    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    for idx, f0 in enumerate(chord):
        w = 1.0 / (1.0 + 0.42 * idx)
        sl, sr_ch = _bowed_string_section(t, f0, pan=((idx % 3) - 1) * 0.35, seed=idx + 20)
        left += w * sl
        right += w * sr_ch

    for onset, pf, pan in ((0.30, 293.66, -0.20), (1.45, 369.99, 0.20)):
        pl, pr = _felt_piano_note(t, onset, pf, pan=pan)
        left += 0.42 * pl
        right += 0.42 * pr

    sos = butter(3, 1120.0 / (sr * 0.5), btype="low", output="sos")
    left = sosfilt(sos, left).astype(np.float32)
    right = sosfilt(sos, right).astype(np.float32)
    left, right = _apply_cinema_hall_reverb(left, right, sr, wet=0.52)

    env = np.ones(n, dtype=np.float32)
    fi = min(int(0.55 * sr), n // 4)
    fo = min(int(1.45 * sr), n // 2)
    if fi > 1:
        env[:fi] = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, fi, dtype=np.float32)))
    if fo > 1:
        env[-fo:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, fo, dtype=np.float32)))

    stereo = np.stack([left * env, right * env], axis=1)
    peak = float(np.max(np.abs(stereo))) + 1e-8
    target_amp = 10.0 ** (-14.8 / 20.0)
    return (stereo / peak * target_amp).astype(np.float32)


def synthesize_map_transition_whoosh(
    duration_sec: float = 1.15,
    sr: int = AUDIO_SPEC.sample_rate,
    seed: int = 101,
) -> np.ndarray:
    """Subtle stereo low-passed air/lens movement for 3D GIS orbital map entries (-27 dBFS)."""
    n = max(1, int(round(duration_sec * sr)))
    t = np.arange(n, dtype=np.float32) / sr
    rng = np.random.default_rng(seed)

    # Smooth stereo-panning filtered air envelope
    bell = np.sin(np.pi * np.clip(t / max(0.2, duration_sec), 0.0, 1.0)) ** 2.2
    raw_l = rng.normal(0.0, 1.0, n).astype(np.float32)
    raw_r = rng.normal(0.0, 1.0, n).astype(np.float32)

    sos_bp = butter(2, [110.0 / (sr * 0.5), 480.0 / (sr * 0.5)], btype="band", output="sos")
    air_l = sosfilt(sos_bp, raw_l).astype(np.float32)
    air_r = sosfilt(sos_bp, raw_r).astype(np.float32)

    # Gentle L -> R pan glide matching orbital camera arc
    pan_u = np.linspace(0.25, 0.75, n, dtype=np.float32)
    sub_swell = 0.35 * np.sin(2.0 * np.pi * (52.0 + 18.0 * pan_u) * t) * bell
    left = (air_l * (1.0 - 0.45 * pan_u) + sub_swell) * bell
    right = (air_r * (0.55 + 0.45 * pan_u) + sub_swell) * bell

    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) + 1e-8
    return (stereo / peak * (10.0 ** (-27.0 / 20.0))).astype(np.float32)


def synthesize_editorial_accent_chime(
    duration_sec: float = 1.75,
    sr: int = AUDIO_SPEC.sample_rate,
) -> np.ndarray:
    """Delicate warm felt-piano fifth interval through concert-hall reverb for quote/telemetry reveals (-27.5 dBFS)."""
    n = max(1, int(round(duration_sec * sr)))
    t = np.arange(n, dtype=np.float32) / sr
    l1, r1 = _felt_piano_note(t, 0.02, 293.66, pan=-0.18)  # D4
    l2, r2 = _felt_piano_note(t, 0.16, 440.00, pan=0.18)   # A4
    left, right = _apply_cinema_hall_reverb(l1 + 0.72 * l2, r1 + 0.72 * r2, sr, wet=0.55)
    fo = min(int(0.45 * sr), n // 3)
    env = np.ones(n, dtype=np.float32)
    if fo > 1:
        env[-fo:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, fo, dtype=np.float32)))
    stereo = np.stack([left * env, right * env], axis=1)
    peak = float(np.max(np.abs(stereo))) + 1e-8
    return (stereo / peak * (10.0 ** (-27.5 / 20.0))).astype(np.float32)
