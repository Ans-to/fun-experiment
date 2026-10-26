"""Multi-Tier Zero-API-Key Neural Voice Studio Engine (Kokoro-82M v1.0 FP32 Primary).

Engineered for human-grade conversational podcast delivery:
  1. Full unquantized 32-bit FP32 `Kokoro-82M v1.0` ONNX (`StyleTTS2 + ISTFTNet`, 24kHz -> 48kHz)
     eliminating both INT8 iSTFT quantization harmonics and FP16 CPU onset drops.
  2. Geodesic SLERP (Spherical Linear Interpolation) on StyleTTS2 voice style manifolds so
     blended host voices retain 100% hypersphere norm, chest warmth, and natural pitch contour.
  3. Abbreviation-safe sentence splitting + mid-sentence stray-pause tightening + context-aware
     human breath pauses with deterministic micro-jitter (never robotic metronome timing).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import wave
from pathlib import Path
from typing import Any

import av
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from voxforge.bootstrap import StudioEnvironment
from voxforge.config import AUDIO_SPEC, HOSTS
from voxforge.models import DialogueTurn

_ABBREV_GUARD = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Mt|vs|etc|approx|ca|c|No|Fig|Gen|Gov|Sen|Rep)\.",
    flags=re.IGNORECASE,
)
_CONTINUATION_START = re.compile(
    r"^(And|But|So|Or|Yet|Because|Plus|Still|Even|While)\b",
    flags=re.IGNORECASE,
)


def _slerp_voice_vectors(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Row-wise Spherical Linear Interpolation (SLERP) preserving StyleTTS2 manifold norm."""
    a = np.asarray(v0, dtype=np.float64)
    b = np.asarray(v1, dtype=np.float64)
    orig_shape = a.shape
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])

    norm_a = np.linalg.norm(a_flat, axis=-1, keepdims=True) + 1e-12
    norm_b = np.linalg.norm(b_flat, axis=-1, keepdims=True) + 1e-12
    u_a = a_flat / norm_a
    u_b = b_flat / norm_b

    dot = np.clip(np.sum(u_a * u_b, axis=-1, keepdims=True), -0.999999, 0.999999)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    use_lerp = sin_omega < 1e-5
    w0 = np.where(use_lerp, 1.0 - t, np.sin((1.0 - t) * omega) / np.maximum(sin_omega, 1e-8))
    w1 = np.where(use_lerp, t, np.sin(t * omega) / np.maximum(sin_omega, 1e-8))

    # Geodesic direction + interpolated hypersphere radius
    target_norm = (1.0 - t) * norm_a + t * norm_b
    blended = (w0 * u_a + w1 * u_b) * target_norm
    return blended.reshape(orig_shape).astype(np.float32)


def _resample_to_broadcast(samples: np.ndarray, orig_sr: int, target_sr: int = AUDIO_SPEC.sample_rate) -> np.ndarray:
    """High-precision polyphase resampling to 48,000 Hz mono float32."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if orig_sr == target_sr or len(x) == 0:
        return x
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(x, up, down).astype(np.float32)


def _tighten_internal_stray_pauses(samples: np.ndarray, sr: int = AUDIO_SPEC.sample_rate) -> np.ndarray:
    """Cap unprompted mid-sentence TTS silences (>320ms) down to a natural 155ms clause breath."""
    if len(samples) < int(sr * 0.8):
        return samples
    hop = max(1, sr // 100)  # 10ms frames
    n_frames = len(samples) // hop
    framed = samples[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    floor = max(10 ** (-47.0 / 20.0), float(np.max(rms)) * 0.015)
    silent = rms < floor

    out_parts: list[np.ndarray] = []
    idx = 0
    max_keep = 16  # 160ms
    stray_min = 33  # 330ms
    while idx < n_frames:
        if not silent[idx]:
            start = idx
            while idx < n_frames and not silent[idx]:
                idx += 1
            out_parts.append(samples[start * hop : idx * hop])
        else:
            start = idx
            while idx < n_frames and silent[idx]:
                idx += 1
            run = idx - start
            if run >= stray_min and 0 < start and idx < n_frames:
                # Smoothly retain 155ms natural breath instead of jarring mid-sentence dead gap
                half = (max_keep * hop) // 2
                left = samples[start * hop : start * hop + half].copy()
                right = samples[idx * hop - half : idx * hop].copy()
                fade = np.linspace(1.0, 0.0, len(left), dtype=np.float32)
                out_parts.append(left * fade)
                out_parts.append(right * fade[::-1])
            else:
                out_parts.append(samples[start * hop : idx * hop])

    if n_frames * hop < len(samples):
        out_parts.append(samples[n_frames * hop :])
    return np.concatenate(out_parts) if out_parts else samples


def _trim_and_declick(samples: np.ndarray, sr: int = AUDIO_SPEC.sample_rate) -> np.ndarray:
    """Trim leading/trailing TTS dead air while keeping natural breath onset/release."""
    if len(samples) == 0:
        return np.zeros(int(sr * 0.25), dtype=np.float32)

    hop = max(1, sr // 100)
    n_frames = len(samples) // hop
    if n_frames > 4:
        framed = samples[: n_frames * hop].reshape(n_frames, hop)
        rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
        thresh = max(10 ** (-48.0 / 20.0), float(np.max(rms)) * 0.014)
        active = np.where(rms > thresh)[0]
        if len(active) > 0:
            pad_lead = int(0.026 * sr)
            pad_trail = int(0.062 * sr)
            start = max(0, int(active[0]) * hop - pad_lead)
            end = min(len(samples), int(active[-1] + 1) * hop + pad_trail)
            samples = samples[start:end]

    fade_n = min(int(0.009 * sr), len(samples) // 4)
    if fade_n > 4:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, fade_n, dtype=np.float32)))
        samples = samples.copy()
        samples[:fade_n] *= ramp
        samples[-fade_n:] *= ramp[::-1]

    return samples


def _split_spoken_sentences(text: str, max_words_per_group: int = 26) -> list[str]:
    """Split text into abbreviation-safe 1-to-2 sentence spoken thoughts (< 220 chars)."""
    protected = _ABBREV_GUARD.sub(lambda m: f"{m.group(1)}<DOT>", text.strip())
    protected = re.sub(r"\b([A-Z])\.\s*(?=[A-Z])", r"\1<DOT> ", protected)
    raw_sents = [
        s.replace("<DOT>", ".").strip()
        for s in re.split(r"(?<=[.!?])\s+", protected)
        if s.strip()
    ]
    if not raw_sents:
        return [text.strip()]

    # Glue ultra-short 1-3 word interjections ("Exactly.", "Think about that.") only if 1-2 words so they don't clip,
    # while keeping 4-26 word sentences distinct so Kokoro resets intonation cleanly at periods!
    groups: list[str] = []
    i = 0
    while i < len(raw_sents):
        s = raw_sents[i]
        wc = len(s.split())
        if wc <= 3 and i + 1 < len(raw_sents) and (wc + len(raw_sents[i + 1].split())) <= max_words_per_group:
            # Join 1-3 word micro-affirmation to next sentence with an expressive dash beat
            stem = s.rstrip(".")
            groups.append(f"{stem} - {raw_sents[i + 1]}" if s.endswith(".") else f"{s} {raw_sents[i + 1]}")
            i += 2
        else:
            groups.append(s)
            i += 1

    return [g.strip() for g in groups if g.strip()]


def _compute_context_pause_sec(prev_sent: str, next_sent: str, seed_idx: int) -> float:
    """Compute context-sensitive human pause between spoken sentences (never robotic metronome)."""
    wc = len(prev_sent.split())
    if prev_sent.endswith("?"):
        base = 0.285
    elif prev_sent.endswith("!"):
        base = 0.220
    elif wc <= 8:
        # Let short punchy hooks ("Picture that scale.", "Here's the catch.") land!
        base = 0.230
    elif wc >= 22:
        # Extra comprehension breath after a rich sentence
        base = 0.245
    else:
        base = 0.185

    if _CONTINUATION_START.match(next_sent):
        base -= 0.040

    # Deterministic +-22ms organic human timing variation
    jitter = (((seed_idx * 37 + len(prev_sent) * 13) % 45) - 22) / 1000.0
    return max(0.115, min(0.360, base + jitter))


class StudioVoiceSynthesizer:
    """Stateful neural TTS synthesizer caching FP32 ONNX sessions and SLERP-blended voice styles."""

    def __init__(self, env: StudioEnvironment, cache_dir: Path) -> None:
        self.env = env
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._kokoro: Any = None
        self._kokoro_styles: dict[str, np.ndarray] = {}
        self._piper_voices: dict[str, Any] = {}
        self.active_backend = env.tts_backend

    def _get_kokoro(self) -> Any:
        if self._kokoro is None:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro

            if self.env.kokoro_model_path is None or self.env.kokoro_voices_path is None:
                raise RuntimeError("Kokoro ONNX model paths are not configured")

            sess_opts = ort.SessionOptions()
            sess_opts.enable_cpu_mem_arena = False
            sess_opts.enable_mem_pattern = False
            sess_opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
            sess_opts.inter_op_num_threads = 1

            session = ort.InferenceSession(
                str(self.env.kokoro_model_path),
                sess_options=sess_opts,
                providers=[("CPUExecutionProvider", {"use_arena": 0})],
            )
            self._kokoro = Kokoro.from_session(session, str(self.env.kokoro_voices_path))
        return self._kokoro

    def _resolve_kokoro_voice_style(self, voice_spec: str) -> Any:
        """Resolve a voice name or multi-voice geodesic SLERP blend on the StyleTTS2 hypersphere."""
        if voice_spec in self._kokoro_styles:
            return self._kokoro_styles[voice_spec]

        kokoro = self._get_kokoro()
        if ":" not in voice_spec and "," not in voice_spec:
            style = kokoro.get_voice_style(voice_spec.strip()).astype(np.float32)
            self._kokoro_styles[voice_spec] = style
            return style

        weighted_vecs: list[tuple[np.ndarray, float]] = []
        for part in voice_spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                vname, wstr = part.split(":", 1)
                w = float(wstr)
            else:
                vname, w = part, 1.0
            if w > 0:
                weighted_vecs.append((kokoro.get_voice_style(vname.strip()).astype(np.float32), w))

        if not weighted_vecs:
            style = kokoro.get_voice_style("af_heart").astype(np.float32)
        else:
            # Iterative spherical SLERP across weighted voice embeddings
            acc_vec, acc_w = weighted_vecs[0]
            for vec, w in weighted_vecs[1:]:
                t = w / (acc_w + w)
                acc_vec = _slerp_voice_vectors(acc_vec, vec, t)
                acc_w += w
            style = acc_vec.astype(np.float32)

        self._kokoro_styles[voice_spec] = style
        return style

    def _get_piper_voice(self, speaker_id: str) -> Any:
        if speaker_id not in self._piper_voices:
            import json as _json
            import onnxruntime
            from piper import PiperVoice
            from piper.config import PiperConfig
            from piper.voice import ESPEAK_DATA_DIR

            onnx_path = (
                self.env.piper_male_onnx
                if speaker_id == "HOST_A"
                else self.env.piper_female_onnx
            )
            if onnx_path is None or not onnx_path.exists():
                raise RuntimeError(f"Piper ONNX model not found for {speaker_id}")

            with open(f"{onnx_path}.json", "r", encoding="utf-8") as cf:
                cfg_dict = _json.load(cf)

            sess_opts = onnxruntime.SessionOptions()
            sess_opts.enable_cpu_mem_arena = False
            sess_opts.enable_mem_pattern = False
            sess_opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 2))
            sess_opts.inter_op_num_threads = 1

            session = onnxruntime.InferenceSession(
                str(onnx_path),
                sess_options=sess_opts,
                providers=[("CPUExecutionProvider", {"use_arena": 0})],
            )
            self._piper_voices[speaker_id] = PiperVoice(
                config=PiperConfig.from_dict(cfg_dict),
                session=session,
                espeak_data_dir=Path(ESPEAK_DATA_DIR),
                download_dir=onnx_path.parent,
            )
        return self._piper_voices[speaker_id]

    def release_memory(self) -> None:
        """Release neural TTS inference sessions prior to heavy audio/video encoding."""
        import gc

        self._piper_voices.clear()
        self._kokoro_styles.clear()
        self._kokoro = None
        gc.collect()

    def _synth_kokoro(self, text: str, speaker_id: str, speed_scale: float = 1.0, turn_id: int = 0) -> np.ndarray:
        persona = HOSTS[speaker_id]
        kokoro = self._get_kokoro()
        voice_style = self._resolve_kokoro_voice_style(persona.kokoro_voice)
        eff_speed = max(0.93, min(1.08, persona.speaking_speed * speed_scale))

        groups = _split_spoken_sentences(text)
        segments_48k: list[np.ndarray] = []
        sr_out = AUDIO_SPEC.sample_rate

        for idx, grp in enumerate(groups):
            wc = len(grp.split())
            # Natural human tempo variation: slightly brisker on questions/short hooks, slightly more measured on dense imagery
            if grp.endswith(("?", "!")):
                cadence_nudge = 1.02
            elif wc >= 22:
                cadence_nudge = 0.975
            else:
                cadence_nudge = 1.0

            samples = np.zeros(0, dtype=np.float32)
            sr = 24000
            for _attempt in range(3):
                raw_s, sr = kokoro.create(
                    grp,
                    voice=voice_style,
                    speed=eff_speed * cadence_nudge,
                    lang="en-us",
                )
                arr_s = np.asarray(raw_s, dtype=np.float32)
                if len(arr_s) > int(sr * 0.20) and not np.isnan(arr_s).any() and float(np.sqrt(np.mean(arr_s * arr_s))) > 0.004:
                    samples = arr_s
                    break

            seg_48k = _resample_to_broadcast(samples, int(sr), sr_out)
            seg_48k = _tighten_internal_stray_pauses(seg_48k, sr_out)
            seg_48k = _trim_and_declick(seg_48k, sr_out)
            segments_48k.append(seg_48k)

            if idx < len(groups) - 1:
                gap_sec = _compute_context_pause_sec(grp, groups[idx + 1], turn_id * 11 + idx)
                segments_48k.append(np.zeros(int(sr_out * gap_sec), dtype=np.float32))

        return np.concatenate(segments_48k) if segments_48k else np.zeros(int(sr_out * 0.25), dtype=np.float32)

    def _synth_piper(self, text: str, speaker_id: str, speed_scale: float = 1.0) -> np.ndarray:
        import io
        from piper.config import SynthesisConfig

        persona = HOSTS[speaker_id]
        voice = self._get_piper_voice(speaker_id)
        length_scale = 1.0 / max(0.90, min(1.15, persona.speaking_speed * speed_scale))
        syn_cfg = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=0.667,
            noise_w_scale=0.8,
        )
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf, syn_config=syn_cfg)
        buf.seek(0)
        data, sr = sf.read(buf, dtype="float32")
        return _resample_to_broadcast(data, int(sr))

    def _synth_edge(self, text: str, speaker_id: str, tmp_mp3: Path) -> np.ndarray:
        import edge_tts

        persona = HOSTS[speaker_id]

        async def _run() -> None:
            comm = edge_tts.Communicate(text, persona.edge_voice)
            await comm.save(str(tmp_mp3))

        asyncio.run(_run())
        container = av.open(str(tmp_mp3))
        frames: list[np.ndarray] = []
        sr = 24000
        for frame in container.decode(audio=0):
            sr = frame.sample_rate
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            frames.append(arr.astype(np.float32))
        container.close()
        tmp_mp3.unlink(missing_ok=True)
        if not frames:
            raise RuntimeError("Empty audio returned by edge-tts")
        joined = np.concatenate(frames)
        if np.max(np.abs(joined)) > 2.0:
            joined = joined / 32768.0
        return _resample_to_broadcast(joined, sr)

    def synthesize_turn(self, turn: DialogueTurn, speed_scale: float = 1.0, force_regen: bool = False) -> Path:
        """Synthesize a single dialogue turn to 48kHz WAV using pure FP32 neural synthesis (zero WSOLA stretch)."""
        persona = HOSTS[turn.speaker]
        model_name = self.env.kokoro_model_path.name if (self.active_backend == "kokoro" and self.env.kokoro_model_path) else "default"
        voice_tag = f"{model_name}:{persona.kokoro_voice}" if self.active_backend == "kokoro" else persona.piper_voice
        key = hashlib.sha1(
            f"v4-slerp:{self.active_backend}:{turn.speaker}:{voice_tag}:{speed_scale:.3f}:{turn.spoken_text}".encode()
        ).hexdigest()[:14]
        out_wav = self.cache_dir / f"utt_{key}.wav"

        if out_wav.exists() and not force_regen and out_wav.stat().st_size > 4096:
            info = sf.info(str(out_wav))
            turn.audio_path = str(out_wav)
            turn.duration_sec = float(info.duration)
            return out_wav

        samples: np.ndarray | None = None
        if self.active_backend == "kokoro":
            try:
                samples = self._synth_kokoro(
                    turn.spoken_text, turn.speaker, speed_scale=speed_scale, turn_id=turn.turn_id
                )
            except Exception:
                if self.env.piper_male_onnx is None:
                    from voxforge.bootstrap import ensure_piper_hifi_voices

                    pp = ensure_piper_hifi_voices()
                    if pp is not None:
                        self.env.piper_male_onnx, self.env.piper_female_onnx = pp
                self.active_backend = "piper"

        if samples is None and self.active_backend == "edge":
            try:
                samples = self._synth_edge(turn.spoken_text, turn.speaker, out_wav.with_suffix(".mp3"))
            except Exception:
                self.active_backend = "piper"

        if samples is None:
            samples = self._synth_piper(turn.spoken_text, turn.speaker, speed_scale=speed_scale)

        cleaned = _trim_and_declick(samples, AUDIO_SPEC.sample_rate)
        sf.write(str(out_wav), cleaned, AUDIO_SPEC.sample_rate, subtype="PCM_16")
        turn.audio_path = str(out_wav)
        turn.duration_sec = round(len(cleaned) / AUDIO_SPEC.sample_rate, 4)
        return out_wav
