"""Central broadcast engineering specifications, paths, and studio configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
ASSETS_DIR = PACKAGE_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
STUDIO_DIR = ASSETS_DIR / "studio"
STUDIO_RIG_DIR = ASSETS_DIR / "studio_rig"
STUDIO_SET_BG_PATH = ASSETS_DIR / "studio_set_bg.jpg"
BROLL_SEED_DIR = ASSETS_DIR / "broll_seed"
WEB_STATIC_DIR = PACKAGE_ROOT / "web" / "static"

CACHE_ROOT = Path(os.environ.get("VOXFORGE_CACHE_DIR", Path.home() / ".cache" / "voxforge"))
VOICES_CACHE_DIR = CACHE_ROOT / "voices"
KOKORO_CACHE_DIR = CACHE_ROOT / "kokoro"
LLM_CACHE_DIR = CACHE_ROOT / "llm"
RESEARCH_CACHE_DIR = CACHE_ROOT / "research"
MEDIA_CACHE_DIR = CACHE_ROOT / "media"
MEDIA_SEED_DIR = CACHE_ROOT / "media_seed"

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output_episodes"

# Compliant descriptive User-Agent required by Wikimedia & Openverse keyless APIs (200 req/min tier)
HTTP_USER_AGENT = (
    "VoxForgeDocumentary/2.0 (Autonomous Open-Source Documentary Film Engine; "
    "https://github.com/Ans-to/fun-experiment) Python-httpx"
)


@dataclass(frozen=True)
class BroadcastAudioSpec:
    """ITU-R BS.1770-4 / Apple Podcasts / Spotify Video Podcast mastering targets."""

    sample_rate: int = 48_000
    channels: int = 2
    target_lufs: float = -16.0
    lufs_tolerance: float = 0.85
    max_true_peak_dbtp: float = -1.0
    target_true_peak_dbtp: float = -1.2
    min_lra_lu: float = 1.5
    max_lra_lu: float = 11.5
    max_dead_air_sec: float = 1.65
    room_tone_dbfs: float = -54.0
    music_bed_dbfs: float = -29.5
    sidechain_duck_db: float = -16.5
    host_a_pan: float = -0.07  # Intimate center-focused headphone stage (Host A slightly left)
    host_b_pan: float = +0.07  # Intimate center-focused headphone stage (Host B slightly right)


@dataclass(frozen=True)
class BroadcastVideoSpec:
    """1080p24 Rec.709 cinema broadcast delivery specifications."""

    width: int = 1920
    height: int = 1080
    fps: int = 24
    video_codec: str = "libx264"
    pixel_format: str = "yuv420p"
    colorspace: str = "bt709"
    color_primaries: str = "bt709"
    color_trc: str = "bt709"
    color_range: str = "tv"
    crf: int = 23
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    intro_bumper_sec: float = 4.2
    outro_card_sec: float = 5.0
    max_static_shot_sec: float = 16.0
    min_broll_width: int = 1024
    min_sharpness_laplacian: float = 32.0


@dataclass(frozen=True)
class HostPersona:
    """Studio host profile, vocal identity, and visual color accent."""

    speaker_id: str
    display_name: str
    short_name: str
    role_title: str
    accent_hex: str
    ass_bgr_color: str  # ASS subtitle &HBBGGRR& format
    kokoro_voice: str
    piper_voice: str
    edge_voice: str
    speaking_speed: float
    portrait_filename: str


HOST_A = HostPersona(
    speaker_id="HOST_A",
    display_name="Alex Vance",
    short_name="ALEX",
    role_title="Lead Narrator & Cultural Historian",
    accent_hex="#E5A93C",  # Warm Cinema Gold
    ass_bgr_color="&H3CA9E5&",
    kokoro_voice="am_michael:0.55,am_fenrir:0.30,am_puck:0.15",
    piper_voice="en_US-hfc_male-medium",
    edge_voice="en-US-AndrewMultilingualNeural",
    speaking_speed=0.97,
    portrait_filename="",
)

HOST_B = HostPersona(
    speaker_id="HOST_B",
    display_name="Dr. Maya Lin",
    short_name="MAYA",
    role_title="Field Archaeologist & Structural Specialist",
    accent_hex="#38BDF8",  # Broadcast Cyan
    ass_bgr_color="&HF8BD38&",
    kokoro_voice="af_heart:0.82,af_bella:0.18",
    piper_voice="en_US-hfc_female-medium",
    edge_voice="en-US-AvaMultilingualNeural",
    speaking_speed=0.98,
    portrait_filename="",
)

HOSTS = {"HOST_A": HOST_A, "HOST_B": HOST_B}

AUDIO_SPEC = BroadcastAudioSpec()
VIDEO_SPEC = BroadcastVideoSpec()

# Model download mirrors (100% Zero-API-Key open weights)
KOKORO_MODEL_URLS = {
    "kokoro-v1.0.onnx": [
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx",
    ],
    "kokoro-v1.0.fp16.onnx": [
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx",
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_fp16.onnx",
    ],
    "kokoro-v1.0.int8.onnx": [
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx",
    ],
    "voices-v1.0.bin": [
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    ],
}

# GitHub codeload tarball mirrors (reachable even on strict GitHub-only networks without LFS/Release redirects)
KOKORO_VOICES_TARBALL_MIRROR = "https://codeload.github.com/donggyun112/tts_server/tar.gz/refs/heads/main"
KOKORO_FP32_TARBALL_MIRROR = "https://codeload.github.com/feureau/Kokoro-TTS-GPU/tar.gz/refs/heads/main"
KOKORO_FP16_TARBALL_MIRROR = "https://codeload.github.com/mercadoa1234-arcDANTE/Kokoro-82B-TTS-Agent-Pack-CPU/tar.gz/refs/heads/main"
KOKORO_INT8_TARBALL_MIRROR = "https://codeload.github.com/Kemerd/jp-edge-tts/tar.gz/refs/heads/main"

PIPER_TARBALL_MIRROR = "https://codeload.github.com/heyseth/Piper_TTS/tar.gz/refs/heads/main"
PIPER_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US"

GGUF_LLM_URLS = [
    (
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ),
    (
        "qwen2.5-3b-instruct-q4_k_m.gguf",
        "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
    ),
]


def ensure_cache_dirs() -> None:
    """Ensure all cache and output directories exist."""
    for d in (
        CACHE_ROOT,
        VOICES_CACHE_DIR,
        KOKORO_CACHE_DIR,
        LLM_CACHE_DIR,
        RESEARCH_CACHE_DIR,
        MEDIA_CACHE_DIR,
        MEDIA_SEED_DIR,
        DEFAULT_OUTPUT_ROOT,
    ):
        d.mkdir(parents=True, exist_ok=True)
