"""Zero-API-Key Autonomous Studio Bootstrapper.

Automatically provisions FFmpeg, studio neural TTS models (Kokoro-82M / Piper Hi-Fi ONNX),
and optional local open-weight GGUF LLMs using only standard internet access.
"""

from __future__ import annotations

import io
import shutil
import ssl
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from voxforge.config import (
    GGUF_LLM_URLS,
    HTTP_USER_AGENT,
    KOKORO_CACHE_DIR,
    KOKORO_FP32_TARBALL_MIRROR,
    KOKORO_INT8_TARBALL_MIRROR,
    KOKORO_MODEL_URLS,
    KOKORO_VOICES_TARBALL_MIRROR,
    LLM_CACHE_DIR,
    PIPER_HF_BASE,
    PIPER_TARBALL_MIRROR,
    VOICES_CACHE_DIR,
    ensure_cache_dirs,
)


@dataclass
class StudioEnvironment:
    """Resolved local paths and active zero-key backends."""

    ffmpeg_bin: str
    tts_backend: str  # "kokoro" | "piper" | "edge"
    kokoro_model_path: Path | None
    kokoro_voices_path: Path | None
    piper_male_onnx: Path | None
    piper_female_onnx: Path | None
    llm_backend: str  # "ollama" | "llamacpp" | "corpus_showrunner"
    llm_model_id_or_path: str


def resolve_ffmpeg_binary() -> str:
    """Locate a full-featured FFmpeg binary (preferring static build with libass + libx264 + loudnorm)."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass

    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    raise RuntimeError(
        "FFmpeg could not be resolved. Install `imageio-ffmpeg` via pip or `ffmpeg` on PATH."
    )


def get_ssl_verify() -> ssl.SSLContext | bool:
    """Return system-CA SSLContext when available so Linux proxy roots validate cleanly."""
    sys_ca = Path("/etc/ssl/certs/ca-certificates.crt")
    if sys_ca.exists():
        try:
            return ssl.create_default_context(cafile=str(sys_ca))
        except Exception:
            pass
    return True


def _download_with_fallback(urls: list[str], dest: Path, min_bytes: int = 1024, timeout: float = 45.0) -> bool:
    """Download a file atomically from the first reachable URL without API keys."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": HTTP_USER_AGENT}

    for url in urls:
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers, verify=get_ssl_verify()) as client:
                with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        continue
                    ctype = resp.headers.get("content-type", "")
                    if "text/html" in ctype and not dest.name.endswith(".html"):
                        continue
                    total = 0
                    with open(tmp_path, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=262_144):
                            f.write(chunk)
                            total += len(chunk)
                    if total >= min_bytes:
                        tmp_path.replace(dest)
                        return True
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            continue

    tmp_path.unlink(missing_ok=True)
    return False


def ensure_piper_hifi_voices() -> tuple[Path, Path] | None:
    """Ensure Hi-Fi Neural Studio Male & Female Piper ONNX voices exist in cache."""
    ensure_cache_dirs()
    male_onnx = VOICES_CACHE_DIR / "en_US-hfc_male-medium.onnx"
    male_json = VOICES_CACHE_DIR / "en_US-hfc_male-medium.onnx.json"
    female_onnx = VOICES_CACHE_DIR / "en_US-hfc_female-medium.onnx"
    female_json = VOICES_CACHE_DIR / "en_US-hfc_female-medium.onnx.json"

    if (
        male_onnx.exists()
        and male_onnx.stat().st_size > 20_000_000
        and female_onnx.exists()
        and female_onnx.stat().st_size > 20_000_000
        and male_json.exists()
        and female_json.exists()
    ):
        return male_onnx, female_onnx

    # Attempt 1: Direct HuggingFace Piper voices repository
    hf_ok = (
        _download_with_fallback(
            [f"{PIPER_HF_BASE}/hfc_male/medium/en_US-hfc_male-medium.onnx"],
            male_onnx,
            min_bytes=20_000_000,
            timeout=25.0,
        )
        and _download_with_fallback(
            [f"{PIPER_HF_BASE}/hfc_male/medium/en_US-hfc_male-medium.onnx.json"],
            male_json,
            min_bytes=500,
            timeout=15.0,
        )
        and _download_with_fallback(
            [f"{PIPER_HF_BASE}/hfc_female/medium/en_US-hfc_female-medium.onnx"],
            female_onnx,
            min_bytes=20_000_000,
            timeout=25.0,
        )
        and _download_with_fallback(
            [f"{PIPER_HF_BASE}/hfc_female/medium/en_US-hfc_female-medium.onnx.json"],
            female_json,
            min_bytes=500,
            timeout=15.0,
        )
    )
    if hf_ok:
        return male_onnx, female_onnx

    # Attempt 2: GitHub codeload tarball mirror (works even when only github.com/codeload is reachable)
    try:
        headers = {"User-Agent": HTTP_USER_AGENT}
        with httpx.Client(follow_redirects=True, timeout=180.0, headers=headers, verify=get_ssl_verify()) as client:
            resp = client.get(PIPER_TARBALL_MIRROR)
            if resp.status_code == 200 and len(resp.content) > 40_000_000:
                wanted = {
                    "en_US-hfc_male-medium.onnx",
                    "en_US-hfc_male-medium.onnx.json",
                    "en_US-hfc_female-medium.onnx",
                    "en_US-hfc_female-medium.onnx.json",
                }
                with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
                    for member in tf.getmembers():
                        base = Path(member.name).name
                        if member.isfile() and base in wanted:
                            extracted = tf.extractfile(member)
                            if extracted is not None:
                                out_file = VOICES_CACHE_DIR / base
                                with open(out_file, "wb") as out:
                                    shutil.copyfileobj(extracted, out)
    except Exception:
        pass

    if male_onnx.exists() and female_onnx.exists() and male_onnx.stat().st_size > 20_000_000:
        return male_onnx, female_onnx
    return None


def try_ensure_kokoro_models(timeout: float = 20.0) -> tuple[Path, Path] | None:
    """Ensure full unquantized FP32 Kokoro-82M v1.0 ONNX (or INT8 fallback) + 54-voice style bundle exist locally."""
    ensure_cache_dirs()
    fp32_path = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
    int8_path = KOKORO_CACHE_DIR / "kokoro-v1.0.int8.onnx"
    voices_path = KOKORO_CACHE_DIR / "voices-v1.0.bin"

    if voices_path.exists() and voices_path.stat().st_size > 20_000_000:
        if fp32_path.exists() and fp32_path.stat().st_size > 300_000_000:
            return fp32_path, voices_path
        if int8_path.exists() and int8_path.stat().st_size > 80_000_000:
            # Prefer upgrading to FP32 if missing, otherwise use INT8 below
            pass

    headers = {"User-Agent": HTTP_USER_AGENT}

    # 1. Ensure voices-v1.0.bin (direct URLs first, then GitHub codeload tarball mirror)
    if not (voices_path.exists() and voices_path.stat().st_size > 20_000_000):
        if not _download_with_fallback(
            KOKORO_MODEL_URLS["voices-v1.0.bin"], voices_path, min_bytes=20_000_000, timeout=timeout
        ):
            try:
                with httpx.Client(follow_redirects=True, timeout=120.0, headers=headers, verify=get_ssl_verify()) as client:
                    resp = client.get(KOKORO_VOICES_TARBALL_MIRROR)
                    if resp.status_code == 200 and len(resp.content) > 20_000_000:
                        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
                            for m in tf.getmembers():
                                if m.isfile() and m.name.endswith("voices-v1.0.bin"):
                                    ext = tf.extractfile(m)
                                    if ext is not None:
                                        voices_path.write_bytes(ext.read())
                                        break
            except Exception:
                pass

    if not (voices_path.exists() and voices_path.stat().st_size > 20_000_000):
        return None

    # 2. Primary: Full unquantized 32-bit FP32 Kokoro-82M v1.0 ONNX (zero iSTFT quantization harmonics, fastest AVX2 RTF)
    if fp32_path.exists() and fp32_path.stat().st_size > 300_000_000:
        return fp32_path, voices_path

    if _download_with_fallback(
        KOKORO_MODEL_URLS["kokoro-v1.0.onnx"], fp32_path, min_bytes=300_000_000, timeout=timeout
    ):
        return fp32_path, voices_path

    try:
        with httpx.Client(follow_redirects=True, timeout=180.0, headers=headers, verify=get_ssl_verify()) as client:
            resp = client.get(KOKORO_FP32_TARBALL_MIRROR)
            if resp.status_code == 200 and len(resp.content) > 250_000_000:
                zparts: dict[str, bytes] = {}
                with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
                    for m in tf.getmembers():
                        bn = Path(m.name).name
                        if m.isfile() and bn.startswith("kokoro-v1.0.zip."):
                            ext = tf.extractfile(m)
                            if ext is not None:
                                zparts[bn] = ext.read()
                if zparts:
                    joined_zip = b"".join(zparts[k] for k in sorted(zparts.keys()))
                    with zipfile.ZipFile(io.BytesIO(joined_zip)) as zf:
                        for zi in zf.infolist():
                            if zi.filename.endswith(".onnx") and zi.file_size > 300_000_000:
                                fp32_path.write_bytes(zf.read(zi.filename))
                                return fp32_path, voices_path
    except Exception:
        pass

    # 3. Fallback: Kokoro-82M v1.0 INT8 ONNX
    if int8_path.exists() and int8_path.stat().st_size > 80_000_000:
        return int8_path, voices_path

    if _download_with_fallback(
        KOKORO_MODEL_URLS["kokoro-v1.0.int8.onnx"], int8_path, min_bytes=80_000_000, timeout=timeout
    ):
        return int8_path, voices_path

    try:
        with httpx.Client(follow_redirects=True, timeout=150.0, headers=headers, verify=get_ssl_verify()) as client:
            resp = client.get(KOKORO_INT8_TARBALL_MIRROR)
            if resp.status_code == 200 and len(resp.content) > 60_000_000:
                with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
                    for m in tf.getmembers():
                        if m.isfile() and m.name.endswith("kokoro-v1.0.int8.onnx"):
                            ext = tf.extractfile(m)
                            if ext is not None:
                                int8_path.write_bytes(ext.read())
                                if int8_path.stat().st_size > 80_000_000:
                                    return int8_path, voices_path
    except Exception:
        pass

    return None


def detect_or_bootstrap_llm(prefer_local_gguf: bool = True) -> tuple[str, str]:
    """Detect zero-key LLM runtime: Ollama -> llama-cpp-python GGUF -> Corpus Showrunner."""
    # Tier 1: Check local Ollama server (http://127.0.0.1:11434)
    try:
        resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=1.5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                return "ollama", models[0]["name"]
            if shutil.which("ollama"):
                subprocess.run(
                    ["ollama", "pull", "qwen2.5:3b"],
                    check=False,
                    timeout=180,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                resp2 = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
                models2 = resp2.json().get("models", [])
                if models2:
                    return "ollama", models2[0]["name"]
    except Exception:
        pass

    # Tier 2: Check llama-cpp-python + cached or downloadable GGUF instruct weights
    if prefer_local_gguf:
        try:
            import llama_cpp  # noqa: F401

            ensure_cache_dirs()
            for fname, url in GGUF_LLM_URLS:
                fpath = LLM_CACHE_DIR / fname
                if fpath.exists() and fpath.stat().st_size > 300_000_000:
                    return "llamacpp", str(fpath)
            # Probe HuggingFace reachability before downloading ~980MB GGUF
            fname, url = GGUF_LLM_URLS[0]
            fpath = LLM_CACHE_DIR / fname
            if _download_with_fallback([url], fpath, min_bytes=300_000_000, timeout=120.0):
                return "llamacpp", str(fpath)
        except Exception:
            pass

    # Tier 3: Deterministic Corpus-Grounded Documentary Showrunner (always online/offline ready)
    return "corpus_showrunner", "voxforge-documentary-discourse-v1"


def bootstrap_studio(preferred_tts: str = "auto") -> StudioEnvironment:
    """Provision all tools and models required for studio production."""
    ensure_cache_dirs()
    ffmpeg_bin = resolve_ffmpeg_binary()

    kokoro_paths: tuple[Path, Path] | None = None
    piper_paths: tuple[Path, Path] | None = None
    tts_backend = "piper"

    if preferred_tts in ("auto", "kokoro"):
        kokoro_paths = try_ensure_kokoro_models(timeout=8.0)
        if kokoro_paths is not None:
            tts_backend = "kokoro"

    if kokoro_paths is None and preferred_tts in ("auto", "piper", "kokoro"):
        piper_paths = ensure_piper_hifi_voices()
        if piper_paths is not None:
            tts_backend = "piper"
        else:
            tts_backend = "edge"
    elif preferred_tts == "edge":
        tts_backend = "edge"
        piper_paths = ensure_piper_hifi_voices()

    llm_backend, llm_model = detect_or_bootstrap_llm()

    return StudioEnvironment(
        ffmpeg_bin=ffmpeg_bin,
        tts_backend=tts_backend,
        kokoro_model_path=kokoro_paths[0] if kokoro_paths else None,
        kokoro_voices_path=kokoro_paths[1] if kokoro_paths else None,
        piper_male_onnx=piper_paths[0] if piper_paths else None,
        piper_female_onnx=piper_paths[1] if piper_paths else None,
        llm_backend=llm_backend,
        llm_model_id_or_path=llm_model,
    )
