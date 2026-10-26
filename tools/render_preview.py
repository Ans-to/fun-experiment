"""Render a short picture-track preview from an already-produced episode.

Re-renders the video from the episode's existing `episode_script.json` and
`documentary_audio.mp3`, truncated to the first N seconds. No TTS, no research,
no LLM — it exists so a visual fix can be checked without a full-length render.

    .venv/bin/python tools/render_preview.py output_episodes/<episode> --seconds 60
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxforge.bootstrap import StudioEnvironment, resolve_ffmpeg_binary  # noqa: E402
from voxforge.models import AudioMasterMeta, EpisodeScript  # noqa: E402
from voxforge.video.renderer import render_episode_video  # noqa: E402


def _preview_env() -> StudioEnvironment:
    """Minimal environment: no TTS/LLM model downloads are needed to re-render pictures."""
    return StudioEnvironment(
        ffmpeg_bin=resolve_ffmpeg_binary(),
        tts_backend="preview",
        kokoro_model_path=None,
        kokoro_voices_path=None,
        piper_male_onnx=None,
        piper_female_onnx=None,
        llm_backend="corpus_showrunner",
        llm_model_id_or_path="offline",
    )


# ---------------------------------------------------------------------------
# generic dataclass rehydration (episode_script.json -> live model objects)
# ---------------------------------------------------------------------------


def _build(cls: type, data: object) -> object:
    """Recursively rebuild a (possibly nested) dataclass tree from plain JSON data."""
    origin = typing.get_origin(cls)
    if origin is list:
        item_cls = typing.get_args(cls)[0]
        return [_build(item_cls, item) for item in (data or [])]
    if origin is tuple:
        args = typing.get_args(cls)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build(args[0], item) for item in (data or []))
        return tuple(_build(a, d) for a, d in zip(args, data or ()))
    if origin is dict:
        k_cls, v_cls = typing.get_args(cls)
        return {_build(k_cls, k): _build(v_cls, v) for k, v in (data or {}).items()}
    if origin is not None:  # Optional[X] / Union[...]
        for arg in typing.get_args(cls):
            if arg is type(None):
                continue
            if data is None:
                return None
            return _build(arg, data)
    if cls is Path:
        return Path(data) if data else None
    if not dataclasses.is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        return data
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, object] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        kwargs[f.name] = _build(hints.get(f.name, object), data[f.name])
    return cls(**kwargs)


def load_script(path: Path) -> EpisodeScript:
    data = json.loads(path.read_text(encoding="utf-8"))
    script = _build(EpisodeScript, data)
    assert isinstance(script, EpisodeScript)
    return script


# ---------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------


def truncate_script(script: EpisodeScript, seconds: float) -> EpisodeScript:
    """Keep only the turns that fall inside the first `seconds` of the film."""
    kept_chapters = []
    for ch in script.chapters:
        turns = []
        for t in ch.turns:
            if t.start_sec >= seconds:
                continue
            if t.end_sec > seconds:
                t.end_sec = float(seconds)
            t.duration_sec = max(0.1, t.end_sec - t.start_sec)
            turns.append(t)
        if not turns:
            continue
        kept_chapters.append(dataclasses.replace(ch, turns=turns))
    if not kept_chapters:
        raise SystemExit(f"no narration falls inside the first {seconds}s")
    return dataclasses.replace(
        script,
        chapters=kept_chapters,
        outro_duration_sec=0.0,
    )


def trim_audio(src: Path, dst: Path, seconds: float, ffmpeg: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error",
            "-i", str(src),
            "-t", f"{seconds:.3f}",
            "-ar", "48000", "-ac", "2",
            str(dst),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episode_dir", type=Path)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out-name", default="preview.mp4")
    args = ap.parse_args(argv)

    ep_dir: Path = args.episode_dir
    seconds: float = args.seconds

    env = _preview_env()
    script = truncate_script(load_script(ep_dir / "episode_script.json"), seconds)

    audio_src = ep_dir / "documentary_audio.mp3"
    if not audio_src.exists():
        raise SystemExit(f"missing {audio_src}")

    work_dir = ep_dir / f"_preview_work_{int(seconds)}"
    out_dir = ep_dir / f"_preview_out_{int(seconds)}"
    wav = work_dir / "preview_audio.wav"
    trim_audio(audio_src, wav, seconds, env.ffmpeg_bin)

    audio_meta = AudioMasterMeta(
        master_wav_path=str(wav),
        master_mp3_path=str(audio_src),
        duration_sec=float(seconds),
        sample_rate=48000,
        channels=2,
        integrated_lufs=-16.0,
        true_peak_dbtp=-1.5,
        loudness_range_lu=6.0,
        max_silence_gap_sec=0.4,
        tts_backend_used="preview",
        utterances_synthesized=0,
    )

    print(f"[PREVIEW] rendering first {seconds:.0f}s of {ep_dir.name}", flush=True)
    mp4, srt, vtt = render_episode_video(
        script=script,
        audio_meta=audio_meta,
        env=env,
        work_dir=work_dir,
        output_dir=out_dir,
        progress_cb=lambda done, total: print(
            f"[PREVIEW] segment {done}/{total}", flush=True
        ),
    )

    final_mp4 = ep_dir / args.out_name
    shutil.move(str(mp4), str(final_mp4))
    for extra in (srt, vtt):
        if extra and Path(extra).exists():
            shutil.move(str(extra), str(ep_dir / f"preview_{Path(extra).name}"))
    shutil.rmtree(out_dir, ignore_errors=True)

    print(f"[PREVIEW] wrote {final_mp4} ({final_mp4.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
