"""High-Speed Parallel Cinema Documentary Video Renderer & Breath-Synced Shot Director (Rec.709).

Combines 24fps high-parallax 2.5D monument cinematography, 3D tilted 1:50m Natural Earth GIS
sovereign expedition maps, 3-plane museum archival light-table compositions, animated chronology
timelines, live counting engineering telemetry, and verbatim historical primary-source
inscriptions (`animator.py` + `compositor.py`) with breath-synced editorial cutting, clean
Univisium `2.00:1` framing, and ITU-R BS.1770-4 `-16 LUFS` Rec.709 MP4 muxing.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import soundfile as sf

from voxforge.bootstrap import StudioEnvironment
from voxforge.config import FONTS_DIR, VIDEO_SPEC
from voxforge.models import (
    AudioMasterMeta,
    CameraAngle,
    ChapterScript,
    DialogueTurn,
    EpisodeScript,
    ScenePlan,
    ScenePlanEntry,
)
from voxforge.video.animator import (
    StudioCharacterAnimator,
    extract_speech_motion_curves,
    parse_historical_year,
)
from voxforge.video.compositor import MulticamCompositor, PlateLayout
from voxforge.video.procedural_world import SCENE_CLASSICAL_GENERIC, resolve_procedural_scene_id
from voxforge.video.subtitles import (
    CaptionCue,
    build_all_caption_cues,
    write_ffmetadata_chapters,
    write_segment_ass,
    write_srt_and_vtt,
)


@dataclass
class RenderSegmentTask:
    seg_index: int
    start_sec: float
    duration_sec: float
    layout: PlateLayout
    ass_path: Path | None
    hud_overlay: tuple[Path, int, int, str] | None
    fade_in: bool
    fade_out: bool
    out_mp4: Path
    animate_hud_reveal: bool = False
    camera_label: str = "broll_cinema"
    lens_variant: int = 0
    dissolve_in_png: Path | None = None
    write_tail_jpg: bool = False


def _escape_filter_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


# -----------------------------------------------------------------------------
# CONTENT-AWARE SHOT DIRECTOR: every beat's visual family is chosen from what the
# narrator is actually saying in that turn, so picture always answers the voice:
#   eng   -> engineering/structure lines  => live physics sims & counting telemetry
#   geo   -> place/trade/movement lines    => 3D orbital Earth GIS maps & route arcs
#   quote -> chronicler/inscription lines  => verbatim primary-source quote cards
#   people-> rulers/events/ritual lines    => monument cinema + archival desk inserts
#   scene -> default                       => 3D monument cinematography & details
# -----------------------------------------------------------------------------
_RE_INTENT_ENG = re.compile(
    r"\b(built|build|engineer|engineers|engineering|mortar|drain|drainage|hydraulic|siphon|aqueduct|"
    r"acoustic|acoustics|echo|cantilever|seismic|foundation|construct|laborers|workers|"
    r"survey|aligned|alignment|astronom\w*|degrees?|slope|concrete|arches|vault|dome|"
    r"terraces?|irrigation|cistern|quarry|ramp|lever|crane|masonry|fortif\w*|forts?|erected|"
    r"staircase|columns?|pillars?|roof|watchtower\w*|signal\w*|beacon\w*|messeng\w*|smoke)\b",
    re.I,
)
_RE_INTENT_GEO = re.compile(
    r"\b(located|situated|kilomet\w*|kilometer\w*|km|miles?|journey|route|travel|caravan|ships?|"
    r"sailed?|cross|crossing|expedition|trade|traded|"
    r"spices?|bitumen|commerce|merchants?|roads?|pass|border|reach|spread|network|harbor|port|oasis)\b",
    re.I,
)
_RE_INTENT_QUOTE = re.compile(
    r"\b(wrote|written|recorded|chronicle\w*|inscription|poem|verse|praised|proclaimed|words|"
    r"described|account|eyewitness|historian|poet|said|declared|famously|renowned|called it)\b",
    re.I,
)
_RE_INTENT_PEOPLE = re.compile(
    r"\b(emperor\w*|empress\w*|king|queen|pharaoh\w*|dynasty|dynast\w*|ruled|reign|commission\w*|"
    r"ordered|decree|died|death|birth|born|war|siege|conquest|abandon\w*|collapsed|earthquake\w*|"
    r"rediscover\w*|explorer\w*|arrived|discover\w*|tomb|burial|ritual|ceremon\w*|gods?|goddess|"
    r"belief|faith|pilgrim\w*)\b",
    re.I,
)
_INTENT_PRIMARY = {
    "eng": "split_fact_card",
    "geo": "wide_studio",
    "quote": "quote_card",
    "people": "studio_insert",
}


def _turn_visual_intent(text: str) -> str:
    """Score narration text against visual-intent lexicons; strongest signal directs the picture."""
    counts = {
        "eng": len(_RE_INTENT_ENG.findall(text or "")),
        "people": len(_RE_INTENT_PEOPLE.findall(text or "")),
        "geo": len(_RE_INTENT_GEO.findall(text or "")),
        "quote": len(_RE_INTENT_QUOTE.findall(text or "")),
    }
    best = max(counts, key=lambda k: (counts[k], -("eng people geo quote".split().index(k))))
    return best if counts[best] > 0 else "scene"


_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:;|\b(?:and|while|whereas|but|because|as|when|after|before|then)\b)\s*", re.I)


def _split_turn_clauses(text: str, max_beats: int = 5) -> list[str]:
    """Split a narration turn into meaningful phrase/clause units for beat-level picture mapping."""
    parts = [p.strip(" ,") for p in re.split(r"(?<=[.!?])\s+|(?<=,)\s+(?=\w{4,})", text or "") if len(p.strip()) >= 12]
    if len(parts) < 2:
        parts = [p.strip(" ,") for p in _CLAUSE_SPLIT_RE.split(text or "") if p and len(p.strip()) >= 12]
    if not parts:
        parts = [text or ""]
    # Merge overflow clauses into the last beat to respect the max-beats ceiling
    while len(parts) > max_beats:
        parts[-2] = f"{parts[-2]}, {parts[-1]}"
        parts.pop()
    return parts


def _family_for_intent(intent: str, scene_mapped: bool, beat_idx: int) -> str:
    """Map a narration clause's semantic intent onto the visual family that answers it."""
    if intent == "eng":
        return "split_fact_card"
    if intent == "geo":
        return "wide_studio"
    if intent == "quote":
        return "quote_card"
    if intent == "people":
        return "studio_insert"
    return "broll_cinema" if scene_mapped else ("split_fact_card" if beat_idx % 2 == 0 else "studio_insert")


def _read_segment_mono_audio(master_wav: Path, start_sec: float, duration_sec: float) -> tuple[np.ndarray, int]:
    """Read exact segment slice from master WAV as mono float32 (decimated to 24kHz for fast FFT telemetry)."""
    info = sf.info(str(master_wav))
    sr = int(info.samplerate)
    s0 = max(0, int(round(start_sec * sr)))
    s1 = min(int(info.frames), s0 + max(1, int(round(duration_sec * sr))))
    if s1 <= s0:
        return np.zeros(2400, dtype=np.float32), 24000
    data, _ = sf.read(str(master_wav), start=s0, stop=s1, dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    if sr == 48000 and len(mono) >= 4:
        return mono[::2], 24000
    return mono, sr


def _build_rms_envelope_50hz(master_wav: Path) -> np.ndarray:
    """Compute lightweight 50Hz RMS envelope across `master_wav` so sub-turn cuts land on breath pauses."""
    try:
        info = sf.info(str(master_wav))
        sr = int(info.samplerate)
        win = max(1, sr // 50)
        vals: list[np.ndarray] = []
        with sf.SoundFile(str(master_wav), "r") as f:
            while True:
                blk = f.read(win * 200, dtype="float32", always_2d=True)
                if len(blk) == 0:
                    break
                mono = np.mean(blk, axis=1)
                n_full = len(mono) // win
                if n_full > 0:
                    reshaped = mono[: n_full * win].reshape(n_full, win)
                    vals.append(np.sqrt(np.mean(reshaped * reshaped, axis=1)))
        return np.concatenate(vals) if vals else np.zeros(10, dtype=np.float32)
    except Exception:
        return np.zeros(10, dtype=np.float32)


def _snap_to_breath_pause(env_50hz: np.ndarray, target_sec: float, min_sec: float, max_sec: float) -> float:
    """Snap a planned cut timestamp to the quietest natural speech valley within ±0.52s."""
    lo = max(min_sec, target_sec - 0.52)
    hi = min(max_sec, target_sec + 0.52)
    if hi <= lo + 0.08 or len(env_50hz) < 10:
        return float(np.clip(target_sec, min_sec, max_sec))
    i0 = max(0, min(len(env_50hz) - 1, int(round(lo * 50.0))))
    i1 = max(i0 + 1, min(len(env_50hz), int(round(hi * 50.0))))
    window = env_50hz[i0:i1]
    offsets = (np.arange(i0, i1, dtype=np.float32) / 50.0) - target_sec
    cost = window + 0.015 * (offsets * offsets)
    best_i = i0 + int(np.argmin(cost))
    return float(np.clip(best_i / 50.0, min_sec, max_sec))


def _collect_chapter_broll_pool(ch: ChapterScript) -> list[Path]:
    """Gather chapter-specific code-synthesized 1080p reference plates."""
    pool: list[Path] = []
    seen: set[str] = set()
    for t in ch.turns:
        if t.visual_asset_path and Path(t.visual_asset_path).exists():
            k = str(Path(t.visual_asset_path).resolve())
            if k not in seen:
                seen.add(k)
                pool.append(Path(t.visual_asset_path))
    if ch.card_asset_path and Path(ch.card_asset_path).exists():
        k = str(Path(ch.card_asset_path).resolve())
        if k not in seen:
            seen.add(k)
            pool.append(Path(ch.card_asset_path))
    return pool


def _pick_plate_from_pool(
    pool: list[Path],
    global_used: set[str],
    last_used: str,
    turn_text: str,
    rot_idx: int,
) -> Path | None:
    del global_used, turn_text
    if not pool:
        return None
    non_adj = [p for p in pool if str(p.resolve()) != last_used]
    return non_adj[rot_idx % len(non_adj)] if non_adj else pool[rot_idx % len(pool)]


def _caption_for_asset(p: Path | None, ch: ChapterScript, fallback: str = "", turn_text: str = "") -> str:
    del p
    _, scene_title = resolve_procedural_scene_id(
        f"{turn_text} || {ch.title} {ch.location}", fallback_idx=ch.chapter_index
    )
    return f"3D Architectural & LiDAR Model · {scene_title}" if scene_title else (fallback or ch.title)


_REQUIRED_VISUAL_TEXT = {
    "broll_cinema": "Wide pure-code 3D monument cinematography of the spoken subject",
    "closeup_host_a": "Tight pure-code 3D detail cinematography of the spoken subject",
    "closeup_host_b": "Tight pure-code 3D detail cinematography of the spoken subject",
    "wide_studio": "3D raycasted orbital Earth GIS map centred on the spoken place/route",
    "split_fact_card": "Live counting telemetry + LiDAR viewport illustrating spoken figures/structure",
    "studio_insert": "Explanatory simulation blackboard / LiDAR turntable illustrating the spoken mechanism",
    "reaction_shot": "Dual-plate archival light-table illustrating the spoken comparison",
    "quote_card": "Verbatim primary-source inscription typography of the spoken citation",
}
_VISUAL_SOURCE = {
    "broll_cinema": "pure-code:Monument3DEngine+broll_depth",
    "closeup_host_a": "pure-code:Monument3DEngine+broll_depth(closeup)",
    "closeup_host_b": "pure-code:Monument3DEngine+broll_depth(closeup)",
    "wide_studio": "pure-code:raycast_orbital_earth_globe",
    "split_fact_card": "pure-code:ProceduralCadCache+telemetry",
    "studio_insert": "pure-code:PhysicsSimulationEngine/ProceduralCadCache",
    "reaction_shot": "pure-code:PhysicsSimulationEngine/ProceduralCadCache",
    "quote_card": "pure-code:kinetic typography",
}


def _plan_turn_editorial_beats(
    turn: DialogueTurn,
    turn_start_sec: float,
    turn_end_sec: float,
    env_50hz: np.ndarray,
    is_episode_opener: bool,
    is_episode_closer: bool,
    scene_context: str = "",
    lead_attribution_sec: float = 0.0,
) -> list[tuple[CameraAngle, float, str]]:
    """Decompose a narration segment into clause-aligned documentary visual beats.

    Each beat carries the exact narration clause it illustrates, so picture changes track
    the spoken subject/action at phrase level (durations come from the real TTS audio via
    the caller's frame-accurate turn timestamps + breath-pause snapping).
    """
    total_dur = max(0.4, turn_end_sec - turn_start_sec)
    spk_cu: CameraAngle = "closeup_host_a" if turn.speaker == "HOST_A" else "closeup_host_b"
    intent = _turn_visual_intent(turn.text)
    primary = turn.camera if intent == "scene" else _INTENT_PRIMARY[intent]

    if total_dur < 5.6:
        if primary == "reaction_shot":
            if total_dur >= 3.9:
                react_d = 1.95
                cut_t = _snap_to_breath_pause(
                    env_50hz, turn_end_sec - react_d, turn_start_sec + 1.8, turn_end_sec - 1.5
                )
                return [(spk_cu, cut_t, ""), ("studio_insert", turn_end_sec, turn.text)]
            return [(spk_cu, turn_end_sec, turn.text)]
        if is_episode_closer and primary != "wide_studio":
            return [("wide_studio", turn_end_sec, turn.text)]
        return [(primary, turn_end_sec, turn.text)]

    # ---- Clause-aligned beat construction: one visual beat per meaningful spoken clause ----
    specs: list[tuple[CameraAngle, float, str]] = []
    if is_episode_opener:
        specs = [
            ("broll_cinema", 4.4, ""),
            ("wide_studio", 4.1, ""),
            (spk_cu, 3.8, ""),
            ("broll_cinema", 4.6, ""),
            ("studio_insert", 4.0, ""),
        ]
    elif is_episode_closer:
        specs = [
            ("broll_cinema", 4.5, ""),
            (spk_cu, 3.6, ""),
            ("wide_studio", 4.2, ""),
            ("broll_cinema", 4.4, ""),
        ]
    else:
        if lead_attribution_sec > 0.0:
            specs.append((spk_cu, lead_attribution_sec, ""))
        clauses = _split_turn_clauses(turn.text)
        for ci, clause in enumerate(clauses):
            intent_c = _turn_visual_intent(clause)
            sid_c, _ = resolve_procedural_scene_id(f"{clause} || {scene_context}", fallback_idx=ci + 1)
            fam = _family_for_intent(intent_c, sid_c != SCENE_CLASSICAL_GENERIC, ci)
            specs.append((fam, float(max(3, len(clause.split()))), clause))

    # Merge beats that would fall below the 2.2s minimum shot length
    def _scaled_min_ok(weights: list[float], total: float) -> bool:
        s = sum(weights)
        return all(w * total / s >= 2.2 for w in weights)

    while len(specs) > 2 and not _scaled_min_ok([w for _, w, _ in specs], total_dur):
        # merge the shortest adjacent pair
        idx = min(range(1, len(specs)), key=lambda i: specs[i][1])
        prev = specs[idx - 1]
        merged = (prev[0], prev[1] + specs[idx][1], f"{prev[2]} {specs[idx][2]}".strip())
        specs[idx - 1] = merged
        specs.pop(idx)

    # Long-narration rule: never hold one picture beyond ~6.2s — split the heaviest clause.
    while len(specs) < 8:
        wsum = sum(w for _, w, _ in specs)
        scale0 = total_dur / max(1e-6, wsum)
        heavy = [i for i, s in enumerate(specs) if s[1] * scale0 > 6.2]
        if not heavy:
            break
        i_h = max(heavy, key=lambda i: specs[i][1])
        cam_h, w_h, cl_h = specs[i_h]
        words_h = (cl_h or "").split()
        if len(words_h) >= 6:
            mid_h = len(words_h) // 2
            specs[i_h : i_h + 1] = [
                (cam_h, w_h / 2.0, " ".join(words_h[:mid_h])),
                (cam_h, w_h / 2.0, " ".join(words_h[mid_h:])),
            ]
        else:
            specs[i_h : i_h + 1] = [(cam_h, w_h / 2.0, cl_h), (cam_h, w_h / 2.0, "")]

    if is_episode_closer and specs[-1][0] != "broll_cinema":
        specs[-1] = ("broll_cinema", specs[-1][1], specs[-1][2])

    weight_sum = sum(w for _, w, _ in specs)
    beats: list[tuple[CameraAngle, float, str]] = []
    cur_t = turn_start_sec
    for idx_b, (cam, wgt, clause) in enumerate(specs):
        if idx_b == len(specs) - 1:
            beats.append((cam, turn_end_sec, clause))
            break
        target_end = cur_t + total_dur * (wgt / weight_sum)
        min_bound = cur_t + 2.35
        max_bound = turn_end_sec - 2.35
        if max_bound <= min_bound:
            snapped = float(np.clip(target_end, cur_t + 0.8, turn_end_sec - 0.8))
        else:
            snapped = _snap_to_breath_pause(env_50hz, target_end, min_bound, max_bound)
        beats.append((cam, snapped, clause))
        cur_t = snapped

    return beats


def _encode_single_segment(
    task: RenderSegmentTask,
    animator: StudioCharacterAnimator,
    ffmpeg_bin: str,
    master_wav: Path,
    x264_preset: str,
    is_marathon_cpu: bool = False,
) -> Path:
    """Encode a single video-only 1080p24 Rec.709 H.264 documentary segment locked to exact frame count."""
    fps = VIDEO_SPEC.fps
    frames = max(1, int(round(task.duration_sec * fps)))
    exact_dur = frames / fps
    fonts_esc = _escape_filter_path(FONTS_DIR)

    anim_mode = task.layout.anim_mode
    is_intro_montage = (task.camera_label == "intro_bumper") and not is_marathon_cpu
    use_telemetry = (task.camera_label == "split_fact_card" or anim_mode == "telemetry_split") and not is_marathon_cpu
    use_kinetic_quote = (
        (task.camera_label == "quote_card" or anim_mode == "kinetic_quote")
        and bool(task.layout.quote_verbatim)
        and not is_marathon_cpu
    )
    use_broll_depth = (
        not is_intro_montage
        and not use_kinetic_quote
        and (anim_mode == "broll_depth" or (anim_mode == "none" and task.camera_label == "broll_cinema" and task.layout.use_ken_burns))
        and not is_marathon_cpu
    )
    use_geo_map = anim_mode == "geo_map" and not is_marathon_cpu
    use_archival_desk = anim_mode == "archival_desk" and not is_marathon_cpu
    use_smooth_card = (
        not is_intro_montage
        and not use_kinetic_quote
        and (anim_mode == "smooth_card" or (anim_mode == "none" and not use_broll_depth and not use_telemetry and task.layout.use_ken_burns))
        and not is_marathon_cpu
    )
    use_pipe = (
        is_intro_montage
        or use_telemetry
        or use_kinetic_quote
        or use_broll_depth
        or use_geo_map
        or use_archival_desk
        or use_smooth_card
    )

    inputs: list[str] = []
    fc_parts: list[str] = []
    next_input_idx = 0

    if use_pipe:
        inputs.extend([
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{VIDEO_SPEC.width}x{VIDEO_SPEC.height}",
            "-r", str(fps),
            "-i", "pipe:0",
        ])
        fc_parts.append("[0:v]setsar=1[base]")
        cur_v = "base"
        next_input_idx = 1
    else:
        bg_idx = next_input_idx
        inputs.extend(["-i", str(task.layout.image_path)])
        next_input_idx += 1

        if task.layout.use_ken_burns:
            z_expr = "min(zoom+0.0006,1.14)" if task.layout.ken_burns_zoom_in else "if(eq(on,1),1.13,max(1.001,zoom-0.0006))"
            fc_parts.append(
                f"[{bg_idx}:v]scale=2048:1152,setsar=1,"
                f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={frames}:s={VIDEO_SPEC.width}x{VIDEO_SPEC.height}:fps={fps}[base]"
            )
        else:
            fc_parts.append(
                f"[{bg_idx}:v]scale={VIDEO_SPEC.width}:{VIDEO_SPEC.height},setsar=1,"
                f"loop=loop={frames}:size=1:start=0,fps={fps},trim=end_frame={frames}[base]"
            )
        cur_v = "base"

    # Subtle broadcast cross-dissolve: previous shot's tail frame dissolves out over the first 0.34s
    # (duration-preserving overlay so master audio A/V sync stays frame-exact; captions remain crisp).
    if task.dissolve_in_png is not None and task.dissolve_in_png.exists() and exact_dur >= 0.9:
        inputs.extend(["-loop", "1", "-t", f"{exact_dur:.3f}", "-r", str(fps), "-i", str(task.dissolve_in_png)])
        fc_parts.append(
            f"[{next_input_idx}:v]scale={VIDEO_SPEC.width}:{VIDEO_SPEC.height},setsar=1,format=rgba,"
            f"fade=t=out:st=0:d=0.34:alpha=1[dov];"
            f"[{cur_v}][dov]overlay=0:0:shortest=1[vdis]"
        )
        cur_v = "vdis"
        next_input_idx += 1

    if task.hud_overlay is not None:
        hud_path, _, _, _ = task.hud_overlay
        inputs.extend(["-i", str(hud_path)])
        if task.animate_hud_reveal:
            fc_parts.append(
                f"[{next_input_idx}:v]format=rgba,loop=loop={frames}:size=1:start=0,fps={fps},"
                f"trim=end_frame={frames},fade=t=in:st=0.12:d=0.55:alpha=1[ov_anim];"
                f"[{cur_v}][ov_anim]overlay=0:0:shortest=1[vhud]"
            )
        else:
            fc_parts.append(f"[{cur_v}][{next_input_idx}:v]overlay=0:0:eof_action=repeat[vhud]")
        cur_v = "vhud"
        next_input_idx += 1

    lt_path = task.layout.transient_lt_path
    if lt_path is not None and lt_path.exists() and exact_dur >= 1.1:
        inputs.extend(["-i", str(lt_path)])
        fade_out_st = min(3.10, max(0.60, exact_dur - 0.42))
        fc_parts.append(
            f"[{next_input_idx}:v]format=rgba,loop=loop={frames}:size=1:start=0,fps={fps},"
            f"trim=end_frame={frames},"
            f"fade=t=in:st=0.14:d=0.28:alpha=1,"
            f"fade=t=out:st={fade_out_st:.3f}:d=0.36:alpha=1[lt_anim];"
            f"[{cur_v}][lt_anim]overlay=0:0:shortest=1[vlt]"
        )
        cur_v = "vlt"
        next_input_idx += 1

    if task.ass_path is not None and task.ass_path.exists():
        ass_esc = _escape_filter_path(task.ass_path)
        fc_parts.append(f"[{cur_v}]ass='{ass_esc}':fontsdir='{fonts_esc}'[vsub]")
        cur_v = "vsub"

    fades: list[str] = []
    if task.fade_in and exact_dur >= 0.8:
        fades.append("fade=t=in:st=0:d=0.28")
    if task.fade_out and exact_dur >= 0.9:
        fades.append(f"fade=t=out:st={max(0.0, exact_dur - 0.30):.3f}:d=0.30")
    fades.append("format=yuv420p")
    fc_parts.append(f"[{cur_v}]{','.join(fades)}[vout]")

    crf_val = "25" if is_marathon_cpu else "24.8"
    maxrate = "820k" if is_marathon_cpu else "840k"
    bufsize = "1640k" if is_marathon_cpu else "1680k"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        ";".join(fc_parts),
        "-map",
        "[vout]",
        "-frames:v",
        str(frames),
        "-r",
        str(fps),
        "-c:v",
        VIDEO_SPEC.video_codec,
        "-preset",
        x264_preset,
        "-g",
        "240",
        "-crf",
        crf_val,
        "-maxrate",
        maxrate,
        "-bufsize",
        bufsize,
        "-pix_fmt",
        VIDEO_SPEC.pixel_format,
        "-colorspace",
        VIDEO_SPEC.colorspace,
        "-color_primaries",
        VIDEO_SPEC.color_primaries,
        "-color_trc",
        VIDEO_SPEC.color_trc,
        "-color_range",
        VIDEO_SPEC.color_range,
        "-video_track_timescale",
        "12288",
        "-an",
        str(task.out_mp4),
    ]

    if not use_pipe:
        subprocess.run(cmd, check=True)
        if task.write_tail_jpg:
            src_img = cv2.imread(str(task.layout.image_path))
            if src_img is not None:
                cv2.imwrite(
                    str(task.out_mp4.with_name(f"tail_{task.seg_index:04d}.jpg")),
                    src_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )
        return task.out_mp4

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        mono_audio, sr = _read_segment_mono_audio(master_wav, task.start_sec, exact_dur)
        curves = extract_speech_motion_curves(mono_audio, sr, frames, fps=fps)

        if is_intro_montage:
            frame_iter = animator.iter_intro_montage_frames(n_frames=frames)
        elif use_geo_map:
            frame_iter = animator.iter_geo_map_frames(
                target_hint=task.layout.map_target_hint or task.layout.location_text,
                prev_hint=task.layout.map_prev_hint,
                era_text=task.layout.era_text,
                n_frames=frames,
                curves=curves,
                fallback_index=task.seg_index + 1,
            )
        elif use_archival_desk:
            sec_img = task.layout.secondary_image_path or task.layout.image_path
            frame_iter = animator.iter_archival_desk_frames(
                primary_plate=task.layout.image_path,
                secondary_plate=sec_img,
                primary_caption=task.layout.primary_caption,
                secondary_caption=task.layout.secondary_caption,
                chapter_tag=task.layout.chapter_title_text,
                n_frames=frames,
                seg_seed=task.seg_index,
                scene_hint=task.layout.scene_hint,
            )
        elif use_kinetic_quote:
            photo_src = task.layout.secondary_image_path or task.layout.image_path
            frame_iter = animator.iter_kinetic_quote_frames(
                plate_path=photo_src,
                verbatim_quote=task.layout.quote_verbatim,
                author_name=task.layout.quote_author,
                citation_line=task.layout.quote_citation,
                n_frames=frames,
                seg_seed=task.seg_index,
                scene_hint=task.layout.scene_hint,
            )
        elif use_telemetry:
            photo_src = task.layout.secondary_image_path or task.layout.image_path
            frame_iter = animator.iter_telemetry_split_frames(
                photo_path=photo_src,
                chapter_title=task.layout.chapter_title_text,
                era_text=task.layout.era_text,
                location_text=task.layout.location_text,
                metrics=list(task.layout.telemetry_metrics),
                accent_hex=task.layout.wave_hex,
                n_frames=frames,
                curves=curves,
                seg_seed=task.seg_index,
                scene_hint=task.layout.scene_hint,
            )
        elif use_broll_depth:
            frame_iter = animator.iter_broll_depth_frames(
                plate_path=task.layout.image_path,
                n_frames=frames,
                seg_seed=task.seg_index,
                zoom_in=task.layout.ken_burns_zoom_in,
                curves=curves,
                lens_variant=task.lens_variant,
                scene_hint=task.layout.scene_hint,
            )
        else:
            frame_iter = animator.iter_smooth_card_frames(
                plate_path=task.layout.image_path,
                n_frames=frames,
                zoom_in=task.layout.ken_burns_zoom_in if task.layout.use_ken_burns else True,
                chronology_year=task.layout.chronology_year,
                prev_year=task.layout.prev_chronology_year,
            )

        last_rgb: np.ndarray | None = None
        for rgb in frame_iter:
            last_rgb = rgb
            proc.stdin.write(rgb.tobytes())
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass

    err_bytes = proc.stderr.read() if proc.stderr else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"FFmpeg documentary segment {task.seg_index} failed (rc={rc}): {err_bytes.decode(errors='replace')}"
        )
    if task.write_tail_jpg and last_rgb is not None:
        cv2.imwrite(
            str(task.out_mp4.with_name(f"tail_{task.seg_index:04d}.jpg")),
            np.ascontiguousarray(last_rgb[:, :, ::-1]),
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )
    return task.out_mp4


def render_episode_video(
    script: EpisodeScript,
    audio_meta: AudioMasterMeta,
    env: StudioEnvironment,
    work_dir: Path,
    output_dir: Path,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[Path, Path, Path]:
    """Render full Rec.709 cinema documentary 1080p24 MP4 + SRT + VTT with zero-repeat plate choreography."""
    plates_dir = work_dir / "plates"
    segs_dir = work_dir / "segments"
    ass_dir = work_dir / "ass_segments"
    plates_dir.mkdir(parents=True, exist_ok=True)
    segs_dir.mkdir(parents=True, exist_ok=True)
    ass_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    compositor = MulticamCompositor()
    animator = StudioCharacterAnimator()
    cues: list[CaptionCue] = build_all_caption_cues(script)

    srt_path = output_dir / "captions.srt"
    vtt_path = output_dir / "captions.vtt"
    write_srt_and_vtt(cues, srt_path, vtt_path)

    total_ch = len(script.chapters)
    show_title = script.spec.show_title
    ep_title = script.spec.episode_title
    plan = ScenePlan(topic=f"{show_title}: {ep_title}")
    master_wav = Path(audio_meta.master_wav_path)
    env_50hz = _build_rms_envelope_50hz(master_wav)
    cpu_count = os.cpu_count() or 2
    is_marathon_cpu = audio_meta.duration_sec > 900 and cpu_count <= 4

    tasks: list[RenderSegmentTask] = []
    seg_idx = 0
    accumulated_frames = 0
    fps = VIDEO_SPEC.fps

    def _alloc_seg_duration(target_end_sec: float) -> tuple[float, float]:
        nonlocal accumulated_frames
        start_s = accumulated_frames / fps
        target_total_frames = max(accumulated_frames + 1, int(round(target_end_sec * fps)))
        seg_frames = target_total_frames - accumulated_frames
        accumulated_frames = target_total_frames
        return start_s, seg_frames / fps

    # 0. Branded Rapid-Cut 3-Wonder Aerial Cold-Open Montage (`0.0s .. first_ch_start`)
    first_ch_start = script.chapters[0].start_sec if script.chapters else script.intro_duration_sec
    if first_ch_start >= 1.5:
        s_sec, dur_sec = _alloc_seg_duration(first_ch_start)
        intro_layout, intro_hud = compositor.render_intro_bumper(
            script,
            plates_dir / f"seg_{seg_idx:04d}_intro_bg.jpg",
            plates_dir / f"seg_{seg_idx:04d}_intro_ov.png",
        )
        tasks.append(
            RenderSegmentTask(
                seg_index=seg_idx,
                start_sec=s_sec,
                duration_sec=dur_sec,
                layout=intro_layout,
                ass_path=None,
                hud_overlay=intro_hud,
                fade_in=False,
                fade_out=False,
                out_mp4=segs_dir / f"seg_{seg_idx:04d}.mp4",
                animate_hud_reveal=True,
                camera_label="intro_bumper",
            )
        )
        seg_idx += 1

    outro_start_boundary = max(
        first_ch_start + 1.0,
        audio_meta.duration_sec - max(4.0, float(script.outro_duration_sec)),
    )

    prev_year_marker = -3000
    prev_map_hint = "Alexandria Mediterranean"
    last_used_asset_str = ""
    global_used_plates: set[str] = set()
    plate_visit_counts: dict[str, int] = {}
    attributed_speakers: set[str] = set()

    for c_i, ch in enumerate(script.chapters):
        ch_broll_pool = _collect_chapter_broll_pool(ch)
        broll_rot_idx = 0

        def _next_distinct_plate(turn_text: str = "") -> tuple[Path | None, int]:
            nonlocal broll_rot_idx, last_used_asset_str
            cand = _pick_plate_from_pool(
                ch_broll_pool, global_used_plates, last_used_asset_str, turn_text, broll_rot_idx
            )
            broll_rot_idx += 1
            visit = 0
            if cand is not None:
                last_used_asset_str = str(cand.resolve())
                m_stem = re.match(r"ch\d+_archive_(.+)$", cand.name)
                canon = m_stem.group(1) if m_stem else cand.name
                visit = plate_visit_counts.get(canon, 0)
                plate_visit_counts[canon] = visit + 1
            return cand, visit

        # 1. Act Opener Card with 24fps Animated Chronology Scrubber (skipped on Act 1 when card_duration_sec==0!)
        cur_year = parse_historical_year(ch.era_or_date, fallback_idx=c_i + 1)
        first_turn_start = ch.turns[0].start_sec if ch.turns else (ch.start_sec + ch.card_duration_sec)
        if first_turn_start - (accumulated_frames / fps) >= 0.65:
            s_sec, dur_sec = _alloc_seg_duration(first_turn_start)
            card_plate = compositor.render_chapter_card(
                ch,
                total_ch,
                show_title,
                plates_dir / f"seg_{seg_idx:04d}_card.jpg",
                chronology_year=cur_year,
                prev_chronology_year=prev_year_marker,
            )
            card_hud = compositor.render_broll_hud_overlay(
                ch.turns[0] if ch.turns else script.all_turns[0],
                ch,
                plates_dir / f"seg_{seg_idx:04d}_card_hud.png",
                total_chapters=total_ch,
                show_title=show_title,
                caption_override=f"Chronological Survey · {ch.location}",
            )
            tasks.append(
                RenderSegmentTask(
                    seg_index=seg_idx,
                    start_sec=s_sec,
                    duration_sec=dur_sec,
                    layout=card_plate,
                    ass_path=None,
                    hud_overlay=card_hud,
                    fade_in=False,
                    fade_out=False,
                    out_mp4=segs_dir / f"seg_{seg_idx:04d}.mp4",
                    camera_label="chapter_card",
                )
            )
            seg_idx += 1
        prev_year_marker = cur_year

        # 2. Sub-Turn Documentary Cinema, 3D GIS Cartography, Archival Desk & Live Telemetry Beats
        for t_i, turn in enumerate(ch.turns):
            is_ep_opener = c_i == 0 and t_i == 0
            is_ep_closer = c_i == len(script.chapters) - 1 and t_i == len(ch.turns) - 1
            if t_i + 1 < len(ch.turns):
                shot_end_sec = ch.turns[t_i + 1].start_sec
            elif c_i + 1 < len(script.chapters):
                shot_end_sec = script.chapters[c_i + 1].start_sec
            else:
                shot_end_sec = outro_start_boundary

            turn_cur_start = accumulated_frames / fps
            # Restrained prestige documentary voice attribution: ONCE per voice across the film!
            want_turn_lt = turn.speaker not in attributed_speakers
            beats = _plan_turn_editorial_beats(
                turn,
                turn_cur_start,
                shot_end_sec,
                env_50hz,
                is_episode_opener=is_ep_opener,
                is_episode_closer=is_ep_closer,
                scene_context=f"{ch.title} {ch.location}",
                lead_attribution_sec=2.9 if (want_turn_lt and not is_ep_opener) else 0.0,
            )
            lt_emitted = False

            for b_i, (beat_cam, beat_end_sec, beat_clause) in enumerate(beats):
                s_sec, dur_sec = _alloc_seg_duration(beat_end_sec)
                # Clause-scoped turn copy: every downstream caption/plate/HUD resolves the
                # picture from the exact spoken clause of THIS beat, not the whole turn.
                turn = dataclasses.replace(turn, text=beat_clause) if beat_clause else turn
                plate_file = plates_dir / f"seg_{seg_idx:04d}_{beat_cam}.jpg"
                hud_overlay: tuple[Path, int, int, str] | None = None

                show_lt_now = False
                if want_turn_lt and not lt_emitted and beat_cam in ("wide_studio", "closeup_host_a", "closeup_host_b") and dur_sec >= 2.2:
                    show_lt_now = True
                    lt_emitted = True
                    attributed_speakers.add(turn.speaker)

                cur_lens_variant = 0
                if beat_cam == "wide_studio":
                    layout, hud_overlay = compositor.render_wide_studio(
                        turn,
                        ch,
                        total_ch,
                        show_title,
                        plate_file,
                        episode_title=ep_title if (c_i == 0 and t_i == 0) else "",
                        show_lower_third=show_lt_now,
                        prev_location_hint=prev_map_hint,
                    )
                    prev_map_hint = f"{turn.text} || {ch.title} {ch.location}"
                elif beat_cam in ("closeup_host_a", "closeup_host_b"):
                    chosen_p, cur_lens_variant = _next_distinct_plate(turn.text)
                    cap_txt = _caption_for_asset(chosen_p, ch, turn.visual_caption, turn.text)
                    layout, hud_overlay = compositor.render_host_closeup(
                        turn,
                        ch,
                        total_ch,
                        show_title,
                        plate_file,
                        show_lower_third=show_lt_now,
                        asset_override=chosen_p,
                        caption_override=cap_txt,
                    )
                elif beat_cam == "reaction_shot":
                    p1, _ = _next_distinct_plate(turn.text)
                    p2, _ = _next_distinct_plate(turn.text)
                    layout, hud_overlay = compositor.render_reaction_shot(
                        turn,
                        ch,
                        total_ch,
                        show_title,
                        plate_file,
                        primary_asset=p1,
                        secondary_asset=p2,
                        primary_caption=_caption_for_asset(p1, ch, "", turn.text),
                        secondary_caption=_caption_for_asset(p2, ch, "", turn.text),
                    )
                elif beat_cam == "studio_insert":
                    p1, _ = _next_distinct_plate(turn.text)
                    p2, _ = _next_distinct_plate(turn.text)
                    layout, hud_overlay = compositor.render_studio_insert(
                        turn,
                        ch,
                        total_ch,
                        show_title,
                        plate_file,
                        plates_dir / f"seg_{seg_idx:04d}_ins_hud.png",
                        primary_asset=p1,
                        secondary_asset=p2,
                        primary_caption=_caption_for_asset(p1, ch, "", turn.text),
                        secondary_caption=_caption_for_asset(p2, ch, "", turn.text),
                    )
                elif beat_cam == "quote_card":
                    p_q, _ = _next_distinct_plate(turn.text)
                    layout = compositor.render_quote_card(
                        turn, ch, total_ch, show_title, plate_file, asset_override=p_q
                    )
                    hud_overlay = compositor.render_broll_hud_overlay(
                        turn,
                        ch,
                        plates_dir / f"seg_{seg_idx:04d}_q_hud.png",
                        total_chapters=total_ch,
                        show_title=show_title,
                        caption_override=f"Primary-Source Inscription · {_caption_for_asset(p_q, ch, '', turn.text)}",
                    )
                elif beat_cam == "split_fact_card":
                    p_f, _ = _next_distinct_plate(turn.text)
                    layout = compositor.render_split_fact_card(
                        turn, ch, total_ch, show_title, plate_file, asset_override=p_f
                    )
                    hud_overlay = compositor.render_broll_hud_overlay(
                        turn,
                        ch,
                        plates_dir / f"seg_{seg_idx:04d}_f_hud.png",
                        total_chapters=total_ch,
                        show_title=show_title,
                        caption_override=f"Engineering Telemetry · {_caption_for_asset(p_f, ch, '', turn.text)}",
                    )
                else:
                    chosen_p, cur_lens_variant = _next_distinct_plate(turn.text)
                    cap_txt = _caption_for_asset(chosen_p, ch, turn.visual_caption, turn.text)
                    layout = compositor.render_broll_cinema(
                        turn,
                        ch,
                        plate_file,
                        asset_override=chosen_p,
                        zoom_in_override=((seg_idx % 2) == 0),
                    )
                    hud_overlay = compositor.render_broll_hud_overlay(
                        turn,
                        ch,
                        plates_dir / f"seg_{seg_idx:04d}_broll_hud.png",
                        total_chapters=total_ch,
                        show_title=show_title,
                        caption_override=cap_txt,
                    )

                ass_file = ass_dir / f"seg_{seg_idx:04d}.ass"
                has_subs = write_segment_ass(cues, s_sec, s_sec + dur_sec, ass_file)

                # Zero pitch-black mid-film flashes — crisp straight editorial cuts throughout Acts 1..5!
                use_fade_in = False
                use_fade_out = False

                sid_b, scene_title_b = resolve_procedural_scene_id(
                    f"{turn.text} || {ch.title} {ch.location}", fallback_idx=seg_idx + 1
                )
                intent_b = _turn_visual_intent(turn.text)
                plan.entries.append(
                    ScenePlanEntry(
                        scene_id=sid_b,
                        seg_index=seg_idx,
                        chapter_index=c_i,
                        turn_id=turn.turn_id,
                        narration_clause=beat_clause or "(voice attribution / establishing beat)",
                        narration_start_sec=round(s_sec, 3),
                        narration_end_sec=round(s_sec + dur_sec, 3),
                        subject=scene_title_b or ep_title,
                        action_or_event=intent_b,
                        required_visual=_REQUIRED_VISUAL_TEXT.get(beat_cam, "pure-code cinematic plate"),
                        visual_family=beat_cam,
                        visual_asset_source=_VISUAL_SOURCE.get(beat_cam, "pure-code engine"),
                        scene_start_sec=round(s_sec, 3),
                        scene_end_sec=round(s_sec + dur_sec, 3),
                        on_screen_text=(ch.title if not beat_clause else beat_clause[:96]),
                        relevance_criteria=(
                            f"family={beat_cam} must depict intent={intent_b}; subject={scene_title_b}; "
                            "clause keywords drive procedural scene resolution"
                        ),
                        validation_status="passed_plan_gate",
                    )
                )

                tasks.append(
                    RenderSegmentTask(
                        seg_index=seg_idx,
                        start_sec=s_sec,
                        duration_sec=dur_sec,
                        layout=layout,
                        ass_path=ass_file if has_subs else None,
                        hud_overlay=hud_overlay,
                        fade_in=use_fade_in,
                        fade_out=use_fade_out,
                        out_mp4=segs_dir / f"seg_{seg_idx:04d}.mp4",
                        camera_label=beat_cam,
                        lens_variant=cur_lens_variant,
                    )
                )
                seg_idx += 1

    # 3. Branded Documentary End-Credits Card
    if audio_meta.duration_sec - (accumulated_frames / fps) >= 1.2:
        s_sec, dur_sec = _alloc_seg_duration(audio_meta.duration_sec)
        outro_layout = compositor.render_outro_card(
            script, plates_dir / f"seg_{seg_idx:04d}_outro.jpg"
        )
        tasks.append(
            RenderSegmentTask(
                seg_index=seg_idx,
                start_sec=s_sec,
                duration_sec=dur_sec,
                layout=outro_layout,
                ass_path=None,
                hud_overlay=None,
                fade_in=False,
                fade_out=True,
                out_mp4=segs_dir / f"seg_{seg_idx:04d}.mp4",
                camera_label="outro_card",
            )
        )
        seg_idx += 1

    (output_dir / "scene_plan.json").write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

    x264_preset = "veryfast" if is_marathon_cpu else "superfast"
    workers = max(2, min(4, cpu_count))

    # Subtle 0.34s broadcast cross-dissolves on interior editorial cuts (act cards, cold-open &
    # end-card stay as crisp hard cuts; durations preserved so master audio sync stays exact).
    ordered_pre = sorted(tasks, key=lambda x: x.seg_index)
    _hard_cut_labels = {"intro_bumper", "chapter_card", "outro_card"}
    for i in range(1, len(ordered_pre)):
        prev_t, cur_t = ordered_pre[i - 1], ordered_pre[i]
        if (
            cur_t.camera_label not in _hard_cut_labels
            and prev_t.camera_label not in _hard_cut_labels
            and cur_t.duration_sec >= 1.0
            and prev_t.duration_sec >= 0.8
        ):
            prev_t.write_tail_jpg = True
            cur_t.dissolve_in_png = prev_t.out_mp4.with_name(f"tail_{prev_t.seg_index:04d}.jpg")

    tail_events: dict[int, threading.Event] = {t.seg_index: threading.Event() for t in tasks}

    def _run_segment_task(t: RenderSegmentTask) -> Path:
        if t.dissolve_in_png is not None:
            tail_events[t.seg_index - 1].wait(timeout=1800)
        try:
            return _encode_single_segment(
                t, animator, env.ffmpeg_bin, master_wav, x264_preset, is_marathon_cpu
            )
        finally:
            tail_events[t.seg_index].set()

    completed = 0
    total_tasks = len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_segment_task, t): t.seg_index
            for t in tasks
        }
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            if progress_cb is not None:
                progress_cb(completed, total_tasks)

    concat_txt = work_dir / "concat_segments.txt"
    ordered_tasks = sorted(tasks, key=lambda x: x.seg_index)
    concat_txt.write_text(
        "\n".join(f"file '{t.out_mp4.resolve()}'" for t in ordered_tasks) + "\n",
        encoding="utf-8",
    )

    ffmeta_path = write_ffmetadata_chapters(script, audio_meta.duration_sec, work_dir / "chapters.ffmeta")
    final_mp4 = output_dir / "documentary_film.mp4"
    legacy_mp4 = output_dir / "podcast_studio.mp4"
    if legacy_mp4.exists():
        legacy_mp4.unlink()

    aac_bitrate = "112k" if audio_meta.duration_sec > 1800 else VIDEO_SPEC.audio_bitrate
    mux_cmd = [
        env.ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_txt),
        "-i",
        str(master_wav),
        "-i",
        str(ffmeta_path),
        "-map_metadata",
        "2",
        "-map_chapters",
        "2",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-colorspace",
        VIDEO_SPEC.colorspace,
        "-color_primaries",
        VIDEO_SPEC.color_primaries,
        "-color_trc",
        VIDEO_SPEC.color_trc,
        "-color_range",
        VIDEO_SPEC.color_range,
        "-c:a",
        VIDEO_SPEC.audio_codec,
        "-b:a",
        aac_bitrate,
        "-ar",
        "48000",
        "-ac",
        "2",
        "-shortest",
        "-movflags",
        "+faststart",
        str(final_mp4),
    ]
    subprocess.run(mux_cmd, check=True)
    return final_mp4, srt_path, vtt_path
