"""Master Autonomous Production Pipeline (Research -> Create -> Audit -> Improve -> Deliver)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from voxforge.audit.audio_qa import audit_and_improve_audio
from voxforge.audit.av_sync_qa import audit_av_sync
from voxforge.audit.deliverable_qa import inspect_final_deliverable, write_audit_reports
from voxforge.audit.editorial import audit_and_improve_script
from voxforge.audit.visual_qa import audit_and_improve_visuals
from voxforge.bootstrap import bootstrap_studio
from voxforge.config import DEFAULT_OUTPUT_ROOT
from voxforge.intent import parse_command_to_spec
from voxforge.models import BuildResult, EpisodeAuditReport
from voxforge.research.engine import build_research_dossier
from voxforge.script.showrunner import generate_episode_script
from voxforge.video.renderer import render_episode_video

StatusCallback = Callable[[str, str], None]


def _noop_status(stage: str, message: str) -> None:
    pass


def run_podcast_pipeline(
    command: str,
    output_root: Path | None = None,
    duration_override_minutes: float | None = None,
    preferred_tts: str = "auto",
    keep_work_files: bool = False,
    on_status: StatusCallback | None = None,
) -> BuildResult:
    """Execute the complete autonomous zero-API-key podcast video production pipeline."""
    notify = on_status or _noop_status

    # 0. Parse command & bootstrap studio environment
    spec = parse_command_to_spec(command, duration_override_minutes=duration_override_minutes)
    out_base = output_root or DEFAULT_OUTPUT_ROOT
    episode_dir = out_base / spec.slug
    episode_dir.mkdir(parents=True, exist_ok=True)

    notify(
        "BOOTSTRAP",
        f"Target: '{spec.topic}' | Runtime target: {spec.target_duration_sec / 60:.1f} min "
        f"(~{spec.target_word_count} words across {spec.chapter_count} acts)",
    )
    env = bootstrap_studio(preferred_tts=preferred_tts)
    notify(
        "BOOTSTRAP",
        f"Provisioned zero-key engines -> TTS: {env.tts_backend} | Brain: {env.llm_backend} | FFmpeg: ready",
    )

    work_dir = Path(tempfile.mkdtemp(prefix=f"voxforge_{spec.slug}_"))
    try:
        # 1. Deep Multi-Source Research & Visual Curation
        notify("RESEARCH", f"Investigating '{spec.topic}' & curating 1080p documentary plates...")
        dossier = build_research_dossier(spec, work_dir)
        dossier_json = episode_dir / "research_dossier.json"
        dossier_json.write_text(json.dumps(dossier.to_dict(), indent=2), encoding="utf-8")
        total_plates = sum(len(s.visual_assets) for s in dossier.subtopics)
        notify(
            "RESEARCH",
            f"Compiled {len(dossier.subtopics)} chapter dossiers & {total_plates} graded 1080p visual plates.",
        )

        # 2. Showrunner Script Creation + Stage A Editorial Audit & Rewrite Loop
        notify("SCRIPT", "Drafting multi-act documentary narration script (Alex Vance & Dr. Maya Lin)...")
        draft_script = generate_episode_script(spec, dossier, env)
        notify("AUDIT_A", f"Auditing draft script ({draft_script.total_words} words) & running editorial improvements...")
        script, stage_a = audit_and_improve_script(draft_script, dossier)
        notify(
            "AUDIT_A",
            f"Editorial QA Score: {stage_a.initial_score:.1f} -> {stage_a.final_score:.1f}/100 "
            f"({script.total_words} words, {len(script.all_turns)} narration segments).",
        )

        # 3. Neural Voice Synthesis, Score & Stage B Broadcast Audio Audit + Mastering Loop
        notify(
            "AUDIO",
            f"Synthesizing {len(script.all_turns)} neural documentary voiceover segments ({env.tts_backend}) + score...",
        )
        audio_meta, stage_b = audit_and_improve_audio(script, env, work_dir, episode_dir)
        notify(
            "AUDIT_B",
            f"Cinema Audio QA Score: {stage_b.initial_score:.1f} -> {stage_b.final_score:.1f}/100 | "
            f"Master: {audio_meta.integrated_lufs:.2f} LUFS, {audio_meta.true_peak_dbtp:.2f} dBTP, "
            f"{audio_meta.duration_sec / 60:.2f} min.",
        )

        # 4. Stage C Visual Asset QA + Cinema Documentary 1080p24 Video Rendering
        notify("AUDIT_C", "Auditing optical sharpness, exposure & documentary shot choreography...")
        stage_c = audit_and_improve_visuals(dossier, script, audio_meta.duration_sec, work_dir)

        # Persist final time-coded & QA-healed script JSON
        script_json = episode_dir / "episode_script.json"
        script_json.write_text(json.dumps(script.to_dict(), indent=2), encoding="utf-8")

        notify(
            "VIDEO",
            f"Rendering 1080p24 2.5D cinema, cartography & telemetry documentary shots in parallel...",
        )

        last_pct = -1

        def _on_seg_progress(done: int, total: int) -> None:
            nonlocal last_pct
            pct = int((done / max(1, total)) * 100)
            if pct >= last_pct + 20 or done == total:
                last_pct = pct
                notify("VIDEO", f"Encoded {done}/{total} documentary segments ({pct}%)...")

        video_mp4, srt_path, vtt_path = render_episode_video(
            script=script,
            audio_meta=audio_meta,
            env=env,
            work_dir=work_dir,
            output_dir=episode_dir,
            progress_cb=_on_seg_progress,
        )

        # 5. Stage D Final Broadcast Deliverable Inspection & Self-Certification
        notify("AUDIT_D", "Inspecting muxed MP4 streams, A/V sync, decoded frames & chapter markers...")
        stage_d = inspect_final_deliverable(video_mp4, srt_path, vtt_path, script, audio_meta)

        # 6. Stage E Narration↔Visual Semantic Sync Gate (final rendered-video frame audit)
        notify("AUDIT_E", "Semantic sync gate: inspecting decoded frames vs the narration scene plan...")
        stage_e = audit_av_sync(video_mp4, episode_dir / "scene_plan.json", srt_path, script, audio_meta)
        notify(
            "AUDIT_E",
            f"Narration↔Visual Sync Gate: {stage_e.final_score:.1f}/100 | Passed={stage_e.passed} "
            f"({len(stage_e.checks)} checks — semantic scene match, drift, black/frozen, captions).",
        )

        stages = [stage_a, stage_b, stage_c, stage_d, stage_e]
        overall_init = round(sum(s.initial_score for s in stages) / len(stages), 1)
        overall_final = round(sum(s.final_score for s in stages) / len(stages), 1)
        certified = all(s.passed for s in stages) and overall_final >= 90.0

        report = EpisodeAuditReport(
            episode_slug=spec.slug,
            overall_initial_score=overall_init,
            overall_final_score=overall_final,
            studio_certified=certified,
            stages=stages,
        )

        audit_json = episode_dir / "audit_report.json"
        audit_md = episode_dir / "audit_report.md"
        write_audit_reports(report, script, audio_meta, audit_json, audit_md)

        notify(
            "COMPLETE",
            f"Studio Certified={certified} | Composite Score: {overall_init:.1f} -> {overall_final:.1f}/100 | "
            f"Saved to {video_mp4}",
        )

        return BuildResult(
            output_dir=episode_dir,
            video_mp4_path=video_mp4,
            audio_mp3_path=Path(audio_meta.master_mp3_path),
            subtitles_srt_path=srt_path,
            subtitles_vtt_path=vtt_path,
            script_json_path=script_json,
            dossier_json_path=dossier_json,
            audit_json_path=audit_json,
            audit_md_path=audit_md,
            duration_sec=audio_meta.duration_sec,
            total_words=script.total_words,
            chapter_count=len(script.chapters),
            audit_report=report,
        )
    finally:
        if not keep_work_files:
            shutil.rmtree(work_dir, ignore_errors=True)
