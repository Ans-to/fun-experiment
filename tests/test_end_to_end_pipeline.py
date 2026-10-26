"""End-to-end integration test of the single-command VoxForge Studio pipeline."""

from pathlib import Path

from voxforge.pipeline import run_podcast_pipeline


def test_full_single_command_studio_pipeline(tmp_path: Path) -> None:
    result = run_podcast_pipeline(
        command="build a 90 second podcast on 7 wonders of the world",
        output_root=tmp_path,
        preferred_tts="auto",
    )

    assert result.video_mp4_path.exists()
    assert result.video_mp4_path.stat().st_size > 250_000
    assert result.audio_mp3_path.exists()
    assert result.subtitles_srt_path.exists()
    assert result.subtitles_vtt_path.exists()
    assert result.audit_json_path.exists()
    assert result.audit_md_path.exists()

    assert result.audit_report.studio_certified is True
    assert result.audit_report.overall_final_score >= 90.0
    assert 65.0 <= result.duration_sec <= 125.0
    md_text = result.audit_md_path.read_text(encoding="utf-8")
    assert "15-Point Studio Broadcast Elements Verification" in md_text
    assert "Branded Intro Bumper" in md_text
    assert "Branded Outro / End Card" in md_text
