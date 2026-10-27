"""Tests for visual QA, procedural 1080p blueprint plates, documentary motion graphics & captioning."""

from pathlib import Path

from voxforge.models import ChapterScript, DialogueTurn, EpisodeScript, KeyMetric, ShowSpec
from voxforge.research.knowledge_packs import get_seven_wonders_dossier
from voxforge.video.compositor import MulticamCompositor
from voxforge.video.infographics import measure_image_quality, render_procedural_documentary_plate
from voxforge.video.subtitles import (
    build_all_caption_cues,
    write_ffmetadata_chapters,
    write_segment_ass,
    write_srt_and_vtt,
)


def test_procedural_plate_hd_and_sharpness(tmp_path: Path) -> None:
    dossier = get_seven_wonders_dossier("7 Wonders of the World")
    out = tmp_path / "plate.jpg"
    render_procedural_documentary_plate(dossier.subtopics[1], out, variant=0)
    w, h, sharp, lum = measure_image_quality(out)
    assert (w, h) == (1920, 1080)
    assert sharp >= 35.0
    assert 20.0 <= lum <= 220.0


def test_subtitles_and_chapters_generation(tmp_path: Path) -> None:
    spec = ShowSpec(
        raw_command="test",
        topic="Test Topic",
        show_title="DEEP HORIZON",
        episode_title="Test Documentary",
        episode_subtitle="Sub",
        target_duration_sec=60.0,
        target_word_count=120,
        chapter_count=1,
    )
    turn = DialogueTurn(
        turn_id=1,
        chapter_index=1,
        speaker="HOST_A",
        text="Across forty-five centuries, ancient engineers raised limestone monuments across the Giza Plateau.",
        spoken_text="Across forty-five centuries, ancient engineers raised limestone monuments across the Giza Plateau.",
        camera="wide_studio",
        start_sec=3.6,
        end_sec=9.4,
        duration_sec=5.8,
    )
    ch = ChapterScript(
        chapter_index=1,
        title="The Great Pyramid",
        subtitle="Precision Stone",
        era_or_date="2560 BCE",
        location="Egypt",
        music_cue="wonder_pad",
        turns=[turn],
        start_sec=0.0,
        end_sec=10.0,
    )
    script = EpisodeScript(spec=spec, chapters=[ch])
    cues = build_all_caption_cues(script)
    assert len(cues) >= 1

    srt = tmp_path / "captions.srt"
    vtt = tmp_path / "captions.vtt"
    ass = tmp_path / "seg.ass"
    ffm = tmp_path / "chapters.ffmeta"

    write_srt_and_vtt(cues, srt, vtt)
    write_segment_ass(cues, 3.0, 10.0, ass)
    write_ffmetadata_chapters(script, 10.0, ffm)

    assert "ALEX VANCE:" in srt.read_text()
    assert vtt.read_text().startswith("WEBVTT")
    assert "Dialogue:" in ass.read_text()
    assert "[CHAPTER]" in ffm.read_text()


def test_compositor_renders_all_15_documentary_shot_types(tmp_path: Path) -> None:
    comp = MulticamCompositor()
    spec = ShowSpec(
        raw_command="build a 15 minute documentary on 7 wonders of the world",
        topic="7 Wonders of the World",
        show_title="DEEP HORIZON",
        episode_title="7 Wonders of the World: Architecture of the Impossible",
        episode_subtitle="Original Documentary",
        target_duration_sec=900.0,
        target_word_count=1800,
        chapter_count=3,
    )
    turn = DialogueTurn(
        turn_id=1,
        chapter_index=1,
        speaker="HOST_A",
        text="Every monument solved a physical impossibility using geometry and local stone.",
        spoken_text="Every monument solved a physical impossibility using geometry and local stone.",
        camera="wide_studio",
        pull_quote="Every monument solved a physical impossibility using geometry and local stone.",
        insert_variant="hands_notes",
        fact_callout_title="The Great Pyramid",
        fact_metrics=[KeyMetric("Original Height", "146.6 m"), KeyMetric("Cardinal Error", "0.05°")],
        is_chapter_opener=True,
    )
    ch = ChapterScript(
        chapter_index=1,
        title="The Great Pyramid of Giza",
        subtitle="Precision Stone",
        era_or_date="c. 2560 BCE",
        location="Giza Plateau, Egypt",
        music_cue="wonder_pad",
        turns=[turn],
    )
    script = EpisodeScript(spec=spec, chapters=[ch])

    intro_l, intro_ov = comp.render_intro_bumper(script, tmp_path / "intro_bg.jpg", tmp_path / "intro_ov.png")
    wide_l, wide_ov = comp.render_wide_studio(turn, ch, 3, "DEEP HORIZON", tmp_path / "wide.jpg", episode_title=spec.episode_title)
    cu_l, cu_ov = comp.render_host_closeup(turn, ch, 3, "DEEP HORIZON", tmp_path / "cu.jpg")
    react_l, react_ov = comp.render_reaction_shot(turn, ch, 3, "DEEP HORIZON", tmp_path / "react.jpg")
    ins_l, ins_ov = comp.render_studio_insert(turn, ch, 3, "DEEP HORIZON", tmp_path / "ins.jpg", tmp_path / "ins_hud.png")
    quote_l = comp.render_quote_card(turn, ch, 3, "DEEP HORIZON", tmp_path / "quote.jpg")
    fact_l = comp.render_split_fact_card(turn, ch, 3, "DEEP HORIZON", tmp_path / "fact.jpg")
    outro_l = comp.render_outro_card(script, tmp_path / "outro.jpg")

    assert intro_l.anim_mode == "broll_depth"
    assert wide_l.anim_mode == "geo_map" and wide_l.transient_lt_path is not None
    assert cu_l.anim_mode == "broll_depth" and cu_l.transient_lt_path is not None
    assert react_l.anim_mode == "archival_desk" and not react_l.anim_is_speaking
    assert ins_l.anim_mode == "archival_desk" and ins_l.anim_is_speaking
    assert quote_l.anim_mode == "none" and quote_l.pip_rect is None and quote_l.wave_x is None
    assert fact_l.anim_mode == "none" and len(fact_l.telemetry_metrics) == 2

    for layout in (intro_l, cu_l, ins_l, quote_l, fact_l, outro_l):
        w, h, _, lum = measure_image_quality(layout.image_path)
        assert (w, h) == (1920, 1080)
        assert 15.0 <= lum <= 235.0
    for ov in (intro_ov, wide_ov, cu_ov, react_ov, ins_ov):
        assert ov[0].exists()


def test_zero_stock_or_seed_photos_in_assets() -> None:
    from voxforge.config import ASSETS_DIR

    static_imgs = [
        p for p in ASSETS_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    assert static_imgs == [], f"Expected zero stock/static photos in assets, found: {static_imgs}"


def test_live_24fps_documentary_motion_engines_not_still(tmp_path: Path) -> None:
    import numpy as np
    from voxforge.video.animator import StudioCharacterAnimator, extract_speech_motion_curves

    sr = 24000
    t = np.linspace(0.0, 2.0, 2 * sr, dtype=np.float32)
    gate = (np.sin(2.0 * np.pi * 2.5 * t) > 0.0).astype(np.float32)
    speech = gate * (0.35 * np.sin(2.0 * np.pi * 210.0 * t) + 0.25 * np.sin(2.0 * np.pi * 720.0 * t))
    curves = extract_speech_motion_curves(speech, sr, 12, fps=24)

    assert float(curves.energy.min()) < 0.05
    assert float(curves.energy.max()) > 0.65

    anim = StudioCharacterAnimator()
    dummy_path = tmp_path / "code_frame.jpg"

    # 1. Pillar 4: True 3D Raycasted Curved-Horizon Planet Earth in Deep Space
    map_frames = list(anim.iter_geo_map_frames("Petra Jordan", "Giza Egypt", "c. 1st Century BCE", 8, curves=curves))
    assert len(map_frames) == 8 and map_frames[0].shape == (1080, 1920, 3)
    assert float(np.mean(np.abs(map_frames[-1].astype(float) - map_frames[0].astype(float)))) > 1.5

    # 2. Pillar 2 & 3: Live Physics Simulation (`seg_seed=0`) & Volumetric 3D LiDAR (`seg_seed=1`)
    sim_frames = list(
        anim.iter_archival_desk_frames(
            dummy_path, dummy_path, "Great Pyramid", "Khufu", "Great Pyramid", 6, seg_seed=0, scene_hint="Khufu Giza"
        )
    )
    assert len(sim_frames) == 6 and sim_frames[0].shape == (1080, 1920, 3)
    assert float(np.mean(np.abs(sim_frames[-1].astype(float) - sim_frames[0].astype(float)))) > 0.4

    lidar_frames = list(
        anim.iter_archival_desk_frames(
            dummy_path, dummy_path, "Taj Mahal", "Agra", "Taj Mahal", 6, seg_seed=1, scene_hint="Taj Mahal"
        )
    )
    assert len(lidar_frames) == 6 and lidar_frames[0].shape == (1080, 1920, 3)
    assert float(np.mean(np.abs(lidar_frames[-1].astype(float) - lidar_frames[0].astype(float)))) > 0.8

    # 3. Staggered 4-Metric Engineering Telemetry + Live Rotating 3D LiDAR Viewport
    tel_frames = list(
        anim.iter_telemetry_split_frames(
            dummy_path,
            "The Roman Colosseum",
            "70-80 CE",
            "Rome, Italy",
            [("Capacity", "50,000+"), ("Concrete Vaults", "80 Arches")],
            "#E5A93C",
            10,
            curves=curves,
            scene_hint="Colosseum Rome",
        )
    )
    assert len(tel_frames) == 10 and tel_frames[0].shape == (1080, 1920, 3)
    assert float(np.mean(np.abs(tel_frames[-1].astype(float) - tel_frames[0].astype(float)))) > 1.0

    # 4. Pillar 1: Live 3D Z-Buffered Architectural Monument Cinema (`Monument3DEngine`)
    broll_frames = list(
        anim.iter_broll_depth_frames(dummy_path, 6, seg_seed=1, curves=curves, scene_hint="Petra Treasury")
    )
    assert len(broll_frames) == 6 and broll_frames[0].shape == (1080, 1920, 3)
    assert float(np.mean(np.abs(broll_frames[-1].astype(float) - broll_frames[0].astype(float)))) > 1.5


def test_subturn_editorial_director_prevents_long_holds() -> None:
    import numpy as np
    from voxforge.video.renderer import _plan_turn_editorial_beats

    turn = DialogueTurn(
        turn_id=4,
        chapter_index=1,
        speaker="HOST_B",
        text="Built around twenty-five sixty BCE for Pharaoh Khufu, the Great Pyramid stood unsurpassed for millennia.",
        spoken_text="Built around twenty-five sixty BCE for Pharaoh Khufu, the Great Pyramid stood unsurpassed for millennia.",
        camera="reaction_shot",
        start_sec=20.0,
        end_sec=41.0,
        duration_sec=21.0,
    )
    env = np.full(2500, 0.1, dtype=np.float32)
    beats = _plan_turn_editorial_beats(
        turn, 20.0, 41.0, env,
        is_episode_opener=False, is_episode_closer=False, lead_attribution_sec=2.9,
    )
    assert len(beats) >= 3
    assert beats[0][0] == "closeup_host_b"
    prev = 20.0
    for cam, end_t, clause in beats:
        dur = end_t - prev
        assert 1.8 <= dur <= 8.0
        prev = end_t
    assert abs(prev - 41.0) < 1e-6


def test_quote_resolution_does_not_borrow_unrelated_chapter_quote() -> None:
    from voxforge.video.compositor import resolve_historical_quote

    unrelated = "The question is simple: what did these builders understand about stone, weight, and time?"
    quote, author, citation = resolve_historical_quote(unrelated, unrelated)
    assert "house of Artemis" not in quote
    assert quote == unrelated
    assert author == "VOXFORGE NARRATION"
    assert citation.startswith("Narration excerpt")

    matched = "Antipater described the house of Artemis and its place among the Seven Wonders."
    quote2, author2, _ = resolve_historical_quote(matched, matched)
    assert "house of Artemis" in quote2
    assert author2 == "ANTIPATER OF SIDON"
