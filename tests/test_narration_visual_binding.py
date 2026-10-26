"""Regression tests for clause-level narration→visual subject binding."""

from __future__ import annotations

import dataclasses

import numpy as np

from voxforge.audit.av_sync_qa import audit_scene_plan_semantics
from voxforge.models import (
    ChapterScript,
    DialogueTurn,
    KeyMetric,
    SubTopicDossier,
    VisualAsset,
)
from voxforge.script.showrunner import assign_multicam_choreography
from voxforge.video.procedural_world import (
    SCENE_CHICHEN_ITZA,
    SCENE_CLASSICAL_GENERIC,
    SCENE_COLOSSEUM,
    SCENE_MACHU_PICCHU,
    SCENE_PETRA,
    SCENE_TAJ_MAHAL,
    detect_explicit_scene_id,
    resolve_monument_scene_id,
    resolve_visual_subject,
)
from voxforge.video.renderer import _bind_beat_turn, _plan_turn_editorial_beats


def _turn(turn_id: int, text: str) -> DialogueTurn:
    return DialogueTurn(
        turn_id=turn_id,
        chapter_index=1,
        speaker="HOST_A",
        text=text,
        spoken_text=text,
        camera="broll_cinema",
        music_cue="wonder_pad",
        start_sec=0.0,
        end_sec=20.0,
        duration_sec=20.0,
    )


def _chapter(turns: list[DialogueTurn]) -> ChapterScript:
    return ChapterScript(
        chapter_index=1,
        title="Petra & The Roman Colosseum",
        subtitle="Stone hydraulics and imperial concrete",
        era_or_date="Ancient Era",
        location="Southern Jordan & Rome, Italy",
        music_cue="wonder_pad",
        turns=turns,
    )


def test_weighted_scene_resolution_follows_spoken_subject() -> None:
    petra = "Hidden behind the Siq gorge, Petra was the Nabataean caravan capital"
    colosseum = "Built over Emperor Nero's lake, the Flavian Amphitheatre dominated Rome"
    trade = (
        "Frankincense, Indian spices and Dead Sea bitumen were prized by Egypt and Rome "
        "|| Petra & The Roman Colosseum: Stone Hydraulics & Imperial Concrete"
    )
    machu = (
        "Perched above the Urubamba River, Machu Picchu is a seismic design triumph "
        "|| Citadels of the Americas: Chichén Itzá & Machu Picchu"
    )
    taj = (
        "Commissioned by Shah Jahan for Mumtaz Mahal, the Taj Mahal rises at Agra "
        "|| Mughal Marble & Modern Horizon: The Taj Mahal & Christ the Redeemer"
    )

    assert resolve_monument_scene_id(petra) == SCENE_PETRA
    assert resolve_monument_scene_id(colosseum) == SCENE_COLOSSEUM
    assert resolve_monument_scene_id(trade) == SCENE_PETRA
    assert resolve_monument_scene_id(machu) == SCENE_MACHU_PICCHU
    assert resolve_monument_scene_id(taj) == SCENE_TAJ_MAHAL
    assert resolve_monument_scene_id("how volcanoes erupt reshape the planet") == SCENE_CLASSICAL_GENERIC
    assert detect_explicit_scene_id("how volcanoes erupt reshape the planet") is None


def test_generic_visual_subject_is_topic_aware() -> None:
    sid, subject = resolve_visual_subject(
        "magma rises through the crust", topic="How Volcanoes Erupt"
    )
    assert sid == SCENE_CLASSICAL_GENERIC
    assert "How Volcanoes Erupt" in subject
    assert "classical sanctuary" not in subject.lower()


def test_showrunner_binds_each_turn_to_its_own_subject() -> None:
    turns = [
        _turn(1, "Hidden behind the Siq gorge, Petra was the Nabataean caravan capital."),
        _turn(2, "Built over Emperor Nero's lake, the Flavian Amphitheatre dominated Rome."),
    ]
    chapter = _chapter(turns)
    asset = VisualAsset(
        asset_id="chapter-level",
        subject_tag="chapter",
        file_path="/tmp/unused.jpg",
        width=1920,
        height=1080,
        sharpness_score=50.0,
        mean_luminance=100.0,
        caption="Volumetric 3D LiDAR & X-Ray Blueprint · Flavian Amphitheatre & Subterranean Hypogeum, Rome",
        credit="VoxForge",
        license_name="100% Algorithmic Code Synthesis",
        is_procedural=True,
    )
    subtopic = SubTopicDossier(
        index=1,
        title=chapter.title,
        subtitle=chapter.subtitle,
        era_or_date=chapter.era_or_date,
        location=chapter.location,
        summary="A mixed-subject act.",
        origin_and_context=[],
        engineering_and_mechanics=[],
        human_stories_and_paradoxes=[],
        legacy_and_modern_science=[],
        metrics=[KeyMetric("Act", "Mixed")],
        visual_queries=["Petra", "Colosseum"],
        visual_assets=[asset],
    )

    assign_multicam_choreography([chapter], [subtopic], episode_topic="Ancient Monuments")

    assert turns[0].visual_scene_id == SCENE_PETRA
    assert "Petra" in turns[0].visual_caption
    assert turns[1].visual_scene_id == SCENE_COLOSSEUM
    assert "Flavian" in turns[1].visual_caption
    assert turns[0].visual_caption != turns[1].visual_caption


def test_beat_binding_is_isolated_from_original_turn() -> None:
    original = _turn(
        7,
        "Petra controlled the incense caravans, while the Flavian Amphitheatre "
        "crowds moved through eighty radial gates.",
    )
    original = dataclasses.replace(
        original,
        visual_caption="Volumetric 3D LiDAR & X-Ray Blueprint · chapter subject",
    )
    chapter = _chapter([original])

    petra_turn, petra_sid, petra_subject = _bind_beat_turn(
        original, "Petra controlled the incense caravans", chapter, "Ancient Monuments", 1
    )
    rome_turn, rome_sid, _ = _bind_beat_turn(
        original,
        "the Flavian Amphitheatre crowds moved through eighty radial gates",
        chapter,
        "Ancient Monuments",
        2,
    )

    assert original.text.startswith("Petra controlled")
    assert original.visual_scene_id is None
    assert petra_turn.text.startswith("Petra")
    assert rome_turn.text.startswith("the Flavian")
    assert (petra_sid, rome_sid) == (SCENE_PETRA, SCENE_COLOSSEUM)
    assert "Petra" in petra_subject


def test_plan_beats_never_use_placeholder_narration() -> None:
    turn = _turn(
        11,
        "The Nabataeans cornered the ancient incense trade, and their engineers cut "
        "water channels into the sandstone. Caravans carried frankincense across the desert, "
        "while cisterns stored every winter rain, and the city grew rich on tolls.",
    )
    env = np.full(1500, 0.12, dtype=np.float32)

    for opener, closer in ((True, False), (False, True), (False, False)):
        beats = _plan_turn_editorial_beats(
            turn,
            0.0,
            30.0,
            env,
            is_episode_opener=opener,
            is_episode_closer=closer,
            scene_context="Petra Jordan",
        )
        assert beats
        assert all(clause and clause.strip() for _, _, clause in beats)
        assert all("(voice attribution" not in clause for _, _, clause in beats)


def test_stage_e_plan_gate_detects_wrong_subject_and_placeholder() -> None:
    plan = {
        "topic": "Ancient Monuments",
        "scenes": [
            {
                "narration_clause": "Hidden behind the Siq gorge, Petra was the caravan capital",
                "scene_id": SCENE_COLOSSEUM,
                "subject": "Flavian Amphitheatre & Subterranean Hypogeum, Rome",
                "visual_family": "broll_cinema",
            },
            {
                "narration_clause": "The Flavian Amphitheatre moved crowds through radial gates",
                "scene_id": SCENE_COLOSSEUM,
                "subject": "Flavian Amphitheatre & Subterranean Hypogeum, Rome",
                "visual_family": "broll_cinema",
            },
            {
                "narration_clause": "Magma rises through the crust",
                "scene_id": SCENE_CLASSICAL_GENERIC,
                "subject": "Volcanoes — explanatory model",
                "visual_family": "split_fact_card",
            },
            {
                "narration_clause": "(voice attribution / establishing beat)",
                "scene_id": SCENE_TAJ_MAHAL,
                "subject": "Taj Mahal, Agra",
                "visual_family": "broll_cinema",
            },
        ],
    }

    failures = audit_scene_plan_semantics(plan)
    reasons = [reason for _, reason in failures]
    assert len(failures) == 2
    assert any("subject_mismatch" in reason for reason in reasons)
    assert any("missing_exact_narration_clause" in reason for reason in reasons)


def test_scene_plan_gate_accepts_chichen_itza_binding() -> None:
    plan = {
        "scenes": [
            {
                "narration_clause": "At Chichén Itzá, the El Castillo steps diffract a handclap",
                "scene_id": SCENE_CHICHEN_ITZA,
                "subject": "El Castillo, Chichén Itzá",
                "visual_family": "split_fact_card",
            }
        ]
    }
    assert audit_scene_plan_semantics(plan) == []
