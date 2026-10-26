"""Tests for the narration-aligned scene plan + Stage E semantic A/V sync gate."""

from __future__ import annotations

import numpy as np

from voxforge.models import DialogueTurn, ScenePlan, ScenePlanEntry


def _turn(text: str, camera: str = "broll_cinema") -> DialogueTurn:
    return DialogueTurn(
        turn_id=1,
        chapter_index=1,
        speaker="HOST_A",
        text=text,
        spoken_text=text,
        camera=camera,  # type: ignore[arg-type]
        start_sec=0.0,
        end_sec=20.0,
        duration_sec=20.0,
    )


def test_split_turn_clauses_caps_and_merges() -> None:
    from voxforge.video.renderer import _split_turn_clauses

    long_text = (
        "The Nabataeans cornered the ancient incense trade, and their engineers cut "
        "water channels into the sandstone. Caravans carried frankincense across the desert, "
        "while cisterns stored every winter rain, and the city grew rich on tolls."
    )
    clauses = _split_turn_clauses(long_text)
    assert 2 <= len(clauses) <= 5
    assert all(len(c) >= 12 for c in clauses)
    assert _split_turn_clauses("Short turn.") == ["Short turn."]


def test_family_for_intent_mapping() -> None:
    from voxforge.video.renderer import _family_for_intent

    assert _family_for_intent("eng", True, 0) == "split_fact_card"
    assert _family_for_intent("geo", True, 0) == "wide_studio"
    assert _family_for_intent("quote", True, 0) == "quote_card"
    assert _family_for_intent("people", True, 0) == "studio_insert"
    assert _family_for_intent("scene", True, 0) == "broll_cinema"
    # Unmapped (generic) scenes never fall back to monument cinema:
    assert _family_for_intent("scene", False, 0) == "split_fact_card"
    assert _family_for_intent("scene", False, 1) == "studio_insert"


def test_planner_returns_clause_triples_covering_turn() -> None:
    from voxforge.video.renderer import _plan_turn_editorial_beats

    turn = _turn(
        "The wall began as rammed-earth forts, and the Ming rebuilt it in stone. "
        "Watchtowers carried smoke signals across the mountains, because generals "
        "needed news within hours of a raid."
    )
    env = np.full(2500, 0.12, dtype=np.float32)
    beats = _plan_turn_editorial_beats(
        turn, 20.0, 40.0, env,
        is_episode_opener=False, is_episode_closer=False,
        scene_context="Great Wall of China Beijing",
    )
    assert len(beats) >= 3
    prev = 20.0
    for cam, end_t, clause in beats:
        assert 2.0 <= end_t - prev <= 8.0
        prev = end_t
    assert abs(prev - 40.0) < 1e-6
    # Geo clause ("mountains"/"across") or eng clause must drive at least one non-broll family
    fams = {cam for cam, _, _ in beats}
    assert fams & {"wide_studio", "split_fact_card", "studio_insert", "quote_card"}


def test_scene_plan_serialization_roundtrip() -> None:
    entry = ScenePlanEntry(
        scene_id=4, seg_index=12, chapter_index=2, turn_id=7,
        narration_clause="Caravans carried frankincense across the desert",
        narration_start_sec=10.5, narration_end_sec=15.0,
        subject="Petra", action_or_event="geo",
        required_visual="3D raycasted orbital Earth GIS map",
        visual_family="wide_studio", visual_asset_source="pure-code:raycast_orbital_earth_globe",
        scene_start_sec=10.5, scene_end_sec=15.0, on_screen_text="Petra",
        relevance_criteria="family=wide_studio must depict intent=geo",
    )
    plan = ScenePlan(topic="T", entries=[entry])
    d = plan.to_dict()
    assert d["scenes"][0]["visual_family"] == "wide_studio"
    assert d["scenes"][0]["validation_status"] == "pending"


def test_frame_group_classifier_separates_families() -> None:
    from voxforge.audit.av_sync_qa import classify_frame_group

    rng = np.random.default_rng(7)
    # geo: dominant blue sphere mass on black starfield
    geo = np.zeros((360, 640, 3), dtype=np.uint8)
    geo[80:300, 160:480] = (180, 90, 30)
    geo += rng.integers(0, 6, geo.shape, dtype=np.uint8)
    assert classify_frame_group(geo) == "geo"
    # monument: warm stone luma everywhere
    mon = np.full((360, 640, 3), 150, dtype=np.uint8)
    mon += rng.integers(0, 20, mon.shape, dtype=np.uint8)
    assert classify_frame_group(mon) == "monument"
    # darkhud: near-black frame with sparse bright cyan/amber glyphs
    hud = np.full((360, 640, 3), 18, dtype=np.uint8)
    hud[50:300:7, 40:600] = (240, 180, 60)
    assert classify_frame_group(hud) == "darkhud"
    # quote: dark backdrop with large white text block
    quote = np.full((360, 640, 3), 22, dtype=np.uint8)
    quote[100:260, 60:560] = (235, 235, 235)
    assert classify_frame_group(quote) == "quote"


def test_generic_explanatory_simulation_renders() -> None:
    from voxforge.video.procedural_world import SCENE_CLASSICAL_GENERIC, PhysicsSimulationEngine

    eng = PhysicsSimulationEngine()
    frame = eng.render_simulation_frame(
        SCENE_CLASSICAL_GENERIC, 0.8, 1.0, 640, 360,
        hint="Trade flows grew 25 percent as the network connected coastal cities",
    )
    assert frame.shape == (360, 640, 3)
    assert float(frame.mean()) > 5.0  # not a black frame; glyphs/panels drawn
