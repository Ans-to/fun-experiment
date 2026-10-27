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


def test_clause_scene_resolution_uses_full_active_turn_before_chapter_fallback() -> None:
    from voxforge.video.procedural_world import SCENE_MACHU_PICCHU, resolve_procedural_scene_id
    from voxforge.video.renderer import _scene_resolution_text, _shot_lens_variant

    clause = "Perched on a narrow granite fault saddle nearly eight"
    active_turn = (
        "Perched on a narrow granite fault saddle nearly eight thousand feet above sea level, "
        "Machu Picchu rises over the Urubamba River."
    )
    broad_chapter = "Citadels of the Americas: Chichén Itzá & Machu Picchu"
    hint = _scene_resolution_text(clause, active_turn, broad_chapter)
    sid, title = resolve_procedural_scene_id(hint)
    assert sid == SCENE_MACHU_PICCHU
    assert "Machu Picchu" in title


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
    for yq in range(110, 250, 20):
        quote[yq : yq + 6, 60:560] = (235, 235, 235)
    assert classify_frame_group(quote) == "quote"
    # noon-bright warm stone must NOT read as quote despite high white-pixel mass
    noon = np.full((360, 640, 3), 205, dtype=np.uint8)
    noon[..., 0] -= 12
    noon += rng.integers(0, 14, noon.shape, dtype=np.uint8)
    assert classify_frame_group(noon) == "monument"
    # night-lit statue: small warm mass on dark sky
    night = np.full((360, 640, 3), 30, dtype=np.uint8)
    night[90:300, 260:380] = (165, 160, 150)
    assert classify_frame_group(night) == "monument"
    # dense blue LiDAR point cloud (no landmass, incoherent blocks) is darkhud, not geo
    lidar = np.full((360, 640, 3), 6, dtype=np.uint8)
    ys, xs = np.mgrid[0:360, 0:640]
    ring = ((ys - 180) ** 2 / 90 ** 2 + (xs - 320) ** 2 / 240 ** 2)
    dots = ((ring > 0.85) & (ring < 1.15) & ((xs + ys) % 5 < 2)) | ((ring > 0.4) & (ring < 0.6) & ((xs * 3 + ys) % 7 < 2))
    lidar[dots] = (230, 140, 40)
    assert classify_frame_group(lidar) == "darkhud"


def test_generic_explanatory_simulation_renders() -> None:
    from voxforge.video.procedural_world import SCENE_CLASSICAL_GENERIC, PhysicsSimulationEngine

    eng = PhysicsSimulationEngine()
    frame = eng.render_simulation_frame(
        SCENE_CLASSICAL_GENERIC, 0.8, 1.0, 640, 360,
        hint="Trade flows grew 25 percent as the network connected coastal cities",
    )
    assert frame.shape == (360, 640, 3)
    assert float(frame.mean()) > 5.0  # not a black frame; glyphs/panels drawn


def test_repeated_landmark_shots_rotate_lens_coverage_by_segment() -> None:
    from voxforge.video.renderer import _shot_lens_variant

    assert _shot_lens_variant(0, 33) != _shot_lens_variant(0, 34)
    assert _shot_lens_variant(0, 42) != _shot_lens_variant(0, 48)


def test_fact_card_metrics_are_filtered_to_the_active_landmark() -> None:
    from types import SimpleNamespace

    from voxforge.video.compositor import _metrics_for_scene

    metrics = [
        SimpleNamespace(label="Central Finial Height", value="73 Meters"),
        SimpleNamespace(label="Artisan Workforce", value="20,000 specialists"),
        SimpleNamespace(label="Corcovado Peak", value="710 m above Rio"),
        SimpleNamespace(label="Arm Wingspan", value="28 meters"),
    ]
    taj = _metrics_for_scene(metrics, "Taj Mahal, Agra")
    christ = _metrics_for_scene(metrics, "Christ the Redeemer, Rio de Janeiro")
    assert [m.label for m in taj] == ["Central Finial Height", "Artisan Workforce"]
    assert [m.label for m in christ] == ["Corcovado Peak", "Arm Wingspan"]
    paired = [
        SimpleNamespace(label="Elevation", value="2,430 m"),
        SimpleNamespace(label="Subsurface Effort", value="60% drains"),
        SimpleNamespace(label="Total Calendar Steps", value="365"),
        SimpleNamespace(label="Pyramid Nested Shells", value="3"),
    ]
    assert [m.label for m in _metrics_for_scene(paired, "Machu Picchu & Huayna Picchu")] == [
        "Elevation", "Subsurface Effort"
    ]
    assert [m.label for m in _metrics_for_scene(paired, "El Castillo, Chichén Itzá")] == [
        "Total Calendar Steps", "Pyramid Nested Shells"
    ]


def test_stage_e_rejects_plan_subject_that_conflicts_with_full_narration_turn() -> None:
    from types import SimpleNamespace

    from voxforge.audit.av_sync_qa import _planned_subject_mismatch
    from voxforge.video.procedural_world import SCENE_CHICHEN_ITZA, SCENE_MACHU_PICCHU

    turn = SimpleNamespace(
        turn_id=13,
        chapter_index=0,
        text=(
            "Perched on a narrow granite fault saddle nearly eight thousand feet above sea level, "
            "Machu Picchu rises over the Urubamba River."
        ),
    )
    chapter = SimpleNamespace(chapter_index=0, title="Citadels of the Americas", location="Chichén Itzá and Machu Picchu")
    script = SimpleNamespace(all_turns=[turn], chapters=[chapter])
    bad_entry = {
        "turn_id": 13,
        "seg_index": 42,
        "narration_clause": "Perched on a narrow granite fault saddle nearly eight",
        "scene_id": SCENE_CHICHEN_ITZA,
        "subject": "El Castillo, Chichén Itzá",
    }
    assert _planned_subject_mismatch(bad_entry, script) is not None
    bad_entry["scene_id"] = SCENE_MACHU_PICCHU
    assert _planned_subject_mismatch(bad_entry, script) is None


def test_stage_e_human_review_queue_on_uncertain_scene(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    import cv2

    from voxforge.audit.av_sync_qa import audit_av_sync

    mp4 = tmp_path / "film.mp4"
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 176))
    globe = np.zeros((176, 320, 3), np.uint8)
    globe[38:138, 110:210] = (180, 90, 30)
    hud = np.full((176, 320, 3), 18, np.uint8)
    hud[30:150:7, 20:300] = (240, 180, 60)
    for i in range(240):  # 4s globe then 6s HUD: partial agreement with a geo plan entry
        vw.write(globe if i < 96 else hud)
    vw.release()

    plan = {
        "topic": "synthetic",
        "scenes": [
            {
                "scene_id": 0, "seg_index": 0, "chapter_index": 0, "turn_id": 1,
                "narration_clause": "synthetic uncertain scene",
                "narration_start_sec": 0.0, "narration_end_sec": 10.0,
                "subject": "synthetic", "action_or_event": "geo",
                "required_visual": "globe", "visual_family": "wide_studio",
                "visual_asset_source": "test", "scene_start_sec": 0.0, "scene_end_sec": 10.0,
                "on_screen_text": "", "relevance_criteria": "", "validation_status": "pending",
            }
        ],
    }
    plan_path = tmp_path / "scene_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    script = SimpleNamespace(all_turns=[])
    audio = SimpleNamespace(duration_sec=10.0)
    stage = audit_av_sync(mp4, plan_path, tmp_path / "c.srt", script, audio)
    out_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert out_plan["scenes"][0]["validation_status"] == "needs_human_review"
    assert out_plan["stage_e"]["human_review_queue"] == 1
    assert stage.passed is False
    frames = sorted((tmp_path / "human_review").glob("seg0000_p*.jpg"))
    assert len(frames) >= 3
