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


def test_camera_for_directive_mapping() -> None:
    from voxforge.video.renderer import _camera_for_directive
    from voxforge.video.semantic_router import (
        MODALITY_DIAGRAM,
        MODALITY_GLOBE,
        MODALITY_LIDAR,
        MODALITY_MONUMENT,
        MODALITY_QUOTE,
        VisualDirective,
    )

    def d(modality: str, treatment: str = "exterior_wide", template: str = "") -> VisualDirective:
        return VisualDirective("k", "Label", modality, -1 if modality != MODALITY_MONUMENT else 0,
                               template, treatment, "Label", 0.9, "semantic_category")

    assert _camera_for_directive(d(MODALITY_DIAGRAM, template="node_network"), "closeup_host_a") == "semantic_diagram"
    assert _camera_for_directive(d(MODALITY_GLOBE, "orbital_context"), "closeup_host_a") == "wide_studio"
    assert _camera_for_directive(d(MODALITY_QUOTE, "kinetic_typography"), "closeup_host_a") == "quote_card"
    assert _camera_for_directive(d(MODALITY_LIDAR, "lidar_scan"), "closeup_host_a") == "studio_insert"
    assert _camera_for_directive(d(MODALITY_MONUMENT, "detail_closeup"), "closeup_host_b") == "closeup_host_b"
    assert _camera_for_directive(d(MODALITY_MONUMENT, "exterior_wide"), "closeup_host_a") == "broll_cinema"


def test_planner_returns_clause_bound_directives_without_placeholders() -> None:
    from voxforge.video.renderer import _plan_turn_editorial_beats
    from voxforge.video.semantic_router import SemanticShotDirector

    turn = _turn(
        "The wall began as rammed-earth forts, and the Ming rebuilt it in stone. "
        "Watchtowers carried smoke signals across the mountains, because generals "
        "needed news within hours of a raid."
    )
    env = np.full(2500, 0.12, dtype=np.float32)
    director = SemanticShotDirector("The Great Wall of China")
    beats = _plan_turn_editorial_beats(
        turn, 20.0, 40.0, env,
        director,
        "The Great Wall",
        "Beijing China",
        scene_context="Great Wall of China Beijing",
    )
    assert len(beats) >= 3
    prev = 20.0
    pairs: list[tuple[str, str]] = []
    for cam, end_t, clause, directive in beats:
        assert 2.0 <= end_t - prev <= 8.0
        prev = end_t
        # Every beat is bound to a real clause and a resolved subject — no placeholders.
        assert clause and len(clause) >= 12
        assert directive.subject_key and directive.subject_label
        pairs.append((directive.subject_key, directive.treatment))
    assert abs(prev - 40.0) < 1e-6
    # No consecutive identical (subject, treatment) pair — the anti-loop invariant.
    assert all(pairs[i] != pairs[i - 1] for i in range(1, len(pairs)))


def test_post_merge_directives_always_match_their_stored_clause() -> None:
    """Regression: directives resolved before beat merging orphaned clauses, producing
    monument bindings whose stored clause named no monument (hard-gate failures seg02/03/14)."""
    from voxforge.audit.av_sync_qa import _subject_hard_gate
    from voxforge.video.renderer import _plan_turn_editorial_beats
    from voxforge.video.semantic_router import SemanticShotDirector

    turn = _turn(
        "Out of the original Seven Wonders cataloged by Greek travelers, earthquakes and fire "
        "destroyed six, leaving only a 4,500-year-old limestone mountain on the edge of Cairo. "
        "The question is simple: what did these builders understand about stone and weight?"
    )
    env = np.full(2500, 0.1, dtype=np.float32)
    director = SemanticShotDirector("Seven Wonders of the Ancient World")
    beats = _plan_turn_editorial_beats(
        turn, 4.0, 26.0, env, director, "Origins of Wonder", "Mediterranean",
        scene_context="Origins of Wonder Mediterranean",
    )
    entries = []
    for i, (cam, end_t, clause, directive) in enumerate(beats, start=1):
        entries.append({
            "scene_id": directive.scene_id, "seg_index": i, "chapter_index": 0, "turn_id": 1,
            "narration_clause": clause, "subject": directive.subject_label, "visual_family": cam,
            "semantic_directive": directive.to_dict(), "validation_status": "pending",
            "scene_start_sec": 0.0, "scene_end_sec": end_t,
        })
    # Every surviving beat's directive must be independently valid for its stored clause.
    assert _subject_hard_gate(entries) == []
    # The Cairo clause is the only fragment naming a monument -> exactly one Giza binding.
    giza = [e for e in entries if e["semantic_directive"]["scene_id"] == 0]
    assert giza and all("cairo" in e["narration_clause"].lower() or "giza" in e["narration_clause"].lower()
                        or e["semantic_directive"]["source"] == "anaphora_continuation" for e in giza)


def test_planner_arbitrary_topic_uses_diagram_family_not_monuments() -> None:
    from voxforge.video.renderer import _plan_turn_editorial_beats
    from voxforge.video.semantic_router import MODALITY_DIAGRAM, SemanticShotDirector

    turn = _turn(
        "Magma buoyancy drives molten rock upward through the crust, while dissolved gases "
        "fragment the lava into ash. The eruption column climbs until it reaches the stable "
        "stratosphere, and then winds distribute tephra across entire continents downwind."
    )
    env = np.full(2500, 0.12, dtype=np.float32)
    director = SemanticShotDirector("How Volcanoes Erupt")
    beats = _plan_turn_editorial_beats(
        turn, 10.0, 34.0, env, director, "The Fiery Engine", "Pacific Ring of Fire",
        scene_context="How Volcanoes Erupt Pacific Ring of Fire",
    )
    assert beats
    for cam, end_t, clause, directive in beats:
        assert directive.scene_id == -1
        assert directive.modality in (MODALITY_DIAGRAM, "orbital_globe", "kinetic_quote")
        if directive.modality == MODALITY_DIAGRAM:
            assert cam == "semantic_diagram"
            assert directive.diagram_template


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


def test_short_quote_card_detected_via_central_band() -> None:
    """A short 2-3 line inscription never reaches the 4.5% whole-frame white mass, but its
    concentrated central-band typography must still read as a quote (regression for the
    Pharos quote scene that previously classified as darkhud)."""
    from voxforge.audit.av_sync_qa import classify_frame_group

    rng = np.random.default_rng(3)
    h, w = 1080, 1920
    # Dark quote backdrop with three short white text rows concentrated in the composition band.
    quote = np.full((h, w, 3), 22, dtype=np.uint8)
    for row_y in (430, 500, 570):
        quote[row_y : row_y + 26, 360:1180] = (248, 250, 252)
    quote += rng.integers(0, 5, quote.shape, dtype=np.uint8)
    assert classify_frame_group(quote) == "quote"

    # Sparse neutral-white telemetry/HUD copy stays below the band threshold -> darkhud.
    hud = np.full((h, w, 3), 16, dtype=np.uint8)
    for row_y in (460, 540):
        hud[row_y : row_y + 10, 360:600] = (235, 235, 235)  # thin, short labels
    assert classify_frame_group(hud) == "darkhud"

    # Amber/cyan HUD glyphs are not neutral white, so they never trigger the quote gate.
    glyph = np.full((h, w, 3), 14, dtype=np.uint8)
    glyph[300:700:40, 300:1500] = (240, 180, 60)
    assert classify_frame_group(glyph) == "darkhud"


def test_generic_explanatory_simulation_renders() -> None:
    from voxforge.video.procedural_world import SCENE_CLASSICAL_GENERIC, PhysicsSimulationEngine

    eng = PhysicsSimulationEngine()
    frame = eng.render_simulation_frame(
        SCENE_CLASSICAL_GENERIC, 0.8, 1.0, 640, 360,
        hint="Trade flows grew 25 percent as the network connected coastal cities",
    )
    assert frame.shape == (360, 640, 3)
    assert float(frame.mean()) > 5.0  # not a black frame; glyphs/panels drawn


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
                "scene_id": -1, "seg_index": 0, "chapter_index": 0, "turn_id": 1,
                "narration_clause": "synthetic uncertain scene with enough clause length",
                "narration_start_sec": 0.0, "narration_end_sec": 10.0,
                "subject": "synthetic", "action_or_event": "geo",
                "required_visual": "globe", "visual_family": "wide_studio",
                "visual_asset_source": "test", "scene_start_sec": 0.0, "scene_end_sec": 10.0,
                "on_screen_text": "", "relevance_criteria": "", "validation_status": "pending",
                "semantic_directive": {
                    "subject_key": "synthetic_place", "subject_label": "synthetic",
                    "modality": "orbital_globe", "scene_id": -1, "diagram_template": "",
                    "treatment": "orbital_context", "on_screen_label": "synthetic",
                    "confidence": 0.8, "source": "place_globe",
                },
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


def _valid_entry(seg: int, clause: str, *, scene_id: int = -1, modality: str = "semantic_diagram",
                 subject: str = "Volcanic Eruptions", key: str = "volcanoes_geoscience",
                 treatment: str = "diagram_terrain_tectonic", template: str = "terrain_tectonic",
                 source: str = "semantic_category") -> dict:
    return {
        "scene_id": scene_id, "seg_index": seg, "chapter_index": 0, "turn_id": seg,
        "narration_clause": clause,
        "narration_start_sec": float(seg), "narration_end_sec": float(seg + 4),
        "subject": subject, "action_or_event": "scene",
        "required_visual": "diagram", "visual_family": "semantic_diagram",
        "visual_asset_source": "pure-code:SemanticDiagramEngine",
        "scene_start_sec": float(seg), "scene_end_sec": float(seg + 4),
        "on_screen_text": clause[:96], "relevance_criteria": "test", "validation_status": "pending",
        "semantic_directive": {
            "subject_key": key, "subject_label": subject, "modality": modality,
            "scene_id": scene_id, "diagram_template": template, "treatment": treatment,
            "on_screen_label": subject, "confidence": 0.75, "source": source,
        },
    }


def test_subject_hard_gate_accepts_clean_subject_bound_plan() -> None:
    from voxforge.audit.av_sync_qa import _subject_hard_gate

    entries = [
        _valid_entry(0, "Magma pressure rises beneath the crust as gases dissolve", treatment="diagram_terrain_tectonic"),
        _valid_entry(1, "The network of magma chambers feeds the central vent system",
                     key="volcanoes_geoscience", treatment="diagram_node_network", template="node_network"),
        _valid_entry(2, "Particles of superheated tephra explode into the upper atmosphere",
                     key="volcanoes_geoscience", treatment="diagram_particle_cluster", template="particle_cluster"),
    ]
    assert _subject_hard_gate(entries) == []
    assert all(e["validation_status"] == "pending" for e in entries)


def test_subject_hard_gate_blocks_placeholder_wrong_subject_and_repeats() -> None:
    from voxforge.audit.av_sync_qa import _PLACEHOLDER_CLAUSE, _subject_hard_gate

    # 1) placeholder-bound shot must fail
    placeholder = _valid_entry(0, _PLACEHOLDER_CLAUSE)
    # 2) monument scene_id on a clause that does not keyword-match that monument
    wrong_monument = _valid_entry(
        1, "Sourdough fermentation depends on wild yeast and lactic acid bacteria",
        scene_id=0, modality="monument_3d", subject="Great Pyramid of Giza",
        key="great_pyramid_of_giza", treatment="exterior_wide", template="", source="wonder_keyword",
    )
    # 3) consecutive identical (subject, treatment) pair
    repeat_a = _valid_entry(2, "The eruption column climbed through the troposphere rapidly",
                            treatment="diagram_stat_dashboard", template="stat_dashboard")
    repeat_b = _valid_entry(3, "Ashfall smothered farmland across three surrounding regions",
                            treatment="diagram_stat_dashboard", template="stat_dashboard")
    entries = [placeholder, wrong_monument, repeat_a, repeat_b]
    failures = dict((s, why) for s, why in _subject_hard_gate(entries))
    assert set(failures) == {0, 1, 3}
    assert "placeholder_or_empty_clause" in failures[0]
    assert "clause_subject_does_not_match_rendered_monument" in failures[1]
    assert "consecutive_identical_subject_treatment" in failures[3]
    assert entries[0]["validation_status"].startswith("failed")
    assert entries[1]["validation_status"].startswith("failed")
    assert entries[2]["validation_status"] == "pending"  # first of the pair is legitimate
    assert entries[3]["validation_status"].startswith("failed")


def test_subject_hard_gate_accepts_clause_keyword_and_anaphora_chain_shots() -> None:
    from voxforge.audit.av_sync_qa import _subject_hard_gate

    # Direct clause-keyword globe: the clause itself names Giza.
    globe = _valid_entry(
        0, "Across the Giza plateau the survey map traces every causeway and harbor basin",
        scene_id=0, modality="orbital_globe", subject="Great Pyramid of Giza",
        key="great_pyramid_of_giza", treatment="orbital_context", template="",
        source="place_globe",
    )
    globe["visual_family"] = "wide_studio"
    assert _subject_hard_gate([globe]) == []

    # Anaphora chain: keyword head followed by lowercase-fragment continuations inherits Giza.
    head = _valid_entry(
        1, "The Great Pyramid at Giza was raised for Pharaoh Khufu from limestone blocks",
        scene_id=0, modality="monument_3d", subject="Great Pyramid of Giza",
        key="great_pyramid_of_giza", treatment="exterior_wide", template="",
        source="wonder_keyword",
    )
    head["visual_family"] = "broll_cinema"
    cont = _valid_entry(
        2, "remaining the tallest human-made structure on Earth for more than 3,800 years",
        scene_id=0, modality="lidar_cad", subject="Great Pyramid of Giza",
        key="great_pyramid_of_giza", treatment="lidar_scan", template="",
        source="anaphora_continuation",
    )
    cont["visual_family"] = "studio_insert"
    assert _subject_hard_gate([head, cont]) == []

    # Broken chain: anaphora source without a monument predecessor must fail.
    orphan = dict(cont)
    orphan["seg_index"] = 3
    failures = _subject_hard_gate([orphan])
    assert failures and "anaphora_chain_broken_or_not_referential" in failures[0][1]

    # Generic-topic globe keeps scene_id -1 and also passes.
    generic_globe = _valid_entry(
        4, "Trade routes stretched across three continents and many remote regions",
        scene_id=-1, modality="orbital_globe", subject="Trade Routes",
        key="trade_routes_place", treatment="orbital_context", template="",
        source="place_globe",
    )
    generic_globe["visual_family"] = "wide_studio"
    assert _subject_hard_gate([generic_globe]) == []


def test_generic_clause_inside_wonder_chapter_never_binds_monument() -> None:
    """Regression for the user-reported mismatch: abstract clauses (cosmology, planets,
    rhetoric) must render neutral diagrams even when the chapter context names a wonder."""
    from voxforge.video.semantic_router import (
        MODALITY_DIAGRAM,
        SemanticShotDirector,
        is_anaphoric_or_continuation,
    )

    director = SemanticShotDirector("Seven Wonders of the Ancient World")
    generic_clauses = [
        "Hellenistic scholars chose the number seven",
        "in classical Greek cosmology it represented completeness and perfection",
        "five known planets plus the Sun and Moon",
        "Proof of what happens when geometry and imagination refuse the impossible",
    ]
    for c in generic_clauses:
        d = director.directive_for_clause(c, "Origins of Wonder · Pharos Lighthouse chapter", "scene")
        assert d.scene_id == -1 and d.modality == MODALITY_DIAGRAM, (c, d)

    # The same chapter's real monument references still bind the monument directly.
    d0 = director.directive_for_clause(
        "Out of the original Seven Wonders, the Giza plateau monument alone survives", "ch", "scene"
    )
    assert d0.scene_id == 0 and d0.source == "wonder_keyword"

    # Referential fragments inherit the previous monument rather than becoming diagrams.
    d1 = director.directive_for_clause("the casing stones caught the sun across the valley", "ch", "scene")
    assert d1.scene_id == 0 and d1.source == "anaphora_continuation"
    assert is_anaphoric_or_continuation("the casing stones caught the sun")
    # Mid-clause complementizer "that" is not anaphora.
    assert not is_anaphoric_or_continuation("A chronicler wrote that the stone gleamed at dawn")


def test_subject_hard_gate_blocks_diagram_bound_to_monument_scene() -> None:
    from voxforge.audit.av_sync_qa import _subject_hard_gate

    bad = _valid_entry(
        0, "Sourdough fermentation depends on wild yeast and lactic acid bacteria colonies",
        scene_id=0, modality="semantic_diagram", subject="Sourdough",
        key="sourdough_food_process", treatment="diagram_process_flow",
        template="process_flow", source="semantic_category",
    )
    failures = _subject_hard_gate([bad])
    assert failures and "non_monument_shot_bound_to_monument_scene_id" in failures[0][1]


def test_subject_hard_gate_accepts_keyword_matched_monument_shots() -> None:
    from voxforge.audit.av_sync_qa import _subject_hard_gate

    entry = _valid_entry(
        0, "The Great Pyramid at Giza was raised for Pharaoh Khufu from limestone blocks",
        scene_id=0, modality="monument_3d", subject="Great Pyramid of Giza",
        key="great_pyramid_of_giza", treatment="exterior_wide", template="", source="wonder_keyword",
    )
    entry["semantic_directive"]["context"] = "Great Pyramid of Giza Necropolis"
    entry["visual_family"] = "broll_cinema"
    assert _subject_hard_gate([entry]) == []


def test_duplicate_frame_pairs_catches_repeated_shots_across_film() -> None:
    from voxforge.audit.av_sync_qa import _frame_fingerprint, duplicate_frame_pairs
    import cv2

    rng = np.random.default_rng(4)
    distinct = [rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8) for _ in range(4)]
    fingers = {i + 1: _frame_fingerprint(f) for i, f in enumerate(distinct)}
    assert duplicate_frame_pairs(fingers) == []

    # A fifth shot that reuses the exact picture of shot 2 (pre-fix subject-seed loop).
    fingers[5] = _frame_fingerprint(distinct[1])
    pairs = duplicate_frame_pairs(fingers)
    assert pairs and pairs[0][:2] == (2, 5)

    # Mildly edited but still near-identical re-encodes trip the gate too.
    soft = cv2.addWeighted(distinct[0], 0.98, np.full_like(distinct[0], 4), 0.02, 0)
    fingers[6] = _frame_fingerprint(soft)
    assert any(a == 1 and b == 6 for a, b, _c in duplicate_frame_pairs(fingers))
