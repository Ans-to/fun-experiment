"""Tests for the six semantic diagram engines: motion, darkness and subject determinism."""

from __future__ import annotations

import numpy as np

from voxforge.video.semantic_diagrams import SemanticDiagramEngine
from voxforge.video.semantic_router import ALL_TEMPLATES, subject_seed

CLAUSE = (
    "The eruption column rose 34 kilometres above the vent while ashfall covered 2.4 million "
    "square kilometres across three regions"
)


def test_every_template_renders_animated_dark_frames() -> None:
    engine = SemanticDiagramEngine()
    for template in ALL_TEMPLATES:
        f0 = engine.render_frame(template, "Volcanic Eruptions", CLAUSE, 0.05, 0.1, seed=42)
        f1 = engine.render_frame(template, "Volcanic Eruptions", CLAUSE, 0.55, 0.7, seed=42)
        f2 = engine.render_frame(template, "Volcanic Eruptions", CLAUSE, 0.95, 1.4, seed=42)
        assert f0.shape == (1080, 1920, 3)
        assert f0.dtype == np.uint8
        # Continuous motion between every sampled frame.
        assert float(np.mean(np.abs(f1.astype(np.int16) - f0.astype(np.int16)))) > 0.4
        assert float(np.mean(np.abs(f2.astype(np.int16) - f1.astype(np.int16)))) > 0.4
        # Obsidian palette: dark enough for the darkhud Stage E evidence group.
        assert f1.mean() < 60.0


def test_subject_seed_changes_geometry() -> None:
    engine = SemanticDiagramEngine()
    a = engine.render_frame(ALL_TEMPLATES[1], "Volcanoes", CLAUSE, 0.5, 0.5, seed=subject_seed("volcanoes"))
    b = engine.render_frame(ALL_TEMPLATES[1], "Quantum Computing", CLAUSE, 0.5, 0.5, seed=subject_seed("quantum_computing"))
    assert float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) > 2.0


def test_every_template_clears_stage_d_black_frame_luma_floor() -> None:
    # Regression for Stage D D3 failure: the sparsest templates (particle/timeline) encoded
    # below the Y>8.0 black-frame gate. Raw frames must clear ~11 so H.264 losses stay safe.
    engine = SemanticDiagramEngine()
    clause = "five known planets plus the Sun and Moon orbit in a stable cosmic system"
    for template in ALL_TEMPLATES:
        for u in (0.15, 0.5, 0.9):
            frame = engine.render_frame(template, "Planet", clause, u, u * 2.0, seed=5)
            assert float(frame.mean()) >= 11.0, (template, u, float(frame.mean()))


def test_every_template_classifies_as_darkhud_for_stage_e() -> None:
    # Regression guard: semantic diagrams must read as the darkhud pixel-evidence group
    # (terrain fills once used neutral slate that the classifier read as monument stone).
    from voxforge.audit.av_sync_qa import classify_frame_group

    engine = SemanticDiagramEngine()
    for template in ALL_TEMPLATES:
        rgb = engine.render_frame(template, "Volcanic Eruptions", CLAUSE, 0.6, 0.9, seed=11)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        assert classify_frame_group(bgr) == "darkhud", template


def test_same_seed_is_deterministic() -> None:
    engine = SemanticDiagramEngine()
    a = engine.render_frame(ALL_TEMPLATES[0], "Volcanoes", CLAUSE, 0.4, 0.4, seed=7)
    b = engine.render_frame(ALL_TEMPLATES[0], "Volcanoes", CLAUSE, 0.4, 0.4, seed=7)
    assert np.array_equal(a, b)


def test_stat_panel_never_fabricates_percentage_without_numbers() -> None:
    from voxforge.video.semantic_diagrams import stat_panel_content

    numeric = stat_panel_content("the pyramid stood 146.6 meters tall", 0.9)
    assert numeric["kind"] == "numeric" and numeric["unit"] == "METERS"

    honest = stat_panel_content("Out of the original Seven Wonders cataloged by travelers", 0.9)
    assert honest["kind"] == "keywords"
    assert honest["terms"] and "PER CENT" not in honest["unit"]


def test_clause_salted_seeds_change_geometry_for_same_subject() -> None:
    # Regression: subject-only seeds made four abstract shots pixel-identical across the film.
    engine = SemanticDiagramEngine()
    base = subject_seed("episode_abstract")
    a = engine.render_frame(
        ALL_TEMPLATES[1], "Deep Horizon", "first clause about wonder and legacy", 0.5, 1.0,
        seed=base ^ subject_seed("clause::first clause about wonder and legacy"),
    )
    b = engine.render_frame(
        ALL_TEMPLATES[1], "Deep Horizon", "second clause about stone and ambition", 0.5, 1.0,
        seed=base ^ subject_seed("clause::second clause about stone and ambition"),
    )
    assert not np.array_equal(a, b)
    assert float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) > 1.0


def test_long_title_always_fits_canvas_width() -> None:
    from PIL import ImageDraw, Image

    engine = SemanticDiagramEngine()
    long_title = "DEEP HORIZON: 7 WONDERS OF THE WORLD: ARCHITECTURE OF THE IMPOSSIBLE"
    img = Image.new("RGB", (1920, 1080))
    draw = ImageDraw.Draw(img)
    lines = engine._fit_title(draw, long_title, 1920 - 172)
    assert lines and all(draw.textlength(text, font=font) <= 1920 - 172 for text, font in lines)


def test_stat_panel_preserves_clause_decimal_precision() -> None:
    from voxforge.video.semantic_diagrams import stat_panel_content

    full = stat_panel_content("the Great Pyramid stood 146.6 meters tall", 1.0)
    assert full["kind"] == "numeric" and full["label"] == "146.6" and full["unit"] == "METERS"
    integer = stat_panel_content("roughly 2 million stones across the site", 1.0)
    assert integer["label"] == "2" and integer["unit"] == "MILLION"
    count = stat_panel_content("the wall spans 21196 kilometres", 1.0)
    assert count["label"] == "21,196" and count["unit"] == "KILOMETRES"
