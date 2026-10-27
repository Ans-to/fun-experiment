"""Tests for the semantic shot director: subject relevance and no-repeat guarantees."""

from __future__ import annotations

from voxforge.video.procedural_world import (
    SCENE_GIZA,
    SCENE_TAJ_MAHAL,
    match_wonder_scene_id,
)
from voxforge.video.semantic_router import (
    ALL_TEMPLATES,
    MODALITY_DIAGRAM,
    MODALITY_GLOBE,
    MODALITY_MONUMENT,
    SemanticShotDirector,
    classify_semantic_category,
    slugify,
    subject_seed,
)


def test_pure_wonder_matcher_has_no_fallback() -> None:
    assert match_wonder_scene_id("Khufu's Great Pyramid at Giza") == SCENE_GIZA
    assert match_wonder_scene_id("the white marble mausoleum at Agra, the Taj Mahal") == SCENE_TAJ_MAHAL
    # Arbitrary topics must NOT resolve to a monument model anymore.
    assert match_wonder_scene_id("how volcanoes erupt and reshape the planet") is None
    assert match_wonder_scene_id("a sourdough loaf needs a strong starter") is None


def test_arbitrary_topics_never_render_monuments() -> None:
    cases = [
        ("how volcanoes erupt and reshape the planet", "geoscience"),
        ("the rise of artificial intelligence and neural networks", "technology"),
        ("how the human immune system fights a virus", "biology_health"),
        ("how to bake sourdough bread at home", "food_process"),
        ("silk road trade networks and merchant caravans", "society_economy"),
    ]
    for topic, category in cases:
        cat, kw = classify_semantic_category(topic)
        assert cat == category, (topic, cat)
        director = SemanticShotDirector(topic)
        d = director.directive_for_clause(topic, topic, "scene")
        assert d.modality == MODALITY_DIAGRAM, (topic, d)
        assert d.scene_id == -1
        assert d.diagram_template in ALL_TEMPLATES
        assert d.subject_key  # always bound to a real subject


def test_same_subject_paragraph_does_not_repeat_treatments() -> None:
    director = SemanticShotDirector("Seven Wonders of the Ancient World")
    clauses = [
        "The Great Pyramid was built for Pharaoh Khufu at Giza.",
        "Its limestone blocks were quarried at Mokattam across the Nile.",
        "Hemiunu, the royal vizier, oversaw the construction crews.",
        "The monument originally stood 146.6 meters above the plateau.",
        "It remained the tallest human-made structure for millennia.",
        "Engineers aligned the casing to true north within a fraction of a degree.",
        "A grand gallery climbs through the heart of the stone mass.",
    ]
    directives = [director.directive_for_clause(c, "Great Pyramid of Giza", "scene") for c in clauses]
    # Every clause binds to the Giza subject with a real clause attached.
    assert all(d.scene_id == SCENE_GIZA for d in directives)
    # No identical treatment repeats inside the sliding window.
    pairs = [(d.subject_key, d.treatment) for d in directives]
    window = 5
    for i in range(len(pairs)):
        assert pairs[i] not in pairs[max(0, i - window):i], pairs
    # And at least three distinct treatments appear across the paragraph.
    assert len({d.treatment for d in directives}) >= 3


def test_place_and_quote_intents_route_to_distinct_modalities() -> None:
    director = SemanticShotDirector("Petra and the Nabataean trade cities")
    geo = director.directive_for_clause(
        "Petra sits beside the trade routes between the Red Sea and Damascus", "Petra Jordan", "geo"
    )
    assert geo.modality == MODALITY_GLOBE
    quote = director.directive_for_clause(
        "A chronicler wrote that the treasury gleamed like rose-red stone", "Petra", "quote"
    )
    assert quote.modality == "kinetic_quote"


def test_attribution_beats_are_bound_not_placeholder() -> None:
    director = SemanticShotDirector("How Volcanoes Erupt")
    d = director.attribution_directive("The Earth's Fiary Engine", "Pacific Ring of Fire", "people")
    assert d.subject_key and d.on_screen_label
    assert d.modality in {MODALITY_DIAGRAM, MODALITY_GLOBE, MODALITY_MONUMENT}
    # Deterministic subject seed: same subject -> same seed, different subject -> different seed.
    assert subject_seed(slugify("Volcanoes")) == subject_seed(slugify("Volcanoes"))
    assert subject_seed(slugify("Volcanoes")) != subject_seed(slugify("Quantum Computing"))


def test_template_ladder_rotates_without_short_term_repetition() -> None:
    director = SemanticShotDirector("Quantum Computing Fundamentals")
    directives = [
        director.directive_for_clause(f"Qubit concept number {i} exploits superposition", "quantum", "scene")
        for i in range(8)
    ]
    templates = [d.diagram_template for d in directives]
    assert all(t in ALL_TEMPLATES for t in templates)
    # Adjacent shots never repeat, and the full category ladder is used before rotation.
    assert all(templates[i] != templates[i - 1] for i in range(1, len(templates)))
    assert len(set(templates[:4])) == 4
