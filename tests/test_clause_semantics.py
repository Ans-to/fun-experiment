"""Tests for clause-level semantic analysis: meaning archetypes, honest labels, quantities."""

from __future__ import annotations

import pytest

from voxforge.video.clause_semantics import (
    ARCH_BUILD,
    ARCH_CATALOG,
    ARCH_COSMOS,
    ARCH_DESTRUCTION,
    ARCH_ENDURE,
    ARCH_HISTORY,
    ARCH_QUANTITY,
    analyze_clause,
    labels_are_clause_derived,
)


@pytest.mark.parametrize(
    "clause,archetype,direction",
    [
        ("fire, and human greed destroyed six, leaving only one", ARCH_DESTRUCTION, "falling"),
        ("Constructed over roughly twenty to twenty-seven years during the Old Kingdom",
         ARCH_BUILD, "rising"),
        ("the World endure as humanity's boldest signature in stone", ARCH_ENDURE, "stable"),
        ("perfection: five known planets plus the Sun and Moon", ARCH_COSMOS, "neutral"),
        ("Out of the original Seven Wonders cataloged by travelers", ARCH_CATALOG, "neutral"),
        ("the Great Pyramid stood 146.6 meters tall over 3,800 years", ARCH_QUANTITY, "stable"),
        ("Egypt's Old Kingdom around 2560 BCE Khufu's horizon monument", ARCH_HISTORY, "neutral"),
    ],
)
def test_archetype_and_direction_match_clause_meaning(clause: str, archetype: str, direction: str) -> None:
    sem = analyze_clause(clause)
    assert sem.archetype == archetype, sem.rationale
    assert sem.direction == direction


def test_every_label_is_clause_derived_across_real_film_clauses() -> None:
    real_clauses = [
        "Out of the original Seven Wonders cataloged",
        "by Greek travelers two thousand years ago earthquakes",
        "fire, and human greed destroyed six — leaving only",
        "The question is simple: what did these builders understand about stone, weight",
        "Origins of Wonder: From Greek Theamata to Global Monuments Mediterranean Basin",
        "in classical Greek cosmology it represented completeness",
        "perfection: five known planets plus the Sun",
        "Constructed over roughly twenty to twenty-seven years during",
        "earthquakes, and desert winds, the 7 Wonders of",
        "the World endure as humanity's boldest signature in stone.",
        "Proof of what happens when geometry and imagination refuse to accept the impossible.",
    ]
    for clause in real_clauses:
        sem = analyze_clause(clause)
        assert len(sem.labels) >= 2, clause
        assert labels_are_clause_derived(clause, sem.labels), (clause, sem.labels)


def test_quantities_keep_precision_and_group_thousands() -> None:
    sem = analyze_clause("the Great Pyramid stood 146.6 meters tall")
    assert sem.quantities and sem.quantities[0].display_number == "146.6"
    assert sem.quantities[0].unit == "meters"

    grouped = analyze_clause("the wall spans 21196 kilometres across the ridges")
    assert grouped.quantities[0].display_number == "21,196"

    counted = analyze_clause("earthquakes, and desert winds, the 7 Wonders of the World")
    assert counted.quantities and counted.quantities[0].number_text == "7"


def test_time_marks_are_verbatim() -> None:
    sem = analyze_clause("the tallest structure for more than 3,800 years since 2560 BCE")
    joined = " ".join(sem.time_marks)
    assert "2560 BCE" in joined
    assert "3,800 years" in joined


def test_terms_preserve_real_entities_and_order_variants() -> None:
    sem = analyze_clause("chief architect Hemiunu built the Great Pyramid near Giza")
    assert "Hemiunu" in sem.terms and "Great Pyramid" in sem.terms or "Pyramid" in sem.terms
    # Clause-order variant keeps narrative sequence for process stages.
    assert sem.ordered_terms[0] in sem.terms


def test_stage_terms_drop_numerals_hedges_and_keep_named_entities() -> None:
    from voxforge.video.clause_semantics import stage_terms

    stages = stage_terms(
        "Constructed over roughly twenty to twenty-seven years during Egypt's Old Kingdom"
    )
    joined = " ".join(stages).lower()
    assert "roughly" not in joined and "twenty" not in joined and "years" not in joined
    assert "constructed" in joined
    # Adjacent capitalized words merge into a single named-entity stage.
    assert any("old kingdom" in s.lower() for s in stages)


def test_entity_terms_promote_star_and_drop_numerals() -> None:
    from voxforge.video.clause_semantics import entity_terms

    bodies = entity_terms("perfection: five known planets plus the Sun and Moon")
    assert bodies[0].lower() == "sun"
    assert "five" not in bodies


def test_hyphenated_duration_is_captured_verbatim_not_truncated() -> None:
    sem = analyze_clause("Constructed over roughly twenty to twenty-seven years of labor")
    assert "twenty-seven years" in sem.time_marks
    assert "seven years" not in sem.time_marks


def test_quantity_keeps_counted_noun_inside_match_chunk() -> None:
    sem = analyze_clause("earthquakes and desert winds, the 7 Wonders of the World endure")
    assert sem.quantities and sem.quantities[0].noun.lower() == "wonders"
    assert sem.quantities[0].display_number == "7"


def test_capitalized_phrases_merge_into_single_terms() -> None:
    sem = analyze_clause("chief architect Hemiunu built the Great Pyramid near Giza")
    assert "Great Pyramid" in sem.terms
    # The merged phrase still tokenizes back to clause words (Stage E honesty).
    assert labels_are_clause_derived("chief architect Hemiunu built the Great Pyramid near Giza",
                                     sem.labels)
