"""Regression tests for live subject fidelity and anti-repetition of visuals.

Two user-reported defects are covered here:

1. LIVE SUBJECT FIDELITY — narration about the wonders that were destroyed must not be
   illustrated with the one wonder that survived. Aggregate/loss clauses get a montage of
   the lost monuments instead.
2. NO VISUAL REPETITION — a long stretch on a single subject must keep changing camera
   family and camera angle rather than holding one holographic screen.
"""

from __future__ import annotations

import numpy as np

from voxforge.models import DialogueTurn
from voxforge.video.procedural_world import SCENE_CAPTIONS
from voxforge.video.renderer import (
    _plan_turn_editorial_beats,
    _plan_turn_editorial_beats_detail,
    is_lost_wonders_reference,
)

WONDERS_CONTEXT = "The Ancient Seven & The Great Pyramid of Giza Giza Plateau, Egypt"

# The exact opening the user flagged: "he said about earthquakes and what not but you keep
# showing pyramid".
OPENING_TEXT = (
    "Out of the original Seven Wonders cataloged by Greek travelers two thousand years ago, "
    "earthquakes, fire, and human greed destroyed six — leaving only a 4,500-year-old "
    "limestone mountain on the edge of Cairo."
)

FLAT_ENV = np.full(3000, 0.12, dtype=np.float32)


def _turn(text: str, camera: str = "broll_cinema", dur: float = 27.0) -> DialogueTurn:
    return DialogueTurn(
        turn_id=1,
        chapter_index=1,
        speaker="HOST_A",
        text=text,
        spoken_text=text,
        camera=camera,
        start_sec=0.0,
        end_sec=dur,
        duration_sec=dur,
    )


# ---------------------------------------------------------------------------
# 1. lost-wonders detection
# ---------------------------------------------------------------------------


def test_detects_destruction_of_the_group() -> None:
    assert is_lost_wonders_reference("earthquakes", WONDERS_CONTEXT)
    assert is_lost_wonders_reference(
        "fire, and human greed destroyed six — leaving only", WONDERS_CONTEXT
    )
    assert is_lost_wonders_reference(
        "the Colossus of Rhodes stood for fifty-four years before an earthquake toppled it",
        WONDERS_CONTEXT,
    )


def test_detects_bare_enumeration_of_the_canon() -> None:
    assert is_lost_wonders_reference(
        "Out of the original Seven Wonders cataloged by Greek travelers", WONDERS_CONTEXT
    )


def test_survivor_only_clause_is_not_a_loss_reference() -> None:
    """A clause that is purely about the surviving monument stays a subject shot."""
    assert not is_lost_wonders_reference(
        "a 4,500-year-old limestone mountain on the edge of Cairo.", WONDERS_CONTEXT
    )
    assert not is_lost_wonders_reference(
        "the Great Pyramid stood 146.6 meters tall", WONDERS_CONTEXT
    )


def test_no_false_positives_outside_wonder_narration() -> None:
    for clause in (
        "Magma rises through the crust and erupts",
        "Bakers fold dough to build gluten strands",
        "Beacon fire carried messages across the mountains",
        "How volcanoes erupt and reshape the planet",
    ):
        assert not is_lost_wonders_reference(clause, "Volcanoes Iceland"), clause


# ---------------------------------------------------------------------------
# 2. the flagged opening no longer shows the pyramid
# ---------------------------------------------------------------------------


def test_opening_runs_montage_while_destruction_is_narrated() -> None:
    beats, flags = _plan_turn_editorial_beats_detail(
        _turn(OPENING_TEXT, camera="wide_studio"),
        0.0,
        27.0,
        FLAT_ENV,
        is_episode_opener=True,
        is_episode_closer=False,
        scene_context=WONDERS_CONTEXT,
    )
    assert len(beats) == len(flags)

    survivor_named = False
    montage_seen = False
    # Beat 0 is the deliberate establishing wide (globe) that opens every episode;
    # every beat after it must be clause-faithful.
    for (cam, _end, clause), flag in list(zip(beats, flags))[1:]:
        is_montage = cam == "broll_cinema" and flag
        if "4,500-year-old" in clause or "Cairo" in clause:
            survivor_named = True
        if survivor_named:
            # The moment the survivor is described, the picture must become a subject shot.
            assert not is_montage, f"montage still running on survivor clause: {clause!r}"
        else:
            assert is_montage, f"no montage while narrating the lost wonders: {clause!r}"
            montage_seen = True
    assert montage_seen


def test_montage_covers_the_first_fifteen_seconds() -> None:
    beats, flags = _plan_turn_editorial_beats_detail(
        _turn(OPENING_TEXT, camera="wide_studio"),
        0.0,
        27.0,
        FLAT_ENV,
        is_episode_opener=True,
        is_episode_closer=False,
        scene_context=WONDERS_CONTEXT,
    )
    prev = 0.0
    montage_seconds = 0.0
    for (cam, end, _clause), flag in zip(beats, flags):
        if cam == "broll_cinema" and flag and prev < 15.0:
            montage_seconds += min(end, 15.0) - prev
        prev = end
    # The destruction narration dominates the first 15s; the survivor must not do the same.
    assert montage_seconds >= 9.0, f"only {montage_seconds:.1f}s of montage in first 15s"


# ---------------------------------------------------------------------------
# 3. anti-repetition on a long single-subject stretch
# ---------------------------------------------------------------------------

GIZA_MONOLOGUE = (
    "Built for Pharaoh Khufu under royal vizier Hemiunu, the Great Pyramid stood 146.6 "
    "meters tall, remaining the tallest human-made structure on Earth for more than 3,800 "
    "years. Its base covers thirteen acres and its courses hold around two million "
    "limestone blocks quarried nearby, each one weighing several tonnes."
)


def test_long_single_subject_never_repeats_a_family_back_to_back() -> None:
    beats = _plan_turn_editorial_beats(
        _turn(GIZA_MONOLOGUE, dur=40.0),
        0.0,
        40.0,
        FLAT_ENV,
        is_episode_opener=False,
        is_episode_closer=False,
        scene_context=WONDERS_CONTEXT,
    )
    families = [cam for cam, _end, _c in beats]
    assert len(families) >= 4, f"too few beats for a 40s stretch: {families}"
    repeats = [
        i for i in range(1, len(families)) if families[i] == families[i - 1]
    ]
    assert not repeats, f"consecutive duplicate visual families: {families}"
    assert len(set(families)) >= 3, f"not enough variety: {families}"


def test_montage_uses_a_distinct_scene_per_lost_wonder() -> None:
    """Regression: all six cards once rendered the SAME Pharos mesh with different captions,
    which is the 'same shape again and again' defect the montage exists to remove."""
    from voxforge.video.animator import StudioCharacterAnimator

    animator = StudioCharacterAnimator()
    entries = animator._LOST_WONDERS
    assert len(entries) == 6
    names = [e[0] for e in entries]
    assert len(set(names)) == 6, f"duplicate wonder names: {names}"
    scene_ids = [e[5] for e in entries]
    assert len(set(scene_ids)) == 6, f"wonders share geometry: {dict(zip(names, scene_ids))}"
    for name, sid in zip(names, scene_ids):
        assert sid in SCENE_CAPTIONS, f"{name} has no registered scene {sid}"


def test_beat_contract_unchanged() -> None:

    """The planner still returns the (camera, end_sec, clause) triples the encoder relies on."""
    beats = _plan_turn_editorial_beats(
        _turn("Short clause about the pyramid.", dur=12.0),
        0.0,
        12.0,
        FLAT_ENV,
        is_episode_opener=False,
        is_episode_closer=False,
        scene_context=WONDERS_CONTEXT,
    )
    assert beats
    for beat in beats:
        assert len(beat) == 3
        cam, end, clause = beat
        assert isinstance(cam, str) and cam
        assert isinstance(end, float)
        assert isinstance(clause, str)
    assert beats[-1][1] == 12.0
