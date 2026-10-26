"""Editorial QA, 1-hour script scaling, and TTS pronunciation normalization tests."""

import tempfile
from pathlib import Path

import numpy as np

from voxforge.audio.tts_engine import _slerp_voice_vectors, _split_spoken_sentences
from voxforge.audit.editorial import audit_and_improve_script
from voxforge.bootstrap import StudioEnvironment
from voxforge.intent import parse_command_to_spec
from voxforge.models import DialogueTurn
from voxforge.research.engine import build_research_dossier
from voxforge.script.showrunner import (
    generate_episode_script,
    normalize_for_tts,
    rewrite_for_spoken_ear,
)


def test_tts_pronunciation_normalizer() -> None:
    raw = "Built around c. 2560 BCE (Fourth Dynasty) under Nebuchadnezzar II — rising ~146.6 m across 21 km at 51°!"
    norm = normalize_for_tts(raw)
    assert "B C E" in norm
    assert "around 25 60 B C E" in norm
    assert "Fourth Dynasty" not in norm
    assert "Nebuchadnezzar the Second" in norm
    assert "meters" in norm
    assert "kilometers" in norm
    assert "degrees" in norm
    assert "—" not in norm

    # Verify abbreviation & title-case hygiene for natural podcast voiceover
    norm2 = normalize_for_tts("Welcome to DEEP HORIZON with Dr. Maya Lin in the 7th Century BCE!")
    assert "Deep Horizon" in norm2
    assert "Doctor Maya Lin" in norm2
    assert "seventh" in norm2.lower()
    assert "B C E" in norm2


def test_spoken_ear_rewriter_and_slerp_voice_manifold() -> None:
    # 1. Rewrite breathless eye-oriented sentences + strip placard parens
    dense = (
        "Six of the original Seven Ancient Wonders — the Colossus of Rhodes, the Lighthouse of Alexandria, "
        "the Mausoleum at Halicarnassus, the Temple of Artemis at Ephesus, the Statue of Zeus at Olympia, "
        "and the Hanging Gardens — were erased by earthquakes and stone-quarrying (c. 226 BCE)."
    )
    spoken = rewrite_for_spoken_ear(dense)
    assert "(c. 226 BCE)" not in spoken
    assert spoken.count(".") >= 2

    # 2. Abbreviation-safe sentence splitter never splits right after 'Dr.'
    chunks = _split_spoken_sentences("For Dr. Maya Lin, I'm Alex Vance. Keep looking deeper.")
    assert not any(c.endswith("Dr.") for c in chunks)

    # 3. Geodesic SLERP preserves hypersphere norm (unlike naive linear averaging which shrinks interior norm)
    rng = np.random.default_rng(7)
    v0 = rng.normal(size=(510, 1, 256)).astype(np.float32)
    v1 = rng.normal(size=(510, 1, 256)).astype(np.float32)
    blended = _slerp_voice_vectors(v0, v1, 0.35)
    n0 = np.linalg.norm(v0, axis=-1).mean()
    n1 = np.linalg.norm(v1, axis=-1).mean()
    nb = np.linalg.norm(blended, axis=-1).mean()
    assert abs(nb - (0.65 * n0 + 0.35 * n1)) < 1e-3


def test_one_hour_script_creation_and_editorial_audit(tmp_path: Path) -> None:
    spec = parse_command_to_spec("build a 1 hour long podcast on 7 wonders of the world")
    env = StudioEnvironment(
        ffmpeg_bin="ffmpeg",
        tts_backend="piper",
        kokoro_model_path=None,
        kokoro_voices_path=None,
        piper_male_onnx=None,
        piper_female_onnx=None,
        llm_backend="corpus_showrunner",
        llm_model_id_or_path="voxforge-v1",
    )
    dossier = build_research_dossier(spec, tmp_path)
    draft = generate_episode_script(spec, dossier, env)

    # Inject a synthetic cliché and an overlong 80-word monologue to verify Stage A repairs both!
    draft.chapters[0].turns.insert(
        2,
        DialogueTurn(
            turn_id=999,
            chapter_index=1,
            speaker="HOST_A",
            text=(
                "Without further ado let's dive in and delve into this rich tapestry because it is a testament "
                "to ancient stonecraft across every single dynasty and empire that ever existed on the Mediterranean "
                "coastline where engineers worked tirelessly day and night moving massive blocks across desert plains "
                "and mountain passes to construct monuments that still astonish modern structural researchers today."
            ),
            spoken_text="",
            camera="wide_studio",
        ),
    )

    improved, audit = audit_and_improve_script(draft, dossier)
    assert audit.passed
    assert audit.final_score >= 90.0
    assert improved.total_words >= int(spec.target_word_count * 0.92)
    # Clichés must be completely purged
    joined = " ".join(t.text.lower() for t in improved.all_turns)
    assert "delve into" not in joined
    assert "rich tapestry" not in joined
    assert "without further ado" not in joined
    # Zero turns exceeding 64 words
    assert all(t.word_count <= 64 for t in improved.all_turns)


def _offline_env() -> StudioEnvironment:
    return StudioEnvironment(
        ffmpeg_bin="ffmpeg",
        tts_backend="piper",
        kokoro_model_path=None,
        kokoro_voices_path=None,
        piper_male_onnx=None,
        piper_female_onnx=None,
        llm_backend="corpus_showrunner",
        llm_model_id_or_path="voxforge-v1",
    )


def test_host_speaker_balance_is_word_weighted_not_turn_weighted(tmp_path: Path) -> None:
    """Regression: alternating turns left Host A with ~67% of spoken words on 9/10 niches.

    Turn-by-turn alternation balances *turn count*, but E4 measures *word* share, so scripts
    whose long analytical turns all landed on Host A failed the 36%-64% band even after healing.
    """
    for topic in (
        "the silk road trade networks of antiquity",
        "the rise of artificial intelligence",
        "the science of baking bread",
    ):
        spec = parse_command_to_spec(f"build a 90 second documentary on {topic}")
        dossier = build_research_dossier(spec, tmp_path / topic.replace(" ", "_"))
        script, audit = audit_and_improve_script(generate_episode_script(spec, dossier, _offline_env()), dossier)

        e4 = next(c for c in audit.checks if c.check_id == "E4")
        ratio_a = sum(t.word_count for t in script.all_turns if t.speaker == "HOST_A") / script.total_words
        assert 0.36 <= ratio_a <= 0.64, f"{topic}: Host A holds {ratio_a * 100:.1f}% of words"
        assert e4.passed, f"{topic}: E4 failed -> {e4.measured_value}"
        assert audit.passed, f"{topic}: Stage A failed at {audit.final_score}"


def test_offline_corpus_shortfall_is_measured_not_guessed() -> None:
    """Long arbitrary topics exceed the offline corpus; the shortfall must be detectable.

    The pipeline warns off the *healed* script rather than an a-priori estimate, because the
    estimated capacity is not knowable up front (the 7 Wonders pack legitimately fills an hour).
    """
    short_spec = parse_command_to_spec("build a 90 second documentary on the history of money and banking")
    hour_spec = parse_command_to_spec("build a 1 hour documentary on the history of money and banking")
    with tempfile.TemporaryDirectory() as td:
        short_dossier = build_research_dossier(short_spec, Path(td) / "short")
        short_script, _ = audit_and_improve_script(
            generate_episode_script(short_spec, short_dossier, _offline_env()), short_dossier
        )
        hour_dossier = build_research_dossier(hour_spec, Path(td) / "hour")
        hour_script, hour_audit = audit_and_improve_script(
            generate_episode_script(hour_spec, hour_dossier, _offline_env()), hour_dossier
        )

    # A 90-second brief fits the offline corpus comfortably (no warning fires).
    assert short_script.total_words >= short_spec.target_word_count * 0.85
    # A one-hour arbitrary topic does not: the healed script is far under budget, so the
    # pipeline emits its offline-corpus warning instead of shipping a silently short film.
    assert hour_script.total_words < hour_spec.target_word_count * 0.85
    e1 = next(c for c in hour_audit.checks if c.check_id == "E1")
    assert not e1.passed
