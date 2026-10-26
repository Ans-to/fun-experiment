"""Strongly-typed domain models for the VoxForge Studio production pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

CameraAngle = Literal[
    "intro_bumper",
    "chapter_card",
    "wide_studio",
    "closeup_host_a",
    "closeup_host_b",
    "reaction_shot",
    "studio_insert",
    "quote_card",
    "broll_cinema",
    "split_fact_card",
    "outro_card",
]

MusicCue = Literal[
    "cold_open_pulse",
    "theme_swell",
    "ambient_warm",
    "wonder_pad",
    "tension_drone",
    "finale_resolve",
    "none",
]


@dataclass
class ShowSpec:
    """Parsed intent and target production parameters for an episode."""

    raw_command: str
    topic: str
    show_title: str
    episode_title: str
    episode_subtitle: str
    target_duration_sec: float
    target_word_count: int
    chapter_count: int
    words_per_minute: float = 144.0
    style: str = "documentary_podcast"
    slug: str = "episode"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KeyMetric:
    """Structured fact metric for broadcast infographic & lower-third callouts."""

    label: str
    value: str


@dataclass
class VisualAsset:
    """Curated visual B-roll asset with quality telemetry and attribution."""

    asset_id: str
    subject_tag: str
    file_path: str
    width: int
    height: int
    sharpness_score: float
    mean_luminance: float
    caption: str
    credit: str
    license_name: str
    source_url: str = ""
    is_procedural: bool = False
    passed_qa: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubTopicDossier:
    """Deep research dossier for a single act / chapter of the episode."""

    index: int
    title: str
    subtitle: str
    era_or_date: str
    location: str
    summary: str
    origin_and_context: list[str]
    engineering_and_mechanics: list[str]
    human_stories_and_paradoxes: list[str]
    legacy_and_modern_science: list[str]
    metrics: list[KeyMetric] = field(default_factory=list)
    visual_queries: list[str] = field(default_factory=list)
    visual_assets: list[VisualAsset] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchDossier:
    """Complete multi-source research corpus for the entire podcast episode."""

    topic: str
    overview: str
    thesis_statement: str
    cold_open_hook_facts: list[str]
    synthesis_comparisons: list[str]
    subtopics: list[SubTopicDossier]
    sources_consulted: list[str] = field(default_factory=list)
    live_web_enriched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DialogueTurn:
    """Single spoken utterance in the show script with full director/camera cues."""

    turn_id: int
    chapter_index: int
    speaker: Literal["HOST_A", "HOST_B"]
    text: str
    spoken_text: str  # Pronunciation-normalized text for neural TTS
    camera: CameraAngle
    visual_asset_path: str = ""
    visual_caption: str = ""
    visual_credit: str = ""
    # Explicit narration→visual binding, resolved from this turn's spoken text rather
    # than inherited from a (possibly mixed-subject) chapter-level asset.
    visual_subject: str = ""
    visual_scene_id: int | None = None
    fact_callout_title: str = ""
    fact_metrics: list[KeyMetric] = field(default_factory=list)
    pull_quote: str = ""
    insert_variant: str = ""
    music_cue: MusicCue = "ambient_warm"
    pause_after_ms: int = 280
    is_chapter_opener: bool = False
    # Populated during audio synthesis & timeline assembly:
    audio_path: str = ""
    duration_sec: float = 0.0
    start_sec: float = 0.0
    end_sec: float = 0.0

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChapterScript:
    """A single dramatic act / chapter of the episode."""

    chapter_index: int
    title: str
    subtitle: str
    era_or_date: str
    location: str
    music_cue: MusicCue
    card_duration_sec: float = 3.6
    card_asset_path: str = ""
    turns: list[DialogueTurn] = field(default_factory=list)
    start_sec: float = 0.0
    end_sec: float = 0.0

    @property
    def word_count(self) -> int:
        return sum(t.word_count for t in self.turns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeScript:
    """Complete multi-chapter broadcast script."""

    spec: ShowSpec
    chapters: list[ChapterScript]
    llm_backend_used: str = "corpus_showrunner"
    revision_number: int = 1
    intro_duration_sec: float = 3.6
    outro_duration_sec: float = 4.6
    next_episode_tease: str = "Lost Megacities of Antiquity: Angkor Wat, Teotihuacan & Alexandria"

    @property
    def all_turns(self) -> list[DialogueTurn]:
        return [t for ch in self.chapters for t in ch.turns]

    @property
    def total_words(self) -> int:
        return sum(ch.word_count for ch in self.chapters)

    def estimated_duration_sec(self) -> float:
        words_sec = (self.total_words / max(self.spec.words_per_minute, 100.0)) * 60.0
        pauses_sec = sum(t.pause_after_ms / 1000.0 for t in self.all_turns)
        cards_sec = sum(ch.card_duration_sec for ch in self.chapters)
        return self.intro_duration_sec + words_sec + pauses_sec + cards_sec + self.outro_duration_sec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AudioMasterMeta:
    """Broadcast loudness & signal telemetry of the mastered stereo soundtrack."""

    master_wav_path: str
    master_mp3_path: str
    duration_sec: float
    sample_rate: int
    channels: int
    integrated_lufs: float
    true_peak_dbtp: float
    loudness_range_lu: float
    max_silence_gap_sec: float
    tts_backend_used: str
    utterances_synthesized: int
    utterances_repaired: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditCheck:
    """Individual measurable broadcast QA check result."""

    check_id: str
    name: str
    passed: bool
    score: float  # 0.0 to 100.0
    measured_value: str
    target_spec: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageAudit:
    """Before-and-after QA audit record for a pipeline stage."""

    stage_name: str
    initial_score: float
    final_score: float
    passed: bool
    iterations: int
    improvements_applied: list[str] = field(default_factory=list)
    checks: list[AuditCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeAuditReport:
    """Full studio audit report proving Create -> Audit -> Improve across all domains."""

    episode_slug: str
    overall_initial_score: float
    overall_final_score: float
    studio_certified: bool
    stages: list[StageAudit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildResult:
    """Final output artifact bundle returned by VoxForge Studio."""

    output_dir: Path
    video_mp4_path: Path
    audio_mp3_path: Path
    subtitles_srt_path: Path
    subtitles_vtt_path: Path
    script_json_path: Path
    dossier_json_path: Path
    audit_json_path: Path
    audit_md_path: Path
    duration_sec: float
    total_words: int
    chapter_count: int
    audit_report: EpisodeAuditReport


@dataclass
class ScenePlanEntry:
    """Authoritative narration→visual mapping record for one rendered scene/beat.

    Maintained per the narration-aligned scene-plan specification: exact spoken clause,
    audio-derived timestamps, semantic subject/action, required + selected visual family,
    asset provenance, on-screen text, relevance criteria and validation status.
    """

    scene_id: int
    seg_index: int
    chapter_index: int
    turn_id: int
    narration_clause: str
    narration_start_sec: float
    narration_end_sec: float
    subject: str
    action_or_event: str
    required_visual: str
    visual_family: str
    visual_asset_source: str
    scene_start_sec: float
    scene_end_sec: float
    on_screen_text: str
    relevance_criteria: str
    validation_status: str = "pending"
    validation_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenePlan:
    """Ordered scene plan binding script ⇄ voice-over ⇄ visuals ⇄ final timeline."""

    topic: str
    entries: list[ScenePlanEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"topic": self.topic, "scenes": [e.to_dict() for e in self.entries]}
