"""Semantic shot director: bind every spoken clause to a relevant, non-repeating visual.

The legacy pipeline resolved pictures with ten wonder keyword rules and otherwise fell back to
a single generic rotunda (or rotated monuments by segment index), so arbitrary-topic films
looped one model and narration could disagree with the picture. This module replaces that with
an explicit, auditable directive per clause:

* wonder clauses  -> the exact monument scene (wide / close / LiDAR / bespoke sim / globe)
* arbitrary topic -> a semantic category routed to a family of clause-driven diagram engines
* place/route clauses -> orbital globe; citation clauses -> kinetic typography
* anything unmatchable -> a NEUTRAL diagram derived from the clause itself, never an unrelated model

A rolling treatment history guarantees the same ``(subject, treatment)`` pair never repeats
within a sliding window, which removes the "same scene on loop" feeling even when one subject
dominates an entire chapter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass

from voxforge.video.procedural_world import SCENE_CAPTIONS, match_wonder_scene_id

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

MODALITY_MONUMENT = "monument_3d"
MODALITY_GLOBE = "orbital_globe"
MODALITY_PHYSICS_SIM = "physics_sim"
MODALITY_LIDAR = "lidar_cad"
MODALITY_QUOTE = "kinetic_quote"
MODALITY_DIAGRAM = "semantic_diagram"

# Six visually distinct generic engines implemented in ``semantic_diagrams.py``.
TPL_TERRAIN = "terrain_tectonic"
TPL_NETWORK = "node_network"
TPL_PARTICLES = "particle_cluster"
TPL_PROCESS = "process_flow"
TPL_STATS = "stat_dashboard"
TPL_TIMELINE = "timeline_arc"

ALL_TEMPLATES = (TPL_TERRAIN, TPL_NETWORK, TPL_PARTICLES, TPL_PROCESS, TPL_STATS, TPL_TIMELINE)

# Semantic categories for arbitrary topics, ordered most-specific first. Each category owns a
# template ladder so consecutive clauses about one topic still receive distinct pictures.
_CATEGORY_LEXICON: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    (
        "geoscience",
        (
            "volcano", "lava", "magma", "eruption", "earthquake", "tectonic", "tsunami",
            "ocean", "sea", "river", "lake", "glacier", "ice", "mountain", "canyon", "erosion",
            "climate", "weather", "storm", "hurricane", "planet", "atmosphere", "fossil",
            "mineral", "rock", "sediment", "geology", "earthquake", "wildfire", "drought",
        ),
        (TPL_TERRAIN, TPL_PARTICLES, TPL_STATS, TPL_TIMELINE),
    ),
    (
        "biology_health",
        (
            "cell", "dna", "gene", "genetic", "immune", "antibod", "virus", "viral", "bacteria",
            "vaccine", "vaccination", "disease", "infection", "brain", "neuron", "organ",
            "blood", "protein", "enzyme", "hormone", "species", "evolution", "ecosystem",
            "habitat", "organism", "biology", "anatomy", "cancer", "metabolism", "microbe",
        ),
        (TPL_PARTICLES, TPL_PROCESS, TPL_STATS, TPL_NETWORK),
    ),
    (
        "food_process",
        (
            "cook", "cooking", "bake", "bread", "dough", "ferment", "recipe", "ingredient",
            "flour", "yeast", "kitchen", "flavor", "flavour", "roast", "simmer", "boil",
            "culinary", "meal", "cuisine", "spice", "pastry", "sourdough",
        ),
        (TPL_PROCESS, TPL_STATS, TPL_PARTICLES, TPL_TIMELINE),
    ),
    (
        "technology",
        (
            "artificial intelligence", "algorithm", "software", "computer", "computing",
            "network", "internet", "data", "digital", "machine learning", "robot", "chip",
            "quantum", "code", "server", "automation", "model", "neural", "silicon",
            "cyber", "satellite", "sensor", "engine", "technology",
        ),
        (TPL_NETWORK, TPL_PROCESS, TPL_STATS, TPL_PARTICLES),
    ),
    (
        "society_economy",
        (
            "trade", "traded", "market", "economy", "economic", "money", "finance", "business",
            "company", "transport", "shipping", "supply", "commerce", "merchant", "currency",
            "bank", "investment", "industry", "urban", "city", "cities", "migration", "population",
        ),
        (TPL_NETWORK, TPL_STATS, TPL_TIMELINE, TPL_PROCESS),
    ),
    (
        "history_humanities",
        (
            "war", "battle", "empire", "emperor", "kingdom", "dynasty", "ancient", "civilization",
            "revolution", "century", "medieval", "army", "conquest", "treaty", "king", "queen",
            "pharaoh", "tribe", "settlement", "archaeolog", "ruler", "reign",
        ),
        (TPL_TIMELINE, TPL_NETWORK, TPL_TERRAIN, TPL_STATS),
    ),
]

_DEFAULT_TEMPLATES = (TPL_STATS, TPL_NETWORK, TPL_PROCESS, TPL_PARTICLES)

# Treatment ladders for matched wonder subjects (each maps to an existing pure-code engine).
_WONDER_TREATMENTS = (
    ("exterior_wide", MODALITY_MONUMENT),
    ("detail_closeup", MODALITY_MONUMENT),
    ("lidar_scan", MODALITY_LIDAR),
    ("physics_simulation", MODALITY_PHYSICS_SIM),
    ("alternate_angle", MODALITY_MONUMENT),
    ("orbital_context", MODALITY_GLOBE),
)
_REPEAT_WINDOW = 5


@dataclass(frozen=True)
class VisualDirective:
    """One clause's fully resolved picture, serialised into ``scene_plan.json`` for auditing."""

    subject_key: str
    subject_label: str
    modality: str
    scene_id: int  # 0-8 wonder id, -1 for generic diagrams/globe/quote
    diagram_template: str
    treatment: str
    on_screen_label: str
    confidence: float
    source: str  # wonder_keyword | semantic_category | place_globe | citation_quote | clause_diagram
    # Exact clause||context hint the router resolved from, enabling independent Stage E re-checks.
    context: str = ""

    def to_dict(self) -> dict:
        return {
            "subject_key": self.subject_key,
            "subject_label": self.subject_label,
            "modality": self.modality,
            "scene_id": self.scene_id,
            "diagram_template": self.diagram_template,
            "treatment": self.treatment,
            "on_screen_label": self.on_screen_label,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "context": self.context,
        }


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s or "subject"


def subject_seed(subject_key: str) -> int:
    """Deterministic per-subject RNG seed so different topics render distinct geometry/palettes."""
    return int(hashlib.sha1(subject_key.encode("utf-8")).hexdigest()[:8], 16)


def classify_semantic_category(text: str) -> tuple[str, str]:
    """Return ``(category, matched_keyword)``; unmatched text lands in ``abstract``."""
    low = f" {text.lower()} "
    best_cat, best_kw, best_hits = "abstract", "", 0
    for cat, keywords, _templates in _CATEGORY_LEXICON:
        hits = [kw for kw in keywords if kw in low]
        if len(hits) > best_hits:
            best_cat, best_kw, best_hits = cat, hits[0], len(hits)
    return best_cat, best_kw


def _category_templates(category: str) -> tuple[str, ...]:
    for cat, _kw, templates in _CATEGORY_LEXICON:
        if cat == category:
            return templates
    return _DEFAULT_TEMPLATES


def _wonder_subject(scene_id: int) -> tuple[str, str]:
    caption = SCENE_CAPTIONS.get(scene_id, "")
    label = caption.split("·", 1)[1].strip() if "·" in caption else caption
    # e.g. "Great Pyramid of Giza & Necropolis (2560 BCE)" -> stable key + short label
    short = re.sub(r"\s*\([^)]*\)", "", label).split("&")[0].strip()
    return slugify(short), short


class SemanticShotDirector:
    """Stateful per-episode director enforcing relevance and the no-repeat treatment window."""

    def __init__(self, episode_subject: str) -> None:
        self.episode_subject = episode_subject or ""
        self.episode_slug = slugify(self.episode_subject)
        self._history: list[VisualDirective] = []

    # -- public API ---------------------------------------------------------

    def directive_for_clause(self, clause: str, context: str, intent: str) -> VisualDirective:
        """Resolve one narration clause. ``intent`` is the renderer lexicon label.

        ``eng`` -> mechanism/simulation, ``geo`` -> globe, ``quote`` -> typography,
        ``people``/``scene`` -> subject-led picture.
        """
        clause = (clause or "").strip()
        hint = f"{clause} || {context}".strip(" |")
        wonder_id = match_wonder_scene_id(hint)

        if wonder_id is not None:
            directive = self._wonder_directive(wonder_id, clause, intent)
        elif intent == "geo" or self._is_place_driven(clause):
            directive = self._globe_directive(clause, context)
        elif intent == "quote":
            directive = self._quote_directive(clause, context)
        else:
            directive = self._diagram_directive(clause, context, intent)

        directive = dataclasses.replace(directive, context=hint)
        self._history.append(directive)
        if len(self._history) > 24:
            self._history = self._history[-24:]
        return directive

    def attribution_directive(self, chapter_title: str, chapter_location: str, intent: str) -> VisualDirective:
        """Bind voice-attribution/establishing beats to the chapter subject (no placeholder shots)."""
        context = f"{chapter_title} {chapter_location}".strip()
        return self.directive_for_clause(context, context, intent if intent != "scene" else "people")

    # -- wonder routing -----------------------------------------------------

    def _wonder_directive(self, scene_id: int, clause: str, intent: str) -> VisualDirective:
        key, label = _wonder_subject(scene_id)
        if intent == "geo":
            directive = VisualDirective(
                key, label, MODALITY_GLOBE, scene_id, "", "orbital_context", label, 0.95, "place_globe"
            )
        elif intent == "quote":
            directive = VisualDirective(
                key, label, MODALITY_QUOTE, scene_id, "", "kinetic_typography", label, 0.9, "citation_quote"
            )
        elif intent == "eng":
            directive = VisualDirective(
                key, label, MODALITY_PHYSICS_SIM, scene_id, "", "physics_simulation", label, 0.95, "wonder_keyword"
            )
        else:
            treatment, modality = self._pick_wonder_treatment(key)
            directive = VisualDirective(key, label, modality, scene_id, "", treatment, label, 0.95, "wonder_keyword")
        return self._avoid_repeat(directive, key, clause)

    def _pick_wonder_treatment(self, key: str) -> tuple[str, str]:
        recent = {(h.subject_key, h.treatment) for h in self._history[-_REPEAT_WINDOW:]}
        for treatment, modality in _WONDER_TREATMENTS:
            if (key, treatment) not in recent:
                return treatment, modality
        return _WONDER_TREATMENTS[len(self._history) % len(_WONDER_TREATMENTS)]

    # -- generic routing ----------------------------------------------------

    def _globe_directive(self, clause: str, context: str) -> VisualDirective:
        category, kw = classify_semantic_category(f"{clause} {context}")
        label = kw.replace("_", " ").title() if kw else self.episode_subject or "Global Context"
        key = f"{self.episode_slug}_place"
        return VisualDirective(key, label, MODALITY_GLOBE, -1, "", "orbital_context", label, 0.8, "place_globe")

    def _quote_directive(self, clause: str, context: str) -> VisualDirective:
        category, kw = classify_semantic_category(f"{clause} {context}")
        label = kw.replace("_", " ").title() if kw else self.episode_subject or "Primary Source"
        key = f"{self.episode_slug}_quote"
        return VisualDirective(key, label, MODALITY_QUOTE, -1, "", "kinetic_typography", label, 0.8, "citation_quote")

    def _diagram_directive(self, clause: str, context: str, intent: str) -> VisualDirective:
        category, kw = classify_semantic_category(f"{clause} {context} {self.episode_subject}")
        templates = _category_templates(category)
        # Mechanism/engineering language prefers process/terrain explanations; number-heavy
        # clauses prefer the stat dashboard.
        if intent == "eng" and TPL_PROCESS in templates:
            preferred = (TPL_PROCESS, *[t for t in templates if t != TPL_PROCESS])
        else:
            preferred = templates
        template = self._pick_template(preferred)
        label = kw.replace("_", " ").title() if kw else (self.episode_subject or "Explainer")
        key = f"{self.episode_slug}_{category}"
        return VisualDirective(
            key, label, MODALITY_DIAGRAM, -1, template, f"diagram_{template}", label, 0.75, "semantic_category"
        )

    def _pick_template(self, templates: tuple[str, ...]) -> str:
        recent = {h.diagram_template for h in self._history[-_REPEAT_WINDOW:] if h.diagram_template}
        for tpl in templates:
            if tpl not in recent:
                return tpl
        # Whole ladder recently used: rotate deterministically rather than repeat the newest one.
        return templates[len(self._history) % len(templates)]

    # -- guards -------------------------------------------------------------

    def _avoid_repeat(self, directive: VisualDirective, key: str, clause: str) -> VisualDirective:
        recent = {(h.subject_key, h.treatment) for h in self._history[-_REPEAT_WINDOW:]}
        if (key, directive.treatment) not in recent:
            return directive
        # Fall through the wonder ladder until we find an unused treatment for this subject.
        if directive.modality in (MODALITY_MONUMENT, MODALITY_LIDAR, MODALITY_PHYSICS_SIM, MODALITY_GLOBE):
            for treatment, modality in _WONDER_TREATMENTS:
                if (key, treatment) not in recent:
                    return VisualDirective(
                        key, directive.subject_label, modality, directive.scene_id, "",
                        treatment, directive.on_screen_label, directive.confidence, directive.source,
                    )
        return directive

    @staticmethod
    def _is_place_driven(clause: str) -> bool:
        place_words = (
            "located", "situated", "region", "across", "border", "coast", "valley", "island",
            "peninsula", "continent", "trade route", "expedition", "migrat", "journey",
        )
        low = clause.lower()
        return any(w in low for w in place_words)
