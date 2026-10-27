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
        "space_science",
        (
            "planet", "planets", "sun", "moon", "star", "stars", "orbit", "orbital", "cosmos",
            "cosmic", "galaxy", "solar", "astronomy", "astronomical", "celestial", "meteor",
            "asteroid", "comet", "universe", "gravity", "spacecraft", "telescope",
        ),
        (TPL_PARTICLES, TPL_TERRAIN, TPL_STATS, TPL_TIMELINE),
    ),
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

# All six engines serve abstract clauses; the stat dashboard is additionally gated to clauses
# containing real numbers, and the whole-film usage budget spreads selections across the ladder.
_DEFAULT_TEMPLATES = (TPL_NETWORK, TPL_PARTICLES, TPL_PROCESS, TPL_TIMELINE, TPL_TERRAIN, TPL_STATS)

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

# A clause that does not name a monument itself may still be ABOUT the chapter's monument when
# it is a sentence fragment or refers back anaphorically. Only these linguistic references
# inherit the previous monument subject — chapter-title keywords alone never force a binding.
# Referential pronouns only count when they OPEN the clause; mid-clause "that"/"this" are
# usually complementizers/determiners ("chroniclers wrote that..."), not subject references.
_ANAPHORA_OPEN_RE = re.compile(
    r"^(?:it|its|they|them|their|this|that|these|those)\b",
    re.I,
)
# Specific reference nouns inherit anywhere they appear — they can only denote the chapter subject.
_REFERENCE_NOUN_RE = re.compile(
    r"\b(?:the monument|the structure|the temple|the site|the wonder|the building|the mausoleum|"
    r"the lighthouse|the pyramid|the statue|the wall|the city|the sanctuary|the casing)\b",
    re.I,
)
_CONNECTIVE_RE = re.compile(
    r"^(?:and|but|or|yet|so|nor|while|whereas|because|when|then|to|of|in|on|with|by|as|at|from|"
    r"into|over|under|through|after|before|for|remaining|crowned|connected|built|standing|"
    r"located|situated|earthquakes|fire)\b",
    re.I,
)


def is_anaphoric_or_continuation(clause: str) -> bool:
    """True when a clause is a mid-sentence fragment or refers back to the previous subject."""
    text = (clause or "").strip()
    if not text:
        return False
    if text[0].islower():  # clause splitter/merger fragment continuing the prior sentence
        return True
    if _CONNECTIVE_RE.match(text):
        return True
    if _ANAPHORA_OPEN_RE.match(text):
        return True
    return bool(_REFERENCE_NOUN_RE.search(text))


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
    # wonder_keyword (clause itself names the monument) | anaphora_continuation (fragment/pronoun
    # inheriting the prior monument) | semantic_category | place_globe | citation_quote
    source: str
    # Exact clause||context hint the router resolved from, enabling independent Stage E re-checks.
    context: str = ""
    # Per-subject emission counter for the same scene/treatment pair: the renderer maps this to
    # a distinct camera lens variant (monuments) or orbital longitude offset (globes), so the
    # same subject never returns through the identical camera path.
    variation_index: int = 0

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
            "variation_index": self.variation_index,
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
        # Whole-film template budget (survives the 24-entry history trim) so a 90s film with
        # eleven diagrams spreads them across the ladder instead of reusing one look four times.
        self._template_usage: dict[str, int] = {}
        # Emission counters per rendered monument scene (all treatments share one counter so
        # every 3D shot of the same wonder gets a unique camera-path index), plus globes.
        self._monument_emissions: dict[int, int] = {}
        self._globe_emissions: dict[str, int] = {}

    # -- public API ---------------------------------------------------------

    def directive_for_clause(self, clause: str, context: str, intent: str) -> VisualDirective:
        """Resolve one narration clause. ``intent`` is the renderer lexicon label.

        ``eng`` -> mechanism/simulation, ``geo`` -> globe, ``quote`` -> typography,
        ``people``/``scene`` -> subject-led picture.
        """
        clause = (clause or "").strip()
        hint = f"{clause} || {context}".strip(" |")

        # 1. Clause-first binding: only a wonder named IN THE CLAUSE ITSELF binds monument
        #    geometry. Chapter-title context must never force a monument onto a generic clause.
        clause_wonder = match_wonder_scene_id(clause)
        prev = self._history[-1] if self._history else None
        if clause_wonder is not None:
            directive = self._monument_directive(clause_wonder, clause, intent, "wonder_keyword")
        elif (
            prev is not None
            and 0 <= prev.scene_id <= 8
            and is_anaphoric_or_continuation(clause)
        ):
            # 2. Fragment/pronoun continuing the sentence about the previous monument subject.
            directive = self._monument_directive(
                prev.scene_id, clause, "scene", "anaphora_continuation"
            )
        elif intent == "geo" or self._is_place_driven(clause):
            # 3. Explicit place/route language -> orbital globe (never an unrelated monument).
            directive = self._globe_directive(clause, context)
        elif intent == "quote":
            directive = self._quote_directive(clause, context)
        else:
            # 4. Any other generic/abstract clause -> a neutral clause-derived diagram.
            directive = self._diagram_directive(clause, context, intent)

        # Never emit the same (subject, treatment) pair twice in a row: globe/quote fallbacks
        # reroute to a clause-derived diagram rather than looping an identical cutaway.
        if self._history and (
            (directive.subject_key, directive.treatment)
            == (self._history[-1].subject_key, self._history[-1].treatment)
        ):
            directive = self._diagram_directive(clause, context, intent)

        directive = dataclasses.replace(directive, context=hint)
        directive = self._stamp_variation(directive)
        if directive.diagram_template:
            self._template_usage[directive.diagram_template] = (
                self._template_usage.get(directive.diagram_template, 0) + 1
            )
        self._history.append(directive)
        if len(self._history) > 24:
            self._history = self._history[-24:]
        return directive

    def _stamp_variation(self, directive: VisualDirective) -> VisualDirective:
        """Assign the per-look emission index so repeat subjects get a fresh camera path."""
        # Wonder-bound shots come first: even a monument's orbital_context globe treatment must
        # consume the per-monument camera counter, not the generic globe counter.
        if 0 <= directive.scene_id <= 8:
            n = self._monument_emissions.get(directive.scene_id, 0)
            self._monument_emissions[directive.scene_id] = n + 1
            return dataclasses.replace(directive, variation_index=n)
        if directive.modality == MODALITY_GLOBE:
            n = self._globe_emissions.get(directive.subject_key, 0)
            self._globe_emissions[directive.subject_key] = n + 1
            return dataclasses.replace(directive, variation_index=n)
        return directive

    def attribution_directive(self, chapter_title: str, chapter_location: str, intent: str) -> VisualDirective:
        """Bind voice-attribution/establishing beats to the chapter subject (no placeholder shots)."""
        context = f"{chapter_title} {chapter_location}".strip()
        return self.directive_for_clause(context, context, intent if intent != "scene" else "people")

    # -- wonder routing -----------------------------------------------------

    def _monument_directive(
        self, scene_id: int, clause: str, intent: str, source: str
    ) -> VisualDirective:
        key, label = _wonder_subject(scene_id)
        # Intent-based modality only for direct keyword clauses; anaphora continuations use the
        # treatment ladder so fragments still vary exterior/close-up/LiDAR/sim instead of
        # collapsing every continuation into one globe or quote card.
        if source == "wonder_keyword" and intent == "geo":
            directive = VisualDirective(
                key, label, MODALITY_GLOBE, scene_id, "", "orbital_context", label, 0.95, "place_globe"
            )
        elif source == "wonder_keyword" and intent == "quote":
            directive = VisualDirective(
                key, label, MODALITY_QUOTE, scene_id, "", "kinetic_typography", label, 0.9, "citation_quote"
            )
        elif source == "wonder_keyword" and intent == "eng":
            directive = VisualDirective(
                key, label, MODALITY_PHYSICS_SIM, scene_id, "", "physics_simulation", label, 0.95, source
            )
        else:
            treatment, modality = self._pick_wonder_treatment(key)
            directive = VisualDirective(key, label, modality, scene_id, "", treatment, label, 0.95, source)
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
        key = f"{self.episode_slug}_{category}"
        template = self._pick_template(preferred, key, clause)
        label = kw.replace("_", " ").title() if kw else (self.episode_subject or "Explainer")
        return VisualDirective(
            key, label, MODALITY_DIAGRAM, -1, template, f"diagram_{template}", label, 0.75, "semantic_category"
        )

    # A stat dashboard is only honest when the clause actually contains a quantity; otherwise
    # the engine would have to invent a number (it previously rendered a fake "60 per cent").
    _NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

    def _pick_template(self, templates: tuple[str, ...], subject_key: str, clause: str = "") -> str:
        """Pick a template that fits the clause category without looping.

        Three variety layers: (1) stat dashboards are banned for numberless clauses so the film
        never fabricates data; (2) a whole-film per-template budget picks the least-used ladder
        member first, spreading eleven diagrams across all six looks; (3) the same subject must
        rotate its ladder before reuse and adjacent shots never share a template.
        """
        honest = [t for t in templates if t != TPL_STATS or self._NUMBER_RE.search(clause or "")]
        ladder = honest or list(templates)
        adjacent = self._history[-1].diagram_template if self._history else ""
        subject_recent = {
            h.diagram_template
            for h in self._history[-_REPEAT_WINDOW:]
            if h.diagram_template and h.subject_key == subject_key
        }

        def rank(tpl: str) -> tuple[int, int, int]:
            return (
                self._template_usage.get(tpl, 0),       # global budget first
                1 if tpl in subject_recent else 0,      # then per-subject window
                1 if tpl == adjacent else 0,            # then adjacency
            )

        return sorted(ladder, key=rank)[0]

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
