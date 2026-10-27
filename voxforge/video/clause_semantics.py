"""Clause-level semantic analysis for meaning-driven documentary diagrams.

The diagram engines must depict what the narrator *means*, not just decorate a clause. This
module turns a spoken clause into a structured, fully auditable payload:

* ``archetype`` — the visual story shape (building, destruction, timeline, quantity, cosmos…);
* ``direction`` — rising / falling / stable / neutral, derived from the clause's verbs;
* ``terms`` — content words taken verbatim from the clause (the only legal node/box labels);
* ``quantities`` — real ``(number, unit, noun)`` triples parsed from the clause text;
* ``time_marks`` — verbatim date/duration expressions for timeline axes;
* ``relations`` — best-effort ``(entity, relation word, entity)`` triples;
* ``labels`` — every on-diagram string, guaranteed to be a substring of the clause so Stage E
  can mechanically verify that nothing shown was invented.

Pure stdlib, deterministic, no I/O — every rule is unit-tested against real narration clauses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Archetypes map 1:1 onto a diagram layout family in ``semantic_diagrams.py``.
ARCH_BUILD = "build_process"
ARCH_DESTRUCTION = "destruction_forces"
ARCH_ENDURE = "enduring_legacy"
ARCH_HISTORY = "history_timeline"
ARCH_QUANTITY = "measured_quantity"
ARCH_CATALOG = "catalog_count"
ARCH_COSMOS = "cosmos_system"
ARCH_GEO = "natural_world"
ARCH_JOURNEY = "journey_route"
ARCH_CONCEPT = "abstract_concept"

KNOWN_ARCHETYPES = frozenset(
    {
        ARCH_BUILD, ARCH_DESTRUCTION, ARCH_ENDURE, ARCH_HISTORY, ARCH_QUANTITY,
        ARCH_CATALOG, ARCH_COSMOS, ARCH_GEO, ARCH_JOURNEY, ARCH_CONCEPT,
    }
)

# Ordered most-specific first; scoring picks the archetype with the strongest lexical evidence.
_ARCH_LEXICON: list[tuple[str, tuple[str, ...]]] = [
    (
        ARCH_COSMOS,
        ("planet", "planets", "sun", "moon", "star", "stars", "cosmos", "cosmic", "cosmology",
         "orbit", "orbital", "solar", "celestial", "galaxy", "universe", "astronom"),
    ),
    (
        ARCH_DESTRUCTION,
        ("destroy", "destruction", "fire", "burn", "earthquake", "earthquakes", "collapse",
         "ruin", "ruins", "fell", "fallen", "war", "greed", "storm", "flood", "eruption",
         "shatter", "crumble", "toppled", "sank", "wreck"),
    ),
    (
        ARCH_BUILD,
        ("build", "built", "construct", "construction", "quarry", "carve", "carved", "raise",
         "raised", "erect", "engineer", "engineered", "craft", "assemble", "stack", "hewn",
         "masonry", "architect", "builders", "necroplis"),
    ),
    (
        ARCH_ENDURE,
        ("endure", "endures", "remain", "remains", "standing", "stands", "stood", "survive",
         "survived", "lasting", "lasts", "signature", "legacy", "boldest", "still", "persist"),
    ),
    (
        ARCH_HISTORY,
        ("millennia", "millennium", "century", "centuries", "era", "dynasty", "kingdom",
         "ancient", "origins", "origin", "history", "empire", "empires", "reign", "bce", "ce ",
         "bc ", "ad ", "chronolog", "antiquity"),
    ),
    (
        ARCH_CATALOG,
        ("catalog", "cataloged", "catalogue", "list", "wonders", "wonder", "original seven",
         "collection", "inventory", "seven"),
    ),
    (
        ARCH_JOURNEY,
        ("traveler", "travelers", "traveller", "journey", "route", "across", "expedition",
         "migrat", "voyage", "mediterranean", "connected", "connection", "via", "harbor",
         "harbour", "basin", "trade"),
    ),
    (
        ARCH_GEO,
        ("stone", "stones", "mountain", "limestone", "river", "nile", "sea", "desert", "wind",
         "winds", "earth", "rock", "rocks", "plateau", "terrain", "cliff", "sand", "water",
         "coast", "valley", "island", "horizon"),
    ),
    (
        ARCH_QUANTITY,
        ("percent", "percentage", "rate", "ratio", "measure", "measurement", "height", "tall",
         "weight", "distance", "scale", "size", "volume", "metric", "figure"),
    ),
    (
        ARCH_CONCEPT,
        ("question", "understand", "understanding", "idea", "imagination", "geometry", "proof",
         "concept", "completeness", "perfection", "represented", "represent", "meaning",
         "impossible", "theory", "knowledge", "wisdom", "thought", "believe"),
    ),
]

# Verbs that imply a visual direction for the composition.
_FALLING_VERBS = frozenset(
    "destroy fire burn earthquake collapse ruin fell fall war greed storm flood eruption shatter crumble topple sank wreck".split()
)
_RISING_VERBS = frozenset(
    "build construct quarry carve raise erect engineer craft assemble stack hew".split()
)
_STABLE_VERBS = frozenset(
    "endure remain standing stands stood survive last signature legacy persist".split()
)

_STOPWORDS = frozenset(
    """a an and or but the of to in on at by for with from into over under through after before
    as is are was were be been being it its they them their this that these those we you your our
    his her hers their ours not no yes than then so yet nor about above below between during
    because while where when what which who whom whose how why can could would should shall will
    may might must do does did done has have had having just only such very more most some any
    each every other another one two also still even ever never often always along plus per onto
    upon out off up down here there their then than during around near since until against
    without across roughly nearly almost approximately approx""".split()
)

_WORD_NUMERALS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_SCALE_WORDS = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_TENS = {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
# Every accepted part of a spoken numeral or measurement phrase (used by duration/stage filters).
_NUM_WORDS = set(_WORD_NUMERALS) | _TENS | set(_SCALE_WORDS)
_KNOWN_UNITS = {
    "%", "percent", "trillion", "billion", "million", "thousand", "hundred",
    "kilometres", "kilometers", "km", "miles", "metres", "meters", "m",
    "years", "year", "days", "day", "hours", "hour", "minutes", "seconds",
    "times", "degrees", "tonnes", "tons", "kg", "stones", "levels",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+|\d+(?:,\d{3})*(?:\.\d+)?")
_YEAR_RE = re.compile(r"\b\d{1,4}\s*(?:BCE|CE|BC|AD)\b", re.I)
# Capture either digits or a spoken numeral (allowing hyphenated compounds like
# "twenty-seven"); word matches are validated afterwards so "limestone years" cannot qualify.
_DURATION_RE = re.compile(
    r"\b((?:\d+(?:,\d{3})*(?:\.\d+)?)|(?:[a-z]+(?:-[a-z]+)?))\s+"
    r"(millennia|millennium|centuries|century|years?|months?|days?|hours?)\b",
    re.I,
)
_ERA_WORD_RE = re.compile(r"\b(?:millennia|millennium|centuries?|ancient era|antiquity)\b", re.I)
_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass(frozen=True)
class Quantity:
    number_text: str   # verbatim text as spoken ("146.6", "4,500")
    unit: str          # normalized unit ("meters", "million") or ""
    noun: str          # the clause word the quantity measures ("stones", "tall") or ""
    value: float       # parsed numeric value for layout scaling

    @property
    def display_number(self) -> str:
        """Group thousands for whole numbers; keep the clause's decimal precision verbatim."""
        if "." in self.number_text:
            head, _, frac = self.number_text.partition(".")
            return f"{int(head.replace(',', '')):,}.{frac}" if head.replace(",", "").isdigit() else self.number_text
        digits = self.number_text.replace(",", "")
        return f"{int(digits):,}" if digits.isdigit() else self.number_text

    def label(self) -> str:
        return f"{self.display_number} {self.unit}".strip().upper()


@dataclass(frozen=True)
class ClauseSemantics:
    archetype: str
    direction: str  # rising | falling | stable | neutral
    terms: list[str] = field(default_factory=list)
    ordered_terms: list[str] = field(default_factory=list)
    quantities: list[Quantity] = field(default_factory=list)
    time_marks: list[str] = field(default_factory=list)
    relations: list[tuple[str, str, str]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "archetype": self.archetype,
            "direction": self.direction,
            "terms": self.terms,
            "ordered_terms": self.ordered_terms,
            "quantities": [
                {"number_text": q.number_text, "unit": q.unit, "noun": q.noun, "value": q.value}
                for q in self.quantities
            ],
            "time_marks": self.time_marks,
            "relations": [list(r) for r in self.relations],
            "labels": self.labels,
            "rationale": self.rationale,
        }

    def node_count(self) -> int:
        """How many repeated entities the layout should draw (clause-derived, never invented)."""
        for q in self.quantities:
            if q.value and q.value <= 12 and q.unit in ("", "times", "stones", "levels"):
                return int(q.value)
        first_num = _WORD_NUMERALS.get((self.terms[0].lower() if self.terms else ""), 0)
        return max(2, min(7, len(self.terms) or first_num or 4))


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_terms(clause: str, limit: int = 7) -> list[str]:
    """Content terms in clause order, de-duplicated, original casing.

    Adjacent capitalized words merge into a single named entity ("Old Kingdom",
    "Great Pyramid") so diagrams label one concept instead of two colliding fragments.
    """
    words: list[str] = []
    for raw in _TOKEN_RE.findall(clause or ""):
        if raw[0].isdigit():
            continue
        word = raw.strip("'-")
        low = word.lower()
        if len(low) < 3 or low in _STOPWORDS:
            words.append("")  # stopwords break capitalized-phrase runs
        else:
            words.append(word)
    out: list[str] = []
    seen: set[str] = set()
    run: list[str] = []

    def _flush(run_words: list[str]) -> None:
        if len(run_words) >= 2 and all(w[0].isupper() for w in run_words):
            phrase = " ".join(run_words)
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                out.append(phrase)
        else:
            for w in run_words:
                key = w.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(w)

    for word in words:
        if not word:
            _flush(run)
            run = []
            continue
        if run and (not word[0].isupper() or not run[-1][0].isupper()):
            _flush(run)
            run = []
        run.append(word)
        # Cap merged named entities at three words so a long title run cannot produce one
        # unfittable label ("Global Monuments Mediterranean Basin ..." -> separate phrases).
        if len(run) >= 3 and all(w[0].isupper() for w in run):
            _flush(run)
            run = []
    _flush(run)
    return out[:limit]


def _is_numberish(word: str) -> bool:
    """True for bare numeral/measurement words that must not masquerade as process stages."""
    parts = re.split(r"[- ]", word.lower())
    return bool(parts) and all(p in _NUM_WORDS or p in _KNOWN_UNITS for p in parts)


# Celestial bodies that anchor a cosmos diagram's central orbit when the clause names them.
_COSMOS_CORE = {"sun", "earth", "core", "galaxy"}
# Generic adjectives/connectors that make weak orbital labels despite being clause content words.
_WEAK_ANCHORS = {"known", "plus", "only", "other", "another", "first", "many", "such"}


def entity_terms(clause: str, limit: int = 6) -> list[str]:
    """Ranked clause entities with spoken numerals removed (for labeled orbital anchors).

    For cosmos clauses the named star ("Sun", "Earth") is promoted to the core position.
    """
    terms = [
        w for w in ranked_terms(clause, limit=20)
        if not _is_numberish(w) and w.lower() not in _WEAK_ANCHORS
    ]
    sem = analyze_clause(clause)
    if sem.archetype == ARCH_COSMOS:
        core = next((w for w in terms if w.lower() in _COSMOS_CORE), None)
        if core:
            terms = [core] + [w for w in terms if w != core]
    return terms[:limit]


def stage_terms(clause: str, limit: int = 5) -> list[str]:
    """Process-stage labels: clause terms with numerals/durations/hedges filtered out.

    "Constructed over roughly twenty to twenty-seven years during Egypt's Old Kingdom"
    -> ["Constructed", "Egypt's", "Old Kingdom"] — honest stages, never "ROUGHLY"/"TWENTY".
    """
    terms = extract_terms(clause, limit=20)
    filtered = [w for w in terms if not _is_numberish(w)]
    return (filtered or terms)[:limit]


def ranked_terms(clause: str, limit: int = 7) -> list[str]:
    """Entity-first ranking used for hubs/headlines: capitalized words, then longer words."""
    ordered = extract_terms(clause, limit=20)
    ranked = sorted(ordered, key=lambda w: (w[0].islower(), -len(w)))
    return ranked[:limit]


def _parse_quantities(clause: str) -> list[Quantity]:
    out: list[Quantity] = []
    tokens = list(re.finditer(r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[A-Za-z%]+)?", clause or ""))
    for m in tokens:
        chunk = m.group(0)
        num_m = _NUMBER_RE.search(chunk)
        if not num_m:
            continue
        number_text = num_m.group(0)
        tail = chunk[num_m.end():].strip().lower().rstrip("%")
        unit = "%" if "%" in chunk else (tail if tail in _KNOWN_UNITS else "")
        # The chunk may already contain the counted noun ("7 Wonders"); otherwise take the first
        # content word after the match ("146.6 meters tall" -> "tall").
        noun = tail if tail and not unit else ""
        following = _TOKEN_RE.findall(clause[m.end(): m.end() + 40])
        if not noun:
            noun = next((w for w in following if w[0].isalpha() and w.lower() not in _STOPWORDS
                         and w.lower() not in _KNOWN_UNITS), "")
        try:
            value = float(number_text.replace(",", ""))
        except ValueError:
            value = 0.0
        out.append(Quantity(number_text, unit, noun[:24], value))
        if len(out) >= 3:
            break
    return out


def _parse_time_marks(clause: str) -> list[str]:
    marks = [m.group(0) for m in _YEAR_RE.finditer(clause or "")]
    for m in _DURATION_RE.finditer(clause or ""):
        head = m.group(1)
        if head[0].isalpha():
            parts = re.split(r"[- ]", head.lower())
            if not all(p in _NUM_WORDS for p in parts):
                continue  # e.g. "limestone years" is not a duration
        marks.append(m.group(0))
    marks += [m.group(0) for m in _ERA_WORD_RE.finditer(clause or "")]
    seen: set[str] = set()
    ordered: list[str] = []
    for mark in marks:
        key = mark.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(mark)
    return ordered[:5]


def _parse_relations(clause: str, terms: list[str]) -> list[tuple[str, str, str]]:
    low = (clause or "").lower()
    out: list[tuple[str, str, str]] = []
    for rel in ("via", "through", "connected", "from", "to"):
        idx = low.find(f" {rel} ")
        if idx < 0 or len(terms) < 2:
            continue
        before = _TOKEN_RE.findall(clause[:idx])
        after = _TOKEN_RE.findall(clause[idx:])
        a = next((w for w in reversed(before) if w in terms), terms[0])
        b = next((w for w in after[1:] if w in terms), terms[-1])
        if a != b:
            out.append((a, rel.upper(), b))
        if len(out) >= 2:
            break
    return out


_TIME_UNITS = {"years", "year", "months", "month", "days", "day", "hours", "hour",
               "minutes", "seconds", "millennia", "millennium", "centuries", "century"}
_VERB_ARCHETYPES = {ARCH_BUILD, ARCH_DESTRUCTION, ARCH_ENDURE}


def _detect_archetype(clause: str, quantities: list[Quantity], time_marks: list[str]) -> tuple[str, str]:
    low = f" {(clause or '').lower()} "
    scores: dict[str, int] = {}
    for archetype, keywords in _ARCH_LEXICON:
        hits = sum(1 for kw in keywords if kw in low)
        if hits:
            scores[archetype] = hits
    # A physical measurement (height, distance, count with a non-time unit) is the clause's
    # dominant visual fact; bare duration phrases merely annotate another archetype.
    if quantities:
        physical = any(q.unit and q.unit not in _TIME_UNITS for q in quantities)
        scores[ARCH_QUANTITY] = scores.get(ARCH_QUANTITY, 0) + (2 if physical else 1)
    # Time marks promote a history timeline only when no action verb already owns the clause;
    # "constructed over twenty years" is a building story annotated with duration, not a timeline.
    verb_owned = any(scores.get(a, 0) > 0 for a in _VERB_ARCHETYPES)
    if time_marks and not verb_owned:
        scores[ARCH_HISTORY] = scores.get(ARCH_HISTORY, 0) + 1
    if not scores:
        return ARCH_CONCEPT, "neutral"
    # Deterministic tie-break by lexicon order (most specific archetypes are listed first).
    order = [a for a, _kw in _ARCH_LEXICON]
    archetype = sorted(scores, key=lambda a: (-scores[a], order.index(a) if a in order else 99))[0]

    words = set(re.findall(r"[a-z]+", low))

    def _hits(verb_set: frozenset) -> bool:
        # Prefix match so inflected forms ("constructed", "endures", "destroyed") resolve to
        # their base verb; the 4-character minimum prevents short-root false positives.
        return any(len(v) >= 4 and any(w.startswith(v) for w in words) for v in verb_set)

    if _hits(_FALLING_VERBS):
        direction = "falling"
    elif _hits(_RISING_VERBS):
        direction = "rising"
    elif _hits(_STABLE_VERBS):
        direction = "stable"
    else:
        direction = "neutral"
    return archetype, direction


def _build_labels(
    terms: list[str],
    quantities: list[Quantity],
    time_marks: list[str],
) -> list[str]:
    """The exact strings an engine may put on screen; every one is clause-verbatim."""
    labels: list[str] = []
    for q in quantities:
        labels.append(q.label())
        if q.noun:
            labels.append(q.noun.upper())
    labels.extend(m.upper() for m in time_marks)
    labels.extend(t.upper() for t in terms)
    # De-duplicate preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for label in labels:
        if label and label not in seen:
            seen.add(label)
            unique.append(label)
    return unique[:9]


def analyze_clause(clause: str) -> ClauseSemantics:
    """Parse one spoken clause into its meaning payload (deterministic, no I/O)."""
    text = (clause or "").strip()
    ordered_terms = extract_terms(text)
    terms = ranked_terms(text)
    quantities = _parse_quantities(text)
    time_marks = _parse_time_marks(text)
    archetype, direction = _detect_archetype(text, quantities, time_marks)
    relations = _parse_relations(text, ordered_terms)
    labels = _build_labels(terms, quantities, time_marks)
    rationale = (
        f"archetype={archetype} direction={direction} "
        f"terms={len(terms)} quantities={len(quantities)} time_marks={len(time_marks)}"
    )
    return ClauseSemantics(
        archetype=archetype,
        direction=direction,
        terms=terms,
        ordered_terms=ordered_terms,
        quantities=quantities,
        time_marks=time_marks,
        relations=relations,
        labels=labels,
        rationale=rationale,
    )


# Preferred diagram template for each archetype; the router applies its no-repeat budget on top.
ARCHETYPE_TEMPLATE = {
    ARCH_COSMOS: "particle_cluster",
    ARCH_DESTRUCTION: "terrain_tectonic",
    ARCH_GEO: "terrain_tectonic",
    ARCH_BUILD: "process_flow",
    ARCH_JOURNEY: "process_flow",
    ARCH_HISTORY: "timeline_arc",
    ARCH_ENDURE: "timeline_arc",
    ARCH_QUANTITY: "stat_dashboard",
    ARCH_CATALOG: "node_network",
    ARCH_CONCEPT: "node_network",
}


def labels_are_clause_derived(clause: str, labels: list[str]) -> bool:
    """Stage E honesty check: every rendered label must tokenize back to clause words."""
    clause_tokens = set(_norm(clause).split())
    for label in labels:
        for token in _norm(label).split():
            if token and not token[0].isdigit() and token not in clause_tokens:
                return False
    return True
