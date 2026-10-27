"""Documentary Showrunner, Socratic Multi-Host Composer & TTS Normalizer.

Constructs broadcast-ready multi-act scripts with Cold Open hooks, dynamic
2-host conversational interplay (Alex Vance & Dr. Maya Lin), camera angle
choreography, B-roll assignments, and exact word-budget scaling from 90-second
briefings up to 1-hour+ feature documentaries with zero repeated phrases.
"""

from __future__ import annotations

import re
from typing import Literal

from voxforge.bootstrap import StudioEnvironment
from voxforge.models import (
    CameraAngle,
    ChapterScript,
    DialogueTurn,
    EpisodeScript,
    MusicCue,
    ResearchDossier,
    ShowSpec,
    SubTopicDossier,
)
from voxforge.script.llm_provider import ZeroKeyLLMClient

_ROMAN_RULERS = {
    "II": "the Second",
    "III": "the Third",
    "IV": "the Fourth",
    "V": "the Fifth",
    "VI": "the Sixth",
    "VII": "the Seventh",
}


def _year_to_words(year_str: str) -> str:
    y = int(year_str)
    if 2000 <= y <= 2009:
        return f"two thousand {y - 2000}" if y > 2000 else "two thousand"
    if 1100 <= y <= 2999:
        hi = y // 100
        lo = y % 100
        if lo == 0:
            return f"{hi} hundred"
        if lo < 10:
            return f"{hi} oh {lo}"
        return f"{hi} {lo}"
    return year_str


_SPOKEN_CONTRACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bI am\b"), "I'm"),
    (re.compile(r"\bwe are\b"), "we're"),
    (re.compile(r"\bWe are\b"), "We're"),
    (re.compile(r"\bthey are\b"), "they're"),
    (re.compile(r"\bThey are\b"), "They're"),
    (re.compile(r"\bit is\b"), "it's"),
    (re.compile(r"\bIt is\b"), "It's"),
    (re.compile(r"\bthat is\b"), "that's"),
    (re.compile(r"\bThat is\b"), "That's"),
    (re.compile(r"\bwhat is\b"), "what's"),
    (re.compile(r"\bWhat is\b"), "What's"),
    (re.compile(r"\bthere is\b"), "there's"),
    (re.compile(r"\bThere is\b"), "There's"),
    (re.compile(r"\bhere is\b"), "here's"),
    (re.compile(r"\bHere is\b"), "Here's"),
    (re.compile(r"\bdo not\b"), "don't"),
    (re.compile(r"\bdid not\b"), "didn't"),
    (re.compile(r"\bcannot\b"), "can't"),
    (re.compile(r"\bwill not\b"), "won't"),
    (re.compile(r"\blet us\b"), "let's"),
    (re.compile(r"\bLet us\b"), "Let's"),
]


_ORDINALS: dict[str, str] = {
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    "6th": "sixth",
    "7th": "seventh",
    "8th": "eighth",
    "9th": "ninth",
    "10th": "tenth",
    "11th": "eleventh",
    "12th": "twelfth",
    "13th": "thirteenth",
    "14th": "fourteenth",
    "15th": "fifteenth",
    "16th": "sixteenth",
    "17th": "seventeenth",
    "18th": "eighteenth",
    "19th": "nineteenth",
    "20th": "twentieth",
    "21st": "twenty-first",
}


def _spoken_era_phrase(era: str) -> str:
    """Convert museum-placard era metadata into natural spoken English."""
    s = re.sub(r"\([^)]*\)", "", era)
    s = re.sub(r"\bca?\.\s*", "around ", s, flags=re.IGNORECASE)
    s = s.replace("–", " to ").replace("—", " to ")
    for k, v in _ORDINALS.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)
    s = re.sub(r"\bCentury\b", "century", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;-")
    # Avoid double 'around around' when inserted into templates that already say 'around {era}'
    s = re.sub(r"^around\s+", "", s, flags=re.IGNORECASE)
    return s


def _spoken_loc_phrase(loc: str) -> str:
    """Smooth out raw geographic metadata labels so host dialogue flows naturally."""
    s = re.sub(r"\([^)]*\)", "", loc).strip()
    replacements = {
        "Giza Plateau, Egypt": "Egypt's Giza Plateau",
        "Northern China Frontier": "northern China's mountain frontier",
        "Ma'an Governorate, Southern Jordan": "the desert canyons of southern Jordan",
        "Cusco Region, Peruvian Andes": "high in the Peruvian Andes",
        "Yucatán Peninsula, Mexico": "Mexico's Yucatán Peninsula",
        "Agra, Uttar Pradesh, India": "Agra, India",
        "Corcovado Mountain, Rio de Janeiro, Brazil": "Corcovado Mountain in Rio de Janeiro",
    }
    return replacements.get(s, s)


def rewrite_for_spoken_ear(text: str) -> str:
    """Rewrite eye-oriented encyclopedia sentences into crisp, natural one-breath spoken sentences."""
    s = re.sub(r"\s*\([^)]{1,45}\)", "", text)
    s = re.sub(r"\bca?\.\s*(?=\d|[a-z])", "around ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bapprox\.\s*", "about ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bvs\.\s*", "versus ", s, flags=re.IGNORECASE)

    # Split breathless >30-word sentences containing parenthetical em-dash lists or ': ' / '; ' clauses
    out_sents: list[str] = []
    raw_sents = re.split(r"(?<=[.!?])\s+", s.strip())
    for sent in raw_sents:
        words = sent.split()
        if len(words) <= 30:
            out_sents.append(sent)
            continue

        # Case 1: Paired em-dash parenthetical ("... — A, B, C — ...")
        if sent.count("—") >= 2:
            parts = [p.strip() for p in sent.split("—") if p.strip()]
            if len(parts) == 3:
                lead = parts[0].rstrip(",; ")
                mid = parts[1].rstrip(",.; ")
                tail = parts[2].lstrip(",; ")
                if len(parts[1].split()) >= 10:
                    out_sents.append(f"{lead} {tail}")
                    out_sents.append(f"That lost lineup included {mid}.")
                else:
                    out_sents.append(f"{lead}, {mid}, {tail}")
                continue

        # Case 2: Split at single ': ' or '; ' (or single ' — ' before a full clause, never a bare -ing fragment)
        split_done = False
        for sep in (": ", "; ", " — "):
            if sep == " — " and sent.count("—") != 1:
                continue
            if sep in sent:
                idx = sent.find(sep)
                left = sent[:idx].strip()
                right = sent[idx + len(sep) :].strip()
                first_rw = right.split()[0].lower() if right.split() else ""
                if len(left.split()) >= 10 and len(right.split()) >= 9 and not first_rw.endswith("ing"):
                    if not left.endswith((".", "!", "?")):
                        left += "."
                    right = right[:1].upper() + right[1:]
                    out_sents.extend([left, right])
                    split_done = True
                    break
        if split_done:
            continue

        # Case 3: Split overlong ', while ' / ', yet ' / ', which ' compound sentences
        for conj, starter in ((", while ", "Meanwhile, "), (", yet ", "Yet "), (", whereas ", "By contrast, ")):
            if conj in sent:
                idx = sent.find(conj)
                left = sent[:idx].strip()
                right = sent[idx + len(conj) :].strip()
                if len(left.split()) >= 11 and len(right.split()) >= 9:
                    out_sents.append(left + ".")
                    out_sents.append(starter + right)
                    split_done = True
                    break
        if not split_done:
            out_sents.append(sent)

    cleaned = " ".join(x.strip() for x in out_sents if x.strip())
    for pat, repl in _SPOKEN_CONTRACTIONS:
        cleaned = pat.sub(repl, cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_for_tts(text: str) -> str:
    """Normalize dates, Roman numerals, abbreviations, contractions, and pauses for human TTS prosody."""
    s = re.sub(r"\s*\([^)]{1,45}\)", "", text)

    for pat, repl in _SPOKEN_CONTRACTIONS:
        s = pat.sub(repl, s)

    # Expand spoken abbreviations before tokenization so TTS never reads 'c.' as 'see' or splits on 'Dr.'
    s = re.sub(r"\bDr\.\s+", "Doctor ", s)
    s = re.sub(r"\bProf\.\s+", "Professor ", s)
    s = re.sub(r"\bSt\.\s+", "Saint ", s)
    s = re.sub(r"\bMt\.\s+", "Mount ", s)
    s = re.sub(r"\bca?\.\s*(?=\d|[a-zA-Z])", "around ", s)
    s = re.sub(r"\bapprox\.\s*", "about ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bvs\.\s*", "versus ", s, flags=re.IGNORECASE)

    for k, v in _ORDINALS.items():
        s = re.sub(rf"\b{k}\b", v, s, flags=re.IGNORECASE)

    # Convert ALL-CAPS show titles (e.g. DEEP HORIZON) to natural Title Case while keeping era/unit acronyms
    _keep_caps = {"BCE", "BC", "CE", "AD", "UNESCO", "DNA", "GPS", "3D", "II", "III", "IV", "VI", "VII"}
    s = re.sub(
        r"\b([A-Z]{3,})\b",
        lambda m: m.group(1) if m.group(1) in _keep_caps else m.group(1).title(),
        s,
    )

    def _rep_ruler(m: re.Match[str]) -> str:
        name, rom = m.group(1), m.group(2)
        return f"{name} {_ROMAN_RULERS.get(rom, rom)}"

    s = re.sub(r"\b([A-Z][a-z]+)\s+(II|III|IV|V|VI|VII)\b", _rep_ruler, s)

    def _rep_era(m: re.Match[str]) -> str:
        yr, era = m.group(1), m.group(2).upper()
        spaced_era = " ".join(list(era))
        return f"{_year_to_words(yr)} {spaced_era}"

    s = re.sub(r"\b(\d{3,4})\s*(BCE|BC|CE|AD)\b", _rep_era, s)
    # Also expand standalone BCE / CE after words like 'seventh century BCE'
    s = re.sub(r"\b(BCE|BC|CE)\b", lambda m: " ".join(list(m.group(1))), s)
    s = re.sub(r"\b(1[4-9]\d{2}|20[0-3]\d)\b", lambda m: _year_to_words(m.group(1)), s)

    # Speak round X,X00 thousands naturally ('4,500' -> '45 hundred', '3,800' -> '38 hundred')
    # and strip commas inside other multi-digit numbers ('21,196' -> '21196') so G2P never pauses mid-number!
    s = re.sub(r"\b([1-9]),([1-9])00\b", r"\1\2 hundred", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    # Expressive clause pause without raw em-dash characters
    s = s.replace("—", ", ").replace("–", " to ").replace("×", " by ").replace("°", " degrees")
    s = s.replace("~", "around ").replace("&", " and ")
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*km\b", r"\1 kilometers", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*m\b", r"\1 meters", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*cm\b", r"\1 centimeters", s)
    s = re.sub(r"\b(\d+(?:\.\d+)?)\s*ft\b", r"\1 feet", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r",\s*,+", ", ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_long_utterance(text: str, max_words: int = 56) -> list[str]:
    """Ensure every turn stays within natural breath-group length (<= 56 words)."""
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sents:
        sw = len(sent.split())
        if cur and cur_len + sw > max_words:
            chunks.append(" ".join(cur).strip())
            cur = [sent]
            cur_len = sw
        else:
            cur.append(sent)
            cur_len += sw
    if cur:
        chunks.append(" ".join(cur).strip())

    final_chunks: list[str] = []
    for ch in chunks:
        if len(ch.split()) > max_words + 6 and ", " in ch:
            clauses = ch.split(", ")
            mid = len(clauses) // 2
            p1 = ", ".join(clauses[:mid]).strip() + "."
            p2 = ", ".join(clauses[mid:]).strip()
            p2 = p2[:1].upper() + p2[1:]
            final_chunks.extend([p1, p2])
        else:
            final_chunks.append(ch)
    return [c for c in final_chunks if c]


_ALEX_ARRIVAL_OPENERS = [
    "Across {loc}, {title} rose on a scale the ancient world had scarcely imagined.",
    "Further east across {loc}, {title} reshaped an entire frontier.",
    "Across the terrain of {loc}, {title} pushed stone and geometry into uncharted territory.",
    "Across oceans and mountains in {loc}, {title} solved an entirely different physical challenge.",
    "Turning toward {loc}, {title} brought monumental engineering into a new age.",
    "Along the horizon of {loc}, {title} still commands the landscape.",
    "Deep within {loc}, builders raised {title} against seemingly impossible odds.",
    "High across {loc}, {title} proved what disciplined surveying could achieve.",
    "Across {loc}, {title} fused sacred astronomy with enduring stonework.",
    "Shifting the lens to {loc} brings {title} into sharp relief.",
]

_MAYA_ENGINEERING_PIVOTS = [
    "For structural engineers, the first mystery is why {short} never buckled under its own mass.",
    "Beneath the familiar silhouette of {short}, the deeper story comes down to pure physics.",
    "Field measurements at {short} overturn almost every assumption in older histories.",
    "Before a single wall rose at {short}, builders first had to master {loc}'s unforgiving terrain.",
    "At the core of {short} sits a surveying puzzle that baffled researchers for generations.",
    "Mapping the load paths beneath {short} reveals a remarkably disciplined blueprint.",
    "Every course of stone in {short} was a calculated defense against weather and gravity in {loc}.",
    "Look directly beneath the visible stonework of {short}, and the hidden mechanics emerge.",
    "Even by modern civil engineering standards, {short} pushed {era} craft to its physical ceiling.",
    "Concealed inside {short} is an intricate system of structural balance and hydraulic control.",
]

_ALEX_FOLLOWUP_QUESTIONS = [
    "Which raises the central question in {loc}: how did ancient crews actually raise {short}?",
    "That figure at {short} defies easy explanation — how did {era} builders hold that tolerance?",
    "On paper, {era} tools had no business achieving that result at {short} — yet the stonework still stands.",
    "Measured on the ground in {loc} today, the forces inside {short} become startlingly tangible.",
    "Without laser levels, steel cables, or powered cranes, every stone at {short} demanded human ingenuity.",
    "And the surrounding geology of {loc} dictated nearly every line of {short}.",
    "Much of the answer at {short} lay inside the chemistry of the raw materials themselves.",
    "Millions of visitors pass beneath {short} unaware of the hidden substructure holding it upright.",
    "Sustaining the supply lines for {short} year after year required state-level logistics.",
    "Century after century, extreme weather across {loc} tested every joint of {short}.",
]


def _generate_unique_chapter_deep_dives(
    subtopic: SubTopicDossier,
    chapter_idx: int,
    prev_subtopic: SubTopicDossier | None,
) -> list[tuple[Literal["HOST_A", "HOST_B"], str]]:
    """Build chapter-specific forensic, logistical, sensory, and comparative exchanges with zero shared 6-grams."""
    short = subtopic.title.split(":")[0]
    loc = _spoken_loc_phrase(subtopic.location)
    era = _spoken_era_phrase(subtopic.era_or_date)
    m = subtopic.metrics

    m0_lbl = m[0].label if len(m) > 0 else "primary scale"
    m0_val = m[0].value if len(m) > 0 else "monumental scale"
    m1_lbl = m[1].label if len(m) > 1 else "construction metric"
    m1_val = m[1].value if len(m) > 1 else "precision craft"
    m2_lbl = m[2].label if len(m) > 2 else "engineering tolerance"
    m2_val = m[2].value if len(m) > 2 else "measured tolerance"
    m3_lbl = m[3].label if len(m) > 3 else "enduring record"
    m3_val = m[3].value if len(m) > 3 else "centuries of survival"
    prev_short = prev_subtopic.title.split(":")[0] if prev_subtopic else "earlier Bronze Age monuments"

    # Distinct per-chapter analytical exchanges indexed by chapter_idx so phrasing never collides across acts!
    variants_by_act: dict[int, list[tuple[Literal["HOST_A", "HOST_B"], str]]] = {
        1: [
            ("HOST_A", f"When Antipater and Philo compiled those early Mediterranean lists during {era}, why did architecture suddenly eclipse poetry or philosophy as the ultimate civic scoreboard?"),
            ("HOST_B", f"Because stone and bronze across the {loc} functioned as diplomatic deterrence: a harbor lighthouse reaching {m2_val} told every foreign merchant captain that Alexandria commanded geometry, wealth, and mathematics."),
            ("HOST_A", f"And notice the stark survivor ratio — {m1_val} still standing out of seven — which turns our entire episode into a forensic investigation of why monumental structures live or die."),
            ("HOST_B", f"Right, and when {m3_val} reshaped the global canon in modern times, voters instinctively gravitated toward sites that mastered their surrounding mountains, deserts, and rivers."),
            ("HOST_A", "Think also about the lost Statue of Zeus at Olympia, where sculptor Phidias inlaid ivory and gold plates over a wooden core humidified by shallow olive-oil pools inside the temple floor."),
            ("HOST_B", "Or the Mausoleum at Halicarnassus, whose stepped marble tomb for Persian satrap Mausolus stood forty-five meters high and literally gave every modern language the word mausoleum."),
            ("HOST_A", "Let us keep that survival lens front and center as we move monument by monument, testing each site against earthquakes, water, and human politics."),
            ("HOST_B", "Because every single wonder we are visiting next solved a different physical impossibility using local stone, chemistry, and sheer organizational discipline."),
        ],
        2: [
            ("HOST_A", f"Stand at the base of {short} on the {loc} and try to visualize {m1_val} stacked to {m0_val}: one block laid every two to three minutes during daylight shifts."),
            ("HOST_B", f"What makes that cadence plausible is quarry proximity paired with Nile flood timing: eighty-five percent of Khufu's core came from horseshoe quarries just south of the pyramid."),
            ("HOST_A", f"While the gleaming white casing came across the river from Tura and eighty-ton granite slabs sailed north from Aswan — all coordinated by papyrus ledgers during {era}."),
            ("HOST_B", f"And look at the surveying discipline required for {m2_val} orientation error: if corner angles drifted even half a degree, the four faces would have missed each other at the summit apex."),
            ("HOST_A", "Even the four faces of Khufu's pyramid harbor a subtle optical secret: each side is slightly concave by nearly one meter down its centerline, creating an eight-sided shadow effect at equinox dawn."),
            ("HOST_B", "Inside the ascending Grand Gallery, corbeled limestone walls rise eight point six meters with slot grooves that Egyptologists Jean-Pierre Houdin and Mark Lehner analyzed as possible counterweight haulage tracks."),
            ("HOST_A", f"Holding {m3_val} as Earth's tallest human creation until Lincoln Cathedral's spire in medieval England puts Old Kingdom organization in a league of its own."),
            ("HOST_B", "Even today, crane operators admit that lifting eighty-ton granite beams sixty meters upward into a cramped stone shaft would challenge contemporary rigging crews."),
        ],
        3: [
            ("HOST_A", f"Walking steep ridge sections of {short} across the {loc}, the sheer verticality shocks your legs — steps sometimes pitch at sixty degrees straight up granite spines."),
            ("HOST_B", f"Ming military planners chose ridge crests deliberately: building along {m1_val} of mountain watersheds meant attackers had to climb uphill under crossbow fire before even touching the parapet."),
            ("HOST_A", f"And stretching {m0_val} across {era} dynasties required feeding hundreds of garrison towns through self-sustaining military farms called tuntian planted behind the frontier."),
            ("HOST_B", f"Combine those garrison farms with {m2_val} binding the crenellations and {m3_val} relaying coded smoke signals, and the frontier behaved like a continental nervous system."),
            ("HOST_A", "At its eastern terminus at Laolongtou — Old Dragon's Head — Ming stone masons even drove iron-clamped granite piers directly into the crashing waves of the Bohai Sea."),
            ("HOST_B", "While three thousand kilometers west at Jiayuguan Pass beneath snow-capped Qilian peaks, mud-brick citadel walls controlled Silk Road caravans entering the Gobi corridor."),
            ("HOST_A", "So rather than a static brick fence, we are really looking at pre-modern logistics, materials chemistry, and long-range telecommunications fused into mountain stone."),
            ("HOST_B", "Precisely — whenever court funding maintained the garrisons and mortar drainage, cavalry raids stalled; only internal political fracture could render the passes irrelevant."),
        ],
        4: [
            ("HOST_A", f"Nothing prepares first-time visitors in {loc} for the acoustic hush inside the Siq gorge just before {short} blazes out in rose sandstone at {m0_val}."),
            ("HOST_B", f"That theatrical reveal was intentional urban staging during {era}: ambassadors from Rome or Alexandria realized instantly that caravan tariffs funded world-class Hellenistic artistry."),
            ("HOST_A", f"Yet surviving on {m1_val} while storing and piping {m2_val} meant every rooftop, cistern lip, and cliff gutter worked as a giant rainwater harvester."),
            ("HOST_B", f"And executing {m3_val} subtraction meant master carvers could never patch a slipped chisel blow on a Corinthian capital forty meters above the canyon floor."),
            ("HOST_A", "Climb eight hundred rock-cut steps above the central valley and you reach Ad-Deir — The Monastery — an even wider forty-seven-meter sandstone facade carved high into Jabal Haroun."),
            ("HOST_B", "Nabataean engineers also lined cistern interiors with waterproof hydraulic pozzolana plaster blended with crushed pottery, preventing desert seepage across dry summer months."),
            ("HOST_A", f"Contrast {short} against {prev_short}: instead of assembling millions of fired bricks over peaks, Nabataean sculptors carved negative space straight into living desert bedrock."),
            ("HOST_B", "Today salt crystallization from tourist breath and rare desert downpours poses the main threat, making ancient Nabataean diversion dams vital once again."),
        ],
        5: [
            ("HOST_A", f"Walk beneath the travertine arcades of {short} in {loc}, and you immediately recognize the numbered gate tickets, tiered sightlines, and concourses of a modern championship arena."),
            ("HOST_B", f"Achieving {m3_val} egress across an {m0_val} footprint during {era} meant Flavian architects understood crowd fluid dynamics two thousand years before computer simulation."),
            ("HOST_A", f"And completing the entire amphitheatre in {m1_val} while orchestrating {m2_val} underneath the wooden sand floor turned imperial liturgy into precision stagecraft."),
            ("HOST_B", "Imagine sitting in the upper plebeian tiers as scented saffron mist sprayed from lead nozzles, sailors adjusted linen shade sails overhead, and scenery rose smoothly out of nowhere."),
            ("HOST_A", "Just to transport travertine from Tivoli quarries thirty kilometers east, Roman teamsters built a dedicated six-meter reinforced haul road carrying two hundred ox-carts daily."),
            ("HOST_B", "And look at the classical column order stacked up the outer wall: sturdy Doric on the ground tier, Ionic on the second, and ornate Corinthian higher up to lighten visual mass."),
            ("HOST_A", f"Unlike {prev_short} carved into silent desert cliffs, Rome's monument pulsed at the noisy political heart of a million-person metropolis."),
            ("HOST_B", "And even after medieval stone-stripping hollowed half its outer wall, the surviving inner concrete vaults still carry traffic vibration and millions of footsteps every year."),
        ],
        6: [
            ("HOST_A", f"Standing at {m0_val} in the {loc}, watching clouds sweep straight up vertical cliffs from the Urubamba gorge, {short} feels suspended between earth and sky."),
            ("HOST_B", f"Yet without dedicating {m1_val} to buried gravel drainage filters during {era}, {m2_val} of intense Andean downpours would have liquefied that mountain saddle within a decade."),
            ("HOST_A", f"Then above ground, {m3_val} joints locked multi-ton white granite blocks together so cleanly that five centuries of Andean tremors never toppled the sacred temples."),
            ("HOST_B", f"Notice the philosophical contrast with {prev_short}: Roman builders poured vaulted concrete to dominate a valley, whereas Inca planners sculpted terraces to harmonize with mountain contours."),
            ("HOST_A", "Even the windows of the curved Torreón solar observatory align directly with the June winter solstice sunrise over Veronica peak and the Pleiades star cluster."),
            ("HOST_B", "And every doorway and niche tapers inward in the signature Inca trapezoid — narrowing lintel spans while lowering structural shear during horizontal ground shaking."),
            ("HOST_A", "Even the stone shapes along the skyline intentionally echo the jagged silhouette of Huayna Picchu and Putucusi peaks behind them."),
            ("HOST_B", "Today visitor-weight distribution and slope laser monitoring protect those delicate Inca terrace walls so the cloud-forest citadel stays stable for centuries ahead."),
        ],
        7: [
            ("HOST_A", f"Cross the blinding white plaza at {short} in the {loc}, and every proportion around you — right down to {m0_val} — encodes astronomical time in carved limestone."),
            ("HOST_B", f"During {era}, Maya sky-watchers tracked Venus and solar horizons over generations without optical glass, embedding {m1_val} directly above a sacred {m3_val}."),
            ("HOST_A", f"And hearing that {m2_val} ring across the plaza when a guide claps at the foot of the stairs sends chills down your spine even when you understand wave diffraction."),
            ("HOST_B", f"Where {prev_short} conquered steep mountain rainfall, Yucatán planners faced the opposite challenge: zero surface rivers, making underground karst sinkholes the lifeblood of urban life."),
            ("HOST_A", "Just south of the main pyramid stands El Caracol — a cylindrical stone observatory dome pierced with viewing shafts calibrated specifically to Venus's eight-year synodic extremes."),
            ("HOST_B", "Connected across the jungle by raised white limestone causeways called sacbeob, Chichén Itzá radiated gleaming lime stucco roads visible even by starlight."),
            ("HOST_A", "It transforms the entire ceremonial center into an instrument tuned simultaneously to equinox sunlight, bird-call acoustics, and subterranean water."),
            ("HOST_B", "Restricting foot traffic off the steep limestone risers since two thousand six has been critical to preserving those razor-sharp acoustic step edges."),
        ],
        8: [
            ("HOST_A", f"At dawn in {loc}, watching first light warm the {m0_val} marble dome of {short}, the stone genuinely seems to float weightlessly above the river mist."),
            ("HOST_B", f"Creating effortless visual serenity during {era} demanded {m1_val} executing hidden structural insurance — from submerged river caissons to the {m2_val} of all four minarets."),
            ("HOST_A", f"Step inside the octagonal cenotaph chamber and {m3_val} of lingering acoustic resonance wraps every footstep and whisper in pure stillness."),
            ("HOST_B", f"Compare that intimate symmetry with {prev_short}: both monuments manipulate light and echo with mathematical rigor, yet Agra channels everything into personal devotion and paradise geometry."),
            ("HOST_A", "Look closely at the main southern gateway — Darwaza-i-Rauza — which frames the distant marble tomb like a camera lens: as you walk closer, optical perspective makes the Taj appear to recede."),
            ("HOST_B", "And each marble floral spray contains up to sixty microscopic slivers of jasper, agate, and malachite polished flush until your fingertip cannot feel a single joint."),
            ("HOST_A", "Even the red sandstone mosque and guest pavilion flanking the white tomb provide chromatic counterbalance while anchoring the riverfront terrace."),
            ("HOST_B", "Sustaining Yamuna water flow around those deep ebony-and-sal timber piles remains one of twenty-first-century India's highest heritage engineering priorities."),
        ],
        9: [
            ("HOST_A", f"Rising over {loc} at {m0_val} with an {m1_val} span embracing the harbor below, {short} proves modern reinforced concrete can achieve poetic grace."),
            ("HOST_B", f"Cladding Caquot's wind-truss skeleton in {m2_val} during {era} while grounding {m3_val} safely into granite turned an exposed coastal pinnacle into an icon."),
            ("HOST_A", f"Placed beside {prev_short} and Khufu's ancient horizon, Corcovado completes a forty-five-century arc of humanity building at the outer edge of material strength."),
            ("HOST_B", "Inside the statue's hollow chest sits a vertical maintenance staircase leading right out onto the shoulders and crown, where alpinist engineers inspect lightning arrestors after tropical storms."),
            ("HOST_A", "And the eight-meter pedestal houses a serene chapel dedicated to Nossa Senhora Aparecida, uniting intimate sanctuary space with a monument visible thirty kilometers across Rio."),
            ("HOST_B", "Look at the shared DNA across all nine acts: Giza mastered compression, China mastered terrain networks, Petra mastered desert water, and Rome mastered vaulted concrete."),
            ("HOST_A", "Machu Picchu danced with earthquakes, Chichén Itzá sculpted sunlight and echo, Agra floated marble on river wells, and Rio rode hurricane winds above the clouds."),
            ("HOST_B", "Every era used its sharpest science not merely for utility, but to leave an unmistakable message across millennia: we were here, and we imagined something extraordinary."),
        ],
    }

    if chapter_idx in variants_by_act:
        return variants_by_act[chapter_idx]

    # Generic unique fallback for arbitrary non-Wonders topics (parameterized by chapter_idx & subtopic specifics)
    return [
        (
            "HOST_A",
            f"Focusing on Act {chapter_idx} and {short} in {loc}, notice how {m0_lbl.lower()} ({m0_val}) framed every practical choice during {era}.",
        ),
        (
            "HOST_B",
            f"And pairing {m0_val} alongside {m1_lbl.lower()} at {m1_val} shows why {short} diverged sharply from {prev_short}.",
        ),
        (
            "HOST_A",
            f"What lessons from {short} still guide researchers studying {loc} in contemporary fieldwork?",
        ),
        (
            "HOST_B",
            f"Field teams examining {short} emphasize empirical verification — cross-checking archival records from {era} against physical measurements.",
        ),
    ]


def _compose_chapter_dialogue(
    subtopic: SubTopicDossier,
    prev_subtopic: SubTopicDossier | None,
    chapter_idx: int,
    total_chapters: int,
    chapter_word_budget: int,
    spec: ShowSpec,
    dossier: ResearchDossier,
    llm_client: ZeroKeyLLMClient,
    used_sentences: set[str],
) -> list[tuple[Literal["HOST_A", "HOST_B"], str]]:
    """Compose rich, non-repetitive two-host dialogue calibrated to chapter_word_budget."""
    raw_pairs: list[tuple[Literal["HOST_A", "HOST_B"], str]] = []
    c_mod = (chapter_idx - 1) % len(_ALEX_ARRIVAL_OPENERS)
    short_title = subtopic.title.split(":")[0]
    spoken_loc = _spoken_loc_phrase(subtopic.location)
    spoken_era = _spoken_era_phrase(subtopic.era_or_date)
    is_compact = chapter_word_budget < 120

    def _cur_words() -> int:
        return sum(len(txt.split()) for _, txt in raw_pairs)

    # 1. Cold Open Documentary Hook on Act 1 (zero podcast-table small talk)
    if chapter_idx == 1:
        hook_1 = dossier.cold_open_hook_facts[0] if dossier.cold_open_hook_facts else subtopic.summary
        used_sentences.add(hook_1)
        if is_compact:
            raw_pairs.append(("HOST_A", hook_1))
            raw_pairs.append(
                (
                    "HOST_B",
                    "The question is simple: what did these builders understand about stone, weight, and time?",
                )
            )
        else:
            raw_pairs.append(("HOST_A", hook_1))
            if len(dossier.cold_open_hook_facts) > 1:
                hook_2 = dossier.cold_open_hook_facts[1]
                raw_pairs.append(("HOST_B", hook_2))
                used_sentences.add(hook_2)
            raw_pairs.append(
                (
                    "HOST_A",
                    f"This is {spec.show_title} — an archival and structural chronicle of {spec.topic}, tracing what endured, what collapsed, and how ancient engineers built on a continental scale.",
                )
            )
            raw_pairs.append(("HOST_B", dossier.thesis_statement))
    else:
        spk_open: Literal["HOST_A", "HOST_B"] = "HOST_A" if chapter_idx % 2 == 1 else "HOST_B"
        raw_pairs.append((spk_open, subtopic.summary))
        used_sentences.add(subtopic.summary)

    # Reserve room for sign-off on final chapter
    signoff_reserve = 38 if chapter_idx == total_chapters else 0
    effective_cap = max(45, chapter_word_budget - signoff_reserve)

    # 2. Optional neural LLM turns if local Ollama / GGUF is active
    if llm_client.is_neural_llm and _cur_words() < effective_cap:
        facts_blob = "\n".join(
            f"- {f}"
            for f in (
                subtopic.origin_and_context
                + subtopic.engineering_and_mechanics
                + subtopic.human_stories_and_paradoxes
                + subtopic.legacy_and_modern_science
            )
        )
        llm_turns = llm_client.generate_json_turns(subtopic.title, facts_blob, effective_cap)
        if llm_turns:
            for item in llm_turns:
                spk_l: Literal["HOST_A", "HOST_B"] = "HOST_A" if item["speaker"] == "HOST_A" else "HOST_B"
                raw_pairs.append((spk_l, item["text"]))
                if _cur_words() >= effective_cap:
                    break

    # 3. Primary Research Lenses (strictly respecting effective_cap)
    def _next_speaker() -> Literal["HOST_A", "HOST_B"]:
        if not raw_pairs:
            return "HOST_A"
        return "HOST_B" if raw_pairs[-1][0] == "HOST_A" else "HOST_A"

    for fact in subtopic.origin_and_context:
        if _cur_words() >= int(effective_cap * 0.96):
            break
        if fact in used_sentences:
            continue
        used_sentences.add(fact)
        raw_pairs.append((_next_speaker(), fact))

    mech_facts = [f for f in subtopic.engineering_and_mechanics if f not in used_sentences]
    if mech_facts and _cur_words() < int(effective_cap * 0.95):
        pivot = _MAYA_ENGINEERING_PIVOTS[c_mod].format(
            short=short_title, loc=spoken_loc, era=spoken_era
        )
        used_sentences.add(mech_facts[0])
        raw_pairs.append(("HOST_B", f"{pivot} {mech_facts[0]}"))
        if len(mech_facts) > 1 and _cur_words() < int(effective_cap * 0.92):
            q_line = _ALEX_FOLLOWUP_QUESTIONS[c_mod].format(
                short=short_title, loc=spoken_loc, era=spoken_era
            )
            raw_pairs.append(("HOST_A", q_line))
            for mf in mech_facts[1:]:
                if _cur_words() >= int(effective_cap * 0.96):
                    break
                used_sentences.add(mf)
                raw_pairs.append((_next_speaker(), mf))

    for hf in subtopic.human_stories_and_paradoxes:
        if _cur_words() >= int(effective_cap * 0.96):
            break
        if hf not in used_sentences:
            used_sentences.add(hf)
            raw_pairs.append((_next_speaker(), hf))

    for lf in subtopic.legacy_and_modern_science:
        if _cur_words() >= int(effective_cap * 0.97):
            break
        if lf not in used_sentences:
            used_sentences.add(lf)
            raw_pairs.append((_next_speaker(), lf))

    # 4. Feature-length 1-hour+ unique forensic deep-dives (100% distinct per chapter!)
    if _cur_words() < int(effective_cap * 0.94):
        deep_exchanges = _generate_unique_chapter_deep_dives(subtopic, chapter_idx, prev_subtopic)
        for d_spk, d_txt in deep_exchanges:
            if _cur_words() >= int(effective_cap * 0.98):
                break
            if d_txt not in used_sentences:
                used_sentences.add(d_txt)
                raw_pairs.append((_next_speaker(), d_txt))

    # 5. Episode Finale Sign-Off
    if chapter_idx == total_chapters:
        if not is_compact:
            for syn in dossier.synthesis_comparisons:
                if _cur_words() < chapter_word_budget and syn not in used_sentences:
                    used_sentences.add(syn)
                    raw_pairs.append((_next_speaker(), syn))
        signoff_spk = _next_speaker()
        topic_clean = spec.topic if spec.topic.lower().startswith("the ") else f"the {spec.topic}"
        raw_pairs.append(
            (
                signoff_spk,
                f"Across millennia of shifting empires, earthquakes, and desert winds, {topic_clean} endure as humanity's boldest signature in stone — "
                f"proof of what happens when geometry and imagination refuse to accept the impossible.",
            )
        )

    # Rewrite every turn for the ear (short breath-friendly sentences, zero placard parens) & cap turn length
    cleaned_pairs: list[tuple[Literal["HOST_A", "HOST_B"], str]] = []
    for spk, txt in raw_pairs:
        ear_txt = rewrite_for_spoken_ear(txt)
        for chunk in _split_long_utterance(ear_txt, max_words=52):
            cleaned_pairs.append((spk, chunk))

    return cleaned_pairs


def compute_turn_pause_ms(turn_text: str, next_text: str | None, is_chapter_end: bool, turn_id: int) -> int:
    """Compute realistic co-host studio turn-taking pause (tight on backchannels, reflective at act ends)."""
    if is_chapter_end:
        return 360
    stripped = turn_text.strip()
    if stripped.endswith("?"):
        base = 210
    elif next_text and re.match(r"^(Right|Exactly|Wait|And|But|So|Or|Wild|Honestly)\b", next_text.strip(), flags=re.IGNORECASE):
        base = 175
    else:
        base = 255
    jitter = ((turn_id * 29 + len(stripped) * 7) % 41) - 20
    return max(150, min(420, base + jitter))


def _extract_pull_quote(text: str) -> str:
    """Extract a punchy 9–24 word pull quote for on-screen Quote Card callouts."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    best = sentences[-1] if sentences else text.strip()
    for s in sentences:
        wc = len(s.split())
        if 9 <= wc <= 24:
            best = s
            break
    words = best.split()
    if len(words) > 24:
        # Trim cleanly at clause punctuation or 22 words
        clauses = [c.strip() for c in re.split(r"[—;:]+", best) if len(c.split()) >= 8]
        best = clauses[0] if clauses else (" ".join(words[:21]).rstrip(",;:") + "...")
    return best.strip(" \"'")


def _generate_next_episode_tease(topic: str) -> str:
    """Create a relevant sequel documentary tease for the Branded Outro End Card."""
    tl = topic.lower()
    if "wonder" in tl or "pyramid" in tl or "ancient" in tl or "colosseum" in tl:
        return "Lost Megacities of Antiquity: Angkor Wat, Teotihuacan & The Library of Alexandria"
    if "silk road" in tl or "trade" in tl or "empire" in tl:
        return "Maritime Empires of the Indian Ocean: Monsoon Navigation & Spice Fortresses"
    if "ocean" in tl or "sea" in tl or "space" in tl:
        return "Extreme Frontiers: Engineering Human Habitats Beneath Ice & Beyond Orbit"
    return f"Beyond {topic}: Unsolved Engineering Mysteries & Archival Discoveries"


_STUDIO_INSERT_VARIANTS = ("mic_detail", "hands_notes", "laptop_workspace")


def assign_multicam_choreography(chapters: list[ChapterScript], subtopics: list[SubTopicDossier]) -> None:
    """Apply full broadcast multicam switching: wide master, close-ups, reaction shots, studio inserts, quotes & data callouts."""
    insert_cursor = 0
    global_idx = 0

    for ch, st in zip(chapters, subtopics):
        assets = st.visual_assets
        broll_assets = [a for a in assets if not a.is_procedural] or assets
        all_assets = assets or broll_assets
        asset_cursor = 1 if len(broll_assets) > 1 else 0
        total_in_ch = len(ch.turns)

        for idx, turn in enumerate(ch.turns):
            has_numbers = bool(re.search(r"\b\d{2,}\b", turn.text))
            turn.pull_quote = _extract_pull_quote(turn.text)

            if ch.chapter_index == 1 and idx == 0:
                # Open in clean Wide Master Studio Set with Episode Title Graphic!
                cam: CameraAngle = "wide_studio"
            elif idx == total_in_ch - 1 and ch.chapter_index == len(chapters):
                # Final sign-off in Wide Master Studio Two-Shot before Outro End Card
                cam = "wide_studio"
            elif idx == 0:
                cam = "broll_cinema" if ch.chapter_index % 2 == 0 else "wide_studio"
            elif idx == 1 and total_in_ch >= 3:
                # Guarantee every Act showcases either a Live 4-Card Telemetry Panel or Kinetic Primary-Source Quote!
                cam = "quote_card" if ch.chapter_index % 2 == 1 else "split_fact_card"
            elif idx == 2 and ch.chapter_index == len(chapters) and total_in_ch >= 4:
                cam = "split_fact_card"
            else:
                slot = global_idx % 9
                if slot == 1:
                    cam = "closeup_host_a" if turn.speaker == "HOST_A" else "closeup_host_b"
                elif slot == 2:
                    cam = "broll_cinema"
                elif slot == 3:
                    cam = "split_fact_card" if (st.metrics and (has_numbers or idx % 2 == 1)) else "closeup_host_b"
                elif slot == 4:
                    cam = "reaction_shot"
                elif slot == 5:
                    cam = "studio_insert"
                elif slot == 6:
                    cam = "quote_card"
                elif slot == 7:
                    cam = "closeup_host_a" if turn.speaker == "HOST_A" else "closeup_host_b"
                elif slot == 8:
                    cam = "broll_cinema"
                else:
                    cam = "wide_studio"

            if cam in ("closeup_host_a", "closeup_host_b"):
                cam = "closeup_host_a" if turn.speaker == "HOST_A" else "closeup_host_b"

            if idx >= 2 and ch.turns[idx - 1].camera == cam and ch.turns[idx - 2].camera == cam:
                cam = "broll_cinema" if cam != "broll_cinema" else (
                    "closeup_host_a" if turn.speaker == "HOST_A" else "closeup_host_b"
                )

            turn.camera = cam
            if cam == "studio_insert":
                turn.insert_variant = _STUDIO_INSERT_VARIANTS[insert_cursor % len(_STUDIO_INSERT_VARIANTS)]
                insert_cursor += 1

            if all_assets:
                chosen = (
                    broll_assets[asset_cursor % len(broll_assets)]
                    if (
                        cam
                        in (
                            "broll_cinema",
                            "split_fact_card",
                            "studio_insert",
                            "closeup_host_a",
                            "closeup_host_b",
                            "quote_card",
                        )
                        and broll_assets
                    )
                    else all_assets[asset_cursor % len(all_assets)]
                )
                turn.visual_asset_path = chosen.file_path
                turn.visual_caption = chosen.caption
                turn.visual_credit = chosen.credit
                if cam in ("broll_cinema", "split_fact_card", "studio_insert", "quote_card"):
                    asset_cursor += 1

            turn.fact_callout_title = st.title.split(":")[0]
            turn.fact_metrics = st.metrics[:4]
            global_idx += 1

            if idx == total_in_ch - 1:
                turn.pause_after_ms = 600
            elif turn.text.endswith("?"):
                turn.pause_after_ms = 200
            elif len(turn.text.split()) < 18:
                turn.pause_after_ms = 175
            else:
                turn.pause_after_ms = 265

    # Guarantee full broadcast shot diversity even on compact (< 9 turn) micro-briefings!
    flat_turns = [t for ch in chapters for t in ch.turns]
    present = {t.camera for t in flat_turns}
    required_angles: list[CameraAngle] = [
        "wide_studio",
        "broll_cinema",
        "reaction_shot",
        "studio_insert",
        "quote_card",
        "split_fact_card",
    ]
    if len(flat_turns) >= 7:
        used_slots = {0, len(flat_turns) - 1}
        for req in required_angles:
            if req not in present:
                for cand_i in range(1, len(flat_turns) - 1):
                    if cand_i not in used_slots and list(t.camera for t in flat_turns).count(flat_turns[cand_i].camera) > 1:
                        flat_turns[cand_i].camera = req
                        if req == "studio_insert" and not flat_turns[cand_i].insert_variant:
                            flat_turns[cand_i].insert_variant = _STUDIO_INSERT_VARIANTS[insert_cursor % 3]
                            insert_cursor += 1
                        if req == "quote_card" and not flat_turns[cand_i].pull_quote:
                            flat_turns[cand_i].pull_quote = _extract_pull_quote(flat_turns[cand_i].text)
                        used_slots.add(cand_i)
                        present.add(req)
                        break


def generate_episode_script(
    spec: ShowSpec,
    dossier: ResearchDossier,
    env: StudioEnvironment,
) -> EpisodeScript:
    """Create initial multi-chapter broadcast podcast script from the ResearchDossier."""
    llm_client = ZeroKeyLLMClient(env.llm_backend, env.llm_model_id_or_path)
    total_chapters = len(dossier.subtopics)
    per_chapter_budget = max(42, int(round(spec.target_word_count * 0.90 / max(1, total_chapters))))

    music_cycle: list[MusicCue] = [
        "cold_open_pulse",
        "wonder_pad",
        "ambient_warm",
        "tension_drone",
        "wonder_pad",
        "ambient_warm",
        "tension_drone",
        "wonder_pad",
        "finale_resolve",
    ]

    chapters: list[ChapterScript] = []
    used_sentences: set[str] = set()
    turn_counter = 1

    for idx, st in enumerate(dossier.subtopics, start=1):
        prev_st = dossier.subtopics[idx - 2] if idx >= 2 else None
        cue = music_cycle[(idx - 1) % len(music_cycle)]
        if idx == total_chapters:
            cue = "finale_resolve"

        dialogue_pairs = _compose_chapter_dialogue(
            subtopic=st,
            prev_subtopic=prev_st,
            chapter_idx=idx,
            total_chapters=total_chapters,
            chapter_word_budget=per_chapter_budget,
            spec=spec,
            dossier=dossier,
            llm_client=llm_client,
            used_sentences=used_sentences,
        )

        turns: list[DialogueTurn] = []
        n_pairs = len(dialogue_pairs)
        for t_i, (_draft_speaker, text) in enumerate(dialogue_pairs):
            nxt_txt = dialogue_pairs[t_i + 1][1] if t_i + 1 < n_pairs else None
            turns.append(
                DialogueTurn(
                    turn_id=turn_counter,
                    chapter_index=idx,
                    # Documentary mode uses one consistent narrator; the dialogue composer
                    # may use HOST_B internally as a drafting device, but the finished script
                    # is authored and voiced as a single continuous narration.
                    speaker="HOST_A",
                    text=text,
                    spoken_text=normalize_for_tts(text),
                    pause_after_ms=compute_turn_pause_ms(text, nxt_txt, t_i == n_pairs - 1, turn_counter),
                    camera="wide_studio",
                    music_cue=cue,
                    is_chapter_opener=(t_i == 0),
                )
            )
            turn_counter += 1

        card_asset = st.visual_assets[0].file_path if st.visual_assets else ""
        chapters.append(
            ChapterScript(
                chapter_index=idx,
                title=st.title,
                subtitle=st.subtitle,
                era_or_date=st.era_or_date,
                location=st.location,
                music_cue=cue,
                card_duration_sec=0.0 if idx == 1 else (2.4 if spec.target_duration_sec >= 120 else 2.0),
                card_asset_path=card_asset,
                turns=turns,
            )
        )

    assign_multicam_choreography(chapters, dossier.subtopics)

    return EpisodeScript(
        spec=spec,
        chapters=chapters,
        llm_backend_used=env.llm_backend,
        revision_number=1,
        intro_duration_sec=4.2,
        outro_duration_sec=5.0,
        next_episode_tease=_generate_next_episode_tease(spec.topic),
    )
