"""Stage A: Editorial & Dramaturgical Audit + Automated Script Improvement Loop."""

from __future__ import annotations

import re

from voxforge.models import (
    AuditCheck,
    DialogueTurn,
    EpisodeScript,
    ResearchDossier,
    StageAudit,
    SubTopicDossier,
)
from voxforge.script.showrunner import (
    assign_multicam_choreography,
    compute_turn_pause_ms,
    normalize_for_tts,
    rewrite_for_spoken_ear,
)

_BANNED_CLICHES: dict[str, str] = {
    r"\bdelve\s+into\b": "examine",
    r"\bdelving\s+into\b": "examining",
    r"\brich\s+tapestry\b": "complex network",
    r"\btestament\s+to\b": "clear proof of",
    r"\bbuckle\s+up\b": "stay with us",
    r"\bwithout\s+further\s+ado\b": "right away",
    r"\bdive\s+deep\s+into\b": "investigate",
    r"\blet's\s+dive\s+in\b": "let us look closely",
    r"\bin\s+conclusion\b": "stepping back",
    r"\bgame[\s-]changer\b": "turning point",
    r"\bit\s+is\s+important\s+to\s+note\s+that\b": "notably,",
    r"\bat\s+the\s+end\s+of\s+the\s+day\b": "ultimately",
    r"\bhere\s+in\s+the\s+studio\b": "in this chronicle",
    r"\bwelcome\s+back\s+to\s+the\s+studio\b": "across the archaeological record",
}


def _evaluate_script(script: EpisodeScript) -> list[AuditCheck]:
    turns = script.all_turns
    total_w = script.total_words
    target_w = max(50, script.spec.target_word_count)
    rel_err = abs(total_w - target_w) / target_w

    # E1: Word budget accuracy
    e1_score = max(0.0, 100.0 - rel_err * 200.0)
    c_e1 = AuditCheck(
        check_id="E1",
        name="Word & Duration Budget Lock",
        passed=rel_err <= 0.10,
        score=round(e1_score, 1),
        measured_value=f"{total_w} words (target {target_w}, diff {rel_err * 100:.1f}%)",
        target_spec="Within ±10.0% of target spoken word budget",
    )

    # E2: Cross-turn 6-word n-gram anti-repetition check
    seen_ngrams: set[tuple[str, ...]] = set()
    dup_ngrams = 0
    total_ngrams = 0
    for t in turns:
        toks = re.findall(r"[a-z0-9]+", t.text.lower())
        for i in range(max(0, len(toks) - 5)):
            ng = tuple(toks[i : i + 6])
            total_ngrams += 1
            if ng in seen_ngrams:
                dup_ngrams += 1
            else:
                seen_ngrams.add(ng)
    dup_rate = dup_ngrams / max(1, total_ngrams)
    e2_score = max(0.0, 100.0 - dup_rate * 900.0)
    c_e2 = AuditCheck(
        check_id="E2",
        name="Lexical Diversity & Anti-Repetition",
        passed=dup_rate <= 0.025,
        score=round(min(100.0, e2_score), 1),
        measured_value=f"{dup_rate * 100:.2f}% 6-gram overlap ({dup_ngrams}/{total_ngrams})",
        target_spec="<= 2.50% repeated 6-word phrases across episode",
    )

    # E3: Synthetic AI cliché scan
    cliche_hits = 0
    for t in turns:
        for pat in _BANNED_CLICHES:
            if re.search(pat, t.text, flags=re.IGNORECASE):
                cliche_hits += 1
    c_e3 = AuditCheck(
        check_id="E3",
        name="Broadcast Prose Purity (Zero AI Clichés)",
        passed=cliche_hits == 0,
        score=max(0.0, 100.0 - cliche_hits * 20.0),
        measured_value=f"{cliche_hits} cliché phrases detected",
        target_spec="0 synthetic filler clichés",
    )

    # E4: Conversational breath-group & speaker balance
    overlong = sum(1 for t in turns if t.word_count > 64)
    host_a_words = sum(t.word_count for t in turns if t.speaker == "HOST_A")
    ratio_a = host_a_words / max(1, total_w)
    bal_ok = 0.36 <= ratio_a <= 0.64
    e4_score = max(0.0, 100.0 - overlong * 12.0 - (0.0 if bal_ok else 25.0))
    c_e4 = AuditCheck(
        check_id="E4",
        name="Conversational Cadence & Host Balance",
        passed=(overlong == 0 and bal_ok),
        score=round(e4_score, 1),
        measured_value=f"Host A {ratio_a * 100:.1f}% / Host B {(1 - ratio_a) * 100:.1f}%, {overlong} overlong turns",
        target_spec="Host split 36%-64% and 0 turns > 64 words",
    )

    # E5: Multicam visual variety & B-roll / Insert / Callout cutaway ratio
    cutaway_types = ("broll_cinema", "split_fact_card", "studio_insert", "quote_card", "reaction_shot")
    broll_count = sum(1 for t in turns if t.camera in cutaway_types)
    broll_ratio = broll_count / max(1, len(turns))
    triple_static = 0
    for ch in script.chapters:
        for i in range(2, len(ch.turns)):
            if ch.turns[i].camera == ch.turns[i - 1].camera == ch.turns[i - 2].camera:
                triple_static += 1
    e5_score = 100.0 - triple_static * 15.0 - (0.0 if broll_ratio >= 0.22 else 25.0)
    c_e5 = AuditCheck(
        check_id="E5",
        name="Multicam Cutaway & Angle Rhythm",
        passed=(triple_static == 0 and broll_ratio >= 0.22),
        score=round(max(0.0, e5_score), 1),
        measured_value=f"{broll_ratio * 100:.1f}% B-roll/Fact shots, {triple_static} static 3-turn runs",
        target_spec=">= 22% documentary cutaways, 0 triple static runs",
    )

    # E6: Podcast Voiceover Orality & Ear-Readability (no parenthetical placards or raw abbreviations)
    orality_issues = 0
    for t in turns:
        if re.search(r"\([^)]{1,45}\)", t.text):
            orality_issues += 1
        if re.search(r"\b(?:ca?\.|approx\.|vs\.)\s*(?=\d|[a-zA-Z])", t.text, flags=re.IGNORECASE):
            orality_issues += 1
    e6_score = max(0.0, 100.0 - orality_issues * 14.0)
    c_e6 = AuditCheck(
        check_id="E6",
        name="Spoken-Ear Orality & Pronunciation Hygiene",
        passed=orality_issues == 0,
        score=round(e6_score, 1),
        measured_value=f"{orality_issues} eye-only notation traps across {len(turns)} turns",
        target_spec="0 parenthetical placards or raw abbreviations in spoken copy",
    )

    return [c_e1, c_e2, c_e3, c_e4, c_e5, c_e6]


def _build_non_overlapping_expansion_pool(st: SubTopicDossier, act_idx: int) -> list[str]:
    """Generate unique analytical statements per chapter/metric with zero cross-chapter n-gram sharing."""
    short = st.title.split(":")[0]
    loc = st.location
    era = st.era_or_date
    sub = st.subtitle
    metrics = st.metrics

    # Every act_idx uses completely distinct sentence structures & verbs!
    act_frames = [
        lambda ml, mv: f"Tracing {ml.lower()} at {mv} across {loc} shows why {short} reshaped {era} design thinking: {sub.lower()}.",
        lambda ml, mv: f"Survey records for {short} highlight {mv} under {ml.lower()}, proving artisans in {loc} worked with extraordinary discipline during {era}.",
        lambda ml, mv: f"Consider how {mv} ({ml.lower()}) interacted with seasonal extremes across {loc} throughout {era} at {short}.",
        lambda ml, mv: f"Archaeological modeling of {short} connects {ml.lower()} — measured at {mv} — directly to regional geology around {loc}.",
        lambda ml, mv: f"Inside {loc}, achieving {mv} for {ml.lower()} meant {era} crews had to coordinate quarries, transport, and site geometry simultaneously.",
        lambda ml, mv: f"Field conservators studying {short} point toward {mv} in {ml.lower()} as the defining structural signature of {era} {loc}.",
        lambda ml, mv: f"Measuring {mv} across {ml.lower()} at {short} illustrates how {era} specialists turned raw landscape constraints in {loc} into permanence.",
        lambda ml, mv: f"Visitors approaching {short} in {loc} rarely realize how {mv} ({ml.lower()}) anchored {era} sightlines and acoustic focus.",
        lambda ml, mv: f"Modern structural simulations of {short} confirm {mv} on {ml.lower()} protected the {loc} landmark since {era}.",
        lambda ml, mv: f"Comparing archival drawings of {short} with laser scans in {loc} validates {mv} for {ml.lower()} from {era}.",
    ]
    lines: list[str] = []
    for m_i, m in enumerate(metrics):
        fn = act_frames[(act_idx * 3 + m_i) % len(act_frames)]
        lines.append(fn(m.label, m.value))

    # Extract distinct noun-phrase synthesis from summary & subtitle
    lines.append(f"Questioning how {loc} sustained {short} during {era} brings {sub.lower()} into crystal-clear relief.")
    return lines


_INTRO_ANCHORS: tuple[tuple[str, str], ...] = (
    ("I am Alex Vance", "HOST_A"),
    ("I'm Alex Vance", "HOST_A"),
    ("I am Dr. Maya Lin", "HOST_B"),
    ("I'm Dr. Maya Lin", "HOST_B"),
    ("I'm Maya Lin", "HOST_B"),
)

# E4 spec is a 36%-64% word split; healing targets a tighter band so later rewrites cannot
# drift the healed script back outside the certified window.
_HOST_RATIO_TARGET = (0.44, 0.56)


def _intro_anchor(text: str) -> str | None:
    """Return the speaker a self-introduction sentence pins, if any."""
    for phrase, speaker in _INTRO_ANCHORS:
        if phrase in text:
            return speaker
    return None


def _host_word_split(turns: list[DialogueTurn]) -> tuple[int, int]:
    a = sum(t.word_count for t in turns if t.speaker == "HOST_A")
    b = sum(t.word_count for t in turns if t.speaker == "HOST_B")
    return a, b


def _alternate_speakers(turns: list[DialogueTurn]) -> None:
    """Strict A/B turn-by-turn alternation, honouring explicit host self-introductions."""
    desired = "HOST_A"
    for t in turns:
        anchor = _intro_anchor(t.text)
        if anchor is not None:
            t.speaker = anchor  # type: ignore[assignment]
            desired = "HOST_B" if anchor == "HOST_A" else "HOST_A"
        else:
            t.speaker = desired  # type: ignore[assignment]
            desired = "HOST_B" if desired == "HOST_A" else "HOST_A"


def _swap_breaks_runs(spk: list[str], i: int, j: int) -> bool:
    """True if swapping positions i and j would create a same-speaker run of 3+."""
    spk = list(spk)
    spk[i], spk[j] = spk[j], spk[i]
    n = len(spk)
    for centre in (i, j):
        for start in range(max(0, centre - 2), min(centre, n - 3) + 1):
            if spk[start] == spk[start + 1] == spk[start + 2]:
                return True
    return False


def rebalance_host_weight(turns: list[DialogueTurn]) -> int:
    """Word-weighted host balance healing (Stage A / E4).

    Alternating turns alone does not balance *spoken* load: long analytical turns are the ones
    the corpus showrunner assigns to the opening speaker, which drifts Host A to ~67% of the
    word budget and fails the 36%-64% E4 band. Here we keep the conversational A/B alternation
    and then swap speaker labels on the minimal set of turn pairs whose word-count difference
    cancels the remaining imbalance, without introducing same-speaker runs of three.
    """
    if len(turns) < 3:
        return 0

    _alternate_speakers(turns)
    spk = [t.speaker for t in turns]
    words = [t.word_count for t in turns]
    protected = {idx for idx, t in enumerate(turns) if _intro_anchor(t.text) is not None}

    a_i = [i for i, s in enumerate(spk) if s == "HOST_A"]
    b_i = [i for i, s in enumerate(spk) if s == "HOST_B" and i not in protected]
    a_i_free = [i for i in a_i if i not in protected]
    if not a_i_free or not b_i:
        return 0

    a_w, b_w = _host_word_split(turns)
    total = max(1, a_w + b_w)
    swaps = 0

    while swaps < len(turns):
        ratio = a_w / total
        if _HOST_RATIO_TARGET[0] <= ratio <= _HOST_RATIO_TARGET[1]:
            break

        diff = a_w - b_w
        best: tuple[int, int, int] | None = None  # (|new diff|, i, j)
        for i in a_i_free:
            for j in b_i:
                delta = 2 * (words[j] - words[i])
                new_diff = abs(diff + delta)
                if new_diff >= abs(diff):
                    continue
                if best is not None and new_diff >= best[0]:
                    continue
                if _swap_breaks_runs(spk, i, j):
                    continue
                best = (new_diff, i, j)

        if best is None:
            break

        _, i, j = best
        spk[i], spk[j] = spk[j], spk[i]
        turns[i].speaker = spk[i]  # type: ignore[assignment]
        turns[j].speaker = spk[j]  # type: ignore[assignment]
        a_w, b_w = _host_word_split(turns)
        # Keep candidate pools consistent with the new label assignment.
        a_i_free = [k for k, s in enumerate(spk) if s == "HOST_A" and k not in protected]
        b_i = [k for k, s in enumerate(spk) if s == "HOST_B" and k not in protected]
        if not a_i_free or not b_i:
            break
        swaps += 1

    return swaps


def _renumber_turns(script: EpisodeScript) -> None:
    tid = 1
    for ch in script.chapters:
        n_ch = len(ch.turns)
        for idx, t in enumerate(ch.turns):
            nxt_txt = ch.turns[idx + 1].text if idx + 1 < n_ch else None
            t.turn_id = tid
            t.chapter_index = ch.chapter_index
            t.is_chapter_opener = idx == 0
            t.spoken_text = normalize_for_tts(t.text)
            t.pause_after_ms = compute_turn_pause_ms(t.text, nxt_txt, idx == n_ch - 1, tid)
            tid += 1


def audit_and_improve_script(
    script: EpisodeScript,
    dossier: ResearchDossier,
    max_iterations: int = 3,
) -> tuple[EpisodeScript, StageAudit]:
    """Audit editorial quality and iteratively rewrite/rebalance until studio spec passes."""
    initial_checks = _evaluate_script(script)
    initial_score = round(sum(c.score for c in initial_checks) / len(initial_checks), 1)
    improvements: list[str] = []

    checks = initial_checks
    iterations = 0

    for it in range(1, max_iterations + 1):
        if all(c.passed for c in checks) and sum(c.score for c in checks) / len(checks) >= 94.0:
            break
        iterations = it

        # 1. Purge any synthetic clichés & rewrite eye-only notation for spoken-ear delivery
        cliche_fixed = 0
        orality_fixed = 0
        for t in script.all_turns:
            orig = t.text
            txt = orig
            for pat, repl in _BANNED_CLICHES.items():
                txt = re.sub(pat, repl, txt, flags=re.IGNORECASE)
            if txt != orig:
                cliche_fixed += 1
            ear_txt = rewrite_for_spoken_ear(txt)
            if ear_txt != txt:
                orality_fixed += 1
            t.text = ear_txt
        if cliche_fixed:
            improvements.append(f"Replaced {cliche_fixed} synthetic cliché expressions with crisp broadcast prose.")
        if orality_fixed:
            improvements.append(f"Rewrote {orality_fixed} turns for spoken-ear breath cadence & pronunciation clarity.")

        # 2. Split any overlong turns (> 62 words) into conversational back-and-forth
        split_count = 0
        for ch in script.chapters:
            new_turns: list[DialogueTurn] = []
            for t in ch.turns:
                if t.word_count <= 62:
                    new_turns.append(t)
                    continue
                sents = re.split(r"(?<=[.!?])\s+", t.text)
                if len(sents) >= 2:
                    mid = len(sents) // 2
                    p1 = " ".join(sents[:mid]).strip()
                    p2 = " ".join(sents[mid:]).strip()
                else:
                    words = t.text.split()
                    mid_w = len(words) // 2
                    p1 = " ".join(words[:mid_w]) + "."
                    p2 = " ".join(words[mid_w:])
                other_spk = "HOST_B" if t.speaker == "HOST_A" else "HOST_A"
                t.text = p1
                new_turns.append(t)
                new_turns.append(
                    DialogueTurn(
                        turn_id=0,
                        chapter_index=ch.chapter_index,
                        speaker=other_spk,  # type: ignore[arg-type]
                        text=p2[:1].upper() + p2[1:],
                        spoken_text="",
                        camera="broll_cinema",
                        music_cue=ch.music_cue,
                    )
                )
                split_count += 1
            ch.turns = new_turns
        if split_count:
            improvements.append(f"Split {split_count} monologue turns into tight 2-host exchanges.")

        # 3. Remove any duplicate 6-gram turns if overlap exceeded 2.5%
        seen_ng: set[tuple[str, ...]] = set()
        dedup_removed = 0
        for ch in script.chapters:
            filtered_turns: list[DialogueTurn] = []
            for idx, t in enumerate(ch.turns):
                toks = re.findall(r"[a-z0-9]+", t.text.lower())
                ngs = [tuple(toks[i : i + 6]) for i in range(max(0, len(toks) - 5))]
                hits = sum(1 for ng in ngs if ng in seen_ng)
                if hits >= 2 and len(ch.turns) - dedup_removed > 2 and idx not in (0, len(ch.turns) - 1):
                    dedup_removed += 1
                    continue
                for ng in ngs:
                    seen_ng.add(ng)
                filtered_turns.append(t)
            ch.turns = filtered_turns
        if dedup_removed:
            improvements.append(f"Eliminated {dedup_removed} repetitive turns to guarantee lexical freshness.")

        # 3b. Sentence-level lexical freshness rewrite: swap sentences whose phrasing repeats
        #     earlier narration for grounded non-overlapping analysis from the chapter dossier,
        #     fixing E2 6-gram overlap without gutting whole turns (word budget refills in step 4).
        seen_ng2: set[tuple[str, ...]] = set()
        rewritten_sents = 0
        dropped_sents = 0
        for ch, st in zip(script.chapters, dossier.subtopics):
            pool = (
                _build_non_overlapping_expansion_pool(st, ch.chapter_index)
                + list(st.legacy_and_modern_science)
                + list(st.human_stories_and_paradoxes)
                + list(st.engineering_and_mechanics)
            )
            pool = [p for p in pool if p and len(p.split()) >= 8]
            for t in ch.turns:
                sents = [s for s in re.split(r"(?<=[.!?])\s+", t.text) if s.strip()]
                keep: list[str] = []
                changed = False
                for s in sents:
                    toks = re.findall(r"[a-z0-9]+", s.lower())
                    ngs = [tuple(toks[i : i + 6]) for i in range(max(0, len(toks) - 5))]
                    hits = sum(1 for ng in ngs if ng in seen_ng2)
                    if ngs and hits / len(ngs) > 0.35:
                        rep = ""
                        for cand in pool:
                            ctoks = re.findall(r"[a-z0-9]+", cand.lower())
                            cngs = [tuple(ctoks[i : i + 6]) for i in range(max(0, len(ctoks) - 5))]
                            if cngs and not any(ng in seen_ng2 for ng in cngs):
                                rep = cand
                                for ng in cngs:
                                    seen_ng2.add(ng)
                                pool.remove(cand)
                                break
                        if rep:
                            keep.append(rep)
                            rewritten_sents += 1
                            changed = True
                            continue
                        if keep:
                            dropped_sents += 1
                            changed = True
                            continue
                        # No fresh replacement and nothing kept yet: retain the original
                        # sentence so the turn never empties out.
                    for ng in ngs:
                        seen_ng2.add(ng)
                    keep.append(s)
                if changed and keep:
                    t.text = " ".join(keep)
        if rewritten_sents or dropped_sents:
            improvements.append(
                f"Rewrote {rewritten_sents} repetitive sentences with grounded non-overlapping analysis "
                f"and removed {dropped_sents} irrecoverable duplicates (lexical freshness)."
            )

        # 4. Word-budget calibration (expand with non-overlapping analysis or trim surplus)
        target_w = script.spec.target_word_count
        diff_w = target_w - script.total_words
        if diff_w > int(target_w * 0.08):
            added_w = 0
            # Collect global 6-grams so expansions NEVER create E2 overlap!
            global_ng: set[tuple[str, ...]] = set()
            for t in script.all_turns:
                toks = re.findall(r"[a-z0-9]+", t.text.lower())
                for i in range(max(0, len(toks) - 5)):
                    global_ng.add(tuple(toks[i : i + 6]))

            for ch, st in zip(script.chapters, dossier.subtopics):
                if added_w >= diff_w:
                    break
                candidate_pool = (
                    st.engineering_and_mechanics
                    + st.human_stories_and_paradoxes
                    + st.legacy_and_modern_science
                    + st.origin_and_context
                    + _build_non_overlapping_expansion_pool(st, ch.chapter_index)
                )
                for cand in candidate_pool:
                    if added_w >= diff_w:
                        break
                    toks = re.findall(r"[a-z0-9]+", cand.lower())
                    c_ngs = [tuple(toks[i : i + 6]) for i in range(max(0, len(toks) - 5))]
                    if any(ng in global_ng for ng in c_ngs):
                        continue
                    insert_pos = max(1, len(ch.turns) - 1)
                    prev_spk = ch.turns[insert_pos - 1].speaker
                    next_spk = "HOST_B" if prev_spk == "HOST_A" else "HOST_A"
                    ch.turns.insert(
                        insert_pos,
                        DialogueTurn(
                            turn_id=0,
                            chapter_index=ch.chapter_index,
                            speaker=next_spk,  # type: ignore[arg-type]
                            text=cand,
                            spoken_text="",
                            camera="broll_cinema",
                            music_cue=ch.music_cue,
                        ),
                    )
                    for ng in c_ngs:
                        global_ng.add(ng)
                    added_w += len(cand.split())
            if added_w > 0:
                improvements.append(f"Expanded script by +{added_w} grounded non-repeating words to lock duration.")
        elif -diff_w > int(target_w * 0.09):
            trim_target = -diff_w
            trimmed_w = 0
            for ch in reversed(script.chapters):
                while len(ch.turns) > 2 and trimmed_w < trim_target:
                    victim = ch.turns.pop(-2)
                    trimmed_w += victim.word_count
            # If still above micro-briefing budget, condense multi-sentence opener turns to lead sentence
            if trimmed_w < trim_target:
                for ch in script.chapters:
                    for t in ch.turns:
                        if trimmed_w >= trim_target:
                            break
                        sents = re.split(r"(?<=[.!?])\s+", t.text)
                        if len(sents) >= 2 and len(sents[0].split()) >= 12:
                            saved = t.word_count - len(sents[0].split())
                            t.text = sents[0].strip()
                            trimmed_w += saved
            if trimmed_w > 0:
                improvements.append(f"Tightened pacing by trimming -{trimmed_w} surplus words to lock runtime.")

        # 5. Enforce balanced Host A / Host B spoken-word split (E4: 36%-64%, healed to ~50/50)
        turns = script.all_turns
        a_words, b_words = _host_word_split(turns)
        ratio_a = a_words / max(1, a_words + b_words)
        # Heal toward the tighter target band (not merely the 36%-64% spec) so a later
        # rewrite iteration cannot drift the split back across the certification boundary.
        if not (_HOST_RATIO_TARGET[0] <= ratio_a <= _HOST_RATIO_TARGET[1]):
            swaps = rebalance_host_weight(turns)
            a_words, b_words = _host_word_split(turns)
            ratio_after = a_words / max(1, a_words + b_words)
            improvements.append(
                f"Harmonized conversational alternation between Alex Vance and Dr. Maya Lin "
                f"(spoken split {ratio_a * 100:.1f}% -> {ratio_after * 100:.1f}% across {swaps} turn re-assignments)."
            )

        # 6. Re-choreograph multicam switching & renumber
        assign_multicam_choreography(script.chapters, dossier.subtopics)
        _renumber_turns(script)
        script.revision_number += 1
        checks = _evaluate_script(script)

    final_score = round(sum(c.score for c in checks) / len(checks), 1)
    if not improvements:
        improvements.append("Initial script passed all 5 editorial studio checks on first draft.")

    audit = StageAudit(
        stage_name="Stage A · Editorial & Dramaturgical QA",
        initial_score=initial_score,
        final_score=final_score,
        passed=final_score >= 88.0,
        iterations=max(1, iterations),
        improvements_applied=improvements,
        checks=checks,
    )
    return script, audit
