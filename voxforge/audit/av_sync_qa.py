"""Stage E — Narration↔Visual Semantic Sync Gate (final rendered-video audit).

This module inspects the FINAL muxed MP4 (decoded frames + actual audio stream) against
the authoritative `scene_plan.json` emitted by the renderer, and refuses certification
when pictures do not meaningfully correspond to the narration at the correct timestamps.

Checks performed:
  1. Semantic family validation — every planned scene's decoded frames are classified by
     pixel evidence (globe=blue sphere mass, sim/telemetry=near-black HUD with sparse
     bright glyphs, monument=warm stone luma under starfield, quote=large white text
     block) and compared against the family the plan promised for that narration clause.
  2. Temporal alignment — samples at 25/50/75% of every scene plus a dense probe 0.2s
     after each scene boundary (cuts), so early/late picture changes are caught.
  3. Frozen / black / drifting picture — inter-sample motion energy and luma floors.
  4. A/V duration drift — decoded container duration vs real master audio duration.
  5. Caption accuracy spot-check — active subtitle cues must show bright text in the
     caption band of the decoded frame.
  6. Subject hard gate — every shot must be bound to a real clause with a complete semantic
     directive, its scene_id/subject must match the directive, monument and wonder-context
     globe/quote shots independently re-match a wonder keyword in their own clause+context,
     neutral diagrams stay on scene_id -1, and no consecutive (subject, treatment) pair repeats.

Every scene's validation status is written back into `scene_plan.json` (never silently
accepted): a single critical semantic mismatch or subject-gate failure fails the stage and
therefore fails studio certification.
"""

from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import numpy as np

from voxforge.models import AudioMasterMeta, EpisodeScript, StageAudit, AuditCheck
from voxforge.video.procedural_world import match_wonder_scene_id
from voxforge.video.semantic_router import ALL_TEMPLATES, is_anaphoric_or_continuation

# Plan visual family -> expected pixel-evidence group produced by the pure-code engine.
_FAMILY_GROUPS: dict[str, set[str]] = {
    "broll_cinema": {"monument"},
    "closeup_host_a": {"monument"},
    "closeup_host_b": {"monument"},
    "wide_studio": {"geo"},
    "split_fact_card": {"darkhud"},
    "studio_insert": {"darkhud"},
    "reaction_shot": {"darkhud"},
    "quote_card": {"quote"},
    "semantic_diagram": {"darkhud"},
}

_PLACEHOLDER_CLAUSE = "(voice attribution / establishing beat)"
_MONUMENT_MODALITIES = {"monument_3d", "lidar_cad", "physics_sim"}
_KNOWN_SOURCES = {
    "wonder_keyword",
    "anaphora_continuation",
    "semantic_category",
    "place_globe",
    "citation_quote",
    "clause_diagram",
}


def classify_frame_group(frame_bgr: np.ndarray) -> str:
    """Classify one decoded frame into a visual-evidence group by pixel statistics.

    Calibrated against real engine output: globe scenes saturate blue across the TOP half,
    twilight monument cinema keeps a dark starfield top with blue-washed stone mid/bottom,
    sim/telemetry HUDs are near-black with sparse glyphs, quote cards carry big white text.
    """
    b = frame_bgr[..., 0].astype(np.float32)
    g = frame_bgr[..., 1].astype(np.float32)
    r = frame_bgr[..., 2].astype(np.float32)
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    mean_lum = float(lum.mean())
    deep_blue_frac = float(np.mean(b - r > 70.0))          # saturated ocean blue of the GIS globe
    blue_frac = float(np.mean((b - r > 25.0) & (b > 90.0)))  # any blue wash (sky gradients count too)
    white_mask = (r > 198.0) & (g > 198.0) & (b > 198.0)
    white_frac = float(np.mean(white_mask))
    hh, ww = lum.shape
    # Quote typography is concentrated in the central composition band, not spread across the
    # whole frame. A short 2-3 line inscription never reaches the 4.5% whole-frame white mass,
    # so measure the neutral-white density where the kinetic quote block actually lives.
    # Amber/cyan HUD glyphs fail the neutral-white mask; bright stone fails the dark-frame gate.
    band = white_mask[int(hh * 0.28) : int(hh * 0.72), int(ww * 0.08) : int(ww * 0.92)]
    quote_band_frac = float(np.mean(band))
    chan_spread = np.maximum(np.abs(r - g), np.maximum(np.abs(g - b), np.abs(r - b)))
    stone_frac = float(np.mean((lum > 70.0) & (lum < 215.0) & (chan_spread < 46.0)))
    dark_frac = float(np.mean(lum < 25.0))
    deep_mask = b - r > 70.0
    coherent_deep = float(np.mean(deep_mask.reshape(hh // 8, 8, ww // 8, 8).mean(axis=(1, 3)) > 0.7))
    land_frac = float(np.mean((g - r > 8.0) & (g > 55.0) & (b - g < 30.0) & (lum > 45.0) & (lum < 210.0)))
    if mean_lum < 75.0 and (white_frac > 0.045 or quote_band_frac > 0.012):
        # Quote cards: concentrated white typography on a dark backdrop. The whole-frame mass
        # test catches dense long inscriptions; the central-band test catches short 2-3 line
        # quotes. Noon-bright stone fails the mean-luma gate; sparse amber/cyan HUD glyphs fail
        # the neutral-white band threshold (measured telemetry band ~0.007 vs quote ~0.019).
        return "quote"
    if deep_blue_frac > 0.06 and blue_frac > 0.12 and (
        land_frac > 0.08 or (coherent_deep > 0.12 and deep_blue_frac > 0.13)
    ):
        # GIS globe: coherent saturated ocean disc with landmasses; sparse blue LiDAR
        # point clouds fail the land/coherence+luma conjunction.
        return "geo"
    if stone_frac > 0.30:
        return "monument"
    if blue_frac > 0.25 and float(lum.std()) > 28.0 and dark_frac < 0.55:
        # Blue-hour monument cinema: broad sky gradient, no saturated ocean disc.
        return "monument"
    stone_dark_frac = float(np.mean((lum > 45.0) & (lum < 215.0) & (chan_spread < 46.0)))
    if stone_dark_frac > 0.09 and mean_lum > 36.0 and dark_frac < 0.55:
        # Night-lit stone/steel monument: warm neutral mass above the HUD luma floor.
        return "monument"
    if stone_dark_frac > 0.07 and float(lum.std()) > 32.0 and mean_lum > 36.0 and dark_frac < 0.5:
        # Tight night closeups: smaller warm mass but strong highlight/shadow gradient
        # (HUD panels are flat-field and never reach this luma std).
        return "monument"
    if mean_lum < 70.0:
        return "darkhud"
    if stone_frac > 0.16:
        return "monument"
    return "unknown"


class _VideoProber:
    """Seeks and decodes frames from the final MP4 at exact timestamps."""

    def __init__(self, mp4_path: Path):
        self.container = av.open(str(mp4_path))
        self.vstream = self.container.streams.video[0]
        self.astream = self.container.streams.audio[0] if self.container.streams.audio else None
        tb = self.vstream.time_base
        self.duration_sec = float(self.container.duration / av.time_base) if self.container.duration else 0.0
        self._tb = float(tb) if tb else 1.0 / 24.0

    def frame_at(self, t_sec: float) -> np.ndarray | None:
        target_pts = int(t_sec / self._tb)
        self.container.seek(max(0, target_pts), stream=self.vstream)
        best: np.ndarray | None = None
        for frame in self.container.decode(self.vstream):
            if frame.pts is None:
                continue
            ft = frame.pts * self._tb
            best = frame.to_ndarray(format="bgr24")
            if ft >= t_sec - 0.02:
                return best
        return best

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:  # noqa: BLE001
            pass


def _chk(check_id: str, name: str, passed: bool, score: float, measured: str, target: str, notes: str = "") -> AuditCheck:
    return AuditCheck(check_id=check_id, name=name, passed=passed, score=score, measured_value=measured, target_spec=target, notes=notes)


def _motion_energy(a: np.ndarray, b: np.ndarray) -> float:
    da = a[::4, ::4].astype(np.float32)
    db = b[::4, ::4].astype(np.float32)
    return float(np.mean(np.abs(da - db)))


def _frame_fingerprint(frame_bgr: np.ndarray) -> np.ndarray:
    """Small grayscale thumbnail used for whole-film near-duplicate detection."""
    small = cv2.resize(frame_bgr, (160, 90))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()


# Measured duplicates on the pre-fix film scored 0.908-0.95; distinct same-family shots stay
# well under this (different camera geometry or clause-salted diagram layout).
_DUPLICATE_CORR_THRESHOLD = 0.90


def duplicate_frame_pairs(
    fingers: dict[int, np.ndarray], threshold: float = _DUPLICATE_CORR_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Return every (seg_a, seg_b, correlation) pair whose mid-shots are near-identical."""
    pairs: list[tuple[int, int, float]] = []
    keys = sorted(fingers)
    for ai in range(len(keys)):
        for bi in range(ai + 1, len(keys)):
            x, y = fingers[keys[ai]], fingers[keys[bi]]
            if x.shape != y.shape:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(corr) and corr >= threshold:
                pairs.append((keys[ai], keys[bi], round(corr, 3)))
    return pairs


def _caption_band_active(frame_bgr: np.ndarray) -> bool:
    band = frame_bgr[int(frame_bgr.shape[0] * 0.84) :, ...]
    r = band[..., 2].astype(np.float32)
    g = band[..., 1].astype(np.float32)
    b = band[..., 0].astype(np.float32)
    bright = np.mean((r > 190.0) & (g > 190.0) & (b > 190.0))
    return bool(bright > 0.0015)


def _subject_hard_gate(entries: list[dict]) -> list[tuple[int, str]]:
    """Subject-level certification gate over the renderer's per-clause semantic directives.

    Every shot must be bound to a real narration clause, its recorded scene_id must match the
    directive, monument shots must independently re-match a wonder keyword in their own clause,
    generic shots must use a known clause-derived template, and no ``(subject, treatment)`` pair
    may repeat on consecutive shots. Returns ``[(seg_index, reason), ...]`` and writes a failing
    ``validation_status``/note onto each offending entry.
    """
    failures: list[tuple[int, str]] = []
    prev_key: tuple[str, str] | None = None
    prev_directive: dict | None = None
    for e in entries:
        seg = int(e.get("seg_index", -1))
        clause = str(e.get("narration_clause", "")).strip()
        d = e.get("semantic_directive") or {}
        reasons: list[str] = []

        if not clause or clause == _PLACEHOLDER_CLAUSE or len(clause) < 12:
            reasons.append("placeholder_or_empty_clause")
        if not d or d.get("subject_key") in (None, "") or d.get("source") not in _KNOWN_SOURCES:
            reasons.append("missing_or_placeholder_directive")

        if d:
            if e.get("scene_id") != d.get("scene_id"):
                reasons.append("scene_id_directive_mismatch")
            if str(e.get("subject", "")) != str(d.get("subject_label", "")):
                reasons.append("subject_label_mismatch")
            modality = d.get("modality", "")
            sid = d.get("scene_id")
            source = d.get("source", "")
            # Clause-first subject verification. Monument geometry is only legal when the clause
            # itself names the wonder, or the clause is an anaphora/continuation of a shot that
            # did. Chapter-title context can never bind a monument onto a generic clause.
            if modality in _MONUMENT_MODALITIES or (
                modality in ("orbital_globe", "kinetic_quote") and isinstance(sid, int) and 0 <= sid <= 8
            ):
                if not isinstance(sid, int) or not 0 <= sid <= 8:
                    reasons.append("monument_shot_without_valid_scene_id")
                elif source == "anaphora_continuation":
                    chain_ok = (
                        is_anaphoric_or_continuation(clause)
                        and prev_directive is not None
                        and prev_directive.get("scene_id") == sid
                        and prev_directive.get("source") in ("wonder_keyword", "anaphora_continuation")
                    )
                    if not chain_ok:
                        reasons.append("anaphora_chain_broken_or_not_referential")
                elif source in ("wonder_keyword", "place_globe", "citation_quote"):
                    # Every non-anaphora monument binding must name the wonder in the clause itself.
                    if match_wonder_scene_id(clause) != sid:
                        reasons.append("clause_subject_does_not_match_rendered_monument")
                else:
                    reasons.append("monument_bound_without_clause_or_anaphora_binding")
            elif modality == "semantic_diagram":
                # Neutral clause-derived diagrams never bind to monument geometry.
                if sid != -1:
                    reasons.append("non_monument_shot_bound_to_monument_scene_id")
                if d.get("diagram_template") not in ALL_TEMPLATES:
                    reasons.append("unknown_diagram_template")
                # Semantic-drift guard: the diagram's meaning payload must exist, use a known
                # archetype, carry at least two clause-derived labels, and every label token
                # must trace back to the spoken clause (nothing invented on screen).
                from voxforge.video.clause_semantics import (
                    KNOWN_ARCHETYPES,
                    analyze_clause,
                    labels_are_clause_derived,
                )
                payload = d.get("semantic_payload")
                if not isinstance(payload, dict):
                    reasons.append("semantic_payload_missing")
                else:
                    if payload.get("archetype") not in KNOWN_ARCHETYPES:
                        reasons.append("semantic_archetype_unknown")
                    payload_labels = payload.get("labels") or []
                    if len(payload_labels) < 2:
                        reasons.append("semantic_payload_too_thin")
                    if not labels_are_clause_derived(clause, payload_labels):
                        reasons.append("semantic_label_not_clause_derived")
                    # The stored payload must match what the clause independently re-analyzes to.
                    fresh = analyze_clause(clause)
                    if fresh.archetype != payload.get("archetype"):
                        reasons.append("semantic_archetype_drift")
                    if sorted(fresh.labels) != sorted(payload_labels):
                        reasons.append("semantic_label_set_drift")
            elif modality in ("orbital_globe", "kinetic_quote"):
                if sid != -1:
                    reasons.append("invalid_scene_id_for_generic_globe_quote_shot")
            else:
                reasons.append("unknown_modality")
            key = (str(d.get("subject_key", "")), str(d.get("treatment", "")))
            if prev_key is not None and key == prev_key:
                reasons.append("consecutive_identical_subject_treatment")
            prev_key = key
            prev_directive = d

        if reasons:
            failures.append((seg, ";".join(reasons)))
            prior = str(e.get("validation_status", ""))
            if not prior.startswith("failed"):
                e["validation_status"] = "failed:subject_gate"
            e["validation_note"] = f"{e.get('validation_note', '')} | subject_gate={';'.join(reasons)}".strip(" |")
    return failures


def audit_av_sync(
    video_mp4: Path,
    scene_plan_json: Path,
    srt_path: Path,
    script: EpisodeScript,
    audio_meta: AudioMasterMeta,
) -> StageAudit:
    """Run the Stage E semantic A/V sync gate over the final rendered film."""
    checks: list[AuditCheck] = []
    improvements: list[str] = []
    mismatches = 0
    uncertain = 0
    subject_gate_hits = 0
    penalties = 0.0

    if not scene_plan_json.exists():
        return StageAudit(
            stage_name="AUDIT_E_AV_SYNC_SEMANTIC",
            initial_score=0.0,
            final_score=0.0,
            passed=False,
            iterations=1,
            improvements_applied=[],
            checks=[_chk("scene_plan_json", "Scene plan present", False, 0.0, "missing", "present", "Renderer did not emit a scene plan — cannot validate narration↔visual sync.")],
        )

    plan = json.loads(scene_plan_json.read_text(encoding="utf-8"))
    entries = plan.get("scenes", [])

    prober = _VideoProber(video_mp4)
    try:
        # --- Global drift + streams ---
        drift = abs(prober.duration_sec - audio_meta.duration_sec)
        checks.append(_chk(
            "av_duration_drift", "A/V duration drift", drift <= 0.15, max(0.0, 100.0 - drift * 400.0),
            f"{drift:.3f}s", "<=0.15s", "Decoded MP4 duration vs real mastered audio duration.",
        ))
        if drift > 0.15:
            penalties += 10.0
        checks.append(_chk(
            "audio_stream_present", "Audio stream present", prober.astream is not None,
            100.0 if prober.astream is not None else 0.0,
            "present" if prober.astream is not None else "MISSING", "AAC track",
        ))
        if prober.astream is None:
            penalties += 25.0

        # --- Per-scene semantic validation (denser sampling near cuts) ---
        validated = 0
        frozen_hits = 0
        black_hits = 0
        fingers: dict[int, np.ndarray] = {}
        for e in entries:
            fam = e.get("visual_family", "")
            expected = _FAMILY_GROUPS.get(fam, {"monument", "darkhud", "geo", "quote"})
            t0 = float(e.get("scene_start_sec", 0.0))
            t1 = float(e.get("scene_end_sec", t0))
            if t1 - t0 < 0.5:
                e["validation_status"] = "skipped:too_short"
                continue
            probes = [t0 + (t1 - t0) * f for f in (0.25, 0.5, 0.75)]
            probes.append(t0 + 0.2)          # dense probe right after the cut-in
            probes.append(t1 - 0.25)         # dense probe right before the cut-out
            groups: list[str] = []
            prev_frame: np.ndarray | None = None
            motion_vals: list[float] = []
            lums: list[float] = []
            for pi, pt in enumerate(probes):
                fr = prober.frame_at(min(max(pt, 0.05), max(0.1, prober.duration_sec - 0.05)))
                if fr is None:
                    continue
                groups.append(classify_frame_group(fr))
                lums.append(float(fr.astype(np.float32).mean()))
                if pi == 1:  # mid-shot frame feeds the whole-film duplicate detector
                    fingers[int(e.get("seg_index", 0))] = _frame_fingerprint(fr)
                if prev_frame is not None and prev_frame.shape == fr.shape:
                    motion_vals.append(_motion_energy(prev_frame, fr))
                prev_frame = fr
            if not groups:
                e["validation_status"] = "failed:no_frames_decoded"
                mismatches += 1
                continue
            core = groups[1:-1] if len(groups) >= 5 else groups
            hit = sum(1 for gg in core if gg in expected)
            frac = hit / max(1, len(core))
            if frac >= 0.5:
                e["validation_status"] = "validated"
                e["validation_note"] = f"frames={','.join(groups)} expected={sorted(expected)}"
                validated += 1
            else:
                e["validation_note"] = f"frames={','.join(groups)} expected={sorted(expected)}"
                if hit == 0:
                    mismatches += 1
                    e["validation_status"] = "failed:semantic_mismatch"
                else:
                    # Uncertain (partial agreement, no majority): never silently accept or
                    # silently fail — queue the decoded probe frames for human review and
                    # block auto-certification until a human clears the queue.
                    uncertain += 1
                    e["validation_status"] = "needs_human_review"
                    review_dir = scene_plan_json.parent / "human_review"
                    review_dir.mkdir(parents=True, exist_ok=True)
                    for k_r, pt_r in enumerate(probes):
                        fr_r = prober.frame_at(min(max(pt_r, 0.05), max(0.1, prober.duration_sec - 0.05)))
                        if fr_r is not None:
                            cv2.imwrite(
                                str(review_dir / f"seg{int(e.get('seg_index', 0)):04d}_p{k_r}.jpg"),
                                fr_r,
                                [cv2.IMWRITE_JPEG_QUALITY, 85],
                            )
                    e["validation_note"] += (
                        f" | probe frames dumped to human_review/seg{int(e.get('seg_index', 0)):04d}_p*.jpg"
                    )
            if lums and max(lums) < 8.0:
                black_hits += 1
                e["validation_status"] = "failed:black_scene"
            if len(motion_vals) >= 2 and max(motion_vals) < 0.35 and (t1 - t0) > 2.0:
                frozen_hits += 1
                e["validation_status"] = "failed:frozen_picture"

        total_scored = max(1, sum(1 for e in entries if not str(e.get("validation_status", "")).startswith("skipped")))
        checks.append(_chk(
            "semantic_scene_match", "Semantic scene match", mismatches == 0,
            100.0 * validated / total_scored, f"{validated}/{total_scored}", f"{total_scored}/{total_scored}",
            "Decoded frames of every planned scene match the family promised for its narration clause.",
        ))
        checks.append(_chk("black_scenes", "No black scenes", black_hits == 0, 100.0 if black_hits == 0 else 0.0, str(black_hits), "0"))
        checks.append(_chk("frozen_scenes", "No frozen scenes", frozen_hits == 0, 100.0 if frozen_hits == 0 else 0.0, str(frozen_hits), "0"))
        checks.append(_chk(
            "human_review_queue", "No scenes awaiting human review", uncertain == 0,
            100.0 if uncertain == 0 else 0.0, str(uncertain), "0",
            "Uncertain scenes dump decoded probe frames into human_review/ for manual clearance.",
        ))
        penalties += 4.0 * black_hits + 4.0 * frozen_hits

        # --- Whole-film near-duplicate gate: the same picture must never return, even far apart ---
        duplicate_pairs = duplicate_frame_pairs(fingers)
        dup_segs = {s for pair in duplicate_pairs for s in pair[:2]}
        for e in entries:
            if int(e.get("seg_index", -1)) in dup_segs and str(e.get("validation_status", "")).startswith("validated"):
                e["validation_status"] = "failed:repeated_shot"
        checks.append(_chk(
            "no_repeated_shots",
            "No repeated shots across the whole film",
            len(duplicate_pairs) == 0,
            100.0 if not duplicate_pairs else 0.0,
            ", ".join(f"seg{a}~seg{b}:{c}" for a, b, c in duplicate_pairs[:6]) or "0",
            "0 pairs",
            "Mid-shot fingerprints of every scene are correlated across the full timeline; "
            "any near-identical pair (same template+seed, same camera path or same globe framing) "
            "blocks certification regardless of distance between the shots.",
        ))
        if duplicate_pairs:
            penalties += 8.0 * len(duplicate_pairs)
            improvements.append(
                "Repeated-shot gate blocked "
                + ", ".join(f"seg{a}~seg{b} ({c})" for a, b, c in duplicate_pairs[:6])
            )

        # --- Subject-level hard gate: clause binding, subject match, no-repetition invariant ---
        subject_gate_failures = _subject_hard_gate(entries)
        subject_gate_hits = len(subject_gate_failures)
        checks.append(_chk(
            "subject_hard_gate",
            "Subject hard gate (clause-bound, no wrong-subject or repeated shots)",
            subject_gate_hits == 0,
            100.0 * (total_scored - subject_gate_hits) / total_scored,
            f"{total_scored - subject_gate_hits}/{total_scored}",
            f"{total_scored}/{total_scored}",
            "Each rendered scene_id must match the clause-resolved subject; placeholder-bound "
            "shots and consecutive identical (subject, treatment) pairs fail certification.",
        ))
        if subject_gate_hits:
            details = ", ".join(f"seg{s}:{why}" for s, why in subject_gate_failures[:6])
            improvements.append(f"Subject hard gate blocked {subject_gate_hits} shot(s): {details}")

        # --- Caption accuracy spot-check against real decoded frames ---
        turns = script.all_turns
        caption_hits = 0
        caption_probes = 0
        for e in entries[:: max(1, len(entries) // 6)][:6]:
            t_mid = 0.5 * (float(e["scene_start_sec"]) + float(e["scene_end_sec"]))
            active = any(t.start_sec - 0.05 <= t_mid <= t.start_sec + max(0.5, t.duration_sec) for t in turns)
            if not active:
                continue
            caption_probes += 1
            # Tolerate inter-cue gaps: caption must be present within ±0.3s of the probe.
            hit = False
            for dt_c in (0.0, -0.3, 0.3):
                fr = prober.frame_at(max(0.05, t_mid + dt_c))
                if fr is not None and _caption_band_active(fr):
                    hit = True
                    break
            if hit:
                caption_hits += 1
        if caption_probes:
            checks.append(_chk(
                "caption_sync_spotcheck", "Caption sync spot-check", caption_hits == caption_probes,
                100.0 * caption_hits / caption_probes, f"{caption_hits}/{caption_probes}", f"{caption_probes}/{caption_probes}",
                "Bright caption text present in decoded frame whenever a narration cue is live.",
            ))
            if caption_hits < caption_probes:
                penalties += 3.0 * (caption_probes - caption_hits)
    finally:
        prober.close()

    final_score = round(
        max(0.0, 100.0 - penalties - (6.0 * mismatches) - (3.0 * uncertain) - (6.0 * subject_gate_hits)), 1
    )
    passed = (
        mismatches == 0
        and uncertain == 0
        and subject_gate_hits == 0
        and not duplicate_pairs
        and final_score >= 90.0
    )
    improvements.append(f"Validated {len(entries)} scene-plan entries against decoded final-render frames.")
    if mismatches:
        improvements.append(f"FLAGGED {mismatches} narration↔visual semantic mismatches (certification blocked).")
    if uncertain:
        improvements.append(
            f"Queued {uncertain} uncertain scene(s) for HUMAN review (probe frames in human_review/); "
            "auto-certification blocked until cleared."
        )

    # Write per-scene validation verdicts back into the authoritative plan file.
    plan["scenes"] = entries
    plan["stage_e"] = {
        "score": final_score,
        "passed": passed,
        "mismatches": mismatches,
        "human_review_queue": uncertain,
        "subject_gate_failures": subject_gate_hits,
        "repeated_shot_pairs": [f"{a}-{b}:{c}" for a, b, c in duplicate_pairs],
        "black_scenes": black_hits,
        "frozen_scenes": frozen_hits,
        "av_drift_sec": round(drift, 3),
    }
    scene_plan_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    return StageAudit(
        stage_name="AUDIT_E_AV_SYNC_SEMANTIC",
        initial_score=final_score,
        final_score=final_score,
        passed=passed,
        iterations=1,
        improvements_applied=improvements,
        checks=checks,
    )
