"""Stage C: Visual Asset & Cinematography Audit + Auto-Grade / Plate Repair Loop."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from voxforge.config import VIDEO_SPEC
from voxforge.models import AuditCheck, EpisodeScript, ResearchDossier, StageAudit
from voxforge.video.infographics import measure_image_quality, render_procedural_documentary_plate


def _evaluate_visuals(dossier: ResearchDossier, script: EpisodeScript, total_dur_sec: float) -> list[AuditCheck]:
    all_assets = [a for st in dossier.subtopics for a in st.visual_assets]
    n_assets = max(1, len(all_assets))

    # V1: 1080p Full-HD resolution lock
    hd_ok = sum(1 for a in all_assets if a.width == VIDEO_SPEC.width and a.height == VIDEO_SPEC.height)
    c_v1 = AuditCheck(
        check_id="V1",
        name="1080p Widescreen Broadcast Resolution",
        passed=hd_ok == len(all_assets),
        score=round((hd_ok / n_assets) * 100.0, 1),
        measured_value=f"{hd_ok}/{len(all_assets)} plates at {VIDEO_SPEC.width}x{VIDEO_SPEC.height}",
        target_spec="100% plates matched to 1920x1080 SAR 1:1",
    )

    # V2: Optical sharpness (Laplacian variance)
    sharp_ok = sum(1 for a in all_assets if a.sharpness_score >= VIDEO_SPEC.min_sharpness_laplacian)
    min_sharp = min((a.sharpness_score for a in all_assets), default=0.0)
    c_v2 = AuditCheck(
        check_id="V2",
        name="Optical Sharpness & Detail (Laplacian Variance)",
        passed=sharp_ok == len(all_assets),
        score=round((sharp_ok / n_assets) * 100.0, 1),
        measured_value=f"{sharp_ok}/{len(all_assets)} sharp (min var={min_sharp:.1f})",
        target_spec=f"Laplacian variance >= {VIDEO_SPEC.min_sharpness_laplacian:.1f}",
    )

    # V3: Broadcast luminance & exposure safety (Rec.709 legal black 16.0 .. white 235.0)
    lum_ok = sum(1 for a in all_assets if 16.0 <= a.mean_luminance <= 228.0)
    c_v3 = AuditCheck(
        check_id="V3",
        name="Broadcast Luminance & Contrast Range",
        passed=lum_ok == len(all_assets),
        score=round((lum_ok / n_assets) * 100.0, 1),
        measured_value=f"{lum_ok}/{len(all_assets)} plates within Rec.709 broadcast luma range",
        target_spec="Mean Y luminance between 16 and 228 (Rec.709 IRE safe)",
    )

    # V4: Retention visual cut frequency (including sub-turn 3.8–5.6s editorial beats)
    subturn_cuts = sum(max(1, int(round(max(4.0, t.duration_sec) / 4.6))) for t in script.all_turns)
    total_shots = 2 + len(script.chapters) + subturn_cuts  # includes Intro Bumper & Outro Card
    avg_shot_sec = total_dur_sec / max(1, total_shots)
    shot_ok = 3.0 <= avg_shot_sec <= 22.0
    c_v4 = AuditCheck(
        check_id="V4",
        name="Visual Pacing & Camera Cut Cadence",
        passed=shot_ok,
        score=100.0 if shot_ok else 78.0,
        measured_value=f"{total_shots} camera cuts (avg {avg_shot_sec:.1f}s / shot)",
        target_spec="Avg shot length 3.0s – 22.0s for viewer retention",
    )

    # V5: Full Multi-Camera Studio Coverage, Inserts & Callouts
    cams = {t.camera for t in script.all_turns}
    has_wide = "wide_studio" in cams
    has_cu = bool({"closeup_host_a", "closeup_host_b"} & cams)
    has_react = "reaction_shot" in cams or len(script.all_turns) < 5
    has_insert = "studio_insert" in cams or len(script.all_turns) < 5
    has_callout = bool({"quote_card", "split_fact_card"} & cams)
    bookends_ok = 3.0 <= script.intro_duration_sec <= 5.0 and script.outro_duration_sec >= 4.0
    cov_hits = sum([has_wide, has_cu, has_react, has_insert, has_callout, bookends_ok])
    c_v5 = AuditCheck(
        check_id="V5",
        name="Multicam Coverage, Inserts, Callouts & Bookends",
        passed=(cov_hits == 6),
        score=round((cov_hits / 6.0) * 100.0, 1),
        measured_value=(
            f"Intro {script.intro_duration_sec:.1f}s, Outro {script.outro_duration_sec:.1f}s, "
            f"Angles={len(cams)} (Wide/CU/React/Insert/Callout={cov_hits}/6)"
        ),
        target_spec="Intro 3-5s + Wide + Close-ups + Reactions + Studio Inserts + Callouts + Outro",
    )

    return [c_v1, c_v2, c_v3, c_v4, c_v5]


def audit_and_improve_visuals(
    dossier: ResearchDossier,
    script: EpisodeScript,
    total_dur_sec: float,
    work_dir: Path,
) -> StageAudit:
    """Inspect every visual plate for blur/exposure/resolution and repair defective plates."""
    initial_checks = _evaluate_visuals(dossier, script, total_dur_sec)
    initial_score = round(sum(c.score for c in initial_checks) / len(initial_checks), 1)
    improvements: list[str] = []
    repaired = 0

    for st in dossier.subtopics:
        for idx, asset in enumerate(st.visual_assets):
            p = Path(asset.file_path)
            if not p.exists():
                render_procedural_documentary_plate(st, p, variant=idx)
                repaired += 1
            w, h, sharp, lum = measure_image_quality(p)
            needs_update = False

            if w != VIDEO_SPEC.width or h != VIDEO_SPEC.height:
                im = Image.open(p).convert("RGB").resize((VIDEO_SPEC.width, VIDEO_SPEC.height), Image.Resampling.LANCZOS)
                im.save(p, quality=93)
                needs_update = True

            if sharp < VIDEO_SPEC.min_sharpness_laplacian:
                # Apply studio unsharp mask; if still soft, replace with crisp architectural blueprint plate
                im = Image.open(p).convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.8, percent=155, threshold=2))
                im.save(p, quality=93)
                _, _, sharp2, _ = measure_image_quality(p)
                if sharp2 < VIDEO_SPEC.min_sharpness_laplacian:
                    render_procedural_documentary_plate(st, p, variant=idx + 3)
                needs_update = True

            if lum < 16.0 or lum > 228.0:
                im = Image.open(p).convert("RGB")
                factor = 1.35 if lum < 16.0 else 0.78
                im = ImageEnhance.Brightness(im).enhance(factor)
                im.save(p, quality=93)
                needs_update = True

            if needs_update:
                repaired += 1
                nw, nh, nsharp, nlum = measure_image_quality(p)
                asset.width = nw
                asset.height = nh
                asset.sharpness_score = round(nsharp, 2)
                asset.mean_luminance = round(nlum, 2)
                asset.passed_qa = True

    if repaired > 0:
        improvements.append(f"Auto-repaired/re-graded {repaired} visual plates for sharpness and broadcast luma.")
    else:
        improvements.append("Verified 100% of 1080p Rec.709 studio, insert, cinema & blueprint plates passed optical QA.")

    # Self-heal multicam coverage if any required angle was omitted on a >=6 turn script
    turns = script.all_turns
    if len(turns) >= 6:
        cams = {t.camera for t in turns}
        needed = [
            ang
            for ang in ("wide_studio", "closeup_host_a", "reaction_shot", "studio_insert", "quote_card")
            if ang not in cams
        ]
        if needed:
            for offset, ang in enumerate(needed, start=2):
                slot_i = min(len(turns) - 2, max(2, offset))
                if ang == "closeup_host_a":
                    for cand_i in range(2, len(turns) - 1):
                        if turns[cand_i].speaker == "HOST_A" and turns[cand_i].camera == "wide_studio":
                            slot_i = cand_i
                            break
                turns[slot_i].camera = ang  # type: ignore[assignment]
                if ang == "studio_insert" and not turns[slot_i].insert_variant:
                    turns[slot_i].insert_variant = "hands_notes"
            improvements.append(f"Injected missing broadcast camera coverage ({', '.join(needed)}).")

    final_checks = _evaluate_visuals(dossier, script, total_dur_sec)
    final_score = round(sum(c.score for c in final_checks) / len(final_checks), 1)

    return StageAudit(
        stage_name="Stage C · Cinematography & Visual Asset QA",
        initial_score=initial_score,
        final_score=final_score,
        passed=final_score >= 90.0,
        iterations=2 if repaired > 0 else 1,
        improvements_applied=improvements,
        checks=final_checks,
    )
