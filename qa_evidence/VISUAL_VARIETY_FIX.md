# Live Subject Fidelity & Visual Anti-Repetition Fix

**Date:** 2026-10-26 · **Scope:** make the picture follow what is being said *right now*, and
stop a single holographic screen looping for a whole minute.

This is the follow-up to `NARRATION_VISUAL_BINDING_FIX.md`. That fix bound each **turn** to a
visual subject; two defects survived it, both reported against the rendered film.

---

## 1. The two reported defects

> *"in the first 10 or 15 seconds he said about earthquakes and what not but you keep showing
> pyramid that's the main thing i wanted you to fix — what narrator is talking about and what
> should be shown on screen should be same... and no shape or visual should repeat again and
> again; if narrator is talking about pyramid for 1 min straight then show different angles,
> don't show only one holographics screen again and again."*

| # | Defect | Cause |
|---|---|---|
| A | Aggregate/loss narration showed the **survivor** | "earthquakes, fire, and human greed destroyed **six**" was illustrated with the one pyramid that survived them. Turn-level binding resolved the whole opening to SCENE_GIZA because the chapter headline mentioned the Great Pyramid. |
| B | One screen held for a full minute | The beat planner hard-coded a canned camera pattern for episode openers/closers, and repeated beats of the same subject reused the identical camera family **and** the identical lens variant. |

## 2. Fix A — lost-wonders montage

Clauses with no surviving physical subject are detected by `is_lost_wonders_reference()` and
rendered as a rapid cut through the six lost wonders. Two shapes are recognised:

* **destruction of the group** — destruction language (`destroyed`, `earthquake`, `toppled`,
  `greed`, `collapsed`…) plus group context (`wonders`, `seven`, `six`, `monuments`);
* **bare enumeration of the canon** — `the original Seven Wonders cataloged by…` with no
  specific monument named.

Detection is deliberately narrow. A clause that is purely about the surviving monument
(`a 4,500-year-old limestone mountain on the edge of Cairo`) is **not** a loss reference and
stays a subject shot.

Three chain rules make it clause-faithful rather than turn-faithful:

1. **Continuation** — a mid-list clause (`by Greek travelers two thousand years ago`) stays
   inside the montage while no monument is named.
2. **Stop at the survivor** — the chain ends at the first clause that names the surviving
   monument, so the montage never bleeds into the subject shot.
3. **Split inheritance** — when a heavy clause is split into two beats, the flag goes to the
   first half and to the second half *only while that half is still part of the list*.

The montage itself (`StudioCharacterAnimator.iter_lost_wonders_montage_frames`) rotates
Pharos → Colossus → Artemis → Babylon → Halicarnassus → Olympia on a ~0.7 s cadence, then cuts
back to the survivor as a bookend. Each cut carries a deliberately **small, dim** lower-third
ID plate: a large bright caption would re-trip the frame classifier into reading the frame as
`quote` instead of `monument` (a failure already hit once during round 1).

### Result — the flagged opening

| Time | Narration | Before | After |
|---:|---|---|---|
| 0.0–2.4 s | *"Out of the original Seven Wonders cataloged by Greek travelers…"* | pyramid | establishing globe |
| 2.4–5.7 s | *"Out of the original Seven Wonders cataloged"* | pyramid | **lost-wonders montage** |
| 5.7–9.1 s | *"by Greek travelers two thousand years ago"* | pyramid | **lost-wonders montage** |
| 9.1–11.5 s | *"earthquakes"* | pyramid | **lost-wonders montage** |
| 11.5–15.9 s | *"fire, and human greed destroyed six — leaving only"* | pyramid | **lost-wonders montage** |
| 15.9–20.2 s | *"a 4,500-year-old limestone mountain on the edge of Cairo."* | pyramid | subject shot (Giza) |

13.5 s of the first 15 s is now the montage; the pyramid appears only when it is named.

## 3. Fix B — anti-repetition

* The hard-coded `is_episode_opener` / `is_episode_closer` camera patterns were **deleted**.
  Openers and closers are clause-driven like every other turn.
* **Family rotation** — when two consecutive beats share camera family *and* subject, the
  second is rotated through `_VARIETY_ROTATION`
  (`broll_cinema → studio_insert → broll_cinema → split_fact_card → …`), so a long stretch
  alternates 3D → LiDAR/CAD → physics/telemetry → globe/quote.
* **Lens rotation** — every repeat of the same subject through the same treatment advances the
  lens variant (orbit crane / LiDAR / satellite push), so even two adjacent `broll_cinema`
  beats of the same monument are different shots.
* Montage beats are exempt from rotation: they are already a multi-subject cut, and rotating
  them away would reintroduce defect A.

Measured on a 40 s single-subject Giza monologue: **6 beats, 3 distinct families, 0
consecutive repeats** (previously it would hold one screen).

## 4. Plumbing

* `_plan_turn_editorial_beats_detail()` returns `(beats, montage_flags)` alongside the beats.
  The montage decision depends on clause-to-clause continuation state inside the planner, so
  re-deriving it at render time would lose information. The original
  `_plan_turn_editorial_beats()` is kept as a thin wrapper for the existing tests.
* `PlateLayout` gained `montage: bool`; `render_broll_cinema(..., montage=True)` sets it.
* The segment encoder dispatches `task.layout.montage` to
  `iter_lost_wonders_montage_frames()` before the `broll_depth` branch.

## 5. Verification

```
.venv/bin/python -m pytest tests/test_visual_variety.py \
  tests/test_narration_visual_binding.py tests/test_av_sync_plan.py \
  tests/test_visual_and_subtitles.py tests/test_editorial_audit.py tests/test_intent.py -q
# 35 passed
```

`tests/test_visual_variety.py` (8 new tests) covers: destruction-of-group detection, canon
enumeration, survivor-only clauses staying subject shots, no false positives on unrelated
narration (volcanoes, baking, beacons), the flagged opening running the montage before the
survivor is named, ≥9 s of montage in the first 15 s, no back-to-back visual families on a
long single-subject stretch, and the unchanged `(camera, end_sec, clause)` beat contract.

Montage frames were also run through the production frame classifier: over 48 frames the
result was **34 `monument` / 14 `darkhud` / 0 `quote`** — the caption plate does not
re-trigger the misclassification seen in round 1.

## 6. Tooling

`tools/render_preview.py` re-renders the picture track from an existing `episode_script.json`
and `documentary_audio.mp3`, truncated to the first N seconds — no TTS, research or LLM, so a
visual fix can be checked without a 30-minute full render.

```
.venv/bin/python tools/render_preview.py output_episodes/7-wonders-of-the-world-5m --seconds 60
```
