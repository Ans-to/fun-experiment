# Narration↔Visual Subject Mismatch Fix

**Date:** 2026-10-26 · **Scope:** eliminate wrong-subject pictures in mixed-topic acts and make
the narration→visual binding auditable at clause level.

---

## 1. Reproduced failures

Static audit of the checked-in 5-minute Wonders episode (`episode_script.json`) found **6 of
18 turns** bound to the wrong procedural scene:

| Turn | Narration subject | Old visual caption |
|---:|---|---|
| 7 | Petra / the Siq | Flavian Amphitheatre |
| 9 | Nabataean / Petra trade network | Flavian Amphitheatre |
| 11 | Machu Picchu | El Castillo, Chichén Itzá |
| 13 | Machu Picchu / Pachacuti | El Castillo, Chichén Itzá |
| 15 | Taj Mahal / Shah Jahan | Rio de Janeiro |
| 17 | Mumtaz Mahal / Taj Mahal | Rio de Janeiro |

The prior 10-niche evidence plans also exposed two structural defects: **100 of 195 scenes**
used the placeholder narration clause `(voice attribution / establishing beat)`, and **195 of
195** recorded the generic `3D Parametric Classical Sanctuary & Rotunda Reconstruction` subject
even for volcanoes, AI, health, cooking and whales.

## 2. Root causes

1. **Chapter-level asset inheritance.** `assign_multicam_choreography()` assigned every turn
   in a paired act the caption and scene of one chapter-level plate. A chapter covering both
   Petra and the Colosseum therefore rendered only the Colosseum.
2. **First-match keyword resolution.** `resolve_monument_scene_id()` returned the first rule
   containing any keyword in the concatenated hint. Incidental words such as `Egypt` could
   override the actual subject, and a generic `wonder` fell back to `fallback_seed % 9`, i.e.
   a random monument.
3. **Mutated beat state.** `render_episode_video()` rebound `turn` to each beat's clause inside
   the beat loop. Later beats inherited an earlier clause, so the scene plan, captions and
   24 fps animation could all resolve from stale narration.
4. **Placeholder clauses.** Episode openers, closers and lead-attribution beats emitted empty
   clauses, making more than half of the authoritative scene plan unauditable.
5. **Family-only Stage E gate.** The final-video gate classified globe / HUD / monument / quote
   families but never checked whether a Petra clause was bound to a Petra scene ID.
6. **Wrong-subject generic geometry.** Unmapped topics fell back to a classical rotunda.

## 3. Fixes

- `DialogueTurn` now carries explicit `visual_scene_id` and `visual_subject`; the showrunner
  resolves them from each utterance and writes a subject-correct `visual_caption`.
  Turn-level telemetry is filtered against the spoken text so mixed acts do not display the
  other subject's figures.
- `procedural_world.py` uses weighted, clause-prioritized scene ranking. Strong proper nouns
  outrank weak geographic aliases; the random `wonder` fallback and classical rotunda are
  gone. Generic topics now render a neutral 3D data sculpture and a topic-aware explanatory
  simulation.
- `renderer.py` creates an isolated clause-scoped turn for every beat from the original turn,
  propagates the explicit scene ID through `PlateLayout`, and records the exact narration
  clause (no placeholders) plus `scene_id` in `scene_plan.json`.
- `compositor.py` and `animator.py` honor the explicit scene ID instead of re-guessing from a
  mixed chapter hint. This applies to B-roll, close-ups, inserts, quote cards, telemetry and
  live 24 fps 3D frames.
- Stage E adds `narration_subject_match`: a plan-level gate that detects missing clauses,
  wrong scene IDs and inconsistent subject labels, in addition to the existing decoded-frame
  family classifier. Any hit blocks studio certification.
- `tools/audit_narration_visual_binding.py` provides a video-free CI audit of any episode
  directory.

## 4. Verification

```bash
.venv/bin/python -m pytest tests/test_narration_visual_binding.py -q
.venv/bin/python -m pytest tests/test_intent.py tests/test_editorial_audit.py \
  tests/test_visual_and_subtitles.py tests/test_av_sync_plan.py \
  tests/test_narration_visual_binding.py -q
.venv/bin/python tools/audit_narration_visual_binding.py \
  output_episodes/7-wonders-of-the-world-5m
```

Results on this branch:

- **27/27** offline unit tests pass.
- The checked-in Wonders episode now has **0/18** mismatched turn bindings (was 6/18).
- New regression tests cover Petra/Colosseum, Machu Picchu/Chichén Itzá, Taj Mahal/Christ the
  Redeemer, generic-topic subjects, beat isolation, placeholder elimination and Stage E
  subject-gate detection.
- The neutral generic 3D frame was rendered at 640×360 and reclassified as `monument` family,
  so the existing Stage E frame gate remains compatible.

## 5. Notes

The 90-second niche evidence sheets in this directory are historical artifacts from the
previous run and still show the old generic subject labels; they were not regenerated because
the film-level matrix requires neural voice-model downloads unavailable in this sandbox. New
runs emit explicit `visual_scene_id`, topic-aware generic subjects and exact clause text, and
the static audit tool can verify both old and new episode directories.
