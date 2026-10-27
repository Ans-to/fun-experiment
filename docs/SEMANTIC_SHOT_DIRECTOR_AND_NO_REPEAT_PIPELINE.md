# Semantic Shot Director & No-Repeat Visual Pipeline

## Why this exists

The legacy resolver knew only ten wonder keyword rules. Any clause that did not name a wonder
fell through to a single generic rotunda scene (or, worse, rotated through unrelated monument
models by segment index), so films on arbitrary topics looped the same model and narration
could describe something completely different from the picture on screen. Audits of the
7-wonders film showed only **2 distinct scenes across 20 shots**, one scene reused 16 times
(including an 11-shot consecutive run), and **9/20 shots bound to a placeholder clause**
`"(voice attribution / establishing beat)"`.

This rebuild makes every picture a clause-derived, subject-bound, non-repeating decision, and
makes Stage E refuse certification when that contract is violated.

## The four components

### 1. Pure wonder matcher — `procedural_world.match_wonder_scene_id`

A side-effect-free keyword match returning `None` when no monument is relevant. The legacy
`resolve_procedural_scene_id` (with its rotating/generic fallbacks) is retained for the plate
caches but is no longer used to decide what a shot depicts.

### 2. Semantic router — `video/semantic_router.py`

**Clause-first binding (the core rule).** A wonder scene is bound only when the wonder is named
*inside the narration clause itself* (`match_wonder_scene_id(clause)`, no context fallback).
Chapter-title keywords never force monument geometry onto an abstract clause — this was the
root cause of shots like a lighthouse under the line "five known planets plus the Sun". When a
clause is a mid-sentence fragment or referential anaphora (lowercase-starting split fragment,
connective, opening pronoun, or a reference noun such as "the monument"/"the casing"), it
inherits the *previous* monument subject; the anaphora chain must originate from a clause that
named the wonder. Everything else routes to a neutral diagram, globe, or quote card.

`SemanticShotDirector.directive_for_clause(clause, context, intent)` returns an immutable
`VisualDirective` per clause:

| Clause kind | Modality | Picture |
|---|---|---|
| Names a wonder | `monument_3d` / `lidar_cad` / `physics_sim` / `orbital_globe` / `kinetic_quote` | exact monument scene (id 0–8), treatment ladder per subject |
| Place/route language or `geo` intent | `orbital_globe` (scene_id −1) | raycast Earth map |
| Citation language or `quote` intent | `kinetic_quote` (scene_id −1) | primary-source typography |
| Anything else | `semantic_diagram` (scene_id −1) | one of six clause-driven diagram engines |

Seven semantic categories (space science, geoscience, biology/health, food/process,
technology, society/economy, history/humanities) each own a template ladder; unmatched text
uses the abstract ladder. Directives are resolved **after** beat merging/splitting, so every
stored clause has exactly one directive and the director's history order matches the
scene-plan order the hard gate validates (resolving before merging orphaned clauses). A rolling 5-shot history guarantees the same `(subject_key, treatment)` pair
never repeats inside the window — full ladder is exhausted before rotation. Openers, closers
and voice-attribution moments are resolved from real chapter/clause text; **empty placeholder
beats no longer exist**.

Template variety is tracked **per subject key**, not globally: one topic must exhaust its
category ladder before reusing a template, while a different topic (e.g. space science) always
gets its most fitting template (orbital `particle_cluster`) even if another topic recently used
that template. The immediately preceding template is still avoided when the ladder offers an
alternative, so adjacent shots never look identical across subject boundaries.

### 3. Six semantic diagram engines — `video/semantic_diagrams.py`

All pure code, animated at 24fps, obsidian HUD palette that classifies as the Stage E
`darkhud` pixel group, seeded per subject so different topics render different geometry:

1. `terrain_tectonic` — layered strata, travelling seismic scanline, shock rings, ejecta
2. `node_network` — 4×6 relationship graph with lit edges and travelling packets
3. `particle_cluster` — orbiting particle system with emergent binding lines
4. `process_flow` — five staged glyph panels with dual packet flow and marching dashes
5. `stat_dashboard` — clause-number count-up (real units preserved), bars, verb-shaped curve
   (rise / fall / cycle morphology from the clause verbs)
6. `timeline_arc` — phase arc with progress beam, pulse ring and drifting ticks

Each frame carries the spoken clause and resolved subject as on-screen chrome, so the picture
is literally labelled with what the narrator is saying. The obsidian base luma (`INK`,
vignette floor and data-dust range) is tuned so every template clears Stage D's decoded
`Y > 8.0` black-frame gate after H.264 losses (raw frame mean ≥ 11) while remaining well inside
the Stage E `darkhud` group (mean ≪ 70); a unit test locks this floor for all six templates.

### 4. Stage E subject hard gate — `audit/av_sync_qa.py::_subject_hard_gate`

Every scene-plan entry must satisfy all of the following or certification fails:

- bound to a real clause (non-empty, ≥ 12 chars, never the placeholder string);
- carries a complete `semantic_directive` with a known resolution source;
- the entry's `scene_id` and `subject` match the directive;
- monument shots with source `wonder_keyword`/`place_globe`/`citation_quote` independently
  re-match a wonder keyword **in the clause alone** — an unrelated monument can never be drawn
  for an abstract clause, regardless of chapter context;
- `anaphora_continuation` shots must actually be referential (same detector the router uses)
  and chain back to an immediately preceding directive bound to the same wonder scene; a broken
  or non-referential chain fails as `anaphora_chain_broken_or_not_referential`;
- neutral diagram shots stay on `scene_id = -1` with a known template; generic globe/quote
  shots also keep `-1`;
- no consecutive identical `(subject_key, treatment)` pair (the anti-loop invariant).

## Narration continuity (no silent scenes)

Standalone interior chapter cards created narration-free 2–2.4 s windows mid-film (the chapter
card rendered over a gap in the spoken timeline). The showrunner now sets
`card_duration_sec = 0.0` for every chapter, so no interior silent card is scheduled; chapter
identity remains visible on every shot's persistent ACT/title HUD graphic. The branded intro
and end-card bookends are tightened to 3.4 s / 4.2 s (still inside the Stage C 3–5 s / ≥4 s
contract) and are music-led by design. Narration coverage per scene is auditable from
`scene_plan.json`; `silencedetect` at −45 dB / 0.35 s reports zero narration-free gaps outside
the two branded bookends.

Failures write `failed:subject_gate` plus the reason into the plan, add a failing
`subject_hard_gate` check, cost 6 points each, and are part of the hard `passed` condition.

## Renderer wiring

- `_plan_turn_editorial_beats` now takes the stateful `SemanticShotDirector` plus chapter
  title/location and returns 4-tuples `(camera, end_sec, clause, VisualDirective)`; clause
  splits, breath-pause snapping, 2.2s minimum-shot merges and the 6.2s long-hold split are
  preserved, and split halves each receive their own fresh directive.
- `_camera_for_directive` maps modalities to render families; the new
  `semantic_diagram` family flows through `compositor.render_semantic_diagram` →
  `PlateLayout(anim_mode="semantic_diagram")` → the rawvideo pipe via
  `iter_semantic_diagram_frames`.
- `ScenePlanEntry.semantic_directive` persists the full verdict (including the exact
  `clause || context` hint used) into `scene_plan.json` for audit.
- Generic-topic captions never name a monument (`_directive_caption`).

## Engine-level subject enforcement

The animators historically re-resolved the wonder scene from text hints inside each frame
iterator (`_resolve_scene` with rotating fallbacks), so a clause merged across subjects could
draw the wrong model — observed as a Khufu engineering panel inside a Pharos LiDAR shot. The
director's verdict is now physically enforced on the rendered pixels:

- `PlateLayout.directive_scene_id` carries the bound scene id onto every segment task;
- `iter_broll_depth_frames`, `iter_archival_desk_frames`, `iter_telemetry_split_frames` and
  `iter_kinetic_quote_frames` all accept `scene_id_override` and draw exactly that scene when
  set (a non-negative id), bypassing legacy hint resolution;
- `PlateLayout.semantic_treatment` drives `force_cad` on the archival engine so a `lidar_scan`
  directive renders the CAD point cloud and a `physics_simulation` directive renders the sim,
  instead of the old even/odd segment-index parity;
- semantic diagrams are directive-native by construction (template + subject seed come
  straight from the directive, no monument engine is ever consulted).

Static fallback plates still resolve through the legacy resolver; they are only used by the
non-pipe marathon-CPU path, never by the standard rawvideo production pipe whose animated
frames Stage E audits.

## Lexicon corrections

- `theamata` (the generic Greek term for the wonders *list*) was removed from the Pharos
  keyword rule — it routed every abstract "origins of the seven wonders" clause to the
  lighthouse model even though the word names no monument.
- The Giza rule's misspelled `hemunu` is corrected to `hemiunu` (old spelling retained as an
  alias) and `akhet` (Akhet Khufu) is added.

## Test coverage

- `tests/test_semantic_router.py` — no-fallback matcher, arbitrary topics never render
  monuments, treatment-window no-repeat, globe/quote routing, attribution binding, ladder
  rotation.
- `tests/test_semantic_diagrams.py` — per-template motion energy, darkness/determinism,
  subject-seed geometry differences, and the classifier regression guard (all six templates
  classify as `darkhud`).
- `tests/test_av_sync_plan.py` — directive→camera mapping, clause-bound planner beats,
  arbitrary-topic diagram planning, and hard-gate accept/block cases including placeholder,
  wrong-subject monument and consecutive-repeat shots.
