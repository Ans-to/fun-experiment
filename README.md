# VoxForge — 100% Pure-Code 3D, Physics, LiDAR & Orbital Earth Documentary Film Engine

**VoxForge** turns a single natural-language command into a **studio-certified 1080p24 cinema documentary film** and **ITU-R BS.1770-4 / EBU R128-mastered stereo audio chronicle** — with **100% algorithmic code-synthesized visuals (zero stock photos, zero static AI images, zero talking-head avatars)**, **zero API keys**, and **closed-loop broadcast quality auditing**.

```bash
./documentary "build a 5 minute documentary on 7 wonders of the world"
```

### Rendered Pure-Code Cinema Documentary Showcase (`7 Wonders of the World: Architecture of the Impossible` — `99.8/100` QA Certified)
- **Download Rendered Documentary Film (`.mp4`, 1080p24 Rec.709, `~46 MB`)**: [`output_episodes/7-wonders-of-the-world-5m/documentary_film.mp4`](output_episodes/7-wonders-of-the-world-5m/documentary_film.mp4)
- **Download Mastered Narration & Chamber Score (`.mp3`, `-15.99 LUFS`)**: [`output_episodes/7-wonders-of-the-world-5m/documentary_audio.mp3`](output_episodes/7-wonders-of-the-world-5m/documentary_audio.mp3)
- **Download Captions (`.srt` / `.vtt`)**: [`captions.srt`](output_episodes/7-wonders-of-the-world-5m/captions.srt) · [`captions.vtt`](output_episodes/7-wonders-of-the-world-5m/captions.vtt)
- **15-Element Cinema Documentary QC Audit Report**: [`audit_report.md`](output_episodes/7-wonders-of-the-world-5m/audit_report.md)

---

## 4-Pillar Pure-Code Visual Engine (`voxforge/video/procedural_world.py`)

Every frame across the film is generated algorithmically in code via `@numba.njit(nogil=True, fastmath=True)` LLVM/AVX2 3D & physics kernels, finished through a cinematic optical chain: **aerial-perspective depth fog**, an analytic **twilight sun disc** with screen-space **volumetric crepuscular god-ray spokes**, **anamorphic cyan lens-flare streaks** on every emissive beacon, **filmic halation** bloom, organic **film-gate weave**, archival grain and a Rec.709 print LUT:
1. **`Monument3DEngine` (Z-Buffered 3D Architectural Reconstructions)**: Multi-thousand-facet 3D architectural reconstructions of the Ancient & New Wonders (Khufu & Khafre with internal King's Chamber/Grand Gallery X-ray, Pharos Lighthouse with rotating volumetric searchlight, Jinshanling Great Wall mountain ridge, Petra's Al-Khazneh in the Siq, Colosseum & subterranean Hypogeum, Machu Picchu & Huayna Picchu terraces, Chichén Itzá's 91-step El Castillo, Taj Mahal with lathe-revolved Mughal onion dome & planar Charbagh water reflections, and Christ the Redeemer on Corcovado) overlaid with 28,000 Z-tested surface LiDAR points and 3D azimuth compass rings.
2. **`PhysicsSimulationEngine` (Live 24fps Engineering & Mathematical Simulations)**: *3Blue1Brown / Sebastian Lague*-style live physical simulations bespoke to each monument — Khufu's `51.84°` Seked `5½` & `0.05°` circumpolar meridian transit, Ming `700 km/h` optical beacon relay, Petra's `45,000,000 L/day` subcritical hydraulic siphon particles, Colosseum's `50,000`-spectator radial vomitoria crowd evacuation & decay curve, Machu Picchu's 4-layer hydrological percolation & mortarless seismic ashlar damping, Chichén Itzá's Bragg acoustic diffraction wavefronts & `1,400 → 900 Hz` Quetzal chirp waveform, Taj Mahal's double-shell dome + ebony river caissons + `2°` outward minaret lean, and Corcovado's `28 m` cantilever truss & lightning Faraday grounding.
3. **`ProceduralCadCache` (Volumetric 3D LiDAR Point-Cloud & X-Ray Blueprints)**: Rotating 28,000-vertex 3D surface LiDAR point clouds with sweeping laser cross-sections and glowing structural wireframes.
4. **`raycast_orbital_earth_globe` (True 3D Raycasted Curved-Horizon Planet Earth in Deep Space)**: Right-handed 3D ECEF spherical Earth raycaster with Rayleigh atmospheric limb scattering, 1:50m Natural Earth vector continents/rivers/lakes (`natural_earth_50m.json.gz`), sovereign amber-gold illumination, and 3D elevated great-circle SLERP flight arcs.

---

## Broadcast Narration, Chamber Score & Closed-Loop QA

- **32-Bit `FP32 Kokoro-82M v1.0` Neural Voiceover + Geodesic SLERP Voice Blending**: Alternates seamlessly between **Alex Vance** (*Lead Narrator & Cultural Historian*) and **Dr. Maya Lin** (*Field Archaeologist & Structural Specialist*) with spoken-ear rewriting and zero podcast banter (**Stage A QA**).
- **5-Stage Vocal Console + Bowed Chamber Score + `-16.0 LUFS` Mastering**: Butterworth HPF → mud cut → optical compression → dynamic de-esser → air shelf → sidechain-ducked chamber strings/piano → two-pass FFmpeg ITU-R BS.1770-4 / EBU R128 loudness mastering (**Stage B QA**).
- **PyAV Stream & Motion Liveness Inspection**: Verifies Rec.709 H.264 `yuv420p` + `48 kHz` stereo AAC sync, continuous 24fps motion liveness across all acts, `100%` verbatim subtitle word fidelity (`captions.srt` / `captions.vtt`), and clickable MP4 chapters (**Stage C & D QA**).
- **Narration-Aligned Scene Plan + Stage E Semantic A/V Sync Gate**: Every rendered scene is bound to its exact spoken clause in `scene_plan.json` (clause text, audio-derived start/end, subject, semantic intent, required visual, pure-code asset source, relevance criteria, validation status). After the mux, Stage E **decodes the final MP4** and validates every scene with pixel-evidence frame classification (globe = deep ocean-blue disc, twilight monument = warm stone mass + sky gradient, sim/telemetry = near-black HUD glyphs, quote = large white typography), dense probes 0.2s after each cut-in, black/frozen-frame detection, A/V drift and caption-presence checks — any unresolved semantic mismatch **blocks studio certification** (**Stage E QA**).

---

## Narration↔Visual Sync Certification (Stage E)

```bash
# Build any film — scene_plan.json + Stage E verdicts are produced automatically
./documentary "build a 5 minute documentary on 7 wonders of the world"

# Re-run the 10-niche certification matrix (evidence sheets land in qa_evidence/)
.venv/bin/python tools/run_niche_matrix.py /tmp/qa_niches

# Clause planner, scene-plan model, subject binding & sync-gate unit tests
.venv/bin/python -m pytest tests/test_av_sync_plan.py tests/test_narration_visual_binding.py -q

# Video-free narration→subject binding audit for any rendered episode
.venv/bin/python tools/audit_narration_visual_binding.py output_episodes/7-wonders-of-the-world-5m
```

### Clause-level subject binding (mismatch fix)

Mixed-subject acts no longer inherit one chapter-level monument. Each `DialogueTurn`
stores an explicit `visual_scene_id` / `visual_subject`; each rendered beat derives an
isolated clause-scoped copy of the original turn and passes that scene ID through the
compositor into the 24 fps animator. A weighted resolver prioritizes proper nouns in the
spoken clause over incidental geography in chapter context, and Stage E adds a
`narration_subject_match` check that blocks certification when an exact clause resolves
to a different scene than the plan records. Unmapped topics now render a neutral 3D data
sculpture plus the narration-derived explanatory simulation instead of a classical
sanctuary that depicts the wrong subject.

---

## Stage A Editorial Certification Matrix (offline-runnable, 10/10)

The film-level matrix above needs network access for the neural TTS voices. The **editorial**
half of the pipeline is dependency-free, so it has its own reproducible certification matrix
that runs anywhere (`qa_evidence/stage_a_matrix.md` + `.json` are the evidence sheets):

```bash
.venv/bin/python tools/run_stage_a_matrix.py                  # all 10 niches (~22 s)
.venv/bin/python tools/run_stage_a_matrix.py --only 03_history,07_cooking
```

Current verdict: **10/10 niches certified**, healing **88.0–90.7 → 94.9–100.0**, with every
one of the six editorial checks (E1 word budget, E2 lexical freshness, E3 cliché purity,
E4 host balance, E5 cutaway rhythm, E6 spoken-ear orality) passing.

### Word-weighted host balancing (E4)

E4 measures each host's share of **spoken words** (36%–64%), not their share of turns.
Balancing by turn count alone left Host A holding ~67% of the word budget on 9 of 10 niches,
because the longest analytical turns all landed on the opening speaker. Stage A now keeps the
conversational A/B alternation and then swaps speaker labels on the minimum set of turn pairs
whose word-count difference cancels the remaining imbalance (never creating a same-speaker run
of three, never touching a scripted self-introduction), healing every niche to 44%–56%.
`tests/test_editorial_audit.py::test_host_speaker_balance_is_word_weighted_not_turn_weighted`
locks this in — it fails at 68.1% against the previous implementation.

### Bounded offline corpus (known limit)

With no LLM attached the showrunner writes from a **finite** corpus, and E2 forbids repeating
6-grams, so an arbitrary topic cannot be stretched to an arbitrary runtime — a 1-hour request
for an unpacked topic heals to ~1.4k of 7.2k words. The pipeline now emits an explicit warning
(`[WARNING] Offline corpus exhausted: …`) instead of quietly shipping a film 20–80% short.
Attach a local LLM (Ollama or a GGUF model) for unlimited rewriting, or stay within the
duration the corpus supports. Curated packs (7 Wonders) legitimately fill an hour.

---

## Quick Start

```bash
# Render a 5-minute or 1-hour pure-code 3D documentary film from a single command
./documentary "build a 5 minute documentary on 7 wonders of the world"
./documentary "build a 1 hour documentary on 7 wonders of the world"

# Launch the Web Documentary Screening & Control Room
./documentary studio --host 0.0.0.0 --port 8000

# Run the automated test suite (verifies zero stock/seed photos + all 4 code-render engines)
.venv/bin/pytest -v
```

### Test ANY topic/script on Google Colab (free CPU, zero setup)

Open [`notebooks/VoxForge_Google_Colab.ipynb`](notebooks/VoxForge_Google_Colab.ipynb), edit the `TOPIC` cell to your own command (any subject, 90 s → 3 h), and run top-to-bottom: it clones this repo, installs the engine, produces the film on the Colab CPU (~45–60 min for a 5-minute film), previews it inline, and downloads the MP4 / MP3 / SRT / QC report to your machine (optional Google Drive backup cell included).
