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

# Clause planner, scene-plan model, frame classifier & sync-gate unit tests
.venv/bin/python -m pytest tests/test_av_sync_plan.py -q
```

---

## Render Quality Tuning (Pure-Code Optical Finishing Chain)

Every plate passes an edge-adaptive optical finish (Lanczos plate upscale + halo-controlled
edge sharpening that leaves skies/starfields/HUD gradients untouched) and a quality-first
H.264 encode (CRF 22, variance-AQ, tuned 3 Mbps VBV window — measured SSIM 0.78→0.91 vs. the
legacy 840k-capped encode, zero Stage E frame-family flips; see
[`docs/VIDEO_FINISHING_AND_ENCODE_UPGRADE.md`](docs/VIDEO_FINISHING_AND_ENCODE_UPGRADE.md)).
Supersampling is CPU-opt-in so free Colab renders stay fast:

| Environment variable | Default | Effect |
|---|---|---|
| `VOXFORGE_RENDER_SCALE` | `0.5` | Monument facet rasterizer internal scale (0.35–1.0); `1.0` = native 1080p SSAA anti-aliasing |
| `VOXFORGE_GLOBE_SCALE` | `0.75` | Orbital Earth raycast scale vs. 1080p (0.5–1.0); 0.75 anti-aliases GIS vector coastlines |
| `VOXFORGE_SHARPEN_AMOUNT` | `0.55` | Edge-adaptive sharpen strength; `0` disables the pass |
| `VOXFORGE_ENCODE_PROFILE` | `quality` | H.264 quality/size tier: `quality` (~4 Mbps, SSIM 0.91), `balanced` (~3 Mbps), `compact` (~2 Mbps for size-budgeted outputs) |

```bash
# Maximum-fidelity offline render (slower, fully anti-aliased geometry)
VOXFORGE_RENDER_SCALE=1.0 ./documentary "build a 5 minute documentary on 7 wonders of the world"
```

Evidence, measured A/B metrics and contact sheets: [`docs/VIDEO_FINISHING_AND_ENCODE_UPGRADE.md`](docs/VIDEO_FINISHING_AND_ENCODE_UPGRADE.md)
and `qa_evidence/quality_stills/` (reproduce with `tools/render_quality_compare.py` /
`tools/globe_quality_compare.py`).

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
