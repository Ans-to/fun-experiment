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

Every frame across the film is generated algorithmically in code via `@numba.njit(nogil=True, fastmath=True)` LLVM/AVX2 3D & physics kernels:
1. **`Monument3DEngine` (Z-Buffered 3D Architectural Reconstructions)**: Multi-thousand-facet 3D architectural reconstructions of the Ancient & New Wonders (Khufu & Khafre with internal King's Chamber/Grand Gallery X-ray, Pharos Lighthouse with rotating volumetric searchlight, Jinshanling Great Wall mountain ridge, Petra's Al-Khazneh in the Siq, Colosseum & subterranean Hypogeum, Machu Picchu & Huayna Picchu terraces, Chichén Itzá's 91-step El Castillo, Taj Mahal with lathe-revolved Mughal onion dome & planar Charbagh water reflections, and Christ the Redeemer on Corcovado) overlaid with 28,000 Z-tested surface LiDAR points and 3D azimuth compass rings.
2. **`PhysicsSimulationEngine` (Live 24fps Engineering & Mathematical Simulations)**: *3Blue1Brown / Sebastian Lague*-style live physical simulations bespoke to each monument — Khufu's `51.84°` Seked `5½` & `0.05°` circumpolar meridian transit, Ming `700 km/h` optical beacon relay, Petra's `45,000,000 L/day` subcritical hydraulic siphon particles, Colosseum's `50,000`-spectator radial vomitoria crowd evacuation & decay curve, Machu Picchu's 4-layer hydrological percolation & mortarless seismic ashlar damping, Chichén Itzá's Bragg acoustic diffraction wavefronts & `1,400 → 900 Hz` Quetzal chirp waveform, Taj Mahal's double-shell dome + ebony river caissons + `2°` outward minaret lean, and Corcovado's `28 m` cantilever truss & lightning Faraday grounding.
3. **`ProceduralCadCache` (Volumetric 3D LiDAR Point-Cloud & X-Ray Blueprints)**: Rotating 28,000-vertex 3D surface LiDAR point clouds with sweeping laser cross-sections and glowing structural wireframes.
4. **`raycast_orbital_earth_globe` (True 3D Raycasted Curved-Horizon Planet Earth in Deep Space)**: Right-handed 3D ECEF spherical Earth raycaster with Rayleigh atmospheric limb scattering, 1:50m Natural Earth vector continents/rivers/lakes (`natural_earth_50m.json.gz`), sovereign amber-gold illumination, and 3D elevated great-circle SLERP flight arcs.

---

## Broadcast Narration, Chamber Score & Closed-Loop QA

- **32-Bit `FP32 Kokoro-82M v1.0` Neural Voiceover + Geodesic SLERP Voice Blending**: Alternates seamlessly between **Alex Vance** (*Lead Narrator & Cultural Historian*) and **Dr. Maya Lin** (*Field Archaeologist & Structural Specialist*) with spoken-ear rewriting and zero podcast banter (**Stage A QA**).
- **5-Stage Vocal Console + Bowed Chamber Score + `-16.0 LUFS` Mastering**: Butterworth HPF → mud cut → optical compression → dynamic de-esser → air shelf → sidechain-ducked chamber strings/piano → two-pass FFmpeg ITU-R BS.1770-4 / EBU R128 loudness mastering (**Stage B QA**).
- **PyAV Stream & Motion Liveness Inspection**: Verifies Rec.709 H.264 `yuv420p` + `48 kHz` stereo AAC sync, continuous 24fps motion liveness across all acts, `100%` verbatim subtitle word fidelity (`captions.srt` / `captions.vtt`), and clickable MP4 chapters (**Stage C & D QA**).

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
