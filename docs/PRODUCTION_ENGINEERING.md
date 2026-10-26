# Broadcast Production Engineering & Closed-Loop QA Architecture

VoxForge Studio is engineered from professional broadcast audio, documentary editing, and streaming video specifications rather than naive slide-concatenation scripts. This document details the broadcast standards and closed-loop **Create → Audit → Improve** algorithms implemented across `voxforge/`.

---

## 1. Broadcast Audio Engineering (`voxforge/audio/`)

### Vocal Signal Chain
Every synthesized host turn (`HOST_A`: Alex Vance, warm baritone; `HOST_B`: Dr. Maya Lin, crisp mezzo-soprano) passes through a dedicated 48 kHz 32-bit float studio channel strip before hitting the mix bus:
1. **Subsonic High-Pass Filter (`80 Hz`, 2nd-order Butterworth)**: Eliminates plosive rumble and DC offset.
2. **Low-Mid Mud Cut (`-1.8 dB` @ `260 Hz`) + Intelligibility Presence Lift (`+2.0 dB` @ `3.4 kHz`)**: Prevents vocal masking over ambient musical pads and sharpens consonant articulation.
3. **Soft-Knee Broadcast Leveler (`-18 dBFS` threshold, `2.8:1` ratio)**: Smooths syllable-to-syllable volume swings without pumping artifacts.
4. **Conversational Stereo Stage Placement**: Constant-power panning places Alex at `-0.14` (subtle left) and Maya at `+0.14` (subtle right), matching their physical left/right seating in the 180-degree studio camera layout while remaining 100% mono-compatible on smartphone speakers.

### Procedural Musical Score, Sidechain Ducking & Room Tone
- **Chapter Bumpers (`score_synth.py`)**: Synthesized harmonic fifth/ninth swells (`D2`, `C2`, `A1`, `F2` tonal centers matched to act mood) play under chapter title cards with a `0.55 s` L-cut overlap beneath the chapter's opening spoken line.
- **Lookahead Sidechain Ducking (`-11.5 dB`)**: A `480 Hz` speech-activity envelope with `60 ms` lookahead attack, `180 ms` hold, and `220 ms` Hanning-window release ducks the ambient underscore cleanly whenever either host speaks.
- **Acoustic Studio Room Tone (`-53 dBFS`)**: Continuous low-passed (`420 Hz`) stereo room floor prevents jarring digital-zero (`-inf dBFS`) dropouts between turns.
- **Constant-RAM `memmap` Mixdown**: Multi-hour 48 kHz stereo mixes stream through temporary disk-backed `float32` `numpy.memmap` buffers in 60-second windows, keeping peak Python RAM under `~90 MB` even on 1–3 hour marathons.
- **Two-Pass ITU-R BS.1770-4 / EBU R128 Mastering**:
  - **Pass 1**: Measures integrated loudness (`input_i`), oversampled true peak (`input_tp`), loudness range (`input_lra`), and gate threshold (`input_thresh`) via FFmpeg `loudnorm=print_format=json`.
  - **Pass 2**: Applies linear loudness normalization (`linear=true`) targeting **`-16.0 LUFS`** integrated loudness followed by a brickwall lookahead true-peak ceiling (`alimiter`) guaranteeing **`≤ -1.0 dBTP`**.

---

## 2. Multicam Cinematography & Retention Editing (`voxforge/video/`)

### 180-Degree Studio Axis & 5-Camera Switching Grammar
Viewer retention drops sharply on static single-image "audiograms." VoxForge enforces a 5-camera switching grammar (`showrunner.py` + `compositor.py`):
1. **`chapter_card`**: Full-screen darkened monument plate with Ken Burns push, act counter, era/location kicker, and fade transitions.
2. **`wide_studio`**: Two-shot broadcast studio desk where Alex (stage left, eyeline right) and Maya (stage right, eyeline left) face one another across a consistent 180-degree axis. The active speaker's frame illuminates with an accent border, `"SPEAKING"` tally badge, and real-time 24fps voice oscilloscope (`showwaves`), while the listening co-host remains softly dimmed (`0.58` brightness).
3. **`closeup_host_a` / `closeup_host_b`**: Single-speaker close-up on one side paired with a live `EVIDENCE MONITOR` panel displaying monument photography and quantitative architectural metrics on the opposite side.
4. **`broll_cinema`**: Full-bleed `1920×1080` cinema-graded documentary plate with top-left archival source attribution pill and top-right Picture-in-Picture speaker waveform badge.
5. **`split_fact_card`**: Dual-panel forensic view pairing high-resolution site photography and speaker telemetry on the left with a 4-card structural metric grid on the right.

### Zero-Drift Parallel Segment Architecture
Concatenating variable-length video clips can accumulate multi-second A/V sync drift over a 1-hour timeline if segment durations round independently. VoxForge eliminates drift structurally:
1. Master audio (`master_48k.wav`) is locked **first**, establishing exact sample-accurate timestamps (`start_sec`, `end_sec`) for every chapter and turn.
2. Segment frame counts are allocated against cumulative integer frame indices (`target_total_frames = round(target_end_sec * 24); seg_frames = target_total_frames - accumulated_frames`), guaranteeing `sum(seg_frames) / 24.0 == total_duration_sec` to within `< 1 frame` (`41.6 ms`) across an entire hour.
3. Segments encode concurrently across worker threads with pinned `-r 24 -pix_fmt yuv420p -video_track_timescale 12288`, then stream-copy (`-c:v copy`) alongside `master_48k.wav` and `FFMETADATA1` chapter markers with `+faststart`.

---

## 3. Closed-Loop 4-Stage Broadcast QA & Self-Healing (`voxforge/audit/`)

Every episode must pass 19 quantitative checks across four domains (`score ≥ 85.0` and zero `FAIL` checks for `STUDIO CERTIFIED` status):

| Stage | Checks | Automated Self-Healing Actions |
|---|---|---|
| **Stage A · Editorial QA** (`editorial.py`) | `E1` Word/Duration Budget (`±10%`), `E2` 6-Gram Lexical Diversity (`≤ 2.5%` overlap), `E3` Zero AI Clichés (`28` banned tropes), `E4` Host Split (`36–64%`) & Turn Length (`≤ 64` words), `E5` Cutaway Rhythm (`≥ 22%` B-roll/Fact, zero 3-turn static runs) | Regex cliché surgery, clause-boundary splitting of monologues into natural two-host exchanges, non-overlapping archival fact expansion or micro-briefing trimming, word-weighted speaker re-assignment (E4 is measured on spoken **words**, so healed scripts are driven to a 44%-56% split rather than a 50/50 turn count), and multicam cutaway injection |
| **Stage B · Audio QA** (`audio_qa.py`) | `A1` Neural Utterance Integrity (`1.1–5.2` words/s, zero clipping/dropouts), `A2` Integrated Loudness (`-16.0 ± 0.85 LUFS`), `A3` True Peak (`≤ -1.0 dBTP`), `A4` Loudness Range (`1.5–11.5 LU`), `A5` Dead-Air Continuity (`≤ 1.65 s`), `A6` Runtime Lock (`±12%`) | Re-synthesizes anomalous turns, adjusts prosody `speed_scale` (`0.88×–1.14×`) when duration drifts outside tolerance, and re-runs two-pass linear `loudnorm` + true-peak limiting |
| **Stage C · Visual QA** (`visual_qa.py`) | `V1` `1920×1080` Resolution Compliance, `V2` Optical Sharpness (`Laplacian variance ≥ 32.0`), `V3` Broadcast IRE Luminance (`Y ∈ [22, 228]`), `V4` Camera Cut Cadence (`3.5–22.0 s` avg shot) | Replaces blurry, dark, or blown-out images with crisp 1080p procedural architectural blueprint plates before encoding |
| **Stage D · Deliverable QC** (`deliverable_qa.py`) | `D1` Stream Spec (`h264 yuv420p` + `aac 48kHz 2ch`), `D2` A/V Sync Lock (`≤ 350 ms` drift over 1 hour), `D3` PyAV Multi-Point Frame Decode (`Y > 8.0`, live motion `> 0.02`), `D4` Embedded MP4 Chapters + `.srt`/`.vtt` | Triggers lossless stream re-mux with explicit audio resampling and `+faststart` metadata repair if container drift is detected |
