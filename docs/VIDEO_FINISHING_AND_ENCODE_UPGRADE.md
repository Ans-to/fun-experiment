# VoxForge: video finishing and encode quality upgrade

**Scope:** picture fidelity of the four pure-code render pillars (3D monument engine, physics
simulation, LiDAR/CAD turntable, orbital Earth globe) and the H.264 delivery encode. Every change
is measured against real engine output, keeps the 1080p24 Rec.709 / yuv420p delivery spec, and
preserves the Stage E frame-family classifier contract (`monument` / `geo` / `darkhud` / `quote`).

## Evidence reviewed

- The shipped `documentary_film.mp4` averages only **~1.16 Mbps** (median packet ≈ 61 kbit at
  24 fps; p95 ≈ 2.3 Mbps) despite CRF 21, because every segment was capped by
  `-maxrate 840k -bufsize 1680k`. Complex monument/sim frames were therefore VBV-starved into
  macroblocking regardless of the CRF target.
- The monument rasterizer renders solid facets at **960×540 then bicubic-upscales to 1080p**,
  while LiDAR points and CAD edges are drawn at native 1080p. The upscale leaves silhouette and
  masonry edges soft relative to the overlay work.
- The globe raycast ran at 1280×720 with a bicubic upscale; hard 1:50m Natural Earth vector
  borders therefore arrived at the delivery raster soft.
- The god-ray pass ran **18 full-1080p resizes plus float round-trips per frame inside an
  18-iteration loop**, pure overhead with no visual benefit.

## Changes in this pass

### 1. Edge-adaptive optical finishing (`voxforge/video/finish.py`, new module)

- `edge_adaptive_sharpen`: halo-controlled unsharp mask gated by a Rec.709-luma Sobel edge map.
  Flat twilight skies, starfields and obsidian HUD gradients carry no edge energy and are left
  bit-identical; geometry edges get a clamped, spatially smoothed detail boost. Grain is applied
  later in `_finish_cinema_frame`, so grain is never sharpened.
- `upscale_plate`: Lanczos-4 for upscaling (tighter than bicubic), area averaging for SSAA
  downscaling (true low-pass, no aliasing).

Measured on real monument frames (Laplacian variance, higher = more resolved detail):

| Scene | Legacy 0.5× + bicubic | 0.5× + new sharpen | 0.75× SSAA + sharpen | Native 1.0× raw |
|---|---:|---:|---:|---:|
| Giza | 4,158 | 7,290 (+75%) | 9,836 (+137%) | 7,058 |
| Taj Mahal | 3,843 | 5,928 (+54%) | 7,552 (+96%) | 5,905 |
| Colosseum | 3,703 | 6,154 (+66%) | 8,489 (+129%) | 6,663 |

Mean per-channel color drift from sharpening stays under 0.6/255; contact sheets under
`qa_evidence/quality_stills/` show no halo rings on the cube/wireframe and marble close-ups.

### 2. Configurable supersampling (`VOXFORGE_RENDER_SCALE`, `VOXFORGE_GLOBE_SCALE`)

- `Monument3DEngine(internal_scale=..., sharpen=...)` replaces the hard-coded half-res path.
  Default stays **0.5** so render time is unchanged on free Colab CPUs; setting
  `VOXFORGE_RENDER_SCALE=0.75` (or `1.0`) supersamples the facet rasterizer before the
  area-averaged downscale, anti-aliasing silhouette edges. Star density is normalized across
  internal resolutions.
- Globe default moves from 1280×720 to **1440×810 (0.75)**. Measurements showed native 1080p
  raycast edges staircase on the GIS vector coastlines (aliasing score 63.4 vs 36.3 at 0.75),
  while 0.75 downscaled cleanly and still lifted Laplacian detail from 323 (legacy raw) to
  ~600–725 with sharpening, at only ~12 ms extra per frame (~58 ms vs ~46 ms; monument frames
  cost 200–900 ms by comparison). Globe sharpening is capped at 0.35 to protect vector borders.

### 3. Quality-first CG rate control (`voxforge/video/renderer.py`)

Tuned with a controlled experiment on real engine frames (`tools/encode_tuning.py`: 108 frames
spanning monument cinema, LiDAR CAD and physics-sim families, PSNR/SSIM against a CRF 12
reference, Stage E family stability):

| Setting | Mean bitrate | PSNR | SSIM | Stage E flips |
|---|---:|---:|---:|---:|
| Legacy (CRF 21, cap 840k, aq1) | 1.16 Mbps | 26.6 | 0.776 | **1** (darkhud→monument from block noise) |
| CRF 23, cap 2500k, veryfast, aq3 | 2.94 Mbps | 29.2 | 0.876 | 0 |
| **CRF 22, cap 3000k, superfast, aq3 (new standard)** | **4.11 Mbps** | **31.4** | **0.911** | **0** |
| CRF 21, cap 4000k, superfast, aq3 | 5.49 Mbps | 32.5 | 0.928 | 0 |

- Standard renders now use **CRF 22 with a `3000k / 6000k` VBV window** — the cap, not the CRF,
  was the binding quality constraint; marathon (>15 min, ≤4-core) renders use CRF 23 with
  `2000k / 4000k`.
- Added `-x264-params aq-mode=3:aq-strength=0.8` (variance adaptive quantization) to protect
  flat twilight skies and dark HUD gradients from banding and macroblocking.
- Segment concat/copy mux, GOP, color tags and all stream-level QA requirements are unchanged.
- Because bitrate now follows content instead of an 840k ceiling, rendered files are larger;
  the committed showcase MP4 in `output_episodes/` is left as-is, and fresh local renders carry
  the upgraded picture. `VOXFORGE_ENCODE_PROFILE` selects the quality/size tradeoff:

| Profile (standard / marathon tier) | CRF | VBV maxrate | Measured mean bitrate |
|---|---|---|---|
| `quality` (default) | 22 / 23 | 3000k / 2000k | ~4.1 Mbps, SSIM 0.911 |
| `balanced` | 23 / 24 | 2500k / 1700k | ~2.9 Mbps, SSIM 0.876 |
| `compact` (size-budgeted, e.g. Git-tracked showcases) | 24 / 25 | 1600k / 1200k | ~2 Mbps, still above the legacy 1.16 Mbps SSIM 0.776 baseline |

### 4. God-ray overhead removed

The radial-blur accumulator now stays at quarter resolution for all 18 samples; the single
full-resolution resize and RGB blend happen once after the loop instead of 18 times per frame.
Because the per-sample contributions are additive, fusing them into one blend is numerically
equivalent except where intermediate uint8 clamping previously saturated (rare on dark twilight
plates); the Stage E suite confirms the frame-family labels and motion liveness are unchanged.

## Tuning knobs (environment variables)

| Variable | Default | Range | Effect |
|---|---|---|---|
| `VOXFORGE_RENDER_SCALE` | `0.5` | 0.35–1.0 | Monument facet rasterizer internal scale; higher = SSAA anti-aliasing, more CPU |
| `VOXFORGE_GLOBE_SCALE` | `0.75` | 0.5–1.0 | Orbital globe raycast scale relative to 1080p |
| `VOXFORGE_SHARPEN_AMOUNT` | `0.55` | 0.0–1.2 | Edge-adaptive sharpen strength; `0` disables the pass |
| `VOXFORGE_ENCODE_PROFILE` | `quality` | quality/balanced/compact | H.264 quality/size tier (CRF + VBV presets, see §3) |

## Validation

- New unit tests in `tests/test_video_finish.py` (5 tests): edge recovery vs. flat-region
  invariance, halo clamp, degenerate input, env parsing, supersampling detail gain, color-drift
  bound and Stage E family safety.
- Existing Stage E classifier still labels the finished globe `geo` and finished monuments
  `monument` on decoded BGR frames (verified by script and the full test suite).
- Reproduce the evidence:
  `PYTHONPATH=. .venv/bin/python tools/render_quality_compare.py` and
  `... tools/globe_quality_compare.py`; contact sheets land in `qa_evidence/quality_stills/`.

### 5. Stage E quote-card detection made layout-aware (`voxforge/audit/av_sync_qa.py`)

The first end-to-end validation run exposed a **pre-existing latent bug** (independent of the
finishing/encode changes — proven by identical white-pixel fractions with sharpening off and
with the old x264 settings): the quote classifier required >4.5% *whole-frame* neutral-white
pixels, which a short 2–3 line inscription physically cannot reach (measured 0.7% even at 92px
type), so the Pharos quote scene was mislabeled `darkhud` and blocked certification.

The detector now also measures neutral-white density inside the central composition band where
the kinetic quote block actually lives. Measured separation: quote band 0.019 vs. telemetry 0.007
vs. physics HUD 0.005; bright stone is still rejected by the `mean_lum < 75` dark-frame gate and
amber/cyan HUD glyphs fail the neutral-white mask. Rule:

```
mean_lum < 75  AND  (global white_frac > 0.045  OR  central-band white_frac > 0.012)  -> quote
```

Re-auditing the already-rendered film with the new classifier restored Stage E to **20/20
semantic matches, score 100.0, zero human-review queue**, with every other family label
unchanged. Regression coverage in `tests/test_av_sync_plan.py::test_short_quote_card_detected_via_central_band`.

## Limits

Laplacian variance and aliasing proxies cannot judge aesthetics; the contact sheets should be
eyeballed on any future parameter change. Default monument scale remains half-resolution for
CPU budget reasons — maximum-fidelity offline renders should opt into `VOXFORGE_RENDER_SCALE=1.0`
and accept the longer render. The encoder now produces larger segment bitstreams on complex
shots; final file size grows because the old cap was the binding quality constraint.
