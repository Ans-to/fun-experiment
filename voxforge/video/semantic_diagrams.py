"""Semantic diagram engines for arbitrary-topic documentaries.

Six visually distinct, fully animated obsidian-blackboard templates chosen by the
:class:`~voxforge.video.semantic_router.SemanticShotDirector`. Every template is seeded by the
resolved subject key so different topics produce different geometry and palettes, while the
same subject stays visually coherent across its shots. Nothing here depicts an unrelated
monument: when the narration cannot bind to a 3D reconstruction, the clause itself drives the
picture (its numbers, verbs, entities and category).

Output convention matches the rest of the video package: RGB uint8 frames (the encoder pipe is
``rgb24``), dark enough to classify as the ``darkhud`` Stage E evidence group, with continuous
24 fps motion in every template.
"""

from __future__ import annotations

import math
import re
import textwrap

import cv2
import numpy as np

from voxforge.video.infographics import load_font
from voxforge.video.semantic_router import (
    TPL_NETWORK,
    TPL_PARTICLES,
    TPL_PROCESS,
    TPL_STATS,
    TPL_TERRAIN,
    TPL_TIMELINE,
    subject_seed,
)

GOLD = (229, 169, 60)
CYAN = (56, 189, 248)
# Floor luma tuned so every template decodes above Stage D's Y>8 black-frame gate even after
# H.264 losses, while remaining well inside the Stage E darkhud group (mean luma well under 70).
INK = (12, 17, 28)
GRID = (33, 48, 70)
PANEL = (17, 26, 44)
TEXT = (238, 242, 248)
MUTED = (158, 176, 200)

# Filler words that carry no visual signal when a numberless clause needs key-term chips.
_STOPWORDS = frozenset(
    """a an and or but the of to in on at by for with from into over under through after before
    as is are was were be been being it its they them their this that these those we you your our
    his her hers their ours not no yes than then so yet nor about above below between during
    because while where when what which who whom whose how why can could would should shall will
    may might must do does did done has have had having just only over such very more most some
    any each every other another one two also still even ever never often always across along""".split()
)
_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([a-zA-Z%]*)")


def extract_key_terms(clause: str, limit: int = 4) -> list[str]:
    """Pick the most informative content words from a clause, preserving original casing."""
    seen: set[str] = set()
    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z'-]+", clause or ""):
        low = raw.lower().strip("'-")
        if len(low) < 4 or low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        terms.append(raw)
    # Prefer capitalised (named) entities, then longer, more specific words.
    terms.sort(key=lambda w: (w[0].islower(), -len(w)))
    return terms[:limit]


def stat_panel_content(clause: str, u: float) -> dict:
    """Pure, testable spec for the stat dashboard's left panel.

    Numeric clauses show their real measured value; numberless clauses get clause-derived key
    terms instead of an invented percentage, so the plate never fabricates data.
    """
    nums = _NUMBER_RE.findall(clause or "")
    nums = [(a, (b or "units").lower()) for a, b in nums if a][:3]
    if nums:
        raw, unit = nums[0]
        try:
            target = float(raw.replace(",", ""))
        except ValueError:
            target, unit = 100.0, "units"
        cur = target * min(1.0, u * 1.15)
        # Preserve the clause's own decimal precision so the plate never contradicts narration
        # ("146.6 meters" must not round to 147); integer counts stay integer with separators.
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        if unit in {"thousand", "million", "billion", "trillion"}:
            label = f"{cur:,.1f}".rstrip("0").rstrip(".") if cur < 100 else f"{cur:,.0f}"
        elif decimals:
            label = f"{cur:,.{decimals}f}"
        else:
            label = f"{cur:,.0f}"
        return {"kind": "numeric", "label": label, "unit": unit.upper()}
    return {"kind": "keywords", "terms": extract_key_terms(clause), "unit": "KEY SIGNALS"}


_TEMPLATE_KICKERS = {
    TPL_TERRAIN: "TECTONIC & TERRAIN MODEL",
    TPL_NETWORK: "RELATIONSHIP NETWORK MODEL",
    TPL_PARTICLES: "PARTICLE & SYSTEM MODEL",
    TPL_PROCESS: "STAGED PROCESS MODEL",
    TPL_STATS: "QUANTITATIVE EVIDENCE MODEL",
    TPL_TIMELINE: "SEQUENCE & TIMELINE MODEL",
}


class SemanticDiagramEngine:
    """Renders animated clause-driven diagrams for the six generic visual templates."""

    def __init__(self) -> None:
        self._f_kicker = load_font("SpaceGrotesk-Bold.ttf", 17)
        self._f_title = load_font("SpaceGrotesk-Bold.ttf", 34)
        # Size ladder lets long episode titles shrink to fit instead of running off-canvas.
        self._f_title_ladder = [load_font("SpaceGrotesk-Bold.ttf", s) for s in (34, 28, 24, 20)]
        self._f_big = load_font("SpaceGrotesk-Bold.ttf", 64)
        self._f_mid = load_font("SpaceGrotesk-Medium.ttf", 22)
        self._f_small = load_font("Inter-Medium.ttf", 17)

    # -- public entry point -------------------------------------------------

    def render_frame(
        self,
        template: str,
        subject_label: str,
        clause: str,
        u: float,
        time_s: float,
        w: int = 1920,
        h: int = 1080,
        seed: int | None = None,
    ) -> np.ndarray:
        rng_seed = seed if seed is not None else subject_seed(subject_label)
        rng = np.random.default_rng(rng_seed)
        canvas = self._base_canvas(w, h, rng)
        draw_fn = {
            TPL_TERRAIN: self._tpl_terrain,
            TPL_NETWORK: self._tpl_network,
            TPL_PARTICLES: self._tpl_particles,
            TPL_PROCESS: self._tpl_process,
            TPL_STATS: self._tpl_stats,
            TPL_TIMELINE: self._tpl_timeline,
        }.get(template, self._tpl_stats)
        draw_fn(canvas, rng, u, time_s, clause)
        # The stat dashboard renders its own key-signal panel; don't duplicate the chips strip.
        if template != TPL_STATS:
            self._draw_key_terms(canvas, extract_key_terms(clause, limit=3), u)
        self._draw_chrome(canvas, template, subject_label, clause)
        return canvas

    def _draw_key_terms(self, canvas: np.ndarray, terms: list[str], u: float) -> None:
        """Clause-derived signal chips: every plate visibly carries its own narration's content
        words, which both improves semantic relevance and structurally distinguishes same-family
        templates built for different clauses."""
        if not terms:
            return
        x, y = 120, 838
        for i, term in enumerate(terms[:3]):
            if u * 1.3 <= i * 0.18:
                break
            chip = term.upper()[:22]
            (tw, th), _bl = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            x1 = min(canvas.shape[1] - 120, x + tw + 40)
            cv2.rectangle(canvas, (x, y), (x1, y + th + 20), PANEL, -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x, y), (x1, y + th + 20), CYAN, 1, cv2.LINE_AA)
            cv2.putText(canvas, chip, (x + 20, y + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 2, cv2.LINE_AA)
            x = x1 + 18
            if x > canvas.shape[1] - 320:
                break

    # -- shared canvas & chrome ---------------------------------------------

    def _base_canvas(self, w: int, h: int, rng: np.random.Generator) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        rad = np.sqrt(((xx - w * 0.5) / 980.0) ** 2 + ((yy - h * 0.52) / 600.0) ** 2)
        vig = np.clip(1.0 - rad * 0.55, 0.44, 1.0)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[..., 0] = (INK[0] * vig).astype(np.uint8)
        canvas[..., 1] = (INK[1] * vig).astype(np.uint8)
        canvas[..., 2] = (INK[2] * vig).astype(np.uint8)
        for gx in range(0, w, 96):
            canvas[:, gx:gx + 1] = np.clip(canvas[:, gx:gx + 1].astype(np.int16) + 10, 0, 255)
        for gy in range(0, h, 96):
            canvas[gy:gy + 1, :] = np.clip(canvas[gy:gy + 1, :].astype(np.int16) + 10, 0, 255)
        # sparse deterministic "data dust" so different subjects have different texture
        for _ in range(90):
            px, py = int(rng.integers(0, w)), int(rng.integers(0, h))
            val = int(rng.integers(36, 82))
            canvas[py, px] = (val, val + 8, min(255, val + 22))
        return canvas

    def _draw_chrome(self, canvas: np.ndarray, template: str, subject_label: str, clause: str) -> None:
        from PIL import Image, ImageDraw

        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        draw.rectangle([64, 56, 72, 92], fill=GOLD)
        draw.text((88, 54), f"PURE-CODE EXPLAINER  ·  {_TEMPLATE_KICKERS.get(template, '')}", font=self._f_kicker, fill=GOLD)
        # Fit the title by measured width (the old fixed 72-char slice still overflowed long
        # episode names); wrap to at most two lines before shrinking the font size.
        max_w = canvas.shape[1] - 172
        title_lines = self._fit_title(draw, (subject_label or "").upper(), max_w)
        y = 96
        for line, font in title_lines[:2]:
            draw.text((86, y), line, font=font, fill=TEXT)
            y += int(font.size * 1.25)
        # The spoken clause is intentionally NOT repeated here: the burned caption already
        # carries it verbatim, and duplicating it collided with the caption band.
        canvas[:] = np.asarray(img)

    def _fit_title(self, draw, text: str, max_w: int) -> list[tuple[str, object]]:
        for font in self._f_title_ladder:
            for width in (48, 38, 30, 24):
                lines = textwrap.wrap(text, width=width) or [text]
                if all(draw.textlength(line, font=font) <= max_w for line in lines):
                    return [(line, font) for line in lines[:2]]
        font = self._f_title_ladder[-1]
        return [(line[:24].rstrip() + "…", font) for line in textwrap.wrap(text, width=24)[:2]]

    # -- template 1: terrain / tectonic -------------------------------------

    def _tpl_terrain(self, c: np.ndarray, rng: np.random.Generator, u: float, t: float, clause: str) -> None:
        h, w = c.shape[:2]
        phase = rng.uniform(0, math.tau, size=4)
        amps = rng.uniform(60, 150, size=4)
        base_y = np.linspace(h * 0.45, h * 0.82, 4)
        xs = np.arange(w, dtype=np.float32)
        glow = np.zeros_like(c)
        for layer in range(4):
            profile = (
                base_y[layer]
                - amps[layer]
                * (0.65 * np.sin(xs / (260 + 60 * layer) + phase[layer])
                   + 0.35 * np.sin(xs / (90 + 24 * layer) + phase[layer] * 1.7))
            )
            # Deep, saturated navy strata: deliberately outside the classifier's neutral-stone
            # band (channel spread >= 46) and dark enough to read as HUD, never monument stone.
            color = (8 + 3 * layer, 14 + 5 * layer, 40 + 12 * layer)
            pts = np.stack([xs, profile], axis=1).astype(np.int32)
            poly = np.vstack([pts, [[w, h], [0, h]]]).astype(np.int32)
            cv2.fillPoly(c, [poly], color)
            cv2.polylines(glow, [pts], False, GOLD if layer == 0 else CYAN, 2, cv2.LINE_AA)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.55, 0)
        # Travelling tectonic pulse + eruption ejecta on the front ridge.
        pulse_x = int((0.12 + 0.76 * u) * w)
        peak_y = int(base_y[0] - amps[0] * 1.4)
        # Expanding shock rings keep the front ridge visibly alive between progress steps.
        for ring in range(3):
            rr = int(20 + ((t * 90 + ring * 60) % 180))
            ring_alpha = 1.0 - (rr - 20) / 180.0
            cv2.circle(glow, (pulse_x, peak_y), rr, GOLD, max(1, int(4 * ring_alpha)), cv2.LINE_AA)
        cv2.circle(glow, (pulse_x, peak_y), 16, GOLD, -1, cv2.LINE_AA)
        # Sweeping seismic scanline crossing the whole profile.
        scan_x = int((t * 240) % w)
        cv2.line(glow, (scan_x, int(h * 0.30)), (scan_x, h), CYAN, 2, cv2.LINE_AA)
        for k in range(26):
            life = (u * 2.4 + k / 26.0) % 1.0
            ex = pulse_x + int(70 * (k % 9 - 4) * life)
            ey = peak_y - int(190 * life)
            alpha = max(0.0, 1.0 - life)
            cv2.circle(glow, (ex, ey), int(3 + 4 * alpha), GOLD, -1, cv2.LINE_AA)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.8, 0)

    # -- template 2: node network -------------------------------------------

    def _tpl_network(self, c: np.ndarray, rng: np.random.Generator, u: float, t: float, clause: str) -> None:
        rows, cols = 4, 6
        nodes: list[tuple[int, int]] = []
        for r in range(rows):
            for col in range(cols):
                nx = int(220 + col * 296 + rng.integers(-46, 46))
                ny = int(300 + r * 150 + rng.integers(-34, 34))
                nodes.append((nx, ny))
        edges = []
        for i, (nx, ny) in enumerate(nodes):
            # connect to right and lower neighbours with deterministic jitter
            for j in (i + 1, i + cols):
                if j < len(nodes) and rng.random() < 0.7:
                    edges.append((i, j))
        for i, j in edges:
            lit = (i / len(nodes)) <= min(1.0, u * 1.25)
            cv2.line(c, nodes[i], nodes[j], GOLD if lit else GRID, 2 if lit else 1, cv2.LINE_AA)
        glow = np.zeros_like(c)
        # packets travelling along edges
        for k, (i, j) in enumerate(edges):
            p = ((t * 0.35 + k * 0.137) % 1.0)
            x = int(nodes[i][0] * (1 - p) + nodes[j][0] * p)
            y = int(nodes[i][1] * (1 - p) + nodes[j][1] * p)
            cv2.circle(glow, (x, y), 4, CYAN, -1, cv2.LINE_AA)
        for idx, (nx, ny) in enumerate(nodes):
            lit = (idx / len(nodes)) <= min(1.0, u * 1.3)
            cv2.circle(c, (nx, ny), 11, GOLD if lit else GRID, -1, cv2.LINE_AA)
            if lit:
                cv2.circle(glow, (nx, ny), 16, GOLD, 2, cv2.LINE_AA)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.75, 0)

    # -- template 3: particle cluster ---------------------------------------

    def _tpl_particles(self, c: np.ndarray, rng: np.random.Generator, u: float, t: float, clause: str) -> None:
        h, w = c.shape[:2]
        cx, cy = w // 2, int(h * 0.52)
        n = 90
        radii = rng.uniform(90, 330, n)
        speeds = rng.uniform(0.25, 1.1, n)
        phases = rng.uniform(0, math.tau, n)
        tilts = rng.uniform(0.3, 1.0, n)
        pts = []
        for k in range(n):
            ang = phases[k] + t * speeds[k]
            rx = radii[k]
            ry = radii[k] * tilts[k] * 0.62
            x = int(cx + rx * math.cos(ang))
            y = int(cy + ry * math.sin(ang))
            pts.append((x, y, k))
        # binding lines between near particles (emergent structure)
        for a in range(0, n, 2):
            for b in range(a + 1, min(n, a + 6)):
                ax, ay, _ = pts[a]
                bx, by, _ = pts[b]
                if abs(ax - bx) < 170 and abs(ay - by) < 110:
                    cv2.line(c, (ax, ay), (bx, by), GRID, 1, cv2.LINE_AA)
        core_r = 26 + int(6 * math.sin(t * 2.2))
        cv2.circle(c, (cx, cy), core_r, PANEL, -1, cv2.LINE_AA)
        glow = np.zeros_like(c)
        cv2.circle(glow, (cx, cy), core_r + 14, CYAN, 2, cv2.LINE_AA)
        for x, y, k in pts:
            born = (k / n) <= min(1.0, u * 1.4)
            if not born:
                continue
            cv2.circle(c, (x, y), 4, CYAN if k % 3 else GOLD, -1, cv2.LINE_AA)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.8, 0)

    # -- template 4: staged process flow ------------------------------------

    def _tpl_process(self, c: np.ndarray, rng: np.random.Generator, u: float, t: float, clause: str) -> None:
        stages = 5
        box_w, box_h = 250, 190
        gap = (1920 - 2 * 120 - stages * box_w) // (stages - 1)
        y0 = 430
        # Seed-driven per-shot variation: glyph order and vertical rhythm must change with the
        # clause salt, otherwise two process plates render pixel-identical regardless of seed.
        glyph_order = rng.permutation(5).tolist()
        row_jitter = rng.integers(-34, 35, size=stages)
        centers = []
        for s in range(stages):
            x0 = 120 + s * (box_w + gap)
            sy = int(y0 + row_jitter[s])
            active = (s / stages) <= min(1.0, u * 1.25)
            col = GOLD if active else GRID
            cv2.rectangle(c, (x0, sy), (x0 + box_w, sy + box_h), PANEL, -1, cv2.LINE_AA)
            cv2.rectangle(c, (x0, sy), (x0 + box_w, sy + box_h), col, 2, cv2.LINE_AA)
            # distinct abstract glyph per stage, order shuffled per clause seed
            gx, gy = x0 + box_w // 2, sy + box_h // 2 - 12
            if glyph_order[s] == 0:
                cv2.circle(c, (gx, gy), 34, col, 3, cv2.LINE_AA)
            elif glyph_order[s] == 1:
                cv2.rectangle(c, (gx - 30, gy - 30), (gx + 30, gy + 30), col, 3, cv2.LINE_AA)
            elif glyph_order[s] == 2:
                cv2.polylines(c, [np.array([(gx, gy - 38), (gx + 36, gy + 28), (gx - 36, gy + 28)])], True, col, 3, cv2.LINE_AA)
            elif glyph_order[s] == 3:
                cv2.line(c, (gx - 34, gy), (gx + 34, gy), col, 3, cv2.LINE_AA)
                cv2.line(c, (gx, gy - 34), (gx, gy + 34), col, 3, cv2.LINE_AA)
            else:
                cv2.circle(c, (gx, gy), 36, col, 3, cv2.LINE_AA)
                cv2.circle(c, (gx, gy), 14, col, -1, cv2.LINE_AA)
            centers.append((x0 + box_w, sy + box_h // 2, x0 + box_w + gap, sy + box_h // 2))
        glow = np.zeros_like(c)
        for ci, (x1, y1, x2, y2) in enumerate(centers):
            cv2.line(c, (x1, y1), (x2, y2), GRID, 2, cv2.LINE_AA)
            # two packets per connector, phase-offset, keep flow visible at all times
            for off in (0.0, 0.5):
                p = (t * 0.45 + ci * 0.11 + off) % 1.0
                cv2.circle(glow, (int(x1 + (x2 - x1) * p), y1), 6, CYAN, -1, cv2.LINE_AA)
            # marching dashes underneath the packets
            dash_p = int((t * 140 + ci * 30) % 48)
            for dx in range(x1, x2, 48):
                dxx = dx + dash_p - 24
                if x1 < dxx < x2:
                    cv2.line(glow, (dxx, y1 + 16), (dxx + 18, y1 + 16), GOLD, 2, cv2.LINE_AA)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.8, 0)

    # -- template 5: stat dashboard -----------------------------------------

    def _tpl_stats(self, c: np.ndarray, rng: np.random.Generator, u: float, t: float, clause: str) -> None:
        spec = stat_panel_content(clause, u)
        # Left panel: real measured readout, or clause-derived key-term chips (never fake data)
        px, py, pw, ph = 120, 300, 760, 460
        cv2.rectangle(c, (px, py), (px + pw, py + ph), PANEL, -1, cv2.LINE_AA)
        cv2.rectangle(c, (px, py), (px + pw, py + ph), GRID, 1, cv2.LINE_AA)
        if spec["kind"] == "numeric":
            self._pil_text(c, spec["label"], (px + 40, py + 120), self._f_big, TEXT)
            self._pil_text(c, spec["unit"], (px + 44, py + 230), self._f_mid, GOLD)
        else:
            self._pil_text(c, spec["unit"], (px + 40, py + 44), self._f_mid, GOLD)
            terms = spec["terms"] or ["CONCEPT", "RELATION", "EVIDENCE"]
            chip_y = py + 110
            for i, term in enumerate(terms[:4]):
                # Stagger chip reveal so the panel animates in sync with the shot's progress.
                if u * 1.35 <= i * 0.22:
                    break
                chip = term.upper()
                (tw, th), _bl = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
                x0, y0 = px + 40, chip_y
                x1, y1 = min(px + pw - 40, x0 + tw + 48), y0 + th + 26
                cv2.rectangle(c, (x0, y0), (x1, y1), GRID, -1, cv2.LINE_AA)
                cv2.rectangle(c, (x0, y0), (x1, y1), GOLD, 1, cv2.LINE_AA)
                cv2.putText(c, chip, (x0 + 24, y1 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.72, TEXT, 2, cv2.LINE_AA)
                chip_y = y1 + 22
        # Right panel: animated bars + verb-shaped curve
        rx, ry, rw, rh = 960, 300, 840, 460
        cv2.rectangle(c, (rx, ry), (rx + rw, ry + rh), PANEL, -1, cv2.LINE_AA)
        cv2.rectangle(c, (rx, ry), (rx + rw, ry + rh), GRID, 1, cv2.LINE_AA)
        n_bars = 7
        targets = rng.uniform(0.25, 0.95, n_bars)
        for b in range(n_bars):
            bh = int(targets[b] * (rh - 140) * min(1.0, max(0.0, u * 1.25 - b * 0.07)))
            bx = rx + 60 + b * 104
            cv2.rectangle(c, (bx, ry + rh - 60 - bh), (bx + 56, ry + rh - 60), CYAN if b % 2 else GOLD, -1, cv2.LINE_AA)
        low = (clause or "").lower()
        if any(k in low for k in ("fell", "drop", "decline", "decrease", "collapse", "lost")):
            shape = lambda tk: math.exp(-3.0 * tk)  # noqa: E731
        elif any(k in low for k in ("cycle", "season", "orbit", "pulse")):
            shape = lambda tk: 0.5 + 0.42 * math.sin(2 * math.pi * 1.6 * tk)  # noqa: E731
        else:
            shape = lambda tk: 1.0 - math.exp(-3.2 * tk)  # noqa: E731
        pts = []
        for k in range(int(max(2, 160 * min(1.0, u * 1.15)))):
            tk = k / 160.0
            pts.append((rx + 50 + int(tk * (rw - 100)), ry + rh - 90 - int(shape(tk) * (rh - 180))))
        if len(pts) >= 2:
            cv2.polylines(c, [np.array(pts, np.int32)], False, GOLD, 3, cv2.LINE_AA)

    # -- template 6: timeline arc -------------------------------------------

    def _tpl_timeline(self, c: np.ndarray, rng: np.random.Generator, u: float, t: float, clause: str) -> None:
        h, w = c.shape[:2]
        # Marker count, spacing jitter and arc heights all consume the clause salt so two
        # timeline plates can never render identically for different clauses.
        n_marks = int(rng.integers(5, 8))
        ax_y = int(h * 0.6)
        x0, x1 = 200, w - 200
        x_jitter = rng.integers(-26, 27, size=n_marks)
        arc_scale = rng.uniform(18.0, 46.0, size=n_marks)
        cv2.line(c, (x0, ax_y), (x1, ax_y), GRID, 2, cv2.LINE_AA)
        for m in range(n_marks):
            frac = m / (n_marks - 1)
            mx = int(x0 + frac * (x1 - x0) + x_jitter[m])
            wobble = int(arc_scale[m] * math.sin(frac * math.pi))
            my = ax_y - wobble
            lit = frac <= min(1.0, u * 1.2)
            cv2.line(c, (mx, ax_y), (mx, my), GRID, 1, cv2.LINE_AA)
            cv2.circle(c, (mx, my), 16, GOLD if lit else GRID, -1, cv2.LINE_AA)
            if lit:
                cv2.circle(c, (mx, my), 24, GOLD, 2, cv2.LINE_AA)
            self._pil_text(c, f"PHASE {m + 1:02d}", (mx - 48, my - 64), self._f_small, MUTED)
        # sweeping progress beam with an expanding pulse ring on the leading marker
        sweep_x = int(x0 + min(1.0, u * 1.15) * (x1 - x0))
        glow = np.zeros_like(c)
        cv2.line(glow, (sweep_x, ax_y - 150), (sweep_x, ax_y + 40), CYAN, 3, cv2.LINE_AA)
        pulse_r = int(18 + (t * 60) % 46)
        cv2.circle(glow, (sweep_x, ax_y), pulse_r, CYAN, 2, cv2.LINE_AA)
        # drifting tick marks along the axis give motion even before the next phase lights
        for k in range(14):
            tx = int(x0 + ((k * 137 + t * 120) % (x1 - x0)))
            cv2.line(glow, (tx, ax_y + 14), (tx, ax_y + 26), GOLD, 2, cv2.LINE_AA)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.7, 0)

    # -- PIL helper ----------------------------------------------------------

    def _pil_text(self, canvas: np.ndarray, text: str, org: tuple[int, int], font, color) -> None:
        from PIL import Image, ImageDraw

        img = Image.fromarray(canvas)
        ImageDraw.Draw(img).text(org, text, font=font, fill=color)
        canvas[:] = np.asarray(img)


def iter_semantic_diagram_frames(
    template: str,
    subject_label: str,
    clause: str,
    seed: int,
    n_frames: int,
    fps: int = 24,
    width: int = 1920,
    height: int = 1080,
):
    """Yield continuous 24fps RGB uint8 frames for one semantic-diagram segment."""
    engine = SemanticDiagramEngine()
    for i in range(max(1, n_frames)):
        u = (i + 1) / max(1, n_frames)
        time_s = i / max(1, fps)
        yield engine.render_frame(template, subject_label, clause, u, time_s, w=width, h=height, seed=seed)
