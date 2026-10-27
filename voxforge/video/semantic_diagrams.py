"""Semantic diagram engines for arbitrary-topic documentaries.

Six visually distinct, fully animated obsidian-blackboard templates chosen by the
:class:`~voxforge.video.semantic_router.SemanticShotDirector`. The pictures are **meaning
driven**: every clause is first parsed by :mod:`voxforge.video.clause_semantics` into an
archetype (building, destruction, timeline, quantity, cosmos, catalog…) plus clause-derived
entities, quantities and time marks. The engines then draw *those exact words and numbers* —
node labels, stage labels, axis marks, readouts — so the diagram depicts what the narrator is
saying instead of decorating the clause with abstract wallpaper. Nothing here depicts an
unrelated monument, and no on-diagram text may be invented (Stage E verifies label honesty).

Output convention matches the rest of the video package: RGB uint8 frames (the encoder pipe is
``rgb24``), dark enough to classify as the ``darkhud`` Stage E evidence group, with continuous
24 fps motion in every template.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass

import cv2
import numpy as np

from voxforge.video.clause_semantics import (
    ARCH_CATALOG,
    ARCH_DESTRUCTION,
    ARCH_ENDURE,
    ClauseSemantics,
    analyze_clause,
    entity_terms,
    stage_terms,
)
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
AMBER = (232, 122, 58)
# Floor luma tuned so every template decodes above Stage D's Y>8 black-frame gate even after
# H.264 losses, while remaining well inside the Stage E darkhud group (mean luma well under 70).
INK = (12, 17, 28)
GRID = (33, 48, 70)
PANEL = (17, 26, 44)
TEXT = (238, 242, 248)
MUTED = (158, 176, 200)

_TEMPLATE_KICKERS = {
    TPL_TERRAIN: "FORCES & TERRAIN MODEL",
    TPL_NETWORK: "ENTITY & RELATION MODEL",
    TPL_PARTICLES: "SYSTEM & ORBIT MODEL",
    TPL_PROCESS: "PROCESS & SEQUENCE MODEL",
    TPL_STATS: "QUANTITATIVE EVIDENCE MODEL",
    TPL_TIMELINE: "HISTORY & DURATION MODEL",
}


def extract_key_terms(clause: str, limit: int = 4) -> list[str]:
    """Backwards-compatible clause term extractor (delegates to the semantics module)."""
    from voxforge.video.clause_semantics import ranked_terms

    return ranked_terms(clause, limit=limit)


def stat_panel_content(clause: str, u: float) -> dict:
    """Pure, testable spec for the stat dashboard's primary readout.

    Numeric clauses show their real measured value with its original precision; numberless
    clauses get clause-derived key terms instead of an invented percentage.
    """
    sem = analyze_clause(clause)
    if sem.quantities:
        q = sem.quantities[0]
        # Only clause-verbatim words may caption the number; omit the line rather than invent one.
        unit_txt = q.unit or q.noun or (sem.terms[0] if sem.terms else "")
        return {"kind": "numeric", "label": q.display_number,
                "unit": unit_txt.upper() if unit_txt else None}
    # Numberless clause: honest key-signal chips, no fabricated header or percentage.
    return {"kind": "keywords", "terms": extract_key_terms(clause), "unit": None}


@dataclass
class _Label:
    text: str
    xy: tuple[int, int]
    font_key: str
    color: tuple[int, int, int]
    anchor: str = "la"
    max_width: int | None = None


class SemanticDiagramEngine:
    """Renders animated, meaning-driven diagrams for the six generic visual templates."""

    def __init__(self) -> None:
        self._f_kicker = load_font("SpaceGrotesk-Bold.ttf", 17)
        self._f_title = load_font("SpaceGrotesk-Bold.ttf", 34)
        # Size ladder lets long episode titles shrink to fit instead of running off-canvas.
        self._f_title_ladder = [load_font("SpaceGrotesk-Bold.ttf", s) for s in (34, 28, 24, 20)]
        self._f_big = load_font("SpaceGrotesk-Bold.ttf", 64)
        self._f_mid = load_font("SpaceGrotesk-Medium.ttf", 22)
        self._f_small = load_font("Inter-Medium.ttf", 17)
        self._f_node = load_font("Inter-SemiBold.ttf", 17)
        self._f_tag = load_font("Inter-Medium.ttf", 15)
        self._f_num = load_font("SpaceGrotesk-Bold.ttf", 22)
        # Shrink ladders let a long clause word fit without an ellipsis: "…" is not a clause
        # token and would break the Stage E label-honesty check, so labels shrink or are dropped.
        self._f_node_shrink = [load_font("Inter-SemiBold.ttf", s) for s in (17, 15, 13, 11)]
        self._f_tag_shrink = [load_font("Inter-Medium.ttf", s) for s in (15, 13, 11, 10)]
        self._f_shrink_ladders = {"node": self._f_node_shrink, "tag": self._f_tag_shrink}
        self._labels: list[_Label] = []

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
        semantics: ClauseSemantics | None = None,
    ) -> np.ndarray:
        rng_seed = seed if seed is not None else subject_seed(subject_label)
        rng = np.random.default_rng(rng_seed)
        canvas = self._base_canvas(w, h, rng)
        sem = semantics or analyze_clause(clause)
        self._labels = []
        draw_fn = {
            TPL_TERRAIN: self._tpl_terrain,
            TPL_NETWORK: self._tpl_network,
            TPL_PARTICLES: self._tpl_particles,
            TPL_PROCESS: self._tpl_process,
            TPL_STATS: self._tpl_stats,
            TPL_TIMELINE: self._tpl_timeline,
        }.get(template, self._tpl_stats)
        draw_fn(canvas, rng, u, time_s, clause, sem)
        self._flush_labels(canvas)
        self._draw_chrome(canvas, template, subject_label, clause)
        return canvas

    # -- label plumbing -----------------------------------------------------

    def _add_label(
        self,
        text: str,
        xy: tuple[int, int],
        font_key: str = "node",
        color: tuple[int, int, int] = TEXT,
        anchor: str = "la",
        max_width: int | None = None,
    ) -> None:
        if text:
            self._labels.append(_Label(str(text), xy, font_key, color, anchor, max_width))

    def _flush_labels(self, canvas: np.ndarray) -> None:
        """Render every clause-derived label in one PIL pass for 24fps performance."""
        if not self._labels:
            return
        from PIL import Image, ImageDraw

        fonts = {
            "kicker": self._f_kicker, "title": self._f_title, "big": self._f_big,
            "mid": self._f_mid, "small": self._f_small, "node": self._f_node,
            "tag": self._f_tag, "num": self._f_num,
        }
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        h, w = canvas.shape[:2]
        # First pass: fit text (font-shrink ladder, never an ellipsis) and compute pixel boxes.
        placed: list[tuple[_Label, object, str, list[int]]] = []
        for lab in self._labels:
            font = fonts.get(lab.font_key, self._f_node)
            text = lab.text
            if lab.max_width:
                ladder = self._f_shrink_ladders.get(lab.font_key, [font])
                font = next((f for f in ladder if draw.textlength(text, font=f) <= lab.max_width),
                            ladder[-1])
                if draw.textlength(text, font=font) > lab.max_width:
                    continue  # still too wide at the smallest size: drop, never truncate
            x0, y0, x1, y1 = draw.textbbox(lab.xy, text, font=font, anchor=lab.anchor)
            placed.append((lab, font, text, [int(x0), int(y0), int(x1), int(y1)]))
        # Relax overlaps: inflate boxes by an anti-collision margin (near-misses like two
        # adjacent orbit labels count), then repeatedly push the lower box downward.
        pad_x, pad_y = 10, 8
        for _ in range(12):
            moved = False
            for i in range(len(placed)):
                for j in range(i + 1, len(placed)):
                    a, b = placed[i][3], placed[j][3]
                    ox = min(a[2], b[2]) - max(a[0], b[0]) + pad_x
                    oy = min(a[3], b[3]) - max(a[1], b[1]) + pad_y
                    if ox > 0 and oy > 0:
                        shift = oy + 6
                        lower = b if b[1] >= a[1] else a
                        lower[1] += shift
                        lower[3] += shift
                        moved = True
            if not moved:
                break
        for lab, font, text, box in placed:
            # Keep every label fully inside the canvas after the relaxation.
            bw, bh = box[2] - box[0], box[3] - box[1]
            box[0] = max(4, min(box[0], w - bw - 4))
            box[1] = max(6, min(box[1], h - bh - 8))
            box[2], box[3] = box[0] + bw, box[1] + bh
            if lab.anchor == "mm":
                xy = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            else:  # left-ascender ("la") and friends: place from the fitted top-left
                xy = (box[0], box[1])
            draw.text(xy, text, font=font, fill=lab.color, anchor=lab.anchor)
        canvas[:] = np.asarray(img)

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
        max_w = canvas.shape[1] - 172
        title_lines = self._fit_title(draw, (subject_label or "").upper(), max_w)
        y = 96
        for line, font in title_lines[:2]:
            draw.text((86, y), line, font=font, fill=TEXT)
            y += int(font.size * 1.25)
        # The spoken clause is intentionally NOT repeated here: the burned caption carries it
        # verbatim, and the diagram itself now carries clause-derived labels in its composition.
        canvas[:] = np.asarray(img)

    def _fit_title(self, draw, text: str, max_w: int) -> list[tuple[str, object]]:
        for font in self._f_title_ladder:
            for width in (48, 38, 30, 24):
                lines = textwrap.wrap(text, width=width) or [text]
                if all(draw.textlength(line, font=font) <= max_w for line in lines):
                    return [(line, font) for line in lines[:2]]
        font = self._f_title_ladder[-1]
        return [(line[:24].rstrip() + "…", font) for line in textwrap.wrap(text, width=24)[:2]]

    # -- template 1: terrain / forces (geo nature, destruction) ------------

    def _tpl_terrain(self, c, rng, u, t, clause, sem: ClauseSemantics) -> None:
        h, w = c.shape[:2]
        falling = sem.direction == "falling" or sem.archetype == ARCH_DESTRUCTION
        phase = rng.uniform(0, math.tau, size=4)
        amps = rng.uniform(60, 150, size=4)
        base_y = np.linspace(h * 0.45, h * 0.82, 4)
        xs = np.arange(w, dtype=np.float32)
        glow = np.zeros_like(c)
        # Numeral words ("six") are not forces/materials; stage_terms already filters them.
        layer_terms = (stage_terms(clause) or sem.terms or ["NATURAL FORCES"])[:4]
        for layer in range(4):
            profile = (
                base_y[layer]
                - amps[layer]
                * (0.65 * np.sin(xs / (260 + 60 * layer) + phase[layer])
                   + 0.35 * np.sin(xs / (90 + 24 * layer) + phase[layer] * 1.7))
            )
            color = (8 + 3 * layer, 14 + 5 * layer, 40 + 12 * layer)
            pts = np.stack([xs, profile], axis=1).astype(np.int32)
            poly = np.vstack([pts, [[w, h], [0, h]]]).astype(np.int32)
            cv2.fillPoly(c, [poly], color)
            ridge_col = AMBER if falling and layer == 0 else (GOLD if layer == 0 else CYAN)
            cv2.polylines(glow, [pts], False, ridge_col, 2, cv2.LINE_AA)
            # Every stratum carries a real clause word naming the force/material it depicts.
            tag_y = int(base_y[layer] - amps[layer] * 0.55)
            if layer < len(layer_terms):
                self._add_label(layer_terms[layer].upper(), (150, max(250, tag_y)),
                                "tag", ridge_col, max_width=260)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.55, 0)
        # A pulse travels continuously along the top ridge (driven by t, not the reveal u)
        # so the plate stays in motion even after the force markers finish their fall.
        travel_x = int((t * 210) % w)
        top_profile = (
            base_y[0]
            - amps[0]
            * (0.65 * np.sin(travel_x / 260 + phase[0])
               + 0.35 * np.sin(travel_x / 90 + phase[0] * 1.7))
        )
        travel_y = int(top_profile)
        for rr, aa in ((22, 1), (14, 2), (7, 3)):
            cv2.circle(glow, (travel_x, travel_y), rr, AMBER if falling else GOLD,
                       aa, cv2.LINE_AA)
        cv2.circle(glow, (travel_x, travel_y), 5, CYAN, -1, cv2.LINE_AA)
        pulse_x = int((0.12 + 0.76 * u) * w)
        peak_y = int(base_y[0] - amps[0] * 1.4)
        scan_x = int((t * 240) % w)
        cv2.line(glow, (scan_x, int(h * 0.30)), (scan_x, h), CYAN, 2, cv2.LINE_AA)
        if falling:
            # Destruction semantics: force markers with downward arrows + falling fragments.
            # Skip terms already naming a stratum so the same word never appears twice; numeral
            # words ("six") are excluded because they name a count, not a destructive force.
            content = stage_terms(clause) or sem.terms or ["FORCE"]
            used = {t.lower() for t in layer_terms}
            force_terms = [t for t in content if t.lower() not in used][:4] or (
                content[:4] or ["FORCE"])
            for k, term in enumerate(force_terms):
                fx = int(w * (0.22 + 0.18 * k))
                life = (u * 1.6 + k * 0.19) % 1.0
                fy = int(300 + 360 * life)
                cv2.arrowedLine(glow, (fx, fy - 46), (fx, fy), AMBER, 3, tipLength=0.5, line_type=cv2.LINE_AA)
                cv2.circle(glow, (fx, fy), 5, AMBER, -1, cv2.LINE_AA)
                self._add_label(term.upper(), (fx, max(230, fy - 78)), "tag", AMBER,
                                anchor="mm", max_width=190)
        else:
            for ring in range(3):
                rr = int(20 + ((t * 90 + ring * 60) % 180))
                ring_alpha = 1.0 - (rr - 20) / 180.0
                cv2.circle(glow, (pulse_x, peak_y), rr, GOLD, max(1, int(4 * ring_alpha)), cv2.LINE_AA)
            cv2.circle(glow, (pulse_x, peak_y), 16, GOLD, -1, cv2.LINE_AA)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.8, 0)

    # -- template 2: node network (catalog count, concept, relations) -------

    def _tpl_network(self, c, rng, u, t, clause, sem: ClauseSemantics) -> None:
        h, w = c.shape[:2]
        terms = sem.terms[:6] or (sem.labels[:6] or ["CONCEPT"])
        glow = np.zeros_like(c)
        if sem.archetype == ARCH_CATALOG:
            n = max(2, min(7, sem.node_count()))
            xs = np.linspace(250, w - 250, n).astype(int)
            y = int(h * 0.56)
            for i in range(n - 1):
                lit = ((i + 1) / n) <= min(1.0, u * 1.2)
                cv2.line(c, (xs[i], y), (xs[i + 1], y), GOLD if lit else GRID, 2, cv2.LINE_AA)
                p = (t * 0.5 + i * 0.19) % 1.0
                cv2.circle(glow, (int(xs[i] + (xs[i + 1] - xs[i]) * p), y), 6, CYAN, -1, cv2.LINE_AA)
            for i, nx in enumerate(xs):
                lit = (i / n) <= min(1.0, u * 1.3)
                col = GOLD if lit else GRID
                cv2.circle(c, (nx, y), 26, PANEL, -1, cv2.LINE_AA)
                cv2.circle(c, (nx, y), 26, col, 2, cv2.LINE_AA)
                self._add_label(str(i + 1), (nx, y), "num", col, anchor="mm")
                if i < len(terms):
                    self._add_label(terms[i].upper(), (nx, y + 62), "node", TEXT,
                                    anchor="mm", max_width=200)
                if lit:
                    ring_r = 32 + int((t * 44 + i * 19) % 26)
                    cv2.circle(glow, (nx, y), ring_r, GOLD, 2, cv2.LINE_AA)
        else:
            # Three structurally distinct, seed-selected layouts so same-family graphs are never
            # the same picture with swapped labels. Every node carries a real clause term.
            node_terms = terms[:6] or ["CONCEPT"]
            n = len(node_terms)
            mode = int(rng.integers(0, 3))
            positions: list[tuple[int, int]] = []
            edges: list[tuple[int, int]] = []
            hub = 0
            if mode == 0:  # hub-and-spoke
                cx, cy = w // 2, int(h * 0.56)
                positions.append((cx, cy))
                radii = rng.uniform(250, 400, max(1, n - 1))
                angle_jit = rng.uniform(-0.4, 0.4, max(1, n - 1))
                for k in range(1, n):
                    ang = math.tau * (k - 1) / max(1, n - 1) + float(angle_jit[k - 1]) - math.pi / 2
                    r = float(radii[k - 1])
                    positions.append((int(cx + r * math.cos(ang)), int(cy + r * 0.6 * math.sin(ang))))
                    edges.append((0, k))
            elif mode == 1:  # horizontal concept chain
                xs = np.linspace(240, w - 240, n).astype(int) + rng.integers(-30, 31, n)
                ys = np.full(n, int(h * 0.55)) + rng.integers(-90, 91, n)
                positions = list(zip(xs.tolist(), ys.tolist()))
                edges = [(k, k + 1) for k in range(n - 1)]
            else:  # tiered rows
                for k in range(n):
                    row = k % 2
                    col = k // 2
                    x = int(300 + col * 440 + rng.integers(-26, 27))
                    y = int(h * (0.38 if row == 0 else 0.7)) + int(rng.integers(-24, 25))
                    positions.append((min(x, w - 200), y))
                edges = [(k, k + 1) for k in range(n - 1)]
            for i, j in edges:
                lit = (j / n) <= min(1.0, u * 1.25)
                cv2.line(c, positions[i], positions[j], GOLD if lit else GRID,
                         3 if lit else 2, cv2.LINE_AA)
                # Two phase-offset packets per edge keep the whole graph in continuous motion.
                for off in (0.0, 0.5):
                    p = (t * 0.5 + i * 0.17 + off) % 1.0
                    mx = int(positions[i][0] + (positions[j][0] - positions[i][0]) * p)
                    my = int(positions[i][1] + (positions[j][1] - positions[i][1]) * p)
                    cv2.circle(glow, (mx, my), 6, CYAN, -1, cv2.LINE_AA)
            for k, (nx, ny) in enumerate(positions):
                lit = (k / n) <= min(1.0, u * 1.3)
                is_hub = mode == 0 and k == hub
                rad = 40 if is_hub else 20
                halo = rad + int(4 * math.sin(t * 3.0 + k))
                cv2.circle(c, (nx, ny), rad, PANEL, -1, cv2.LINE_AA)
                cv2.circle(glow, (nx, ny), halo, CYAN if is_hub else GOLD, 2, cv2.LINE_AA)
                # Expanding pulse ring per node for frame-to-frame energy.
                ring_r = rad + 8 + int((t * 46 + k * 23) % 42)
                ring_a = max(1, 3 - int((t * 46 + k * 23) % 42) // 14)
                cv2.circle(glow, (nx, ny), ring_r, CYAN if is_hub else GOLD,
                           ring_a, cv2.LINE_AA)
                cv2.circle(c, (nx, ny), rad, CYAN if is_hub else (GOLD if lit else GRID),
                           3 if is_hub else 2, cv2.LINE_AA)
                if is_hub:
                    self._add_label(node_terms[k].upper(), (nx, ny), "node", CYAN,
                                    anchor="mm", max_width=130)
                else:
                    self._add_label(node_terms[k].upper(), (nx, ny + 42), "node", TEXT,
                                    anchor="mm", max_width=220)
            for a, rel, b in sem.relations[:2]:
                self._add_label(rel, (int(w * 0.5), int(h * 0.18)), "tag", GOLD, anchor="mm")
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.75, 0)

    # -- template 3: particle system (cosmos, collectives) ------------------

    def _tpl_particles(self, c, rng, u, t, clause, sem: ClauseSemantics) -> None:
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
            x = int(cx + radii[k] * math.cos(ang))
            y = int(cy + radii[k] * tilts[k] * 0.62 * math.sin(ang))
            pts.append((x, y, k))
        for a in range(0, n, 2):
            for b in range(a + 1, min(a + 6, n)):
                ax, ay, _ = pts[a]
                bx, by, _ = pts[b]
                if abs(ax - bx) < 170 and abs(ay - by) < 110:
                    cv2.line(c, (ax, ay), (bx, by), GRID, 1, cv2.LINE_AA)
        # Labeled orbital anchors carry the clause's actual bodies/entities; the rest are dust.
        # Numeral words ("five") are dropped and the named star is promoted to the core.
        label_terms = entity_terms(clause, limit=5) or sem.terms[:5] or ["SYSTEM"]
        n_anchor = min(4, max(1, len(label_terms) - 1))
        # Each anchor owns its own angular sector so labels never stack on the same side;
        # a slow shared rotation keeps the system alive without letting anchors converge.
        sector_jit = rng.uniform(-0.18, 0.18, n_anchor)
        for k in range(n_anchor):
            rr = 150 + k * 48
            cv2.ellipse(c, (cx, cy), (rr, int(rr * 0.62)), 0, 0, 360, GRID, 1, cv2.LINE_AA)
            base_ang = -math.pi / 2 + k * math.tau / n_anchor + float(sector_jit[k])
            ang = base_ang + t * 0.22
            ax = int(cx + rr * math.cos(ang))
            ay = int(cy + rr * 0.62 * math.sin(ang))
            cv2.circle(c, (ax, ay), 9, CYAN if k % 2 else GOLD, -1, cv2.LINE_AA)
            # Alternate labels above/below their body for clean separation.
            ly = ay - 28 if k % 2 == 0 else ay + 34
            self._add_label(label_terms[k + 1].upper(), (ax, ly), "tag",
                            CYAN if k % 2 else GOLD, anchor="mm", max_width=180)
        core_r = 26 + int(6 * math.sin(t * 2.2))
        cv2.circle(c, (cx, cy), core_r, PANEL, -1, cv2.LINE_AA)
        glow = np.zeros_like(c)
        cv2.circle(glow, (cx, cy), core_r + 14, CYAN, 2, cv2.LINE_AA)
        for x, y, k in pts:
            if (k / n) <= min(1.0, u * 1.4):
                cv2.circle(c, (x, y), 4, CYAN if k % 3 else GOLD, -1, cv2.LINE_AA)
        self._add_label(label_terms[0].upper(), (cx, cy), "node", CYAN, anchor="mm", max_width=150)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.8, 0)

    # -- template 4: process flow (building, journeys) ----------------------

    def _tpl_process(self, c, rng, u, t, clause, sem: ClauseSemantics) -> None:
        # Stage labels exclude spoken numerals/durations ("twenty", "years") and hedges
        # ("roughly") so the staircase reads as real build stages, not a word salad.
        steps = stage_terms(clause) or (sem.terms or ["PROCESS"])[:5]
        n_stages = max(1, len(steps))
        h, w = c.shape[:2]
        box_w, box_h = 270, 150
        # Fixed gap with the whole staircase centered: sparse (2-stage) builds must not get
        # stretched to opposite canvas edges; dense (5-stage) rows still fit with breathing room.
        gap = 110 if n_stages > 1 else 0
        margin = max(90, (w - n_stages * box_w - (n_stages - 1) * gap) // 2)
        # Seed-driven stage offsets keep same-family process plates structurally distinct.
        x_jit = rng.integers(-26, 27, n_stages)
        y_jit = rng.integers(-30, 31, n_stages)
        centers = []
        box_rects: list[tuple[int, int]] = []
        glow = np.zeros_like(c)
        for s in range(n_stages):
            x0 = margin + s * (box_w + gap) + int(x_jit[s])
            # Verb direction shapes the staircase: building rises, destruction falls, else flat.
            if sem.direction == "rising":
                y0 = int(620 - s * 70)
            elif sem.direction == "falling":
                y0 = int(360 + s * 70)
            else:
                y0 = 470
            y0 += int(y_jit[s])
            active = (s / n_stages) <= min(1.0, u * 1.25)
            col = GOLD if active else GRID
            cv2.rectangle(c, (x0, y0), (x0 + box_w, y0 + box_h), PANEL, -1, cv2.LINE_AA)
            cv2.rectangle(c, (x0, y0), (x0 + box_w, y0 + box_h), col, 2, cv2.LINE_AA)
            self._add_label(f"{s + 1:02d}", (x0 + 22, y0 + 18), "num", col)
            self._add_label(steps[s].upper(), (x0 + box_w // 2, y0 + box_h // 2 + 12),
                            "node", TEXT, anchor="mm", max_width=box_w - 36)
            if s < len(sem.time_marks):
                self._add_label(sem.time_marks[s].upper(), (x0 + box_w // 2, y0 + box_h + 34),
                                "tag", MUTED, anchor="mm", max_width=box_w)
            box_rects.append((x0, y0))
            centers.append((x0 + box_w, y0 + box_h // 2, x0 + box_w + gap, y0 + box_h // 2))
        for ci, (x1, y1, x2, y2) in enumerate(centers[:-1]):
            cv2.arrowedLine(c, (x1 + 8, y1), (x2 - 8, y2), GRID, 2, tipLength=0.25, line_type=cv2.LINE_AA)
            # Dual flowing packets plus marching dashes keep every connector visibly alive.
            for off in (0.0, 0.5):
                p = (t * 0.5 + ci * 0.11 + off) % 1.0
                cv2.circle(glow, (int(x1 + (x2 - x1) * p), y1), 6, CYAN, -1, cv2.LINE_AA)
            dash = int((t * 150 + ci * 30) % 52)
            for dx in range(x1, x2, 52):
                dxx = dx + dash - 26
                if x1 < dxx < x2:
                    cv2.line(glow, (dxx, y1 + 18), (dxx + 20, y1 + 18), GOLD, 2, cv2.LINE_AA)
        # Breathing highlight on the active stage border (same jittered box, never drifting).
        active_idx = min(n_stages - 1, int(u * n_stages))
        ax0, ay0 = box_rects[active_idx]
        cv2.rectangle(glow, (ax0 - 6, ay0 - 6), (ax0 + box_w + 6, ay0 + box_h + 6),
                      GOLD, 2, cv2.LINE_AA)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.8, 0)

    # -- template 5: stat dashboard (real measured quantities only) --------

    def _tpl_stats(self, c, rng, u, t, clause, sem: ClauseSemantics) -> None:
        spec = stat_panel_content(clause, u)
        px, py, pw, ph = 120, 300, 760, 460
        cv2.rectangle(c, (px, py), (px + pw, py + ph), PANEL, -1, cv2.LINE_AA)
        cv2.rectangle(c, (px, py), (px + pw, py + ph), GRID, 1, cv2.LINE_AA)
        if spec["kind"] == "numeric":
            self._add_label(spec["label"], (px + 40, py + 118), "big", TEXT)
            if spec["unit"]:
                self._add_label(spec["unit"], (px + 44, py + 232), "mid", GOLD)
            # Context line only when it adds information beyond the unit/noun already shown.
            if sem.terms and sem.terms[0].upper() != (spec["unit"] or ""):
                self._add_label(sem.terms[0].upper(), (px + 44, py + 300), "tag", MUTED, max_width=pw - 80)
        else:
            # No invented section header: the chips are the clause's own terms, nothing else.
            for i, term in enumerate(spec["terms"][:4]):
                if u * 1.35 <= i * 0.22:
                    break
                self._add_label("• " + term.upper(), (px + 44, py + 90 + i * 72), "node", TEXT,
                                max_width=pw - 80)
        # Right panel: bars sized ONLY from real clause quantities; no invented series.
        rx, ry, rw, rh = 960, 300, 840, 460
        cv2.rectangle(c, (rx, ry), (rx + rw, ry + rh), PANEL, -1, cv2.LINE_AA)
        cv2.rectangle(c, (rx, ry), (rx + rw, ry + rh), GRID, 1, cv2.LINE_AA)
        qs = [q for q in sem.quantities if q.value > 0]
        if len(qs) >= 2:
            vmax = max(q.value for q in qs)
            for b, q in enumerate(qs[:6]):
                bh = int(max(0.08, q.value / vmax) * (rh - 180) * min(1.0, u * 1.25))
                bx = rx + 70 + b * 120
                cv2.rectangle(c, (bx, ry + rh - 70 - bh), (bx + 64, ry + rh - 70),
                              CYAN if b % 2 else GOLD, -1, cv2.LINE_AA)
                self._add_label(q.display_number, (bx + 32, ry + rh - 84), "tag", TEXT,
                                anchor="mm", max_width=110)
                bar_name = q.noun or q.unit or (sem.terms[b].upper() if b < len(sem.terms) else "")
                if bar_name:
                    self._add_label(bar_name.upper(), (bx + 32, ry + rh - 44),
                                    "tag", MUTED, anchor="mm", max_width=110)
        elif qs:
            q = qs[0]
            bh = int((rh - 200) * min(1.0, u * 1.2))
            bx = rx + rw // 2 - 44
            cv2.rectangle(c, (bx, ry + rh - 90 - bh), (bx + 88, ry + rh - 90), GOLD, -1, cv2.LINE_AA)
            bar_name = q.noun or q.unit or (sem.terms[0] if sem.terms else "")
            if bar_name:
                self._add_label(bar_name.upper(), (rx + rw // 2, ry + rh - 60),
                                "node", GOLD, anchor="mm", max_width=300)
        else:
            # Numberless clause: non-data motif only — labeled key signals, never fake bars.
            for i, term in enumerate(spec.get("terms", [])[:4]):
                self._add_label("• " + term.upper(), (rx + 70, ry + 110 + i * 72), "node", TEXT,
                                max_width=rw - 120)
        # Continuous non-data motion: sweeping panel glint + breathing border so the dashboard
        # never freezes even after its (real) bars finish growing.
        glow = np.zeros_like(c)
        glint_x = rx + int((t * 160) % rw)
        cv2.line(glow, (glint_x, ry + 12), (glint_x, ry + rh - 12), CYAN, 2, cv2.LINE_AA)
        pulse = 3 + int(2 * math.sin(t * 2.6))
        cv2.rectangle(glow, (px - pulse, py - pulse), (px + pw + pulse, py + ph + pulse),
                      GOLD, 1, cv2.LINE_AA)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.7, 0)

    # -- template 6: timeline (history, endurance) --------------------------

    def _tpl_timeline(self, c, rng, u, t, clause, sem: ClauseSemantics) -> None:
        h, w = c.shape[:2]
        marks = sem.time_marks[:6] or [t.upper() for t in (sem.terms[:6] or ["PHASE"])]
        n_marks = max(2, len(marks))
        ax_y = int(h * 0.6)
        x0, x1 = 220, w - 220
        cv2.line(c, (x0, ax_y), (x1, ax_y), GRID, 2, cv2.LINE_AA)
        glow = np.zeros_like(c)
        # Seed-driven spacing jitter and arc heights make each timeline unique per clause.
        spacing_jit = rng.integers(-34, 35, n_marks)
        arc_scale = rng.uniform(18.0, 44.0, n_marks)
        positions = np.linspace(x0, x1, n_marks).astype(int) + spacing_jit
        for m, mx in enumerate(positions):
            wobble = int(arc_scale[m] * math.sin((m / max(1, n_marks - 1)) * math.pi))
            my = ax_y - wobble
            lit = (m / n_marks) <= min(1.0, u * 1.2)
            cv2.line(c, (mx, ax_y), (mx, my), GRID, 1, cv2.LINE_AA)
            cv2.circle(c, (mx, my), 15, GOLD if lit else GRID, -1, cv2.LINE_AA)
            if lit:
                cv2.circle(glow, (mx, my), 23, GOLD, 2, cv2.LINE_AA)
            # Clears the sweep pulse ring (max radius ~64 at the axis) plus label padding.
            self._add_label(marks[m], (mx, my - 62), "tag", MUTED if not lit else TEXT,
                            anchor="mm", max_width=210)
        # Two phase-offset packets travel continuously along the axis so the timeline keeps
        # frame-to-frame energy even after the reveal cursor has finished its sweep.
        for off in (0.0, 0.5):
            px = x0 + int(((t * 0.28 + off) % 1.0) * (x1 - x0))
            cv2.circle(glow, (px, ax_y), 7, CYAN, -1, cv2.LINE_AA)
            cv2.circle(glow, (px, ax_y), 13, GOLD, 1, cv2.LINE_AA)
        sweep_x = int(x0 + min(1.0, u * 1.15) * (x1 - x0))
        # Short vertical stem so the moving cursor never crosses the mark labels above the axis.
        cv2.line(glow, (sweep_x, ax_y - 30), (sweep_x, ax_y + 40), CYAN, 3, cv2.LINE_AA)
        pulse_r = int(18 + (t * 60) % 46)
        cv2.circle(glow, (sweep_x, ax_y), pulse_r, CYAN, 2, cv2.LINE_AA)
        if sem.archetype == ARCH_ENDURE or sem.direction == "stable":
            # Endurance = a persistent unbroken band carrying a legacy term across time.
            cv2.line(glow, (x0, ax_y + 64), (x1, ax_y + 64), GOLD, 4, cv2.LINE_AA)
            mark_words = {m.upper() for m in marks}
            band_term = next((w for w in sem.terms if w.upper() not in mark_words), None)
            if band_term:
                self._add_label(band_term.upper(), ((x0 + x1) // 2, ax_y + 108),
                                "node", GOLD, anchor="mm", max_width=420)
        c[:] = cv2.addWeighted(c, 1.0, glow, 0.75, 0)


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
    sem = analyze_clause(clause)
    for i in range(max(1, n_frames)):
        u = (i + 1) / max(1, n_frames)
        time_s = i / max(1, fps)
        yield engine.render_frame(
            template, subject_label, clause, u, time_s, w=width, h=height,
            seed=seed, semantics=sem,
        )
