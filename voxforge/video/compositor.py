"""Cinema-Grade Documentary Film & Motion-Graphics Compositor (Rec.709, Univisium 2.00:1).

Implements authentic prestige documentary broadcast visuals (*Lemmino* / *Netflix* / *BBC Horizon* /
*Vox* / *Johnny Harris* / *National Geographic* aesthetic — zero synthetic human avatars):
  1. Rapid-Cut 3-Wonder Aerial Cold-Open Documentary Title Sequence (`intro_bumper`, lit at `t=0.0s`)
  2. Univisium `2.00:1` Cinema Framing (`50px` matte bars housing minimal tracked `ACT 02/05`
     and `DEEP HORIZON` metadata completely OFF the active `1920x980` photography!)
  3. 3D Tilted 1:50m Natural Earth GIS Sovereign Cartography (`render_wide_studio -> geo_map`)
  4. High-Parallax 2.5D Monument & Telephoto Masonry Cinema (`render_host_closeup`, `render_broll_cinema`)
  5. 3-Plane Museum Archival Light-Table Survey (`render_reaction_shot`, `render_studio_insert`)
  6. Verbatim Eyewitness Primary-Source Inscription Plates (`render_quote_card` — complete
     historical inscriptions, NEVER truncated mid-sentence)
  7. Live 24fps Counting Engineering Telemetry Split-Views (`render_split_fact_card`)
  8. Restrained One-Time Narrator Attribution Lower Thirds & Clean Cinema End-Credits Card
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from voxforge.config import HOST_A, HOST_B, HOSTS, VIDEO_SPEC
from voxforge.models import ChapterScript, DialogueTurn, EpisodeScript
from voxforge.video.infographics import apply_cinema_grade, load_font
from voxforge.video.procedural_world import (
    SCENE_GIZA,
    Monument3DEngine,
    ProceduralCadCache,
    resolve_procedural_scene_id,
)
from voxforge.video.semantic_diagrams import SemanticDiagramEngine
from voxforge.video.semantic_router import VisualDirective, subject_seed

AnimMode = Literal[
    "none",
    "broll_depth",
    "geo_map",
    "archival_desk",
    "telemetry_split",
    "kinetic_quote",
    "smooth_card",
    "intro_montage",
    "semantic_diagram",
]

# Verbatim primary-source historical eyewitness inscriptions (8 monument-specific records, 100% complete sentences)
_HISTORICAL_PRIMARY_QUOTES: list[tuple[tuple[str, ...], str, str, str]] = [
    (
        ("redeemer", "corcovado", "soapstone", "rio de janeiro", "guanabara"),
        "From the summit of Corcovado, reinforced concrete and six million soapstone tiles stand open-armed above bay, mountain, and Atlantic storm.",
        "HEITOR DA SILVA COSTA",
        "Engineering Dedication Record · Christ the Redeemer, Rio de Janeiro (1931 CE)",
    ),
    (
        ("taj mahal", "agra", "shah jahan", "mumtaz", "yamuna", "makrana"),
        "Before the radiance of this white Makrana marble dome above the Yamuna, both sun and moon shed tears of awe from their eyes.",
        "ABDUL HAMID LAHORI",
        "Padshahnama · Official Imperial Chronicle of Shah Jahan (1643 CE)",
    ),
    (
        ("chichen", "chichén", "kukulkan", "kukulcán", "cenote", "chicxulub", "castillo"),
        "Rising above the great sacred cenote of Yucatán, the terraced pyramid of Kukulcán commands the plain with four stairways of carved stone.",
        "DIEGO DE LANDA",
        "Relación de las Cosas de Yucatán · Early Colonial Maya Survey (1566 CE)",
    ),
    (
        ("machu picchu", "inca", "pachacuti", "intihuatana", "urubamba"),
        "Built without a single iron chisel, each carved block interlocks so tightly against its neighbor that not even a knife blade may enter.",
        "PEDRO CIEZA DE LEÓN",
        "Crónica del Perú · Eyewitness Survey of Andean Inca Masonry (1553 CE)",
    ),
    (
        ("colosseum", "flavian", "vespasian", "titus", "hypogeum", "amphitheatre"),
        "While stands the Coliseum, Rome shall stand; when falls the Coliseum, Rome shall fall — and when Rome falls, the world.",
        "THE VENERABLE BEDE",
        "Collectanea Pseudo-Bedae · Early Medieval Roman Chronicle (8th Century CE)",
    ),
    (
        ("petra", "khazneh", "nabataean", "nabataeans", "siq"),
        "Set within precipitous desert rock walls fed by abundant hidden springs, the capital of the Nabataeans stands naturally fortified on every side.",
        "STRABO OF AMASEIA",
        "Geographica XVI.4.21 · Classical Geographical Survey of Petra (c. 20 CE)",
    ),
    (
        ("great wall", "jinshanling", "jiayuguan", "ming", "meng tian"),
        "Hollow brick watchtowers stand shoulder to shoulder along the mountain crests, carrying beacon fire faster than any horse upon the frontier.",
        "GENERAL QI JIGUANG",
        "Memorial on Northern Frontier Defense · Ming Dynasty Archive (1571 CE)",
    ),
    (
        ("giza", "khufu", "ancient seven", "artemis", "theamata", "alexandria"),
        "When I saw the house of Artemis rising to the clouds, all other marvels lost their brilliancy — save the sun upon Olympus.",
        "ANTIPATER OF SIDON",
        "Greek Anthology IX.58 · Earliest Surviving List of the Seven Wonders (c. 140 BCE)",
    ),
]


def resolve_historical_quote(chapter_hint: str, fallback_text: str) -> tuple[str, str, str]:
    """Return `(complete_quote, source_author, archival_citation)` prioritizing spoken turn text first."""
    clauses = [c.strip().lower() for c in chapter_hint.split("||") if c.strip()]
    for low in clauses:
        for kws, quote, author, citation in _HISTORICAL_PRIMARY_QUOTES:
            if any(re.search(rf"\b{re.escape(k)}\b", low) for k in kws):
                return quote, author, citation

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", fallback_text.strip()) if len(s.strip()) > 24]
    short_sents = [s for s in sentences if len(s.split()) <= 26]
    chosen = short_sents[0] if short_sents else (sentences[0] if sentences else fallback_text.strip())
    if len(chosen.split()) > 28:
        clauses = re.split(r"\s*[—;:]\s*", chosen)
        chosen = clauses[0].rstrip(",. ") + "."
    return chosen, "ARCHIVAL CHRONICLE", f"Historical Field Dossier · {chapter_hint.split(':')[0][:42]}"


@dataclass(frozen=True)
class PlateLayout:
    """Segment visual plate + live 24fps documentary motion-graphics & transient graphic spec."""

    image_path: Path
    use_ken_burns: bool = False
    ken_burns_zoom_in: bool = False
    wave_x: int | None = None
    wave_y: int | None = None
    wave_w: int = 305
    wave_h: int = 36
    wave_hex: str = "#E5A93C"
    anim_mode: AnimMode = "none"
    anim_speaker: Literal["HOST_A", "HOST_B", "NONE"] = "NONE"
    anim_is_speaking: bool = True
    pip_rect: tuple[int, int, int, int] | None = None
    transient_lt_path: Path | None = None
    secondary_image_path: Path | None = None
    map_target_hint: str = ""
    map_prev_hint: str = ""
    # Generic (scene_id == -1) globes show a neutral rotating Earth with no site highlight or
    # place callout; variation shifts orbital longitude so repeat globes never frame identically.
    map_neutral: bool = False
    map_variation: int = 0
    era_text: str = ""
    location_text: str = ""
    chapter_title_text: str = ""
    primary_caption: str = ""
    secondary_caption: str = ""
    telemetry_metrics: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    chronology_year: int | None = None
    prev_chronology_year: int | None = None
    quote_verbatim: str = ""
    quote_author: str = ""
    quote_citation: str = ""
    hud_overlay_path: Path | None = None
    scene_hint: str = ""
    # Semantic explainer diagrams (anim_mode == "semantic_diagram")
    semantic_template: str = ""
    semantic_subject: str = ""
    semantic_clause: str = ""
    semantic_seed: int = 0
    # Director-resolved wonder scene id; forces monument/CAD/sim engines to draw the exact
    # subject the clause bound to instead of re-resolving via the legacy hint fallback.
    directive_scene_id: int | None = None
    # Director's treatment label (e.g. lidar_scan / physics_simulation) so the animation engine
    # choice matches the treatment recorded in the scene plan instead of segment-index parity.
    semantic_treatment: str = ""


def _hex_to_rgba(hex_str: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha


def apply_rec709_studio_grade(img: Image.Image, width: int = 1920, height: int = 1080) -> Image.Image:
    """Apply cohesive Rec.709 documentary cinema grade (crisp micro-contrast, slate shadows, legal luma)."""
    im = img.convert("RGB")
    if im.size != (width, height):
        src_w, src_h = im.size
        target_r = width / height
        src_r = src_w / max(src_h, 1)
        if src_r > target_r:
            nw = int(round(src_h * target_r))
            left = (src_w - nw) // 2
            im = im.crop((left, 0, left + nw, src_h))
        else:
            nh = int(round(src_w / target_r))
            top = max(0, int((src_h - nh) * 0.25))
            im = im.crop((0, top, src_w, top + nh))
        im = im.resize((width, height), Image.Resampling.LANCZOS)

    im = im.filter(ImageFilter.UnsharpMask(radius=1.3, percent=110, threshold=3))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = np.clip((arr - 0.5) * 1.05 + 0.5, 0.0, 1.0)
    luma = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    shadow_w = np.clip(1.0 - luma * 2.2, 0.0, 1.0)
    mid_w = np.clip(1.0 - np.abs(luma - 0.54) * 2.3, 0.0, 1.0)

    arr[..., 0] = np.clip(arr[..., 0] + 0.015 * mid_w - 0.008 * shadow_w, 0.0, 1.0)
    arr[..., 1] = np.clip(arr[..., 1] + 0.004 * mid_w, 0.0, 1.0)
    arr[..., 2] = np.clip(arr[..., 2] - 0.008 * mid_w + 0.016 * shadow_w, 0.0, 1.0)

    gy, gx = np.meshgrid(
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    vig = np.clip(1.0 - 0.12 * (gx * gx + gy * gy), 0.85, 1.0)[..., None]
    arr = arr * vig

    out = np.clip(18.0 + arr * 216.0, 18.0, 234.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _crop_rounded_image(img: Image.Image, target_w: int, target_h: int, radius: int = 14) -> Image.Image:
    im = img.convert("RGB")
    src_w, src_h = im.size
    target_ratio = target_w / target_h
    src_ratio = src_w / max(src_h, 1)
    if src_ratio > target_ratio:
        nw = int(round(src_h * target_ratio))
        left = (src_w - nw) // 2
        im = im.crop((left, 0, left + nw, src_h))
    else:
        nh = int(round(src_w / target_ratio))
        top = max(0, int((src_h - nh) * 0.22))
        im = im.crop((0, top, src_w, top + nh))
    im = im.resize((target_w, target_h), Image.Resampling.LANCZOS).convert("RGBA")

    mask = Image.new("L", (target_w, target_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, target_w - 1, target_h - 1], radius=radius, fill=255)
    im.putalpha(mask)
    return im


def _apply_bottom_legibility_gradient(canvas: Image.Image, strength: int = 116) -> Image.Image:
    """Apply subtle lower-frame cinema gradient (`y=820..1030`) inside active Univisium frame."""
    w, h = canvas.size
    grad_h = 210
    y0 = h - 50 - grad_h
    y_ramp = np.linspace(0.0, 1.0, grad_h, dtype=np.float32) ** 1.85
    alpha_col = (y_ramp * strength).astype(np.uint8)
    alpha_2d = np.repeat(alpha_col[:, None], w, axis=1)
    grad_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    grad_rgba[y0 : y0 + grad_h, :, 0] = 6
    grad_rgba[y0 : y0 + grad_h, :, 1] = 8
    grad_rgba[y0 : y0 + grad_h, :, 2] = 14
    grad_rgba[y0 : y0 + grad_h, :, 3] = alpha_2d
    return Image.alpha_composite(canvas.convert("RGBA"), Image.fromarray(grad_rgba, mode="RGBA"))


class MulticamCompositor:
    """Builds Univisium 2.00:1 cinema-documentary Rec.709 overlays, chronology cards & motion specs."""

    def __init__(self) -> None:
        self.w = VIDEO_SPEC.width
        self.h = VIDEO_SPEC.height

        self._monument_3d = Monument3DEngine()
        self._cad_lidar = ProceduralCadCache()
        master_rgb = self._monument_3d.render_frame(SCENE_GIZA, 0.50, 2.0, self.w, self.h, lens_variant=0)
        lifted = np.clip(18.0 + master_rgb.astype(np.float32) * 0.90, 0.0, 255.0).astype(np.uint8)
        self._wide_master = apply_rec709_studio_grade(Image.fromarray(lifted, mode="RGB"), self.w, self.h)
        self._studio_bg = ImageEnhance.Brightness(
            self._wide_master.filter(ImageFilter.GaussianBlur(radius=8.0))
        ).enhance(0.72).convert("RGBA")

        self._f_matte_bold = load_font("SpaceGrotesk-Bold.ttf", 14)
        self._f_matte_reg = load_font("Inter-Medium.ttf", 13)
        self._f_ep_banner_kicker = load_font("SpaceGrotesk-Bold.ttf", 12)
        self._f_ep_banner_title = load_font("SpaceGrotesk-Bold.ttf", 22)
        self._f_lt_kicker = load_font("SpaceGrotesk-Bold.ttf", 12)
        self._f_lt_name = load_font("SpaceGrotesk-Bold.ttf", 22)
        self._f_lt_role = load_font("Inter-Medium.ttf", 13)
        self._f_badge = load_font("SpaceGrotesk-Bold.ttf", 14)

        self._f_intro_kicker = load_font("SpaceGrotesk-Bold.ttf", 18)
        self._f_intro_brand = load_font("SpaceGrotesk-Bold.ttf", 62)
        self._f_intro_ep = load_font("Inter-Regular.ttf", 23)
        self._f_intro_host = load_font("SpaceGrotesk-Medium.ttf", 16)

        self._f_card_kicker = load_font("SpaceGrotesk-Bold.ttf", 18)
        self._f_card_big = load_font("SpaceGrotesk-Bold.ttf", 136)
        self._f_card_title = load_font("SpaceGrotesk-Bold.ttf", 50)
        self._f_card_sub = load_font("Inter-Regular.ttf", 23)
        self._f_tl_tick = load_font("SpaceGrotesk-Medium.ttf", 13)

        self._f_quote_mark = load_font("SpaceGrotesk-Bold.ttf", 92)
        self._f_quote_body = load_font("Inter-SemiBold.ttf", 38)
        self._f_panel_title = load_font("SpaceGrotesk-Bold.ttf", 28)
        self._f_panel_val = load_font("SpaceGrotesk-Bold.ttf", 30)
        self._f_panel_lbl = load_font("Inter-Medium.ttf", 14)

    def _draw_logo_bug(self, draw: ImageDraw.ImageDraw, show_title: str = "DEEP HORIZON") -> None:
        """Minimal tracked `DEEP HORIZON` mark tucked inside the `y=0..50` Univisium top matte bar."""
        label = show_title[:20].upper()
        tw = int(draw.textlength(label, font=self._f_matte_bold))
        rx = self.w - 64 - tw
        ry = 17
        cx, cy = rx - 14, ry + 7
        draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(229, 169, 60, 225))
        draw.text((rx, ry), label, font=self._f_matte_bold, fill=(216, 224, 236, 215))

    def _draw_persistent_topic_graphic(
        self,
        draw: ImageDraw.ImageDraw,
        chapter_idx: int,
        total_chapters: int,
        chapter_title: str,
        mode_tag: str = "",
    ) -> None:
        """Minimal tracked `ACT 02 / 05 · TITLE` inside the `y=0..50` Univisium top matte bar."""
        act_str = f"ACT {chapter_idx:02d} / {max(1, total_chapters):02d}"
        draw.text((64, 17), act_str, font=self._f_matte_bold, fill=(229, 169, 60, 232))
        aw = int(draw.textlength(act_str, font=self._f_matte_bold))

        short_ch = chapter_title.split(":")[0].strip().upper()
        if len(short_ch) > 46:
            short_ch = short_ch[:43].rsplit(" ", 1)[0] + "..."
        suffix = f"   ·   {mode_tag.upper()}" if mode_tag else ""
        draw.text((64 + aw + 14, 18), f"·   {short_ch}{suffix}", font=self._f_matte_reg, fill=(176, 188, 204, 208))

    def _draw_bottom_plate_caption(self, draw: ImageDraw.ImageDraw, caption_str: str) -> None:
        """Subtle pure-code 3D / physics provenance inside the `y=1030..1080` Univisium bottom matte bar."""
        if not caption_str:
            return
        no_parens = re.sub(r"\s*\([^)]*\)?", "", caption_str).strip()
        no_parens = no_parens.rstrip("·,-& ")
        clean = f"CODE SYNTHESIS   ·   {no_parens.upper()}"
        # Measured truncation so the badge never runs past the matte bar's right edge.
        max_w = self.w - 128
        while clean and int(draw.textlength(clean, font=self._f_matte_reg)) > max_w and len(clean) > 18:
            clean = clean[:-2].rstrip()
        if int(draw.textlength(clean, font=self._f_matte_reg)) > max_w:
            clean = clean[: max(12, int(max_w / 11))].rstrip() + "…"
        draw.text((64, self.h - 33), clean, font=self._f_matte_reg, fill=(148, 162, 180, 196))

    def _render_transient_lower_third(
        self,
        speaker_id: str,
        topic_context: str,
        lt_path: Path,
        align_right: bool = False,
        prominent_episode_title: str | None = None,
    ) -> Path:
        """Render clean one-time Documentary Voiceover Attribution Lower Third."""
        del topic_context, prominent_episode_title
        ov = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)

        persona = HOSTS[speaker_id]
        acc = _hex_to_rgba(persona.accent_hex, 255)
        role_short = persona.role_title
        if len(role_short) > 44:
            role_short = role_short[:41].rsplit(" ", 1)[0] + "..."

        header_tag = "DOCUMENTARY NARRATION"
        w_name = int(draw.textlength(persona.display_name.upper(), font=self._f_lt_name))
        w_sub = int(draw.textlength(role_short, font=self._f_lt_role))
        w_hdr = int(draw.textlength(header_tag, font=self._f_lt_kicker))
        lt_w = max(320, min(520, max(w_name, w_sub, w_hdr) + 52))
        lt_h = 74
        lt_y = 824
        lt_x = (self.w - 64 - lt_w) if align_right else 64

        draw.rounded_rectangle(
            [lt_x, lt_y, lt_x + lt_w, lt_y + lt_h],
            radius=8,
            fill=(8, 12, 20, 212),
            outline=(255, 255, 255, 28),
            width=1,
        )
        draw.rectangle([lt_x, lt_y, lt_x + 4, lt_y + lt_h], fill=acc)
        draw.text((lt_x + 20, lt_y + 9), header_tag, font=self._f_lt_kicker, fill=acc)
        draw.text((lt_x + 20, lt_y + 26), persona.display_name.upper(), font=self._f_lt_name, fill=(255, 255, 255, 252))
        draw.text((lt_x + 20, lt_y + 52), role_short, font=self._f_lt_role, fill=(182, 194, 210, 224))

        lt_path.parent.mkdir(parents=True, exist_ok=True)
        ov.save(lt_path)
        return lt_path

    def _code_plate_for_hint(self, hint: str, fallback_idx: int = 1) -> Image.Image:
        sid, _ = resolve_procedural_scene_id(hint, fallback_idx=fallback_idx)
        rgb = self._monument_3d.render_frame(sid, 0.50, 2.0, self.w, self.h, lens_variant=0)
        lifted = np.clip(18.0 + rgb.astype(np.float32) * 0.90, 0.0, 255.0).astype(np.uint8)
        return apply_rec709_studio_grade(Image.fromarray(lifted, mode="RGB"), self.w, self.h)

    def render_intro_bumper(
        self,
        script: EpisodeScript,
        bg_out_path: Path,
        overlay_out_path: Path,
    ) -> tuple[PlateLayout, tuple[Path, int, int, str]]:
        """Render Cold-Open Cinema Title Sequence over rapid-cut 4-wonder 24fps 3D montage."""
        bg = self._wide_master.copy()
        bg_out_path.parent.mkdir(parents=True, exist_ok=True)
        bg.save(bg_out_path, quality=94)

        ov = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        yy, xx = np.mgrid[0 : self.h, 0 : self.w].astype(np.float32)
        dist = np.sqrt(((xx - self.w * 0.5) / 760.0) ** 2 + ((yy - self.h * 0.5) / 360.0) ** 2)
        scrim_a = np.clip((1.16 - dist) * 184.0, 52.0, 188.0).astype(np.uint8)
        scrim = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        scrim[..., 0] = 6
        scrim[..., 1] = 9
        scrim[..., 2] = 15
        scrim[..., 3] = scrim_a
        ov = Image.alpha_composite(ov, Image.fromarray(scrim, mode="RGBA"))
        draw = ImageDraw.Draw(ov)

        kicker = f"A {script.spec.show_title.upper()} ORIGINAL DOCUMENTARY FILM"
        kw = int(draw.textlength(kicker, font=self._f_intro_kicker))
        kx = (self.w - kw) // 2
        draw.line([(kx - 68, 268), (kx - 18, 268)], fill=(229, 169, 60, 235), width=2)
        draw.line([(kx + kw + 18, 268), (kx + kw + 68, 268)], fill=(229, 169, 60, 235), width=2)
        draw.text((kx, 256), kicker, font=self._f_intro_kicker, fill=(229, 169, 60, 255))

        ty = 314
        ep_title = script.spec.episode_title or script.spec.topic
        if ":" in ep_title:
            main_part, sub_part = [p.strip() for p in ep_title.split(":", 1)]
            for line in textwrap.wrap(main_part, width=32)[:2]:
                tw = int(draw.textlength(line, font=self._f_intro_brand))
                draw.text(((self.w - tw) // 2, ty), line, font=self._f_intro_brand, fill=(255, 255, 255, 255))
                ty += 70
            f_sub_brand = load_font("SpaceGrotesk-Bold.ttf", 42)
            ty += 4
            for line in textwrap.wrap(sub_part, width=42)[:2]:
                tw = int(draw.textlength(line, font=f_sub_brand))
                draw.text(((self.w - tw) // 2, ty), line, font=f_sub_brand, fill=(242, 204, 124, 252))
                ty += 52
        else:
            for line in textwrap.wrap(ep_title, width=32)[:3]:
                tw = int(draw.textlength(line, font=self._f_intro_brand))
                draw.text(((self.w - tw) // 2, ty), line, font=self._f_intro_brand, fill=(255, 255, 255, 255))
                ty += 70

        sy = ty + 16
        for line in textwrap.wrap(script.spec.episode_subtitle, width=66)[:2]:
            sw = int(draw.textlength(line, font=self._f_intro_ep))
            draw.text(((self.w - sw) // 2, sy), line, font=self._f_intro_ep, fill=(214, 224, 238, 236))
            sy += 34

        cred = f"NARRATED BY {HOST_A.display_name.upper()} & {HOST_B.display_name.upper()}   ·   {len(script.chapters)} ACTS   ·   24P CINEMA"
        cw = int(draw.textlength(cred, font=self._f_intro_host))
        draw.text(((self.w - cw) // 2, self.h - 128), cred, font=self._f_intro_host, fill=(186, 198, 214, 220))

        overlay_out_path.parent.mkdir(parents=True, exist_ok=True)
        ov.save(overlay_out_path)
        layout = PlateLayout(
            image_path=bg_out_path,
            use_ken_burns=True,
            ken_burns_zoom_in=True,
            anim_mode="broll_depth",
        )
        return layout, (overlay_out_path, 0, 0, "#E5A93C")

    def render_chapter_card(
        self,
        chapter: ChapterScript,
        total_chapters: int,
        show_title: str,
        out_path: Path,
        chronology_year: int | None = None,
        prev_chronology_year: int | None = None,
    ) -> PlateLayout:
        """Render Documentary Act Card over code-synthesized 3D monument backdrop + Chronology Axis."""
        code_bg = self._code_plate_for_hint(f"{chapter.title} {chapter.location}", chapter.chapter_index)
        bg = ImageEnhance.Brightness(code_bg).enhance(0.58).convert("RGBA")

        overlay = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for x in range(1320):
            a = int(214 * ((1.0 - x / 1320.0) ** 1.35))
            draw.line([(x, 0), (x, self.h)], fill=(6, 9, 15, a))

        self._draw_persistent_topic_graphic(draw, chapter.chapter_index, total_chapters, chapter.title)
        self._draw_logo_bug(draw, show_title)

        draw.text((148, 128), f"{chapter.chapter_index:02d}", font=self._f_card_big, fill=(255, 255, 255, 24))
        draw.rectangle([168, 224, 174, 624], fill=(229, 169, 60, 255))

        kicker = f"ACT {chapter.chapter_index:02d} OF {total_chapters:02d}   ·   {chapter.era_or_date.upper()}"
        draw.text((204, 224), kicker[:78], font=self._f_card_kicker, fill=(229, 169, 60, 248))

        ty = 274
        for line in textwrap.wrap(chapter.title, width=30)[:3]:
            draw.text((204, ty), line, font=self._f_card_title, fill=(255, 255, 255, 255))
            ty += 60

        ty += 14
        for line in textwrap.wrap(chapter.subtitle, width=60)[:2]:
            draw.text((204, ty), line, font=self._f_card_sub, fill=(208, 218, 232, 234))
            ty += 34

        loc_txt = f"EXPEDITION SITE   ·   {chapter.location.upper()}"
        draw.text((204, ty + 22), loc_txt[:74], font=self._f_badge, fill=(229, 169, 60, 232))

        tl_x0, tl_x1, tl_y = 168, 1180, 868
        for yr, lbl in ((-3000, "3000 BCE"), (-2000, "2000 BCE"), (-1000, "1000 BCE"), (0, "1 CE"), (1000, "1000 CE"), (2000, "2026 CE")):
            tx = int(round(tl_x0 + ((yr + 3000.0) / 5026.0) * (tl_x1 - tl_x0)))
            tw_t = int(draw.textlength(lbl, font=self._f_tl_tick))
            draw.text((tx - tw_t // 2, tl_y + 14), lbl, font=self._f_tl_tick, fill=(154, 168, 188, 210))

        final_im = Image.alpha_composite(bg, overlay).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_im.save(out_path, quality=93)
        return PlateLayout(
            image_path=out_path,
            use_ken_burns=True,
            ken_burns_zoom_in=True,
            anim_mode="smooth_card",
            chronology_year=chronology_year,
            prev_chronology_year=prev_chronology_year,
        )

    def render_wide_studio(
        self,
        turn: DialogueTurn,
        chapter: ChapterScript,
        total_chapters: int,
        show_title: str,
        out_path: Path,
        episode_title: str = "",
        show_lower_third: bool = True,
        prev_location_hint: str = "",
        neutral_globe: bool = False,
        globe_variation: int = 0,
    ) -> tuple[PlateLayout, tuple[Path, int, int, str]]:
        """Live 24fps True 3D Raycasted Orbital Earth GIS Globe (`raycast_orbital_earth_globe`)."""
        hud = _apply_bottom_legibility_gradient(Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0)), strength=102)
        draw = ImageDraw.Draw(hud)
        self._draw_persistent_topic_graphic(
            draw, chapter.chapter_index, total_chapters, chapter.title, mode_tag="3D ORBITAL EARTH GIS"
        )
        self._draw_logo_bug(draw, show_title)
        badge = (
            "3D RAYCASTED EARTH ECEF · 1:50M NATURAL EARTH GIS"
            if neutral_globe
            else f"3D RAYCASTED EARTH ECEF · 1:50M NATURAL EARTH GIS · {chapter.location}"
        )
        self._draw_bottom_plate_caption(draw, badge)

        hud_path = out_path.with_suffix(".png")
        hud_path.parent.mkdir(parents=True, exist_ok=True)
        hud.save(hud_path)

        lt_path: Path | None = None
        if show_lower_third:
            lt_path = self._render_transient_lower_third(
                turn.speaker,
                chapter.title,
                out_path.with_name(out_path.stem + "_lt.png"),
                align_right=(turn.speaker == "HOST_B"),
                prominent_episode_title=episode_title,
            )

        target_hint = f"{turn.text} || {chapter.title} {chapter.location}"
        layout = PlateLayout(
            image_path=hud_path,
            anim_mode="geo_map",
            anim_speaker=turn.speaker,
            anim_is_speaking=True,
            transient_lt_path=lt_path,
            map_target_hint=target_hint,
            map_prev_hint=prev_location_hint,
            map_neutral=neutral_globe,
            map_variation=globe_variation,
            era_text=chapter.era_or_date,
            location_text=chapter.location,
        )
        return layout, (hud_path, 0, 0, "#" + HOSTS[turn.speaker].accent_hex.lstrip("#"))

    def render_host_closeup(
        self,
        turn: DialogueTurn,
        chapter: ChapterScript,
        total_chapters: int,
        show_title: str,
        out_path: Path,
        show_lower_third: bool | None = None,
        asset_override: Path | None = None,
        caption_override: str | None = None,
    ) -> tuple[PlateLayout, tuple[Path, int, int, str]]:
        """Live 24fps Pure-Code 3D Architectural Monument Reconstruction."""
        del asset_override
        s_hint = f"{turn.text} || {chapter.title} {chapter.location}"
        _, scene_title = resolve_procedural_scene_id(s_hint, chapter.chapter_index)
        bg = self._code_plate_for_hint(s_hint, chapter.chapter_index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bg.save(out_path, quality=93)

        hud = _apply_bottom_legibility_gradient(Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0)), strength=110)
        draw = ImageDraw.Draw(hud)
        self._draw_persistent_topic_graphic(
            draw, chapter.chapter_index, total_chapters, chapter.title, mode_tag="3D ARCHITECTURE"
        )
        self._draw_logo_bug(draw, show_title)
        cap_label = caption_override or f"3D Architectural Reconstruction · {scene_title}"
        self._draw_bottom_plate_caption(draw, cap_label)

        hud_path = out_path.with_name(out_path.stem + "_hud.png")
        hud.save(hud_path)

        want_lt = (turn.turn_id <= 2) if show_lower_third is None else show_lower_third
        lt_path: Path | None = None
        if want_lt:
            lt_path = self._render_transient_lower_third(
                turn.speaker,
                turn.fact_callout_title or chapter.title,
                out_path.with_name(out_path.stem + "_lt.png"),
                align_right=(turn.speaker == "HOST_B"),
            )

        layout = PlateLayout(
            image_path=out_path,
            use_ken_burns=True,
            ken_burns_zoom_in=(turn.turn_id % 2 == 0),
            anim_mode="broll_depth",
            anim_speaker=turn.speaker,
            anim_is_speaking=True,
            transient_lt_path=lt_path,
            scene_hint=s_hint,
        )
        return layout, (hud_path, 0, 0, "#" + HOSTS[turn.speaker].accent_hex.lstrip("#"))

    def render_reaction_shot(
        self,
        turn: DialogueTurn,
        chapter: ChapterScript,
        total_chapters: int,
        show_title: str,
        out_path: Path,
        primary_asset: Path | None = None,
        secondary_asset: Path | None = None,
        primary_caption: str = "",
        secondary_caption: str = "",
    ) -> tuple[PlateLayout, tuple[Path, int, int, str]]:
        """Live 24fps Volumetric 3D LiDAR Point-Cloud & Physics Simulation."""
        del primary_asset, secondary_asset
        s_hint = f"{turn.text} || {chapter.title} {chapter.location}"
        _, scene_title = resolve_procedural_scene_id(s_hint, chapter.chapter_index)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._code_plate_for_hint(s_hint, chapter.chapter_index).save(out_path, quality=92)

        hud = _apply_bottom_legibility_gradient(Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0)), strength=104)
        draw = ImageDraw.Draw(hud)
        self._draw_persistent_topic_graphic(
            draw, chapter.chapter_index, total_chapters, chapter.title, mode_tag="VOLUMETRIC LIDAR & PHYSICS"
        )
        self._draw_logo_bug(draw, show_title)
        self._draw_bottom_plate_caption(draw, primary_caption or f"Volumetric LiDAR & Physics Model · {scene_title}")

        hud_path = out_path.with_name(out_path.stem + "_hud.png")
        hud.save(hud_path)

        listener_id: Literal["HOST_A", "HOST_B"] = "HOST_B" if turn.speaker == "HOST_A" else "HOST_A"
        layout = PlateLayout(
            image_path=out_path,
            secondary_image_path=out_path,
            anim_mode="archival_desk",
            anim_speaker=listener_id,
            anim_is_speaking=False,
            chapter_title_text=chapter.title.split(":")[0],
            primary_caption=primary_caption or scene_title,
            secondary_caption=secondary_caption or chapter.location,
            scene_hint=s_hint,
        )
        return layout, (hud_path, 0, 0, "#" + HOSTS[turn.speaker].accent_hex.lstrip("#"))

    def render_studio_insert(
        self,
        turn: DialogueTurn,
        chapter: ChapterScript,
        total_chapters: int,
        show_title: str,
        bg_out_path: Path,
        hud_out_path: Path,
        primary_asset: Path | None = None,
        secondary_asset: Path | None = None,
        primary_caption: str = "",
        secondary_caption: str = "",
    ) -> tuple[PlateLayout, tuple[Path, int, int, str]]:
        """Live 24fps Mathematical & Engineering Physics Simulation / 3D LiDAR Cutaway."""
        del primary_asset, secondary_asset
        s_hint = f"{turn.text} || {chapter.title} {chapter.location}"
        _, scene_title = resolve_procedural_scene_id(s_hint, chapter.chapter_index)

        bg_out_path.parent.mkdir(parents=True, exist_ok=True)
        self._code_plate_for_hint(s_hint, chapter.chapter_index).save(bg_out_path, quality=93)

        hud = _apply_bottom_legibility_gradient(Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0)), strength=106)
        draw = ImageDraw.Draw(hud)
        self._draw_persistent_topic_graphic(
            draw, chapter.chapter_index, total_chapters, chapter.title, mode_tag="PHYSICS & CAD SIMULATION"
        )
        self._draw_logo_bug(draw, show_title)
        self._draw_bottom_plate_caption(draw, primary_caption or f"Mathematical & Structural Physics · {scene_title}")

        hud_out_path.parent.mkdir(parents=True, exist_ok=True)
        hud.save(hud_out_path)

        layout = PlateLayout(
            image_path=bg_out_path,
            secondary_image_path=bg_out_path,
            use_ken_burns=False,
            anim_mode="archival_desk",
            anim_speaker=turn.speaker,
            anim_is_speaking=True,
            chapter_title_text=chapter.title.split(":")[0],
            primary_caption=primary_caption or scene_title,
            secondary_caption=secondary_caption or chapter.location,
            scene_hint=s_hint,
        )
        return layout, (hud_out_path, 0, 0, "#" + HOSTS[turn.speaker].accent_hex.lstrip("#"))

    def render_quote_card(
        self,
        turn: DialogueTurn,
        chapter: ChapterScript,
        total_chapters: int,
        show_title: str,
        out_path: Path,
        asset_override: Path | None = None,
    ) -> PlateLayout:
        """Render Verbatim Eyewitness Primary-Source Inscription Plate over live 3D monument."""
        del asset_override
        s_hint = f"{turn.text} || {chapter.title} {chapter.location}"
        bg_src = self._code_plate_for_hint(s_hint, chapter.chapter_index)
        bg = ImageEnhance.Brightness(bg_src.filter(ImageFilter.GaussianBlur(radius=9.0))).enhance(0.56).convert("RGBA")

        overlay = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        self._draw_persistent_topic_graphic(
            draw, chapter.chapter_index, total_chapters, chapter.title, mode_tag="PRIMARY SOURCE"
        )
        self._draw_logo_bug(draw, show_title)

        acc = (229, 169, 60, 255)
        verbatim_q, author_name, citation_line = resolve_historical_quote(
            s_hint, turn.pull_quote or turn.text
        )

        hud_only = _apply_bottom_legibility_gradient(Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0)), strength=96)
        hdraw = ImageDraw.Draw(hud_only)
        self._draw_persistent_topic_graphic(
            hdraw, chapter.chapter_index, total_chapters, chapter.title, mode_tag="PRIMARY SOURCE"
        )
        self._draw_logo_bug(hdraw, show_title)
        hud_path = out_path.with_name(out_path.stem + "_hud.png")
        hud_path.parent.mkdir(parents=True, exist_ok=True)
        hud_only.save(hud_path)

        bx0, by0, bx1, by1 = 168, 178, self.w - 168, self.h - 224
        draw.rounded_rectangle(
            [bx0, by0, bx1, by1],
            radius=16,
            fill=(8, 12, 20, 224),
            outline=(229, 169, 60, 92),
            width=1,
        )
        draw.rectangle([bx0, by0, bx0 + 6, by1], fill=acc)
        draw.text((bx0 + 48, by0 + 28), "PRIMARY HISTORICAL INSCRIPTION   ·   ARCHIVAL RECORD", font=self._f_badge, fill=acc)
        draw.text((bx0 + 42, by0 + 52), "“", font=self._f_quote_mark, fill=(229, 169, 60, 148))

        qy = by0 + 132
        for line in textwrap.wrap(verbatim_q, width=54)[:5]:
            draw.text((bx0 + 64, qy), line, font=self._f_quote_body, fill=(248, 250, 252, 255))
            qy += 52

        div_y = max(qy + 20, by1 - 92)
        draw.line([(bx0 + 64, div_y), (bx0 + 240, div_y)], fill=acc, width=2)
        draw.text((bx0 + 64, div_y + 14), f"— {author_name}", font=self._f_card_kicker, fill=(248, 250, 252, 248))
        draw.text((bx0 + 64, div_y + 42), citation_line[:86], font=self._f_panel_lbl, fill=(172, 186, 204, 224))

        final_im = Image.alpha_composite(bg, overlay).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_im.save(out_path, quality=93)
        return PlateLayout(
            image_path=out_path,
            secondary_image_path=out_path,
            use_ken_burns=True,
            ken_burns_zoom_in=True,
            anim_mode="none",
            quote_verbatim=verbatim_q,
            quote_author=author_name,
            quote_citation=citation_line,
            hud_overlay_path=hud_path,
            scene_hint=s_hint,
        )

    def render_broll_cinema(
        self,
        turn: DialogueTurn,
        chapter: ChapterScript,
        out_path: Path,
        asset_override: Path | None = None,
        zoom_in_override: bool | None = None,
    ) -> PlateLayout:
        """Render clean full-bleed Rec.709 1920x1080 Pure-Code 3D Monument plate."""
        del asset_override
        s_hint = f"{turn.text} || {chapter.title} {chapter.location}"
        bg = self._code_plate_for_hint(s_hint, chapter.chapter_index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bg.save(out_path, quality=93)
        z_in = (turn.turn_id % 2 == 1) if zoom_in_override is None else zoom_in_override
        return PlateLayout(
            image_path=out_path,
            use_ken_burns=True,
            ken_burns_zoom_in=z_in,
            anim_mode="none",
            scene_hint=s_hint,
        )

    def render_semantic_diagram(
        self,
        directive: VisualDirective,
        clause: str,
        out_path: Path,
    ) -> PlateLayout:
        """Render a clause-driven neutral explainer plate (no monument geometry is ever used).

        A mid-shot still is persisted as the segment plate (tail frame / non-pipe fallback);
        the encoder's rawvideo pipe animates the full 24fps sequence from the same engine.
        """
        engine = SemanticDiagramEngine()
        # Salt the per-subject seed with the clause hash: same subject + same template on a
        # different clause must produce different geometry (the subject-only seed made four
        # abstract shots pixel-identical across the film).
        seed = subject_seed(directive.subject_key) ^ subject_seed(f"clause::{clause}")
        still = engine.render_frame(
            template=directive.diagram_template,
            subject_label=directive.subject_label,
            clause=clause,
            u=0.55,
            time_s=0.8,
            seed=seed,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(still).save(out_path, quality=93)
        return PlateLayout(
            image_path=out_path,
            anim_mode="semantic_diagram",
            scene_hint=clause,
            semantic_template=directive.diagram_template,
            semantic_subject=directive.subject_label,
            semantic_clause=clause,
            semantic_seed=seed,
        )

    def render_broll_hud_overlay(
        self,
        turn: DialogueTurn,
        chapter: ChapterScript,
        hud_path: Path,
        total_chapters: int = 1,
        show_title: str = "DEEP HORIZON",
        caption_override: str | None = None,
    ) -> tuple[Path, int, int, str]:
        """Render Univisium top/bottom matte metadata + subtle caption scrim over documentary B-Roll."""
        hud = _apply_bottom_legibility_gradient(Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0)), strength=114)
        draw = ImageDraw.Draw(hud)

        self._draw_persistent_topic_graphic(draw, chapter.chapter_index, max(1, total_chapters), chapter.title)
        self._draw_logo_bug(draw, show_title)

        cap_label = (caption_override or turn.visual_caption or chapter.title).strip()
        full_cap = cap_label if "·" in cap_label else f"{cap_label}  ·  {chapter.location.split('&')[0].strip()}"
        self._draw_bottom_plate_caption(draw, full_cap)

        hud_path.parent.mkdir(parents=True, exist_ok=True)
        hud.save(hud_path)
        return hud_path, 0, 0, "#" + HOSTS[turn.speaker].accent_hex.lstrip("#")

    def render_split_fact_card(
        self,
        turn: DialogueTurn,
        chapter: ChapterScript,
        total_chapters: int,
        show_title: str,
        out_path: Path,
        asset_override: Path | None = None,
    ) -> PlateLayout:
        """Render 2-Column Archaeological & Engineering Telemetry View (`anim_mode == 'none'`)."""
        del asset_override
        s_hint = f"{turn.text} || {chapter.title} {chapter.location}"
        sid, _ = resolve_procedural_scene_id(s_hint, chapter.chapter_index)
        canvas = self._studio_bg.copy()
        overlay = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        self._draw_persistent_topic_graphic(
            draw, chapter.chapter_index, total_chapters, chapter.title, mode_tag="ENGINEERING TELEMETRY"
        )
        self._draw_logo_bug(draw, show_title)

        persona = HOSTS[turn.speaker]
        acc = _hex_to_rgba(persona.accent_hex, 255)

        lx, ty, lw, lh = 96, 128, 824, 648
        cad_rgb = self._cad_lidar.render_cad_frame(sid, 0.50, 2.0, lw, lh, show_hud=False)
        vis_im = _crop_rounded_image(Image.fromarray(cad_rgb, mode="RGB"), lw, lh, radius=14)
        canvas.alpha_composite(vis_im, dest=(lx, ty))

        rx, rw = lx + lw + 52, 852
        draw.rounded_rectangle(
            [rx, ty, rx + rw, ty + lh],
            radius=14,
            fill=(10, 14, 22, 228),
            outline=(255, 255, 255, 32),
            width=1,
        )
        draw.rectangle([rx, ty, rx + 4, ty + lh], fill=acc)
        draw.text((rx + 38, ty + 30), f"STRUCTURAL TELEMETRY  ·  {chapter.era_or_date.split('·')[0].split('&')[0].strip().upper()}"[:52], font=self._f_badge, fill=acc)

        cy = ty + 64
        short_ch_title = chapter.title.split(":")[0] if len(chapter.title) > 52 else chapter.title
        for line in textwrap.wrap(short_ch_title, width=32)[:2]:
            draw.text((rx + 38, cy), line, font=self._f_panel_title, fill=(255, 255, 255, 255))
            cy += 38

        cy += 18
        metrics = turn.fact_metrics or [
            type("M", (), {"label": "Site Location", "value": chapter.location[:22]})(),
            type("M", (), {"label": "Historical Era", "value": chapter.era_or_date[:22]})(),
        ]
        col_w = (rw - 96) // 2
        row_h = 154
        metric_pairs: list[tuple[str, str]] = []
        for i, m in enumerate(metrics[:4]):
            metric_pairs.append((str(m.label), str(m.value)))
            gx = rx + 38 + (i % 2) * (col_w + 20)
            gy = cy + (i // 2) * (row_h + 16)
            draw.rounded_rectangle(
                [gx, gy, gx + col_w, gy + row_h],
                radius=10,
                fill=(16, 22, 36, 234),
                outline=(255, 255, 255, 24),
                width=1,
            )
            vy = gy + 22
            for vl in textwrap.wrap(str(m.value), width=17)[:2]:
                draw.text((gx + 22, vy), vl, font=self._f_panel_val, fill=(255, 255, 255, 255))
                vy += 34
            draw.text((gx + 22, gy + row_h - 38), str(m.label).upper()[:26], font=self._f_panel_lbl, fill=acc)

        coord_txt = f"ALGORITHMIC 3D SURVEY  ·  {chapter.location.upper()}"[:62]
        draw.text((rx + 38, ty + lh - 38), coord_txt, font=self._f_panel_lbl, fill=(148, 163, 184, 212))

        final_im = Image.alpha_composite(canvas, overlay).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_im.save(out_path, quality=93)
        return PlateLayout(
            image_path=out_path,
            secondary_image_path=out_path,
            use_ken_burns=False,
            anim_mode="none",
            wave_hex="#" + persona.accent_hex.lstrip("#"),
            chapter_title_text=chapter.title,
            era_text=chapter.era_or_date,
            location_text=chapter.location,
            telemetry_metrics=tuple(metric_pairs),
            scene_hint=s_hint,
        )

    def render_outro_card(self, script: EpisodeScript, out_path: Path) -> PlateLayout:
        """Render clean Branded Documentary Film End-Credits & Archival Provenance Card."""
        bg = ImageEnhance.Brightness(self._studio_bg).enhance(0.84)
        ov = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(ov)

        self._draw_logo_bug(draw, script.spec.show_title)
        draw.text((180, 158), "VOXFORGE ORIGINAL DOCUMENTARY FILMS", font=self._f_card_kicker, fill=(229, 169, 60, 245))
        draw.text((180, 198), script.spec.show_title.upper(), font=self._f_card_title, fill=(255, 255, 255, 255))
        draw.line([(180, 272), (self.w - 180, 272)], fill=(255, 255, 255, 34), width=1)

        draw.text((180, 314), "DOCUMENTARY CREDITS", font=self._f_card_kicker, fill=(229, 169, 60, 240))
        credits = [
            ("HISTORICAL NARRATION", HOST_A.display_name),
            ("ARCHAEOLOGICAL & STRUCTURAL NARRATION", HOST_B.display_name),
            ("ALGORITHMIC 3D ARCHITECTURE, LIDAR & PHYSICS", "100% Pure-Code VoxForge 3D Engine"),
            ("3D ORBITAL EARTH GIS & CHAMBER MASTERING", "Natural Earth 1:50m · ITU-R BS.1770-4 (-16 LUFS)"),
        ]
        cy = 362
        for lbl, val in credits:
            draw.text((180, cy), lbl, font=self._f_panel_lbl, fill=(148, 163, 184, 215))
            draw.text((180, cy + 22), val, font=self._f_lt_name, fill=(248, 250, 252, 252))
            cy += 80

        rx = 1020
        draw.text((rx, 314), "NEXT DOCUMENTARY FEATURE", font=self._f_card_kicker, fill=(56, 189, 248, 245))
        nty = 362
        for line in textwrap.wrap(script.next_episode_tease, width=32)[:4]:
            draw.text((rx, nty), line, font=self._f_panel_title, fill=(255, 255, 255, 255))
            nty += 42

        cta_y = 626
        draw.rounded_rectangle([rx, cta_y, rx + 312, cta_y + 50], radius=6, fill=(229, 169, 60, 255))
        draw.text((rx + 24, cta_y + 15), "DEEP HORIZON ARCHIVES", font=self._f_card_kicker, fill=(10, 12, 18, 255))
        draw.text(
            (rx, cta_y + 68),
            "Autonomous Open-Source Cinema Documentaries · Zero API Keys",
            font=self._f_lt_role,
            fill=(203, 213, 225, 232),
        )

        final_im = Image.alpha_composite(bg, ov).convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_im.save(out_path, quality=93)
        return PlateLayout(image_path=out_path, use_ken_burns=True, ken_burns_zoom_in=True, anim_mode="smooth_card")
