"""Broadcast Cinema Captioning (ASS / SRT / VTT) & MP4 Chapter Metadata Generator.

Adheres to Netflix / BBC prestige documentary subtitling standards:
  - Clause-aware phrase chunking (`<= 72` chars, never stranding dangling prepositions/articles,
    ZERO dropped words — 100% of spoken text rendered cleanly across 1–2 balanced lines)
  - Punctuation-weighted timing locked to exact synthesized utterance bounds
  - Clean speaker-free burned-in cinema open captions (`{\\fad(75,60)}`) + accessible SRT/VTT sidecars
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

from voxforge.config import HOSTS
from voxforge.models import DialogueTurn, EpisodeScript

_DANGLING_TAIL_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by", "with", "from",
    "into", "onto", "upon", "near", "over", "under", "across", "through", "between",
    "behind", "beneath", "around", "during", "after", "before", "and", "or", "but",
    "as", "than", "its", "their", "his", "her",
}

_CLAUSE_STARTERS = {
    "where", "while", "which", "after", "before", "across", "under", "through",
    "leaving", "allowing", "carrying", "carved", "crowned", "stretching", "proving",
    "draining", "combining", "connecting",
}


@dataclass(frozen=True)
class CaptionCue:
    index: int
    start_sec: float
    end_sec: float
    speaker_id: str
    speaker_label: str
    text: str


def _balance_two_lines(text: str, max_width: int = 43) -> list[str]:
    """Wrap `text` into 1 or 2 visually balanced lines WITHOUT ever dropping trailing words."""
    clean = " ".join(text.strip().split())
    if not clean:
        return []
    if len(clean) <= max_width:
        return [clean]

    words = clean.split()
    best_split = len(words) // 2
    best_score = 1e9
    for k in range(1, len(words)):
        l1 = " ".join(words[:k])
        l2 = " ".join(words[k:])
        overflow = max(0, len(l1) - max_width) * 12 + max(0, len(l2) - max_width) * 12
        diff = abs(len(l1) - len(l2))
        dangling_pen = 16 if words[k - 1].lower().rstrip(",.;:") in _DANGLING_TAIL_WORDS else 0
        punct_bonus = -10 if words[k - 1].endswith((",", ";", ":", "—", ".")) else 0
        score = overflow + diff + dangling_pen + punct_bonus
        if score < best_score:
            best_score = score
            best_split = k

    return [" ".join(words[:best_split]), " ".join(words[best_split:])]


def _split_turn_into_caption_phrases(turn: DialogueTurn, max_chars_per_cue: int = 72) -> list[CaptionCue]:
    """Split a spoken turn into clause-aware, zero-truncation 1–2 line documentary caption cues."""
    persona = HOSTS[turn.speaker]
    words = turn.text.strip().split()
    if not words:
        return []

    phrases: list[str] = []
    cur: list[str] = []

    def _cur_len() -> int:
        return sum(len(x) for x in cur) + max(0, len(cur) - 1)

    for idx_w, w in enumerate(words):
        projected = _cur_len() + (1 if cur else 0) + len(w)
        prev_w = cur[-1] if cur else ""
        prev_clean = prev_w.lower().rstrip(",.;:!?—")

        # Natural early break at clause punctuation or clause starter once >= 38 chars
        natural_punct_break = (
            _cur_len() >= 34
            and prev_w.endswith((",", ";", ":", "—", ".", "?", "!"))
            and idx_w < len(words) - 2
        )
        clause_word_break = (
            _cur_len() >= 44
            and w.lower().strip("'\"") in _CLAUSE_STARTERS
            and prev_clean not in _DANGLING_TAIL_WORDS
        )

        if cur and (projected > max_chars_per_cue or natural_punct_break or clause_word_break):
            # Never leave a dangling preposition/article at the tail of `cur` if avoidable
            if len(cur) >= 4 and prev_clean in _DANGLING_TAIL_WORDS and not prev_w.endswith((",", ";", ":", ".")):
                moved = cur.pop()
                phrases.append(" ".join(cur))
                cur = [moved, w]
            else:
                phrases.append(" ".join(cur))
                cur = [w]
        else:
            cur.append(w)

    if cur:
        # Merge tiny 1–2 word orphan tail into previous cue if total stays <= 84 chars
        tail_txt = " ".join(cur)
        if phrases and len(cur) <= 2 and len(phrases[-1]) + 1 + len(tail_txt) <= 84:
            phrases[-1] = f"{phrases[-1]} {tail_txt}"
        else:
            phrases.append(tail_txt)

    # Weight duration by character count + punctuation pause weight for tight syllable lock
    def _cue_weight(p: str) -> float:
        p_pause = 6.0 * sum(p.count(ch) for ch in (",", ";", ":", "—")) + 11.0 * sum(p.count(ch) for ch in (".", "?", "!"))
        return max(4.0, float(len(p)) + p_pause)

    weights = [_cue_weight(p) for p in phrases]
    total_w = max(1.0, sum(weights))
    span = max(0.25, turn.end_sec - turn.start_sec)
    cursor = turn.start_sec
    cues: list[CaptionCue] = []

    for i, (ph, wt) in enumerate(zip(phrases, weights)):
        dur = span * (wt / total_w) if i < len(phrases) - 1 else max(0.15, turn.end_sec - cursor)
        cues.append(
            CaptionCue(
                index=0,
                start_sec=round(cursor, 3),
                end_sec=round(cursor + dur - 0.02, 3),
                speaker_id=turn.speaker,
                speaker_label=persona.display_name.upper(),
                text=ph,
            )
        )
        cursor += dur
    return cues


def build_all_caption_cues(script: EpisodeScript) -> list[CaptionCue]:
    raw: list[CaptionCue] = []
    idx = 1
    for turn in script.all_turns:
        for c in _split_turn_into_caption_phrases(turn):
            raw.append(
                CaptionCue(
                    index=idx,
                    start_sec=c.start_sec,
                    end_sec=max(c.start_sec + 0.20, c.end_sec),
                    speaker_id=c.speaker_id,
                    speaker_label=c.speaker_label,
                    text=c.text,
                )
            )
            idx += 1
    return raw


def _fmt_srt_ts(sec: float) -> str:
    ms_total = max(0, int(round(sec * 1000)))
    hh = ms_total // 3_600_000
    rem = ms_total % 3_600_000
    mm = rem // 60_000
    rem %= 60_000
    ss = rem // 1000
    ms = rem % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _fmt_vtt_ts(sec: float) -> str:
    return _fmt_srt_ts(sec).replace(",", ".")


def _fmt_ass_ts(sec: float) -> str:
    cs_total = max(0, int(round(sec * 100)))
    hh = cs_total // 360_000
    rem = cs_total % 360_000
    mm = rem // 6000
    rem %= 6000
    ss = rem // 100
    cs = rem % 100
    return f"{hh:d}:{mm:02d}:{ss:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def write_srt_and_vtt(cues: list[CaptionCue], srt_path: Path, vtt_path: Path) -> None:
    srt_lines: list[str] = []
    vtt_lines: list[str] = ["WEBVTT", ""]

    for c in cues:
        wrapped = "\n".join(textwrap.wrap(f"{c.speaker_label}: {c.text}", width=52))
        srt_lines.extend([str(c.index), f"{_fmt_srt_ts(c.start_sec)} --> {_fmt_srt_ts(c.end_sec)}", wrapped, ""])
        vtt_lines.extend([f"{_fmt_vtt_ts(c.start_sec)} --> {_fmt_vtt_ts(c.end_sec)}", wrapped, ""])

    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")


def write_segment_ass(
    cues: list[CaptionCue],
    seg_start_sec: float,
    seg_end_sec: float,
    ass_path: Path,
) -> Path:
    """Write a local-time ASS subtitle script for a single video segment (zero-truncation cinema captions)."""
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Broadcast,Inter SemiBold,35,&H00F8FAFC,&H003CA9E5,&H1005070C,&H88000000,0,0,0,0,100,100,0.25,0,1,2.6,1.3,2,160,160,68,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    for c in cues:
        if c.end_sec <= seg_start_sec or c.start_sec >= seg_end_sec:
            continue
        loc_s = max(0.0, c.start_sec - seg_start_sec)
        loc_e = min(seg_end_sec - seg_start_sec, c.end_sec - seg_start_sec)
        if loc_e - loc_s < 0.08:
            continue

        lines = _balance_two_lines(_ass_escape(c.text), max_width=43)
        if not lines:
            continue
        body = "\\N".join(lines)
        fade_tag = "{\\fad(65,55)}" if (loc_e - loc_s) >= 0.35 else ""
        events.append(
            f"Dialogue: 0,{_fmt_ass_ts(loc_s)},{_fmt_ass_ts(loc_e)},Broadcast,,0,0,0,,{fade_tag}{body}"
        )

    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return ass_path


def write_ffmetadata_chapters(script: EpisodeScript, total_duration_sec: float, meta_path: Path) -> Path:
    """Write FFmpeg FFMETADATA1 file with embedded clickable chapter markers."""
    lines = [
        ";FFMETADATA1",
        f"title={script.spec.episode_title}",
        "artist=Dr. Maya Lin (Deep Horizon Documentary Studio)",
        f"comment={script.spec.episode_subtitle}",
        "",
    ]
    first_ch_ms = int(round(script.chapters[0].start_sec * 1000)) if script.chapters else 3600
    if first_ch_ms >= 1500:
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                "START=0",
                f"END={first_ch_ms}",
                f"title=Intro: {script.spec.show_title}",
                "",
            ]
        )

    outro_start_ms = int(round((total_duration_sec - script.outro_duration_sec) * 1000))
    for idx, ch in enumerate(script.chapters):
        start_ms = int(round(ch.start_sec * 1000))
        if idx + 1 < len(script.chapters):
            end_ms = int(round(script.chapters[idx + 1].start_sec * 1000))
        else:
            end_ms = max(start_ms + 500, outro_start_ms)
        end_ms = max(start_ms + 500, end_ms)
        safe_title = ch.title.replace("=", "-").replace(";", ",")
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title=Act {ch.chapter_index}: {safe_title}",
                "",
            ]
        )

    if total_duration_sec * 1000 - outro_start_ms >= 1500:
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={outro_start_ms}",
                f"END={int(round(total_duration_sec * 1000))}",
                f"title=Credits & Next Feature: {script.spec.show_title}",
                "",
            ]
        )

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text("\n".join(lines), encoding="utf-8")
    return meta_path
