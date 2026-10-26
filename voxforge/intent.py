"""Natural-language single-command parser for VoxForge Studio.

Translates plain-English user commands such as:
  "build a 1 hour long podcast on 7 wonders of the world"
  "create a 30 minute deep dive video podcast about the Silk Road"
  "make a 5 min podcast exploring black holes"
into a calibrated ShowSpec with exact duration, word-count, and act budgets.
"""

from __future__ import annotations

import math
import re

from voxforge.models import ShowSpec

_WORD_NUMBERS: dict[str, float] = {
    "a": 1.0,
    "an": 1.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "twelve": 12.0,
    "fifteen": 15.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "forty": 40.0,
    "forty-five": 45.0,
    "fifty": 50.0,
    "sixty": 60.0,
    "ninety": 90.0,
    "half": 0.5,
}

_HR_MIN_PATTERN = re.compile(
    r"\b(?P<hr>\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\s*(?:and\s*)?(?P<min>\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m)\b",
    re.IGNORECASE,
)
_HALF_HOUR_PATTERN = re.compile(r"\b(?:a\s+)?half\s+(?:an?\s+)?hour(?:\s+long)?\b", re.IGNORECASE)
_DURATION_PATTERN = re.compile(
    r"\b(?P<val>\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"twelve|fifteen|twenty|thirty|forty|forty-five|fifty|sixty|ninety|an?)"
    r"[\s-]*(?P<unit>hours?|hrs?|hr|minutes?|mins?|min|seconds?|secs?|sec)"
    r"(?:\s+long|\s+length)?\b",
    re.IGNORECASE,
)


def parse_duration_seconds(command: str, default_minutes: float = 15.0) -> tuple[float, str]:
    """Extract requested duration in seconds and return (seconds, cleaned_text_without_duration)."""
    text = command.strip()

    m_hm = _HR_MIN_PATTERN.search(text)
    if m_hm:
        secs = float(m_hm.group("hr")) * 3600.0 + float(m_hm.group("min")) * 60.0
        cleaned = (text[: m_hm.start()] + " " + text[m_hm.end() :]).strip()
        return max(45.0, secs), cleaned

    m_half = _HALF_HOUR_PATTERN.search(text)
    if m_half:
        cleaned = (text[: m_half.start()] + " " + text[m_half.end() :]).strip()
        return 1800.0, cleaned

    for m in _DURATION_PATTERN.finditer(text):
        raw_val = m.group("val").lower()
        unit = m.group("unit").lower()
        val = float(raw_val) if raw_val.replace(".", "", 1).isdigit() else _WORD_NUMBERS.get(raw_val, 1.0)
        if unit.startswith("h"):
            secs = val * 3600.0
        elif unit.startswith("m"):
            secs = val * 60.0
        else:
            secs = val
        cleaned = (text[: m.start()] + " " + text[m.end() :]).strip()
        return max(45.0, secs), cleaned

    return default_minutes * 60.0, text


def extract_topic(cleaned_text: str) -> str:
    """Extract clean documentary topic from natural-language command text."""
    t = re.sub(r"\s+", " ", cleaned_text).strip()

    # Prefer explicit "... documentary/film/podcast/episode/video on/about/exploring <TOPIC>" match first
    m_explicit = re.search(
        r"\b(?:documentary|film|feature|podcast|episode|show|video)\s+(?:film\s+|episode\s+|video\s+|series\s+)?(?:on|about|covering|exploring|investigating|regarding|of)\s+(.+)$",
        t,
        flags=re.IGNORECASE,
    )
    if m_explicit:
        t = m_explicit.group(1).strip()
    else:
        prefix_patterns = [
            r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?(?:build|create|make|generate|produce|render|write|craft)\s+(?:me\s+)?",
            r"^(?:a\s+|an\s+|the\s+)?(?:professional\s+|cinema(?:\s*-?\s*grade)?\s+|studio(?:\s*-?\s*level|\s*-?\s*grade)?\s+|high[\s-]quality\s+|deep[\s-]dive\s+|documentary\s+|full\s+|complete\s+|long\s+)*(?:video\s+)?(?:documentary|film|feature|podcast|episode|show|video)(?:\s+film|\s+episode|\s+video)?\s*",
            r"^(?:on|about|covering|exploring|investigating|regarding)\s+",
        ]
        for pat in prefix_patterns:
            new_t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
            if new_t:
                t = new_t

    t = re.sub(
        r"\s+(?:and\s+(?:audit|improve|deliver|verify).*|with\s+no\s+api.*|without\s+any\s+api.*|at\s+studio\s+level.*)$",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip(" .,!?:;\"'")

    if not t:
        t = "The Seven Wonders of the World"

    small_words = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "vs"}
    words = t.split()
    formatted: list[str] = []
    for i, w in enumerate(words):
        if w.isupper() and len(w) > 1:
            formatted.append(w)
        elif i > 0 and w.lower() in small_words:
            formatted.append(w.lower())
        else:
            formatted.append(w[:1].upper() + w[1:])
    return " ".join(formatted)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:56] or "podcast-episode"


def parse_command_to_spec(
    command: str,
    duration_override_minutes: float | None = None,
    wpm: float = 140.0,
) -> ShowSpec:
    """Convert a single natural-language user prompt into a complete ShowSpec."""
    parsed_sec, without_dur = parse_duration_seconds(command, default_minutes=15.0)
    if duration_override_minutes is not None and duration_override_minutes > 0:
        target_sec = max(45.0, float(duration_override_minutes) * 60.0)
    else:
        target_sec = parsed_sec

    topic = extract_topic(without_dur)
    slug = slugify(topic)

    minutes = target_sec / 60.0
    if minutes <= 2.2:
        chapter_count = 2
    elif minutes <= 4.5:
        chapter_count = 3
    elif minutes <= 10.0:
        chapter_count = 5
    elif minutes <= 25.0:
        chapter_count = 7
    elif minutes <= 75.0:
        chapter_count = 9
    else:
        chapter_count = min(14, max(10, int(round(minutes / 8.0))))

    card_dur = 2.5 if target_sec >= 120 else 2.2
    bookend_overhead_sec = 8.4 if target_sec < 120 else 9.2
    cards_overhead_sec = chapter_count * card_dur + bookend_overhead_sec
    net_speech_and_pauses_sec = max(22.0, target_sec - cards_overhead_sec)
    effective_spoken_sec = net_speech_and_pauses_sec * 0.87
    target_words = max(80, int(round((effective_spoken_sec / 60.0) * wpm)))

    if minutes >= 60 and abs(minutes % 60) < 2:
        dur_label = f"{int(round(minutes / 60))}h"
    elif minutes < 2.0:
        dur_label = f"{int(round(target_sec))}s"
    else:
        dur_label = f"{int(round(minutes))}m"

    episode_title = f"{topic}: Architecture of the Impossible"
    episode_subtitle = f"Deep Horizon Original Documentary · {max(1, round(minutes))}-Minute Cinema Edition"

    return ShowSpec(
        raw_command=command,
        topic=topic,
        show_title="DEEP HORIZON",
        episode_title=episode_title,
        episode_subtitle=episode_subtitle,
        target_duration_sec=target_sec,
        target_word_count=target_words,
        chapter_count=chapter_count,
        words_per_minute=wpm,
        style="cinema_documentary",
        slug=f"{slug}-{dur_label}",
    )
