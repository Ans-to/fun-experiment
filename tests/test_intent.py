"""Unit tests for natural-language single-command intent parsing."""

from voxforge.intent import parse_command_to_spec


def test_parse_one_hour_seven_wonders() -> None:
    spec = parse_command_to_spec("build a 1 hour long podcast on 7 wonders of the world")
    assert spec.target_duration_sec == 3600.0
    assert "7 Wonders of the World" in spec.topic
    assert spec.chapter_count == 9
    assert 6500 <= spec.target_word_count <= 8500
    assert spec.slug.endswith("-1h")


def test_parse_word_numbers_and_half_hour() -> None:
    s1 = parse_command_to_spec("create a half an hour documentary podcast about the Silk Road")
    assert s1.target_duration_sec == 1800.0
    assert s1.topic == "The Silk Road"

    s2 = parse_command_to_spec("make a forty-five minute video podcast exploring Black Holes")
    assert s2.target_duration_sec == 2700.0
    assert s2.topic == "Black Holes"


def test_parse_short_briefing_and_override() -> None:
    s3 = parse_command_to_spec("build a 3 minute podcast on 7 wonders of the world")
    assert s3.target_duration_sec == 180.0
    assert "7 Wonders" in s3.topic
    assert s3.chapter_count == 3

    s4 = parse_command_to_spec("Ancient Rome", duration_override_minutes=20.0)
    assert s4.target_duration_sec == 1200.0
    assert s4.topic == "Ancient Rome"
