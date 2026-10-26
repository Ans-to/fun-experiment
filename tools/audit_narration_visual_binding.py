#!/usr/bin/env python3
"""Static narration→visual binding audit for a rendered episode directory.

Checks `episode_script.json` and (when present) `scene_plan.json` for exact-clause
placeholders and spoken-subject / procedural-scene mismatches.  No video decoding is
required, so this can run in CI before an expensive render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from voxforge.audit.av_sync_qa import audit_scene_plan_semantics  # noqa: E402
from voxforge.video.procedural_world import (  # noqa: E402
    SCENE_CAPTIONS,
    resolve_monument_scene_id,
)


def _caption_scene_id(caption: str) -> int | None:
    low = caption.lower()
    for sid, raw in SCENE_CAPTIONS.items():
        tail = raw.split("·", 1)[1].strip() if "·" in raw else raw
        if tail.lower() in low or low in tail.lower():
            return sid
    return None


def _expected_scene_id(turn: dict, chapter: dict) -> int:
    hint = " || ".join(
        part.strip()
        for part in (
            turn.get("text", ""),
            f"{chapter.get('title', '')} {chapter.get('subtitle', '')}",
            chapter.get("location", ""),
        )
        if part and part.strip()
    )
    return resolve_monument_scene_id(hint)


def audit_script(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    for chapter in data.get("chapters", []):
        for turn in chapter.get("turns", []):
            tid = turn.get("turn_id", "?")
            expected = _expected_scene_id(turn, chapter)
            recorded = turn.get("visual_scene_id")
            if recorded is None:
                recorded = _caption_scene_id(str(turn.get("visual_caption", "")))
                source = "caption"
            else:
                source = "visual_scene_id"
            try:
                recorded_i = int(recorded) if recorded is not None else None
            except (TypeError, ValueError):
                recorded_i = None
            if recorded_i != expected:
                issues.append(
                    f"episode_script turn {tid}: expected scene {expected} "
                    f"({SCENE_CAPTIONS.get(expected, '?')}), {source} records {recorded_i}"
                )
            subject = str(turn.get("visual_subject", "") or "")
            if "classical sanctuary" in subject.lower():
                issues.append(f"episode_script turn {tid}: stale classical-sanctuary subject")
    return issues


def audit_scene_plan(path: Path) -> list[str]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    return [
        f"scene_plan[{idx}]: {reason}"
        for idx, reason in audit_scene_plan_semantics(plan)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episode_dir", type=Path, help="Directory containing episode_script.json / scene_plan.json")
    args = ap.parse_args()

    script_path = args.episode_dir / "episode_script.json"
    plan_path = args.episode_dir / "scene_plan.json"
    if not script_path.exists() and not plan_path.exists():
        print(f"ERROR: neither {script_path} nor {plan_path} exists", file=sys.stderr)
        return 2

    issues: list[str] = []
    if script_path.exists():
        issues.extend(audit_script(script_path))
    if plan_path.exists():
        issues.extend(audit_scene_plan(plan_path))

    if issues:
        print(f"FAIL: {len(issues)} narration→visual binding issue(s)")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("PASS: narration→visual bindings are subject-consistent and clause-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
