"""VoxForge Stage A editorial certification matrix — 10 niches, offline-runnable.

The film-level matrix (`tools/run_niche_matrix.py`) needs network access for neural TTS
models, so it cannot run in an air-gapped sandbox. Stage A (script writing + editorial
audit + rewrite loop) is completely dependency-free apart from numpy/cv2, which makes it
the part of the certification chain that *must* be reproducible anywhere.

This tool runs the real single-command pipeline's editorial half for every niche:

    parse command -> research dossier -> showrunner script -> Stage A audit/heal loop

and records the initial vs. healed verdicts for all six editorial checks (E1 word budget,
E2 6-gram lexical freshness, E3 cliché purity, E4 host balance/cadence, E5 multicam
cutaway rhythm, E6 spoken-ear orality) into `qa_evidence/`.

Usage:
    python tools/run_stage_a_matrix.py                 # all 10 niches
    python tools/run_stage_a_matrix.py --only 03_history,07_cooking
    python tools/run_stage_a_matrix.py --json /tmp/stage_a.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools.run_niche_matrix import NICHES  # noqa: E402

from voxforge.audit.editorial import _evaluate_script, audit_and_improve_script  # noqa: E402
from voxforge.bootstrap import StudioEnvironment  # noqa: E402
from voxforge.intent import parse_command_to_spec  # noqa: E402
from voxforge.research.engine import build_research_dossier  # noqa: E402
from voxforge.script.showrunner import generate_episode_script  # noqa: E402

EVIDENCE_DIR = REPO / "qa_evidence"

# Stage A passes when the healed script clears the studio threshold and every check passes.
PASS_SCORE = 88.0


def _offline_env() -> StudioEnvironment:
    """StudioEnvironment with the always-available corpus showrunner (no model downloads)."""
    return StudioEnvironment(
        ffmpeg_bin="ffmpeg",
        tts_backend="piper",
        kokoro_model_path=None,
        kokoro_voices_path=None,
        piper_male_onnx=None,
        piper_female_onnx=None,
        llm_backend="corpus_showrunner",
        llm_model_id_or_path="voxforge-documentary-discourse-v1",
    )


def _check_row(c) -> dict:
    return {
        "check_id": c.check_id,
        "name": c.name,
        "passed": bool(c.passed),
        "score": c.score,
        "measured": c.measured_value,
    }


def run_niche(tag: str, topic: str) -> dict:
    """Run the editorial half of the pipeline for one niche and return its verdict record."""
    t0 = time.time()
    rec: dict = {"niche": tag, "topic": topic, "command": f"build a 90 second documentary on {topic}"}
    try:
        spec = parse_command_to_spec(rec["command"])
        with tempfile.TemporaryDirectory(prefix="voxforge_stagea_") as td:
            dossier = build_research_dossier(spec, Path(td))
            draft = generate_episode_script(spec, dossier, _offline_env())
            pre = _evaluate_script(draft)
            pre_score = round(sum(c.score for c in pre) / len(pre), 1)
            script, audit = audit_and_improve_script(draft, dossier)

        failed = [c.check_id for c in audit.checks if not c.passed]
        rec.update(
            ok=True,
            stage_a_passed=bool(audit.passed and audit.final_score >= PASS_SCORE and not failed),
            initial_score=pre_score,
            final_score=audit.final_score,
            iterations=audit.iterations,
            total_words=script.total_words,
            target_words=spec.target_word_count,
            chapters=len(script.chapters),
            turns=len(script.all_turns),
            failed_checks=failed,
            pre_checks=[_check_row(c) for c in pre],
            post_checks=[_check_row(c) for c in audit.checks],
            improvements=list(audit.improvements_applied),
        )
    except Exception:  # noqa: BLE001 — a crashing niche must not abort the whole matrix
        rec.update(ok=False, stage_a_passed=False, error=traceback.format_exc()[-900:])
    rec["secs"] = round(time.time() - t0, 2)
    return rec


def write_markdown(records: list[dict], path: Path) -> None:
    total = len(records)
    passed = sum(1 for r in records if r.get("stage_a_passed"))
    lines = [
        "# VoxForge Stage A Editorial Certification Matrix",
        "",
        f"**{passed}/{total} niches certified** (Stage A threshold: score >= {PASS_SCORE} and all six checks pass).",
        "",
        "Generated offline via `tools/run_stage_a_matrix.py` — the editorial half of the pipeline",
        "(intent parse → research dossier → showrunner script → audit/heal loop).",
        "",
        "| Niche | Topic | Initial | Final | E1 | E2 | E3 | E4 | E5 | E6 | Certified |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if not r.get("ok"):
            lines.append(
                f"| `{r['niche']}` | {r['topic']} | — | — | — | — | — | — | — | — | **ERROR** |"
            )
            continue
        post = {c["check_id"]: c for c in r["post_checks"]}
        marks = [("PASS" if post[k]["passed"] else "**FAIL**") for k in ("E1", "E2", "E3", "E4", "E5", "E6")]
        lines.append(
            f"| `{r['niche']}` | {r['topic']} | {r['initial_score']} | {r['final_score']} | "
            + " | ".join(marks)
            + f" | {'YES' if r['stage_a_passed'] else 'NO'} |"
        )
    lines += [
        "",
        "## Lexical freshness detail (E2 · 6-gram overlap, spec <= 2.50%)",
        "",
        "| Niche | Initial | Final | Measured (post) | Words / Target |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        if not r.get("ok"):
            continue
        pre_e2 = next(c for c in r["pre_checks"] if c["check_id"] == "E2")
        post_e2 = next(c for c in r["post_checks"] if c["check_id"] == "E2")
        lines.append(
            f"| `{r['niche']}` | {pre_e2['score']} | {post_e2['score']} | "
            f"{post_e2['measured']} | {r['total_words']} / {r['target_words']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage A 10-niche editorial certification matrix")
    ap.add_argument("--only", default=None, help="comma-separated niche tags to run/resume")
    ap.add_argument("--json", default=None, help="results path (default: qa_evidence/stage_a_matrix.json)")
    ap.add_argument("--md", default=None, help="markdown path (default: qa_evidence/stage_a_matrix.md)")
    args = ap.parse_args()

    json_path = Path(args.json) if args.json else EVIDENCE_DIR / "stage_a_matrix.json"
    md_path = Path(args.md) if args.md else EVIDENCE_DIR / "stage_a_matrix.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    only = {t.strip() for t in args.only.split(",") if t.strip()} if args.only else None

    # Merge across runs so a resumed/partial run never loses earlier verdicts.
    by_tag: dict[str, dict] = {}
    if json_path.exists():
        try:
            for prev in json.loads(json_path.read_text(encoding="utf-8")):
                by_tag[prev["niche"]] = prev
        except (json.JSONDecodeError, KeyError):
            by_tag = {}

    for tag, topic in NICHES:
        if only and tag not in only:
            continue
        print(f"\n=== [{tag}] {topic} ===", flush=True)
        rec = run_niche(tag, topic)
        if rec.get("ok"):
            print(
                f"  Stage A {rec['initial_score']} -> {rec['final_score']} | "
                f"{'CERTIFIED' if rec['stage_a_passed'] else 'FAILED ' + ','.join(rec['failed_checks'])} "
                f"| {rec['total_words']}/{rec['target_words']} words | {rec['secs']}s",
                flush=True,
            )
        else:
            print(f"  ERROR: {rec['error'].splitlines()[-1]}", flush=True)
        by_tag[tag] = rec
        merged = [by_tag[t] for t, _ in NICHES if t in by_tag]
        json_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    merged = [by_tag[t] for t, _ in NICHES if t in by_tag]
    write_markdown(merged, md_path)
    ok = sum(1 for r in merged if r.get("stage_a_passed"))
    print(f"\n=== STAGE A MATRIX: {ok}/{len(merged)} certified ===", flush=True)
    print(f"Evidence: {json_path} | {md_path}", flush=True)
    return 0 if ok == len(merged) else 1


if __name__ == "__main__":
    raise SystemExit(main())
