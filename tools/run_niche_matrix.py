"""VoxForge A/V-sync certification matrix — 10 niches × 90-second films.

Runs the full single-command pipeline for each niche, then audits the FINAL rendered
video with the Stage E narration↔visual semantic gate (scene_plan.json + decoded
frames), and saves frame-inspection evidence sheets + per-scene validation verdicts
into `qa_evidence/` for the deliverables report.

Usage:  python tools/run_niche_matrix.py [output_root]
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from voxforge.pipeline import run_podcast_pipeline  # noqa: E402

NICHES: list[tuple[str, str]] = [
    ("01_science_education", "how volcanoes erupt and reshape the planet"),
    ("02_tech_ai", "the rise of artificial intelligence"),
    ("03_history", "the silk road trade networks of antiquity"),
    ("04_business_finance", "the history of money and banking"),
    ("05_health", "how the human heart works"),
    ("06_travel_geography", "the nile river and ancient egypt"),
    ("07_cooking", "the science of baking bread"),
    ("08_wildlife_nature", "the great whale migrations"),
    ("09_documentary_storytelling", "the lost libraries of the ancient world"),
    ("10_motivational", "stoic philosophy for modern life"),
]

EVIDENCE_DIR = REPO / "qa_evidence"


def _frame_at(container, vstream, tb: float, t: float):
    container.seek(int(t / tb), stream=vstream)
    for f in container.decode(vstream):
        if f.pts is not None and f.pts * tb >= t - 0.02:
            return f.to_ndarray(format="bgr24")
    return None


def _make_sheet(mp4: Path, plan: dict, out_path: Path, n: int = 4) -> None:
    import av
    import cv2
    import numpy as np

    c = av.open(str(mp4))
    vs = c.streams.video[0]
    tb = float(vs.time_base)
    entries = [e for e in plan.get("scenes", []) if e["scene_end_sec"] - e["scene_start_sec"] >= 1.5]
    picks = entries[:: max(1, len(entries) // n)][:n]
    tiles = []
    for e in picks:
        fr = _frame_at(c, vs, tb, 0.5 * (e["scene_start_sec"] + e["scene_end_sec"]))
        if fr is None:
            continue
        small = cv2.resize(fr, (640, 360))
        clause = (e.get("narration_clause") or "")[:110]
        tag = f"[{e.get('visual_family')} @ {e.get('scene_start_sec')}s -> {e.get('validation_status')}]"
        cv2.rectangle(small, (0, 306), (640, 360), (0, 0, 0), -1)
        cv2.putText(small, clause[:66], (8, 324), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(small, clause[66:] or tag, (8, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(small, tag, (8, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 220, 255), 1, cv2.LINE_AA)
        tiles.append(small)
    c.close()
    while len(tiles) < n:
        tiles.append(np.zeros((360, 640, 3), dtype=np.uint8))
    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, n, 2)]
    sheet = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])


def main() -> int:
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/user/qa_niches")
    out_root.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for tag, topic in NICHES:
        t0 = time.time()
        rec: dict = {"niche": tag, "topic": topic}
        print(f"\n=== [{tag}] {topic} ===", flush=True)
        try:
            r = run_podcast_pipeline(
                command=f"build a 90 second documentary on {topic}",
                output_root=out_root,
                preferred_tts="auto",
            )
            rep = r.audit_report
            plan_path = r.output_dir / "scene_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
            stage_e = plan.get("stage_e", {})
            scene_statuses = [str(e.get("validation_status", "")) for e in plan.get("scenes", [])]
            rec.update(
                certified=bool(rep.studio_certified),
                overall_final=rep.overall_final_score,
                stage_scores={s.stage_name: s.final_score for s in rep.stages},
                stage_e_score=stage_e.get("score"),
                stage_e_passed=stage_e.get("passed"),
                semantic_mismatches=stage_e.get("mismatches"),
                av_drift_sec=stage_e.get("av_drift_sec"),
                scenes_planned=len(scene_statuses),
                scenes_validated=sum(1 for s in scene_statuses if s == "validated"),
                output_dir=str(r.output_dir),
                secs=round(time.time() - t0),
            )
            _make_sheet(r.video_mp4_path, plan, EVIDENCE_DIR / f"{tag}_sheet.jpg")
            shutil.copy(plan_path, EVIDENCE_DIR / f"{tag}_scene_plan.json")
            shutil.copy(r.audit_json_path, EVIDENCE_DIR / f"{tag}_audit_report.json")
        except Exception:  # noqa: BLE001
            rec.update(certified=False, error=traceback.format_exc()[-800:], secs=round(time.time() - t0))
        results.append(rec)
        print("RESULT " + json.dumps(rec), flush=True)
        (EVIDENCE_DIR / "niche_matrix_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== MATRIX COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
