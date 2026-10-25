"""FastAPI Broadcast Control Room Server (Preview-Safe, 0.0.0.0, Relative URLs)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from voxforge.config import DEFAULT_OUTPUT_ROOT, WEB_STATIC_DIR, ensure_cache_dirs
from voxforge.pipeline import run_podcast_pipeline

app = FastAPI(title="VoxForge Broadcast Studio", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


class GenerateRequest(BaseModel):
    command: str
    duration_minutes: float | None = None
    tts_backend: str = "auto"


def _scan_episodes() -> list[dict[str, Any]]:
    ensure_cache_dirs()
    episodes: list[dict[str, Any]] = []
    if not DEFAULT_OUTPUT_ROOT.exists():
        return episodes

    for d in sorted(DEFAULT_OUTPUT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        mp4 = d / "documentary_film.mp4"
        if not mp4.exists():
            mp4 = d / "podcast_studio.mp4"
        script_f = d / "episode_script.json"
        audit_f = d / "audit_report.json"
        if not (mp4.exists() and script_f.exists() and audit_f.exists()):
            continue
        try:
            script_data = json.loads(script_f.read_text(encoding="utf-8"))
            audit_data = json.loads(audit_f.read_text(encoding="utf-8"))
            spec = script_data.get("spec", {})
            chapters = script_data.get("chapters", [])
            dur = chapters[-1]["end_sec"] if chapters else spec.get("target_duration_sec", 0.0)
            episodes.append(
                {
                    "slug": d.name,
                    "title": spec.get("episode_title", d.name),
                    "subtitle": spec.get("episode_subtitle", ""),
                    "topic": spec.get("topic", d.name),
                    "raw_command": spec.get("raw_command", ""),
                    "duration_sec": round(float(dur), 1),
                    "chapter_count": len(chapters),
                    "overall_initial_score": audit_data.get("overall_initial_score", 0.0),
                    "overall_final_score": audit_data.get("overall_final_score", 0.0),
                    "studio_certified": audit_data.get("studio_certified", True),
                    "video_size_mb": round(mp4.stat().st_size / (1024 * 1024), 2),
                }
            )
        except Exception:
            continue
    return episodes


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = WEB_STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/episodes")
def list_episodes() -> dict[str, Any]:
    return {"episodes": _scan_episodes()}


@app.get("/api/episodes/{slug}")
def get_episode_detail(slug: str) -> dict[str, Any]:
    ep_dir = (DEFAULT_OUTPUT_ROOT / slug).resolve()
    if not ep_dir.exists() or ep_dir.parent != DEFAULT_OUTPUT_ROOT.resolve():
        raise HTTPException(status_code=404, detail="Episode not found")

    script_data = json.loads((ep_dir / "episode_script.json").read_text(encoding="utf-8"))
    audit_data = json.loads((ep_dir / "audit_report.json").read_text(encoding="utf-8"))
    dossier_data = (
        json.loads((ep_dir / "research_dossier.json").read_text(encoding="utf-8"))
        if (ep_dir / "research_dossier.json").exists()
        else {}
    )
    return {
        "slug": slug,
        "script": script_data,
        "audit": audit_data,
        "dossier": dossier_data,
    }


@app.api_route("/api/episodes/{slug}/media/{filename}", methods=["GET", "HEAD"])
def serve_episode_media(slug: str, filename: str, download: int = 0) -> FileResponse:
    allowed = {
        "documentary_film.mp4": "video/mp4",
        "podcast_studio.mp4": "video/mp4",
        "documentary_audio.mp3": "audio/mpeg",
        "podcast_audio.mp3": "audio/mpeg",
        "captions.srt": "text/plain; charset=utf-8",
        "captions.vtt": "text/vtt; charset=utf-8",
        "audit_report.md": "text/markdown; charset=utf-8",
    }
    if filename not in allowed or "/" in slug or "\\" in slug or ".." in slug:
        raise HTTPException(status_code=400, detail="Unsupported artifact")

    ep_dir = (DEFAULT_OUTPUT_ROOT / slug).resolve()
    if ep_dir.parent != DEFAULT_OUTPUT_ROOT.resolve():
        raise HTTPException(status_code=404, detail="Episode not found")

    fpath = (ep_dir / filename).resolve()
    if not fpath.exists():
        alt_map = {
            "documentary_film.mp4": "podcast_studio.mp4",
            "podcast_studio.mp4": "documentary_film.mp4",
            "documentary_audio.mp3": "podcast_audio.mp3",
            "podcast_audio.mp3": "documentary_audio.mp3",
        }
        if filename in alt_map and (ep_dir / alt_map[filename]).exists():
            fpath = (ep_dir / alt_map[filename]).resolve()
        else:
            raise HTTPException(status_code=404, detail="File not found")

    if download:
        return FileResponse(
            str(fpath),
            media_type=allowed[filename],
            filename=f"{slug}-{filename}",
            content_disposition_type="attachment",
        )
    return FileResponse(str(fpath), media_type=allowed[filename])


@app.api_route("/api/episodes/{slug}/download/{filename}", methods=["GET", "HEAD"])
def download_episode_media(slug: str, filename: str) -> FileResponse:
    return serve_episode_media(slug=slug, filename=filename, download=1)


@app.post("/api/jobs")
def start_generation_job(req: GenerateRequest) -> dict[str, Any]:
    if not req.command.strip():
        raise HTTPException(status_code=400, detail="Command cannot be empty")

    job_id = uuid.uuid4().hex[:10]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "command": req.command,
            "status": "running",
            "stage": "QUEUED",
            "logs": ["Queued autonomous studio production pipeline..."],
            "episode_slug": None,
            "error": None,
            "started_at": time.time(),
        }

    def _worker() -> None:
        def _cb(stage: str, msg: str) -> None:
            with _JOBS_LOCK:
                _JOBS[job_id]["stage"] = stage
                _JOBS[job_id]["logs"].append(f"[{stage}] {msg}")

        try:
            res = run_podcast_pipeline(
                command=req.command,
                duration_override_minutes=req.duration_minutes,
                preferred_tts=req.tts_backend,
                on_status=_cb,
            )
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "completed"
                _JOBS[job_id]["stage"] = "COMPLETE"
                _JOBS[job_id]["episode_slug"] = res.output_dir.name
        except Exception as exc:
            with _JOBS_LOCK:
                _JOBS[job_id]["status"] = "failed"
                _JOBS[job_id]["error"] = str(exc)
                _JOBS[job_id]["logs"].append(f"[ERROR] {exc}")

    threading.Thread(target=_worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    with _JOBS_LOCK:
        return {"jobs": list(_JOBS.values())}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        if job_id not in _JOBS:
            raise HTTPException(status_code=404, detail="Job not found")
        return _JOBS[job_id]
