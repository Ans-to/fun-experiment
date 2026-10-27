"""Single-Command CLI for VoxForge Documentary Film Studio.

Supports natural-language single-command execution:
    ./documentary "build a 1 hour documentary on 7 wonders of the world"
    ./podcast "build a 5 minute documentary on 7 wonders of the world"
and interactive Documentary Screening & Control Room server mode:
    ./documentary studio --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from voxforge.pipeline import run_podcast_pipeline

console = Console()


def _run_studio_server(host: str, port: int) -> None:
    import uvicorn

    console.print(
        Panel.fit(
            f"[bold gold1]VoxForge Broadcast Control Room[/bold gold1]\n"
            f"Listening on [cyan]http://{host}:{port}[/cyan] (Zero API Keys · Zero Subscriptions)",
            border_style="gold1",
        )
    )
    uvicorn.run("voxforge.web.server:app", host=host, port=port, log_level="info")


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)

    if args_list and args_list[0].lower() in ("studio", "serve"):
        parser_s = argparse.ArgumentParser(prog="podcast studio", description="Launch VoxForge Web Control Room")
        parser_s.add_argument("studio", help="Launch web studio")
        parser_s.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
        parser_s.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
        ns = parser_s.parse_args(args_list)
        _run_studio_server(ns.host, ns.port)
        return 0

    parser = argparse.ArgumentParser(
        prog="podcast",
        description=(
            "VoxForge Studio — Single-Command Zero-API-Key Broadcast Video Podcast Producer.\n"
            'Example: ./podcast "build a 1 hour long podcast on 7 wonders of the world"'
        ),
    )
    parser.add_argument(
        "command",
        nargs="+",
        help='Natural-language prompt, e.g. "build a 1 hour long podcast on 7 wonders of the world"',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom root directory for output episode artifacts",
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=None,
        help="Optional explicit duration override in minutes (otherwise parsed from prompt)",
    )
    parser.add_argument(
        "--tts",
        choices=["auto", "kokoro", "piper", "edge"],
        default="auto",
        help="Zero-key neural voice backend preference (default: auto)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Retain intermediate segment render workspace for inspection",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Open Broadcast Control Room web UI immediately after rendering completes",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port when --serve is enabled (default: 8000)",
    )

    ns = parser.parse_args(args_list)
    full_command = " ".join(ns.command).strip()

    console.print(
        Panel.fit(
            f"[bold gold1]VOXFORGE · AUTONOMOUS DOCUMENTARY FILM ENGINE[/bold gold1]\n"
            f"[white]Command:[/white] [italic cyan]\"{full_command}\"[/italic cyan]\n"
            f"[dim]Zero API Keys · Deep Research · 24fps 2.5D & Cartography · -16 LUFS / BS.1770-4 Mastering · Closed-Loop QA[/dim]",
            border_style="gold1",
        )
    )

    def _status_printer(stage: str, message: str) -> None:
        console.print(f"[bold gold1][{stage:<9}][/bold gold1] {message}")

    result = run_podcast_pipeline(
        command=full_command,
        output_root=ns.output_dir,
        duration_override_minutes=ns.duration_min,
        preferred_tts=ns.tts,
        keep_work_files=ns.keep_temp,
        on_status=_status_printer,
    )

    # Render Audit & Self-Improvement Summary Table
    table = Table(title="Create → Audit → Improve Broadcast QA Certification", border_style="cyan")
    table.add_column("Domain Stage", style="bold white")
    table.add_column("Initial", justify="right", style="yellow")
    table.add_column("Improved", justify="right", style="bold green")
    table.add_column("Passes", justify="center")
    table.add_column("Key Improvements Applied", style="dim white")

    for st in result.audit_report.stages:
        table.add_row(
            st.stage_name,
            f"{st.initial_score:.1f}",
            f"{st.final_score:.1f}/100",
            str(st.iterations),
            st.improvements_applied[0] if st.improvements_applied else "Verified",
        )

    console.print()
    console.print(table)

    console.print(
        Panel(
            f"[bold green]STUDIO CERTIFIED DELIVERABLE READY[/bold green] "
            f"(Composite Score: [bold]{result.audit_report.overall_initial_score:.1f}[/bold] -> "
            f"[bold green]{result.audit_report.overall_final_score:.1f}/100[/bold green])\n\n"
            f"  [bold]Documentary Film (1080p24 MP4):[/bold] {result.video_mp4_path}\n"
            f"  [bold]Mastered Narration & Score (MP3):[/bold] {result.audio_mp3_path}\n"
            f"  [bold]Subtitles (SRT / WebVTT):[/bold]     {result.subtitles_srt_path} | {result.subtitles_vtt_path}\n"
            f"  [bold]Studio QC Audit Report:[/bold]       {result.audit_md_path}\n"
            f"  [bold]Master Runtime & Scale:[/bold]       {result.duration_sec / 60:.2f} min ({result.duration_sec:.1f}s) · "
            f"{result.total_words} words · {result.chapter_count} Acts",
            border_style="green",
        )
    )

    if ns.serve:
        _run_studio_server("0.0.0.0", ns.port)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
