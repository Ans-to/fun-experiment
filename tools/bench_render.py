"""Benchmark monument 3D frame render cost (default engine configuration)."""
import os
import time

from voxforge.video.procedural_world import (
    Monument3DEngine,
    SCENE_GIZA,
    SCENE_TAJ_MAHAL,
    SCENE_COLOSSEUM,
)


def main() -> None:
    eng = Monument3DEngine()
    # warmup (numba compile)
    _ = eng.render_frame(SCENE_GIZA, 0.5, 1.0, 1920, 1080)
    scenes = [("Giza", SCENE_GIZA), ("Taj", SCENE_TAJ_MAHAL), ("Colosseum", SCENE_COLOSSEUM)]
    n = 12
    for name, sid in scenes:
        t0 = time.perf_counter()
        for i in range(n):
            _ = eng.render_frame(sid, 0.3 + 0.4 * i / n, i / 24.0, 1920, 1080)
        dt = (time.perf_counter() - t0) / n * 1000
        print(f"{name:10s} scale={eng._internal_scale:4.2f}: {dt:7.1f} ms/frame")
    nproc = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"nproc={nproc}")


if __name__ == "__main__":
    main()
