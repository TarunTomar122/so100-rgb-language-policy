"""Record actual policy runs and their first/last frames. Requires ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from so100.action_demo import ActionDemo
from so100.train_rgb_target import SIZE

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("go-near-blue", 313000, "go near the blue cube", ("reach",)),
    ("lift-red", 307000, "please grab the red block and hold it up", ("reach", "lower", "close", "lift")),
    ("move-green-right", 307002, "lift the green tube and put it down on the right",
     ("reach", "lower", "close", "lift", "right", "open")),
)


def record(app: ActionDemo, slug: str, seed: int, instruction: str, expected: tuple[str, ...]) -> None:
    app.seed = seed - 1
    app.reset()
    app.command(instruction)
    before = app.world.render(SIZE)
    output = ROOT / "media" / f"{slug}.mp4"
    process = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pixel_format", "rgb24", "-video_size", f"{SIZE}x{SIZE}", "-framerate", "12",
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    ticks = 0

    def frame(force: bool = False) -> None:
        nonlocal ticks
        ticks += 1
        if force or ticks % 2 == 0:
            process.stdin.write(np.ascontiguousarray(app.world.render(SIZE)).tobytes())

    execute = app.executor.execute
    goto = app.executor.goto
    app.executor.execute = lambda action, on_frame=None: execute(action, on_frame=frame)
    app.executor.goto = lambda goal, on_frame=None: goto(goal, on_frame=frame)
    peak = 0.0
    try:
        for _ in range(12):
            frame(force=True)
        for _ in range(13):
            if app._predict() is None:
                break
            app.step()
            peak = max(peak, app._target_rise() or 0.0)
            frame(force=True)
        for _ in range(12):
            frame(force=True)
    finally:
        app.executor.execute = execute
        app.executor.goto = goto
        process.stdin.close()
        assert process.wait() == 0, f"ffmpeg failed for {slug}"

    assert app.finished and not app.failed and tuple(app.history) == expected, (instruction, app.history, app.status)
    if "lift" in expected:
        assert peak >= 30, (instruction, peak)
    else:
        assert peak < 5 and app.world.jaw() > 1.0, (instruction, peak)
    after = app.world.render(SIZE)
    Image.fromarray(np.concatenate([before, after], axis=1)).save(ROOT / "media" / f"{slug}.jpg", quality=88)
    print(f"{output.name}: {app.history}, peak rise {peak:.0f} mm, {output.stat().st_size // 1024} KiB", flush=True)


if __name__ == "__main__":
    (ROOT / "media").mkdir(exist_ok=True)
    demo = ActionDemo()
    for case in CASES:
        record(demo, *case)
