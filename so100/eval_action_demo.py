"""Fresh-scene, end-to-end check of the browser policy and visible outcomes.

Run: source scripts/vulkan_env.sh && PYTHONPATH=. .venv/bin/python -m so100.eval_action_demo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from so100.action_demo import ActionDemo
from so100.sim import MOVABLES
from so100.train_rgb_target import SIZE, scene


# Fresh wording and seeds; these are never fed to training.
CASES = (
    ("please pick up the red cube", "red_cube", ("reach", "lower", "close", "lift")),
    ("could you grab the blue cube for me", "blue_cube", ("reach", "lower", "close", "lift")),
    ("pick up the green cylinder and move left", "green_cylinder", ("reach", "lower", "close", "lift", "left")),
    ("lift the yellow block, move right, then let go", "yellow_block", ("reach", "lower", "close", "lift", "right", "open")),
    ("grab the red cube and set it down to the left", "red_cube", ("reach", "lower", "close", "lift", "left", "open")),
    ("take the blue cube off the table and put it to the right", "blue_cube", ("reach", "lower", "close", "lift", "right", "open")),
    ("move the arm left", None, ("left",)),
    ("open the gripper", None, ("open",)),
    ("close the jaws", None, ("close",)),
    ("move right and then go left", None, ("right", "left")),
    ("please move the arm up", None, ("up",)),
)

BLIND_CASES = (
    ("could you raise the red cube and keep holding it", "red_cube", ("reach", "lower", "close", "lift")),
    ("pick that blue block up for me", "blue_cube", ("reach", "lower", "close", "lift")),
    ("bring the green cylinder up off the table", "green_cylinder", ("reach", "lower", "close", "lift")),
    ("please take the yellow block and carry it rightward", "yellow_block", ("reach", "lower", "close", "lift", "right")),
    ("place the red box down to your left", "red_cube", ("reach", "lower", "close", "lift", "left", "open")),
    ("fetch the blue cube and leave it on the right", "blue_cube", ("reach", "lower", "close", "lift", "right", "open")),
    ("move the green cylinder left and release your grip", "green_cylinder", ("reach", "lower", "close", "lift", "left", "open")),
    ("take the yellow block, shift right, then open your fingers", "yellow_block", ("reach", "lower", "close", "lift", "right", "open")),
    ("lift the red cube and release your hold", "red_cube", ("reach", "lower", "close", "lift", "open")),
    ("pick up the blue cube and shift it right", "blue_cube", ("reach", "lower", "close", "lift", "right")),
    ("move the arm left and then raise it", None, ("left", "up")),
    ("raise the gripper before moving left", None, ("up", "left")),
    ("lower the end effector", None, ("down",)),
    ("open your fingers", None, ("open",)),
    ("clamp the jaws", None, ("close",)),
)

FINAL_CASES = (
    ("please grab the red block and hold it up", "red_cube", ("reach", "lower", "close", "lift")),
    ("raise the blue box and move it to the left", "blue_cube", ("reach", "lower", "close", "lift", "left")),
    ("lift the green tube and put it down on the right", "green_cylinder", ("reach", "lower", "close", "lift", "right", "open")),
    ("pick the yellow rectangular block up and drop it", "yellow_block", ("reach", "lower", "close", "lift", "open")),
    ("bring the red cube off the table, go right, and let go", "red_cube", ("reach", "lower", "close", "lift", "right", "open")),
    ("could you take the blue cube and shift it leftward", "blue_cube", ("reach", "lower", "close", "lift", "left")),
    ("pick up the green cylinder and keep holding it", "green_cylinder", ("reach", "lower", "close", "lift")),
    ("move the yellow block to the left, then release it", "yellow_block", ("reach", "lower", "close", "lift", "left", "open")),
    ("take the red cube off the table and leave it on the right", "red_cube", ("reach", "lower", "close", "lift", "right", "open")),
    ("fetch the blue block and put it down to your left", "blue_cube", ("reach", "lower", "close", "lift", "left", "open")),
    ("shift the arm right, then raise the gripper", None, ("right", "up")),
    ("go left followed by right", None, ("left", "right")),
    ("lower the gripper a little", None, ("down",)),
    ("raise the arm and then lower it", None, ("up", "down")),
    ("open your fingers", None, ("open",)),
    ("squeeze the gripper closed", None, ("close",)),
)

NEAR_CASES = (
    ("go near the blue cube", "blue_cube", ("reach",)),
    ("move close to the red block", "red_cube", ("reach",)),
    ("bring the gripper near the green cylinder", "green_cylinder", ("reach",)),
    ("hover over the yellow rectangular block", "yellow_block", ("reach",)),
    ("approach the red cube but leave it alone", "red_cube", ("reach",)),
    ("get close to the blue box", "blue_cube", ("reach",)),
    ("position the arm by the green tube", "green_cylinder", ("reach",)),
    ("reach toward the yellow box and stop", "yellow_block", ("reach",)),
)


def main() -> None:
    blind = "--blind" in sys.argv
    sweep = "--sweep" in sys.argv
    wide_hold = "--wide-hold" in sys.argv
    wide = "--wide" in sys.argv or wide_hold
    final = "--final" in sys.argv
    near = "--near" in sys.argv
    cases = ([(f"lift the {name.replace('_', ' ')}", name,
               ("reach", "lower", "close", "lift")) for _ in range(20) for name in MOVABLES]
             if sweep or wide else NEAR_CASES if near else FINAL_CASES if final else BLIND_CASES if blind else CASES)
    args = [arg for arg in sys.argv[1:] if arg not in ("--blind", "--sweep", "--wide", "--wide-hold", "--final", "--near")]
    limit = int(args[0]) if args else len(cases)
    app = ActionDemo()
    out = Path(__file__).resolve().parents[1] / "data" / "action-head" / (
        "wide-hold-eval" if wide_hold else "wide-eval" if wide else "sweep-eval" if sweep else "near-eval" if near else "final-eval" if final
        else "blind-eval" if blind else "fresh-eval")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (text, target, expected) in enumerate(cases[:limit]):
        app.seed = ((309000 if wide_hold else 308000) + i // 4 if wide else 300000 + i // 4 if sweep else 313000 + i if near else 307000 + i if final
                    else 298000 + i if blind else 291000 + i) - 1
        app.reset()
        if wide:
            scene(app.world, app.seed, annotate=False, wide=True)
        before = None if target is None else app.world.object_xyz(target).copy()
        first = app.world.render(SIZE)
        app.command(text)
        peak = 0.0
        predictions = []
        for _ in range(13):
            choice = app._predict()
            if choice is None:
                break
            predictions.append((choice[0], round(choice[1], 3)))
            app.step()
            peak = max(peak, app._target_rise() or 0.0)
        last = app.world.render(SIZE)
        if target is not None:
            movement = (app.world.object_xyz(target) - before) * 1000
            selected = app.target_name == target
            physical = (peak < 5 and app.world.jaw() > 1.0) if expected == ("reach",) else peak >= 30
            if "left" in expected:
                physical &= movement[1] <= -15
            if "right" in expected:
                physical &= movement[1] >= 15
        else:
            movement = None
            selected = True
            physical = True
        sequence = tuple(app.history) == expected and app.finished
        passed = bool(sequence and selected and physical)
        Image.fromarray(np.concatenate([first, last], axis=1)).save(out / f"{i:02d}.jpg", quality=88)
        row = {"seed": app.seed, "command": text, "expected": expected, "actual": app.history,
               "predictions": predictions,
               "selected": app.target_name, "peak_rise_mm": round(peak, 1),
               "target_movement_mm": None if movement is None else np.round(movement, 1).tolist(),
               "grasp_retries": app.grasp_retries, "status": app.status, "pass": passed}
        print(json.dumps(row), flush=True)
        results.append(row)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"PASS {sum(r['pass'] for r in results)}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
