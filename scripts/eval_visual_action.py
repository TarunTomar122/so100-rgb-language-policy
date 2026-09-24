"""Held-out physical rollouts for the visual action head and old baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from so100.action_demo import ActionDemo
from so100.sim import TABLE_TOP
from so100.train_rgb_target import SIZE
from so100.visual_action_demo import VisualActionDemo

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (280001, "go near the blue cube", "blue_cube", "near", False),
    (280002, "lift the red cube", "red_cube", "lift", False),
    (280003, "lift the green cylinder and move left", "green_cylinder", "left", False),
    (280004, "lift the yellow block and move right then drop it", "yellow_block", "drop", False),
    (280005, "shift the arm left", None, "arm-left", False),
    (280006, "raise the gripper a little", None, "arm-up", False),
    (280007, "close the gripper", None, "grip-close", False),
    (280008, "pick up the red cube, move left, then let go", "red_cube", "drop-left", False),
    (280009, "lift the blue cube and move right then drop it", "blue_cube", "drop", False),
    (280101, "lift the blue cube", "blue_cube", "lift", True),
    (280102, "lift the red cube", "red_cube", "lift", True),
)


def run(app, case: tuple, folder: Path, slip: bool = False) -> dict:
    seed, text, target, task, moved = case
    app.seed = seed - 1
    app.reset()
    world = app.world
    before = None if target is None else world.object_xyz(target)
    before_tip = world.tcp()
    first = world.render(SIZE)
    app.command(text)
    peak = 0.0
    perturbed = False
    for _ in range(13):
        if app._predict() is None:
            break
        app.step()
        if target is not None:
            peak = max(peak, (world.object_xyz(target)[2] - before[2]) * 1000)
        if moved and not perturbed and app.history == ["reach"]:
            pose = world.object_pose(target)
            pose[0] = float(np.clip(pose[0] + 0.03, -0.085, 0.085))
            pose[2] = TABLE_TOP + 0.020
            world.set_object(target, pose[:3], pose[3:])
            mujoco.mj_forward(world.model, world.data)
            perturbed = True
        if slip and not perturbed and app.history == ["reach", "lower", "close"]:
            pose = world.object_pose(target)
            pose[0] = float(np.clip(pose[0] + 0.035, -0.085, 0.085))
            pose[2] = TABLE_TOP + 0.020
            world.set_object(target, pose[:3], pose[3:])
            mujoco.mj_forward(world.model, world.data)
            # Slip occurs after the model picked lift, while that action begins.
            app.last_frame = world.render(SIZE)
            perturbed = True
        if app.finished or app.failed:
            break
    final = world.render(SIZE)
    image_name = f"{seed}-{task}.jpg" if slip else f"{seed}.jpg"
    Image.fromarray(np.concatenate((first, final), axis=1)).save(folder / image_name, quality=88)
    delta = (world.object_xyz(target) - before) * 1000 if target is not None else (world.tcp() - before_tip) * 1000
    ok = bool(app.finished and (target is None or app.target_name == target))
    if task == "near":
        ok &= app.history == ["reach"] and world.jaw() > 1.0
    elif task == "arm-left":
        ok &= delta[1] <= -15 and app.history == ["left"]
    elif task == "arm-up":
        ok &= delta[2] >= 15 and app.history == ["up"]
    elif task == "grip-close":
        ok &= world.jaw() < 0.1 and app.history == ["close"]
    else:
        ok &= peak >= 30
        if task == "left":
            ok &= delta[1] <= -15
        if task == "drop":
            ok &= delta[1] >= 15 and world.jaw() > 1.0
        if task == "drop-left":
            ok &= delta[1] <= -15 and world.jaw() > 1.0
    if moved:
        ok &= perturbed
    if slip:
        ok &= perturbed and app.history.count("lift") >= 2
    row = {"seed": seed, "command": text, "target": target, "task": task,
           "moved_mid_task": moved, "slipped_after_grasp": slip, "perturbed": perturbed,
           "actions": app.history, "reapproaches": max(0, app.history.count("reach") - 1),
           "selected": app.target_name,
           "peak_rise_mm": round(peak, 1), "delta_mm": np.round(delta, 1).tolist(),
           "finished": app.finished, "failed": app.failed, "status": app.status,
           "pass": bool(ok), "image": image_name}
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    baseline = "--baseline" in sys.argv
    suite = next((name for name in ("ood", "holdout", "stress")
                  if f"--{name}" in sys.argv), None)
    if suite is not None:
        from scripts.eval_ood import CASES as STRESS_CASES, FRESH_CASES, HOLDOUT_CASES, run as run_ood

        cases = {"ood": FRESH_CASES, "holdout": HOLDOUT_CASES,
                 "stress": STRESS_CASES}[suite]
        folder = ROOT / "eval" / "visual-action-v3" / suite
        folder.mkdir(parents=True, exist_ok=True)
        app = VisualActionDemo()
        rows = []
        for case in cases:
            row = run_ood(app, case, folder)
            near = case[3].startswith("go near")
            row["physical_pass"] = bool(row["finished"] and row["selected_slot"] == case[4]
                                        and (row["expected_target_peak_rise_mm"] < 5 and row["jaw"] > 1
                                             if near else row["expected_target_final_rise_mm"] >= 30))
            rows.append(row)
        (folder / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"PHYSICAL {sum(row['physical_pass'] for row in rows)}/{len(rows)}", flush=True)
        return
    slip = "--slip" in sys.argv
    folder = ROOT / "eval" / ("visual-action-v1" if baseline else "visual-action-v3") / ("slip" if slip else "baseline" if baseline else "head")
    folder.mkdir(parents=True, exist_ok=True)
    app = ActionDemo() if baseline else VisualActionDemo()
    cases = ((CASES[1], (280002, CASES[7][1], "red_cube", "drop-left", False), CASES[7])
             if slip else (*CASES[7:9], (290000, "lift the red cube then move left then drop it",
                                        "red_cube", "drop-left", False))
             if "--placements" in sys.argv else CASES)
    rows = [run(app, case, folder, slip=slip) for case in cases]
    filename = "placements.json" if "--placements" in sys.argv else "results.json"
    (folder / filename).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"PASS {sum(row['pass'] for row in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
