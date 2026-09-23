"""Seeded, no-training stress test of the frozen RGB + language policy."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from so100.action_demo import ActionDemo
from so100.rgb_target import candidate_masks
from so100.sim import TABLE_TOP
from so100.train_rgb_target import SIZE, scene

ROOT = Path(__file__).resolve().parents[1]
PURPLE = (.55, .20, .65, 1)
ORANGE = (.92, .35, .08, 1)
CYAN = (.08, .62, .70, 1)
PINK = (.86, .32, .58, 1)

# Conditions were fixed before looking at outcomes. Slots are MuJoCo identities
# for grading; the policy receives only the rendered RGB and instruction.
CASES = [
    ("baseline-red", 321000, "baseline", "lift the red cube", "red_cube", {}),
    ("baseline-near", 321001, "baseline", "go near the blue cube", "blue_cube", {}),
    ("red-ball", 321010, "new shape", "lift the red ball", "red_cube",
     {"object": ("red_cube", "sphere", (.018, 0, 0), .018, None)}),
    ("blue-tube", 321011, "new shape", "lift the blue cylinder", "blue_cube",
     {"object": ("blue_cube", "cylinder", (.011, .021, 0), .021, None)}),
    ("green-ball", 321012, "new shape", "pick up the green ball", "green_cylinder",
     {"object": ("green_cylinder", "sphere", (.017, 0, 0), .017, None)}),
    ("yellow-bar", 321013, "new shape", "lift the yellow bar", "yellow_block",
     {"object": ("yellow_block", "box", (.010, .024, .012), .012, None)}),
    ("red-egg", 321014, "new shape", "lift the red egg", "red_cube",
     {"object": ("red_cube", "ellipsoid", (.013, .013, .020), .020, None)}),
    ("purple-cube", 321020, "new color", "lift the purple cube", "red_cube",
     {"object": ("red_cube", "box", (.015, .015, .015), .015, PURPLE)}),
    ("orange-cube", 321021, "new color", "lift the orange cube", "blue_cube",
     {"object": ("blue_cube", "box", (.015, .015, .015), .015, ORANGE)}),
    ("cyan-cylinder", 321022, "new color", "lift the cyan cylinder", "green_cylinder",
     {"object": ("green_cylinder", "cylinder", (.013, .015, 0), .015, CYAN)}),
    ("pink-block", 321023, "new color", "lift the pink block", "yellow_block",
     {"object": ("yellow_block", "box", (.012, .022, .015), .015, PINK)}),
    ("purple-ball", 321030, "new color + shape", "lift the purple ball", "red_cube",
     {"object": ("red_cube", "sphere", (.018, 0, 0), .018, PURPLE)}),
    ("cyan-puck", 321031, "new color + shape", "pick up the cyan puck", "green_cylinder",
     {"object": ("green_cylinder", "cylinder", (.018, .007, 0), .007, CYAN)}),
    ("orange-capsule", 321032, "new color + shape", "lift the orange capsule", "blue_cube",
     {"object": ("blue_cube", "capsule", (.011, .012, 0), .023, ORANGE)}),
    ("camera-shift", 321040, "environment", "lift the red cube", "red_cube",
     {"camera": ((.63, -.09, .32), (0, -.23, .09))}),
    ("camera-low", 321041, "environment", "lift the blue cube", "blue_cube",
     {"camera": ((.70, -.21, .22), (0, -.27, .05))}),
    ("teal-table", 321042, "environment", "lift the green cylinder", "green_cylinder",
     {"table": (.18, .42, .34, 1)}),
    ("dim-light", 321043, "environment", "lift the yellow block", "yellow_block",
     {"light": .35}),
    ("wide-placement", 321044, "environment", "lift the red cube", "red_cube",
     {"wide": True}),
    ("extra-distractor", 321045, "environment", "lift the blue cube", "blue_cube",
     {"marker": True}),
    ("low-friction", 321046, "environment", "lift the green cylinder", "green_cylinder",
     {"friction": .25}),
    ("sensor-noise-8", 321050, "image noise", "lift the red cube", "red_cube",
     {"noise": "gaussian8"}),
    ("sensor-noise-25", 321051, "image noise", "lift the blue cube", "blue_cube",
     {"noise": "gaussian25"}),
    ("blur", 321052, "image noise", "lift the green cylinder", "green_cylinder",
     {"noise": "blur"}),
    ("jpeg-25", 321053, "image noise", "lift the yellow block", "yellow_block",
     {"noise": "jpeg25"}),
    ("dark-noisy", 321054, "image noise", "lift the red cube", "red_cube",
     {"noise": "dark_noise"}),
    ("wide-shift-noise", 321060, "combined", "lift the blue cube", "blue_cube",
     {"wide": True, "camera": ((.63, -.09, .32), (0, -.23, .09)), "noise": "gaussian8"}),
    ("purple-ball-dim-jpeg", 321061, "combined", "lift the purple ball", "red_cube",
     {"object": ("red_cube", "sphere", (.018, 0, 0), .018, PURPLE), "light": .35, "noise": "jpeg25"}),
]

# Fixed before running the revised policy. These scene seeds and combinations
# were not used to train the repo's heads or choose this policy revision.
HOLDOUT_CASES = [
    ("violet-cube", 331100, "new color", "lift the violet cube", "red_cube",
     {"object": ("red_cube", "box", (.015, .015, .015), .015, (.42, .18, .72, 1))}),
    ("coral-block", 331101, "new color", "pick up the coral block", "yellow_block",
     {"object": ("yellow_block", "box", (.012, .022, .015), .015, (.92, .30, .28, 1))}),
    ("lime-cylinder", 331102, "new color", "lift the lime cylinder", "green_cylinder",
     {"object": ("green_cylinder", "cylinder", (.013, .015, 0), .015, (.55, .85, .12, 1))}),
    ("red-sphere-near-cube", 331103, "similar color", "lift the red ball", "blue_cube",
     {"object": ("blue_cube", "sphere", (.018, 0, 0), .018, (.8, .12, .12, 1))}),
    ("blue-capsule", 331104, "new shape", "lift the blue capsule", "blue_cube",
     {"object": ("blue_cube", "capsule", (.011, .012, 0), .023, None)}),
    ("tall-cyan-tube", 331105, "new shape + height", "pick up the cyan tube", "green_cylinder",
     {"object": ("green_cylinder", "cylinder", (.012, .028, 0), .028, CYAN)}),
    ("flat-amber-puck", 331106, "new shape + height", "lift the amber puck", "yellow_block",
     {"object": ("yellow_block", "cylinder", (.018, .008, 0), .008, (.83, .48, .07, 1))}),
    ("low-camera-purple-sphere", 331107, "combined", "lift the purple sphere", "red_cube",
     {"object": ("red_cube", "sphere", (.018, 0, 0), .018, PURPLE),
      "camera": ((.70, -.21, .22), (0, -.27, .05))}),
    ("teal-table-yellow", 331108, "environment", "lift the yellow block", "yellow_block",
     {"table": (.18, .42, .34, 1)}),
    ("dim-violet-cube", 331109, "combined", "lift the violet cube", "red_cube",
     {"object": ("red_cube", "box", (.015, .015, .015), .015, (.42, .18, .72, 1)), "light": .35}),
    ("jpeg-red-egg", 331110, "combined", "pick up the red egg", "red_cube",
     {"object": ("red_cube", "ellipsoid", (.013, .013, .020), .020, None), "noise": "jpeg25"}),
    ("shift-extra-distractor", 331111, "combined", "lift the blue cube", "blue_cube",
     {"camera": ((.63, -.09, .32), (0, -.23, .09)), "marker": True}),
    ("wide-noisy-blue", 331112, "combined", "lift the blue cube", "blue_cube",
     {"wide": True, "noise": "gaussian8"}),
]

# Final, untouched check after the implementation and the first holdout review.
FRESH_CASES = [
    ("gold-cube", 341200, "new color", "lift the gold cube", "red_cube",
     {"object": ("red_cube", "box", (.015, .015, .015), .015, (.86, .63, .06, 1))}),
    ("lilac-block", 341201, "new color", "pick up the lilac block", "yellow_block",
     {"object": ("yellow_block", "box", (.012, .022, .015), .015, (.60, .42, .80, 1))}),
    ("red-capsule", 341202, "new shape", "lift the red capsule", "red_cube",
     {"object": ("red_cube", "capsule", (.011, .012, 0), .023, None)}),
    ("blue-ellipsoid", 341203, "new shape", "lift the blue egg", "blue_cube",
     {"object": ("blue_cube", "ellipsoid", (.013, .013, .020), .020, None)}),
    ("shift-orange-cube", 341204, "combined", "lift the orange cube", "blue_cube",
     {"object": ("blue_cube", "box", (.015, .015, .015), .015, ORANGE),
      "camera": ((.63, -.09, .32), (0, -.23, .09))}),
    ("teal-table-red-ball", 341205, "combined", "lift the red ball", "red_cube",
     {"object": ("red_cube", "sphere", (.018, 0, 0), .018, None),
      "table": (.18, .42, .34, 1)}),
    ("dim-cyan-cylinder", 341206, "combined", "lift the cyan cylinder", "green_cylinder",
     {"object": ("green_cylinder", "cylinder", (.013, .015, 0), .015, CYAN), "light": .35}),
    ("jpeg-purple-cube", 341207, "combined", "lift the purple cube", "red_cube",
     {"object": ("red_cube", "box", (.015, .015, .015), .015, PURPLE), "noise": "jpeg25"}),
    ("wide-blue-capsule", 341208, "combined", "lift the blue capsule", "blue_cube",
     {"object": ("blue_cube", "capsule", (.011, .012, 0), .023, None), "wide": True}),
    ("near-lilac-block", 341209, "instruction", "go near the lilac block", "yellow_block",
     {"object": ("yellow_block", "box", (.012, .022, .015), .015, (.60, .42, .80, 1))}),
]


def corrupt(image: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind.startswith("gaussian"):
        sigma = int(kind.removeprefix("gaussian"))
        return np.clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape), 0, 255).astype(np.uint8)
    if kind == "blur":
        return cv2.GaussianBlur(image, (9, 9), 2.0)
    if kind == "jpeg25":
        encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 25])[1]
        return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "dark_noise":
        return np.clip(image.astype(np.float32) * .45 + rng.normal(0, 9, image.shape), 0, 255).astype(np.uint8)
    raise ValueError(kind)


def run(app: ActionDemo, case: tuple, output: Path, original_render, original: dict, video: bool = False) -> dict:
    slug, seed, group, instruction, target, change = case
    world = app.world
    model = world.model
    world.render = original_render
    for key, value in original.items():
        if key == "headlight_diffuse":
            model.vis.headlight.diffuse[:] = value
        elif key == "headlight_ambient":
            model.vis.headlight.ambient[:] = value
        else:
            getattr(model, key)[:] = value
    app.seed = seed - 1
    app.reset()
    if change.get("wide"):
        scene(world, seed, annotate=False, wide=True)
    if "object" in change:
        slot, shape, dims, half_height, rgba = change["object"]
        gid = world.obj_geom[slot]
        model.geom_type[gid] = int(getattr(mujoco.mjtGeom, f"mjGEOM_{shape.upper()}"))
        model.geom_size[gid] = dims
        if rgba is not None:
            model.geom_rgba[gid] = rgba
        pose = world.object_pose(slot)
        world.set_object(slot, np.array([pose[0], pose[1], TABLE_TOP + half_height + .004]), pose[3:])
        world.step(100)
    if "camera" in change:
        eye, look = change["camera"]
        world.set_camera(np.array(eye), np.array(look))
    if "table" in change:
        model.mat_rgba[model.geom_matid[world._table]] = change["table"]
    if "light" in change:
        model.light_diffuse[:] *= change["light"]
        model.vis.headlight.diffuse[:] *= change["light"]
        model.vis.headlight.ambient[:] *= change["light"]
    if change.get("marker"):
        world.set_mocap("marker", np.array([0.0, -.215, TABLE_TOP + .004]))
    if "friction" in change:
        model.geom_friction[world.obj_geom[target], 0] = change["friction"]
    mujoco.mj_forward(model, world.data)
    # Evaluation setup mirrors taking a real empty-table calibration frame
    # after changing the camera or lighting; the policy never receives poses.
    saved = world.snapshot()
    world.park_scene()
    mujoco.mj_forward(model, world.data)
    app.background = original_render(SIZE)
    world.restore(saved)
    clean_before = original_render(SIZE)
    if "noise" in change:
        rng = np.random.default_rng(seed + 1000)
        noise = change["noise"]
        world.render = lambda size=256, camera="main": corrupt(original_render(size, camera), noise, rng) if camera == "main" else original_render(size, camera)
    observed_before = world.render(SIZE)
    masks = candidate_masks(observed_before, background=app.background)
    mask_count = len(masks)
    mask_areas = [int(mask.sum()) for mask in masks]
    before_xyz = world.object_xyz(target)
    before_z = float(before_xyz[2])
    app.command(instruction)
    process = None
    execute = app.executor.execute
    goto = app.executor.goto
    find_grasp = app._find_grasp
    grasp_estimates = []

    def observed_find_grasp(retry: bool = False) -> None:
        find_grasp(retry)
        if app.grasp is not None:
            grasp_estimates.append({"retry": retry, "xyz": np.round(app.grasp["xyz"], 4).tolist(),
                                    "yaw_deg": round(float(app.grasp["yaw_deg"]), 1)})

    app._find_grasp = observed_find_grasp
    if video:
        process = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
             "-pixel_format", "rgb24", "-video_size", f"{SIZE}x{SIZE}", "-framerate", "12",
             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / f"{slug}.mp4")],
            stdin=subprocess.PIPE,
        )
        assert process.stdin is not None
        ticks = 0

        def frame(force: bool = False) -> None:
            nonlocal ticks
            ticks += 1
            if force or ticks % 2 == 0:
                process.stdin.write(np.ascontiguousarray(world.render(SIZE)).tobytes())

        app.executor.execute = lambda action, on_frame=None: execute(action, on_frame=frame)
        app.executor.goto = lambda goal, on_frame=None: goto(goal, on_frame=frame)
        for _ in range(12):
            frame(force=True)
    predictions = []
    peak_rise = 0.0
    exception = None
    try:
        for _ in range(13):
            choice = app._predict()
            if choice is None:
                break
            predictions.append([choice[0], None if choice[1] is None else round(choice[1], 3)])
            try:
                app.step()
            except Exception as exc:  # Keep the next seeded trial running and record the crash.
                exception = f"{type(exc).__name__}: {exc}"
                break
            peak_rise = max(peak_rise, (float(world.object_xyz(target)[2]) - before_z) * 1000)
    finally:
        app.executor.execute = execute
        app.executor.goto = goto
        app._find_grasp = find_grasp
        if process is not None:
            for _ in range(12):
                frame(force=True)
            process.stdin.close()
            assert process.wait() == 0, f"ffmpeg failed for {slug}"
    final_rise = (float(world.object_xyz(target)[2]) - before_z) * 1000
    after = original_render(SIZE)
    montage = np.concatenate([clean_before, observed_before, after], axis=1)
    Image.fromarray(montage).save(output / f"{slug}.jpg", quality=86)
    expected = ("reach",) if instruction.startswith("go near") else ("reach", "lower", "close", "lift")
    passed = (not exception and app.finished and not app.failed and tuple(app.history) == expected
              and app.target_name == target and
              ((peak_rise < 5 and world.jaw() > 1.0) if len(expected) == 1
               else final_rise >= 30 and world.jaw() < 1.0))
    result = {
        "case": slug, "group": group, "seed": seed, "instruction": instruction,
        "target_slot": target, "condition": change, "mask_count_initial": mask_count,
        "mask_areas_initial": mask_areas,
        "predictions": predictions, "actual_actions": list(app.history), "selected_slot": app.target_name,
        "target_confidence": app.target_confidence, "target_uv": app.target_uv,
        "expected_target_initial_xyz": np.round(before_xyz, 4).tolist(),
        "grasp_estimates": grasp_estimates,
        "grasp_xyz": None if app.grasp is None else np.round(app.grasp["xyz"], 4).tolist(),
        "expected_target_peak_rise_mm": round(peak_rise, 1),
        "expected_target_final_rise_mm": round(final_rise, 1), "jaw": round(world.jaw(), 3),
        "grasp_retries": app.grasp_retries, "status": app.status, "exception": exception,
        "lift_unverified": app.lift_unverified,
        "finished": app.finished, "failed": app.failed, "pass": bool(passed),
        "image": f"{slug}.jpg", "video": f"{slug}.mp4" if video else None,
    }
    print(f"{slug}: {'PASS' if passed else 'FAIL'} | masks {mask_count} | {app.history} | "
          f"selected {app.target_name} | final rise {final_rise:.0f} mm | {exception or app.status}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stress", "control", "known-word", "holdout", "fresh"), default="stress")
    parser.add_argument("--select", nargs="*", default=[])
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--limit", type=int, default=len(CASES))
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "ood-v1")
    args = parser.parse_args()
    cases = CASES
    if args.mode == "control":
        cases = [(f"control-{slug}", seed, "control", f"lift the {target.replace('_', ' ')}", target, {})
                 for slug, seed, _group, _instruction, target, _change in CASES[2:]]
    elif args.mode == "known-word":
        cases = [(f"known-word-{slug}", seed, "known-word", f"lift the {target.replace('_', ' ')}", target, change)
                 for slug, seed, _group, _instruction, target, change in CASES[2:] if "object" in change]
    elif args.mode == "holdout":
        cases = HOLDOUT_CASES
    elif args.mode == "fresh":
        cases = FRESH_CASES
    if args.select:
        cases = [case for case in cases if case[0] in args.select]
    args.output.mkdir(parents=True, exist_ok=True)
    app = ActionDemo()
    model = app.world.model
    original_render = app.world.render
    original = {key: getattr(model, key).copy() for key in
                ("geom_type", "geom_size", "geom_rgba", "geom_friction", "mat_rgba", "light_diffuse")}
    original["headlight_diffuse"] = model.vis.headlight.diffuse.copy()
    original["headlight_ambient"] = model.vis.headlight.ambient.copy()
    results = []
    for case in cases[:args.limit]:
        results.append(run(app, case, args.output, original_render, original, args.video))
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"TOTAL {sum(row['pass'] for row in results)}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
