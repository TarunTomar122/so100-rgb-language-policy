"""Train a small stop decision on frozen RGB/action features."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import torch

from so100.action_head import SKILLS, execute_skill
from so100.encode import Siglip2
from so100.executor import Executor
from so100.sim import TABLE_TOP
from so100.train_rgb_target import scene, setup_world
from so100.train_visual_action import CACHE as BASE_CACHE, CHECKPOINT, pack, target_grasp
from so100.train_visual_action_recovery import CACHE as SLIP_CACHE
from so100.visual_action_head import SIZE, DoneCalibrator, LanguagePrior, VisualActionHead, observe

DONE_CHECKPOINT = CHECKPOINT.with_name("done.pt")
VARIED_CACHE = BASE_CACHE.with_name("so100-done-varied-v3.pt")

# Separate training scenes. Evaluation seeds and object combinations stay out.
VARIANTS = (
    ("red_cube", "violet cube", "box", (.015, .015, .015), .015, (.43, .20, .69, 1)),
    ("yellow_block", "coral block", "box", (.012, .022, .015), .015, (.90, .34, .27, 1)),
    ("red_cube", "gold ball", "sphere", (.018, 0, 0), .018, (.87, .65, .08, 1)),
    ("blue_cube", "blue sphere", "sphere", (.018, 0, 0), .018, None),
    ("green_cylinder", "cyan tube", "cylinder", (.012, .024, 0), .024, (.10, .60, .72, 1)),
    ("yellow_block", "amber puck", "cylinder", (.018, .009, 0), .009, (.80, .50, .09, 1)),
    ("blue_cube", "blue capsule", "capsule", (.011, .012, 0), .023, None),
    ("red_cube", "red egg", "ellipsoid", (.013, .013, .020), .020, None),
)


def collect_varied(eyes: Siglip2, seeds: range) -> tuple[list, dict]:
    prior = LanguagePrior(eyes.device)
    rows = []
    counts = {"episodes": 0, "verified_done": 0, "failed_grasp": 0, "skipped": 0}
    for seed in seeds:
        slot, noun, shape, dims, half_height, color = VARIANTS[seed % len(VARIANTS)]
        world = setup_world((slot, shape, dims, half_height, color))
        executor = Executor(world)
        empty = []
        scene(world, seed, on_empty=lambda _: empty.append(world.render(SIZE)))
        pose = world.object_pose(slot)
        world.set_object(slot, np.array([pose[0], pose[1], TABLE_TOP + half_height + .004]), pose[3:])
        world.step(100)
        mujoco.mj_forward(world.model, world.data)
        start = world.snapshot()
        before_z = float(world.object_xyz(slot)[2])
        grasp = target_grasp(world, slot, world.render(SIZE))
        if grasp is None:
            counts["skipped"] += 1
            continue
        near = seed % 7 == 0
        verb = ("go near" if near else ("lift", "pick up", "grab")[(seed // len(VARIANTS)) % 3])
        instruction = f"{verb} the {noun}"
        text_feature = eyes.embed_text_mean([instruction])[0]
        prior.prepare(instruction)
        history = []
        previous_uv = None
        counts["episodes"] += 1

        def record(skill: str, last_ok: bool = True) -> None:
            nonlocal previous_uv
            features = observe(world, eyes, instruction, empty[-1], history, last_ok,
                               text_feature, previous_uv)
            previous_uv = features[5]
            rows.append((*features[:4], prior.logits(world, history), SKILLS.index(skill), False))

        program = ("reach",) if near else ("reach", "lower", "close", "lift")
        for skill in program:
            record(skill)
            if not execute_skill(world, executor, skill, grasp):
                break
            history.append(skill)
        else:
            lifted = float(world.object_xyz(slot)[2]) - before_z >= .03
            if near or lifted:
                record("done")
                counts["verified_done"] += 1
            else:
                record("reach", last_ok=False)
                counts["failed_grasp"] += 1
        if seed % 4 == 0:
            world.restore(start)
            instruction = ("raise the gripper slightly", "lift the gripper a little")[(seed // 4) % 2]
            text_feature = eyes.embed_text_mean([instruction])[0]
            prior.prepare(instruction)
            history = []
            previous_uv = None
            before_tip = world.tcp()[2]
            if execute_skill(world, executor, "up", None) and world.tcp()[2] - before_tip >= .015:
                history.append("up")
                record("done")
        print(f"varied done data {seed}: {len(rows)} rows", flush=True)
    return rows, counts


def main() -> None:
    torch.manual_seed(2026)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if not BASE_CACHE.exists() or not SLIP_CACHE.exists():
        raise SystemExit("Run so100.train_visual_action and so100.train_visual_action_recovery first")
    base_train, _, base_val, _ = torch.load(BASE_CACHE, weights_only=False)
    slip_train, slip_val = torch.load(SLIP_CACHE, weights_only=False)
    if VARIED_CACHE.exists():
        varied_train, train_counts, varied_val, val_counts = torch.load(VARIED_CACHE, weights_only=False)
    else:
        eyes = Siglip2(device)
        varied_train, train_counts = collect_varied(eyes, range(360000, 360048))
        varied_val, val_counts = collect_varied(eyes, range(370000, 370012))
        torch.save((varied_train, train_counts, varied_val, val_counts), VARIED_CACHE)
    train = pack(base_train + slip_train + varied_train, device)
    val = pack(base_val + slip_val + varied_val, device)
    base = VisualActionHead().to(device).eval()
    base.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True)["state"])

    def encode(data):
        with torch.no_grad():
            return torch.cat([base(*(x[i:i+128] for x in data[:5]))
                              for i in range(0, len(data[5]), 128)])

    train_logits, val_logits = encode(train), encode(val)
    done = SKILLS.index("done")
    positive = torch.where(train[5] == done)[0]
    negative = torch.where(train[5] != done)[0]
    head = DoneCalibrator().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3)
    best = -1
    for step in range(601):
        if step:
            ids = torch.cat((positive[torch.randint(len(positive), (16,), device=device)],
                             negative[torch.randint(len(negative), (48,), device=device)]))
            logits = head(train[1][ids], train[2][ids], train[3][ids], train_logits[ids])
            loss = torch.nn.functional.cross_entropy(logits, train[5][ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % 50 == 0:
            head.eval()
            with torch.no_grad():
                pred = head(val[1], val[2], val[3], val_logits).argmax(-1)
            correct = int((pred == val[5]).sum())
            false_stop = int(((pred == done) & (val[5] != done)).sum())
            missed_stop = int(((pred != done) & (val[5] == done)).sum())
            print(f"step={step} next={correct}/{len(pred)} false_done={false_stop} missed_done={missed_stop}", flush=True)
            rank = correct - false_stop
            if rank > best:
                best = rank
                torch.save({"state": {k: v.cpu() for k, v in head.state_dict().items()},
                            "validation": {"correct": correct, "total": len(pred),
                                           "false_done": false_stop,
                                           "missed_done": missed_stop}, "step": step},
                           DONE_CHECKPOINT)
            head.train()
    DONE_CHECKPOINT.with_suffix(".json").write_text(json.dumps({
        "train": len(train[5]), "validation": len(val[5]), "best_rank": best,
        "varied_train": train_counts, "varied_validation": val_counts,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
