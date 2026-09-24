"""Train the image-conditioned one-step head from varied simulator demonstrations.

Only the teacher sees simulator object identities. Inference uses RGB, instruction,
and arm sensors. Run after sourcing scripts/vulkan_env.sh.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from so100.action_head import SKILLS, execute_skill
from so100.encode import Siglip2
from so100.executor import Executor
from so100.rgb_grasp import estimate_grasp
from so100.rgb_target import candidate_masks
from so100.sim import MOVABLES, TABLE_TOP
from so100.train_action_head import approach_phrases, phrases, simple_phrases
from so100.train_rgb_target import scene, setup_world
from so100.vision import project_xyz
from so100.visual_action_head import SIZE, LanguagePrior, VisualActionHead, observe

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data" / "visual-action" / "head.pt"
CACHE = Path("/tmp/so100-visual-action-features-v2.pt")


def target_grasp(world, name: str, image: np.ndarray) -> dict | None:
    """Training teacher only: identify which RGB mask belongs to a simulator object."""
    masks = candidate_masks(image)
    uv = project_xyz(world, world.object_xyz(name), len(image))
    if not masks or uv is None:
        return None
    centers = [np.array([np.nonzero(mask)[1].mean(), np.nonzero(mask)[0].mean()]) for mask in masks]
    index = int(np.argmin([np.linalg.norm(center - uv) for center in centers]))
    if np.linalg.norm(centers[index] - uv) > len(image) * 0.08:
        return None
    try:
        return estimate_grasp(world, image, masks[index])
    except ValueError:
        return None


def collect(eyes: Siglip2, seeds: range, heldout: bool = False) -> tuple[list, dict]:
    world = setup_world()
    executor = Executor(world)
    prior = LanguagePrior(eyes.device)
    rows = []
    counts = {skill: 0 for skill in SKILLS}
    for seed in seeds:
        empty = []
        image, _, _ = scene(world, seed, on_empty=lambda _: empty.append(world.render(SIZE)))
        background = empty[-1]
        start = world.snapshot()
        for obj_index, name in enumerate(MOVABLES):
            world.restore(start)
            options = [None, (), ("left",), ("right",), ("open",), ("left", "open"), ("right", "open")]
            suffix = options[(seed * 4 + obj_index) % len(options)]
            texts = (approach_phrases(name, heldout) if suffix is None
                     else phrases(name, heldout)[suffix])
            instruction = texts[(seed + obj_index) % len(texts)]
            text_feature = eyes.embed_text_mean([instruction])[0]
            prior.prepare(instruction)
            program = (["reach"] if suffix is None else ["reach", "lower", "close", "lift", *suffix])
            grasp = target_grasp(world, name, image)
            if grasp is None:
                continue
            history: list[str] = []
            last_ok = True
            previous_uv = None

            def record(skill: str, recovery: bool = False) -> None:
                nonlocal previous_uv
                features = observe(world, eyes, instruction, background, history, last_ok,
                                   text_feature, previous_uv)
                previous_uv = features[5]
                rows.append((*features[:4], prior.logits(world, history), SKILLS.index(skill), recovery))
                counts[skill] += 1

            for skill in program:
                # Perturb a fraction of training episodes after reaching. The
                # same command/history now needs a fresh visual decision.
                if skill == "lower" and (seed + obj_index) % 5 == 0:
                    pose = world.object_pose(name)
                    pose[0] = float(np.clip(pose[0] + 0.028, -0.085, 0.085))
                    world.set_object(name, pose[:3], pose[3:])
                    mujoco.mj_forward(world.model, world.data)
                    moved = target_grasp(world, name, world.render(len(image)))
                    if moved is not None:
                        record("reach", recovery=True)
                        last_ok = execute_skill(world, executor, "reach", moved)
                        history.append("reach")
                        grasp = moved
                        if not last_ok:
                            break
                record(skill)
                last_ok = execute_skill(world, executor, skill, grasp)
                history.append(skill)
                if not last_ok:
                    break
            else:
                record("done")

        # Arm-only instructions provide examples with no named object.
        world.restore(start)
        options = list(simple_phrases(heldout).items())
        program, texts = options[seed % len(options)]
        instruction = texts[seed % len(texts)]
        text_feature = eyes.embed_text_mean([instruction])[0]
        prior.prepare(instruction)
        history = []
        previous_uv = None
        for skill in (*program, "done"):
            features = observe(world, eyes, instruction, background, history, True,
                               text_feature, previous_uv)
            previous_uv = features[5]
            rows.append((*features[:4], prior.logits(world, history), SKILLS.index(skill), False))
            counts[skill] += 1
            if skill != "done":
                if not execute_skill(world, executor, skill, None):
                    break
                history.append(skill)
        print(f"visual action data {seed}: {len(rows)} rows", flush=True)
    return rows, counts


def pack(rows: list, device: str):
    return tuple(torch.tensor(np.stack([row[i] for row in rows]), dtype=torch.float32, device=device)
                 for i in range(5)) + (torch.tensor([row[5] for row in rows], dtype=torch.long, device=device),
                                       torch.tensor([row[6] for row in rows], dtype=torch.bool, device=device))


def score(head: VisualActionHead, data: tuple[torch.Tensor, ...]) -> dict:
    head.eval()
    with torch.no_grad():
        predictions = torch.cat([head(*(x[i:i+128] for x in data[:5])).argmax(-1)
                                 for i in range(0, len(data[5]), 128)])
    correct = predictions == data[5]
    prior_correct = data[4].argmax(-1) == data[5]
    recovery = data[6]
    return {"normal_correct": int(correct[~recovery].sum()), "normal_total": int((~recovery).sum()),
            "recovery_correct": int(correct[recovery].sum()), "recovery_total": int(recovery.sum()),
            "prior_normal": int(prior_correct[~recovery].sum()),
            "prior_recovery": int(prior_correct[recovery].sum())}


def main() -> None:
    torch.manual_seed(2026)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if CACHE.exists():
        train_rows, train_counts, val_rows, val_counts = torch.load(CACHE, weights_only=False)
        print(f"loaded cached RGB features from {CACHE}", flush=True)
    else:
        eyes = Siglip2(device)
        train_rows, train_counts = collect(eyes, range(260000, 260096))
        val_rows, val_counts = collect(eyes, range(270000, 270016), heldout=True)
        torch.save((train_rows, train_counts, val_rows, val_counts), CACHE)
    train, val = pack(train_rows, device), pack(val_rows, device)
    head = VisualActionHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3)
    counts = torch.bincount(train[5], minlength=len(SKILLS)).float().clamp_min(1)
    weights = (counts.sum() / (len(SKILLS) * counts)).clamp(max=6)
    recovery_ids = torch.where(train[6])[0]
    assert len(recovery_ids) > 0
    best = -1
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    for step in range(401):
        if step:
            ids = torch.cat((torch.randint(len(train[5]), (48,), device=device),
                             recovery_ids[torch.randint(len(recovery_ids), (16,), device=device)]))
            logits = head(*(x[ids] for x in train[:5]))
            loss = (torch.nn.functional.cross_entropy(logits, train[5][ids], weight=weights)
                    + 0.005 * (logits - train[4][ids] * 0.5).square().mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % 50 == 0:
            result = score(head, val)
            print(f"step={step} heldout={result}", flush=True)
            rank = result["normal_correct"] + 2 * result["recovery_correct"]
            if rank > best:
                best = rank
                torch.save({"state": {k: v.cpu() for k, v in head.state_dict().items()},
                            "validation": result, "step": step}, CHECKPOINT)
    report = {"train": len(train_rows), "validation": len(val_rows),
              "train_labels": train_counts, "validation_labels": val_counts,
              "best_weighted_score": best, "checkpoint": str(CHECKPOINT)}
    CHECKPOINT.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
