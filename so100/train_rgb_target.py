"""Train a small language target head on frozen, single-pass SigLIP RGB features.

Run: source scripts/vulkan_env.sh && PYTHONPATH=. .venv/bin/python -m so100.train_rgb_target
"""

from __future__ import annotations

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from so100.encode import Siglip2
from so100.rgb_target import TargetHead, candidate_masks, pool_patches
from so100.sim import MOVABLES, TABLE_TOP, XML, Tabletop
from so100.vision import project_xyz

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "data" / "rgb-target" / "target.pt"
ANCHORS = ((-0.035, -0.34), (0.035, -0.34), (-0.035, -0.28), (0.035, -0.28))
SIZE = 512


def command(name: str, verb: str = "pick up") -> str:
    return f"{verb} the {name.replace('_', ' ')}"


def setup_world(variant: tuple | None = None) -> Tabletop:
    if variant is None:
        world = Tabletop()
    else:
        # MuJoCo collision constants are compiled from XML. Mutating geom_type
        # at runtime changes the rendering but leaves intermittent contacts.
        tree = ET.parse(XML)
        slot, shape, dims, _half_height, color = variant
        geom = tree.find(f".//geom[@name='{slot}']")
        assert geom is not None
        geom.set("type", shape)
        geom.set("size", " ".join(str(value) for value in dims if value != 0))
        if color is not None:
            geom.set("rgba", " ".join(str(value) for value in color))
        with tempfile.NamedTemporaryFile(dir=ROOT, suffix=".xml") as source:
            tree.write(source.name)
            world = Tabletop(Path(source.name))
    world.model.site_pos[world._tcp] = [0.0089, -0.1064, 0.0]
    return world


def scene(world: Tabletop, seed: int, annotate: bool = True, wide: bool = False,
          on_empty=None, allow_merged: bool = False) -> tuple[np.ndarray, list[np.ndarray], list[int] | None]:
    rng = np.random.default_rng(seed)
    for _ in range(100 if wide else 30):
        world.home()
        world.park_scene()
        world.data.qpos[5] = 1.4
        world.hold()
        names = list(MOVABLES)
        rng.shuffle(names)
        points = []
        if wide:
            for _ in MOVABLES:
                for _ in range(100):
                    candidate = rng.uniform((-0.065, -0.375), (0.065, -0.245))
                    if all(np.linalg.norm(candidate - point) >= 0.052 for point in points):
                        points.append(candidate)
                        break
            if len(points) != len(MOVABLES):
                continue
            placements = [(name, x, y, float(rng.uniform(0, np.pi / 2)))
                          for name, (x, y) in zip(names, points)]
        else:
            placements = []
            for name, (ax, ay) in zip(names, ANCHORS):
                yaw = float(rng.uniform(0, np.pi / 2))
                x, y = ax + float(rng.uniform(-0.005, 0.005)), ay + float(rng.uniform(-0.005, 0.005))
                placements.append((name, x, y, yaw))
        world.set_camera(np.array([0.75, -0.13, 0.34]), np.array([0.0, -0.19, 0.10]))
        mujoco.mj_forward(world.model, world.data)
        if on_empty is not None:
            on_empty(world.render(SIZE))
        for name, x, y, yaw in placements:
            world.set_object(name, np.array([x, y, TABLE_TOP + 0.020]), np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        world.step(80)
        mujoco.mj_forward(world.model, world.data)
        image = world.render(SIZE)
        masks = candidate_masks(image)
        if len(masks) != 4 and not (allow_merged and not annotate):
            continue
        if not annotate:
            return image, masks, None
        centers = np.array([[np.nonzero(m)[1].mean(), np.nonzero(m)[0].mean()] for m in masks])
        actual = np.array([project_xyz(world, world.object_xyz(name), SIZE) for name in MOVABLES])
        cost = np.linalg.norm(actual[:, None, :] - centers[None, :, :], axis=-1)
        rows, cols = linear_sum_assignment(cost)
        if max(cost[rows, cols]) > 24:
            continue
        labels = [int(cols[np.where(rows == i)[0][0]]) for i in range(4)]
        return image, masks, labels
    raise RuntimeError(f"cannot render four separated RGB masks for seed {seed}")


def collect(eyes: Siglip2, world: Tabletop, seeds: range, verbs: tuple[str, ...]):
    visual, textual, labels = [], [], []
    for index, seed in enumerate(seeds):
        image, masks, targets = scene(world, seed)
        assert targets is not None
        prompts = [command(name, verbs[index % len(verbs)]) for name in MOVABLES]
        vis, txt = eyes.embed([Image.fromarray(image)], prompts)
        pooled = pool_patches(vis[0], masks)
        visual.extend([pooled] * 4)
        textual.extend(txt.mean(axis=1))
        labels.extend(targets)
        if (index + 1) % 40 == 0:
            print(f"embedded {index + 1}/{len(seeds)} scenes", flush=True)
    return (
        torch.tensor(np.asarray(visual), dtype=torch.float32),
        torch.tensor(np.asarray(textual), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


@torch.no_grad()
def evaluate(head: TargetHead, data: tuple[torch.Tensor, ...], device: str) -> dict:
    head.eval()
    visual, textual, labels = (x.to(device) for x in data)
    logits = head(visual, textual)
    predictions = logits.argmax(-1)
    hit = (predictions == labels).cpu().numpy().reshape(-1, 4)
    return {"correct": int(hit.sum()), "total": int(hit.size), "all_four": int(hit.all(axis=1).sum()), "scenes": int(len(hit)), "nll": float(torch.nn.functional.cross_entropy(logits, labels))}


def main() -> None:
    torch.manual_seed(2026)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading frozen SigLIP on {device}", flush=True)
    eyes = Siglip2(device)
    world = setup_world()
    train = collect(eyes, world, range(220000, 220160), ("pick up", "lift", "grab"))
    val = collect(eyes, world, range(230000, 230040), ("lift", "grab"))
    head = TargetHead().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    x, t, y = (v.to(device) for v in train)
    best = (-1, float("-inf"))
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, 501):
        ids = torch.randint(len(y), (64,), device=device)
        logits = head(x[ids], t[ids])
        loss = torch.nn.functional.cross_entropy(logits, y[ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            score = evaluate(head, val, device)
            print(f"step={step} loss={loss.item():.3f} val={score}", flush=True)
            rank = (score["correct"], -score["nll"])
            if rank > best:
                best = rank
                torch.save({"state": head.cpu().state_dict(), "validation": score, "step": step}, CHECKPOINT)
                head.to(device)
    chosen = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    head.load_state_dict(chosen["state"])
    hold = collect(eyes, world, range(300000, 300040), ("pick up", "lift", "grab"))
    report = {"train_scenes": 160, "validation": chosen["validation"], "selected_step": chosen["step"], "holdout": evaluate(head, hold, device), "checkpoint": str(CHECKPOINT)}
    CHECKPOINT.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
