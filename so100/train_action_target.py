"""Adapt the RGB target head to complete multi-action sentences.

Run: source scripts/vulkan_env.sh && PYTHONPATH=. .venv/bin/python -m so100.train_action_target
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from so100.encode import Siglip2
from so100.rgb_target import TargetHead, pool_patches
from so100.sim import MOVABLES
from so100.train_action_head import phrases
from so100.train_rgb_target import evaluate, scene, setup_world

CHECKPOINT = Path(__file__).resolve().parents[1] / "data" / "rgb-target" / "action-target-v4.pt"


def texts(name: str, heldout: bool = False) -> list[str]:
    return [text for variants in phrases(name, heldout).values() for text in variants]


def collect(eyes: Siglip2, seeds: range, heldout: bool = False, wide: bool = False):
    world = setup_world()
    options = {name: texts(name, heldout) for name in MOVABLES}
    prompts = list(dict.fromkeys(text for group in options.values() for text in group))
    lookup = {}
    for i in range(0, len(prompts), 32):
        batch = eyes.embed_text_mean(prompts[i:i + 32])
        lookup.update(zip(prompts[i:i + 32], batch))
    visual, textual, labels = [], [], []
    for i, seed in enumerate(seeds):
        image, masks, targets = scene(world, seed, wide=wide)
        assert targets is not None
        vis = eyes.embed_image([Image.fromarray(image)])[0]
        pooled = pool_patches(vis, masks)
        for j, name in enumerate(MOVABLES):
            group = options[name]
            for k in range(4 if heldout else 8):
                prompt = group[(i * 7 + k * 61 + j * 13) % len(group)]
                visual.append(pooled)
                textual.append(lookup[prompt])
                labels.append(targets[j])
        if (i + 1) % 40 == 0:
            print(f"grounding embedded {i + 1}/{len(seeds)} scenes", flush=True)
    return (torch.tensor(np.asarray(visual), dtype=torch.float32),
            torch.tensor(np.asarray(textual), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long))


def main() -> None:
    torch.manual_seed(2027)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    eyes = Siglip2(device)
    narrow = collect(eyes, range(294000, 294320))
    spread = collect(eyes, range(310000, 310160), wide=True)
    train = tuple(torch.cat((a, b)) for a, b in zip(narrow, spread))
    val = collect(eyes, range(311000, 311080), heldout=True, wide=True)
    head = TargetHead().to(device)
    prior = CHECKPOINT.with_name("action-target-v3.pt")
    head.load_state_dict(torch.load(prior, map_location=device, weights_only=False)["state"])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-4)
    x, t, y = (item.to(device) for item in train)
    best = (-1, float("-inf"))
    for step in range(1, 801):
        ids = torch.randint(len(y), (128,), device=device)
        logits = head(x[ids], t[ids])
        loss = torch.nn.functional.cross_entropy(logits, y[ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            score = evaluate(head, val, device)
            rank = (score["correct"], -score["nll"])
            print(f"step={step} val={score}", flush=True)
            if rank > best:
                best = rank
                torch.save({"state": head.cpu().state_dict(), "validation": score, "step": step}, CHECKPOINT)
                head.to(device)
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    head.load_state_dict(saved["state"])
    hold = collect(eyes, range(312000, 312080), heldout=True, wide=True)
    report = {"train_scenes": 480, "train_rows": len(y), "validation": saved["validation"], "selected_step": saved["step"],
              "holdout": evaluate(head, hold, device), "checkpoint": str(CHECKPOINT)}
    CHECKPOINT.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
