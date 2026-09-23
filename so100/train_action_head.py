"""Imitate verified RGB-grasp/IK demonstrations with a language-conditioned skill head.

Run: source scripts/vulkan_env.sh && PYTHONPATH=. .venv/bin/python -m so100.train_action_head
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from so100.action_head import ActionHead, ActionText, PICK, SKILLS, execute_skill, state_vector
from so100.executor import Executor
from so100.rgb_grasp import estimate_grasp
from so100.sim import MOVABLES
from so100.train_rgb_target import scene, setup_world

CHECKPOINT = Path(__file__).resolve().parents[1] / "data" / "action-head" / "action-v10.pt"

ALIASES = {
    "red_cube": ("red cube", "red box", "red block"),
    "blue_cube": ("blue cube", "blue box", "blue block"),
    "green_cylinder": ("green cylinder", "green round object", "green tube"),
    "yellow_block": ("yellow block", "yellow box", "yellow rectangular block"),
}


def phrases(name: str, heldout: bool = False) -> dict[tuple[str, ...], list[str]]:
    obj = name.replace("_", " ")
    if heldout:
        return {
            (): [f"raise the {obj}", f"take the {obj} off the table"],
            ("left",): [f"raise the {obj} and shift it left"],
            ("right",): [f"take the {obj} and move it right"],
            ("open",): [f"raise the {obj} and release it"],
            ("left", "open"): [f"take the {obj}, go left, and let go"],
            ("right", "open"): [f"raise the {obj}, go right, then release it"],
        }
    aliases = ALIASES[name]
    result: dict[tuple[str, ...], list[str]] = {
        (): [], ("left",): [], ("right",): [], ("open",): [],
        ("left", "open"): [], ("right", "open"): [],
    }
    for item in aliases:
        beginnings = [
            f"lift the {item}", f"pick up the {item}", f"grab the {item}",
            f"raise the {item} from the table", f"take the {item} off the table",
            f"hold the {item}", f"please lift the {item}",
            f"can you pick up the {item}", f"could you lift the {item}",
            f"would you grab the {item}", f"lift {item} for me", f"get the {item}",
            f"pick that {item} up", f"take hold of the {item}",
            f"could you raise the {item}", f"bring the {item} up off the table",
            f"fetch the {item}",
        ]
        directions = {
            "left": ("then move left", "and move it to the left", "and carry it left",
                     "then shift left", "and bring it left", "then go left",
                     "and carry it leftward"),
            "right": ("then move right", "and move it to the right", "and carry it right",
                      "then shift right", "and bring it right", "then go right",
                      "and carry it rightward"),
        }
        releases = ("then drop it", "and release it", "and let go of it",
                    "and set it down", "then put it down", "and open the gripper",
                    "and release your grip", "and let go of your hold")
        result[()].extend(beginnings)
        result[()].extend([f"lift the {item} and keep holding it",
                           f"raise the {item} without letting go",
                           f"take the {item} and hold onto it"])
        for i, beginning in enumerate(beginnings):
            result[("open",)].append(f"{beginning} {releases[i % len(releases)]}")
            for direction in ("left", "right"):
                move = directions[direction]
                for j in (i % len(move), (i + 3) % len(move)):
                    result[(direction,)].append(f"{beginning} {move[j]}")
                    result[(direction, "open")].append(
                        f"{beginning} {move[j]} {releases[(i + j) % len(releases)]}")
        for direction in ("left", "right"):
            result[(direction, "open")].extend([
                f"put the {item} down on the {direction}",
                f"place the {item} to the {direction}",
                f"set the {item} down to the {direction}",
                f"carry the {item} to the {direction} and release it",
                f"move the {item} to the {direction} and let go",
                f"grab the {item} and put it to the {direction}",
                f"pick up the {item} and set it down on the {direction}",
                f"take the {item} away and leave it to the {direction}",
                f"get the {item} and leave it on your {direction}",
                f"bring the {item} over to the {direction} and let go",
            ])
    return result


def approach_phrases(name: str, heldout: bool = False) -> list[str]:
    if heldout:
        obj = name.replace("_", " ")
        return [f"get near the {obj}", f"position the gripper by the {obj}"]
    return [text for obj in ALIASES[name] for text in (
        f"approach the {obj}", f"move near the {obj}", f"go close to the {obj}",
        f"bring the gripper close to the {obj}", f"hover above the {obj}",
        f"reach toward the {obj}", f"move the arm toward the {obj}",
        f"approach the {obj} without grabbing it", f"move near the {obj} and stop there",
        f"hover over the {obj} without touching it",
        f"move toward the {obj} and leave it untouched",
        f"get close to the {obj} but do not pick it up",
    )]


def simple_phrases(heldout: bool = False) -> dict[tuple[str, ...], list[str]]:
    if heldout:
        return {
            ("left",): ["shift the arm to the left"], ("right",): ["shift the arm to the right"],
            ("up",): ["raise the gripper a little"], ("down",): ["lower the gripper a little"],
            ("close",): ["shut the gripper"], ("open",): ["release the gripper"],
            ("left", "right"): ["go left followed by right"],
            ("right", "left"): ["go right followed by left"],
            ("up", "down"): ["raise the gripper and lower it"],
            ("right", "up"): ["nudge the arm right before lifting the gripper"],
            ("up", "right"): ["lift the wrist, then nudge right"],
        }
    return {
        ("left",): ["move left", "move the arm left", "shift left", "slide left",
                    "shift the gripper left", "slide the robot arm left", "nudge the arm left"],
        ("right",): ["move right", "move the arm right", "shift right", "slide right",
                     "move the gripper rightward"],
        ("up",): ["move up", "move the gripper up", "raise the arm",
                  "raise the end effector", "lift the gripper", "bring the arm up"],
        ("down",): ["move down", "move the gripper down", "lower the arm",
                    "lower the end effector", "bring the wrist down"],
        ("close",): ["close the gripper", "close grippers", "shut the jaws",
                     "clamp the jaws", "squeeze the gripper", "close your fingers"],
        ("open",): ["open the gripper", "open grippers", "drop it", "release the object",
                    "open your fingers", "release your grip", "spread the jaws"],
        ("left", "right"): ["move left then move right", "go left and then right",
                            "shift the arm left before moving right", "first left, then right",
                            "move the gripper left followed by right"],
        ("right", "left"): ["move right then move left", "go right and then left",
                            "shift the arm right before moving left", "first right, then left",
                            "move the gripper right followed by left"],
        ("up", "down"): ["move up then move down", "raise the arm then lower it"],
        ("left", "up"): ["move left and then raise the arm", "shift left before going up",
                         "go left then lift the gripper"],
        ("up", "left"): ["raise the arm before moving left", "lift the gripper then go left",
                         "move up followed by left"],
        ("right", "up"): ["move right and then raise the arm", "go right then lift the gripper",
                            "shift the gripper right before raising it", "first move the arm right, then lift it",
                            "slide right and afterward move up", "move toward the right and then raise the gripper"],
        ("up", "right"): ["raise the arm before moving right", "lift the gripper then go right",
                            "first raise the wrist, then shift right", "move up and afterward slide right"],
    }


def add_rows(out: list, texts: list[str], records: list[tuple[np.ndarray, str]], repeat: int = 1) -> None:
    for text in texts:
        out.extend((text, state, SKILLS.index(label)) for _ in range(repeat) for state, label in records)


def collect(seeds: range, heldout: bool = False) -> list[tuple[str, np.ndarray, int]]:
    world = setup_world()
    executor = Executor(world)
    rows: list[tuple[str, np.ndarray, int]] = []
    for seed in seeds:
        image, masks, labels = scene(world, seed)
        assert labels is not None
        start = world.snapshot()
        for name, mask_index in zip(MOVABLES, labels):
            world.restore(start)
            grasp = estimate_grasp(world, image, masks[mask_index])
            initial_z = float(world.object_xyz(name)[2])
            history: list[str] = []
            pick_records = []
            for skill in PICK:
                pick_records.append((state_vector(world, history), skill))
                if not execute_skill(world, executor, skill, grasp):
                    break
                history.append(skill)
            if len(pick_records) >= 2:
                add_rows(rows, approach_phrases(name, heldout),
                         [(pick_records[0][0], "reach"), (pick_records[1][0], "done")],
                         repeat=1 if heldout else 5)
            if len(history) != len(PICK) or float(world.object_xyz(name)[2]) - initial_z < 0.03:
                continue
            lifted = world.snapshot()
            for suffix, texts in phrases(name, heldout).items():
                world.restore(lifted)
                branch = pick_records.copy()
                completed = list(PICK)
                valid = True
                for skill in suffix:
                    branch.append((state_vector(world, completed), skill))
                    if not execute_skill(world, executor, skill, grasp):
                        valid = False
                        break
                    completed.append(skill)
                if valid:
                    branch.append((state_vector(world, completed), "done"))
                    add_rows(rows, texts, branch)
            world.restore(lifted)
            drop_records = [(state_vector(world, []), "open")]
            if execute_skill(world, executor, "open", None):
                drop_records.append((state_vector(world, ["open"]), "done"))
                add_rows(rows, [f"drop the {name.replace('_', ' ')}", "drop it"], drop_records)

        for program, texts in simple_phrases(heldout).items():
            world.restore(start)
            completed = []
            records = []
            for skill in program:
                records.append((state_vector(world, completed), skill))
                if not execute_skill(world, executor, skill, None):
                    break
                completed.append(skill)
            if len(completed) == len(program):
                records.append((state_vector(world, completed), "done"))
                add_rows(rows, texts, records, repeat=10)
        print(f"action demonstrations {seed}: {len(rows)} samples", flush=True)
    return rows


def main() -> None:
    torch.manual_seed(2026)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train = collect(range(240000, 240060))
    val = collect(range(250000, 250012), heldout=True)
    texts = list(dict.fromkeys(row[0] for row in train + val))
    language = ActionText(device)
    tokens = np.concatenate([language.embed(texts[i:i + 16]) for i in range(0, len(texts), 16)])
    lookup = {text: i for i, text in enumerate(texts)}

    def pack(rows: list[tuple[str, np.ndarray, int]]):
        return (torch.tensor([lookup[row[0]] for row in rows], dtype=torch.long, device=device),
                torch.tensor(np.stack([row[1] for row in rows]), device=device),
                torch.tensor([row[2] for row in rows], dtype=torch.long, device=device))

    tx, sx, y = pack(train)
    vx, vs, vy = pack(val)
    tok = torch.tensor(tokens, device=device)
    head = ActionHead().to(device)
    prior = torch.load(CHECKPOINT.with_name("action-v9.pt"), map_location=device, weights_only=False)["state"]
    prior["state.0.weight"] = prior["state.0.weight"][:, :-1]
    prior["out.2.weight"] = prior["out.2.weight"][:-1]
    prior["out.2.bias"] = prior["out.2.bias"][:-1]
    head.load_state_dict(prior)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    counts = torch.bincount(y, minlength=len(SKILLS)).float().clamp_min(1)
    weights = (counts.sum() / (len(SKILLS) * counts)).clamp(max=8)
    best = -1
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    for step in range(0, 801):
        if step:
            ids = torch.randint(len(y), (128,), device=device)
            logits = head(tok[tx[ids]], sx[ids])
            loss = torch.nn.functional.cross_entropy(logits, y[ids], weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % 100 == 0:
            head.eval()
            with torch.no_grad():
                pred = torch.cat([head(tok[vx[i:i + 256]], vs[i:i + 256]).argmax(-1)
                                  for i in range(0, len(vy), 256)])
                score = int((pred == vy).sum().item())
            print(f"step={step} heldout={score}/{len(vy)}", flush=True)
            if score > best:
                best = score
                torch.save({"state": head.cpu().state_dict(), "heldout_correct": score,
                            "heldout_total": len(vy), "step": step}, CHECKPOINT)
                head.to(device)
            head.train()
    report = {"train_rows": len(y), "heldout_rows": len(vy), "heldout_correct": best,
              "checkpoint": str(CHECKPOINT)}
    CHECKPOINT.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
