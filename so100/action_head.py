"""Learned language + arm-state predictor over executable SO-100 skills."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from so100.executor import Executor
from so100.sim import Tabletop

SKILLS = ("reach", "lower", "close", "lift", "left", "right", "up", "down", "open", "done")
PICK = ("reach", "lower", "close", "lift")
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ponytail: fixed-camera grasp calibration; tune this on physical SO-100 trials.
PINCH_BIAS_M = 0.003


def pinch_from_xyz(world: Tabletop, xyz: np.ndarray, along: float) -> np.ndarray:
    """Place the fixed finger pad beside the RGB-estimated grasp centre."""
    xyz = np.asarray(xyz, dtype=np.float64)
    fixed = world.data.geom_xpos[world.finger_geoms[0]]
    moving = world.data.geom_xpos[world.finger_geoms[4]]
    axis = fixed[:2] - moving[:2]
    norm = float(np.linalg.norm(axis))
    axis = axis / norm if norm >= 1e-6 else np.array([1.0, 0.0])
    desired_pad = xyz + np.array([axis[0] * along, axis[1] * along, 0.0])
    return world.tcp() + (desired_pad - fixed)


class ActionText:
    def __init__(self, device: str) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(texts, padding="max_length", truncation=True,
                               max_length=48, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        tokens = self.model(**batch).last_hidden_state.float()
        return (tokens * batch["attention_mask"][:, :, None]).cpu().numpy()


def state_vector(world: Tabletop, completed: list[str]) -> np.ndarray:
    history = np.array([min(completed.count(skill), 3) for skill in SKILLS], dtype=np.float32)
    return np.concatenate([[world.tcp()[2] * 4, world.jaw() / 1.5], history]).astype(np.float32)


class ActionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text = nn.Linear(384, 128)
        self.sequence = nn.GRU(128, 128, batch_first=True)
        self.state = nn.Sequential(nn.Linear(2 + len(SKILLS), 64), nn.GELU())
        self.out = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, len(SKILLS)))

    def forward(self, tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        lengths = (tokens.abs().sum(-1) > 0).sum(-1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.text(tokens), lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.sequence(packed)
        return self.out(torch.cat([hidden[-1], self.state(state)], dim=-1))


def execute_skill(world: Tabletop, executor: Executor, skill: str, grasp: dict | None) -> bool:
    """Physical meaning of each learned skill. Targets come from RGB grasp, not object pose."""
    tcp = world.tcp()
    if skill == "reach":
        if grasp is None:
            return False
        desired_roll = float(world.model.key("home").qpos[4]) + np.deg2rad(grasp["yaw_deg"])
        delta = desired_roll - float(world.joints()[4])
        if not executor.execute(np.array([0, 0, 0, delta, 0])):
            return False
        grasp["pinch"] = pinch_from_xyz(
            world, grasp["xyz"], grasp["along"] + grasp.get("pinch_bias_m", PINCH_BIAS_M))
        return executor.goto(grasp["pinch"] + np.array([0.0, 0.0, 0.06]))
    if skill == "lower":
        return grasp is not None and "pinch" in grasp and executor.goto(grasp["pinch"])
    if skill == "close":
        return executor.execute(np.array([0, 0, 0, 0, -1.0]))
    if skill == "lift":
        return executor.goto(tcp + np.array([0.0, 0.0, 0.06]))
    if skill == "open":
        return executor.execute(np.array([0, 0, 0, 0, 1.0]))
    # In this fixed camera, screen-left is -Y and screen-right is +Y.
    delta = {"left": (0, -0.03, 0), "right": (0, 0.03, 0),
             "up": (0, 0, 0.03), "down": (0, 0, -0.03)}.get(skill)
    if delta is not None:
        goal = tcp + np.asarray(delta)
        if skill == "down":
            goal[2] = max(goal[2], 0.045)
            if tcp[2] - goal[2] < 0.003:
                return False
        return executor.goto(goal)
    return skill == "done"
