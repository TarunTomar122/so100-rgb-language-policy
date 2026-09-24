"""Closed-loop RGB action-head experiment. Run on port 8773 after training."""

from __future__ import annotations

import sys

import numpy as np
import torch
from PIL import Image

from so100.action_demo import ActionDemo, PAGE
from so100.action_head import SKILLS, execute_skill
from so100.encode import Siglip2
from so100.executor import Executor
from so100.ik_point_demo import serve
from so100.train_rgb_target import setup_world
from so100.train_visual_action import CHECKPOINT
from so100.train_visual_action_recovery import RECOVERY_CHECKPOINT
from so100.visual_action_head import SIZE, LanguagePrior, RecoveryActionHead, VisualActionHead, observe


class VisualActionDemo(ActionDemo):
    def __init__(self) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.eyes = Siglip2(self.device)
        self.prior = LanguagePrior(self.device)
        self.head = VisualActionHead().to(self.device).eval()
        saved = torch.load(CHECKPOINT, map_location=self.device, weights_only=True)
        self.head.load_state_dict(saved["state"])
        self.recovery_head = RecoveryActionHead().to(self.device).eval()
        self.recovery_head.load_state_dict(torch.load(
            RECOVERY_CHECKPOINT, map_location=self.device, weights_only=True)["state"])
        self.world = setup_world()
        self.executor = Executor(self.world)
        self.seed = 289999
        self.next_choice: tuple[str, float] | None = None
        self.text_feature: np.ndarray | None = None
        self.previous_uv: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.last_ok = True
        self.recovering = False
        self.target_identity: np.ndarray | None = None
        self.decision_history: list[str] = []
        self.pick_start = 0
        self.reset()

    def reset(self) -> dict:
        self.next_choice = None
        self.text_feature = None
        self.previous_uv = None
        self.last_frame = None
        self.last_ok = True
        self.recovering = False
        self.target_identity = None
        self.decision_history = []
        self.pick_start = 0
        return super().reset()

    def command(self, text: str) -> dict:
        text = " ".join(text.strip().split())
        if not text or len(text) > 200:
            raise ValueError("Enter an instruction of at most 200 characters")
        self.instruction = text
        self.text_feature = self.eyes.embed_text_mean([text])[0]
        self.prior.prepare(text)
        self.previous_uv = None
        self.last_frame = None
        self.history = []
        self.grasp = None
        self.target_uv = None
        self.target_confidence = None
        self.target_name = None
        self.initial_target_z = None
        self.target_chroma = None
        self.target_initial_y = None
        self.visual_rise_px = None
        self.lift_unverified = False
        self.grasp_retries = 0
        self.last_model_choice = None
        self.last_ok = True
        self.recovering = False
        self.target_identity = None
        self.decision_history = []
        self.pick_start = 0
        self.failed = False
        self.finished = False
        self.status = "Instruction received"
        self._infer()
        return self.state()

    def _predict(self) -> tuple[str, float] | None:
        return self.next_choice

    def _select_target(self, image: np.ndarray, masks: list[np.ndarray],
                       crops: list[Image.Image], logits: torch.Tensor, retry: bool) -> int:
        features = self.eyes.embed_image(crops).mean(axis=1)
        features /= np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-6)
        if retry and self.target_identity is not None:
            scores = features @ self.target_identity
            if self.target_chroma is not None:
                colors = np.array([image[mask].mean(axis=0) for mask in masks])
                chromas = colors / np.maximum(colors.sum(axis=1, keepdims=True), 1)
                distances = np.linalg.norm(chromas - self.target_chroma, axis=1)
                scores[distances > 0.08] = -np.inf
                if np.isfinite(scores).any():
                    return int(np.argmax(scores))
            return int(np.argmax(features @ self.target_identity))
        index = super()._select_target(image, masks, crops, logits, retry)
        self.target_identity = features[index].copy()
        return index

    def _infer(self) -> None:
        assert self.text_feature is not None
        history = self.history if self.recovering else self.decision_history
        scene, target, text, state, _, uv = observe(
            self.world, self.eyes, self.instruction, self.background,
            history, self.last_ok and not self.recovering,
            self.text_feature, self.previous_uv,
        )
        self.previous_uv = uv
        self.last_frame = self.world.render(SIZE)
        with torch.no_grad():
            inputs = [torch.from_numpy(x).to(self.device)[None] for x in (scene, target, text, state)]
            prior = torch.from_numpy(self.prior.logits(self.world, history)).to(self.device)[None]
            base_logits = self.head(*inputs, prior)
            logits = (self.recovery_head(inputs[3], base_logits)
                      if self.recovering else base_logits)
            probabilities = logits.softmax(-1)[0]
        index = int(probabilities.argmax().item())
        self.next_choice = (SKILLS[index], float(probabilities[index].item()))

    def step(self) -> dict:
        if self.next_choice is not None and (self.last_frame is None or
                not np.array_equal(self.world.render(SIZE), self.last_frame)):
            self._infer()
        choice = self.next_choice
        if choice is None:
            return self.state()
        skill, _ = choice
        self.last_model_choice = skill
        self.next_choice = None
        if skill == "done":
            self.finished = True
            self.status = "Model chose to stop"
            return self.state()
        if len(self.history) >= 12:
            self.failed = True
            self.status = "Stopped after 12 decisions"
            return self.state()
        try:
            if skill == "reach":
                if self.grasp is None:
                    self.pick_start = len(self.decision_history)
                self._find_grasp(retry=self.grasp is not None)
            was_recovering = self.recovering
            self.last_ok = execute_skill(self.world, self.executor, skill, self.grasp)
            self.history.append(skill)
            if self.last_ok and skill == "lift" and self.grasp is not None:
                lifted = self._visual_lifted()
                if lifted is False or (lifted is None and self.world.finger_gap() < 0.015):
                    self.last_ok = False
                    self.recovering = True
                elif lifted is True:
                    self.recovering = False
            if was_recovering:
                if not self.recovering and skill == "lift":
                    self.decision_history.extend(("reach", "lower", "close", "lift"))
            elif self.recovering:
                # Failed physical attempts did not complete the pick subtask.
                self.decision_history = self.decision_history[:self.pick_start]
            else:
                self.decision_history.append(skill)
            self.status = (f"Executed {skill}" if self.last_ok
                           else f"{skill} did not achieve its expected result")
        except ValueError as exc:
            self.last_ok = False
            self.history.append(skill)
            self.status = str(exc)
        self._infer()
        return self.state()


if __name__ == "__main__":
    app = VisualActionDemo()
    if "--check" in sys.argv:
        app.command("go near the blue cube")
        for _ in range(12):
            if app._predict() is None:
                break
            app.step()
            if app.finished or app.failed:
                break
        print(app.history, app.status)
    else:
        serve(app, PAGE, 8773)
