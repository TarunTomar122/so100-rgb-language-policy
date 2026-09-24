"""Closed-loop RGB action-head experiment. Run on port 8773 after training."""

from __future__ import annotations

import sys

import mujoco
import numpy as np
import torch

from so100.action_demo import ActionDemo, PAGE
from so100.action_head import SKILLS, execute_skill
from so100.encode import Siglip2
from so100.executor import Executor
from so100.ik_point_demo import serve
from so100.rgb_target import TargetHead
from so100.train_action_target import CHECKPOINT as TARGET_CHECKPOINT
from so100.sim import MOVABLES, TABLE_TOP, TABLE_X, TABLE_Y
from so100.train_rgb_target import setup_world
from so100.train_visual_action import CHECKPOINT
from so100.train_visual_action_recovery import RECOVERY_CHECKPOINT
from so100.train_done_head import DONE_CHECKPOINT
from so100.visual_action_head import SIZE, DoneCalibrator, LanguagePrior, RecoveryActionHead, VisualActionHead, observe
from so100.vision import project_xyz, unproject_pixels

FRAME_SIZE = 512


class VisualActionDemo(ActionDemo):
    def __init__(self) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.eyes = Siglip2(self.device)
        self.target_head = TargetHead().to(self.device).eval()
        self.target_head.load_state_dict(torch.load(
            TARGET_CHECKPOINT, map_location=self.device, weights_only=True)["state"])
        self.prior = LanguagePrior(self.device)
        self.head = VisualActionHead().to(self.device).eval()
        saved = torch.load(CHECKPOINT, map_location=self.device, weights_only=True)
        self.head.load_state_dict(saved["state"])
        self.recovery_head = RecoveryActionHead().to(self.device).eval()
        self.recovery_head.load_state_dict(torch.load(
            RECOVERY_CHECKPOINT, map_location=self.device, weights_only=True)["state"])
        self.done_head = DoneCalibrator().to(self.device).eval()
        self.done_head.load_state_dict(torch.load(
            DONE_CHECKPOINT, map_location=self.device, weights_only=True)["state"])
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
        self.selected_mask: np.ndarray | None = None
        self.selected_feature: np.ndarray | None = None
        self.move_selected: str | None = None
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
        self.selected_mask = None
        self.selected_feature = None
        self.move_selected = None
        self.decision_history = []
        self.pick_start = 0
        self.carrying = False
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
        self.selected_mask = None
        self.selected_feature = None
        self.move_selected = None
        self.decision_history = []
        self.pick_start = 0
        self.carrying = False
        self.failed = False
        self.finished = False
        self.status = "Instruction received"
        self._infer()
        return self.state()

    def _predict(self) -> tuple[str, float] | None:
        return self.next_choice

    def move_click(self, u: float, v: float) -> dict:
        """Simulator UI intervention; object poses never enter policy inference."""
        if not np.isfinite(u) or not np.isfinite(v) or not (0 <= u < FRAME_SIZE and 0 <= v < FRAME_SIZE):
            raise ValueError("Click inside the simulator image")
        if self.move_selected is None:
            projected = [(name, project_xyz(self.world, self.world.object_xyz(name), FRAME_SIZE))
                         for name in MOVABLES]
            visible = [(name, np.linalg.norm(np.array(point) - [u, v]))
                       for name, point in projected if point is not None]
            if not visible:
                raise ValueError("No object is visible")
            name, distance = min(visible, key=lambda item: item[1])
            if distance > 28:
                raise ValueError("Click a cube or other object first")
            self.move_selected = name
            self.status = "Click a new spot on the table"
            return self.state()
        xyz = unproject_pixels(self.world, np.array([u]), np.array([v]), TABLE_TOP, FRAME_SIZE)[0]
        if not np.isfinite(xyz).all() or not (TABLE_X[0] + 0.02 <= xyz[0] <= TABLE_X[1] - 0.02
                                             and TABLE_Y[0] + 0.02 <= xyz[1] <= TABLE_Y[1] - 0.02):
            raise ValueError("Click an open spot on the table")
        name = self.move_selected
        pose = self.world.object_pose(name)
        pose[:2] = xyz[:2]
        self.world.set_object(name, pose[:3], pose[3:])
        mujoco.mj_forward(self.world.model, self.world.data)
        self.move_selected = None
        self.status = f"Moved {name.replace('_', ' ')}"
        if self.text_feature is not None and not self.finished and not self.failed:
            self._infer()
        return self.state()

    def state(self) -> dict:
        result = super().state()
        result["can_move_objects"] = True
        result["move_selected"] = self.move_selected
        result["move_uv"] = (None if self.move_selected is None else
                             project_xyz(self.world, self.world.object_xyz(self.move_selected), FRAME_SIZE))
        return result

    def _find_grasp(self, retry: bool = False) -> None:
        if self.selected_mask is None:
            raise ValueError("RGB camera found no selected target in this view")
        super()._find_grasp(retry=retry, selected_mask=self.selected_mask)

    def _infer(self) -> None:
        assert self.text_feature is not None
        history = self.history if self.recovering else self.decision_history
        scene, target, text, state, _, uv, self.selected_mask, self.selected_feature = observe(
            self.world, self.eyes, self.instruction, self.background,
            history, self.last_ok and not self.recovering,
            self.text_feature, self.previous_uv, self.target_identity, self.target_chroma,
            self.target_head,
        )
        self.previous_uv = uv
        self.last_frame = self.world.render(SIZE)
        with torch.no_grad():
            inputs = [torch.from_numpy(x).to(self.device)[None] for x in (scene, target, text, state)]
            prior = torch.from_numpy(self.prior.logits(self.world, history)).to(self.device)[None]
            base_logits = self.head(*inputs, prior)
            logits = (self.recovery_head(inputs[3], base_logits)
                      if self.recovering else self.done_head(
                          inputs[1], inputs[2], inputs[3], base_logits))
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
                    self.target_identity = self.selected_feature
                self._find_grasp(retry=self.grasp is not None)
            was_recovering = self.recovering
            self.last_ok = execute_skill(self.world, self.executor, skill, self.grasp)
            self.history.append(skill)
            if self.last_ok and skill == "lift" and self.grasp is not None:
                lifted = self._visual_lifted()
                if lifted is False or (lifted is None and self.world.finger_gap() < 0.015):
                    self.last_ok = False
                    self.recovering = True
                    self.carrying = False
                else:
                    self.recovering = False
                    self.carrying = True
            elif self.last_ok and self.carrying and skill in ("left", "right", "up", "down"):
                lifted = self._visual_lifted()
                if lifted is False or (lifted is None and self.world.finger_gap() < 0.015):
                    self.last_ok = False
                    self.recovering = True
                    self.carrying = False
            elif skill == "open":
                self.carrying = False
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
        app.seed = 280001
        app.reset()
        before = app.world.object_xyz("red_cube")
        app.move_click(*project_xyz(app.world, before, FRAME_SIZE))
        destination = before.copy()
        destination[0] += 0.025
        destination[2] = TABLE_TOP
        app.move_click(*project_xyz(app.world, destination, FRAME_SIZE))
        assert np.linalg.norm(app.world.object_xyz("red_cube")[:2] - destination[:2]) < 0.002
        app.command("go near the red cube")
        app.step()
        assert app.target_name == "red_cube" and app.history == ["reach"]
        print("click-to-move and RGB retarget ok")
    else:
        serve(app, PAGE, 8773)
