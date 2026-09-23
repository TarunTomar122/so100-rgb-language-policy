"""Free-text, learned next-skill SO-100 tabletop demo.

Run: source scripts/vulkan_env.sh && PYTHONPATH=. .venv/bin/python -m so100.action_demo
Open: http://127.0.0.1:8772/
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from so100.action_head import ActionHead, ActionText, SKILLS, execute_skill, state_vector
from so100.encode import Siglip2
from so100.executor import Executor
from so100.ik_point_demo import serve
from so100.rgb_grasp import estimate_grasp
from so100.rgb_target import TargetHead, candidate_masks, pool_patches
from so100.train_action_head import CHECKPOINT as ACTION_CHECKPOINT
from so100.train_action_target import CHECKPOINT as TARGET_CHECKPOINT
from so100.train_rgb_target import SIZE, scene, setup_world
from so100.sim import MOVABLES
from so100.vision import project_xyz

PAGE = Path(__file__).resolve().parents[1] / "demo" / "action_demo.html"


class ActionDemo:
    def __init__(self) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.eyes = Siglip2(self.device)
        self.language = ActionText(self.device)
        self.target_head = TargetHead().to(self.device).eval()
        self.target_head.load_state_dict(torch.load(TARGET_CHECKPOINT, map_location=self.device, weights_only=False)["state"])
        self.action_head = ActionHead().to(self.device).eval()
        self.action_head.load_state_dict(torch.load(ACTION_CHECKPOINT, map_location=self.device, weights_only=False)["state"])
        self.world = setup_world()
        self.executor = Executor(self.world)
        self.seed = 289999
        self.reset()

    def reset(self) -> dict:
        self.seed += 1
        scene(self.world, self.seed, annotate=False)
        self.instruction = ""
        self.tokens: torch.Tensor | None = None
        self.history: list[str] = []
        self.grasp: dict | None = None
        self.target_uv: list[float] | None = None
        self.target_confidence: float | None = None
        self.target_name: str | None = None
        self.initial_target_z: float | None = None
        self.target_chroma: np.ndarray | None = None
        self.target_initial_y: float | None = None
        self.visual_rise_px: float | None = None
        self.grasp_retries = 0
        self.last_model_choice: str | None = None
        self.failed = False
        self.finished = False
        self.status = "Enter an instruction"
        return self.state()

    def command(self, text: str) -> dict:
        text = " ".join(text.strip().split())
        if not text or len(text) > 200:
            raise ValueError("Enter an instruction of at most 200 characters")
        self.instruction = text
        self.tokens = torch.tensor(self.language.embed([text]), device=self.device)
        self.history = []
        self.grasp = None
        self.target_uv = None
        self.target_confidence = None
        self.target_name = None
        self.initial_target_z = None
        self.target_chroma = None
        self.target_initial_y = None
        self.visual_rise_px = None
        self.grasp_retries = 0
        self.last_model_choice = None
        self.failed = False
        self.finished = False
        self.status = "Instruction loaded. The action head will predict the first step."
        return self.state()

    @torch.no_grad()
    def _predict(self) -> tuple[str, float] | None:
        if self.tokens is None or self.failed or self.finished:
            return None
        state = torch.tensor(state_vector(self.world, self.history)[None], device=self.device)
        logits = self.action_head(self.tokens, state)[0]
        prob = torch.softmax(logits, -1)
        index = int(prob.argmax().item())
        return SKILLS[index], float(prob[index].item())

    def _find_grasp(self, retry: bool = False) -> None:
        image = self.world.render(SIZE)
        masks = candidate_masks(image)
        if len(masks) != 4:
            raise ValueError("RGB camera cannot separate four object candidates in this view")
        # One image-encoder pass for this pickup. Later steps use cached text and arm state.
        vis = self.eyes.embed_image([Image.fromarray(image)])[0]
        txt = self.eyes.embed_text_mean([self.instruction])[0]
        candidates = pool_patches(vis, masks)
        with torch.no_grad():
            logits = self.target_head(
                torch.tensor(candidates[None], device=self.device),
                torch.tensor(txt[None], device=self.device),
            )[0]
            prob = torch.softmax(logits, -1)
            index = int(prob.argmax().item())
            confidence = float(prob[index].item())
        ys, xs = np.nonzero(masks[index])
        self.target_uv = [float(xs.mean()), float(ys.mean())]
        self.target_confidence = confidence
        color = image[masks[index]].mean(axis=0)
        self.target_chroma = color / max(float(color.sum()), 1.0)
        self.target_initial_y = float(ys.mean())
        self.grasp = estimate_grasp(self.world, image, masks[index])
        # Simulator pose is used only to report the actual lift, never to choose motion.
        if not retry:
            self.target_name = min(MOVABLES, key=lambda name: np.linalg.norm(
                np.asarray(project_xyz(self.world, self.world.object_xyz(name), SIZE)) - self.target_uv
            ))
            self.initial_target_z = float(self.world.object_xyz(self.target_name)[2])

    def _visual_lifted(self) -> bool:
        assert self.target_chroma is not None and self.target_initial_y is not None
        image = self.world.render(SIZE)
        masks = candidate_masks(image)
        if len(masks) != 4:
            raise ValueError("RGB camera lost the target after lifting")
        colors = [image[mask].mean(axis=0) for mask in masks]
        chromas = [color / max(float(color.sum()), 1.0) for color in colors]
        distances = [float(np.linalg.norm(color - self.target_chroma)) for color in chromas]
        index = int(np.argmin(distances))
        if distances[index] > 0.08:
            raise ValueError("RGB camera cannot track the lifted target")
        self.visual_rise_px = self.target_initial_y - float(np.nonzero(masks[index])[0].mean())
        # ponytail: fixed side camera, 18 px observed lift threshold; recalibrate on a real camera.
        return self.visual_rise_px >= 18

    def _retry_grasp(self, yaw_offset: float, bias: float) -> bool:
        # Retarget from a new RGB frame; simulator object coordinates are never used.
        self.grasp_retries += 1
        if not execute_skill(self.world, self.executor, "open", None):
            return False
        if not self.executor.goto(self.world.tcp() + np.array([0.0, 0.0, 0.08])):
            return False
        if not self.executor.goto(np.array([0.0, -0.20, 0.18])):
            return False
        self._find_grasp(retry=True)
        assert self.grasp is not None
        self.grasp["yaw_deg"] += yaw_offset
        self.grasp["pinch_bias_m"] = bias
        return all(execute_skill(self.world, self.executor, skill, self.grasp)
                   for skill in ("reach", "lower", "close"))

    def step(self) -> dict:
        choice = self._predict()
        if choice is None:
            return self.state()
        skill, confidence = choice
        self.last_model_choice = skill
        if len(self.history) >= 12:
            self.failed = True
            self.status = "Stopped after 12 actions; the model did not finish the instruction"
        elif skill == "done":
            self.finished = True
            self.status = "Instruction complete" if self.history else f"Model chose done ({confidence:.0%})"
        else:
            try:
                if skill == "reach" and self.grasp is None:
                    self._find_grasp()
                success = execute_skill(self.world, self.executor, skill, self.grasp)
                retried = False
                if (success and skill == "close" and self.grasp is not None
                        and self.grasp["circularity"] < 0.92 and self.world.finger_gap() < 0.015):
                    retried = True
                    success = self._retry_grasp(0, 0.008)
                if success and skill == "lift" and self.grasp is not None:
                    lifted = self._visual_lifted()
                    retries = ((0, 0.008), (60, 0.003), (-60, 0.003), (90, 0.008))
                    while not lifted and self.grasp_retries < len(retries):
                        retried = True
                        offset, bias = retries[self.grasp_retries]
                        success = self._retry_grasp(offset, bias)
                        if not success:
                            break
                        success = execute_skill(self.world, self.executor, "lift", self.grasp)
                        if not success:
                            break
                        lifted = self._visual_lifted()
                    if success and not lifted:
                        self.failed = True
                        self.status = f"RGB sees no lift after {len(retries)} grasp retries"
                if not success:
                    self.failed = True
                    self.status = ("RGB grasp retry failed" if retried
                                   else f"{skill} failed: IK or physical limit")
                else:
                    self.history.append(skill)
                    rise = self._target_rise()
                    if not self.failed and skill == "lift" and rise is not None and rise < 30:
                        self.failed = True
                        self.status = f"Grasp missed: selected object rose {rise:.0f} mm"
                    elif not self.failed:
                        suffix = " after RGB retry" if retried else ""
                        self.status = f"Executed {skill}{suffix} ({confidence:.0%} model confidence)"
            except ValueError as exc:
                self.failed = True
                self.status = str(exc)
        return self.state()

    def _target_rise(self) -> float | None:
        if self.target_name is None or self.initial_target_z is None:
            return None
        return (float(self.world.object_xyz(self.target_name)[2]) - self.initial_target_z) * 1000

    def state(self) -> dict:
        frame = io.BytesIO()
        Image.fromarray(self.world.render(SIZE)).save(frame, format="JPEG", quality=85)
        choice = self._predict()
        return {
            "frame": "data:image/jpeg;base64," + base64.b64encode(frame.getvalue()).decode("ascii"),
            "instruction": self.instruction,
            "seed": self.seed,
            "history": self.history,
            "next_action": None if choice is None else choice[0],
            "action_confidence": None if choice is None else choice[1],
            "last_model_choice": self.last_model_choice,
            "tip_uv": project_xyz(self.world, self.world.tcp(), SIZE),
            "target_uv": self.target_uv,
            "grasp_uv": None if self.grasp is None else project_xyz(self.world, self.grasp["xyz"], SIZE),
            "target_confidence": self.target_confidence,
            "selected_object": self.target_name,
            "target_rise_mm": self._target_rise(),
            "grasp_retries": self.grasp_retries,
            "visual_rise_px": self.visual_rise_px,
            "status": self.status,
            "finished": self.finished,
            "failed": self.failed,
        }


if __name__ == "__main__":
    app = ActionDemo()
    if "--check" in sys.argv:
        cases = [
            ("lift the red cube", ("reach", "lower", "close", "lift"), "red_cube", 0),
            ("lift the blue cube then move left", ("reach", "lower", "close", "lift", "left"), "blue_cube", -1),
            ("pick up the green cylinder then move right and drop it", ("reach", "lower", "close", "lift", "right", "open"), "green_cylinder", 1),
            ("close the gripper", ("close",), None, 0),
            ("move left then move right", ("left", "right"), None, 0),
        ]
        for i, (text, expected, target, direction) in enumerate(cases):
            app.seed = 289999 + i
            app.reset()
            before = None if target is None else app.world.object_xyz(target)
            app.command(text)
            max_rise = 0.0
            while app._predict() is not None:
                app.step()
                max_rise = max(max_rise, app._target_rise() or 0)
            print(f"{text}: {app.history} | {app.status} | peak rise {max_rise:.0f} mm", flush=True)
            assert tuple(app.history) == expected, text
            if target is None:
                assert app.finished
            else:
                assert app.finished and app.target_name == target and max_rise >= 30, text
                if direction:
                    assert direction * (app.world.object_xyz(target)[1] - before[1]) >= 0.015, text
        print("ACTION DEMO CHECK 5/5", flush=True)
    else:
        serve(app, PAGE, 8772)
