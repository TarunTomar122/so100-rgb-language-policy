"""Deterministic local controller.

Turns a numeric action into joint targets using IK and the SO-100 joint
limits. It reads the arm. It does not read object poses or the task.
"""

from __future__ import annotations

import mujoco
import numpy as np

from so100.actions import Action, CATALOG
from so100.sim import JAW_DOF, ROLL_DOF, Tabletop

# A free-space close needs ~170 steps at 2 ms. Returning earlier reports
# success while the jaw is still open.
SETTLE_STEPS = 240
# Kinematic plan is reachable if the TCP can get this close to the commanded point.
IK_ABS = 0.004
IK_FRAC = 0.40
PENETRATION = 0.003
JAW_OPEN = 1.55
JAW_CLOSE = -0.10


class Executor:
    def __init__(self, world: Tabletop) -> None:
        self.world = world

    def plan(self, vec: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return (arm ctrl target, mechanically valid) without physics."""
        w = self.world
        vec = np.asarray(vec, dtype=np.float64)
        q = w.joints()
        target_q = q.copy()
        if abs(vec[4]) > 0.5:
            target_q[JAW_DOF] = JAW_OPEN if vec[4] > 0 else JAW_CLOSE
            lo, hi = w.model.jnt_range[JAW_DOF]
            target_q[JAW_DOF] = float(np.clip(target_q[JAW_DOF], lo, hi))
            return target_q, True
        if abs(vec[3]) > 1e-8 and np.linalg.norm(vec[:3]) < 1e-8:
            nxt = q[ROLL_DOF] + float(vec[3])
            lo, hi = w.model.jnt_range[ROLL_DOF]
            if nxt < lo - 1e-4 or nxt > hi + 1e-4:
                return target_q, False
            target_q[ROLL_DOF] = float(np.clip(nxt, lo, hi))
            return target_q, True
        snap = w.snapshot()
        goal = w.tcp() + vec[:3]
        solved, residual = w.ik_to(goal, lock_roll=True)
        penetrate = w.table_penetration()
        w.restore(snap)
        cmd = float(np.linalg.norm(vec[:3]))
        ok = residual <= max(IK_ABS, IK_FRAC * cmd) and penetrate <= PENETRATION
        if not ok:
            return q, False
        target_q = solved
        target_q[ROLL_DOF] = q[ROLL_DOF]
        # A close stalls against the object, so qpos is short of the close
        # target. Copying that qpos into ctrl lets go. Keep the squeeze.
        grip = float(snap["ctrl"][JAW_DOF])
        target_q[JAW_DOF] = grip if grip < 0.0 else q[JAW_DOF]
        return target_q, True

    def _settle(self, target_q: np.ndarray, on_frame=None, steps: int | None = None) -> None:
        w = self.world
        w.data.ctrl[:] = target_q
        n = SETTLE_STEPS if steps is None else steps
        for i in range(n):
            mujoco.mj_step(w.model, w.data)
            if on_frame is not None and i % 12 == 0:
                on_frame()
            if i > 15 and i % 5 == 0:
                if float(np.linalg.norm(w.data.qvel[:6])) < 0.04:
                    if float(np.max(np.abs(w.data.qpos[:6] - target_q))) < 0.03:
                        break

    def execute(self, action: Action | np.ndarray, on_frame=None) -> bool:
        """Apply one action from the current state. Invalid actions are not applied."""
        vec = action.array if isinstance(action, Action) else np.asarray(action, dtype=np.float64)
        w = self.world
        snap = w.snapshot()
        target_q, valid = self.plan(vec)
        w.restore(snap)
        if not valid:
            return False
        self._settle(target_q, on_frame=on_frame)
        return True

    def goto(self, goal: np.ndarray, on_frame=None) -> bool:
        """One IK solve to a TCP target. Keeps jaw and wrist roll. No object pose."""
        w = self.world
        goal = np.asarray(goal, dtype=np.float64)
        jaw = float(w.data.ctrl[JAW_DOF])
        roll = float(w.joints()[ROLL_DOF])

        def _one(target: np.ndarray) -> bool:
            snap = w.snapshot()
            solved, residual = w.ik_to(target, lock_roll=True)
            penetrate = w.table_penetration()
            w.restore(snap)
            if residual > 0.02 or penetrate > PENETRATION:
                return False
            solved[ROLL_DOF] = roll
            solved[JAW_DOF] = jaw if jaw < 0.0 else float(w.joints()[JAW_DOF])
            self._settle(solved, on_frame=on_frame, steps=SETTLE_STEPS)
            return float(np.linalg.norm(w.tcp() - target)) < 0.012

        if _one(goal):
            return True
        start = w.tcp()
        high_z = max(float(start[2]), float(goal[2]), 0.12)
        _one(np.array([start[0], start[1], high_z]))
        _one(np.array([goal[0], goal[1], high_z]))
        return _one(goal)

    def valid_mask(self) -> np.ndarray:
        """Which catalog actions the current arm can execute. No object pose."""
        snap = self.world.snapshot()
        out = np.zeros(len(CATALOG), dtype=bool)
        for i, action in enumerate(CATALOG):
            self.world.restore(snap)
            _, out[i] = self.plan(action.array)
        self.world.restore(snap)
        return out
