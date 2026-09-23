"""MuJoCo SO-100 tabletop. State copy, camera, object poses.

The executor is the only thing that applies an action. This module does not
choose actions and does not know the task.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "tabletop.xml"

TABLE_TOP = 0.01
TABLE_X = (-0.16, 0.16)
TABLE_Y = (-0.38, -0.16)

MOVABLES = ("red_cube", "blue_cube", "green_cylinder", "yellow_block")
HALF_Z = {
    "red_cube": 0.015,
    "blue_cube": 0.015,
    "green_cylinder": 0.02,
    "yellow_block": 0.018,
}
MOCAPS = ("bowl", "marker", "platform")
FINGER_GEOMS = (
    "fixed_jaw_pad_1",
    "fixed_jaw_pad_2",
    "fixed_jaw_pad_3",
    "fixed_jaw_pad_4",
    "moving_jaw_pad_1",
    "moving_jaw_pad_2",
    "moving_jaw_pad_3",
    "moving_jaw_pad_4",
)

# Side of the table. Over-the-shoulder hid the target (~20% of frames).
# Image-left is world -Y here; left_of still uses world -X in the judge.
DEFAULT_EYE = np.array([0.48, -0.18, 0.22])
DEFAULT_TARGET = np.array([0.00, -0.28, 0.03])
WATCH_EYE = DEFAULT_EYE.copy()
WATCH_TARGET = DEFAULT_TARGET.copy()

ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
# Translation IK uses the first four joints and holds wrist roll.
POS_DOFS = (0, 1, 2, 3)
ROLL_DOF = 4
JAW_DOF = 5


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return quat


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Standard look-at. Use for the watch camera; not for language-left = image-left."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    z = -(np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64))
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-6:
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
        n = np.linalg.norm(x)
    x = x / n
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1)
    return _mat_to_quat(mat)


def look_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera +X aligned with world +X. Camera looks toward target."""
    forward = target - eye
    z = -forward
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.array([1.0, 0.0, 0.0])
    x = x - z * float(np.dot(x, z))
    n = np.linalg.norm(x)
    if n < 1e-6:
        x = np.array([0.0, 1.0, 0.0])
        x = x - z * float(np.dot(x, z))
        n = np.linalg.norm(x)
    x = x / n
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1)
    return _mat_to_quat(mat)


class Tabletop:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self._cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "main")
        self._watch = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "watch")
        self._tcp = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._table = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.obj_qadr = {}
        self.obj_geom = {}
        for name in MOVABLES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_j")
            self.obj_qadr[name] = int(self.model.jnt_qposadr[jid])
            self.obj_geom[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.mocap_id = {}
        for name in MOCAPS:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.mocap_id[name] = int(self.model.body_mocapid[bid])
        self.finger_geoms = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FINGER_GEOMS
        ]
        self.arm_collision = []
        for i in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[i])
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if bname in ("Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw"):
                if int(self.model.geom_contype[i]) != 0:
                    self.arm_collision.append(i)
        self.renderer: mujoco.Renderer | None = None
        self.home()

    def home(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.ctrl[:] = self.data.qpos[:6]
        self.park_scene()
        self.set_camera(DEFAULT_EYE, DEFAULT_TARGET)
        if self._watch >= 0:
            self.model.cam_pos[self._watch] = WATCH_EYE
            self.model.cam_quat[self._watch] = look_at(WATCH_EYE, WATCH_TARGET)
        mujoco.mj_forward(self.model, self.data)

    def park_scene(self) -> None:
        for i, name in enumerate(MOVABLES):
            self.set_object(name, np.array([2.0 + 0.3 * i, 2.0, 0.2]))
        for name in MOCAPS:
            self.set_mocap(name, np.array([2.0, 2.5, -0.5]))

    def snapshot(self) -> dict:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
            "mocap_pos": self.data.mocap_pos.copy(),
            "mocap_quat": self.data.mocap_quat.copy(),
            "cam_pos": self.model.cam_pos[self._cam].copy(),
            "cam_quat": self.model.cam_quat[self._cam].copy(),
        }

    def restore(self, snap: dict) -> None:
        self.data.qpos[:] = snap["qpos"]
        self.data.qvel[:] = snap["qvel"]
        self.data.ctrl[:] = snap["ctrl"]
        self.data.time = snap["time"]
        self.data.mocap_pos[:] = snap["mocap_pos"]
        self.data.mocap_quat[:] = snap["mocap_quat"]
        self.model.cam_pos[self._cam] = snap["cam_pos"]
        self.model.cam_quat[self._cam] = snap["cam_quat"]
        self.data.qacc_warmstart[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def tcp(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp].copy()

    def joints(self) -> np.ndarray:
        return self.data.qpos[:6].copy()

    def jaw(self) -> float:
        return float(self.data.qpos[JAW_DOF])

    def set_object(self, name: str, xyz: np.ndarray, quat: np.ndarray | None = None) -> None:
        adr = self.obj_qadr[name]
        self.data.qpos[adr : adr + 3] = xyz
        if quat is None:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            q = np.asarray(quat, dtype=np.float64)
            n = float(np.linalg.norm(q))
            q = q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[adr + 3 : adr + 7] = q
        self.data.qvel[:] = 0.0

    def object_pose(self, name: str) -> np.ndarray:
        adr = self.obj_qadr[name]
        return self.data.qpos[adr : adr + 7].copy()

    def object_xyz(self, name: str) -> np.ndarray:
        return self.object_pose(name)[:3].copy()

    def set_mocap(self, name: str, xyz: np.ndarray) -> None:
        mid = self.mocap_id[name]
        self.data.mocap_pos[mid] = xyz
        self.data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0])

    def mocap_xyz(self, name: str) -> np.ndarray:
        return self.data.mocap_pos[self.mocap_id[name]].copy()

    def set_camera(self, eye: np.ndarray, target: np.ndarray) -> None:
        self.model.cam_pos[self._cam] = np.asarray(eye, dtype=np.float64)
        self.model.cam_quat[self._cam] = look_at(np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64))

    def camera(self) -> tuple[np.ndarray, np.ndarray]:
        return self.model.cam_pos[self._cam].copy(), self.model.cam_quat[self._cam].copy()

    def finger_gap(self) -> float:
        a = self.data.geom_xpos[self.finger_geoms[0]]
        b = self.data.geom_xpos[self.finger_geoms[4]]
        return float(np.linalg.norm(a - b))

    def finger_contact(self, name: str) -> bool:
        gid = self.obj_geom[name]
        fingers = set(self.finger_geoms)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if gid in pair and pair & fingers:
                return True
        return False

    def both_jaws_contact(self, name: str) -> bool:
        gid = self.obj_geom[name]
        fixed = set(self.finger_geoms[:4])
        moving = set(self.finger_geoms[4:])
        hit_f = hit_m = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if gid not in pair:
                continue
            other = (pair - {gid}).pop()
            hit_f = hit_f or other in fixed
            hit_m = hit_m or other in moving
        return hit_f and hit_m

    def table_penetration(self) -> float:
        """Positive means the arm is inside the table, in metres."""
        worst = 0.0
        fromto = np.zeros(6)
        for g in self.arm_collision:
            dist = mujoco.mj_geomDistance(self.model, self.data, g, self._table, 0.05, fromto)
            if dist < 0:
                worst = max(worst, float(-dist))
        return worst

    def render(self, size: int = 256, camera: str = "main") -> np.ndarray:
        if self.renderer is None or self.renderer.height != size:
            self.renderer = mujoco.Renderer(self.model, size, size)
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def render_depth(self, size: int = 256, camera: str = "main") -> np.ndarray:
        """Meters along the camera ray. Same pose as render()."""
        if self.renderer is None or self.renderer.height != size:
            self.renderer = mujoco.Renderer(self.model, size, size)
        self.renderer.update_scene(self.data, camera=camera)
        self.renderer.enable_depth_rendering()
        try:
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()

    def hold(self) -> None:
        self.data.ctrl[:] = self.data.qpos[:6]
        self.data.qvel[:] = 0.0

    def step(self, n: int) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def ik_to(self, target: np.ndarray, lock_roll: bool = True, iters: int = 25) -> tuple[np.ndarray, float]:
        """Move arm qpos toward a world TCP target. Returns (q6, residual metres).

        Does not simulate. Caller decides whether the residual is acceptable.
        """
        dofs = POS_DOFS if lock_roll else POS_DOFS + (ROLL_DOF,)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        lam = 1e-3
        eye = np.eye(3)
        for _ in range(iters):
            mujoco.mj_forward(self.model, self.data)
            err = target - self.data.site_xpos[self._tcp]
            if float(np.linalg.norm(err)) < 1e-3:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._tcp)
            J = jacp[:, list(dofs)]
            dq = J.T @ np.linalg.solve(J @ J.T + lam * eye, err)
            step = np.clip(dq, -0.2, 0.2)
            for k, j in enumerate(dofs):
                lo, hi = self.model.jnt_range[j]
                self.data.qpos[j] = float(np.clip(self.data.qpos[j] + step[k], lo, hi))
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        residual = float(np.linalg.norm(target - self.data.site_xpos[self._tcp]))
        return self.data.qpos[:6].copy(), residual
