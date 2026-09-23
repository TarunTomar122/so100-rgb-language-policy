"""Pinhole projection through the calibrated fixed RGB camera."""

from __future__ import annotations

import mujoco
import numpy as np

from so100.sim import Tabletop


def _cam_rot(world: Tabletop) -> np.ndarray:
    quat = world.model.cam_quat[world._cam]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def project_xyz(world: Tabletop, xyz: np.ndarray, size: int) -> tuple[float, float] | None:
    """Project a world point into the fixed camera; used for checks and overlays."""
    origin, _ = world.camera()
    cam = _cam_rot(world).T @ (np.asarray(xyz, dtype=np.float64) - origin)
    if cam[2] >= -1e-4:
        return None
    fovy = float(np.deg2rad(world.model.cam_fovy[world._cam]))
    f = (size / 2) / np.tan(fovy / 2)
    cx = cy = (size - 1) / 2
    u = f * cam[0] / (-cam[2]) + cx
    v = f * (-cam[1]) / (-cam[2]) + cy
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return float(u), float(v)


def unproject_pixels(world: Tabletop, us: np.ndarray, vs: np.ndarray, z_world: float, size: int) -> np.ndarray:
    """Intersect RGB pixel rays with an assumed world-height plane; no depth image."""
    fovy = float(np.deg2rad(world.model.cam_fovy[world._cam]))
    f = (size / 2) / np.tan(fovy / 2)
    cx = cy = (size - 1) / 2
    x = (us.astype(np.float64) - cx) / f
    y = -(vs.astype(np.float64) - cy) / f
    rays = np.stack([x, y, -np.ones_like(x)], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    origin = world.model.cam_pos[world._cam].copy()
    direction = rays @ _cam_rot(world).T
    dz = direction[:, 2]
    t = np.full(len(us), np.nan)
    ok = np.abs(dz) > 1e-8
    t[ok] = (z_world - origin[2]) / dz[ok]
    t[t <= 0] = np.nan
    return origin + t[:, None] * direction
