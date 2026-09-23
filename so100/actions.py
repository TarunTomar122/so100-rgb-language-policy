"""The 30 local SO-100 actions. Numeric vectors, not class ids.

Stored and shown to the model as
[dx_m, dy_m, dz_m, droll_rad, gripper]
with gripper +1 open, -1 close, 0 when the action is not a gripper command.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Metres and radians, as specified. The network rescales internally.
XYZ_MM = (2, 5, 10, 30)
ROLL_DEG = (5, 15)


@dataclass(frozen=True)
class Action:
    name: str
    vec: tuple[float, float, float, float, float]

    @property
    def array(self) -> np.ndarray:
        return np.asarray(self.vec, dtype=np.float64)


def _build() -> list[Action]:
    out: list[Action] = []
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        for mm in XYZ_MM:
            for sign, tag in ((1, "+"), (-1, "-")):
                v = [0.0, 0.0, 0.0, 0.0, 0.0]
                v[idx] = sign * mm / 1000.0
                out.append(Action(f"{axis}{tag}{mm}mm", tuple(v)))
    for deg in ROLL_DEG:
        for sign, tag in ((1, "+"), (-1, "-")):
            v = [0.0, 0.0, 0.0, sign * np.deg2rad(deg), 0.0]
            out.append(Action(f"roll{tag}{deg}deg", tuple(v)))
    out.append(Action("open", (0.0, 0.0, 0.0, 0.0, 1.0)))
    out.append(Action("close", (0.0, 0.0, 0.0, 0.0, -1.0)))
    if len(out) != 30:
        raise RuntimeError(f"expected 30 actions, got {len(out)}")
    return out


CATALOG: list[Action] = _build()
ACTION_NAMES: list[str] = [a.name for a in CATALOG]
ACTION_MATRIX: np.ndarray = np.stack([a.array for a in CATALOG], axis=0)


def by_name(name: str) -> Action:
    for a in CATALOG:
        if a.name == name:
            return a
    raise KeyError(name)
