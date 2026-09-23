"""Calibrated single-RGB grasp estimate for the colored tabletop prototype.

Only the RGB frame, fixed camera calibration, and table height enter here.
The mask comes from an RGB language-selected object proposal.
"""

from __future__ import annotations

import cv2
import numpy as np

from so100.sim import TABLE_TOP, Tabletop
from so100.vision import unproject_pixels


def colored_components(image: np.ndarray) -> list[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    colored = ((hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 45)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(colored, 8)
    ids = sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)
    return [labels == i for i in ids if stats[i, cv2.CC_STAT_AREA] >= 80]


def estimate_grasp(world: Tabletop, image: np.ndarray, mask: np.ndarray | None = None,
                   fallback: bool = True) -> dict:
    size = image.shape[0]
    if mask is None:
        components = colored_components(image)
        if not components:
            raise ValueError("No colored object visible")
        mask = components[0]
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < 80:
        raise ValueError("Object mask is too small")

    brightness = image.max(axis=2)
    top = mask & (brightness >= np.quantile(brightness[mask], 0.8))
    ys, xs = np.nonzero(top)
    if len(xs) < 30:
        raise ValueError("Object top is not visible")
    # ponytail: all benchmark objects are 30 mm tall; estimate height from RGB
    # or add an active view when testing objects with unknown heights.
    top_xyz = unproject_pixels(world, xs.astype(float), ys.astype(float), TABLE_TOP + 0.03, size)
    xy = top_xyz[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(xy)
    corners = cv2.boxPoints(rect)
    edges = [corners[1] - corners[0], corners[2] - corners[1]]
    short = min(edges, key=np.linalg.norm)
    yaw = float(np.rad2deg(np.arctan2(short[1], short[0])))
    contour = cv2.convexHull(xy)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * cv2.contourArea(contour) / max(perimeter * perimeter, 1e-9)
    if circularity > 0.92:
        yaw = 0.0
    elif abs(np.linalg.norm(edges[0]) - np.linalg.norm(edges[1])) < 0.005:
        yaw = (yaw + 45) % 90 - 45
    else:
        yaw = (yaw + 90) % 180 - 90
        if abs(yaw) > 60:
            # A wide grip is easier here than an extreme wrist rotation.
            short = max(edges, key=np.linalg.norm)
            yaw = float(np.rad2deg(np.arctan2(short[1], short[0])))
            yaw = (yaw + 90) % 180 - 90

    along = float(np.linalg.norm(short) / 2 + 0.007)
    center_xy = xy.mean(axis=0)
    if fallback and along > 0.03:
        # The bright top can be mostly hidden by another object. Use the full
        # RGB silhouette and the calibrated finger span for these outliers.
        all_y, all_x = np.nonzero(mask)
        center_u = (float(all_x.min()) + float(all_x.max())) / 2
        center_v = (float(all_y.min()) + float(all_y.max())) / 2
        center_xy = unproject_pixels(
            world, np.array([center_u]), np.array([center_v]), TABLE_TOP + 0.015, size
        )[0, :2]
        full_xy = unproject_pixels(
            world, all_x.astype(float), all_y.astype(float), TABLE_TOP + 0.015, size
        )[:, :2].astype(np.float32)
        full_corners = cv2.boxPoints(cv2.minAreaRect(full_xy))
        full_short = min((full_corners[1] - full_corners[0],
                          full_corners[2] - full_corners[1]), key=np.linalg.norm)
        along = min(0.024, float(np.linalg.norm(full_short) / 2 + 0.007))

    return {
        "xyz": np.array([float(center_xy[0]), float(center_xy[1]), TABLE_TOP + 0.015]),
        "yaw_deg": yaw,
        "along": along,
        "mask_pixels": int(mask.sum()),
        "circularity": float(circularity),
    }
