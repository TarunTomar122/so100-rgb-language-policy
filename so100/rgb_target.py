"""RGB object proposals and the historical learned target head."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


def candidate_masks(image: np.ndarray, count: int = 4,
                    background: np.ndarray | None = None) -> list[np.ndarray]:
    """Split saturated tabletop objects by color without object names or poses."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    visible = (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 45)
    if background is not None:
        current = cv2.GaussianBlur(image, (5, 5), 0).astype(np.int16)
        empty = cv2.GaussianBlur(background, (5, 5), 0).astype(np.int16)
        changed = (np.max(np.abs(current - empty), axis=2) > 28).astype(np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # ponytail: a fixed camera needs an empty-table frame at the same pose/light.
        if count * 80 <= int(changed.sum()) < image.shape[0] * image.shape[1] // 4:
            visible &= changed.astype(bool)
    ys, xs = np.nonzero(visible)
    if len(xs) < count * 80:
        return []
    rgb = image[visible].astype(np.float32)
    chroma = rgb / (rgb.sum(axis=1, keepdims=True) + 1)
    cv2.setRNGSeed(7)
    _, clusters, _ = cv2.kmeans(
        chroma, count, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3),
        4, cv2.KMEANS_PP_CENTERS,
    )
    out = []
    for k in range(count):
        raw = np.zeros(visible.shape, np.uint8)
        chosen = clusters[:, 0] == k
        raw[ys[chosen], xs[chosen]] = 1
        n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        if n < 2:
            continue
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[component, cv2.CC_STAT_AREA] >= 80:
            out.append(labels == component)
    return sorted(out, key=lambda mask: float(np.nonzero(mask)[1].mean()))


def pool_patches(vis: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    patches = vis[-256:].reshape(16, 16, -1)
    pooled = []
    for mask in masks:
        weights = cv2.resize(mask.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)
        weights /= max(float(weights.sum()), 1e-6)
        pooled.append((patches * weights[:, :, None]).sum(axis=(0, 1)))
    return np.asarray(pooled, dtype=np.float32)


class TargetHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = nn.Linear(768, 128)
        self.text = nn.Linear(768, 128)
        self.log_temp = nn.Parameter(torch.tensor(2.0))

    def forward(self, candidates: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        v = nn.functional.normalize(self.vision(candidates), dim=-1)
        t = nn.functional.normalize(self.text(text), dim=-1)
        return torch.einsum("...cd,...d->...c", v, t) * self.log_temp.exp()
