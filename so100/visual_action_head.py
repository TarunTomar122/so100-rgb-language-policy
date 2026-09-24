"""Image-conditioned, one-step skill prediction for the RGB tabletop experiment."""

from __future__ import annotations

import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
from torch import nn

from so100.action_head import ActionHead, ActionText, SKILLS, state_vector
from so100.encode import Siglip2
from so100.rgb_target import TargetHead, candidate_masks, pool_patches
from so100.sim import Tabletop
from so100.vision import project_xyz

SIZE = 256
FEATURES = 768
STATE_DIM = 6 + 3 + 1 + 2 + 1 + 4 + len(SKILLS) * 2 + 1
PRIOR_CHECKPOINT = Path(__file__).resolve().parents[1] / "data" / "action-head" / "action-v10.pt"


class LanguagePrior:
    """Reuse the earlier trained skill semantics; RGB learns when to correct it."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.text = ActionText(device)
        self.head = ActionHead().to(device).eval()
        self.head.load_state_dict(torch.load(PRIOR_CHECKPOINT, map_location=device,
                                             weights_only=True)["state"])
        self.tokens: torch.Tensor | None = None

    def prepare(self, instruction: str) -> None:
        self.tokens = torch.tensor(self.text.embed([instruction]), device=self.device)

    @torch.no_grad()
    def logits(self, world: Tabletop, history: list[str]) -> np.ndarray:
        assert self.tokens is not None
        state = torch.tensor(state_vector(world, history), device=self.device)[None]
        return self.head(self.tokens, state).float().cpu().numpy()[0]


def observe(world: Tabletop, eyes: Siglip2, instruction: str,
            background: np.ndarray, history: list[str], last_ok: bool,
            text_feature: np.ndarray | None = None,
            previous_uv: np.ndarray | None = None,
            target_identity: np.ndarray | None = None,
            target_chroma: np.ndarray | None = None,
            target_head: TargetHead | None = None) -> tuple:
    """Only RGB, the instruction, and robot sensors enter the learned decision."""
    image = world.render(SIZE)
    target_image = world.render(512) if target_head is not None else image
    if background.shape != target_image.shape:
        background = cv2.resize(background, target_image.shape[:2][::-1], interpolation=cv2.INTER_AREA)
    masks = candidate_masks(target_image, background=background)
    patches = eyes.embed_image([Image.fromarray(image)])[0]
    scene = patches.mean(axis=0)
    target = np.zeros(FEATURES, np.float32)
    uv = np.zeros(2, np.float32)
    area = 0.0
    selected = None
    selected_feature = None
    if text_feature is None:
        text_feature = eyes.embed_text_mean([instruction])[0]
    if masks:
        if target_head is not None:
            target_patches = eyes.embed_image([Image.fromarray(target_image)])[0]
            features = pool_patches(target_patches, masks)
            with torch.no_grad():
                scores = target_head(
                    torch.from_numpy(features).to(eyes.device)[None],
                    torch.from_numpy(text_feature).to(eyes.device)[None],
                )[0].float().cpu().numpy()
        else:
            crops = []
            for mask in masks:
                ys, xs = np.nonzero(mask)
                crops.append(Image.fromarray(image[max(0, ys.min()-12):min(SIZE, ys.max()+13),
                                                   max(0, xs.min()-12):min(SIZE, xs.max()+13)]))
            batch = eyes.processor(images=crops, text=[instruction], padding="max_length",
                                   max_length=48, return_tensors="pt")
            with torch.no_grad():
                output = eyes.model(**{key: value.to(eyes.device) for key, value in batch.items()})
            features = output.image_embeds.float().cpu().numpy()
            scores = output.logits_per_image[:, 0].float().cpu().numpy()
        features /= np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-6)
        if target_identity is not None:
            scores = features @ target_identity
            if target_chroma is not None:
                colors = np.array([target_image[mask].mean(axis=0) for mask in masks])
                chromas = colors / np.maximum(colors.sum(axis=1, keepdims=True), 1)
                distances = np.linalg.norm(chromas - target_chroma, axis=1)
                matching = distances <= 0.08
                if matching.any():
                    scores = np.where(matching, scores, -np.inf)
        selected = int(scores.argmax())
        selected_feature = features[selected].copy()
        mask = masks[selected]
        action_mask = (cv2.resize(mask.astype(np.uint8), (SIZE, SIZE),
                                  interpolation=cv2.INTER_AREA) > 0.5
                       if target_head is not None else mask)
        target = pool_patches(patches, [action_mask])[0]
        ys, xs = np.nonzero(action_mask)
        uv = np.array([xs.mean() / SIZE, ys.mean() / SIZE], np.float32)
        area = float(mask.mean())
    tip_uv = project_xyz(world, world.tcp(), SIZE)
    tip_uv = np.asarray(tip_uv, np.float32) / SIZE if tip_uv is not None else np.zeros(2, np.float32)
    target_to_tip = (uv - tip_uv) * 5 if selected is not None else np.zeros(2, np.float32)
    target_motion = (uv - previous_uv) * 10 if selected is not None and previous_uv is not None else np.zeros(2, np.float32)
    joints = world.joints().astype(np.float32)
    tcp = world.tcp().astype(np.float32)
    counts = np.array([min(history.count(skill), 3) / 3 for skill in SKILLS], np.float32)
    recent = np.zeros(len(SKILLS), np.float32)
    if history:
        recent[SKILLS.index(history[-1])] = 1
    state = np.concatenate((joints / 2, tcp * 4, [world.finger_gap() * 10],
                            uv, [area * 10], target_to_tip, target_motion,
                            counts, recent, [float(last_ok)])).astype(np.float32)
    assert state.shape == (STATE_DIM,)
    return (scene.astype(np.float32), target.astype(np.float32), text_feature.astype(np.float32),
            state, selected, uv, None if selected is None else masks[selected], selected_feature)


class VisualActionHead(nn.Module):
    """A visual correction to the earlier frozen language-and-state action head."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURES * 3 + STATE_DIM + len(SKILLS), 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, len(SKILLS)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, scene: torch.Tensor, target: torch.Tensor,
                text: torch.Tensor, state: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        return prior * 0.5 + self.net(torch.cat((scene, target, text, state, prior * 0.1), dim=-1))


class RecoveryActionHead(nn.Module):
    """Small learned retry policy using RGB-derived state and the visual head's logits."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATE_DIM + len(SKILLS), 64), nn.GELU(),
                                 nn.Linear(64, len(SKILLS)))

    def forward(self, state: torch.Tensor, visual_logits: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, visual_logits), dim=-1))


class DoneCalibrator(nn.Module):
    """Learn when the instruction's observed outcome permits stopping."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURES * 2 + STATE_DIM + len(SKILLS), 128),
                                 nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, target: torch.Tensor, text: torch.Tensor,
                state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        correction = self.net(torch.cat((target, text, state, logits), dim=-1))
        result = logits.clone()
        result[..., SKILLS.index("done")] += correction[..., 0]
        return result


if __name__ == "__main__":
    head = VisualActionHead()
    logits = head(*(torch.zeros(2, n) for n in (FEATURES, FEATURES, FEATURES, STATE_DIM, len(SKILLS))))
    assert logits.shape == (2, len(SKILLS))
    assert RecoveryActionHead()(torch.zeros(2, STATE_DIM), logits).shape == logits.shape
    assert DoneCalibrator()(torch.zeros(2, FEATURES), torch.zeros(2, FEATURES),
                            torch.zeros(2, STATE_DIM), logits).shape == logits.shape
    print("visual action head ok")
