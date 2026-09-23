"""Frozen SigLIP 2 image and text encoder."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-base-patch16-256"


class Siglip2:
    def __init__(self, device: str) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, images: list[Image.Image], texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        vis, txt = self._forward(images, texts)
        return vis.float().cpu().numpy(), txt.float().cpu().numpy()

    @torch.no_grad()
    def embed_image(self, images: list[Image.Image]) -> np.ndarray:
        batch = self.processor(images=images, return_tensors="pt")
        pixels = batch["pixel_values"].to(self.device)
        return self.model.vision_model(pixel_values=pixels).last_hidden_state.float().cpu().numpy()

    @torch.no_grad()
    def embed_text(self, texts: list[str], max_length: int = 32, mask_pad: bool = True) -> np.ndarray:
        batch = self.processor(
            text=texts, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        mask = (batch["input_ids"] != 0).long() if mask_pad else None
        tokens = self.model.text_model(
            input_ids=batch["input_ids"], attention_mask=mask
        ).last_hidden_state.float()
        if mask_pad:
            tokens = tokens * mask[:, :, None]
        return tokens.cpu().numpy()

    def embed_text_mean(self, texts: list[str], max_length: int = 48) -> np.ndarray:
        tokens = self.embed_text(texts, max_length=max_length, mask_pad=True)
        lengths = (np.linalg.norm(tokens, axis=-1) > 0).sum(axis=1).clip(min=1)
        return (tokens.sum(axis=1) / lengths[:, None]).astype(np.float32)

    def _forward(self, images: list[Image.Image], texts: list[str]):
        batch = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            max_length=16,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        vis = self.model.vision_model(pixel_values=batch["pixel_values"]).last_hidden_state
        txt = self.model.text_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        ).last_hidden_state
        return vis, txt

    def embed_train(self, images: list[Image.Image], texts: list[str]):
        return self._forward(images, texts)

    def unfreeze_last(self, n: int) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        layers = self.model.vision_model.encoder.layers
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in self.model.vision_model.post_layernorm.parameters():
            p.requires_grad_(True)
        self.model.vision_model.train()
        n_on = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"unfroze last {n} vision blocks params {n_on}", flush=True)
