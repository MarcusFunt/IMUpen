"""TS2Vec-style hierarchical temporal/instance contrast for IMU sequences."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TS2VecProjectionHead(nn.Module):
    """Projection head used only while fitting the self-supervised objective."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def context_crop_bounds(
    sequence_length: int,
    rng: np.random.Generator,
    *,
    downsample_factor: int = 4,
    temporal_unit: int = 0,
) -> tuple[int, int, int, int, int]:
    """Sample two aligned context crops and return ``(a0,b0,a1,b1,overlap)``.

    Crop boundaries are aligned to the recognizer's total stride so the common
    segment has the same latent timestep coordinates in both encoded views.
    """
    if sequence_length < 1 or downsample_factor < 1 or temporal_unit < 0:
        raise ValueError("sequence length/factor must be positive and temporal_unit non-negative")
    latent_length = (sequence_length + downsample_factor - 1) // downsample_factor
    if latent_length < 2:
        raise ValueError("TS2Vec requires at least two encoded timesteps per sequence")
    minimum_overlap = 2 ** (temporal_unit + 1)
    minimum_overlap = min(max(minimum_overlap, 2), latent_length)
    overlap = int(rng.integers(minimum_overlap, latent_length + 1))
    left = int(rng.integers(0, latent_length - overlap + 1))
    right = left + overlap
    left_context = int(rng.integers(0, left + 1))
    right_context = int(rng.integers(right, latent_length + 1))

    shared_start = left * downsample_factor
    shared_end = min(sequence_length, right * downsample_factor)
    view1_start = (left - left_context) * downsample_factor
    view1_end = shared_end
    view2_start = shared_start
    view2_end = min(sequence_length, right_context * downsample_factor)
    return view1_start, view1_end, view2_start, view2_end, overlap


def mask_timestamps(
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float = 0.15,
) -> np.ndarray:
    """Mask random normalized input timestamps with the training mean (zero)."""
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
        raise ValueError("timestamp masking expects non-empty (timesteps, 6) values")
    if not 0.0 <= probability < 1.0:
        raise ValueError("mask probability must be in [0, 1)")
    result = np.asarray(values, dtype=np.float32).copy()
    count = min(len(result), int(round(len(result) * probability)))
    if count:
        selected = rng.choice(len(result), size=count, replace=False)
        result[selected] = 0.0
    return result


def hierarchical_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    lengths: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Average symmetric instance and temporal InfoNCE losses across scales."""
    if first.ndim != 3 or second.shape != first.shape:
        raise ValueError("both views must have matching (batch, time, features) shapes")
    if lengths.ndim != 1 or lengths.numel() != first.shape[0]:
        raise ValueError("lengths must contain one value per batch item")
    if torch.any(lengths < 1) or torch.any(lengths > first.shape[1]):
        raise ValueError("contrastive lengths must fit within the padded timestep dimension")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    view1, view2 = first, second
    level_lengths = lengths.to(device=first.device, dtype=torch.long)
    scale_losses: list[torch.Tensor] = []
    while True:
        component_losses = [
            value
            for value in (
                _instance_contrastive_loss(view1, view2, level_lengths, temperature),
                _temporal_contrastive_loss(view1, view2, level_lengths, temperature),
            )
            if value is not None
        ]
        if component_losses:
            scale_losses.append(torch.stack(component_losses).mean())
        if view1.shape[1] <= 1:
            break
        view1, pooled_lengths = _masked_max_pool(view1, level_lengths)
        view2, _ = _masked_max_pool(view2, level_lengths)
        level_lengths = pooled_lengths
    if not scale_losses:
        raise ValueError("contrastive batch needs a negative pair at one or more scales")
    return torch.stack(scale_losses).mean()


def _instance_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    lengths: torch.Tensor,
    temperature: float,
) -> torch.Tensor | None:
    losses = []
    for timestep in range(first.shape[1]):
        valid = torch.nonzero(lengths > timestep, as_tuple=False).flatten()
        if valid.numel() < 2:
            continue
        left = F.normalize(first[valid, timestep], dim=-1)
        right = F.normalize(second[valid, timestep], dim=-1)
        logits = left @ right.transpose(0, 1) / temperature
        labels = torch.arange(valid.numel(), device=first.device)
        losses.append(
            0.5
            * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.transpose(0, 1), labels)
            )
        )
    return torch.stack(losses).mean() if losses else None


def _temporal_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    lengths: torch.Tensor,
    temperature: float,
) -> torch.Tensor | None:
    losses = []
    for sample_index, raw_length in enumerate(lengths.tolist()):
        if raw_length < 2:
            continue
        left = F.normalize(first[sample_index, :raw_length], dim=-1)
        right = F.normalize(second[sample_index, :raw_length], dim=-1)
        logits = left @ right.transpose(0, 1) / temperature
        labels = torch.arange(raw_length, device=first.device)
        losses.append(
            0.5
            * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.transpose(0, 1), labels)
            )
        )
    return torch.stack(losses).mean() if losses else None


def _masked_max_pool(
    values: torch.Tensor, lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    time = values.shape[1]
    valid_steps = torch.arange(time, device=values.device)[None, :] < lengths[:, None]
    masked = values.masked_fill(~valid_steps[:, :, None], -torch.inf)
    pooled = F.max_pool1d(masked.transpose(1, 2), kernel_size=2, stride=2, ceil_mode=True)
    pooled = pooled.transpose(1, 2)
    pooled_lengths = (lengths + 1) // 2
    pooled_valid = torch.arange(pooled.shape[1], device=values.device)[None, :] < pooled_lengths[:, None]
    pooled = pooled.masked_fill(~pooled_valid[:, :, None], 0.0)
    return pooled, pooled_lengths
