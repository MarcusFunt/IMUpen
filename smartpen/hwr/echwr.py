"""Training-only sensor/text contrastive modules for M6-D."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def generate_edit_distance_one_negatives(
    text: str,
    charset: str,
    rng: np.random.Generator,
    *,
    sets: int = 1,
) -> list[str]:
    """Generate deletion, insertion, and substitution negatives per set."""
    if sets < 1:
        raise ValueError("sets must be positive")
    if len(set(charset)) != len(charset) or len(charset) < 2:
        raise ValueError("charset must contain at least two unique characters")
    if not text:
        raise ValueError("hard-negative generation requires a non-empty transcript")
    invalid = set(text) - set(charset)
    if invalid:
        raise ValueError(f"transcript contains character outside charset: {sorted(invalid)[0]!r}")

    negatives: list[str] = []
    choices = np.asarray(list(charset), dtype=object)
    for _ in range(sets):
        delete_at = int(rng.integers(len(text)))
        negatives.append(text[:delete_at] + text[delete_at + 1 :])

        insert_at = int(rng.integers(len(text) + 1))
        inserted = str(rng.choice(choices))
        negatives.append(text[:insert_at] + inserted + text[insert_at:])

        substitute_at = int(rng.integers(len(text)))
        original = text[substitute_at]
        replacements = [character for character in charset if character != original]
        replacement = str(rng.choice(replacements))
        negatives.append(
            text[:substitute_at] + replacement + text[substitute_at + 1 :]
        )
    return negatives


class ECHWRAuxiliaryBranch(nn.Module):
    """Project temporal IMU and character-level text into a shared space.

    This module exists only during contrastive training. The recognizer's
    encoder and CTC classifier remain the complete inference model.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        charset: str,
        embedding_dim: int = 128,
        attention_heads: int = 4,
        text_layers: int = 2,
        max_text_characters: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or embedding_dim < 4:
            raise ValueError("input and embedding dimensions must be positive")
        if attention_heads < 1 or embedding_dim % attention_heads:
            raise ValueError("embedding dimension must be divisible by attention heads")
        if text_layers < 1 or max_text_characters < 1:
            raise ValueError("text layers and maximum text length must be positive")
        if len(set(charset)) != len(charset) or len(charset) < 2:
            raise ValueError("charset must contain at least two unique characters")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.charset = charset
        self.char_to_id = {character: index + 1 for index, character in enumerate(charset)}
        self.cls_id = len(charset) + 1
        self.embedding_dim = embedding_dim
        self.max_text_characters = max_text_characters

        self.sensor_projection = nn.Linear(input_dim, embedding_dim)
        self.sensor_attention = nn.MultiheadAttention(
            embedding_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.sensor_norm = nn.LayerNorm(embedding_dim)

        self.char_embedding = nn.Embedding(
            len(charset) + 2, embedding_dim, padding_idx=0
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.position_embedding = nn.Embedding(max_text_characters + 2, embedding_dim)
        text_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(
            text_layer, num_layers=text_layers, enable_nested_tensor=False
        )
        self.text_norm = nn.LayerNorm(embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        nn.init.normal_(self.cls_token, std=0.02)

    def encode_sensor(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.sensor_projection.in_features:
            raise ValueError("sensor features have an incompatible shape")
        if lengths.ndim != 1 or lengths.numel() != features.shape[0]:
            raise ValueError("sensor lengths must contain one value per sequence")
        if torch.any(lengths < 1) or torch.any(lengths > features.shape[1]):
            raise ValueError("sensor lengths must fit the padded temporal dimension")

        lengths = lengths.to(device=features.device, dtype=torch.long)
        projected = self.sensor_projection(features)
        time_positions = torch.arange(
            features.shape[1], device=features.device, dtype=torch.float32
        )[:, None]
        even_dimensions = torch.arange(
            0, self.embedding_dim, 2, device=features.device, dtype=torch.float32
        )
        frequencies = torch.exp(
            -math.log(10_000.0) * even_dimensions / max(self.embedding_dim, 1)
        )
        positional = torch.zeros(
            (features.shape[1], self.embedding_dim),
            device=features.device,
            dtype=torch.float32,
        )
        positional[:, 0::2] = torch.sin(time_positions * frequencies)
        if self.embedding_dim > 1:
            positional[:, 1::2] = torch.cos(
                time_positions * frequencies[: positional[:, 1::2].shape[1]]
            )
        projected = projected + positional.to(dtype=projected.dtype)[None, :, :]
        valid = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
        masked = projected.masked_fill(~valid[:, :, None], 0.0)
        query = masked.sum(dim=1, keepdim=True) / lengths[:, None, None].to(projected.dtype)
        pooled, _ = self.sensor_attention(
            query,
            projected,
            projected,
            key_padding_mask=~valid,
            need_weights=False,
        )
        pooled = self.sensor_norm(pooled[:, 0])
        return F.normalize(pooled.float(), dim=-1)

    def encode_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        if not texts:
            raise ValueError("text batch cannot be empty")
        tokenized: list[list[int]] = []
        for text in texts:
            invalid = set(text) - self.char_to_id.keys()
            if invalid:
                raise ValueError(
                    f"transcript contains character outside charset: {sorted(invalid)[0]!r}"
                )
            if len(text) > self.max_text_characters:
                raise ValueError("transcript exceeds configured text encoder length")
            tokenized.append([self.cls_id, *(self.char_to_id[char] for char in text)])

        width = max(map(len, tokenized))
        ids = torch.zeros((len(texts), width), dtype=torch.long, device=device)
        for index, tokens in enumerate(tokenized):
            ids[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        positions = torch.arange(width, device=device)[None, :]
        values = self.char_embedding(ids) + self.position_embedding(positions)
        padding_mask = ids.eq(0)
        encoded = self.text_encoder(values, src_key_padding_mask=padding_mask)
        return F.normalize(self.text_norm(encoded[:, 0]).float(), dim=-1)

    def scale(self) -> torch.Tensor:
        """Return the bounded log-scale parameter used by contrastive logits."""
        return self.logit_scale.float().clamp(max=math.log(100.0))


def in_batch_contrastive_loss(
    sensor_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    texts: list[str],
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric sensor/text InfoNCE after removing duplicate transcript rows."""
    if sensor_embeddings.ndim != 2 or sensor_embeddings.shape != text_embeddings.shape:
        raise ValueError("sensor and text embeddings must have matching (batch, dim) shapes")
    if len(texts) != sensor_embeddings.shape[0]:
        raise ValueError("transcripts must match the embedding batch size")

    first_by_text: dict[str, int] = {}
    for index, text in enumerate(texts):
        first_by_text.setdefault(text, index)
    if len(first_by_text) < 2:
        return (sensor_embeddings.sum() + text_embeddings.sum()) * 0.0

    indices = torch.tensor(list(first_by_text.values()), device=sensor_embeddings.device)
    sensor = F.normalize(sensor_embeddings[indices].float(), dim=-1)
    text = F.normalize(text_embeddings[indices].float(), dim=-1)
    logits = logit_scale.float().clamp(max=math.log(100.0)).exp() * sensor @ text.T
    targets = torch.arange(len(indices), device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)
    )


def error_based_contrastive_loss(
    sensor_embeddings: torch.Tensor,
    positive_text_embeddings: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Classify each transcript embedding above its edit-distance-one negatives."""
    if sensor_embeddings.ndim != 2 or positive_text_embeddings.shape != sensor_embeddings.shape:
        raise ValueError("positive sensor/text embeddings must have matching (batch, dim) shapes")
    if (
        negative_text_embeddings.ndim != 3
        or negative_text_embeddings.shape[0] != sensor_embeddings.shape[0]
        or negative_text_embeddings.shape[2] != sensor_embeddings.shape[1]
        or negative_text_embeddings.shape[1] < 1
    ):
        raise ValueError("negative text embeddings must have shape (batch, negatives, dim)")

    sensor = F.normalize(sensor_embeddings.float(), dim=-1)
    candidates = torch.cat(
        (positive_text_embeddings[:, None, :], negative_text_embeddings), dim=1
    )
    candidates = F.normalize(candidates.float(), dim=-1)
    logits = logit_scale.float().clamp(max=math.log(100.0)).exp() * (
        sensor[:, None, :] * candidates
    ).sum(dim=-1)
    targets = torch.zeros(sensor.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)
