"""Reproducible, training-only measured-sensor augmentations for M6-B."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from smartpen.hwr.data import ONHW_CHARSET, SequenceRecord, encode_text


ONLINE_AUGMENTATIONS = (
    "temporal_stretch",
    "gaussian_noise",
    "sensor_bias_gain",
    "rotation_3d",
)
SINGLE_AUGMENTATIONS = (*ONLINE_AUGMENTATIONS, "sequence_concat")
AUGMENTATION_NAMES = ("none", *SINGLE_AUGMENTATIONS, "combined")


def augmentation_plan(name: str) -> tuple[tuple[str, ...], bool]:
    """Return per-record transforms and whether to add concatenated records."""
    if name == "none":
        return (), False
    if name in ONLINE_AUGMENTATIONS:
        return (name,), False
    if name == "sequence_concat":
        return (), True
    if name == "combined":
        return ONLINE_AUGMENTATIONS, True
    raise ValueError(f"unknown augmentation {name!r}; choose from {AUGMENTATION_NAMES}")


def apply_augmentation(
    record: SequenceRecord,
    name: str,
    rng: np.random.Generator,
    training_std: np.ndarray,
) -> SequenceRecord:
    """Apply a compatible training-only transform, preserving record metadata."""
    values = np.asarray(record.features, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] < 1 or len(values) == 0:
        raise ValueError("augmentation expects a non-empty [timesteps, channels] sequence")
    channels = values.shape[1]
    scale = np.asarray(training_std, dtype=np.float32)
    if scale.shape != (channels,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(
            f"training_std must contain {channels} positive finite channel scales"
        )

    if name == "temporal_stretch":
        factor = float(rng.uniform(0.9, 1.1))
        new_length = max(1, int(round(len(values) * factor)))
        if new_length != len(values):
            source_time = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
            target_time = np.linspace(0.0, 1.0, new_length, dtype=np.float64)
            values = np.stack(
                [np.interp(target_time, source_time, values[:, axis]) for axis in range(channels)],
                axis=1,
            ).astype(np.float32)
    elif name == "gaussian_noise":
        values += rng.normal(0.0, 0.01 * scale, size=values.shape).astype(np.float32)
    elif name == "sensor_bias_gain":
        gains = rng.uniform(0.97, 1.03, size=channels).astype(np.float32)
        offsets = rng.normal(0.0, 0.02 * scale, size=channels).astype(np.float32)
        values = values * gains + offsets
    elif name == "rotation_3d":
        rotation = _small_rotation(rng)
        vector_groups = _vector_channel_groups(record, channels)
        if not vector_groups:
            raise ValueError(
                "rotation_3d requires at least one complete measured xyz sensor group"
            )
        for indices in vector_groups:
            values[:, indices] = values[:, indices] @ rotation.T
    else:
        raise ValueError(f"{name!r} is not a per-record augmentation")

    return replace(
        record,
        features=np.ascontiguousarray(values, dtype=np.float32),
        content_sha256=None,
    )


def _vector_channel_groups(record: SequenceRecord, channel_count: int) -> tuple[tuple[int, ...], ...]:
    """Find complete xyz sensor vectors using semantic names when available."""
    if not record.channel_names:
        if channel_count == 6:
            return (tuple(range(3)), tuple(range(3, 6)))
        return ()
    groups = []
    for prefix in ("acc_front", "acc_rear", "gyro", "mag"):
        indices = tuple(
            record.channel_names.index(f"{prefix}_{axis}")
            for axis in ("x", "y", "z")
            if f"{prefix}_{axis}" in record.channel_names
        )
        if len(indices) == 3:
            groups.append(indices)
    return tuple(groups)


def _small_rotation(rng: np.random.Generator) -> np.ndarray:
    """Draw one shared, small rigid rotation for accel and gyro coordinates."""
    roll, pitch, yaw = np.deg2rad(rng.uniform(-5.0, 5.0, size=3))
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]]
    )
    ry = np.array(
        [[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]]
    )
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    return (rz @ ry @ rx).astype(np.float32)


def concatenate_training_records(
    records: Sequence[SequenceRecord],
    *,
    seed: int,
    charset: str = ONHW_CHARSET,
    delimiter: str = "",
    gap_frames: int = 8,
) -> list[SequenceRecord]:
    """Create deterministic same-writer/session synthetic word sequences.

    A short linear sensor bridge represents a controlled pause/transition.
    Only callers' training records are accepted by this function; validation
    records should never be passed here.
    """
    if gap_frames < 0:
        raise ValueError("gap_frames must be non-negative")
    if delimiter and len(delimiter) != 1:
        raise ValueError("delimiter must be empty or a single character")

    groups: dict[tuple[str, str], list[SequenceRecord]] = defaultdict(list)
    for record in records:
        encode_text(record.text, charset)
        groups[(str(record.writer_id), record.session_id or "")].append(record)

    rng = np.random.default_rng(seed)
    next_index = max((record.sample_index for record in records), default=-1) + 1
    combined: list[SequenceRecord] = []
    for group_key in sorted(groups):
        group = groups[group_key]
        if len(group) < 2:
            continue
        permutation = rng.permutation(len(group)).tolist()
        shift = int(rng.integers(1, len(group)))
        for pair_number, left_position in enumerate(permutation):
            right_position = permutation[(pair_number + shift) % len(group)]
            left, right = group[left_position], group[right_position]
            text = left.text + delimiter + right.text
            encode_text(text, charset)
            bridge = np.linspace(
                left.features[-1], right.features[0], gap_frames + 2, dtype=np.float32
            )[1:-1]
            features = np.concatenate((left.features, bridge, right.features), axis=0)
            left_id = left.sample_id or str(left.sample_index)
            right_id = right.sample_id or str(right.sample_index)
            combined.append(
                replace(
                    left,
                    features=np.ascontiguousarray(features, dtype=np.float32),
                    text=text,
                    sample_index=next_index,
                    sample_id=f"m6b-concat-{left_id}-{right_id}",
                    source_id=None,
                    content_sha256=None,
                )
            )
            next_index += 1
    return combined
