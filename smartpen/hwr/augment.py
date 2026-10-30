"""Reproducible, training-only six-axis IMU augmentations for M6-B."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

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
    """Apply one named transform to a raw six-axis record, preserving its label."""
    values = np.asarray(record.features, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
        raise ValueError("augmentation expects a non-empty (timesteps, 6) sequence")
    scale = np.asarray(training_std, dtype=np.float32)
    if scale.shape != (6,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("training_std must contain six positive finite axis scales")

    if name == "temporal_stretch":
        factor = float(rng.uniform(0.9, 1.1))
        new_length = max(1, int(round(len(values) * factor)))
        if new_length != len(values):
            source_time = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
            target_time = np.linspace(0.0, 1.0, new_length, dtype=np.float64)
            values = np.stack(
                [np.interp(target_time, source_time, values[:, axis]) for axis in range(6)],
                axis=1,
            ).astype(np.float32)
    elif name == "gaussian_noise":
        values += rng.normal(0.0, 0.01 * scale, size=values.shape).astype(np.float32)
    elif name == "sensor_bias_gain":
        gains = rng.uniform(0.97, 1.03, size=6).astype(np.float32)
        offsets = rng.normal(0.0, 0.02 * scale, size=6).astype(np.float32)
        values = values * gains + offsets
    elif name == "rotation_3d":
        rotation = _small_rotation(rng)
        values[:, :3] = values[:, :3] @ rotation.T
        values[:, 3:] = values[:, 3:] @ rotation.T
    else:
        raise ValueError(f"{name!r} is not a per-record augmentation")

    return SequenceRecord(
        features=np.ascontiguousarray(values, dtype=np.float32),
        text=record.text,
        writer_id=record.writer_id,
        sample_index=record.sample_index,
        session_id=record.session_id,
        sample_id=record.sample_id,
    )


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
                SequenceRecord(
                    features=np.ascontiguousarray(features, dtype=np.float32),
                    text=text,
                    writer_id=left.writer_id,
                    sample_index=next_index,
                    session_id=left.session_id,
                    sample_id=f"m6b-concat-{left_id}-{right_id}",
                )
            )
            next_index += 1
    return combined
