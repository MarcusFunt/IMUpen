"""Reproducible CPU/GPU training and held-out evaluation for OnHW-Words500."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
import time
from contextlib import nullcontext
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from smartpen.hwr.augment import (
    AUGMENTATION_NAMES,
    SINGLE_AUGMENTATIONS,
    apply_augmentation,
    augmentation_plan,
    concatenate_training_records,
)
from smartpen.hwr.data import (
    ONHW_BLANK_INDEX,
    ONHW_CHARSET,
    OnHWFold,
    SequenceRecord,
    apply_sensor_profile,
    encode_text,
    load_onhw_words500,
)
from smartpen.hwr.sensors import (
    ONHW_SAMPLE_RATE_HZ,
    PRIMARY_SENSOR_COMPARISONS,
    resolve_sensor_profile,
)
from smartpen.hwr.metrics import (
    corpus_error_rates,
    paired_writer_bootstrap,
    per_writer_error_rates,
    single_character_metrics,
)
from smartpen.hwr.model import (
    MODEL_TYPES,
    build_model,
    count_parameters,
    ctc_greedy_decode,
    default_model_config,
    encoded_length,
)
from smartpen.hwr.provenance import code_provenance


def seed_everything(seed: int, device_type: str = "cpu") -> None:
    """Seed Python, NumPy, and PyTorch; request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    """Use the current scaler API and retain compatibility with PyTorch 2.2."""
    scaler_type = getattr(getattr(torch, "amp", None), "GradScaler", None)
    if scaler_type is not None:
        return scaler_type("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _ctc_minimum_frames(tokens: Sequence[int]) -> int:
    """Minimum CTC frames: one per label plus one between repeated labels."""
    return len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:]))


def keep_ctc_feasible(
    records: Sequence[SequenceRecord], charset: str = ONHW_CHARSET
) -> tuple[list[SequenceRecord], int]:
    kept = []
    for record in records:
        tokens = encode_text(record.text, charset)
        if encoded_length(len(record.features)) >= _ctc_minimum_frames(tokens):
            kept.append(record)
    return kept, len(records) - len(kept)


def _official_split_summary(fold_data: OnHWFold) -> dict:
    if fold_data.split_family not in {"writer_dependent", "writer_independent"}:
        return {
            "official_protocol": None,
            "custom_protocol": fold_data.split_family,
            "interpretation": "custom split; not an official Words500 benchmark",
        }
    train_vocab = {record.text for record in fold_data.train}
    validation_vocab = {record.text for record in fold_data.validation}
    summary = {
        "official_protocol": fold_data.split_family,
        "train_writer_count": len(fold_data.train_writers),
        "validation_writer_count": len(fold_data.validation_writers),
        "writer_overlap_count": len(fold_data.train_writers & fold_data.validation_writers),
        "train_vocabulary_size": len(train_vocab),
        "validation_vocabulary_size": len(validation_vocab),
        "shared_vocabulary_size": len(train_vocab & validation_vocab),
    }
    if fold_data.split_family == "writer_dependent":
        train_counts = {
            writer: len({record.text for record in fold_data.train if record.writer_id == writer})
            for writer in fold_data.train_writers
        }
        validation_counts = {
            writer: len({record.text for record in fold_data.validation if record.writer_id == writer})
            for writer in fold_data.validation_writers
        }
        summary.update(
            {
                "interpretation": "familiar writers, held-out word identities",
                "published_word_count_target_per_writer": {"train": 400, "validation": 100},
                "observed_distinct_words_per_writer": {
                    "train_min": min(train_counts.values(), default=0),
                    "train_max": max(train_counts.values(), default=0),
                    "validation_min": min(validation_counts.values(), default=0),
                    "validation_max": max(validation_counts.values(), default=0),
                },
                "matches_published_word_counts": bool(train_counts)
                and all(value == 400 for value in train_counts.values())
                and all(value == 100 for value in validation_counts.values()),
            }
        )
    else:
        summary.update(
            {
                "interpretation": "unseen writers, familiar word identities",
                "writers_are_disjoint": fold_data.train_writers.isdisjoint(
                    fold_data.validation_writers
                ),
                "vocabulary_is_shared": bool(train_vocab & validation_vocab),
                "note": (
                    "Official WI is a writer-held-out/common-vocabulary task. This report "
                    "measures actual vocabulary overlap instead of treating WI versus WD "
                    "as a causal writer-familiarity contrast."
                ),
            }
        )
    return summary


def select_records(
    records: Sequence[SequenceRecord], limit: int | None, seed: int
) -> tuple[list[SequenceRecord], list[int]]:
    """Choose a stable, seeded subset and return its original sample indices."""
    if limit is None or limit >= len(records):
        selected = list(records)
    elif limit < 1:
        raise ValueError("sample limits must be positive")
    else:
        rng = np.random.default_rng(seed)
        positions = sorted(rng.choice(len(records), size=limit, replace=False).tolist())
        selected = [records[position] for position in positions]
    return selected, [record.sample_index for record in selected]


def feature_normalization(records: Sequence[SequenceRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-axis mean/std from fitting samples only."""
    if not records:
        raise ValueError("cannot normalize an empty training partition")
    first = np.asarray(records[0].features)
    if first.ndim != 2 or first.shape[1] < 1 or len(first) == 0:
        raise ValueError("each training sequence must be a non-empty [time, channels] array")
    channel_count = first.shape[1]
    sums = np.zeros(channel_count, dtype=np.float64)
    sum_squares = np.zeros(channel_count, dtype=np.float64)
    count = 0
    for record in records:
        values = np.asarray(record.features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != channel_count or len(values) == 0:
            raise ValueError(
                f"each training sequence must have shape (timesteps, {channel_count})"
            )
        if not np.isfinite(values).all():
            raise ValueError("normalization input contains non-finite sensor values")
        sums += values.sum(axis=0)
        sum_squares += np.square(values).sum(axis=0)
        count += len(values)
    mean = sums / count
    variance = np.maximum(sum_squares / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class _RecordDataset(Dataset):
    def __init__(self, records: Sequence[SequenceRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SequenceRecord:
        return self.records[index]


class _AugmentedRecordDataset(_RecordDataset):
    """Sample-stable, epoch-varying transforms for fitting records only."""

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        *,
        transforms: Sequence[str],
        training_std: np.ndarray,
        seed: int,
        charset: str,
    ) -> None:
        super().__init__(records)
        self.transforms = tuple(transforms)
        self.training_std = training_std
        self.seed = seed
        self.charset = charset
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> SequenceRecord:
        record = self.records[index]
        if not self.transforms:
            return record
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        augmented = record
        for transform in self.transforms:
            augmented = apply_augmentation(augmented, transform, rng, self.training_std)
        tokens = encode_text(augmented.text, self.charset)
        if encoded_length(len(augmented.features)) < _ctc_minimum_frames(tokens):
            # A shortening time warp may violate CTC's repeated-label minimum.
            return record
        return augmented


class _LengthBucketBatchSampler(Sampler[list[int]]):
    """Group similar lengths to limit padding, shuffling batches per epoch."""

    def __init__(self, records: Sequence[SequenceRecord], batch_size: int, seed: int):
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        order = sorted(range(len(self.records)), key=lambda i: len(self.records[i].features))
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size


def _collate_records(mean: np.ndarray, std: np.ndarray, charset: str = ONHW_CHARSET):
    def collate(records: list[SequenceRecord]) -> dict[str, torch.Tensor | list]:
        lengths = torch.tensor([len(record.features) for record in records], dtype=torch.long)
        max_length = int(lengths.max())
        channels = len(mean)
        if len(std) != channels:
            raise ValueError("normalization mean and standard deviation dimensions differ")
        inputs = torch.zeros((len(records), max_length, channels), dtype=torch.float32)
        token_sequences = [encode_text(record.text, charset) for record in records]
        targets = torch.tensor(
            [token for sequence in token_sequences for token in sequence], dtype=torch.long
        )
        target_lengths = torch.tensor(
            [len(sequence) for sequence in token_sequences], dtype=torch.long
        )
        for index, record in enumerate(records):
            values = (record.features - mean) / std
            inputs[index, : len(values)] = torch.from_numpy(
                np.ascontiguousarray(values, dtype=np.float32)
            )
        return {
            "inputs": inputs,
            "lengths": lengths,
            "targets": targets,
            "target_lengths": target_lengths,
            "texts": [record.text for record in records],
            "writer_ids": [record.writer_id for record in records],
            "sample_ids": [
                record.sample_id or record.source_id or str(record.sample_index)
                for record in records
            ],
        }

    return collate


def _loader(
    records: Sequence[SequenceRecord],
    batch_size: int,
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
    charset: str = ONHW_CHARSET,
    augmentation_transforms: Sequence[str] = (),
    augmentation_seed: int = 0,
) -> tuple[DataLoader, _LengthBucketBatchSampler]:
    sampler = _LengthBucketBatchSampler(records, batch_size, seed)
    loader = DataLoader(
        _AugmentedRecordDataset(
            records,
            transforms=augmentation_transforms,
            training_std=std,
            seed=augmentation_seed,
            charset=charset,
        ),
        batch_sampler=sampler,
        collate_fn=_collate_records(mean, std, charset),
        num_workers=0,
    )
    return loader, sampler


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.CTCLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    mixed_precision: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    charset: str = ONHW_CHARSET,
    blank_index: int = ONHW_BLANK_INDEX,
    include_metadata: bool = False,
) -> tuple[float, list[str], list[str]] | tuple[
    float, list[str], list[str], list[int | str], list[str]
]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    sample_count = 0
    predictions: list[str] = []
    references: list[str] = []
    writer_ids: list[int | str] = []
    sample_ids: list[str] = []
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            inputs = batch["inputs"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision
                else nullcontext()
            )
            with autocast_context:
                logits, output_lengths = model(inputs, lengths)
            # CTCLoss is kept in float32 for stable log-probabilities and gradients.
            log_probabilities = logits.float().log_softmax(dim=-1).transpose(0, 1)
            loss = loss_function(
                log_probabilities,
                targets,
                output_lengths.to(device="cpu"),
                target_lengths.to(device="cpu"),
            )
            if optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            batch_size = len(batch["texts"])
            total_loss += float(loss.detach()) * batch_size
            sample_count += batch_size
            token_ids = logits.detach().argmax(dim=-1).cpu()
            for index, reference in enumerate(batch["texts"]):
                output_length = int(output_lengths[index].detach().cpu())
                predictions.append(
                    ctc_greedy_decode(
                        token_ids[index, :output_length].tolist(), blank_index, charset
                    )
                )
                references.append(reference)
                if include_metadata:
                    writer_ids.append(batch["writer_ids"][index])
                    sample_ids.append(batch["sample_ids"][index])
    result = total_loss / max(sample_count, 1), predictions, references
    if include_metadata:
        return (*result, writer_ids, sample_ids)
    return result


def _sha256(path: Path) -> str | None:
    if path.is_dir():
        digest = hashlib.sha256()
        files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
        for candidate in files:
            digest.update(candidate.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_torch_save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-model-", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_training(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    fold: int = 0,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 1337,
    device_name: str = "auto",
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    num_threads: int = 4,
    model_type: str = "cnn_bilstm_ctc",
    sensor_profile: str = "front_acc_gyro6",
    mixed_precision: bool | None = None,
    training_stage: str = "supervised",
    model_config: dict | None = None,
    augmentation: str = "none",
    concat_delimiter: str = "",
    concat_gap_frames: int = 8,
    resume: bool = False,
    records_override: OnHWFold | None = None,
    custom_split_manifest: dict | None = None,
    dataset_name: str = "OnHW-Words500",
    protocol_label: str | None = None,
    dataset_summary: dict | None = None,
    writer_ids_available: bool = True,
    metric_mode: str = "word_sequence",
    label_charset: str | None = None,
) -> dict:
    """Train a CTC model on one declared OnHW split and sensor profile."""
    if epochs < 1 or batch_size < 1 or num_threads < 1:
        raise ValueError("epochs, batch_size, and num_threads must be positive")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unknown model type {model_type!r}; choose from {MODEL_TYPES}")
    if metric_mode not in {"word_sequence", "single_character"}:
        raise ValueError("metric_mode must be word_sequence or single_character")
    active_charset = ONHW_CHARSET if label_charset is None else label_charset
    if not active_charset or len(active_charset) != len(set(active_charset)):
        raise ValueError("label_charset must contain unique characters")
    if not writer_ids_available and custom_split_manifest is not None:
        raise ValueError("custom writer/vocabulary factorial training requires writer IDs")
    transforms, add_concatenations = augmentation_plan(augmentation)
    if augmentation not in AUGMENTATION_NAMES:
        raise ValueError(f"unknown augmentation {augmentation!r}")
    if concat_gap_frames < 0:
        raise ValueError("concat_gap_frames must be non-negative")
    seed_everything(seed, device.type)
    torch.set_num_threads(num_threads)

    profile = resolve_sensor_profile(sensor_profile)
    if "rotation_3d" in transforms and not any(
        group.code in profile.group_code for group in profile.groups
        if group.code in {"F", "R", "G", "M"}
    ):
        raise ValueError(
            f"rotation_3d is incompatible with sensor profile {profile.profile_id}"
        )
    fold_data: OnHWFold = records_override or load_onhw_words500(data_path, fold=fold)
    train_writers = fold_data.train_writers
    validation_writers = fold_data.validation_writers
    writer_overlap = (
        not train_writers.isdisjoint(validation_writers) if writer_ids_available else None
    )
    if (
        writer_ids_available
        and fold_data.split_family == "writer_independent"
        and writer_overlap
    ):
        raise ValueError("writer leakage between official writer-independent partitions")
    if (
        writer_ids_available
        and fold_data.split_family == "writer_dependent"
        and not writer_overlap
    ):
        raise ValueError("official writer-dependent partitions must share writers")
    if not writer_ids_available and any(
        record.writer_id is not None for record in fold_data.train + fold_data.validation
    ):
        raise ValueError("writer_ids_available=False requires unknown writer IDs to remain null")
    if custom_split_manifest is not None:
        if max_train_samples is not None:
            raise ValueError(
                "sample-capped custom training could drop required exposure anchors; "
                "set the matched budget when building the factorial split instead"
            )
        if custom_split_manifest.get("protocol") != "custom_writer_vocabulary_factorial":
            raise ValueError("unsupported custom split manifest protocol")
        expected_writer_exposure = bool(custom_split_manifest.get("writer_exposure"))
        expected_vocabulary_exposure = bool(custom_split_manifest.get("vocabulary_exposure"))
        actual_vocabulary_exposure = bool(
            {record.text for record in fold_data.train}
            & {record.text for record in fold_data.validation}
        )
        if writer_overlap != expected_writer_exposure:
            raise ValueError("custom split writer exposure does not match its manifest")
        if actual_vocabulary_exposure != expected_vocabulary_exposure:
            raise ValueError("custom split vocabulary exposure does not match its manifest")
        train_fingerprints = {record.content_sha256 for record in fold_data.train}
        query_fingerprints = {record.content_sha256 for record in fold_data.validation}
        train_fingerprints.discard(None)
        query_fingerprints.discard(None)
        if train_fingerprints & query_fingerprints:
            raise ValueError("custom split reuses a fingerprinted recording across train/query")

    profiled_train = apply_sensor_profile(fold_data.train, profile.profile_id)
    profiled_validation = apply_sensor_profile(fold_data.validation, profile.profile_id)
    if metric_mode == "single_character" and any(
        len(record.text) != 1 for record in profiled_train + profiled_validation
    ):
        raise ValueError("single-character classification requires one-character references")
    if any(character not in active_charset for record in profiled_train + profiled_validation
           for character in record.text):
        raise ValueError("a reference contains a character outside label_charset")
    for partition, records in (("train", profiled_train), ("validation", profiled_validation)):
        for record in records:
            values = np.asarray(record.features)
            if (
                values.ndim != 2
                or values.shape[1] != profile.input_channels
                or len(values) == 0
            ):
                raise ValueError(
                    f"{partition} sample {record.sample_id or record.sample_index} does not "
                    f"match sensor profile {profile.profile_id}"
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{partition} sample {record.sample_id or record.sample_index} "
                    "contains non-finite sensor values"
                )
    blank_index = len(active_charset)
    feasible_train, dropped_train_ctc = keep_ctc_feasible(profiled_train, active_charset)
    feasible_validation, dropped_validation_ctc = keep_ctc_feasible(
        profiled_validation, active_charset
    )
    train_records, train_indices = select_records(feasible_train, max_train_samples, seed)
    validation_records, validation_indices = select_records(
        feasible_validation, max_validation_samples, seed + 1
    )
    if not train_records or not validation_records:
        raise ValueError("training and validation subsets must both be non-empty")
    # Fit the same original-data statistics for every ablation. Augmentations
    # are applied afterwards and only to fitting records.
    mean, std = feature_normalization(train_records)
    concatenated_records = (
        concatenate_training_records(
            train_records,
            seed=seed,
            charset=active_charset,
            delimiter=concat_delimiter,
            gap_frames=concat_gap_frames,
        )
        if add_concatenations
        else []
    )
    concatenated_records, dropped_concatenations_ctc = keep_ctc_feasible(
        concatenated_records
    )
    augmented_train_records = train_records + concatenated_records

    config = default_model_config(
        model_type, blank_index + 1, input_channels=profile.input_channels
    )
    if model_config:
        config.update(model_config)
    if int(config.get("input_channels", -1)) != profile.input_channels:
        raise ValueError(
            "model input_channels does not match the selected sensor profile "
            f"({profile.profile_id}: {profile.input_channels})"
        )
    model = build_model(model_type, config).to(device)
    parameter_count = count_parameters(model)
    use_mixed_precision = (
        device.type == "cuda" if mixed_precision is None else mixed_precision
    )
    if use_mixed_precision and device.type != "cuda":
        raise ValueError("mixed precision is supported only for CUDA training")
    scaler = _make_grad_scaler(use_mixed_precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    train_loader, train_sampler = _loader(
        augmented_train_records,
        batch_size,
        seed,
        mean,
        std,
        charset=active_charset,
        augmentation_transforms=transforms,
        augmentation_seed=seed,
    )
    validation_loader, _ = _loader(
        validation_records, batch_size, seed + 1, mean, std, charset=active_charset
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.pt"
    data_hash = _sha256(Path(data_path))
    resume_signature = {
        "archive_sha256": data_hash,
        "fold": fold,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_threads": num_threads,
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "mixed_precision": use_mixed_precision,
        "model_type": model_type,
        "model_config": config,
        "sensor_profile": profile.profile_id,
        "train_sample_ids": [record.sample_id or record.source_id for record in train_records],
        "validation_sample_ids": [record.sample_id or record.source_id for record in validation_records],
        "augmentation": augmentation,
        "concat_delimiter": concat_delimiter if add_concatenations else None,
        "concat_gap_frames": concat_gap_frames if add_concatenations else None,
        "custom_split_manifest": custom_split_manifest,
        "dataset_name": dataset_name,
        "protocol_label": protocol_label,
        "writer_ids_available": writer_ids_available,
        "metric_mode": metric_mode,
        "label_charset": active_charset,
        "code_provenance": code_provenance(),
    }
    resume_signature_sha256 = hashlib.sha256(
        json.dumps(resume_signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    epoch_results: list[dict] = []
    best_model_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_wer = float("inf")
    best_validation_loss = float("inf")
    best_predictions: list[str] = []
    best_references: list[str] = []
    best_writer_ids: list[int | str] = []
    best_sample_ids: list[str] = []
    start_epoch = 0
    if not resume and progress_path.exists():
        progress_path.unlink()
    if resume and progress_path.is_file():
        progress = torch.load(progress_path, map_location="cpu", weights_only=True)
        if progress.get("resume_signature_sha256") != resume_signature_sha256:
            raise ValueError(
                "saved training progress does not match this dataset, split, model, "
                "or training configuration"
            )
        model.load_state_dict(progress["model_state"])
        optimizer.load_state_dict(progress["optimizer_state"])
        scaler.load_state_dict(progress["scaler_state"])
        epoch_results = progress["epoch_results"]
        best_model_state = progress["best_model_state"]
        best_epoch = int(progress["best_epoch"])
        best_validation_wer = float(progress["best_validation_wer"])
        best_validation_loss = float(progress["best_validation_loss"])
        best_predictions = progress["best_predictions"]
        best_references = progress["best_references"]
        best_writer_ids = progress["best_writer_ids"]
        best_sample_ids = progress["best_sample_ids"]
        start_epoch = int(progress["completed_epochs"])
        torch.set_rng_state(progress["torch_rng_state"])
        if device.type == "cuda" and progress.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(progress["cuda_rng_states"])
        if start_epoch > epochs:
            raise ValueError("saved progress completed more epochs than requested")

    _synchronize_device(device)
    training_started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        train_loader.dataset.set_epoch(epoch)
        train_loss, _, _ = _run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
            mixed_precision=use_mixed_precision,
            scaler=scaler,
            charset=active_charset,
            blank_index=blank_index,
        )
        _synchronize_device(device)
        validation_started = time.perf_counter()
        validation_loss, hypotheses, references, writer_ids, sample_ids = _run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            mixed_precision=use_mixed_precision,
            charset=active_charset,
            blank_index=blank_index,
            include_metadata=True,
        )
        metrics = corpus_error_rates(hypotheses, references)
        if metric_mode == "single_character":
            character_metrics = single_character_metrics(
                hypotheses,
                references,
                labels=list(active_charset),
                summary_only=True,
            )
            selection_value = 1.0 - character_metrics["accuracy"]
        else:
            character_metrics = None
            selection_value = float(metrics["wer"])
        _synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_report = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_samples": len(references),
            "epoch_seconds": epoch_seconds,
            "validation_seconds": validation_seconds,
            "validation_samples_per_second": len(references) / max(validation_seconds, 1e-12),
        }
        if metric_mode == "single_character":
            epoch_report["validation_character_metrics"] = character_metrics
        else:
            epoch_report.update(
                {
                    "validation_cer": metrics["cer"],
                    "validation_wer": metrics["wer"],
                    "character_errors": metrics["character_errors"],
                    "reference_characters": metrics["reference_characters"],
                    "word_errors": metrics["word_errors"],
                    "reference_words": metrics["reference_words"],
                    "exact_word_accuracy": metrics["exact_word_accuracy"],
                    "exact_word_count": metrics["exact_word_count"],
                }
            )
        epoch_results.append(epoch_report)

        if (selection_value, validation_loss) < (
            best_validation_wer,
            best_validation_loss,
        ):
            best_validation_wer = float(selection_value)
            best_validation_loss = float(validation_loss)
            best_epoch = epoch + 1
            best_predictions = list(hypotheses)
            best_references = list(references)
            best_writer_ids = list(writer_ids)
            best_sample_ids = list(sample_ids)
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if best_model_state is None:
            raise RuntimeError("training did not produce a selectable checkpoint")
        progress_value = {
            "resume_signature_sha256": resume_signature_sha256,
            "completed_epochs": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch_results": epoch_results,
            "best_model_state": best_model_state,
            "best_epoch": best_epoch,
            "best_validation_wer": best_validation_wer,
            "best_validation_loss": best_validation_loss,
            "best_predictions": best_predictions,
            "best_references": best_references,
            "best_writer_ids": best_writer_ids,
            "best_sample_ids": best_sample_ids,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        _atomic_torch_save(progress_path, progress_value)

    _synchronize_device(device)
    training_seconds_this_attempt = time.perf_counter() - training_started
    if best_model_state is None:
        raise RuntimeError("no completed epoch is available for checkpoint selection")
    model.load_state_dict(best_model_state)
    final_metrics = (
        single_character_metrics(
            best_predictions,
            best_references,
                labels=list(active_charset),
        )
        if metric_mode == "single_character"
        else corpus_error_rates(best_predictions, best_references)
    )
    channel_source = profiled_train[0] if profiled_train else None
    effective_protocol = protocol_label or fold_data.split_family
    effective_summary = dataset_summary or _official_split_summary(fold_data)
    if metric_mode == "single_character":
        checkpoint_criterion = (
            "maximum exact-character accuracy, then minimum validation loss"
        )
    else:
        checkpoint_criterion = "minimum validation WER, then validation loss"
    writer_metrics = (
        {
            metric: float(
                np.mean(
                    [
                        result[metric]
                        for result in per_writer_error_rates(
                            best_predictions, best_references, best_writer_ids
                        ).values()
                    ]
                )
            )
            for metric in ("wer", "cer")
        }
        if writer_ids_available
        else None
    )
    per_writer_metrics = (
        per_writer_error_rates(best_predictions, best_references, best_writer_ids)
        if writer_ids_available
        else None
    )

    report = {
        "schema_version": 1,
        "run_kind": "sample" if max_train_samples or max_validation_samples else "full",
        "dataset": dataset_name,
        "task_type": (
            "single-character classification"
            if metric_mode == "single_character"
            else "word-sequence recognition"
        ),
        "metric_mode": metric_mode,
        "protocol": effective_protocol,
        "source_archive": Path(data_path).name,
        "archive_sha256": _sha256(Path(data_path)),
        "fold": fold,
        "split": protocol_label or (
            "custom writer/vocabulary factorial cell; query recordings are fixed across cells"
            if custom_split_manifest is not None
            else (
                "official writer-independent split; writers are disjoint and vocabulary is shared"
                if fold_data.split_family == "writer_independent"
                else "official writer-dependent split; writers overlap and word identities are held out"
            )
        ),
        "writer_overlap": writer_overlap,
        "writer_ids_available": writer_ids_available,
        "official_split_summary": effective_summary,
        "custom_split_manifest": custom_split_manifest,
        "analysis_registration": {
            "primary_sensor_comparisons": list(PRIMARY_SENSOR_COMPARISONS),
            "full_31_profile_ranking": "exploratory_until_independent_confirmation",
        },
        "train_samples_in_archive": len(fold_data.train) + fold_data.dropped_train,
        "validation_samples_in_archive": (
            len(fold_data.validation) + fold_data.dropped_validation
        ),
        "train_samples_after_empty_filter": len(fold_data.train),
        "validation_samples_after_empty_filter": len(fold_data.validation),
        "train_samples_ctc_feasible": len(feasible_train),
        "validation_samples_ctc_feasible": len(feasible_validation),
        "train_samples_used": len(train_records),
        "train_sequences_after_augmentation": len(augmented_train_records),
        "sequence_concatenations_added": len(concatenated_records),
        "sequence_concatenations_dropped_ctc": dropped_concatenations_ctc,
        "validation_samples_used": len(validation_records),
        "train_writers_available": len(train_writers) if writer_ids_available else None,
        "validation_writers_available": len(validation_writers) if writer_ids_available else None,
        "train_writers_sampled": (
            len({record.writer_id for record in train_records}) if writer_ids_available else None
        ),
        "validation_writers_sampled": (
            len({record.writer_id for record in validation_records})
            if writer_ids_available else None
        ),
        "writer_leakage": (
            fold_data.split_family == "writer_independent" and bool(writer_overlap)
            if writer_ids_available
            else None
        ),
        "sensor_profile": {
            "profile_id": profile.profile_id,
            "group_code": profile.group_code,
            "channel_indices": list(profile.channel_indices),
            "channel_names": list(channel_source.channel_names)
            if channel_source is not None and channel_source.channel_names
            else list(profile.channel_names),
            "units": list(channel_source.channel_units)
            if channel_source is not None and channel_source.channel_units
            else list(profile.units),
            "sample_rate_hz": channel_source.sample_rate_hz
            if channel_source is not None
            else ONHW_SAMPLE_RATE_HZ,
            "physical_description": profile.physical_description,
            "preprocessing": dict(profile.preprocessing),
            "dataset_compatibility": (
                "onhw_chars_13_channel"
                if dataset_name == "OnHW-Chars"
                else profile.dataset_compatibility
            ),
        },
        "identity_policy": {
            "stable_source_ids_available": False,
            "source_id": "null when the archive has no authoritative recording identifier",
            "sample_id": (
                "dataset/archive + split family + fold + partition + split-local index; "
                "unique only within that published partition identity"
            ),
            "content_fingerprint": "SHA-256 of canonical float32 little-endian [time, 13] data",
            "cross_fold_limit": (
                "exact fingerprint matches can identify unchanged recordings; transformed "
                "signals cannot be reliably reconstructed as the same source recording"
            ),
        },
        "empty_recordings_dropped": {
            "train": fold_data.dropped_train,
            "validation": fold_data.dropped_validation,
        },
        "ctc_infeasible_recordings_dropped": {
            "train": dropped_train_ctc,
            "validation": dropped_validation_ctc,
        },
        "sample_indices": {
            "train": train_indices,
            "validation": validation_indices,
        },
        "source_ids": {
            "train": [record.source_id for record in train_records],
            "validation": [record.source_id for record in validation_records],
        },
        "sample_ids": {
            "train": [record.sample_id for record in train_records],
            "validation": [record.sample_id for record in validation_records],
        },
        "content_fingerprints": {
            "train": [record.content_sha256 for record in train_records],
            "validation": [record.content_sha256 for record in validation_records],
        },
        "seed": seed,
        "num_threads": num_threads,
        "batch_size": batch_size,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "deterministic_algorithms": True,
        "normalization": {
            "source": "selected training samples only",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "model_type": model_type,
        "training_stage": training_stage,
        "augmentation": {
            "name": augmentation,
            "online_transforms": list(transforms),
            "sequence_concatenation": add_concatenations,
            "concat_delimiter": concat_delimiter if add_concatenations else None,
            "concat_gap_frames": concat_gap_frames if add_concatenations else None,
            "seed": seed,
            "applied_only_to_training": True,
            "normalization_fit_before_augmentation": True,
            "parameters": {
                "temporal_stretch_factor": [0.9, 1.1],
                "gaussian_noise_axis_std_fraction": 0.01,
                "sensor_gain_range": [0.97, 1.03],
                "sensor_bias_std_fraction": 0.02,
                "rotation_euler_degrees": [-5.0, 5.0],
            },
        },
        "model": config,
        "model_parameters": parameter_count,
        "mixed_precision": use_mixed_precision,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "ctc": {"blank_index": blank_index, "zero_infinity": True},
        "label_charset": active_charset,
        "epochs": epoch_results,
        "final_metrics": final_metrics,
        "macro_writer_metrics": writer_metrics,
        "per_writer_metrics": per_writer_metrics,
        "validation_outputs": {
            "sample_ids": best_sample_ids,
            "writer_ids": best_writer_ids,
            "hypotheses": best_predictions,
            "references": best_references,
        },
        "checkpoint_selection": {
            "criterion": checkpoint_criterion,
            "selected_epoch": best_epoch,
            "selected_validation_loss": best_validation_loss,
            "selected_error_score": best_validation_wer,
            "selected_error_score_name": (
                "character_error_rate"
                if metric_mode == "single_character"
                else "word_error_rate"
            ),
        },
        "resumed_from_epoch": start_epoch,
        "training_seconds_this_attempt": training_seconds_this_attempt,
        "training_signature_sha256": resume_signature_sha256,
        "code_provenance": code_provenance(),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Sample-capped integration smoke test only; not an accuracy or full-fold benchmark."
            if report_is_sample(max_train_samples, max_validation_samples)
            else "Measured on this official fold only; no cross-fold average is implied."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_type": model_type,
        "training_stage": training_stage,
        "augmentation": report["augmentation"],
        "model_config": config,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "charset": active_charset,
        "blank_index": blank_index,
        "seed": seed,
        "fold": fold,
        "sensor_profile_id": profile.profile_id,
        "sensor_channel_indices": list(profile.channel_indices),
        "sensor_channel_names": list(profile.channel_names),
        "sensor_channel_units": list(profile.units),
        "sample_rate_hz": ONHW_SAMPLE_RATE_HZ,
        "sensor_profile": report["sensor_profile"],
        "training_report": report,
        "training_signature_sha256": resume_signature_sha256,
        "selected_epoch": best_epoch,
    }
    checkpoint_path = output / "model.pt"
    _atomic_torch_save(checkpoint_path, checkpoint)
    report["checkpoint"] = checkpoint_path.name
    _atomic_json(output / "metrics.json", report)
    if progress_path.exists():
        progress_path.unlink()
    return report


def report_is_sample(
    max_train_samples: int | None, max_validation_samples: int | None
) -> bool:
    return max_train_samples is not None or max_validation_samples is not None


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    data_path: str | Path,
    fold: int | None = None,
    max_validation_samples: int | None = None,
    seed: int | None = None,
    device_name: str = "auto",
    batch_size: int = 32,
    num_threads: int = 4,
    mixed_precision: bool | None = None,
    records_override: OnHWFold | None = None,
    dataset_name: str = "OnHW-Words500",
    protocol_label: str | None = None,
    dataset_summary: dict | None = None,
    writer_ids_available: bool = True,
    metric_mode: str = "word_sequence",
    label_charset: str | None = None,
) -> dict:
    """Evaluate a checkpoint on an official or explicitly supplied held-out split."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selected_fold = int(checkpoint["fold"] if fold is None else fold)
    selected_seed = int(checkpoint["seed"] if seed is None else seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    use_mixed_precision = (
        device.type == "cuda" if mixed_precision is None else mixed_precision
    )
    if use_mixed_precision and device.type != "cuda":
        raise ValueError("mixed precision is supported only for CUDA evaluation")
    if metric_mode not in {"word_sequence", "single_character"}:
        raise ValueError("metric_mode must be word_sequence or single_character")
    checkpoint_charset = checkpoint.get("charset", ONHW_CHARSET)
    active_charset = checkpoint_charset if label_charset is None else label_charset
    if active_charset != checkpoint_charset:
        raise ValueError("evaluation charset does not match the checkpoint charset")
    if not active_charset or len(active_charset) != len(set(active_charset)):
        raise ValueError("label_charset must contain unique characters")
    seed_everything(selected_seed, device.type)
    torch.set_num_threads(num_threads)
    fold_data = records_override or load_onhw_words500(data_path, selected_fold)
    profile = resolve_sensor_profile(
        checkpoint.get("sensor_profile_id", "front_acc_gyro6")
    )
    profiled_validation = apply_sensor_profile(fold_data.validation, profile.profile_id)
    if any(character not in active_charset for record in profiled_validation
           for character in record.text):
        raise ValueError("a reference contains a character outside label_charset")
    blank_index = int(checkpoint.get("blank_index", len(active_charset)))
    if blank_index != len(active_charset):
        raise ValueError("checkpoint CTC blank index does not follow its label charset")
    records, dropped_ctc = keep_ctc_feasible(profiled_validation, active_charset)
    records, indices = select_records(records, max_validation_samples, selected_seed + 1)
    model_type = checkpoint.get("model_type", "cnn_bilstm_ctc")
    model = build_model(model_type, checkpoint["model_config"]).to(device)
    if int(checkpoint["model_config"].get("input_channels", -1)) != profile.input_channels:
        raise ValueError("checkpoint model input dimensions do not match its sensor profile")
    model.load_state_dict(checkpoint["model_state"])
    mean = checkpoint["normalization_mean"].numpy().astype(np.float32)
    std = checkpoint["normalization_std"].numpy().astype(np.float32)
    loader, _ = _loader(
        records, batch_size, selected_seed + 1, mean, std, charset=active_charset
    )
    loss_function = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    _synchronize_device(device)
    inference_started = time.perf_counter()
    loss, hypotheses, references, writer_ids, sample_ids = _run_epoch(
        model,
        loader,
        loss_function,
        device,
        mixed_precision=use_mixed_precision,
        charset=active_charset,
        blank_index=blank_index,
        include_metadata=True,
    )
    _synchronize_device(device)
    inference_seconds = time.perf_counter() - inference_started
    result_metrics = (
        single_character_metrics(
            hypotheses, references, labels=list(active_charset)
        )
        if metric_mode == "single_character"
        else corpus_error_rates(hypotheses, references)
    )
    per_writer_metrics = (
        per_writer_error_rates(hypotheses, references, writer_ids)
        if writer_ids_available
        else None
    )
    macro_writer_metrics = (
        {
            metric: float(np.mean([values[metric] for values in per_writer_metrics.values()]))
            for metric in ("wer", "cer")
        }
        if per_writer_metrics is not None
        else None
    )
    result = {
        "dataset": dataset_name,
        "task_type": (
            "single-character classification"
            if metric_mode == "single_character"
            else "word-sequence recognition"
        ),
        "metric_mode": metric_mode,
        "protocol": protocol_label or fold_data.split_family,
        "official_split_summary": dataset_summary or _official_split_summary(fold_data),
        "sensor_profile_id": profile.profile_id,
        "model_type": model_type,
        "model_parameters": count_parameters(model),
        "fold": selected_fold,
        "validation_samples": len(records),
        "writer_ids_available": writer_ids_available,
        "validation_writers": (
            len({record.writer_id for record in records}) if writer_ids_available else None
        ),
        "writer_overlap": (
            not fold_data.train_writers.isdisjoint(fold_data.validation_writers)
            if writer_ids_available
            else None
        ),
        "writer_leakage": (
            fold_data.split_family == "writer_independent"
            and not fold_data.train_writers.isdisjoint(fold_data.validation_writers)
            if writer_ids_available
            else None
        ),
        "ctc_infeasible_recordings_dropped": dropped_ctc,
        "sample_indices": indices,
        "validation_loss": loss,
        "inference_seconds": inference_seconds,
        "inference_samples_per_second": len(records) / max(inference_seconds, 1e-12),
        "metrics": result_metrics,
        "macro_writer_metrics": macro_writer_metrics,
        "per_writer_metrics": per_writer_metrics,
        "validation_outputs": {
            "sample_ids": sample_ids,
            "writer_ids": writer_ids,
            "hypotheses": hypotheses,
            "references": references,
        },
        "seed": selected_seed,
        "device": str(device),
        "mixed_precision": use_mixed_precision,
    }
    return result


def run_comparison(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    fold: int = 0,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 1337,
    device_name: str = "auto",
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    num_threads: int = 4,
    mixed_precision: bool | None = None,
    sensor_profile: str = "front_acc_gyro6",
) -> dict:
    """Train M4 and M5 sequentially on identical samples and write a comparison."""
    output = Path(output_dir)
    runs = {}
    for model_type in MODEL_TYPES:
        runs[model_type] = run_training(
            data_path=data_path,
            output_dir=output / model_type,
            fold=fold,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device_name=device_name,
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            num_threads=num_threads,
            model_type=model_type,
            sensor_profile=sensor_profile,
            mixed_precision=mixed_precision,
        )

    baseline = runs["cnn_bilstm_ctc"]
    resnet = runs["resnet_bilstm_ctc"]
    same_split = (
        baseline["sample_indices"] == resnet["sample_indices"]
        and baseline["archive_sha256"] == resnet["archive_sha256"]
        and baseline["fold"] == resnet["fold"]
        and baseline["seed"] == resnet["seed"]
        and baseline["normalization"] == resnet["normalization"]
        and baseline["sensor_profile"] == resnet["sensor_profile"]
    )
    if not same_split:
        raise RuntimeError("M4/M5 comparison did not use identical data and split settings")
    same_optimization = (
        len(baseline["epochs"]) == len(resnet["epochs"]) == epochs
        and all(
            baseline[key] == resnet[key]
            for key in ("mixed_precision", "num_threads", "batch_size", "device")
        )
    )
    if not same_optimization:
        raise RuntimeError("M4/M5 comparison settings differ")
    if baseline["writer_leakage"] or resnet["writer_leakage"]:
        raise RuntimeError("M4/M5 comparison includes writer leakage")

    paired_analysis = None
    if all("validation_outputs" in report for report in runs.values()):
        first_outputs = baseline["validation_outputs"]
        second_outputs = resnet["validation_outputs"]
        for field in ("sample_ids", "writer_ids", "references"):
            if first_outputs[field] != second_outputs[field]:
                raise RuntimeError(f"M4/M5 paired validation {field} are not aligned")
        paired_analysis = paired_writer_bootstrap(
            first_outputs["hypotheses"],
            second_outputs["hypotheses"],
            first_outputs["references"],
            first_outputs["writer_ids"],
            seed=seed,
        )

    comparison = {
        "schema_version": 1,
        "run_kind": baseline["run_kind"],
        "dataset": baseline["dataset"],
        "protocol": baseline["protocol"],
        "sensor_profile": baseline["sensor_profile"],
        "archive_sha256": baseline["archive_sha256"],
        "fold": fold,
        "seed": seed,
        "device": baseline["device"],
        "device_name": baseline["device_name"],
        "batch_size": batch_size,
        "epochs": epochs,
        "mixed_precision": baseline["mixed_precision"],
        "python_version": baseline["python_version"],
        "torch_version": baseline["torch_version"],
        "numpy_version": baseline["numpy_version"],
        "deterministic_algorithms": baseline["deterministic_algorithms"],
        "normalization": baseline["normalization"],
        "train_sample_count": baseline["train_samples_used"],
        "validation_sample_count": baseline["validation_samples_used"],
        "train_writer_count": baseline["train_writers_sampled"],
        "validation_writer_count": baseline["validation_writers_sampled"],
        "train_sample_indices": baseline["sample_indices"]["train"],
        "validation_sample_indices": baseline["sample_indices"]["validation"],
        "fairness_checks": {
            "same_archive_sha256": True,
            "same_official_fold_and_sample_indices": True,
            "same_train_only_normalization": True,
            "same_seed_batch_size_epochs_and_device": True,
            "writer_leakage": False,
        },
        "paired_writer_analysis": paired_analysis,
        "models": {
            model_type: {
                "parameters": report["model_parameters"],
                "cuda_peak_memory_bytes": report["cuda_peak_memory_bytes"],
                "configuration": report["model"],
                "final_metrics": report["final_metrics"],
                "macro_writer_metrics": report.get("macro_writer_metrics"),
                "per_writer_metrics": report.get("per_writer_metrics"),
                "final_metric_counts": {
                    key: report["final_metrics"].get(key)
                    for key in (
                        "character_errors",
                        "reference_characters",
                        "word_errors",
                        "reference_words",
                        "exact_word_count",
                        "sample_count",
                    )
                },
                "epoch_metrics": report["epochs"],
                "run_directory": model_type,
            }
            for model_type, report in runs.items()
        },
        "interpretation": (
            "Sample-capped integration smoke test only; not an accuracy benchmark or a "
            "claim of improved handwriting accuracy."
            if report_is_sample(max_train_samples, max_validation_samples)
            else "Measured on this full official fold only; no cross-fold average is implied."
        ),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "comparison.json", comparison)
    return comparison


def run_augmentation_ablation(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    fold: int = 0,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 1337,
    device_name: str = "auto",
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    num_threads: int = 4,
    mixed_precision: bool | None = None,
    model_type: str = "resnet_bilstm_ctc",
    include_combined: bool = False,
    concat_delimiter: str = "",
    concat_gap_frames: int = 8,
    sensor_profile: str = "front_acc_gyro6",
) -> dict:
    """Run M6-B baseline and isolated augmentation experiments on one split."""
    output = Path(output_dir)
    variants = ["none", *SINGLE_AUGMENTATIONS]
    if include_combined:
        variants.append("combined")
    reports: dict[str, dict] = {}
    for variant in variants:
        reports[variant] = run_training(
            data_path=data_path,
            output_dir=output / variant,
            fold=fold,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device_name=device_name,
            max_train_samples=max_train_samples,
            max_validation_samples=max_validation_samples,
            num_threads=num_threads,
            model_type=model_type,
            sensor_profile=sensor_profile,
            mixed_precision=mixed_precision,
            training_stage="m6b_augmentation_ablation",
            augmentation=variant,
            concat_delimiter=concat_delimiter,
            concat_gap_frames=concat_gap_frames,
        )

    reference = reports["none"]
    common_keys = (
        "archive_sha256",
        "fold",
        "seed",
        "sample_indices",
        "normalization",
        "sensor_profile",
        "model_type",
        "model",
        "batch_size",
        "mixed_precision",
        "num_threads",
        "device",
        "optimizer",
    )
    same_setup = all(
        all(report[key] == reference[key] for key in common_keys)
        and len(report["epochs"]) == epochs
        for report in reports.values()
    )
    writer_safe = all(not report["writer_leakage"] for report in reports.values())
    validation_unaugmented = all(
        report["augmentation"]["applied_only_to_training"]
        and report["augmentation"]["normalization_fit_before_augmentation"]
        for report in reports.values()
    )
    if not same_setup or not writer_safe or not validation_unaugmented:
        raise RuntimeError("M6-B ablations failed their shared-split/fairness checks")

    result = {
        "schema_version": 1,
        "experiment": "M6-B independent six-axis IMU augmentation ablation",
        "run_kind": reference["run_kind"],
        "dataset": reference["dataset"],
        "source_archive": Path(data_path).name,
        "archive_sha256": reference["archive_sha256"],
        "fold": fold,
        "seed": seed,
        "model_type": model_type,
        "model_parameters": reference["model_parameters"],
        "model_configuration": reference["model"],
        "device": reference["device"],
        "batch_size": batch_size,
        "epochs": epochs,
        "num_threads": num_threads,
        "mixed_precision": reference["mixed_precision"],
        "optimizer": reference["optimizer"],
        "python_version": reference["python_version"],
        "torch_version": reference["torch_version"],
        "numpy_version": reference["numpy_version"],
        "deterministic_algorithms": reference["deterministic_algorithms"],
        "train_sample_count": reference["train_samples_used"],
        "validation_sample_count": reference["validation_samples_used"],
        "train_sample_indices": reference["sample_indices"]["train"],
        "validation_sample_indices": reference["sample_indices"]["validation"],
        "normalization": reference["normalization"],
        "variants": {
            variant: {
                "augmentation": report["augmentation"],
                "final_metrics": report["final_metrics"],
                "final_metric_counts": {
                    key: report["epochs"][-1][key]
                    for key in (
                        "character_errors",
                        "reference_characters",
                        "word_errors",
                        "reference_words",
                    )
                },
                "epoch_metrics": report["epochs"],
                "train_sequences_after_augmentation": report[
                    "train_sequences_after_augmentation"
                ],
                "sequence_concatenations_added": report[
                    "sequence_concatenations_added"
                ],
                "sequence_concatenations_dropped_ctc": report[
                    "sequence_concatenations_dropped_ctc"
                ],
                "run_directory": variant,
            }
            for variant, report in reports.items()
        },
        "fairness_checks": {
            "same_archive_fold_and_sample_indices": same_setup,
            "same_model_optimizer_seed_and_epoch_count": same_setup,
            "same_training_only_normalization": same_setup,
            "validation_not_augmented": validation_unaugmented,
            "writer_leakage": not writer_safe,
            "sequence_concatenation_uses_training_only_same_writer_session": True,
        },
        "interpretation": (
            "Sample-capped implementation smoke experiment. Results verify the "
            "pipeline and are not benchmark or accuracy claims."
            if report_is_sample(max_train_samples, max_validation_samples)
            else "Single-fold ablation results only; no cross-fold average or "
            "accuracy improvement is implied."
        ),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "augmentation_ablation.json", result)
    return result
