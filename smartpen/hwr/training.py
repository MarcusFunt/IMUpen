"""Reproducible CPU/GPU training and held-out evaluation for OnHW-Words500."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import tempfile
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from smartpen.hwr.data import (
    ONHW_BLANK_INDEX,
    OnHWFold,
    SequenceRecord,
    encode_text,
    load_onhw_words500,
)
from smartpen.hwr.metrics import corpus_error_rates
from smartpen.hwr.model import CNNBiLSTMCTC, ctc_greedy_decode, encoded_length


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


def _ctc_minimum_frames(tokens: Sequence[int]) -> int:
    """Minimum CTC frames: one per label plus one between repeated labels."""
    return len(tokens) + sum(left == right for left, right in zip(tokens, tokens[1:]))


def keep_ctc_feasible(records: Sequence[SequenceRecord]) -> tuple[list[SequenceRecord], int]:
    kept = []
    for record in records:
        tokens = encode_text(record.text)
        if encoded_length(len(record.features)) >= _ctc_minimum_frames(tokens):
            kept.append(record)
    return kept, len(records) - len(kept)


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
    sums = np.zeros(6, dtype=np.float64)
    sum_squares = np.zeros(6, dtype=np.float64)
    count = 0
    for record in records:
        values = np.asarray(record.features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
            raise ValueError("each training sequence must have shape (timesteps, 6)")
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


def _collate_records(mean: np.ndarray, std: np.ndarray):
    def collate(records: list[SequenceRecord]) -> dict[str, torch.Tensor | list]:
        lengths = torch.tensor([len(record.features) for record in records], dtype=torch.long)
        max_length = int(lengths.max())
        inputs = torch.zeros((len(records), max_length, 6), dtype=torch.float32)
        token_sequences = [encode_text(record.text) for record in records]
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
        }

    return collate


def _loader(
    records: Sequence[SequenceRecord],
    batch_size: int,
    seed: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[DataLoader, _LengthBucketBatchSampler]:
    sampler = _LengthBucketBatchSampler(records, batch_size, seed)
    loader = DataLoader(
        _RecordDataset(records),
        batch_sampler=sampler,
        collate_fn=_collate_records(mean, std),
        num_workers=0,
    )
    return loader, sampler


def _run_epoch(
    model: CNNBiLSTMCTC,
    loader: DataLoader,
    loss_function: nn.CTCLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[str], list[str]]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    sample_count = 0
    predictions: list[str] = []
    references: list[str] = []
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            inputs = batch["inputs"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits, output_lengths = model(inputs, lengths)
            log_probabilities = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = loss_function(
                log_probabilities,
                targets,
                output_lengths.to(device="cpu"),
                target_lengths.to(device="cpu"),
            )
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_size = len(batch["texts"])
            total_loss += float(loss.detach()) * batch_size
            sample_count += batch_size
            token_ids = logits.detach().argmax(dim=-1).cpu()
            for index, reference in enumerate(batch["texts"]):
                output_length = int(output_lengths[index].detach().cpu())
                predictions.append(
                    ctc_greedy_decode(
                        token_ids[index, :output_length].tolist(), ONHW_BLANK_INDEX
                    )
                )
                references.append(reference)
    return total_loss / max(sample_count, 1), predictions, references


def _sha256(path: Path) -> str | None:
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
    model_config: dict | None = None,
) -> dict:
    """Train and evaluate a CTC model on an official OnHW writer-independent fold."""
    if epochs < 1 or batch_size < 1 or num_threads < 1:
        raise ValueError("epochs, batch_size, and num_threads must be positive")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    seed_everything(seed, device.type)
    torch.set_num_threads(num_threads)

    fold_data: OnHWFold = load_onhw_words500(data_path, fold=fold)
    train_writers = fold_data.train_writers
    validation_writers = fold_data.validation_writers
    if not train_writers.isdisjoint(validation_writers):
        raise ValueError("writer leakage between official train and validation partitions")

    feasible_train, dropped_train_ctc = keep_ctc_feasible(fold_data.train)
    feasible_validation, dropped_validation_ctc = keep_ctc_feasible(fold_data.validation)
    train_records, train_indices = select_records(feasible_train, max_train_samples, seed)
    validation_records, validation_indices = select_records(
        feasible_validation, max_validation_samples, seed + 1
    )
    if not train_records or not validation_records:
        raise ValueError("training and validation subsets must both be non-empty")
    mean, std = feature_normalization(train_records)

    config = {
        "num_classes": ONHW_BLANK_INDEX + 1,
        "input_channels": 6,
        "conv_channels": 32,
        "hidden_size": 64,
        "lstm_layers": 2,
        "dropout": 0.1,
    }
    if model_config:
        config.update(model_config)
    model = CNNBiLSTMCTC(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.CTCLoss(blank=ONHW_BLANK_INDEX, zero_infinity=True)
    train_loader, train_sampler = _loader(train_records, batch_size, seed, mean, std)
    validation_loader, _ = _loader(
        validation_records, batch_size, seed + 1, mean, std
    )

    epoch_results: list[dict] = []
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_loss, _, _ = _run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss, hypotheses, references = _run_epoch(
            model, validation_loader, loss_function, device
        )
        metrics = corpus_error_rates(hypotheses, references)
        epoch_results.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_cer": metrics["cer"],
                "validation_wer": metrics["wer"],
                "validation_samples": len(references),
            }
        )

    report = {
        "schema_version": 1,
        "run_kind": "sample" if max_train_samples or max_validation_samples else "full",
        "dataset": "OnHW-Words500 writer-independent",
        "source_archive": Path(data_path).name,
        "archive_sha256": _sha256(Path(data_path)),
        "fold": fold,
        "split": "official train/validation split; writer IDs are disjoint",
        "train_samples_in_archive": len(fold_data.train) + fold_data.dropped_train,
        "validation_samples_in_archive": (
            len(fold_data.validation) + fold_data.dropped_validation
        ),
        "train_samples_after_empty_filter": len(fold_data.train),
        "validation_samples_after_empty_filter": len(fold_data.validation),
        "train_samples_ctc_feasible": len(feasible_train),
        "validation_samples_ctc_feasible": len(feasible_validation),
        "train_samples_used": len(train_records),
        "validation_samples_used": len(validation_records),
        "train_writers_available": len(train_writers),
        "validation_writers_available": len(validation_writers),
        "train_writers_sampled": len({record.writer_id for record in train_records}),
        "validation_writers_sampled": len(
            {record.writer_id for record in validation_records}
        ),
        "writer_leakage": False,
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
        "seed": seed,
        "num_threads": num_threads,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "device": str(device),
        "deterministic_algorithms": True,
        "normalization": {
            "source": "selected training samples only",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "model": config,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "ctc": {"blank_index": ONHW_BLANK_INDEX, "zero_infinity": True},
        "epochs": epoch_results,
        "final_metrics": {
            "cer": epoch_results[-1]["validation_cer"],
            "wer": epoch_results[-1]["validation_wer"],
        },
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Measured on this run's sampled held-out writers; not a full-fold "
            "benchmark and not directly comparable to published results."
            if report_is_sample(max_train_samples, max_validation_samples)
            else "Measured on the full official fold; no cross-fold average is implied."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": config,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß",
        "blank_index": ONHW_BLANK_INDEX,
        "seed": seed,
        "fold": fold,
        "training_report": report,
    }
    checkpoint_path = output / "model.pt"
    fd, temporary = tempfile.mkstemp(prefix="tmp-model-", suffix=".pt", dir=output)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, checkpoint_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report["checkpoint"] = checkpoint_path.name
    _atomic_json(output / "metrics.json", report)
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
) -> dict:
    """Evaluate a saved checkpoint on its held-out official writer split."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    selected_fold = int(checkpoint["fold"] if fold is None else fold)
    selected_seed = int(checkpoint["seed"] if seed is None else seed)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    seed_everything(selected_seed, device.type)
    torch.set_num_threads(num_threads)
    fold_data = load_onhw_words500(data_path, selected_fold)
    records, dropped_ctc = keep_ctc_feasible(fold_data.validation)
    records, indices = select_records(records, max_validation_samples, selected_seed + 1)
    model = CNNBiLSTMCTC(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    mean = checkpoint["normalization_mean"].numpy().astype(np.float32)
    std = checkpoint["normalization_std"].numpy().astype(np.float32)
    loader, _ = _loader(records, batch_size, selected_seed + 1, mean, std)
    loss_function = nn.CTCLoss(blank=ONHW_BLANK_INDEX, zero_infinity=True)
    loss, hypotheses, references = _run_epoch(model, loader, loss_function, device)
    result = {
        "dataset": "OnHW-Words500 writer-independent",
        "fold": selected_fold,
        "validation_samples": len(records),
        "validation_writers": len({record.writer_id for record in records}),
        "writer_leakage": not fold_data.train_writers.isdisjoint(
            fold_data.validation_writers
        ),
        "ctc_infeasible_recordings_dropped": dropped_ctc,
        "sample_indices": indices,
        "validation_loss": loss,
        "metrics": corpus_error_rates(hypotheses, references),
        "seed": selected_seed,
        "device": str(device),
    }
    return result
