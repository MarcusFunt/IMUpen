"""OnHW encoder pretraining and session-held-out personal fine-tuning."""
from __future__ import annotations

import json
import os
import platform
import tempfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from smartpen.hwr.augment import apply_augmentation
from smartpen.hwr.data import (
    PERSONAL_CHARSET,
    PersonalSessionSplit,
    SequenceRecord,
    load_personal_recordings,
    load_unlabeled_personal_recordings,
    split_personal_by_session,
    validate_charset,
)
from smartpen.hwr.metrics import corpus_error_rates
from smartpen.hwr.model import (
    build_model,
    count_parameters,
    default_model_config,
    encoded_length,
)
from smartpen.hwr.ts2vec import (
    TS2VecProjectionHead,
    context_crop_bounds,
    hierarchical_contrastive_loss,
    mask_timestamps,
)
from smartpen.hwr.training import (
    _LengthBucketBatchSampler,
    _atomic_json,
    _loader,
    _make_grad_scaler,
    _run_epoch,
    _sha256,
    feature_normalization,
    keep_ctc_feasible,
    select_records,
    run_training,
    seed_everything,
)


_ENCODER_PREFIXES = ("stem.", "residual_blocks.", "recurrent.")
_ENCODER_CONFIG_KEYS = ("input_channels", "channels", "hidden_size", "lstm_layers")


def run_onhw_encoder_pretraining(
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
    model_config: dict | None = None,
) -> dict:
    """Train the M5 encoder on OnHW and mark its checkpoint for transfer."""
    return run_training(
        data_path=data_path,
        output_dir=output_dir,
        fold=fold,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device_name=device_name,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        num_threads=num_threads,
        model_type="resnet_bilstm_ctc",
        mixed_precision=mixed_precision,
        training_stage="onhw_encoder_pretraining",
        model_config=model_config,
    )


def load_pretrained_encoder(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Copy compatible OnHW CNN/BiLSTM weights without transferring the CTC head."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_type = checkpoint.get("model_type", "cnn_bilstm_ctc")
    if model_type != "resnet_bilstm_ctc":
        raise ValueError("M6-A requires an OnHW ResNet-BiLSTM-CTC checkpoint")
    training_report = checkpoint.get("training_report", {})
    if training_report.get("dataset") != "OnHW-Words500 writer-independent":
        raise ValueError("pretraining checkpoint is not from OnHW-Words500")
    source_config = checkpoint.get("model_config", {})
    target_config = getattr(model, "_m6_model_config", None)
    if target_config is not None:
        mismatched = [
            key
            for key in _ENCODER_CONFIG_KEYS
            if source_config.get(key) != target_config.get(key)
        ]
        if mismatched:
            raise ValueError(
                "pretrained encoder shape mismatch for: " + ", ".join(mismatched)
            )

    source_state = checkpoint.get("model_state", {})
    encoder_state = {
        key: value
        for key, value in source_state.items()
        if key.startswith(_ENCODER_PREFIXES)
    }
    expected_encoder_keys = {
        key for key in model.state_dict() if key.startswith(_ENCODER_PREFIXES)
    }
    if set(encoder_state) != expected_encoder_keys:
        raise ValueError("pretrained checkpoint encoder layers do not match the model")
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    expected_missing = set(model.state_dict()) - expected_encoder_keys
    if set(missing) != expected_missing or unexpected:
        raise ValueError("failed to load all encoder weights from the pretraining checkpoint")

    return {
        "checkpoint_file": Path(checkpoint_path).name,
        "checkpoint_sha256": _sha256(Path(checkpoint_path)),
        "source_training_stage": checkpoint.get("training_stage", "unknown"),
        "source_archive_sha256": training_report.get("archive_sha256"),
        "source_fold": checkpoint.get("fold"),
        "loaded_encoder_tensors": len(encoder_state),
        "classifier_head_transferred": False,
    }


def load_personal_ts2vec_encoder(
    model: nn.Module, checkpoint_path: str | Path
) -> dict[str, Any]:
    """Copy the personal TS2Vec encoder while leaving both projection heads out."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("training_stage") != "personal_ts2vec_pretraining":
        raise ValueError("checkpoint is not from personal TS2Vec pretraining")
    source_config = checkpoint.get("model_config", {})
    target_config = getattr(model, "_m6_model_config", None)
    if target_config is not None:
        mismatched = [
            key
            for key in _ENCODER_CONFIG_KEYS
            if source_config.get(key) != target_config.get(key)
        ]
        if mismatched:
            raise ValueError("TS2Vec encoder shape mismatch for: " + ", ".join(mismatched))

    encoder_state = checkpoint.get("encoder_state", {})
    expected_keys = {
        key for key in model.state_dict() if key.startswith(_ENCODER_PREFIXES)
    }
    if set(encoder_state) != expected_keys:
        raise ValueError("TS2Vec checkpoint encoder layers do not match the model")
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    expected_missing = set(model.state_dict()) - expected_keys
    if set(missing) != expected_missing or unexpected:
        raise ValueError("failed to load all personal TS2Vec encoder weights")
    training_report = checkpoint.get("training_report", {})
    return {
        "checkpoint_file": Path(checkpoint_path).name,
        "checkpoint_sha256": _sha256(Path(checkpoint_path)),
        "source_training_stage": checkpoint["training_stage"],
        "self_supervised_loss": training_report.get("final_loss"),
        "self_supervised_record_count": training_report.get("training_record_count"),
        "loaded_encoder_tensors": len(encoder_state),
        "classifier_head_transferred": False,
        "projection_head_transferred": False,
    }


def _session_manifest(
    split: PersonalSessionSplit, *, seed: int, validation_fraction: float
) -> dict:
    def groups(values: frozenset[tuple[str, str]]) -> list[dict[str, str]]:
        return [
            {"writer_id": writer_id, "session_id": session_id}
            for writer_id, session_id in sorted(values)
        ]

    return {
        "schema_version": 1,
        "split_policy": "writer/session groups held out from training",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_sessions": groups(split.train_sessions),
        "validation_sessions": groups(split.validation_sessions),
        "train_sample_ids": sorted(record.sample_id for record in split.train),
        "validation_sample_ids": sorted(record.sample_id for record in split.validation),
    }


def _apply_session_manifest(
    records: list[SequenceRecord], manifest_path: str | Path
) -> PersonalSessionSplit:
    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid personal split manifest: {manifest_file}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported personal split manifest schema")

    train_id_list = manifest.get("train_sample_ids", [])
    validation_id_list = manifest.get("validation_sample_ids", [])
    if not isinstance(train_id_list, list) or not isinstance(validation_id_list, list):
        raise ValueError("personal split manifest sample IDs must be lists")
    train_ids = set(train_id_list)
    validation_ids = set(validation_id_list)
    if len(train_ids) != len(train_id_list) or len(validation_ids) != len(validation_id_list):
        raise ValueError("personal split manifest contains duplicate sample IDs")
    all_ids = {record.sample_id for record in records}
    if train_ids & validation_ids:
        raise ValueError("personal split manifest places a recording in both partitions")
    if train_ids | validation_ids != all_ids:
        raise ValueError("personal split manifest does not exactly cover current recordings")

    train = [record for record in records if record.sample_id in train_ids]
    validation = [record for record in records if record.sample_id in validation_ids]
    train_sessions = frozenset(
        (str(record.writer_id), str(record.session_id)) for record in train
    )
    validation_sessions = frozenset(
        (str(record.writer_id), str(record.session_id)) for record in validation
    )
    if train_sessions & validation_sessions:
        raise ValueError("personal split manifest leaks a writer/session group")
    try:
        listed_train_sessions = frozenset(
            (str(group["writer_id"]), str(group["session_id"]))
            for group in manifest["train_sessions"]
        )
        listed_validation_sessions = frozenset(
            (str(group["writer_id"]), str(group["session_id"]))
            for group in manifest["validation_sessions"]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("personal split manifest has invalid session groups") from exc
    if listed_train_sessions != train_sessions or listed_validation_sessions != validation_sessions:
        raise ValueError("personal split manifest session groups do not match sample IDs")
    if not train or not validation:
        raise ValueError("personal split manifest must include both partitions")
    return PersonalSessionSplit(train, validation, train_sessions, validation_sessions)


def _pad_feature_views(
    values: list[np.ndarray], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(value) for value in values], dtype=torch.long)
    padded = torch.zeros((len(values), int(lengths.max()), 6), dtype=torch.float32)
    for index, value in enumerate(values):
        padded[index, : len(value)] = torch.from_numpy(
            np.ascontiguousarray(value, dtype=np.float32)
        )
    return padded.to(device), lengths.to(device)


def _make_personal_context_views(
    records: list[SequenceRecord],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Build aligned two-context views using raw-unit transforms before scaling."""
    first_views: list[np.ndarray] = []
    second_views: list[np.ndarray] = []
    overlaps: list[int] = []
    feature_std = np.maximum(std, 1e-6)
    for record in records:
        start1, end1, start2, end2, overlap = context_crop_bounds(
            len(record.features), rng
        )
        contexts = [
            record.features[start1:end1],
            record.features[start2:end2],
        ]
        normalized_views = []
        for context in contexts:
            view = SequenceRecord(
                features=np.ascontiguousarray(context, dtype=np.float32),
                text="",
                writer_id=record.writer_id,
                sample_index=record.sample_index,
                session_id=record.session_id,
                sample_id=record.sample_id,
            )
            for transform in ("gaussian_noise", "sensor_bias_gain"):
                view = apply_augmentation(view, transform, rng, feature_std)
            if rng.random() < 0.5:
                view = apply_augmentation(view, "rotation_3d", rng, feature_std)
            normalized = (view.features - mean) / std
            normalized_views.append(mask_timestamps(normalized, rng))
        first_views.append(normalized_views[0])
        second_views.append(normalized_views[1])
        overlaps.append(overlap)
    return first_views, second_views, overlaps


def _run_personal_ts2vec_pretraining(
    *,
    records: list[SequenceRecord],
    output_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    model_config: dict,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    num_threads: int,
    mixed_precision: bool,
    eligible_sessions: list[dict[str, str]],
    excluded_validation_samples: int,
    rejected_recordings: int,
) -> dict:
    if not records:
        raise ValueError("personal TS2Vec pretraining has no eligible unlabeled sequences")
    short_records = [record for record in records if encoded_length(len(record.features)) < 2]
    records = [record for record in records if encoded_length(len(record.features)) >= 2]
    if not records:
        raise ValueError("all unlabeled sequences are too short for TS2Vec temporal views")
    if epochs < 1 or batch_size < 1 or num_threads < 1:
        raise ValueError("TS2Vec epochs, batch size, and thread count must be positive")

    seed_everything(seed, device.type)
    torch.set_num_threads(num_threads)
    backbone = build_model("resnet_bilstm_ctc", model_config).to(device)
    for parameter in backbone.classifier.parameters():
        parameter.requires_grad_(False)
    projector = TS2VecProjectionHead(
        input_dim=int(model_config["hidden_size"]) * 2,
        hidden_dim=128,
        output_dim=128,
    ).to(device)
    parameters = [
        *(parameter for parameter in backbone.parameters() if parameter.requires_grad),
        *projector.parameters(),
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    scaler = _make_grad_scaler(mixed_precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sampler = _LengthBucketBatchSampler(records, batch_size, seed)
    loader = DataLoader(
        records,
        batch_sampler=sampler,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )
    epoch_results = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        backbone.train()
        projector.train()
        total_loss = 0.0
        batch_count = 0
        for batch_index, batch_records in enumerate(loader):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, epoch, batch_index])
            )
            view1, view2, overlaps = _make_personal_context_views(
                batch_records, mean=mean, std=std, rng=rng
            )
            inputs1, lengths1 = _pad_feature_views(view1, device)
            inputs2, lengths2 = _pad_feature_views(view2, device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision
                else nullcontext()
            )
            with autocast:
                features1, encoded_lengths1 = backbone.encode_temporal(inputs1, lengths1)
                features2, encoded_lengths2 = backbone.encode_temporal(inputs2, lengths2)
                projected1 = projector(features1).float()
                projected2 = projector(features2).float()
            aligned1 = []
            aligned2 = []
            for index, overlap in enumerate(overlaps):
                length1 = int(encoded_lengths1[index].item())
                length2 = int(encoded_lengths2[index].item())
                if overlap > length1 or overlap > length2:
                    raise RuntimeError("context crops did not preserve their aligned segment")
                aligned1.append(projected1[index, length1 - overlap : length1])
                aligned2.append(projected2[index, :overlap])
            max_overlap = max(overlaps)
            contrast1 = torch.zeros(
                (len(overlaps), max_overlap, projected1.shape[-1]),
                dtype=projected1.dtype,
                device=device,
            )
            contrast2 = torch.zeros_like(contrast1)
            for index, overlap in enumerate(overlaps):
                contrast1[index, :overlap] = aligned1[index]
                contrast2[index, :overlap] = aligned2[index]
            overlap_lengths = torch.tensor(overlaps, dtype=torch.long, device=device)
            loss = hierarchical_contrastive_loss(
                contrast1, contrast2, overlap_lengths, temperature=0.2
            )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            total_loss += float(loss.detach())
            batch_count += 1
        if batch_count == 0:
            raise ValueError("TS2Vec produced no training batches")
        epoch_results.append(
            {"epoch": epoch + 1, "contrastive_loss": total_loss / batch_count, "batches": batch_count}
        )

    report = {
        "schema_version": 1,
        "dataset": "unlabeled personal recorder recordings",
        "training_stage": "personal_ts2vec_pretraining",
        "objective": "hierarchical temporal and instance-wise contrastive loss",
        "view_generation": {
            "context_crop_stride": 4,
            "minimum_shared_encoded_timesteps": 2,
            "timestamp_mask_probability": 0.15,
            "gaussian_noise_axis_std_fraction": 0.01,
            "sensor_gain_range": [0.97, 1.03],
            "sensor_bias_std_fraction": 0.02,
            "small_rotation_probability": 0.5,
            "rotation_euler_degrees": [-5.0, 5.0],
        },
        "model_type": "resnet_bilstm_ctc temporal encoder",
        "model_config": model_config,
        "model_parameters": count_parameters(backbone),
        "projection_head": {"hidden_dim": 128, "output_dim": 128},
        "training_record_count": len(records),
        "training_sample_ids": sorted(str(record.sample_id) for record in records),
        "training_sessions": eligible_sessions,
        "validation_session_samples_excluded": excluded_validation_samples,
        "rejected_recordings_skipped": rejected_recordings,
        "too_short_sequences_dropped": len(short_records),
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "deterministic_algorithms": True,
        "mixed_precision": mixed_precision,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "num_threads": num_threads,
        "normalization": {
            "source": "labeled personal training sessions only; shared with both supervised runs",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "temperature": 0.2,
        "gradient_clip_norm": 5.0,
        "epochs": epoch_results,
        "final_loss": epoch_results[-1]["contrastive_loss"],
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_state = {
        key: value
        for key, value in backbone.state_dict().items()
        if key.startswith(_ENCODER_PREFIXES)
    }
    checkpoint = {
        "training_stage": "personal_ts2vec_pretraining",
        "model_type": "resnet_bilstm_ctc",
        "model_config": model_config,
        "encoder_state": encoder_state,
        "projection_state": projector.state_dict(),
        "training_report": report,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "seed": seed,
    }
    checkpoint_path = output_dir / "model.pt"
    fd, temporary = tempfile.mkstemp(prefix="tmp-model-", suffix=".pt", dir=output_dir)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, checkpoint_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report["checkpoint"] = checkpoint_path.name
    _atomic_json(output_dir / "metrics.json", report)
    return report


def _train_personal_variant(
    *,
    model_type: str,
    initialization: str,
    initialization_checkpoint: Path | None,
    train_records: list[SequenceRecord],
    validation_records: list[SequenceRecord],
    charset: str,
    blank_index: int,
    mean: np.ndarray,
    std: np.ndarray,
    split_manifest: dict,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
    num_threads: int,
    mixed_precision: bool,
    model_config_override: dict | None,
) -> dict:
    seed_everything(seed, device.type)
    torch.set_num_threads(num_threads)
    config = default_model_config(model_type, blank_index + 1)
    if model_config_override:
        config.update(model_config_override)
    model = build_model(model_type, config)
    # Used only to verify dimensional compatibility before loading transferred weights.
    model._m6_model_config = config
    transfer_metadata = None
    if initialization == "onhw_pretrained_encoder":
        if initialization_checkpoint is None:
            raise ValueError("OnHW encoder initialization requires a checkpoint")
        transfer_metadata = load_pretrained_encoder(model, initialization_checkpoint)
    elif initialization == "personal_ts2vec_encoder":
        if initialization_checkpoint is None:
            raise ValueError("personal TS2Vec initialization requires a checkpoint")
        transfer_metadata = load_personal_ts2vec_encoder(
            model, initialization_checkpoint
        )
    model = model.to(device)
    parameter_count = count_parameters(model)
    scaler = _make_grad_scaler(mixed_precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    train_loader, train_sampler = _loader(
        train_records, batch_size, seed, mean, std, charset
    )
    validation_loader, _ = _loader(
        validation_records, batch_size, seed + 1, mean, std, charset
    )
    epoch_results = []
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_loss, _, _ = _run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
            mixed_precision=mixed_precision,
            scaler=scaler,
            charset=charset,
            blank_index=blank_index,
        )
        validation_loss, hypotheses, references = _run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            mixed_precision=mixed_precision,
            charset=charset,
            blank_index=blank_index,
        )
        metrics = corpus_error_rates(hypotheses, references)
        epoch_results.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_metrics": metrics,
                "validation_samples": len(references),
            }
        )

    report = {
        "schema_version": 1,
        "dataset": "IMUpen personal recorder recordings",
        "initialization": initialization,
        "model_type": model_type,
        "model_config": config,
        "model_parameters": parameter_count,
        "charset": charset,
        "blank_index": blank_index,
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "deterministic_algorithms": True,
        "mixed_precision": mixed_precision,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "num_threads": num_threads,
        "training_sample_ids": sorted(record.sample_id for record in train_records),
        "validation_sample_ids": sorted(record.sample_id for record in validation_records),
        "training_writer_count": len({str(record.writer_id) for record in train_records}),
        "validation_writer_count": len(
            {str(record.writer_id) for record in validation_records}
        ),
        "training_sessions": split_manifest["train_sessions"],
        "validation_sessions": split_manifest["validation_sessions"],
        "session_leakage": bool(
            {
                (group["writer_id"], group["session_id"])
                for group in split_manifest["train_sessions"]
            }
            & {
                (group["writer_id"], group["session_id"])
                for group in split_manifest["validation_sessions"]
            }
        ),
        "normalization": {"source": "personal training sessions only", "mean": mean.tolist(), "std": std.tolist()},
        "pretraining": transfer_metadata,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "gradient_clip_norm": 5.0,
        "epochs": epoch_results,
        "final_metrics": epoch_results[-1]["validation_metrics"],
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_type": model_type,
        "training_stage": "personal_finetuning",
        "model_config": config,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "charset": charset,
        "blank_index": blank_index,
        "seed": seed,
        "training_report": report,
        "session_split": split_manifest,
        "pretraining": transfer_metadata,
    }
    checkpoint_path = output_dir / "model.pt"
    fd, temporary = tempfile.mkstemp(prefix="tmp-model-", suffix=".pt", dir=output_dir)
    os.close(fd)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, checkpoint_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report["checkpoint"] = checkpoint_path.name
    _atomic_json(output_dir / "metrics.json", report)
    return report


def run_personal_comparison(
    *,
    recordings_dir: str | Path,
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    charset: str = PERSONAL_CHARSET,
    validation_fraction: float = 0.2,
    split_manifest_path: str | Path | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    seed: int = 1337,
    device_name: str = "auto",
    num_threads: int = 4,
    mixed_precision: bool | None = None,
    model_config: dict | None = None,
) -> dict:
    """Compare OnHW-encoder fine-tuning with same-seed personal training from scratch."""
    if epochs < 1 or batch_size < 1 or num_threads < 1:
        raise ValueError("epochs, batch_size, and num_threads must be positive")
    charset = validate_charset(charset)
    blank_index = len(charset)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    use_mixed_precision = (
        device.type == "cuda" if mixed_precision is None else mixed_precision
    )
    if use_mixed_precision and device.type != "cuda":
        raise ValueError("mixed precision is supported only for CUDA training")

    personal_data = load_personal_recordings(recordings_dir, charset)
    if split_manifest_path is None:
        split = split_personal_by_session(
            personal_data.records,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        split_manifest = _session_manifest(
            split, seed=seed, validation_fraction=validation_fraction
        )
    else:
        split = _apply_session_manifest(personal_data.records, split_manifest_path)
        split_manifest = json.loads(
            Path(split_manifest_path).read_text(encoding="utf-8")
        )
    output = Path(output_dir)
    _atomic_json(output / "session_split.json", split_manifest)

    train_records, dropped_train = keep_ctc_feasible(split.train, charset)
    validation_records, dropped_validation = keep_ctc_feasible(split.validation, charset)
    if not train_records or not validation_records:
        raise ValueError("session split has no CTC-feasible train or validation samples")
    if not split.train_sessions.isdisjoint(split.validation_sessions):
        raise ValueError("writer/session leakage in personal split")
    mean, std = feature_normalization(train_records)
    config = default_model_config("resnet_bilstm_ctc", blank_index + 1)
    if model_config:
        config.update(model_config)

    pretrained_path = Path(pretrained_checkpoint).resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(pretrained_path)
    # Validate transfer compatibility before starting either experiment.
    probe_model = build_model("resnet_bilstm_ctc", config)
    probe_model._m6_model_config = config
    transfer_metadata = load_pretrained_encoder(probe_model, pretrained_path)
    del probe_model

    results = {}
    initializations = ("from_scratch", "onhw_pretrained_encoder")
    for initialization in initializations:
        results[initialization] = _train_personal_variant(
            model_type="resnet_bilstm_ctc",
            initialization=initialization,
            initialization_checkpoint=(
                pretrained_path if initialization == "onhw_pretrained_encoder" else None
            ),
            train_records=train_records,
            validation_records=validation_records,
            charset=charset,
            blank_index=blank_index,
            mean=mean,
            std=std,
            split_manifest=split_manifest,
            output_dir=output / initialization,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            num_threads=num_threads,
            mixed_precision=use_mixed_precision,
            model_config_override=config,
        )

    scratch = results["from_scratch"]
    pretrained = results["onhw_pretrained_encoder"]
    same_data = (
        scratch["training_sample_ids"] == pretrained["training_sample_ids"]
        and scratch["validation_sample_ids"] == pretrained["validation_sample_ids"]
        and scratch["normalization"] == pretrained["normalization"]
        and scratch["charset"] == pretrained["charset"]
    )
    if not same_data:
        raise RuntimeError("scratch and pretrained runs used different data or preprocessing")

    comparison = {
        "schema_version": 1,
        "experiment": "M6-A OnHW encoder pretraining vs personal training from scratch",
        "dataset": "IMUpen personal recorder recordings",
        "training_sample_count": len(train_records),
        "validation_sample_count": len(validation_records),
        "rejected_recordings_skipped": personal_data.rejected_recordings,
        "ctc_infeasible_recordings_dropped": {
            "train": dropped_train,
            "validation": dropped_validation,
        },
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "mixed_precision": use_mixed_precision,
        "archive_pretraining": transfer_metadata,
        "session_split_manifest": "session_split.json",
        "fairness_checks": {
            "same_personal_train_and_validation_recordings": same_data,
            "same_session_split": not bool(split.train_sessions & split.validation_sessions),
            "same_charset_and_train_only_normalization": same_data,
            "same_seed_and_optimizer_settings": True,
            "personal_session_leakage": False,
            "classifier_head_reinitialized_for_both_runs": True,
        },
        "models": {
            name: {
                "model_parameters": result["model_parameters"],
                "final_metrics": result["final_metrics"],
                "epochs": result["epochs"],
                "cuda_peak_memory_bytes": result["cuda_peak_memory_bytes"],
                "checkpoint": f"{name}/model.pt",
            }
            for name, result in results.items()
        },
        "interpretation": (
            "Personal validation metrics are measured on the held-out sessions in this "
            "recording set. They are not a public benchmark and may be unstable for small datasets."
        ),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "comparison.json", comparison)
    return comparison


def run_personal_ts2vec_comparison(
    *,
    recordings_dir: str | Path,
    unlabeled_recordings_dir: str | Path,
    output_dir: str | Path,
    charset: str = PERSONAL_CHARSET,
    validation_fraction: float = 0.2,
    split_manifest_path: str | Path | None = None,
    ssl_epochs: int = 20,
    epochs: int = 20,
    batch_size: int = 16,
    max_unlabeled_samples: int | None = None,
    seed: int = 1337,
    device_name: str = "auto",
    num_threads: int = 4,
    mixed_precision: bool | None = None,
    model_config: dict | None = None,
) -> dict:
    """Compare personal TS2Vec encoder initialization with supervised scratch.

    Unlabeled sequences from the held-out writer/session groups are removed
    before self-supervised training. Both downstream runs use the same labeled
    split, original training-only normalization, optimizer, seed, and settings.
    """
    if ssl_epochs < 1 or epochs < 1 or batch_size < 1 or num_threads < 1:
        raise ValueError("SSL epochs, fine-tune epochs, batch size, and threads must be positive")
    if max_unlabeled_samples is not None and max_unlabeled_samples < 1:
        raise ValueError("max_unlabeled_samples must be positive")
    charset = validate_charset(charset)
    blank_index = len(charset)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    use_mixed_precision = (
        device.type == "cuda" if mixed_precision is None else mixed_precision
    )
    if use_mixed_precision and device.type != "cuda":
        raise ValueError("mixed precision is supported only for CUDA training")

    labeled = load_personal_recordings(recordings_dir, charset)
    if split_manifest_path is None:
        split = split_personal_by_session(
            labeled.records,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        split_manifest = _session_manifest(
            split, seed=seed, validation_fraction=validation_fraction
        )
    else:
        split = _apply_session_manifest(labeled.records, split_manifest_path)
        split_manifest = json.loads(
            Path(split_manifest_path).read_text(encoding="utf-8")
        )

    train_records, dropped_train = keep_ctc_feasible(split.train, charset)
    validation_records, dropped_validation = keep_ctc_feasible(
        split.validation, charset
    )
    if not train_records or not validation_records:
        raise ValueError("session split has no CTC-feasible train or validation samples")
    if not split.train_sessions.isdisjoint(split.validation_sessions):
        raise ValueError("writer/session leakage in personal split")
    mean, std = feature_normalization(train_records)

    unlabeled = load_unlabeled_personal_recordings(unlabeled_recordings_dir)
    validation_groups = set(split.validation_sessions)
    training_writers = {str(record.writer_id) for record in split.train}
    validation_overlap = [
        record
        for record in unlabeled.records
        if (str(record.writer_id), str(record.session_id)) in validation_groups
    ]
    other_writer_samples = [
        record
        for record in unlabeled.records
        if (str(record.writer_id), str(record.session_id)) not in validation_groups
        and str(record.writer_id) not in training_writers
    ]
    eligible_ssl_records = [
        record
        for record in unlabeled.records
        if (str(record.writer_id), str(record.session_id)) not in validation_groups
        and str(record.writer_id) in training_writers
    ]
    if not eligible_ssl_records:
        raise ValueError(
            "no unlabeled recordings belong to a labeled training writer outside "
            "held-out validation sessions; provide eligible training data"
        )
    ssl_records, ssl_indices = select_records(
        eligible_ssl_records, max_unlabeled_samples, seed + 2
    )
    if not ssl_records:
        raise ValueError("no unlabeled recordings selected for TS2Vec")

    output = Path(output_dir)
    _atomic_json(output / "session_split.json", split_manifest)
    config = default_model_config("resnet_bilstm_ctc", blank_index + 1)
    if model_config:
        config.update(model_config)
    eligible_ssl_sessions = [
        {"writer_id": writer_id, "session_id": session_id}
        for writer_id, session_id in sorted(
            {(str(record.writer_id), str(record.session_id)) for record in ssl_records}
        )
    ]
    ssl_report = _run_personal_ts2vec_pretraining(
        records=ssl_records,
        output_dir=output / "personal_ts2vec_pretraining",
        mean=mean,
        std=std,
        model_config=config,
        seed=seed,
        epochs=ssl_epochs,
        batch_size=batch_size,
        device=device,
        num_threads=num_threads,
        mixed_precision=use_mixed_precision,
        eligible_sessions=eligible_ssl_sessions,
        excluded_validation_samples=len(validation_overlap),
        rejected_recordings=unlabeled.rejected_recordings,
    )

    results = {}
    initializations = ("from_scratch", "personal_ts2vec_encoder")
    for initialization in initializations:
        results[initialization] = _train_personal_variant(
            model_type="resnet_bilstm_ctc",
            initialization=initialization,
            initialization_checkpoint=(
                output / "personal_ts2vec_pretraining" / "model.pt"
                if initialization == "personal_ts2vec_encoder"
                else None
            ),
            train_records=train_records,
            validation_records=validation_records,
            charset=charset,
            blank_index=blank_index,
            mean=mean,
            std=std,
            split_manifest=split_manifest,
            output_dir=output / initialization,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            num_threads=num_threads,
            mixed_precision=use_mixed_precision,
            model_config_override=config,
        )

    scratch = results["from_scratch"]
    ts2vec = results["personal_ts2vec_encoder"]
    same_data = (
        scratch["training_sample_ids"] == ts2vec["training_sample_ids"]
        and scratch["validation_sample_ids"] == ts2vec["validation_sample_ids"]
        and scratch["normalization"] == ts2vec["normalization"]
        and scratch["charset"] == ts2vec["charset"]
    )
    ssl_normalization_shared = (
        ssl_report["normalization"]["mean"] == scratch["normalization"]["mean"]
        and ssl_report["normalization"]["std"] == scratch["normalization"]["std"]
    )
    ssl_has_no_validation_sessions = not bool(
        set(split.validation_sessions)
        & {
            (str(record.writer_id), str(record.session_id))
            for record in ssl_records
        }
    )
    if not same_data or not ssl_has_no_validation_sessions or not ssl_normalization_shared:
        raise RuntimeError("M6-C comparison failed its shared-data/session checks")

    comparison = {
        "schema_version": 1,
        "experiment": "M6-C personal TS2Vec self-supervised encoder vs supervised scratch",
        "dataset": "IMUpen personal recorder recordings",
        "training_sample_count": len(train_records),
        "validation_sample_count": len(validation_records),
        "unlabeled_training_sample_count": len(ssl_records),
        "unlabeled_training_sample_indices": ssl_indices,
        "unlabeled_validation_session_samples_excluded": len(validation_overlap),
        "unlabeled_other_writer_samples_excluded": len(other_writer_samples),
        "rejected_labeled_recordings_skipped": labeled.rejected_recordings,
        "rejected_unlabeled_recordings_skipped": unlabeled.rejected_recordings,
        "ctc_infeasible_recordings_dropped": {
            "train": dropped_train,
            "validation": dropped_validation,
        },
        "seed": seed,
        "ssl_epochs": ssl_epochs,
        "fine_tune_epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "mixed_precision": use_mixed_precision,
        "session_split_manifest": "session_split.json",
        "self_supervised_report": "personal_ts2vec_pretraining/metrics.json",
        "fairness_checks": {
            "same_labeled_train_and_validation_recordings": same_data,
            "same_session_split": not bool(split.train_sessions & split.validation_sessions),
            "same_charset_and_training_only_normalization": same_data,
            "self_supervised_and_supervised_training_share_normalization": ssl_normalization_shared,
            "same_seed_optimizer_and_supervised_settings": True,
            "validation_sessions_excluded_from_self_supervised_pretraining": ssl_has_no_validation_sessions,
            "classifier_and_projection_heads_reinitialized_or_dropped": True,
            "personal_session_leakage": False,
        },
        "self_supervised_pretraining": {
            "checkpoint": "personal_ts2vec_pretraining/model.pt",
            "final_loss": ssl_report["final_loss"],
            "training_sample_count": ssl_report["training_record_count"],
            "training_sessions": eligible_ssl_sessions,
        },
        "models": {
            name: {
                "model_parameters": result["model_parameters"],
                "final_metrics": result["final_metrics"],
                "epochs": result["epochs"],
                "cuda_peak_memory_bytes": result["cuda_peak_memory_bytes"],
                "checkpoint": f"{name}/model.pt",
            }
            for name, result in results.items()
        },
        "interpretation": (
            "Metrics are measured on the labeled held-out sessions in this recording set. "
            "No personal recordings are bundled, and small-data results may be unstable."
        ),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "comparison.json", comparison)
    return comparison
