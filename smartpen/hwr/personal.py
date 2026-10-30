"""OnHW encoder pretraining and session-held-out personal fine-tuning."""
from __future__ import annotations

import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from smartpen.hwr.data import (
    PERSONAL_CHARSET,
    PersonalSessionSplit,
    SequenceRecord,
    load_personal_recordings,
    split_personal_by_session,
    validate_charset,
)
from smartpen.hwr.metrics import corpus_error_rates
from smartpen.hwr.model import (
    build_model,
    count_parameters,
    default_model_config,
)
from smartpen.hwr.training import (
    _atomic_json,
    _loader,
    _make_grad_scaler,
    _run_epoch,
    _sha256,
    feature_normalization,
    keep_ctc_feasible,
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


def _train_personal_variant(
    *,
    model_type: str,
    initialization: str,
    pretrained_checkpoint: Path,
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
        transfer_metadata = load_pretrained_encoder(model, pretrained_checkpoint)
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
            pretrained_checkpoint=pretrained_path,
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
