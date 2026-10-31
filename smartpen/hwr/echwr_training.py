"""Session-held-out M6-D personal ECHWR-style training comparisons."""
from __future__ import annotations

import json
import os
import platform
import tempfile
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from smartpen.hwr.augment import AUGMENTATION_NAMES, augmentation_plan, concatenate_training_records
from smartpen.hwr.data import (
    PERSONAL_CHARSET,
    SequenceRecord,
    load_personal_recordings,
    split_personal_by_session,
    validate_charset,
)
from smartpen.hwr.echwr import (
    ECHWRAuxiliaryBranch,
    error_based_contrastive_loss,
    generate_edit_distance_one_negatives,
    in_batch_contrastive_loss,
)
from smartpen.hwr.metrics import corpus_error_rates
from smartpen.hwr.model import build_model, count_parameters, default_model_config
from smartpen.hwr.personal import _apply_session_manifest, _session_manifest, load_pretrained_encoder
from smartpen.hwr.training import (
    _atomic_json,
    _loader,
    _make_grad_scaler,
    _run_epoch,
    _sha256,
    feature_normalization,
    keep_ctc_feasible,
    select_records,
    seed_everything,
)


def _train_echwr_variant(
    *,
    name: str,
    use_batch_contrastive: bool,
    use_error_contrastive: bool,
    initialization_checkpoint: Path | None,
    fit_records: list[SequenceRecord],
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
    augmentation: str,
    batch_contrastive_weight: float,
    error_contrastive_weight: float,
    negative_sets: int,
    model_config: dict,
) -> dict:
    seed_everything(seed, device.type)
    torch.set_num_threads(num_threads)
    model = build_model("resnet_bilstm_ctc", model_config)
    model._m6_model_config = model_config
    transfer_metadata = None
    if initialization_checkpoint is not None:
        transfer_metadata = load_pretrained_encoder(model, initialization_checkpoint)
    model = model.to(device)

    auxiliary = None
    if use_batch_contrastive or use_error_contrastive:
        maximum_text_length = max(len(record.text) for record in fit_records) + 1
        auxiliary = ECHWRAuxiliaryBranch(
            input_dim=int(model_config["hidden_size"]) * 2,
            charset=charset,
            max_text_characters=maximum_text_length,
        ).to(device)
    optimizer_parameters = list(model.parameters())
    if auxiliary is not None:
        optimizer_parameters.extend(auxiliary.parameters())
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=1e-3, weight_decay=1e-4)
    scaler = _make_grad_scaler(mixed_precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    transforms, add_concatenations = augmentation_plan(augmentation)
    train_loader, train_sampler = _loader(
        fit_records,
        batch_size,
        seed,
        mean,
        std,
        charset,
        augmentation_transforms=transforms,
        augmentation_seed=seed,
    )
    validation_loader, _ = _loader(
        validation_records, batch_size, seed + 1, mean, std, charset
    )
    loss_function = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    epoch_results = []
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        train_loader.dataset.set_epoch(epoch)
        model.train()
        if auxiliary is not None:
            auxiliary.train()
        totals = {"total": 0.0, "ctc": 0.0, "batch_contrastive": 0.0, "error_contrastive": 0.0}
        sample_count = 0
        contrastive_active_batches = 0
        contrastive_skipped_batches = 0

        for batch_index, batch in enumerate(train_loader):
            inputs = batch["inputs"].to(device)
            lengths = batch["lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            texts = batch["texts"]
            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision
                else nullcontext()
            )
            with autocast_context:
                temporal_features, output_lengths = model.encode_temporal(inputs, lengths)
                logits = model.classifier(model.dropout(temporal_features))
                if auxiliary is not None:
                    sensor_embeddings = auxiliary.encode_sensor(
                        temporal_features, output_lengths
                    )
                    positive_embeddings = auxiliary.encode_texts(texts, device)
                else:
                    sensor_embeddings = None
                    positive_embeddings = None

            log_probabilities = logits.float().log_softmax(dim=-1).transpose(0, 1)
            ctc_loss = loss_function(
                log_probabilities,
                targets,
                output_lengths.to(device="cpu"),
                target_lengths.to(device="cpu"),
            )
            batch_loss = ctc_loss
            batch_contrastive = ctc_loss.new_zeros(())
            error_contrastive = ctc_loss.new_zeros(())

            if auxiliary is not None and use_batch_contrastive:
                batch_contrastive = in_batch_contrastive_loss(
                    sensor_embeddings,
                    positive_embeddings,
                    texts,
                    auxiliary.scale(),
                )
                if len(set(texts)) >= 2:
                    contrastive_active_batches += 1
                else:
                    contrastive_skipped_batches += 1
                batch_loss = batch_loss + batch_contrastive_weight * batch_contrastive

            if auxiliary is not None and use_error_contrastive:
                rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, batch_index]))
                negative_texts = [
                    generate_edit_distance_one_negatives(
                        text, charset, rng, sets=negative_sets
                    )
                    for text in texts
                ]
                negative_count = len(negative_texts[0])
                negative_embeddings = auxiliary.encode_texts(
                    [negative for sample in negative_texts for negative in sample], device
                ).reshape(len(texts), negative_count, -1)
                error_contrastive = error_based_contrastive_loss(
                    sensor_embeddings,
                    positive_embeddings,
                    negative_embeddings,
                    auxiliary.scale(),
                )
                batch_loss = batch_loss + error_contrastive_weight * error_contrastive

            if scaler.is_enabled():
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
            else:
                batch_loss.backward()
            nn.utils.clip_grad_norm_(optimizer_parameters, max_norm=5.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            size = len(texts)
            sample_count += size
            totals["total"] += float(batch_loss.detach()) * size
            totals["ctc"] += float(ctc_loss.detach()) * size
            totals["batch_contrastive"] += float(batch_contrastive.detach()) * size
            totals["error_contrastive"] += float(error_contrastive.detach()) * size

        validation_loss, hypotheses, references = _run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            mixed_precision=mixed_precision,
            charset=charset,
            blank_index=blank_index,
        )
        epoch_results.append(
            {
                "epoch": epoch + 1,
                "train_total_loss": totals["total"] / max(sample_count, 1),
                "train_ctc_loss": totals["ctc"] / max(sample_count, 1),
                "train_batch_contrastive_loss": totals["batch_contrastive"] / max(sample_count, 1),
                "train_error_contrastive_loss": totals["error_contrastive"] / max(sample_count, 1),
                "validation_ctc_loss": validation_loss,
                "validation_metrics": corpus_error_rates(hypotheses, references),
                "validation_samples": len(references),
                "batch_contrastive_active_batches": contrastive_active_batches,
                "batch_contrastive_skipped_batches": contrastive_skipped_batches,
            }
        )

    report = {
        "schema_version": 1,
        "experiment": "M6-D ECHWR-style personal fine-tuning",
        "model_variant": name,
        "dataset": "IMUpen personal recorder recordings",
        "initialization": "onhw_pretrained_encoder" if transfer_metadata else "from_scratch",
        "pretraining": transfer_metadata,
        "model_type": "resnet_bilstm_ctc",
        "model_config": model_config,
        "model_parameters": count_parameters(model),
        "auxiliary_training_parameters": count_parameters(auxiliary) if auxiliary is not None else 0,
        "auxiliary_branch_included_in_inference": False,
        "charset": charset,
        "blank_index": blank_index,
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "deterministic_algorithms": True,
        "mixed_precision": mixed_precision,
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "num_threads": num_threads,
        "training_sample_ids": sorted(record.sample_id for record in fit_records),
        "validation_sample_ids": sorted(record.sample_id for record in validation_records),
        "training_sessions": split_manifest["train_sessions"],
        "validation_sessions": split_manifest["validation_sessions"],
        "session_leakage": False,
        "normalization": {
            "source": "original personal training sessions only",
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "augmentation": {
            "name": augmentation,
            "per_record_transforms": list(transforms),
            "concatenations_enabled": add_concatenations,
            "applied_only_to_training": True,
        },
        "contrastive": {
            "batch_contrastive_enabled": use_batch_contrastive,
            "error_contrastive_enabled": use_error_contrastive,
            "batch_contrastive_weight": batch_contrastive_weight if use_batch_contrastive else 0.0,
            "error_contrastive_weight": error_contrastive_weight if use_error_contrastive else 0.0,
            "synthetic_edit_negative_sets_per_sample": negative_sets if use_error_contrastive else 0,
        },
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "gradient_clip_norm": 5.0,
        "epochs": epoch_results,
        "final_metrics": epoch_results[-1]["validation_metrics"],
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "model_type": "resnet_bilstm_ctc",
        "training_stage": "personal_echwr_finetuning",
        "model_config": model_config,
        "normalization_mean": torch.from_numpy(mean),
        "normalization_std": torch.from_numpy(std),
        "charset": charset,
        "blank_index": blank_index,
        "seed": seed,
        "training_report": report,
        "session_split": split_manifest,
        "pretraining": transfer_metadata,
        "training_auxiliary_state": auxiliary.state_dict() if auxiliary is not None else None,
        "auxiliary_config": (
            {
                "input_dim": int(model_config["hidden_size"]) * 2,
                "charset": charset,
                "max_text_characters": max(len(record.text) for record in fit_records) + 1,
            }
            if auxiliary is not None
            else None
        ),
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


def run_personal_echwr_comparison(
    *,
    recordings_dir: str | Path,
    output_dir: str | Path,
    pretrained_checkpoint: str | Path | None = None,
    charset: str = PERSONAL_CHARSET,
    validation_fraction: float = 0.2,
    split_manifest_path: str | Path | None = None,
    epochs: int = 20,
    batch_size: int = 16,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    seed: int = 1337,
    device_name: str = "auto",
    num_threads: int = 4,
    mixed_precision: bool | None = None,
    batch_contrastive_weight: float = 1.0,
    error_contrastive_weight: float = 1.0,
    negative_sets: int = 1,
    augmentation: str = "none",
    model_config: dict | None = None,
) -> dict:
    """Compare CTC, CTC+BC, and CTC+BC+EC on one session-held-out split."""
    if epochs < 1 or batch_size < 1 or num_threads < 1:
        raise ValueError("epochs, batch size, and thread count must be positive")
    if batch_contrastive_weight <= 0 or error_contrastive_weight <= 0:
        raise ValueError("contrastive loss weights must be positive")
    if negative_sets < 1:
        raise ValueError("negative_sets must be positive")
    if max_train_samples is not None and max_train_samples < 1:
        raise ValueError("max_train_samples must be positive")
    if max_validation_samples is not None and max_validation_samples < 1:
        raise ValueError("max_validation_samples must be positive")
    augmentation_plan(augmentation)
    if augmentation not in AUGMENTATION_NAMES:
        raise ValueError(f"unknown augmentation {augmentation!r}")

    charset = validate_charset(charset)
    blank_index = len(charset)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    use_mixed_precision = device.type == "cuda" if mixed_precision is None else mixed_precision
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
        split_manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))

    train_records, dropped_train = keep_ctc_feasible(split.train, charset)
    validation_records, dropped_validation = keep_ctc_feasible(split.validation, charset)
    train_records, selected_train_indices = select_records(train_records, max_train_samples, seed)
    validation_records, selected_validation_indices = select_records(
        validation_records, max_validation_samples, seed + 1
    )
    if not train_records or not validation_records:
        raise ValueError("session split has no CTC-feasible train or validation samples")
    if not split.train_sessions.isdisjoint(split.validation_sessions):
        raise ValueError("writer/session leakage in personal split")

    mean, std = feature_normalization(train_records)
    transforms, add_concatenations = augmentation_plan(augmentation)
    concatenated = (
        concatenate_training_records(train_records, seed=seed, charset=charset)
        if add_concatenations
        else []
    )
    concatenated, dropped_concatenations = keep_ctc_feasible(concatenated, charset)
    fit_records = train_records + concatenated

    config = default_model_config("resnet_bilstm_ctc", blank_index + 1)
    if model_config:
        config.update(model_config)
    checkpoint_path = None
    transfer_metadata = None
    if pretrained_checkpoint is not None:
        checkpoint_path = Path(pretrained_checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        probe = build_model("resnet_bilstm_ctc", config)
        probe._m6_model_config = config
        transfer_metadata = load_pretrained_encoder(probe, checkpoint_path)
        del probe

    output = Path(output_dir)
    _atomic_json(output / "session_split.json", split_manifest)
    specifications = (
        ("ctc", False, False),
        ("ctc_batch_contrastive", True, False),
        ("ctc_batch_and_error_contrastive", True, True),
    )
    results = {}
    for name, use_batch_contrastive, use_error_contrastive in specifications:
        results[name] = _train_echwr_variant(
            name=name,
            use_batch_contrastive=use_batch_contrastive,
            use_error_contrastive=use_error_contrastive,
            initialization_checkpoint=checkpoint_path,
            fit_records=fit_records,
            validation_records=validation_records,
            charset=charset,
            blank_index=blank_index,
            mean=mean,
            std=std,
            split_manifest=split_manifest,
            output_dir=output / name,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            num_threads=num_threads,
            mixed_precision=use_mixed_precision,
            augmentation=augmentation,
            batch_contrastive_weight=batch_contrastive_weight,
            error_contrastive_weight=error_contrastive_weight,
            negative_sets=negative_sets,
            model_config=config,
        )

    first = results["ctc"]
    same_data = all(
        result["training_sample_ids"] == first["training_sample_ids"]
        and result["validation_sample_ids"] == first["validation_sample_ids"]
        and result["normalization"] == first["normalization"]
        and result["charset"] == first["charset"]
        and result["augmentation"] == first["augmentation"]
        and result["pretraining"] == first["pretraining"]
        for result in results.values()
    )
    if not same_data:
        raise RuntimeError("M6-D variants used different data, preprocessing, or initialization")

    comparison = {
        "schema_version": 1,
        "experiment": "M6-D ECHWR-style contrastive text alignment",
        "dataset": "IMUpen personal recorder recordings",
        "training_sample_count": len(train_records),
        "augmented_training_sample_count": len(fit_records),
        "validation_sample_count": len(validation_records),
        "rejected_recordings_skipped": personal_data.rejected_recordings,
        "ctc_infeasible_recordings_dropped": {
            "train": dropped_train,
            "validation": dropped_validation,
            "synthetic_concatenations": dropped_concatenations,
        },
        "sample_limit_indices": {
            "train": selected_train_indices,
            "validation": selected_validation_indices,
        },
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "mixed_precision": use_mixed_precision,
        "pretrained_checkpoint_sha256": _sha256(checkpoint_path) if checkpoint_path else None,
        "session_split_manifest": "session_split.json",
        "augmentation": {
            "name": augmentation,
            "per_record_transforms": list(transforms),
            "concatenations_enabled": add_concatenations,
            "applied_only_to_training": True,
        },
        "contrastive_settings": {
            "batch_contrastive_weight": batch_contrastive_weight,
            "error_contrastive_weight": error_contrastive_weight,
            "negative_sets_per_sample": negative_sets,
            "hard_negative_edits": ["deletion", "insertion", "substitution"],
        },
        "fairness_checks": {
            "same_personal_train_and_validation_recordings": same_data,
            "same_writer_session_split": True,
            "same_charset_and_train_only_normalization": same_data,
            "same_seed_optimizer_and_augmentation": True,
            "same_encoder_initialization": same_data,
            "personal_session_leakage": False,
            "auxiliary_branch_removed_from_inference": all(
                result["auxiliary_branch_included_in_inference"] is False
                for result in results.values()
            ),
        },
        "models": {
            name: {
                "model_parameters": result["model_parameters"],
                "auxiliary_training_parameters": result["auxiliary_training_parameters"],
                "final_metrics": result["final_metrics"],
                "epochs": result["epochs"],
                "checkpoint": f"{name}/model.pt",
            }
            for name, result in results.items()
        },
        "interpretation": (
            "Personal held-out-session metrics are experiment-specific and may be unstable "
            "for small datasets. No accuracy improvement is assumed."
        ),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not comparison["fairness_checks"]["auxiliary_branch_removed_from_inference"]:
        raise RuntimeError("M6-D auxiliary branch unexpectedly changed inference architecture")
    _atomic_json(output / "comparison.json", comparison)
    return comparison
