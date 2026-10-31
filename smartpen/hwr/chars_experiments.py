"""Reproducible official-fold OnHW-Chars screening experiments."""
from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from smartpen.hwr.chars_data import (
    ONHW_CHARS_ALPHABETS,
    ONHW_CHARS_PROTOCOLS,
    OnHWCharsFold,
    load_onhw_chars,
)
from smartpen.hwr.data import OnHWFold, SequenceRecord
from smartpen.hwr.experiments import _canonical_json
from smartpen.hwr.metrics import single_character_metrics
from smartpen.hwr.model import MODEL_TYPES
from smartpen.hwr.provenance import code_provenance
from smartpen.hwr.sensors import PRIMARY_SENSOR_COMPARISONS, SENSOR_PROFILES, resolve_sensor_profile
from smartpen.hwr.training import (
    _atomic_json,
    _sha256,
    evaluate_checkpoint,
    keep_ctc_feasible,
    run_training,
)


_CHAR_SWEEP_KEYS = {
    "data",
    "corpora",
    "protocols",
    "folds",
    "sensor_profiles",
    "model_types",
    "seeds",
    "epochs",
    "batch_size",
    "device",
    "num_threads",
    "development_fraction",
    "development_seed",
    "model_config",
    "max_train_samples",
    "max_validation_samples",
}


def _array_values(config: dict, key: str, default: list):
    value = config.get(key, default)
    if value == "all" and key == "sensor_profiles":
        return [profile.profile_id for profile in SENSOR_PROFILES]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty JSON array")
    encoded = [json.dumps(item, sort_keys=True) for item in value]
    if len(encoded) != len(set(encoded)):
        raise ValueError(f"{key} must not contain duplicates")
    return value


def _load_sweep_config(config_path: str | Path) -> tuple[dict, Path]:
    config_file = Path(config_path).resolve()
    try:
        source = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read OnHW-Chars JSON config: {config_file}") from exc
    if not isinstance(source, dict):
        raise ValueError("OnHW-Chars sweep configuration must be a JSON object")
    unknown = set(source) - _CHAR_SWEEP_KEYS
    if unknown:
        raise ValueError(f"unsupported OnHW-Chars config keys: {sorted(unknown)}")
    if not isinstance(source.get("data"), str) or not source["data"].strip():
        raise ValueError("OnHW-Chars sweep configuration requires a data archive path")
    data_path = Path(source["data"])
    if not data_path.is_absolute():
        data_path = (config_file.parent / data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    normalized = {
        "data": str(data_path),
        "corpora": _array_values(source, "corpora", ["both"]),
        "protocols": _array_values(source, "protocols", ["indep"]),
        "folds": _array_values(source, "folds", [0, 1, 2, 3, 4]),
        "sensor_profiles": _array_values(source, "sensor_profiles", "all"),
        "model_types": _array_values(source, "model_types", ["resnet_bilstm_ctc"]),
        "seeds": _array_values(source, "seeds", [1337]),
        "epochs": int(source.get("epochs", 10)),
        "batch_size": int(source.get("batch_size", 32)),
        "device": source.get("device", "auto"),
        "num_threads": int(source.get("num_threads", 8)),
        "development_fraction": float(source.get("development_fraction", 0.1)),
        "development_seed": int(source.get("development_seed", 2026)),
        "model_config": source.get("model_config"),
        "max_train_samples": source.get("max_train_samples"),
        "max_validation_samples": source.get("max_validation_samples"),
    }
    if any(corpus not in ONHW_CHARS_ALPHABETS for corpus in normalized["corpora"]):
        raise ValueError(f"corpora must be one of {tuple(ONHW_CHARS_ALPHABETS)}")
    if any(protocol not in ONHW_CHARS_PROTOCOLS for protocol in normalized["protocols"]):
        raise ValueError(f"protocols must be one of {tuple(ONHW_CHARS_PROTOCOLS)}")
    if any(not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(5)
           for fold in normalized["folds"]):
        raise ValueError("folds must contain integers from 0 through 4")
    profiles = [resolve_sensor_profile(value).profile_id for value in normalized["sensor_profiles"]]
    if len(profiles) != len(set(profiles)):
        raise ValueError("sensor_profiles must not contain aliases for the same profile")
    normalized["sensor_profiles"] = profiles
    if any(model not in MODEL_TYPES for model in normalized["model_types"]):
        raise ValueError(f"model_types must contain values from {MODEL_TYPES}")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in normalized["seeds"]):
        raise ValueError("seeds must contain integers")
    if normalized["epochs"] < 1 or normalized["batch_size"] < 1 or normalized["num_threads"] < 1:
        raise ValueError("epochs, batch_size, and num_threads must be positive")
    if normalized["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if not 0.02 <= normalized["development_fraction"] <= 0.5:
        raise ValueError("development_fraction must be between 0.02 and 0.5")
    for key in ("max_train_samples", "max_validation_samples"):
        if normalized[key] is not None:
            normalized[key] = int(normalized[key])
            if normalized[key] < 1:
                raise ValueError(f"{key} must be positive when supplied")
    return normalized, data_path


def _stratified_internal_development(
    records: list[SequenceRecord], *, fraction: float, seed: int
) -> tuple[list[SequenceRecord], list[SequenceRecord]]:
    """Partition official training rows into fixed, class-stratified train/dev sets."""
    groups: dict[str, list[SequenceRecord]] = defaultdict(list)
    for record in records:
        groups[record.text].append(record)
    train: list[SequenceRecord] = []
    development: list[SequenceRecord] = []
    for class_index, label in enumerate(sorted(groups)):
        ordered = sorted(groups[label], key=lambda record: record.sample_index)
        rng = np.random.default_rng(seed + class_index * 1009)
        order = rng.permutation(len(ordered))
        dev_count = max(1, int(round(len(ordered) * fraction)))
        if dev_count >= len(ordered):
            raise ValueError(f"class {label!r} has too few samples for internal development split")
        dev_indices = set(int(value) for value in order[:dev_count])
        development.extend(record for index, record in enumerate(ordered) if index in dev_indices)
        train.extend(record for index, record in enumerate(ordered) if index not in dev_indices)
    train.sort(key=lambda record: record.sample_id or "")
    development.sort(key=lambda record: record.sample_id or "")
    train_ids = {record.sample_id for record in train}
    development_ids = {record.sample_id for record in development}
    train_hashes = {record.content_sha256 for record in train}
    development_hashes = {record.content_sha256 for record in development}
    if train_ids & development_ids or train_hashes & development_hashes:
        raise ValueError("exact recording leakage detected in internal development split")
    return train, development


def _class_counts(records: list[SequenceRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.text for record in records).items()))


def _runtime_provenance(device: str) -> dict:
    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "device": selected,
        "device_name": torch.cuda.get_device_name() if selected == "cuda" else platform.processor(),
    }


def _chars_run_config(
    config: dict,
    fold: OnHWCharsFold,
    *,
    model_type: str,
    sensor_profile: str,
    seed: int,
) -> dict:
    return {
        "dataset": "OnHW-Chars",
        "label_charset": "".join(ONHW_CHARS_ALPHABETS[fold.corpus]),
        "archive_name": fold.archive_name,
        "corpus": fold.corpus,
        "protocol": fold.protocol,
        "fold": fold.fold,
        "model_type": model_type,
        "sensor_profile": sensor_profile,
        "seed": seed,
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "device": config["device"],
        "num_threads": config["num_threads"],
        "development_fraction": config["development_fraction"],
        "development_seed": config["development_seed"],
        "model_config": config["model_config"],
        "max_train_samples": config["max_train_samples"],
        "max_validation_samples": config["max_validation_samples"],
        "official_train_sample_ids": [record.sample_id for record in fold.train],
        "official_test_sample_ids": [record.sample_id for record in fold.test],
        "data_quality": fold.data_quality,
    }


def _load_completed_run(run_dir: Path, expected_fingerprint: str) -> tuple[dict, dict] | None:
    marker_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    evaluation_path = run_dir / "evaluation.json"
    if not marker_path.is_file() or not metrics_path.is_file() or not evaluation_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("status") != "complete"
            or marker.get("run_fingerprint") != expected_fingerprint
            or marker.get("metrics_sha256") != _sha256(metrics_path)
            or marker.get("evaluation_sha256") != _sha256(evaluation_path)
        ):
            return None
        return (
            json.loads(metrics_path.read_text(encoding="utf-8")),
            json.loads(evaluation_path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError):
        return None


def _run_chars_cell(
    *,
    config: dict,
    fold_data: OnHWCharsFold,
    train_records: list[SequenceRecord],
    development_records: list[SequenceRecord],
    model_type: str,
    sensor_profile: str,
    seed: int,
    archive_sha256: str,
    run_dir: Path,
    dry_run: bool,
) -> tuple[dict, dict | None]:
    run_config = _chars_run_config(
        config,
        fold_data,
        model_type=model_type,
        sensor_profile=sensor_profile,
        seed=seed,
    )
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "config": run_config,
                "archive_sha256": archive_sha256,
                "code_provenance": code_provenance(),
                "runtime_provenance": _runtime_provenance(config["device"]),
            }
        ).encode("utf-8")
    ).hexdigest()
    run_id = fingerprint[:20]
    row = {
        "run_id": run_id,
        "configuration": {
            key: run_config[key]
            for key in ("corpus", "protocol", "fold", "model_type", "sensor_profile", "seed")
        },
        "status": "planned" if dry_run else "running",
    }
    if dry_run:
        return row, None

    run_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_run(run_dir, fingerprint)
    if completed is not None:
        train_report, evaluation = completed
        row["status"] = "skipped_completed"
    else:
        marker = {
            "schema_version": 1,
            "run_id": run_id,
            "run_fingerprint": fingerprint,
            "status": "running",
            "configuration": run_config,
            "archive_sha256": archive_sha256,
            "code_provenance": code_provenance(),
            "runtime_provenance": _runtime_provenance(config["device"]),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        marker_path = run_dir / "run_manifest.json"
        _atomic_json(marker_path, marker)
        protocol = (
            f"OnHW-Chars {fold_data.corpus} {ONHW_CHARS_PROTOCOLS[fold_data.protocol]} "
            f"fold {fold_data.fold}; internal stratified development partition"
        )
        split_summary = {
            **fold_data.data_quality,
            "official_train_class_counts": _class_counts(fold_data.train),
            "official_test_class_counts": _class_counts(fold_data.test),
            "model_fit_class_counts": _class_counts(train_records),
            "internal_development_class_counts": _class_counts(development_records),
            "internal_development_fraction": config["development_fraction"],
            "internal_development_seed": config["development_seed"],
            "official_test_used_for_checkpoint_selection": False,
            "writer_ids_available": False,
            "writer_level_uncertainty_available": False,
        }
        train_fold = OnHWFold(
            train=train_records,
            validation=development_records,
            fold=fold_data.fold,
            split_family="onhw_chars_internal_development",
            archive_name=fold_data.archive_name,
        )
        try:
            train_report = run_training(
                data_path=config["data"],
                output_dir=run_dir,
                fold=fold_data.fold,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                seed=seed,
                device_name=config["device"],
                max_train_samples=config["max_train_samples"],
                max_validation_samples=config["max_validation_samples"],
                num_threads=config["num_threads"],
                model_type=model_type,
                sensor_profile=sensor_profile,
                training_stage="onhw_chars_single_character_ctc",
                model_config=config["model_config"],
                resume=True,
                records_override=train_fold,
                dataset_name="OnHW-Chars",
                protocol_label=protocol,
                dataset_summary=split_summary,
                writer_ids_available=False,
                metric_mode="single_character",
                label_charset="".join(ONHW_CHARS_ALPHABETS[fold_data.corpus]),
            )
            test_fold = OnHWFold(
                train=train_records,
                validation=fold_data.test,
                fold=fold_data.fold,
                split_family="onhw_chars_official_test",
                archive_name=fold_data.archive_name,
            )
            test_summary = {
                **split_summary,
                "evaluation_partition": "official test",
                "evaluation_sample_count": len(fold_data.test),
                "evaluation_class_counts": _class_counts(fold_data.test),
            }
            evaluation = evaluate_checkpoint(
                checkpoint_path=run_dir / "model.pt",
                data_path=config["data"],
                fold=fold_data.fold,
                max_validation_samples=config["max_validation_samples"],
                seed=seed,
                device_name=config["device"],
                batch_size=config["batch_size"],
                num_threads=config["num_threads"],
                records_override=test_fold,
                dataset_name="OnHW-Chars",
                protocol_label=(
                    f"OnHW-Chars {fold_data.corpus} "
                    f"{ONHW_CHARS_PROTOCOLS[fold_data.protocol]} official test fold {fold_data.fold}"
                ),
                dataset_summary=test_summary,
                writer_ids_available=False,
                metric_mode="single_character",
                label_charset="".join(ONHW_CHARS_ALPHABETS[fold_data.corpus]),
            )
            _atomic_json(run_dir / "evaluation.json", evaluation)
            if config["max_train_samples"] is None and config["max_validation_samples"] is None:
                (run_dir / "model.pt").unlink(missing_ok=True)
            marker.update(
                {
                    "status": "complete",
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "selected_epoch": train_report["checkpoint_selection"]["selected_epoch"],
                    "test_metrics": evaluation["metrics"],
                    "metrics_sha256": _sha256(run_dir / "metrics.json"),
                    "evaluation_sha256": _sha256(run_dir / "evaluation.json"),
                    "checkpoint_sha256": _sha256(run_dir / "model.pt")
                    if (run_dir / "model.pt").is_file()
                    else None,
                    "checkpoint_retained": (run_dir / "model.pt").is_file(),
                }
            )
            _atomic_json(marker_path, marker)
        except Exception as exc:
            marker.update(
                {
                    "status": "failed_resumable"
                    if (run_dir / "progress.pt").is_file()
                    else "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            _atomic_json(marker_path, marker)
            raise

    metrics = evaluation["metrics"]
    correctness = {
        sample_id: hypothesis == reference
        for sample_id, hypothesis, reference in zip(
            evaluation["validation_outputs"]["sample_ids"],
            evaluation["validation_outputs"]["hypotheses"],
            evaluation["validation_outputs"]["references"],
        )
    }
    row.update(
        {
            "status": row["status"] if row["status"] == "skipped_completed" else "complete",
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "correct_count": metrics["correct_count"],
            "sample_count": metrics["sample_count"],
            "invalid_prediction_count": metrics["invalid_prediction_count"],
            "labels": metrics["labels"],
            "confusion_matrix": metrics["confusion_matrix"],
            "model_parameters": evaluation["model_parameters"],
            "training_seconds_total_recorded": sum(
                epoch["epoch_seconds"] for epoch in train_report.get("epochs", [])
            ),
            "training_seconds_this_attempt": train_report.get("training_seconds_this_attempt"),
            "inference_seconds": evaluation["inference_seconds"],
            "epochs_completed": len(train_report.get("epochs", [])),
            "selected_epoch": train_report["checkpoint_selection"]["selected_epoch"],
            "train_samples_used": train_report["train_samples_used"],
            "internal_development_samples": train_report["validation_samples_used"],
            "per_class": metrics["per_class"],
            "run_manifest": str(run_dir / "run_manifest.json"),
        }
    )
    return row, correctness


def _aggregate_chars_runs(rows: list[dict], correctness: dict[str, dict[str, bool]]) -> dict:
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    indexed: dict[tuple[str, str, str, int, int, str], dict] = {}
    for row in rows:
        cfg = row["configuration"]
        if row.get("accuracy") is None:
            continue
        key = (cfg["corpus"], cfg["protocol"], cfg["model_type"], cfg["sensor_profile"])
        groups[key].append(row)
        indexed[(cfg["corpus"], cfg["protocol"], cfg["model_type"], cfg["fold"], cfg["seed"], cfg["sensor_profile"])] = row

    profile_summaries = {}
    per_class_summaries = {}
    confusion_summaries = {}
    for (corpus, protocol, model, profile), entries in sorted(groups.items()):
        metrics = {}
        for key in ("accuracy", "balanced_accuracy", "macro_f1"):
            values = [entry[key] for entry in entries]
            metrics[key] = {
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "fold_seed_run_count": len(values),
            }
        summary_key = f"{corpus}:{protocol}:{model}:{profile}"
        profile_summaries[summary_key] = {
            "corpus": corpus,
            "protocol": protocol,
            "model_type": model,
            "sensor_profile": profile,
            "metrics": metrics,
            "sample_count_total": sum(entry["sample_count"] for entry in entries),
            "model_parameters": entries[0]["model_parameters"],
        }
        class_accumulator: dict[str, dict[str, int]] = defaultdict(
            lambda: {"correct_count": 0, "reference_count": 0, "predicted_count": 0}
        )
        for entry in entries:
            for label, values in entry["per_class"].items():
                for field in ("correct_count", "reference_count", "predicted_count"):
                    class_accumulator[label][field] += values[field]
        per_class_summaries[summary_key] = {
            label: {
                **values,
                "recall": values["correct_count"] / max(values["reference_count"], 1),
                "precision": values["correct_count"] / max(values["predicted_count"], 1),
                "f1": (
                    2
                    * values["correct_count"] / max(values["predicted_count"], 1)
                    * values["correct_count"] / max(values["reference_count"], 1)
                    / max(
                        values["correct_count"] / max(values["predicted_count"], 1)
                        + values["correct_count"] / max(values["reference_count"], 1),
                        1e-12,
                    )
                ),
            }
            for label, values in sorted(class_accumulator.items())
        }
        labels = entries[0]["labels"]
        pooled_confusion = [
            [0 for _ in labels]
            for _ in labels
        ]
        for entry in entries:
            if entry["labels"] != labels:
                raise ValueError("OnHW-Chars label order differs within an aggregate group")
            for row_index, confusion_row in enumerate(entry["confusion_matrix"]):
                for column_index, value in enumerate(confusion_row):
                    pooled_confusion[row_index][column_index] += int(value)
        confusion_summaries[summary_key] = {
            "labels": labels,
            "confusion_matrix": pooled_confusion,
            "interpretation": "sum over official held-out fold evaluations; not deduplicated recordings",
        }

    paired_contrasts = []
    for (corpus, protocol, model) in sorted(
        {(key[0], key[1], key[2]) for key in groups}
    ):
        for comparison in PRIMARY_SENSOR_COMPARISONS:
            left = resolve_sensor_profile(comparison["first"]).profile_id
            right = resolve_sensor_profile(comparison["second"]).profile_id
            paired = []
            keys = {
                (fold, seed)
                for c, p, m, fold, seed, profile in indexed
                if (c, p, m, profile) == (corpus, protocol, model, left)
            } & {
                (fold, seed)
                for c, p, m, fold, seed, profile in indexed
                if (c, p, m, profile) == (corpus, protocol, model, right)
            }
            for fold, seed in sorted(keys):
                left_row = indexed[(corpus, protocol, model, fold, seed, left)]
                right_row = indexed[(corpus, protocol, model, fold, seed, right)]
                left_correct = correctness.get(left_row["run_id"], {})
                right_correct = correctness.get(right_row["run_id"], {})
                shared_ids = sorted(set(left_correct) & set(right_correct))
                left_wins = sum(left_correct[key] and not right_correct[key] for key in shared_ids)
                right_wins = sum(right_correct[key] and not left_correct[key] for key in shared_ids)
                paired.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "left_accuracy": left_row["accuracy"],
                        "right_accuracy": right_row["accuracy"],
                        "accuracy_difference_right_minus_left": (
                            right_row["accuracy"] - left_row["accuracy"]
                        ),
                        "paired_test_sample_count": len(shared_ids),
                        "left_only_correct": left_wins,
                        "right_only_correct": right_wins,
                    }
                )
            paired_contrasts.append(
                {
                    "corpus": corpus,
                    "protocol": protocol,
                    "model_type": model,
                    "left_profile": left,
                    "right_profile": right,
                    "question": comparison["question"],
                    "paired_fold_seed_count": len(paired),
                    "mean_accuracy_difference_right_minus_left": (
                        float(np.mean([
                            item["accuracy_difference_right_minus_left"] for item in paired
                        ]))
                        if paired
                        else None
                    ),
                    "paired_differences": paired,
                }
            )

    return {
        "metric": "single_character_classification",
        "profile_summaries": profile_summaries,
        "per_class_summaries": per_class_summaries,
        "confusion_summaries": confusion_summaries,
        "paired_sensor_contrasts": paired_contrasts,
        "uncertainty_note": (
            "Writer IDs are absent in this archive. Five-fold means/standard deviations and "
            "same-test-sample differences are descriptive; writer-clustered confidence intervals "
            "cannot be computed from these files."
        ),
    }


def run_onhw_chars_sweep(
    *, config_path: str | Path, output_dir: str | Path, dry_run: bool = False
) -> dict:
    """Run a resumable sensor-profile × official-fold OnHW-Chars study.

    The official test fold is held out from model selection. A fixed, stratified
    10% subset of the official training fold is used for checkpoint selection.
    """
    config, data_path = _load_sweep_config(config_path)
    archive_sha256 = _sha256(data_path)
    if archive_sha256 is None:
        raise ValueError("data path must be an OnHW-Chars ZIP archive")
    combinations = [
        {"corpus": corpus, "protocol": protocol, "fold": fold,
         "model_type": model, "sensor_profile": profile, "seed": seed}
        for corpus in config["corpora"]
        for protocol in config["protocols"]
        for fold in config["folds"]
        for model in config["model_types"]
        for profile in config["sensor_profiles"]
        for seed in config["seeds"]
    ]
    sweep_fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "config": config,
                "archive_sha256": archive_sha256,
                "code_provenance": code_provenance(),
                "runtime_provenance": _runtime_provenance(config["device"]),
            }
        ).encode("utf-8")
    ).hexdigest()
    sweep_id = sweep_fingerprint[:20]
    root = Path(output_dir).resolve() / sweep_id
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "chars_sweep_manifest.json"
    aggregate_path = root / "chars_sweep_aggregate.json"
    manifest = {
        "schema_version": 1,
        "sweep_id": sweep_id,
        "sweep_fingerprint": sweep_fingerprint,
        "status": "planned" if dry_run else "running",
        "dataset": "OnHW-Chars",
        "archive_name": data_path.name,
        "archive_sha256": archive_sha256,
        "configuration": config,
        "run_count": len(combinations),
        "dry_run": dry_run,
        "runs": [],
        "code_provenance": code_provenance(),
        "runtime_provenance": _runtime_provenance(config["device"]),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(manifest_path, manifest)

    rows: list[dict] = []
    correctness_by_run: dict[str, dict[str, bool]] = {}
    failures = 0
    grouped_combinations: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for combination in combinations:
        grouped_combinations[(combination["corpus"], combination["protocol"], combination["fold"])].append(
            combination
        )
    for cell_key, cell_combinations in sorted(grouped_combinations.items()):
        corpus, protocol, fold = cell_key
        fold_data = load_onhw_chars(
            data_path, corpus=corpus, protocol=protocol, fold=fold
        )
        train_records, development_records = _stratified_internal_development(
            fold_data.train,
            fraction=config["development_fraction"],
            seed=config["development_seed"],
        )
        ctc_train, _ = keep_ctc_feasible(train_records)
        ctc_development, _ = keep_ctc_feasible(development_records)
        ctc_test, _ = keep_ctc_feasible(fold_data.test)
        if not ctc_train or not ctc_development or not ctc_test:
            raise ValueError(f"empty CTC-feasible partition in {corpus}/{protocol}/{fold}")
        for combination in cell_combinations:
            row_name = (
                f"{corpus}:{protocol}:fold-{fold}:{combination['model_type']}:"
                f"{combination['sensor_profile']}:seed-{combination['seed']}"
            )
            config_fingerprint = hashlib.sha256(
                _canonical_json(combination).encode("utf-8")
            ).hexdigest()[:16]
            run_dir = root / "runs" / config_fingerprint
            try:
                row, correctness = _run_chars_cell(
                    config=config,
                    fold_data=fold_data,
                    train_records=train_records,
                    development_records=development_records,
                    model_type=combination["model_type"],
                    sensor_profile=combination["sensor_profile"],
                    seed=combination["seed"],
                    archive_sha256=archive_sha256,
                    run_dir=run_dir,
                    dry_run=dry_run,
                )
                row["run_name"] = row_name
                if correctness is not None:
                    correctness_by_run[row["run_id"]] = correctness
            except Exception as exc:
                failures += 1
                row = {
                    "run_id": config_fingerprint,
                    "run_name": row_name,
                    "configuration": combination,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            manifest["runs"] = list(rows)
            manifest["completed_run_count"] = sum(
                item["status"] in {"complete", "skipped_completed", "planned"}
                for item in rows
            )
            manifest["failed_run_count"] = failures
            _atomic_json(manifest_path, manifest)
        del fold_data

    aggregation = _aggregate_chars_runs(rows, correctness_by_run)
    manifest.update(
        {
            "status": "planned" if dry_run else ("complete_with_failures" if failures else "complete"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failed_run_count": failures,
            "aggregation_file": aggregate_path.name,
            "aggregation": aggregation,
            "interpretation": (
                "Full OnHW-Chars character-level experiment; no word-recognition claims."
                if not config["max_train_samples"] and not config["max_validation_samples"]
                else "Sample-capped integration check only; not a benchmark."
            ),
        }
    )
    _atomic_json(aggregate_path, aggregation)
    _atomic_json(manifest_path, manifest)
    return manifest
