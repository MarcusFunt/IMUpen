"""Minimal configuration-driven, resumable OnHW M5 experiment runner."""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from smartpen.hwr.data import OnHWFold, apply_sensor_profile, load_onhw_words500
from smartpen.hwr.metrics import factorial_writer_bootstrap
from smartpen.hwr.model import MODEL_TYPES
from smartpen.hwr.provenance import code_provenance
from smartpen.hwr.protocols import (
    build_writer_adaptation_splits,
    build_writer_vocabulary_factorial,
    write_factorial_manifests,
    write_writer_adaptation_manifests,
)
from smartpen.hwr.sensors import (
    PRIMARY_SENSOR_COMPARISONS,
    SENSOR_PROFILES,
    resolve_sensor_profile,
)
from smartpen.hwr.training import (
    _atomic_json,
    _sha256,
    evaluate_checkpoint,
    keep_ctc_feasible,
    run_training,
    select_records,
)


_ALLOWED_CONFIG_KEYS = {
    "data",
    "fold",
    "sensor_profile",
    "seed",
    "epochs",
    "batch_size",
    "device",
    "num_threads",
    "max_train_samples",
    "max_validation_samples",
    "protocol",
    "model_type",
}


def _canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _runtime_provenance(device_name: str) -> dict:
    selected_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device_name == "auto" else device_name
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "device": selected_device,
        "device_name": (
            torch.cuda.get_device_name(torch.device(selected_device))
            if selected_device == "cuda" and torch.cuda.is_available()
            else None
        ),
    }


def _recover_completed_training(
    run_dir: Path, config: dict, archive_sha256: str, profile_id: str
) -> dict | None:
    """Recover the short crash window after training artifacts are committed."""
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "model.pt"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        return None
    try:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, json.JSONDecodeError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
        return None
    if (
        report.get("archive_sha256") != archive_sha256
        or report.get("fold") != config["fold"]
        or report.get("seed") != config["seed"]
        or report.get("model_type") != config["model_type"]
        or report.get("training_stage") != "supervised_ctc"
        or report.get("sensor_profile", {}).get("profile_id") != profile_id
        or len(report.get("epochs", [])) != config["epochs"]
        or report.get("training_signature_sha256")
        != checkpoint.get("training_signature_sha256")
        or report.get("code_provenance") != code_provenance()
    ):
        return None
    return report


def _recover_completed_factorial_cell(
    cell_dir: Path,
    *,
    archive_sha256: str,
    fold: int,
    seed: int,
    model_type: str,
    profile_id: str,
    epochs: int,
    split_manifest: dict,
) -> dict | None:
    metrics_path = cell_dir / "metrics.json"
    checkpoint_path = cell_dir / "model.pt"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        return None
    try:
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (
        OSError,
        json.JSONDecodeError,
        RuntimeError,
        EOFError,
        ValueError,
        pickle.UnpicklingError,
    ):
        return None
    if (
        report.get("archive_sha256") != archive_sha256
        or report.get("fold") != fold
        or report.get("seed") != seed
        or report.get("model_type") != model_type
        or report.get("sensor_profile", {}).get("profile_id") != profile_id
        or report.get("custom_split_manifest") != split_manifest
        or len(report.get("epochs", [])) != epochs
        or report.get("training_signature_sha256")
        != checkpoint.get("training_signature_sha256")
        or report.get("code_provenance") != code_provenance()
    ):
        return None
    return report


def _load_config(config_path: str | Path) -> tuple[dict, Path, Path]:
    config_file = Path(config_path).resolve()
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON experiment configuration: {config_file}") from exc
    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a JSON object")
    unknown = set(config) - _ALLOWED_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unsupported experiment configuration keys: {sorted(unknown)}")
    if not isinstance(config.get("data"), str) or not config["data"].strip():
        raise ValueError("experiment configuration requires a data path")
    data_path = Path(config["data"])
    if not data_path.is_absolute():
        data_path = (config_file.parent / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    normalized = {
        "data": data_path.name,
        "fold": int(config.get("fold", 0)),
        "sensor_profile": resolve_sensor_profile(
            config.get("sensor_profile", "front_acc_gyro6")
        ).profile_id,
        "seed": int(config.get("seed", 1337)),
        "epochs": int(config.get("epochs", 10)),
        "batch_size": int(config.get("batch_size", 32)),
        "device": str(config.get("device", "auto")),
        "num_threads": int(config.get("num_threads", 4)),
        "max_train_samples": config.get("max_train_samples"),
        "max_validation_samples": config.get("max_validation_samples"),
        "protocol": config.get("protocol", "official"),
        "model_type": config.get("model_type", "resnet_bilstm_ctc"),
    }
    if normalized["fold"] not in range(5):
        raise ValueError("fold must be from 0 through 4")
    if normalized["epochs"] < 1 or normalized["batch_size"] < 1 or normalized["num_threads"] < 1:
        raise ValueError("epochs, batch_size, and num_threads must be positive")
    for key in ("max_train_samples", "max_validation_samples"):
        if normalized[key] is not None and int(normalized[key]) < 1:
            raise ValueError(f"{key} must be a positive integer when supplied")
        if normalized[key] is not None:
            normalized[key] = int(normalized[key])
    if normalized["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if normalized["model_type"] not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")
    if not isinstance(normalized["protocol"], str) or normalized["protocol"] not in {
        "official",
        "writer_dependent",
        "writer_independent",
    }:
        raise ValueError("protocol must be official, writer_dependent, or writer_independent")
    return normalized, data_path, config_file


def run_experiment(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = True,
) -> dict:
    """Validate, train, evaluate, and record one reproducible official-fold M4/M5 run."""
    config, data_path, _ = _load_config(config_path)
    archive_sha256 = _sha256(data_path)
    if archive_sha256 is None:
        raise ValueError("data path must be an OnHW ZIP archive or extracted dataset directory")
    fold_data = load_onhw_words500(data_path, config["fold"])
    if config["protocol"] != "official" and config["protocol"] != fold_data.split_family:
        raise ValueError(
            f"configuration requests {config['protocol']} but the selected archive is "
            f"{fold_data.split_family}"
        )
    profile = resolve_sensor_profile(config["sensor_profile"])
    model_label = "M4" if config["model_type"] == "cnn_bilstm_ctc" else "M5"
    train_records = apply_sensor_profile(fold_data.train, profile.profile_id)
    validation_records = apply_sensor_profile(fold_data.validation, profile.profile_id)
    train_records, dropped_train = keep_ctc_feasible(train_records)
    validation_records, dropped_validation = keep_ctc_feasible(validation_records)
    ctc_train_count = len(train_records)
    ctc_validation_count = len(validation_records)
    train_records, train_indices = select_records(
        train_records, config["max_train_samples"], config["seed"]
    )
    validation_records, validation_indices = select_records(
        validation_records, config["max_validation_samples"], config["seed"] + 1
    )
    if not train_records or not validation_records:
        raise ValueError("experiment split must have CTC-feasible training and validation samples")

    run_fingerprint = hashlib.sha256(
        _canonical_json({
            "config": config,
            "archive_sha256": archive_sha256,
            "code_provenance": code_provenance(),
            "runtime_provenance": _runtime_provenance(config["device"]),
        }).encode("utf-8")
    ).hexdigest()
    run_id = run_fingerprint[:20]
    output_root = Path(output_dir).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != run_fingerprint:
            raise ValueError("run directory contains a manifest for a different configuration")
        if (
            previous.get("status") == "complete"
            and (run_dir / "model.pt").is_file()
            and (run_dir / "metrics.json").is_file()
            and (run_dir / "evaluation.json").is_file()
            and previous.get("checkpoint_sha256") == _sha256(run_dir / "model.pt")
            and previous.get("metrics_sha256") == _sha256(run_dir / "metrics.json")
            and previous.get("evaluation_sha256") == _sha256(run_dir / "evaluation.json")
        ):
            return {**previous, "status": "skipped_completed"}

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "run_fingerprint": run_fingerprint,
        "status": "planned" if dry_run else "running",
        "experiment": f"OnHW official-fold {model_label} supervised CTC",
        "configuration": config,
        "code_provenance": code_provenance(),
        "runtime_provenance": _runtime_provenance(config["device"]),
        "dataset": {
            "path_name": data_path.name,
            "sha256": archive_sha256,
            "protocol": fold_data.split_family,
            "fold": fold_data.fold,
            "train_samples_available": len(fold_data.train),
            "validation_samples_available": len(fold_data.validation),
            "train_samples_ctc_feasible": ctc_train_count,
            "validation_samples_ctc_feasible": ctc_validation_count,
            "ctc_infeasible_dropped": {
                "train": dropped_train,
                "validation": dropped_validation,
            },
            "selected_train_sample_indices": train_indices,
            "selected_validation_sample_indices": validation_indices,
        },
        "sensor_profile": {
            "profile_id": profile.profile_id,
            "group_code": profile.group_code,
            "groups": [group.name for group in profile.groups],
            "channel_indices": list(profile.channel_indices),
            "channel_names": list(profile.channel_names),
            "units": list(profile.units),
            "sample_rate_hz": 100.0,
            "physical_description": profile.physical_description,
        },
        "analysis_registration": {
            "primary_sensor_comparisons": list(PRIMARY_SENSOR_COMPARISONS),
            "full_31_profile_ranking": "exploratory_until_independent_confirmation",
        },
        "checkpoint": "model.pt",
        "result_report": "metrics.json",
        "evaluation_report": "evaluation.json",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(manifest_path, manifest)
    if dry_run:
        manifest["status"] = "planned"
        _atomic_json(manifest_path, manifest)
        return manifest

    try:
        report = (
            _recover_completed_training(run_dir, config, archive_sha256, profile.profile_id)
            if resume
            else None
        )
        if report is None:
            report = run_training(
                data_path=data_path,
                output_dir=run_dir,
                fold=config["fold"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                seed=config["seed"],
                device_name=config["device"],
                max_train_samples=config["max_train_samples"],
                max_validation_samples=config["max_validation_samples"],
                num_threads=config["num_threads"],
                model_type=config["model_type"],
                sensor_profile=profile.profile_id,
                training_stage="supervised_ctc",
                resume=resume,
            )
        evaluation = evaluate_checkpoint(
            checkpoint_path=run_dir / "model.pt",
            data_path=data_path,
            fold=config["fold"],
            max_validation_samples=config["max_validation_samples"],
            seed=config["seed"],
            device_name=config["device"],
            batch_size=config["batch_size"],
            num_threads=config["num_threads"],
        )
        _atomic_json(run_dir / "evaluation.json", evaluation)
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_sha256": _sha256(run_dir / "model.pt"),
                "metrics_sha256": _sha256(run_dir / "metrics.json"),
                "evaluation_sha256": _sha256(run_dir / "evaluation.json"),
                "selected_epoch": report["checkpoint_selection"]["selected_epoch"],
                "final_metrics": report["final_metrics"],
                "evaluation_metrics": evaluation["metrics"],
                "interpretation": report["interpretation"],
            }
        )
        _atomic_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest.update(
            {
                "status": "failed_resumable" if (run_dir / "progress.pt").is_file() else "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise


def _sweep_configs(config_path: str | Path) -> tuple[list[dict], Path, str]:
    config_file = Path(config_path).resolve()
    try:
        source = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON sweep configuration: {config_file}") from exc
    if not isinstance(source, dict):
        raise ValueError("sweep configuration must be a JSON object")
    axis_keys = {"sensor_profiles", "model_types", "folds", "seeds"}
    unknown = set(source) - _ALLOWED_CONFIG_KEYS - axis_keys
    if unknown:
        raise ValueError(f"unsupported sweep configuration keys: {sorted(unknown)}")
    if not isinstance(source.get("data"), str) or not source["data"].strip():
        raise ValueError("sweep configuration requires a data path")
    data_path = Path(source["data"])
    if not data_path.is_absolute():
        data_path = (config_file.parent / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    def values(axis_name: str, scalar_name: str, default):
        axis = source.get(axis_name)
        if axis_name == "sensor_profiles" and axis == "all":
            return [profile.profile_id for profile in SENSOR_PROFILES]
        if axis is None:
            value = source.get(scalar_name, default)
            return [value]
        if not isinstance(axis, list) or not axis:
            raise ValueError(f"{axis_name} must be a non-empty JSON array")
        serialized = [json.dumps(value, sort_keys=True) for value in axis]
        if len(axis) != len(set(serialized)):
            raise ValueError(f"{axis_name} must not contain duplicates")
        return axis

    profiles = values("sensor_profiles", "sensor_profile", "front_acc_gyro6")
    models = values("model_types", "model_type", "resnet_bilstm_ctc")
    folds = values("folds", "fold", 0)
    seeds = values("seeds", "seed", 1337)
    if any(not isinstance(profile, str) for profile in profiles):
        raise ValueError("sensor_profiles must contain profile IDs or group codes")
    profiles = [resolve_sensor_profile(profile).profile_id for profile in profiles]
    if len(profiles) != len(set(profiles)):
        raise ValueError("sensor_profiles must not contain aliases for the same profile")
    if any(not isinstance(model, str) or model not in MODEL_TYPES for model in models):
        raise ValueError(f"model_types must contain values from {MODEL_TYPES}")
    if any(not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(5) for fold in folds):
        raise ValueError("folds must contain integers from 0 through 4")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must contain integers")

    base = {key: value for key, value in source.items() if key not in axis_keys}
    base["data"] = str(data_path)
    configs = []
    for model in models:
        for profile in profiles:
            for fold in folds:
                for seed in seeds:
                    configs.append(
                        {
                            **base,
                            "model_type": model,
                            "sensor_profile": resolve_sensor_profile(profile).profile_id,
                            "fold": fold,
                            "seed": seed,
                        }
                    )
    return configs, data_path, config_file.name


def _aggregate_sweep_results(runs: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {}
    indexed: dict[tuple[str, str, int, int], dict] = {}
    for run in runs:
        model = run["configuration"]["model_type"]
        profile = run["configuration"]["sensor_profile"]
        grouped.setdefault((model, profile), []).append(run)
        if run.get("metrics") is not None:
            indexed[(model, profile, run["configuration"]["fold"], run["configuration"]["seed"])] = run

    profile_summaries = {}
    for (model, profile), entries in sorted(grouped.items()):
        metric_summaries = {}
        for metric in ("wer", "cer", "exact_word_accuracy"):
            values = [entry["metrics"][metric] for entry in entries if entry.get("metrics") is not None]
            if values:
                metric_summaries[metric] = {
                    "mean": float(np.mean(values)),
                    "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "run_count": len(values),
                }
        profile_summaries[f"{model}:{profile}"] = {
            "model_type": model,
            "sensor_profile": profile,
            "metrics": metric_summaries,
            "completed_run_count": sum(entry.get("metrics") is not None for entry in entries),
            "failed_run_count": sum(entry["status"] == "failed" for entry in entries),
        }

    primary_comparisons = []
    for model in MODEL_TYPES:
        for comparison in PRIMARY_SENSOR_COMPARISONS:
            left = comparison["first"]
            right = comparison["second"]
            left_profile = resolve_sensor_profile(left).profile_id
            right_profile = resolve_sensor_profile(right).profile_id
            paired = []
            keys = {
                (fold, seed)
                for run_model, profile, fold, seed in indexed
                if run_model == model and profile == left_profile
            } & {
                (fold, seed)
                for run_model, profile, fold, seed in indexed
                if run_model == model and profile == right_profile
            }
            for fold, seed in sorted(keys):
                left_run = indexed[(model, left_profile, fold, seed)]
                right_run = indexed[(model, right_profile, fold, seed)]
                paired.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "wer_difference_right_minus_left": (
                            right_run["metrics"]["wer"] - left_run["metrics"]["wer"]
                        ),
                        "cer_difference_right_minus_left": (
                            right_run["metrics"]["cer"] - left_run["metrics"]["cer"]
                        ),
                    }
                )
            primary_comparisons.append(
                {
                    "model_type": model,
                    "left_profile": left_profile,
                    "right_profile": right_profile,
                    "rationale": comparison["question"],
                    "paired_run_count": len(paired),
                    "paired_differences": paired,
                    "mean_wer_difference_right_minus_left": (
                        float(np.mean([row["wer_difference_right_minus_left"] for row in paired]))
                        if paired else None
                    ),
                    "interpretation": "paired descriptive development estimate; confirm independently",
                }
            )
    return {
        "profile_summaries": profile_summaries,
        "primary_sensor_comparisons": primary_comparisons,
    }


def run_sweep(
    *, config_path: str | Path, output_dir: str | Path, dry_run: bool = False
) -> dict:
    """Run a resumable M4/M5 profile × fold × seed screening sweep.

    Each child is the same validated official-fold runner. Failures are retained
    per configuration and do not hide successful or negative results.
    """
    configs, data_path, config_name = _sweep_configs(config_path)
    archive_sha256 = _sha256(data_path)
    if archive_sha256 is None:
        raise ValueError("data path must be an OnHW ZIP archive or extracted dataset directory")
    fingerprint = hashlib.sha256(
        _canonical_json(
            {
                "configs": configs,
                "archive_sha256": archive_sha256,
                "code_provenance": code_provenance(),
            }
        ).encode("utf-8")
    ).hexdigest()
    sweep_id = fingerprint[:20]
    root = Path(output_dir).resolve() / sweep_id
    configs_dir = root / "configs"
    runs_dir = root / "runs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "sweep_manifest.json"
    run_rows = []
    manifest = {
        "schema_version": 1,
        "sweep_id": sweep_id,
        "sweep_fingerprint": fingerprint,
        "status": "planned" if dry_run else "running",
        "configuration_file": config_name,
        "archive_sha256": archive_sha256,
        "run_count": len(configs),
        "dry_run": dry_run,
        "runs": [],
        "code_provenance": code_provenance(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(manifest_path, manifest)
    for config in configs:
        config_fingerprint = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
        config_id = config_fingerprint[:16]
        child_config_path = configs_dir / f"{config_id}.json"
        _atomic_json(child_config_path, config)
        row = {
            "configuration_id": config_id,
            "configuration": {
                key: config[key]
                for key in ("model_type", "sensor_profile", "fold", "seed")
            },
            "status": "running",
        }
        manifest["runs"] = [*run_rows, row]
        _atomic_json(manifest_path, manifest)
        try:
            result = run_experiment(
                config_path=child_config_path,
                output_dir=runs_dir / config_id,
                dry_run=dry_run,
                resume=True,
            )
            row["status"] = result["status"]
            row["run_id"] = result.get("run_id")
            row["run_manifest"] = str(
                Path("runs") / config_id / result.get("run_id", "") / "run_manifest.json"
            )
            eval_metrics = result.get("evaluation_metrics")
            row["metrics"] = eval_metrics.get("metrics") if isinstance(eval_metrics, dict) else None
            row["interpretation"] = result.get("interpretation")
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
        run_rows.append(row)
        manifest["runs"] = list(run_rows)
        _atomic_json(manifest_path, manifest)

    aggregation = _aggregate_sweep_results(run_rows)
    failures = sum(row["status"] == "failed" for row in run_rows)
    manifest.update(
        {
            "status": (
                "planned" if dry_run else ("complete_with_failures" if failures else "complete")
            ),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "failed_run_count": failures,
            "aggregation": aggregation,
            "aggregation_file": "sweep_aggregation.json",
        }
    )
    _atomic_json(root / "sweep_aggregation.json", aggregation)
    _atomic_json(manifest_path, manifest)
    return manifest


def run_writer_vocabulary_factorial(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    target_writers: list[int | str],
    fold: int = 0,
    query_per_writer: int = 1,
    support_per_writer: int = 5,
    training_samples_per_cell: int | None = None,
    allow_unbalanced: bool = True,
    sensor_profile: str = "front_acc_gyro6",
    model_type: str = "resnet_bilstm_ctc",
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 1337,
    device_name: str = "auto",
    num_threads: int = 4,
    model_config: dict | None = None,
) -> dict:
    """Train all four matched W × V cells and report paired writer-level effects.

    This is a custom development experiment over the selected official fold's
    records. Its results do not replace or emulate official WD/WI benchmarks.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")
    data_path = Path(data_path).resolve()
    archive_sha256 = _sha256(data_path)
    if archive_sha256 is None:
        raise ValueError("data path must be an OnHW archive or extracted dataset directory")
    fold_data = load_onhw_words500(data_path, fold)
    profile = resolve_sensor_profile(sensor_profile)
    all_records = [*fold_data.train, *fold_data.validation]
    cells = build_writer_vocabulary_factorial(
        all_records,
        target_writers=target_writers,
        query_per_writer=query_per_writer,
        support_per_writer=support_per_writer,
        training_samples_per_cell=training_samples_per_cell,
        allow_unbalanced=allow_unbalanced,
        seed=seed,
    )
    cell_training_sample_counts = {
        name: cell.manifest["training_sample_count"] for name, cell in sorted(cells.items())
    }
    training_samples_balanced = len(set(cell_training_sample_counts.values())) == 1
    design_config = {
        "protocol": "custom_writer_vocabulary_factorial",
        "archive_sha256": archive_sha256,
        "archive_name": data_path.name,
        "fold": fold,
        "target_writers": sorted(str(writer) for writer in target_writers),
        "query_per_writer": query_per_writer,
        "support_per_writer": support_per_writer,
        "training_samples_per_cell_requested": training_samples_per_cell,
        "training_samples_target": next(iter(cells.values())).manifest[
            "training_sample_target"
        ],
        "cell_training_sample_counts": cell_training_sample_counts,
        "training_samples_balanced": training_samples_balanced,
        "sensor_profile": profile.profile_id,
        "model_type": model_type,
        "model_config": model_config,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "device": device_name,
        "num_threads": num_threads,
        "code_provenance": code_provenance(),
        "runtime_provenance": _runtime_provenance(device_name),
        "query_sample_ids": next(iter(cells.values())).manifest["query_sample_ids"],
        "cell_train_sample_ids": {
            name: cell.manifest["training_sample_ids"] for name, cell in sorted(cells.items())
        },
    }
    fingerprint = hashlib.sha256(_canonical_json(design_config).encode("utf-8")).hexdigest()
    run_id = fingerprint[:20]
    root = Path(output_dir).resolve() / run_id
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "factorial_run_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != fingerprint:
            raise ValueError("factorial run directory contains a different configuration")
        aggregate_path = root / "factorial_comparison.json"
        cell_artifacts_valid = all(
            (root / f"{name}_run_manifest.json").is_file()
            and json.loads((root / f"{name}_run_manifest.json").read_text(encoding="utf-8")).get(
                "run_fingerprint"
            ) == fingerprint
            and json.loads((root / f"{name}_run_manifest.json").read_text(encoding="utf-8")).get(
                "status"
            ) == "complete"
            and json.loads((root / f"{name}_run_manifest.json").read_text(encoding="utf-8")).get(
                "metrics_sha256"
            ) == _sha256(root / name / "metrics.json")
            and json.loads((root / f"{name}_run_manifest.json").read_text(encoding="utf-8")).get(
                "checkpoint_sha256"
            ) == _sha256(root / name / "model.pt")
            for name in ("W0V0", "W0V1", "W1V0", "W1V1")
        )
        if (
            previous.get("status") == "complete"
            and aggregate_path.is_file()
            and cell_artifacts_valid
            and previous.get("comparison_sha256") == _sha256(aggregate_path)
        ):
            return {**previous, "status": "skipped_completed"}

    write_factorial_manifests(cells, root / "splits")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "run_fingerprint": fingerprint,
        "status": "running",
        "experiment": "Matched writer × vocabulary exposure factorial",
        "configuration": design_config,
        "code_provenance": code_provenance(),
        "runtime_provenance": _runtime_provenance(device_name),
        "split_manifests": "splits/factorial.json",
        "cells": {},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(manifest_path, manifest)
    try:
        reports: dict[str, dict] = {}
        for name, cell in sorted(cells.items()):
            cell_dir = root / name
            cell_marker = root / f"{name}_run_manifest.json"
            if cell_marker.is_file():
                previous_cell = json.loads(cell_marker.read_text(encoding="utf-8"))
                if (
                    previous_cell.get("run_fingerprint") == fingerprint
                    and previous_cell.get("status") == "complete"
                    and previous_cell.get("metrics_sha256") == _sha256(cell_dir / "metrics.json")
                    and previous_cell.get("checkpoint_sha256") == _sha256(cell_dir / "model.pt")
                ):
                    reports[name] = json.loads(
                        (cell_dir / "metrics.json").read_text(encoding="utf-8")
                    )
                    manifest["cells"][name] = {"status": "skipped_completed"}
                    _atomic_json(manifest_path, manifest)
                    continue

            recovered = _recover_completed_factorial_cell(
                cell_dir,
                archive_sha256=archive_sha256,
                fold=fold,
                seed=seed,
                model_type=model_type,
                profile_id=profile.profile_id,
                epochs=epochs,
                split_manifest=cell.manifest,
            )
            if recovered is not None:
                reports[name] = recovered
                marker = {
                    "schema_version": 1,
                    "run_fingerprint": fingerprint,
                    "cell": name,
                    "status": "complete",
                    "metrics_sha256": _sha256(cell_dir / "metrics.json"),
                    "checkpoint_sha256": _sha256(cell_dir / "model.pt"),
                    "selected_epoch": recovered["checkpoint_selection"]["selected_epoch"],
                    "recovered_after_interruption": True,
                }
                _atomic_json(cell_marker, marker)
                manifest["cells"][name] = {"status": "skipped_completed"}
                _atomic_json(manifest_path, manifest)
                continue

            override = OnHWFold(
                train=cell.training_records,
                validation=cell.query_records,
                fold=fold,
                split_family="custom_writer_vocabulary_factorial",
                archive_name=data_path.name,
            )
            report = run_training(
                data_path=data_path,
                output_dir=cell_dir,
                fold=fold,
                epochs=epochs,
                batch_size=batch_size,
                seed=seed,
                device_name=device_name,
                num_threads=num_threads,
                model_type=model_type,
                sensor_profile=profile.profile_id,
                training_stage="custom_writer_vocabulary_factorial",
                model_config=model_config,
                resume=True,
                records_override=override,
                custom_split_manifest=cell.manifest,
            )
            reports[name] = report
            marker = {
                "schema_version": 1,
                "run_fingerprint": fingerprint,
                "cell": name,
                "status": "complete",
                "metrics_sha256": _sha256(cell_dir / "metrics.json"),
                "checkpoint_sha256": _sha256(cell_dir / "model.pt"),
                "selected_epoch": report["checkpoint_selection"]["selected_epoch"],
            }
            _atomic_json(cell_marker, marker)
            manifest["cells"][name] = {
                "status": "complete",
                "metrics_sha256": marker["metrics_sha256"],
                "checkpoint_sha256": marker["checkpoint_sha256"],
            }
            _atomic_json(manifest_path, manifest)

        query_outputs = {
            name: reports[name]["validation_outputs"] for name in sorted(reports)
        }
        expected_ids = query_outputs["W0V0"]["sample_ids"]
        expected_references = query_outputs["W0V0"]["references"]
        expected_writers = query_outputs["W0V0"]["writer_ids"]
        for name, outputs in query_outputs.items():
            if (
                outputs["sample_ids"] != expected_ids
                or outputs["references"] != expected_references
                or outputs["writer_ids"] != expected_writers
            ):
                raise RuntimeError(f"factorial cell {name} evaluated different query records")
        paired_analysis = factorial_writer_bootstrap(
            {name: outputs["hypotheses"] for name, outputs in query_outputs.items()},
            expected_references,
            expected_writers,
            seed=seed,
        )
        comparison = {
            "schema_version": 1,
            "protocol": "custom_writer_vocabulary_factorial",
            "official_benchmark": False,
            "run_id": run_id,
            "dataset": "OnHW-Words500",
            "archive_sha256": archive_sha256,
            "fold": fold,
            "sensor_profile": profile.profile_id,
            "model_type": model_type,
            "seed": seed,
            "fixed_query_sample_ids": expected_ids,
            "fixed_query_references": expected_references,
            "fixed_query_writer_ids": expected_writers,
            "training_sample_counts": cell_training_sample_counts,
            "training_samples_balanced": training_samples_balanced,
            "paired_macro_writer_analysis": paired_analysis,
            "cells": {
                name: {
                    "manifest": f"splits/{name}.json",
                    "metrics": f"{name}/metrics.json",
                    "checkpoint": f"{name}/model.pt",
                    "final_metrics": reports[name]["final_metrics"],
                    "macro_writer_metrics": reports[name]["macro_writer_metrics"],
                    "per_writer_metrics": reports[name]["per_writer_metrics"],
                }
                for name in sorted(reports)
            },
            "interpretation": (
                "Custom matched development experiment only. W1 support is held fixed "
                "across vocabulary conditions; V1 vocabulary anchors come from eligible "
                "non-target writers. Estimates do not replace official WD/WI results."
                + (
                    " Training sample counts differ across cells, so exposure contrasts may "
                    "also reflect training-data quantity."
                    if not training_samples_balanced
                    else ""
                )
            ),
        }
        comparison_path = root / "factorial_comparison.json"
        _atomic_json(comparison_path, comparison)
        manifest.update(
            {
                "status": "complete",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "comparison": comparison_path.name,
                "comparison_sha256": _sha256(comparison_path),
                "paired_macro_writer_effects": paired_analysis["macro_writer_effects"],
            }
        )
        _atomic_json(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest.update(
            {
                "status": "failed_resumable" if any(
                    path.is_file() for path in root.glob("*/progress.pt")
                ) else "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise


def build_adaptation_manifests(
    *,
    data_path: str | Path,
    output_dir: str | Path,
    target_writer: int | str,
    fold: int = 0,
    query_count: int = 10,
    support_budgets: tuple[int, ...] = (1, 5, 10, 25, 50, 100),
    seed: int = 1337,
) -> dict:
    """Write leakage-checked P1–P6 writer-adaptation split manifests."""
    data_path = Path(data_path).resolve()
    archive_sha256 = _sha256(data_path)
    if archive_sha256 is None:
        raise ValueError("data path must be an OnHW archive or extracted dataset directory")
    fold_data = load_onhw_words500(data_path, fold)
    splits, summary = build_writer_adaptation_splits(
        [*fold_data.train, *fold_data.validation],
        target_writer=target_writer,
        query_count=query_count,
        support_budgets=support_budgets,
        seed=seed,
    )
    summary["dataset"] = "OnHW-Words500"
    summary["archive_name"] = data_path.name
    summary["archive_sha256"] = archive_sha256
    summary["official_source_protocol"] = fold_data.split_family
    summary["code_provenance"] = code_provenance()
    summary["runtime_provenance"] = _runtime_provenance("auto")
    summary["fold"] = fold
    write_writer_adaptation_manifests(splits, summary, output_dir)
    return summary
