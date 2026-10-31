"""Model, OnHW adapter, decoding, and metric contracts."""

import json
import pickle
import platform
import zipfile

import numpy as np
import torch
import pytest

import smartpen.hwr.training as hwr_training
from smartpen.hwr.experiments import _sweep_configs, run_experiment, run_sweep
from smartpen.hwr.data import (
    ONHW_BLANK_INDEX,
    ONHW_CHARSET,
    decode_onhw_label,
    encode_text,
    load_onhw_words500,
    select_six_axis,
)
from smartpen.hwr.metrics import corpus_error_rates
from smartpen.hwr.model import (
    CNNBiLSTMCTC,
    ResNetBiLSTMCTC,
    build_model,
    count_parameters,
    ctc_greedy_decode,
    encoded_length,
)
from smartpen.hwr.sensors import (
    ONHW_CHANNEL_NAMES,
    SENSOR_PROFILES,
    extract_profile,
    resolve_sensor_profile,
)
from smartpen.hwr.training import evaluate_checkpoint, run_comparison, run_training


def _write_tiny_onhw_archive(
    path, train_writer_ids=(1, 2), validation_writer_ids=(90, 91)
):
    variant = "dep" if "_dep" in path.name.lower() else "indep"
    with zipfile.ZipFile(path, "w") as archive:
        for split, writer_ids in (
            ("train", train_writer_ids),
            ("val", validation_writer_ids),
        ):
            values = [
                np.full((12, 13), index + (0.25 if split == "train" else 10.25), dtype=np.float32)
                for index, _writer_id in enumerate(writer_ids)
            ]
            labels = [
                encode_text(
                    ("A" if index % 2 == 0 else "B")
                    if split == "train"
                    else ("C" if index % 2 == 0 else "D")
                )
                + [ONHW_BLANK_INDEX] * 18
                for index, _writer_id in enumerate(writer_ids)
            ]
            members = {
                f"all_x_dat_{split}_imu.pkl": values,
                f"all_{split}_gt.pkl": labels,
                f"{split}_ids.pkl": list(writer_ids),
            }
            for name, value in members.items():
                archive.writestr(
                    f"Words500_{variant}_02/0/{name}", pickle.dumps(value, protocol=4)
                )


def test_onhw_label_padding_is_removed_and_charset_is_preserved():
    assert len(ONHW_CHARSET) == 59
    assert ONHW_BLANK_INDEX == 59
    assert decode_onhw_label([32, 30, 43, 26, 29, 30, 59, 59]) == "gerade"


def test_onhw_adapter_selects_front_accelerometer_and_gyroscope_axes():
    sample = np.arange(2 * 13, dtype=np.float64).reshape(2, 13)

    adapted = select_six_axis(sample)

    np.testing.assert_array_equal(adapted, sample[:, [0, 1, 2, 6, 7, 8]])
    assert adapted.dtype == np.float32


def test_onhw_zip_loader_keeps_official_writer_independent_partitions(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    _write_tiny_onhw_archive(
        archive_path, train_writer_ids=(12,), validation_writer_ids=(97,)
    )

    fold = load_onhw_words500(archive_path, fold=0)

    assert fold.train[0].text == "A"
    assert fold.validation[0].writer_id == 97
    assert fold.train_writers.isdisjoint(fold.validation_writers)
    assert fold.train[0].features.shape == (12, 13)
    assert fold.train[0].channel_names == ONHW_CHANNEL_NAMES
    assert fold.train[0].sample_rate_hz == 100.0
    assert fold.train[0].archive_name == archive_path.name
    assert fold.train[0].source_id is None
    assert fold.train[0].sample_id == (
        "onhw-words500:OnHW-Words500_indep.zip:writer_independent:fold-0:train:0"
    )


def test_onhw_writer_dependent_loader_accepts_shared_writers_and_disjoint_words(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_dep.zip"
    _write_tiny_onhw_archive(
        archive_path, train_writer_ids=(12, 13), validation_writer_ids=(12, 13)
    )

    fold = load_onhw_words500(archive_path, fold=0)

    assert fold.split_family == "writer_dependent"
    assert fold.train_writers == fold.validation_writers == {12, 13}
    assert {record.text for record in fold.train}.isdisjoint(
        record.text for record in fold.validation
    )
    assert fold.validation[0].partition == "val"


def test_onhw_writer_independent_loader_rejects_writer_leakage(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for split in ("train", "val"):
            members = {
                f"all_x_dat_{split}_imu.pkl": [np.zeros((6, 13), dtype=np.float32)],
                f"all_{split}_gt.pkl": [encode_text("A") + [ONHW_BLANK_INDEX] * 18],
                f"{split}_ids.pkl": [4],
            }
            for name, value in members.items():
                archive.writestr(
                    f"Words500_indep_02/0/{name}", pickle.dumps(value, protocol=4)
                )

    with pytest.raises(ValueError, match="leaks 1 writers"):
        load_onhw_words500(archive_path, fold=0)


def test_onhw_loader_rejects_archive_name_and_internal_variant_disagreement(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for split, writer in (("train", 1), ("val", 2)):
            values = {
                f"all_x_dat_{split}_imu.pkl": [np.ones((8, 13), dtype=np.float32) * writer],
                f"all_{split}_gt.pkl": [encode_text("A") + [ONHW_BLANK_INDEX] * 18],
                f"{split}_ids.pkl": [writer],
            }
            for name, value in values.items():
                archive.writestr(
                    f"Words500_dep_02/0/{name}", pickle.dumps(value, protocol=4)
                )

    with pytest.raises(ValueError, match="filename indicates writer_independent"):
        load_onhw_words500(archive_path, fold=0)


def test_training_and_checkpoint_evaluation_run_on_writer_held_out_fold(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    _write_tiny_onhw_archive(archive_path)

    report = run_training(
        data_path=archive_path,
        output_dir=tmp_path / "run",
        epochs=1,
        batch_size=2,
        seed=17,
        device_name="cpu",
        num_threads=1,
        max_train_samples=2,
        max_validation_samples=2,
        model_config={
            "conv_channels": 8,
            "hidden_size": 8,
            "lstm_layers": 1,
        },
    )
    result = evaluate_checkpoint(
        checkpoint_path=tmp_path / "run" / "model.pt",
        data_path=archive_path,
        max_validation_samples=2,
        seed=17,
        device_name="cpu",
        batch_size=2,
        num_threads=1,
    )

    assert report["writer_leakage"] is False
    assert report["final_metrics"]["cer"] >= 0.0
    assert result["writer_leakage"] is False
    assert result["validation_samples"] == 2
    assert (tmp_path / "run" / "metrics.json").is_file()


def test_resnet_bilstm_default_matches_m5_parameter_budget_and_downsampling():
    model = ResNetBiLSTMCTC(num_classes=ONHW_BLANK_INDEX + 1)
    inputs = torch.randn(2, 25, 6)
    lengths = torch.tensor([25, 17])

    logits, output_lengths = model(inputs, lengths)

    assert 4_000_000 <= count_parameters(model) <= 8_000_000
    assert logits.shape == (2, 7, ONHW_BLANK_INDEX + 1)
    assert output_lengths.tolist() == [7, 5]


def test_resnet_bilstm_padding_does_not_change_short_sequence_logits():
    model = ResNetBiLSTMCTC(
        num_classes=ONHW_BLANK_INDEX + 1,
        channels=(8, 16, 24),
        hidden_size=8,
        lstm_layers=1,
        dropout=0.0,
    ).eval()
    short = torch.randn(1, 17, 6)
    batched = torch.zeros(2, 25, 6)
    batched[0, :17] = short
    batched[1] = torch.randn(25, 6)

    with torch.no_grad():
        alone, _ = model(short, torch.tensor([17]))
        together, _ = model(batched, torch.tensor([17, 25]))

    torch.testing.assert_close(alone[0, :5], together[0, :5], atol=1e-6, rtol=1e-6)


def test_resnet_checkpoint_trains_and_evaluates_through_shared_pipeline(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    _write_tiny_onhw_archive(archive_path)

    report = run_training(
        data_path=archive_path,
        output_dir=tmp_path / "resnet-run",
        epochs=1,
        batch_size=2,
        seed=23,
        device_name="cpu",
        num_threads=1,
        max_train_samples=2,
        max_validation_samples=2,
        model_type="resnet_bilstm_ctc",
        model_config={
            "channels": [8, 16, 24],
            "hidden_size": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
        },
    )
    result = evaluate_checkpoint(
        checkpoint_path=tmp_path / "resnet-run" / "model.pt",
        data_path=archive_path,
        max_validation_samples=2,
        seed=23,
        device_name="cpu",
        batch_size=2,
        num_threads=1,
    )

    assert report["model_type"] == "resnet_bilstm_ctc"
    assert report["writer_leakage"] is False
    assert result["model_type"] == "resnet_bilstm_ctc"
    assert result["writer_leakage"] is False
    assert result["validation_samples"] == 2
    assert result["metrics"]["cer"] >= 0.0


def test_full_13_channel_m5_checkpoint_trains_and_evaluates(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    _write_tiny_onhw_archive(archive_path)

    report = run_training(
        data_path=archive_path,
        output_dir=tmp_path / "full13-run",
        epochs=1,
        batch_size=2,
        seed=31,
        device_name="cpu",
        num_threads=1,
        max_train_samples=2,
        max_validation_samples=2,
        model_type="resnet_bilstm_ctc",
        sensor_profile="full13",
        model_config={
            "channels": [8, 16, 24],
            "hidden_size": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
        },
    )
    evaluation = evaluate_checkpoint(
        checkpoint_path=tmp_path / "full13-run" / "model.pt",
        data_path=archive_path,
        device_name="cpu",
        batch_size=2,
        num_threads=1,
    )

    checkpoint = torch.load(
        tmp_path / "full13-run" / "model.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["model_config"]["input_channels"] == 13
    assert report["sensor_profile"]["channel_indices"] == list(range(13))
    assert report["checkpoint_selection"]["selected_epoch"] == 1
    assert evaluation["sensor_profile_id"] == "full13"
    assert report["final_metrics"]["sample_count"] == 2
    assert report["macro_writer_metrics"]["wer"] >= 0


def test_ctc_decoder_collapses_repeats_only_across_nonblank_runs():
    assert ctc_greedy_decode([0, 0, 59, 0, 1, 1, 59, 1]) == "AABB"


def test_corpus_cer_and_wer_use_aggregate_edit_counts():
    rates = corpus_error_rates(["hello", "cut"], ["Hello", "cat"])

    assert rates["cer"] == 2 / 8
    assert rates["wer"] == 1.0
    assert rates["character_errors"] == 2
    assert rates["word_errors"] == 2


def test_cnn_bilstm_output_lengths_match_two_stride_two_convolutions():
    model = CNNBiLSTMCTC(
        num_classes=ONHW_BLANK_INDEX + 1,
        conv_channels=8,
        hidden_size=8,
        lstm_layers=1,
    )
    inputs = torch.randn(2, 25, 6)
    lengths = torch.tensor([25, 17])

    logits, output_lengths = model(inputs, lengths)

    assert logits.shape == (2, 7, ONHW_BLANK_INDEX + 1)
    assert output_lengths.tolist() == [encoded_length(25), encoded_length(17)]
    assert output_lengths.tolist() == [7, 5]


@pytest.mark.parametrize("profile", SENSOR_PROFILES, ids=lambda item: item.profile_id)
def test_every_measured_sensor_profile_selects_registered_channels(profile):
    sequence = np.arange(3 * 13, dtype=np.float32).reshape(3, 13)

    selected = extract_profile(sequence, profile)

    np.testing.assert_array_equal(selected, sequence[:, profile.channel_indices])
    assert selected.shape == (3, profile.input_channels)
    assert len(profile.channel_names) == profile.input_channels
    assert resolve_sensor_profile(profile.group_code) == profile


def test_sensor_registry_contains_exactly_all_31_nonempty_group_subsets():
    assert len(SENSOR_PROFILES) == 31
    assert len({profile.group_code for profile in SENSOR_PROFILES}) == 31
    with pytest.raises(ValueError, match="unknown OnHW sensor profile"):
        resolve_sensor_profile("second_gyro")
    rear = resolve_sensor_profile("RG")
    both_accelerometers = resolve_sensor_profile("FRGM")
    assert rear.profile_id == "rear_acc_shared_gyro6"
    assert rear.channel_indices == (3, 4, 5, 6, 7, 8)
    assert "single shared gyroscope" in rear.physical_description
    assert both_accelerometers.channel_indices == tuple(range(12))
    assert "not a rear-mounted gyro" in both_accelerometers.physical_description
    assert dict(rear.preprocessing)["normalization"] == "train_partition_zscore"


@pytest.mark.parametrize("channels", [1, 3, 4, 6, 7, 9, 10, 12, 13])
def test_m4_and_m5_accept_variable_input_dimensions(channels):
    for model_type, config in (
        (
            "cnn_bilstm_ctc",
            {"conv_channels": 4, "hidden_size": 4, "lstm_layers": 1, "dropout": 0.0},
        ),
        (
            "resnet_bilstm_ctc",
            {"channels": [4, 6, 8], "hidden_size": 4, "lstm_layers": 1, "dropout": 0.0},
        ),
    ):
        model = build_model(
            model_type,
            {"num_classes": ONHW_BLANK_INDEX + 1, "input_channels": channels, **config},
        )
        logits, lengths = model(torch.randn(2, 20, channels), torch.tensor([20, 16]))
        assert logits.shape == (2, 5, ONHW_BLANK_INDEX + 1)
        assert lengths.tolist() == [5, 4]


def test_cnn_bilstm_padding_does_not_change_short_sequence_logits():
    model = CNNBiLSTMCTC(
        num_classes=ONHW_BLANK_INDEX + 1,
        conv_channels=8,
        hidden_size=8,
        lstm_layers=1,
    ).eval()
    short = torch.randn(1, 17, 6)
    batched = torch.zeros(2, 25, 6)
    batched[0, :17] = short
    batched[1] = torch.randn(25, 6)

    with torch.no_grad():
        alone, _ = model(short, torch.tensor([17]))
        together, _ = model(batched, torch.tensor([17, 25]))

    torch.testing.assert_close(alone[0, :5], together[0, :5], atol=1e-6, rtol=1e-6)


def test_build_model_selects_the_saved_architecture():
    model = build_model(
        "resnet_bilstm_ctc",
        {
            "num_classes": ONHW_BLANK_INDEX + 1,
            "channels": [8, 16, 24],
            "hidden_size": 8,
            "lstm_layers": 1,
        },
    )

    assert isinstance(model, ResNetBiLSTMCTC)


def test_comparison_records_identical_split_and_common_settings(tmp_path, monkeypatch):
    def fake_run_training(**kwargs):
        return {
            "run_kind": "sample",
            "dataset": "OnHW-Words500 writer-independent",
            "protocol": "writer_independent",
            "sensor_profile": {"profile_id": kwargs["sensor_profile"]},
            "archive_sha256": "same-archive",
            "fold": kwargs["fold"],
            "seed": kwargs["seed"],
            "sample_indices": {"train": [1, 3], "validation": [4]},
            "train_samples_used": 2,
            "validation_samples_used": 1,
            "train_writers_sampled": 2,
            "validation_writers_sampled": 1,
            "writer_leakage": False,
            "epochs": [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "validation_loss": 1.0,
                    "character_errors": 1,
                    "reference_characters": 1,
                    "word_errors": 1,
                    "reference_words": 1,
                }
            ],
            "mixed_precision": False,
            "num_threads": kwargs["num_threads"],
            "batch_size": kwargs["batch_size"],
            "device": "cpu",
            "device_name": None,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "deterministic_algorithms": True,
            "normalization": {"mean": [0.0], "std": [1.0]},
            "cuda_peak_memory_bytes": None,
            "model_parameters": 10 if kwargs["model_type"] == "cnn_bilstm_ctc" else 20,
            "model": {"type": kwargs["model_type"]},
            "final_metrics": {"cer": 1.0, "wer": 1.0},
        }

    monkeypatch.setattr(hwr_training, "run_training", fake_run_training)

    report = run_comparison(
        data_path="unused.zip",
        output_dir=tmp_path / "comparison",
        epochs=1,
        batch_size=2,
        seed=17,
        device_name="cpu",
        max_train_samples=2,
        max_validation_samples=1,
        num_threads=1,
    )

    assert report["fairness_checks"]["same_official_fold_and_sample_indices"] is True
    assert report["fairness_checks"]["writer_leakage"] is False
    assert report["models"]["cnn_bilstm_ctc"]["parameters"] == 10
    assert report["models"]["resnet_bilstm_ctc"]["parameters"] == 20
    assert (tmp_path / "comparison" / "comparison.json").is_file()


def test_training_resumes_from_the_last_completed_epoch(tmp_path, monkeypatch):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    _write_tiny_onhw_archive(archive_path)
    original_run_epoch = hwr_training._run_epoch
    calls = 0

    def interrupt_on_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return original_run_epoch(*args, **kwargs)

    monkeypatch.setattr(hwr_training, "_run_epoch", interrupt_on_second_epoch)
    run_args = {
        "data_path": archive_path,
        "output_dir": tmp_path / "resume-run",
        "epochs": 2,
        "batch_size": 2,
        "seed": 37,
        "device_name": "cpu",
        "num_threads": 1,
        "max_train_samples": 2,
        "max_validation_samples": 2,
        "model_config": {
            "conv_channels": 8,
            "hidden_size": 8,
            "lstm_layers": 1,
        },
        "resume": True,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_training(**run_args)
    assert (tmp_path / "resume-run" / "progress.pt").is_file()

    monkeypatch.setattr(hwr_training, "_run_epoch", original_run_epoch)
    report = run_training(**run_args)
    assert report["resumed_from_epoch"] == 1
    assert len(report["epochs"]) == 2
    assert not (tmp_path / "resume-run" / "progress.pt").exists()


def test_json_experiment_runner_dry_run_executes_and_skips_completed_run(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    _write_tiny_onhw_archive(archive_path)
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "data": archive_path.name,
                "fold": 0,
                "sensor_profile": "front_acc_gyro6",
                "seed": 43,
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
                "num_threads": 1,
                "max_train_samples": 2,
                "max_validation_samples": 2,
                "protocol": "writer_independent",
            }
        ),
        encoding="utf-8",
    )

    plan = run_experiment(config_path, tmp_path / "experiments", dry_run=True)
    assert plan["status"] == "planned"
    completed = run_experiment(config_path, tmp_path / "experiments")
    assert completed["status"] == "complete"
    assert completed["experiment"] == "OnHW official-fold M5 supervised CTC"
    assert (tmp_path / "experiments" / completed["run_id"] / "evaluation.json").is_file()
    skipped = run_experiment(config_path, tmp_path / "experiments")
    assert skipped["status"] == "skipped_completed"


def test_json_sweep_expands_models_and_profiles_and_records_dry_run(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    _write_tiny_onhw_archive(archive_path)
    config_path = tmp_path / "sweep.json"
    config_path.write_text(
        json.dumps(
            {
                "data": archive_path.name,
                "sensor_profiles": ["FG", "RG"],
                "model_types": ["cnn_bilstm_ctc", "resnet_bilstm_ctc"],
                "folds": [0],
                "seeds": [43],
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
                "num_threads": 1,
                "max_train_samples": 2,
                "max_validation_samples": 2,
                "protocol": "writer_independent",
            }
        ),
        encoding="utf-8",
    )

    report = run_sweep(
        config_path=config_path,
        output_dir=tmp_path / "sweeps",
        dry_run=True,
    )

    assert report["status"] == "planned"
    assert report["run_count"] == 4
    assert report["failed_run_count"] == 0
    assert len(report["runs"]) == 4
    assert all(run["status"] == "planned" for run in report["runs"])
    assert {run["configuration"]["sensor_profile"] for run in report["runs"]} == {
        "front_acc_gyro6",
        "rear_acc_shared_gyro6",
    }
    assert {run["configuration"]["model_type"] for run in report["runs"]} == {
        "cnn_bilstm_ctc",
        "resnet_bilstm_ctc",
    }
    assert (tmp_path / "sweeps" / report["sweep_id"] / "sweep_aggregation.json").is_file()

    config_path.write_text(
        json.dumps(
            {
                "data": archive_path.name,
                "sensor_profiles": "all",
                "model_types": ["resnet_bilstm_ctc"],
                "folds": [0],
                "seeds": [43],
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
                "num_threads": 1,
                "max_train_samples": 2,
                "max_validation_samples": 2,
                "protocol": "writer_independent",
            }
        ),
        encoding="utf-8",
    )
    all_profile_configs, _, _ = _sweep_configs(config_path)
    assert len(all_profile_configs) == 31

    config_path.write_text(
        json.dumps(
            {
                "data": archive_path.name,
                "sensor_profiles": ["FG"],
                "model_types": ["cnn_bilstm_ctc"],
                "folds": [0],
                "seeds": [43],
                "epochs": 1,
                "batch_size": 2,
                "device": "cpu",
                "num_threads": 1,
                "max_train_samples": 2,
                "max_validation_samples": 2,
                "protocol": "writer_independent",
            }
        ),
        encoding="utf-8",
    )
    executed = run_sweep(config_path=config_path, output_dir=tmp_path / "sweeps")
    assert executed["status"] == "complete"
    assert executed["aggregation"]["profile_summaries"]
    assert any(
        comparison["paired_run_count"] == 0
        for comparison in executed["aggregation"]["primary_sensor_comparisons"]
    )
    resumed = run_sweep(config_path=config_path, output_dir=tmp_path / "sweeps")
    assert resumed["status"] == "complete"
    assert resumed["runs"][0]["status"] == "skipped_completed"
