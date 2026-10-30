"""Model, OnHW adapter, decoding, and metric contracts."""

import pickle
import platform
import zipfile

import numpy as np
import torch
import pytest

import smartpen.hwr.training as hwr_training
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
from smartpen.hwr.training import evaluate_checkpoint, run_comparison, run_training


def _write_tiny_onhw_archive(
    path, train_writer_ids=(1, 2), validation_writer_ids=(90, 91)
):
    with zipfile.ZipFile(path, "w") as archive:
        for split, writer_ids in (
            ("train", train_writer_ids),
            ("val", validation_writer_ids),
        ):
            values = [
                np.full((12, 13), index + 0.25, dtype=np.float32)
                for index, _writer_id in enumerate(writer_ids)
            ]
            labels = [
                encode_text("A" if index % 2 == 0 else "B")
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
                    f"Words500_indep_02/0/{name}", pickle.dumps(value, protocol=4)
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
    assert fold.train[0].features.shape == (12, 6)


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
