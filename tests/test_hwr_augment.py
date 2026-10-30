"""M6-B IMU augmentation semantics and ablation fairness tests."""
import pickle
import zipfile

import numpy as np

import smartpen.hwr.training as hwr_training
from smartpen.hwr.augment import (
    ONLINE_AUGMENTATIONS,
    apply_augmentation,
    concatenate_training_records,
)
from smartpen.hwr.data import ONHW_BLANK_INDEX, SequenceRecord, encode_text
from smartpen.hwr.training import (
    _AugmentedRecordDataset,
    run_augmentation_ablation,
    run_training,
)


def _record(text="A", writer_id="w1", session_id="s1", sample_index=0, seed=0):
    rng = np.random.default_rng(seed)
    return SequenceRecord(
        features=rng.normal(size=(40, 6)).astype(np.float32),
        text=text,
        writer_id=writer_id,
        session_id=session_id,
        sample_index=sample_index,
        sample_id=f"sample-{sample_index}",
    )


def test_each_online_augmentation_is_seeded_and_preserves_text():
    record = _record()
    scale = np.ones(6, dtype=np.float32)

    for augmentation in ONLINE_AUGMENTATIONS:
        first = apply_augmentation(
            record, augmentation, np.random.default_rng(813), scale
        )
        repeat = apply_augmentation(
            record, augmentation, np.random.default_rng(813), scale
        )

        assert first.text == record.text
        assert first.writer_id == record.writer_id
        assert first.features.shape[1] == 6
        np.testing.assert_array_equal(first.features, repeat.features)


def test_rotation_applies_one_rigid_transform_to_acceleration_and_gyro():
    values = np.array(
        [[1.0, 2.0, 3.0, -2.0, 1.0, 0.5], [4.0, -1.0, 2.0, 0.0, 3.0, -1.0]],
        dtype=np.float32,
    )
    record = SequenceRecord(values, "A", "w1")

    rotated = apply_augmentation(
        record, "rotation_3d", np.random.default_rng(6), np.ones(6)
    ).features

    for start in (0, 3):
        np.testing.assert_allclose(
            np.linalg.norm(values[:, start : start + 3], axis=1),
            np.linalg.norm(rotated[:, start : start + 3], axis=1),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            values[0, start : start + 3] @ values[1, start : start + 3],
            rotated[0, start : start + 3] @ rotated[1, start : start + 3],
            rtol=1e-6,
        )


def test_concatenation_is_reproducible_and_stays_within_writer_session():
    records = [
        _record("A", "w1", "s1", 0, 1),
        _record("B", "w1", "s1", 1, 2),
        _record("C", "w1", "s2", 2, 3),
        _record("D", "w2", "s1", 3, 4),
    ]

    combined = concatenate_training_records(records, seed=91, gap_frames=4)
    repeated = concatenate_training_records(records, seed=91, gap_frames=4)

    assert len(combined) == 2
    assert all(record.writer_id == "w1" for record in combined)
    assert all(record.session_id == "s1" for record in combined)
    assert {record.text for record in combined} == {"AB", "BA"}
    assert all(len(record.features) == 40 + 4 + 40 for record in combined)
    for first, second in zip(combined, repeated):
        np.testing.assert_array_equal(first.features, second.features)


def test_online_dataset_varies_by_epoch_but_repeats_for_same_seed():
    record = _record()
    kwargs = {
        "transforms": ("gaussian_noise",),
        "training_std": np.ones(6, dtype=np.float32),
        "seed": 44,
        "charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß",
    }
    first_dataset = _AugmentedRecordDataset([record], **kwargs)
    repeat_dataset = _AugmentedRecordDataset([record], **kwargs)
    epoch_zero = first_dataset[0].features
    assert np.array_equal(epoch_zero, repeat_dataset[0].features)

    first_dataset.set_epoch(1)
    assert not np.array_equal(epoch_zero, first_dataset[0].features)


def test_ablation_runner_uses_common_data_and_records_all_independent_variants(
    tmp_path, monkeypatch
):
    def fake_run_training(**kwargs):
        variant = kwargs["augmentation"]
        return {
            "run_kind": "sample",
            "dataset": "OnHW-Words500 writer-independent",
            "archive_sha256": "archive",
            "fold": kwargs["fold"],
            "seed": kwargs["seed"],
            "sample_indices": {"train": [1, 2], "validation": [3]},
            "train_samples_used": 2,
            "train_sequences_after_augmentation": 2 + (variant == "sequence_concat"),
            "sequence_concatenations_added": int(variant == "sequence_concat"),
            "sequence_concatenations_dropped_ctc": 0,
            "validation_samples_used": 1,
            "writer_leakage": False,
            "normalization": {"mean": [0.0], "std": [1.0]},
            "model_type": kwargs["model_type"],
            "model": {"type": kwargs["model_type"]},
            "model_parameters": 20,
            "batch_size": kwargs["batch_size"],
            "epochs": [{
                "epoch": 1,
                "validation_cer": 1.0,
                "validation_wer": 1.0,
                "character_errors": 1,
                "reference_characters": 1,
                "word_errors": 1,
                "reference_words": 1,
            }],
            "mixed_precision": False,
            "num_threads": kwargs["num_threads"],
            "device": "cpu",
            "optimizer": "AdamW",
            "python_version": "3.13",
            "torch_version": "2.13",
            "numpy_version": "2.2",
            "deterministic_algorithms": True,
            "augmentation": {
                "name": variant,
                "applied_only_to_training": True,
                "normalization_fit_before_augmentation": True,
            },
            "final_metrics": {"cer": 1.0, "wer": 1.0},
        }

    monkeypatch.setattr(hwr_training, "run_training", fake_run_training)

    report = run_augmentation_ablation(
        data_path="unused.zip",
        output_dir=tmp_path / "ablation",
        epochs=1,
        batch_size=2,
        seed=17,
        device_name="cpu",
        max_train_samples=2,
        max_validation_samples=1,
        num_threads=1,
    )

    assert set(report["variants"]) == {
        "none",
        "temporal_stretch",
        "gaussian_noise",
        "sensor_bias_gain",
        "rotation_3d",
        "sequence_concat",
    }
    assert report["fairness_checks"]["same_archive_fold_and_sample_indices"]
    assert report["fairness_checks"]["validation_not_augmented"]
    assert report["fairness_checks"]["writer_leakage"] is False
    assert (tmp_path / "ablation" / "augmentation_ablation.json").is_file()


def test_sequence_concat_runs_through_ctc_training_without_touching_validation(tmp_path):
    archive_path = tmp_path / "tiny-onhw.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for split, writer_ids in (("train", (1, 1)), ("val", (90, 90))):
            sequences = [
                np.full((32, 13), index + 0.25, dtype=np.float32)
                for index, _ in enumerate(writer_ids)
            ]
            labels = [
                encode_text("A" if index == 0 else "B")
                + [ONHW_BLANK_INDEX] * 18
                for index, _ in enumerate(writer_ids)
            ]
            members = {
                f"all_x_dat_{split}_imu.pkl": sequences,
                f"all_{split}_gt.pkl": labels,
                f"{split}_ids.pkl": list(writer_ids),
            }
            for name, value in members.items():
                archive.writestr(
                    f"Words500_indep_02/0/{name}", pickle.dumps(value, protocol=4)
                )

    report = run_training(
        data_path=archive_path,
        output_dir=tmp_path / "concat-run",
        epochs=1,
        batch_size=2,
        seed=11,
        device_name="cpu",
        num_threads=1,
        augmentation="sequence_concat",
        model_config={"conv_channels": 8, "hidden_size": 8, "lstm_layers": 1},
    )

    assert report["train_samples_used"] == 2
    assert report["train_sequences_after_augmentation"] == 4
    assert report["sequence_concatenations_added"] == 2
    assert report["validation_samples_used"] == 2
    assert report["augmentation"]["applied_only_to_training"] is True
    assert report["writer_leakage"] is False
