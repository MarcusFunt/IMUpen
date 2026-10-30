"""M6-A recorder-data loading, session splits, and encoder transfer tests."""
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from smartpen.hwr.data import (
    ONHW_BLANK_INDEX,
    PERSONAL_CHARSET,
    SequenceRecord,
    encode_text,
    load_personal_recordings,
    split_personal_by_session,
)
from smartpen.hwr.model import ResNetBiLSTMCTC, build_model
from smartpen.hwr.personal import (
    _apply_session_manifest,
    _session_manifest,
    load_pretrained_encoder,
    run_onhw_encoder_pretraining,
    run_personal_comparison,
)


def _write_personal_record(
    root, *, sample_id, session_id, text, axes=None, accepted=True, array_file=None
):
    metadata_path = root / "metadata" / session_id / f"{sample_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if array_file is None:
        array_file = Path("processed") / session_id / f"{sample_id}.npz"
        array_path = root / array_file
        array_path.parent.mkdir(parents=True, exist_ok=True)
        values = np.asarray(axes if axes is not None else np.ones((24, 6)), dtype=np.int16)
        np.savez_compressed(array_path, axes=values)
    metadata = {
        "schema_version": 1,
        "sample_id": sample_id,
        "writer_id": "p01",
        "session_id": session_id,
        "prompt": text,
        "accepted": accepted,
        "sample_count": len(axes) if axes is not None else 24,
        "array_file": str(array_file).replace("\\", "/"),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _write_tiny_onhw_archive(path):
    with zipfile.ZipFile(path, "w") as archive:
        for split, writer_ids in (("train", [1, 2]), ("val", [90, 91])):
            sequences = [np.ones((24, 13), dtype=np.float32) * index for index in writer_ids]
            labels = [
                encode_text("A" if index % 2 == 0 else "B")
                + [ONHW_BLANK_INDEX] * 18
                for index, _ in enumerate(writer_ids)
            ]
            members = {
                f"all_x_dat_{split}_imu.pkl": sequences,
                f"all_{split}_gt.pkl": labels,
                f"{split}_ids.pkl": writer_ids,
            }
            for name, value in members.items():
                archive.writestr(
                    f"Words500_indep_02/0/{name}", pickle.dumps(value, protocol=4)
                )


def test_default_personal_charset_encodes_danish_letters_and_space():
    encoded = encode_text("ÆØÅ æøå", PERSONAL_CHARSET)

    assert len(encoded) == len("ÆØÅ æøå")
    assert all(token < len(PERSONAL_CHARSET) for token in encoded)


def test_personal_loader_reads_accepted_six_axis_recordings_and_skips_rejected(tmp_path):
    values = np.arange(24 * 6, dtype=np.int16).reshape(24, 6)
    _write_personal_record(
        tmp_path, sample_id="sample-01", session_id="day-01", text="A B", axes=values
    )
    _write_personal_record(
        tmp_path,
        sample_id="sample-02",
        session_id="day-02",
        text="discard",
        accepted=False,
    )

    dataset = load_personal_recordings(tmp_path)

    assert len(dataset.records) == 1
    assert dataset.rejected_recordings == 1
    assert dataset.records[0].text == "A B"
    assert dataset.records[0].writer_id == "p01"
    assert dataset.records[0].session_id == "day-01"
    assert dataset.records[0].features.dtype == np.float32
    np.testing.assert_array_equal(dataset.records[0].features, values.astype(np.float32))


def test_personal_loader_rejects_recording_paths_that_escape_root(tmp_path):
    _write_personal_record(
        tmp_path,
        sample_id="escape",
        session_id="day-01",
        text="A",
        array_file="../../outside.npz",
    )

    with pytest.raises(ValueError, match="escapes dataset root"):
        load_personal_recordings(tmp_path)


def test_personal_split_is_seeded_and_keeps_writer_sessions_disjoint(tmp_path):
    records = [
        SequenceRecord(
            np.ones((24, 6), dtype=np.float32),
            "A",
            "p01",
            index,
            session_id,
            f"sample-{index}",
        )
        for index, session_id in enumerate(("day-01", "day-02", "day-03", "day-04"))
    ]

    first = split_personal_by_session(records, validation_fraction=0.25, seed=31)
    again = split_personal_by_session(records, validation_fraction=0.25, seed=31)

    assert first.train_sessions.isdisjoint(first.validation_sessions)
    assert first.train_sessions == again.train_sessions
    assert first.validation_sessions == again.validation_sessions
    manifest = _session_manifest(first, seed=31, validation_fraction=0.25)
    restored = _apply_session_manifest(
        records, manifest_path=_write_manifest(tmp_path, manifest)
    )
    assert {record.sample_id for record in restored.train} == {
        record.sample_id for record in first.train
    }


def _write_manifest(tmp_path, manifest):
    path = tmp_path / "personal_session_split_test.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_pretrained_encoder_copies_only_non_classifier_weights(tmp_path):
    source_config = {
        "num_classes": ONHW_BLANK_INDEX + 1,
        "input_channels": 6,
        "channels": [8, 16, 24],
        "hidden_size": 8,
        "lstm_layers": 1,
        "dropout": 0.0,
    }
    source = build_model("resnet_bilstm_ctc", source_config)
    checkpoint_path = tmp_path / "onhw-pretrain.pt"
    torch.save(
        {
            "model_type": "resnet_bilstm_ctc",
            "training_stage": "onhw_encoder_pretraining",
            "model_config": source_config,
            "model_state": source.state_dict(),
            "training_report": {"dataset": "OnHW-Words500 writer-independent"},
        },
        checkpoint_path,
    )
    target_config = dict(source_config, num_classes=len(PERSONAL_CHARSET) + 1)
    target = ResNetBiLSTMCTC(**target_config)
    target._m6_model_config = target_config
    initial_head = {key: value.clone() for key, value in target.classifier.state_dict().items()}

    metadata = load_pretrained_encoder(target, checkpoint_path)

    torch.testing.assert_close(target.stem[0].weight, source.stem[0].weight)
    torch.testing.assert_close(
        target.recurrent.weight_ih_l0, source.recurrent.weight_ih_l0
    )
    for key, value in target.classifier.state_dict().items():
        torch.testing.assert_close(value, initial_head[key])
    assert metadata["classifier_head_transferred"] is False


def test_m6a_personal_comparison_runs_both_initializations_on_held_out_sessions(tmp_path):
    onhw_archive = tmp_path / "OnHW-Words500_indep.zip"
    _write_tiny_onhw_archive(onhw_archive)
    small_model = {
        "channels": [8, 16, 24],
        "hidden_size": 8,
        "lstm_layers": 1,
        "dropout": 0.0,
    }
    pretrain = run_onhw_encoder_pretraining(
        data_path=onhw_archive,
        output_dir=tmp_path / "pretrain",
        epochs=1,
        batch_size=2,
        seed=19,
        device_name="cpu",
        num_threads=1,
        max_train_samples=2,
        max_validation_samples=2,
        model_config=small_model,
    )
    recordings = tmp_path / "recordings"
    for sample_id, session, text in (
        ("sample-01", "day-01", "A"),
        ("sample-02", "day-02", "B"),
        ("sample-03", "day-03", "A B"),
    ):
        _write_personal_record(
            recordings,
            sample_id=sample_id,
            session_id=session,
            text=text,
            axes=np.random.default_rng(1).normal(size=(24, 6)).astype(np.float32),
        )

    comparison = run_personal_comparison(
        recordings_dir=recordings,
        pretrained_checkpoint=tmp_path / "pretrain" / "model.pt",
        output_dir=tmp_path / "personal-runs",
        validation_fraction=0.34,
        epochs=1,
        batch_size=2,
        seed=19,
        device_name="cpu",
        num_threads=1,
        model_config=small_model,
    )

    assert pretrain["training_stage"] == "onhw_encoder_pretraining"
    assert comparison["fairness_checks"]["same_personal_train_and_validation_recordings"]
    assert comparison["fairness_checks"]["personal_session_leakage"] is False
    assert comparison["training_sample_count"] == 2
    assert comparison["validation_sample_count"] == 1
    assert comparison["models"]["onhw_pretrained_encoder"]["final_metrics"]["wer"] >= 0
    fine_tuned = torch.load(
        tmp_path / "personal-runs" / "onhw_pretrained_encoder" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert fine_tuned["charset"] == PERSONAL_CHARSET
    assert fine_tuned["pretraining"]["classifier_head_transferred"] is False
