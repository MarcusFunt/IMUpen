"""M6-C unlabeled personal data and hierarchical TS2Vec pretraining tests."""
import json

import numpy as np
import torch

from smartpen.hwr.data import (
    PERSONAL_CHARSET,
    load_unlabeled_personal_recordings,
)
from smartpen.hwr.personal import run_personal_ts2vec_comparison
from smartpen.hwr.ts2vec import (
    context_crop_bounds,
    hierarchical_contrastive_loss,
    mask_timestamps,
)


def _write_record(
    root, *, sample_id, session_id, seed, prompt=None, accepted=True, writer_id="p01"
):
    metadata_path = root / "metadata" / session_id / f"{sample_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    array_file = f"processed/{session_id}/{sample_id}.npz"
    array_path = root / array_file
    array_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(seed).normal(size=(48, 6)).astype(np.float32)
    np.savez_compressed(array_path, axes=values)
    metadata = {
        "schema_version": 1,
        "sample_id": sample_id,
        "writer_id": writer_id,
        "session_id": session_id,
        "accepted": accepted,
        "sample_count": len(values),
        "array_file": array_file,
    }
    if prompt is not None:
        metadata["prompt"] = prompt
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_unlabeled_loader_ignores_prompt_and_skips_rejected_records(tmp_path):
    _write_record(
        tmp_path,
        sample_id="unlabeled-01",
        session_id="day-01",
        seed=1,
    )
    _write_record(
        tmp_path,
        sample_id="rejected-01",
        session_id="day-02",
        seed=2,
        accepted=False,
    )

    result = load_unlabeled_personal_recordings(tmp_path)

    assert len(result.records) == 1
    assert result.records[0].text == ""
    assert result.records[0].features.shape == (48, 6)
    assert result.rejected_recordings == 1


def test_context_crops_keep_the_shared_sensor_segment_aligned():
    rng = np.random.default_rng(23)
    values = np.arange(101 * 6, dtype=np.float32).reshape(101, 6)

    start1, end1, start2, end2, overlap = context_crop_bounds(101, rng)
    view1 = values[start1:end1]
    view2 = values[start2:end2]
    shared = values[start2:end1]

    assert start1 % 4 == start2 % 4 == 0
    assert end1 <= end2 or start2 == start1
    assert overlap >= 2
    np.testing.assert_array_equal(view1[-len(shared) :], shared)
    np.testing.assert_array_equal(view2[: len(shared)], shared)


def test_hierarchical_contrastive_loss_is_finite_and_differentiable():
    torch.manual_seed(3)
    first = torch.randn(3, 9, 12, requires_grad=True)
    second = (first.detach() + torch.randn(3, 9, 12) * 0.2).requires_grad_()
    lengths = torch.tensor([9, 7, 3])

    loss = hierarchical_contrastive_loss(first, second, lengths)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(first.grad).all()
    assert torch.isfinite(second.grad).all()


def test_timestamp_mask_is_seeded_and_does_not_mutate_its_input():
    values = np.ones((20, 6), dtype=np.float32)
    first = mask_timestamps(values, np.random.default_rng(5))
    repeat = mask_timestamps(values, np.random.default_rng(5))

    np.testing.assert_array_equal(first, repeat)
    np.testing.assert_array_equal(values, np.ones_like(values))
    assert np.count_nonzero(np.all(first == 0, axis=1)) > 0


def test_m6c_excludes_validation_sessions_and_fine_tunes_both_initializations(tmp_path):
    labeled_root = tmp_path / "labeled"
    unlabeled_root = tmp_path / "unlabeled"
    for index, session in enumerate(("day-01", "day-02", "day-03")):
        _write_record(
            labeled_root,
            sample_id=f"labeled-{index}",
            session_id=session,
            seed=10 + index,
            prompt="A" if index % 2 == 0 else "B",
        )
        _write_record(
            unlabeled_root,
            sample_id=f"ssl-{index}",
            session_id=session,
            seed=20 + index,
            prompt="should be ignored",
        )
    # Extra unlabeled-only training session; it has no transcript.
    _write_record(
        unlabeled_root,
        sample_id="ssl-extra",
        session_id="day-04",
        seed=24,
    )
    _write_record(
        unlabeled_root,
        sample_id="ssl-other-writer",
        session_id="day-05",
        seed=25,
        writer_id="p02",
    )

    comparison = run_personal_ts2vec_comparison(
        recordings_dir=labeled_root,
        unlabeled_recordings_dir=unlabeled_root,
        output_dir=tmp_path / "m6c",
        validation_fraction=0.34,
        ssl_epochs=1,
        epochs=1,
        batch_size=2,
        seed=19,
        device_name="cpu",
        num_threads=1,
        model_config={
            "channels": [8, 16, 24],
            "hidden_size": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
        },
    )

    assert comparison["fairness_checks"]["same_labeled_train_and_validation_recordings"]
    assert comparison["fairness_checks"]["validation_sessions_excluded_from_self_supervised_pretraining"]
    assert comparison["unlabeled_validation_session_samples_excluded"] == 1
    assert comparison["unlabeled_other_writer_samples_excluded"] == 1
    assert comparison["models"]["personal_ts2vec_encoder"]["final_metrics"]["wer"] >= 0
    pretraining = torch.load(
        tmp_path / "m6c" / "personal_ts2vec_pretraining" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    fine_tuned = torch.load(
        tmp_path / "m6c" / "personal_ts2vec_encoder" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert pretraining["training_stage"] == "personal_ts2vec_pretraining"
    assert pretraining["training_report"]["final_loss"] > 0
    assert all(
        key.startswith(("stem.", "residual_blocks.", "recurrent."))
        for key in pretraining["encoder_state"]
    )
    assert fine_tuned["pretraining"]["projection_head_transferred"] is False
    assert fine_tuned["charset"] == PERSONAL_CHARSET
