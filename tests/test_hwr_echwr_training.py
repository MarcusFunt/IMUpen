"""M6-D personal training comparison integration tests."""
import json

import numpy as np
import torch

from smartpen.hwr.echwr_training import run_personal_echwr_comparison


def _write_record(root, *, sample_id, session_id, seed, prompt, writer_id="p01"):
    metadata_path = root / "metadata" / session_id / f"{sample_id}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    array_file = f"processed/{session_id}/{sample_id}.npz"
    array_path = root / array_file
    array_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(seed).normal(size=(48, 6)).astype(np.float32)
    np.savez_compressed(array_path, axes=values)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "writer_id": writer_id,
                "session_id": session_id,
                "accepted": True,
                "sample_count": len(values),
                "array_file": array_file,
                "prompt": prompt,
            }
        ),
        encoding="utf-8",
    )


def test_m6d_runs_fair_session_held_out_comparison_and_saves_encoder_only(tmp_path):
    recordings = tmp_path / "recordings"
    for index, prompt in enumerate(("A", "B", "A", "B")):
        _write_record(
            recordings,
            sample_id=f"sample-{index}",
            session_id=f"day-{index + 1}",
            seed=30 + index,
            prompt=prompt,
        )

    comparison = run_personal_echwr_comparison(
        recordings_dir=recordings,
        output_dir=tmp_path / "m6d",
        charset="AB",
        validation_fraction=0.25,
        epochs=1,
        batch_size=2,
        seed=29,
        device_name="cpu",
        num_threads=1,
        model_config={
            "channels": [8, 16, 24],
            "hidden_size": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
        },
    )

    assert comparison["fairness_checks"]["same_personal_train_and_validation_recordings"]
    assert comparison["fairness_checks"]["same_writer_session_split"]
    assert comparison["fairness_checks"]["same_encoder_initialization"]
    assert comparison["fairness_checks"]["auxiliary_branch_removed_from_inference"]
    assert set(comparison["models"]) == {
        "ctc",
        "ctc_batch_contrastive",
        "ctc_batch_and_error_contrastive",
    }
    assert all(model["final_metrics"]["cer"] >= 0 for model in comparison["models"].values())

    baseline = torch.load(
        tmp_path / "m6d" / "ctc" / "model.pt", map_location="cpu", weights_only=True
    )
    contrastive = torch.load(
        tmp_path / "m6d" / "ctc_batch_and_error_contrastive" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert baseline["training_auxiliary_state"] is None
    assert contrastive["training_auxiliary_state"]
    assert set(contrastive["model_state"]) == set(baseline["model_state"])
    assert contrastive["training_stage"] == "personal_echwr_finetuning"
