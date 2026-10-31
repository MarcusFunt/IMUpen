import json
import pickle
import zipfile

import numpy as np

from smartpen.hwr.data import SequenceRecord
from smartpen.hwr.data import ONHW_BLANK_INDEX, encode_text
from smartpen.hwr.experiments import run_writer_vocabulary_factorial
from smartpen.hwr.protocols import (
    build_writer_adaptation_splits,
    build_writer_vocabulary_factorial,
    write_factorial_manifests,
    write_writer_adaptation_manifests,
)


def _factorial_records():
    records = []
    words = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for writer in (1, 2, 3, 4):
        for index, word in enumerate(words):
            features = np.full((8, 13), writer * 100 + index, dtype=np.float32)
            records.append(
                SequenceRecord(
                    features=features,
                    text=word,
                    writer_id=writer,
                    sample_index=index,
                    source_id=f"w{writer}-s{index}",
                    dataset_variant="test",
                    fold=0,
                    partition="all",
                )
            )
    return records


def test_factorial_cells_match_query_set_and_isolate_writer_vocabulary_exposure(tmp_path):
    cells = build_writer_vocabulary_factorial(
        _factorial_records(),
        target_writers=(1, 2),
        query_per_writer=1,
        support_per_writer=1,
        training_samples_per_cell=10,
        seed=17,
    )

    assert set(cells) == {"W0V0", "W0V1", "W1V0", "W1V1"}
    query_sets = {
        tuple(record.source_id for record in cell.query_records)
        for cell in cells.values()
    }
    assert len(query_sets) == 1
    for name, cell in cells.items():
        assert len(cell.training_records) == 10
        assert len(cell.query_records) == 2
        assert set(record.source_id for record in cell.training_records).isdisjoint(
            record.source_id for record in cell.query_records
        )
        assert cell.manifest["writer_exposure"] is (name[1] == "1")
        assert cell.manifest["vocabulary_exposure"] is (name[3] == "1")
        assert bool(cell.manifest["writer_overlap"]) == (name[1] == "1")
        assert bool(cell.manifest["vocabulary_overlap"]) == (name[3] == "1")
        assert cell.manifest["target_support_count"] == (2 if name[1] == "1" else 0)

    write_factorial_manifests(cells, tmp_path)
    top = json.loads((tmp_path / "factorial.json").read_text(encoding="utf-8"))
    assert top["protocol"] == "custom_writer_vocabulary_factorial"
    assert top["interpretation"].startswith("Custom matched experiment")
    assert all((tmp_path / f"W{w}V{v}.json").is_file() for w in (0, 1) for v in (0, 1))
    assert top["training_samples_balanced"] is True
    assert set(top["training_sample_counts"].values()) == {10}


def test_factorial_records_unavoidable_training_count_imbalance_and_can_reject_it(tmp_path):
    cells = build_writer_vocabulary_factorial(
        _factorial_records(),
        target_writers=(1, 2, 3),
        query_per_writer=1,
        support_per_writer=2,
        seed=17,
    )
    counts = {name: len(cell.training_records) for name, cell in cells.items()}
    assert len(set(counts.values())) > 1
    assert all(
        cell.manifest["training_samples_balanced"] is False for cell in cells.values()
    )

    write_factorial_manifests(cells, tmp_path)
    top = json.loads((tmp_path / "factorial.json").read_text(encoding="utf-8"))
    assert top["training_samples_balanced"] is False
    assert top["training_sample_counts"] == counts
    assert "data quantity" in top["imbalance_interpretation"]

    try:
        build_writer_vocabulary_factorial(
            _factorial_records(),
            target_writers=(1, 2, 3),
            query_per_writer=1,
            support_per_writer=2,
            allow_unbalanced=False,
            seed=17,
        )
    except ValueError as error:
        assert "exactly matched training counts are infeasible" in str(error)
    else:
        raise AssertionError("strict balancing must reject a design with unequal cell counts")


def test_factorial_rejects_duplicate_content_fingerprint_with_distinct_ids():
    records = _factorial_records()
    records[1] = SequenceRecord(
        features=records[0].features.copy(),
        text=records[1].text,
        writer_id=records[1].writer_id,
        sample_index=records[1].sample_index,
        source_id=records[1].source_id,
    )

    try:
        build_writer_vocabulary_factorial(
            records,
            target_writers=(1,),
            query_per_writer=1,
            support_per_writer=0,
        )
    except ValueError as error:
        assert "duplicate sensor fingerprints" in str(error)
    else:
        raise AssertionError("duplicate exact-content records must be rejected")


def _write_onhw_factorial_archive(path):
    with zipfile.ZipFile(path, "w") as archive:
        for split, writers in (("train", (1, 2)), ("val", (90, 91))):
            sequences = []
            labels = []
            writer_ids = []
            for writer in writers:
                for index, word in enumerate("ABCDEFGH"):
                    sequences.append(
                        np.full((16, 13), writer * 100 + index, dtype=np.float32)
                    )
                    labels.append(encode_text(word) + [ONHW_BLANK_INDEX] * 18)
                    writer_ids.append(writer)
            values = {
                f"all_x_dat_{split}_imu.pkl": sequences,
                f"all_{split}_gt.pkl": labels,
                f"{split}_ids.pkl": writer_ids,
            }
            for name, value in values.items():
                archive.writestr(
                    f"Words500_indep_02/0/{name}", pickle.dumps(value, protocol=4)
                )


def test_factorial_runner_trains_cells_and_reports_paired_writer_effects(tmp_path):
    archive_path = tmp_path / "OnHW-Words500_indep.zip"
    _write_onhw_factorial_archive(archive_path)
    options = {
        "data_path": archive_path,
        "output_dir": tmp_path / "factorial-runs",
        "target_writers": [90, 91],
        "query_per_writer": 1,
        "support_per_writer": 1,
        "training_samples_per_cell": 10,
        "sensor_profile": "front_acc_gyro6",
        "model_type": "resnet_bilstm_ctc",
        "epochs": 1,
        "batch_size": 2,
        "seed": 53,
        "device_name": "cpu",
        "num_threads": 1,
        "model_config": {
            "channels": [8, 16, 24],
            "hidden_size": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
        },
    }

    report = run_writer_vocabulary_factorial(**options)

    assert report["status"] == "complete"
    comparison = json.loads(
        (
            tmp_path
            / "factorial-runs"
            / report["run_id"]
            / "factorial_comparison.json"
        ).read_text(encoding="utf-8")
    )
    assert comparison["official_benchmark"] is False
    assert comparison["training_samples_balanced"] is True
    assert set(comparison["training_sample_counts"].values()) == {10}
    assert comparison["paired_macro_writer_analysis"]["writer_count"] == 2
    assert "writer_vocabulary_interaction" in comparison[
        "paired_macro_writer_analysis"
    ]["macro_writer_effects"]["wer"]
    assert set(comparison["cells"]) == {"W0V0", "W0V1", "W1V0", "W1V1"}
    skipped = run_writer_vocabulary_factorial(**options)
    assert skipped["status"] == "skipped_completed"


def test_writer_adaptation_manifests_cover_protocols_and_hold_query_fixed(tmp_path):
    splits, summary = build_writer_adaptation_splits(
        _factorial_records(),
        target_writer=1,
        query_count=2,
        support_budgets=(1, 5, 10),
        seed=61,
    )

    query_ids = set(summary["query_sample_ids"])
    assert summary["support_budgets_feasible"] == [1, 5]
    assert summary["support_budgets_skipped"] == [10]
    assert splits["P1_single_writer_scratch"].manifest["source_writer_count"] == 0
    assert splits["P2_cross_writer_zero_shot"].manifest["writer_overlap"] is False
    assert splits["P3_fewshot_5"].manifest["target_writer_support_count"] == 5
    assert splits["P5_ssl_5"].manifest["target_writer_unlabeled_support_count"] == 5
    assert all(not record.text for record in splits["P5_ssl_5"].unlabeled_records)
    assert splits["P6_contrastive_5"].manifest[
        "query_traces_excluded_from_all_training_stages"
    ]
    assert all(
        query_ids.isdisjoint(split.manifest["training_sample_ids"])
        and query_ids.isdisjoint(split.manifest["unlabeled_training_sample_ids"])
        for split in splits.values()
    )
    write_writer_adaptation_manifests(splits, summary, tmp_path)
    saved = json.loads((tmp_path / "adaptation_protocols.json").read_text(encoding="utf-8"))
    assert saved["session_metadata_available"] is False
