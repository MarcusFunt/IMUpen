"""Leakage-checked custom writer/vocabulary experiment splits."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from smartpen.hwr.data import SequenceRecord


@dataclass(frozen=True)
class FactorialCell:
    """One W × V cell with immutable query membership and a JSON manifest."""

    training_records: list[SequenceRecord]
    query_records: list[SequenceRecord]
    manifest: dict


@dataclass(frozen=True)
class WriterAdaptationSplit:
    """Exact labeled/unlabeled training and fixed query sets for writer studies."""

    training_records: list[SequenceRecord]
    query_records: list[SequenceRecord]
    unlabeled_records: list[SequenceRecord]
    manifest: dict


def _identity(record: SequenceRecord) -> str:
    return record.source_id or record.sample_id or (
        f"{record.dataset_variant}:{record.fold}:{record.partition}:{record.sample_index}"
    )


def _fingerprint(record: SequenceRecord) -> str:
    if record.content_sha256:
        return record.content_sha256
    values = np.asarray(record.features, dtype="<f4", order="C")
    digest = hashlib.sha256()
    digest.update(f"{values.shape[0]}x{values.shape[1]}|f32le|".encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _ordered(records: Sequence[SequenceRecord], seed: int) -> list[SequenceRecord]:
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{seed}:{_identity(record)}".encode("utf-8")
        ).hexdigest(),
    )


def build_writer_vocabulary_factorial(
    records: Sequence[SequenceRecord],
    *,
    target_writers: Sequence[int | str],
    query_per_writer: int,
    support_per_writer: int = 0,
    training_samples_per_cell: int | None = None,
    allow_unbalanced: bool = True,
    seed: int = 1337,
) -> dict[str, FactorialCell]:
    """Build matched W0V0/W0V1/W1V0/W1V1 splits over a fixed query set.

    W1 support recordings are chosen once per target writer and reused in both
    vocabulary conditions. V1 vocabulary anchors come from non-target writers.
    Query words therefore mean vocabulary seen somewhere in training, not words
    personally written by the target writer.
    """
    if not records or not target_writers:
        raise ValueError("records and target_writers must both be non-empty")
    if query_per_writer < 1 or support_per_writer < 0:
        raise ValueError("query_per_writer must be positive and support non-negative")
    targets = {str(writer) for writer in target_writers}
    if len(targets) != len(target_writers):
        raise ValueError("target_writers must not contain duplicates")
    identities = [_identity(record) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("source/sample identities must be unique within the split pool")
    all_fingerprints: dict[str, str] = {}
    for record in records:
        fingerprint = _fingerprint(record)
        identity = _identity(record)
        previous = all_fingerprints.setdefault(fingerprint, identity)
        if previous != identity:
            raise ValueError(
                "duplicate sensor fingerprints occur in the split pool; resolve them "
                "before constructing a factorial experiment"
            )

    query_by_writer: dict[str, list[SequenceRecord]] = {}
    for writer in sorted(targets):
        writer_records = [record for record in records if str(record.writer_id) == writer]
        if len(writer_records) <= support_per_writer:
            raise ValueError(f"writer {writer} has too few recordings for query and support")
        ordered = _ordered(writer_records, seed ^ int(hashlib.sha256(writer.encode()).hexdigest()[:8], 16))
        if len(ordered) < query_per_writer + support_per_writer:
            raise ValueError(f"writer {writer} lacks disjoint query and support recordings")
        query_by_writer[writer] = ordered[:query_per_writer]
    query_records = [record for writer in sorted(targets) for record in query_by_writer[writer]]
    query_ids = {_identity(record) for record in query_records}
    query_words = {record.text for record in query_records}

    support_by_writer: dict[str, list[SequenceRecord]] = {}
    if support_per_writer:
        for writer in sorted(targets):
            candidates = [
                record
                for record in records
                if str(record.writer_id) == writer
                and _identity(record) not in query_ids
                and record.text not in query_words
            ]
            if len(candidates) < support_per_writer:
                raise ValueError(
                    f"writer {writer} lacks {support_per_writer} non-query-word support "
                    "recordings required to keep support matched across vocabulary cells"
                )
            support_by_writer[writer] = _ordered(
                candidates, seed + int(hashlib.sha256(writer.encode()).hexdigest()[:8], 16)
            )[:support_per_writer]
    support_records = [record for writer in sorted(targets) for record in support_by_writer.get(writer, [])]
    support_ids = {_identity(record) for record in support_records}

    source_records = [record for record in records if str(record.writer_id) not in targets]
    vocabulary_anchors: dict[str, SequenceRecord] = {}
    for word in sorted(query_words):
        candidates = [record for record in source_records if record.text == word]
        if not candidates:
            raise ValueError(
                f"query word {word!r} is absent from eligible source writers; W1V1/W0V1 "
                "cannot represent familiar vocabulary without target query leakage"
            )
        vocabulary_anchors[word] = _ordered(candidates, seed)[0]
    anchors = list(vocabulary_anchors.values())

    candidate_by_cell: dict[str, list[SequenceRecord]] = {}
    mandatory_by_cell: dict[str, list[SequenceRecord]] = {}
    for writer_exposure in (0, 1):
        for vocabulary_exposure in (0, 1):
            key = f"W{writer_exposure}V{vocabulary_exposure}"
            allowed = [
                record
                for record in records
                if _identity(record) not in query_ids
                and (
                    str(record.writer_id) not in targets
                    or (writer_exposure == 1 and _identity(record) in support_ids)
                )
                and (vocabulary_exposure == 1 or record.text not in query_words)
            ]
            mandatory = []
            if writer_exposure:
                mandatory.extend(support_records)
            if vocabulary_exposure:
                mandatory.extend(anchors)
            allowed_ids = {_identity(record) for record in allowed}
            if any(_identity(record) not in allowed_ids for record in mandatory):
                raise ValueError(f"mandatory training samples are ineligible for {key}")
            mandatory_by_cell[key] = list({
                _identity(record): record for record in mandatory
            }.values())
            candidate_by_cell[key] = allowed

    candidate_counts = {key: len(value) for key, value in candidate_by_cell.items()}
    mandatory_counts = {key: len(value) for key, value in mandatory_by_cell.items()}
    if training_samples_per_cell is not None and training_samples_per_cell < 1:
        raise ValueError("training_samples_per_cell must be positive")

    # Match counts at the requested budget when feasible. Mandatory W1 support
    # and V1 anchors can force a cell above that budget, while small eligible
    # pools can force it below. Any resulting imbalance is reported explicitly.
    requested_budget = training_samples_per_cell
    if requested_budget is None:
        requested_budget = min(candidate_counts.values())
    cell_training_counts = {
        key: max(min(requested_budget, candidate_counts[key]), mandatory_counts[key])
        for key in candidate_by_cell
    }
    if any(cell_training_counts[key] < mandatory_counts[key] for key in candidate_by_cell):
        raise ValueError("a factorial cell cannot fit its mandatory support and vocabulary anchors")
    counts_are_balanced = len(set(cell_training_counts.values())) == 1
    if not counts_are_balanced and not allow_unbalanced:
        raise ValueError(
            "exactly matched training counts are infeasible; actual cell counts would be "
            f"{dict(sorted(cell_training_counts.items()))}"
        )

    result: dict[str, FactorialCell] = {}
    for writer_exposure in (0, 1):
        for vocabulary_exposure in (0, 1):
            key = f"W{writer_exposure}V{vocabulary_exposure}"
            mandatory = mandatory_by_cell[key]
            cell_budget = cell_training_counts[key]
            mandatory_ids = {_identity(record) for record in mandatory}
            extras = [
                record
                for record in candidate_by_cell[key]
                if _identity(record) not in mandatory_ids
            ]
            extras = _ordered(extras, seed + writer_exposure * 101 + vocabulary_exposure * 1009)
            train = mandatory + extras[: cell_budget - len(mandatory)]
            train = sorted(train, key=lambda record: _identity(record))
            train_ids = {_identity(record) for record in train}
            train_fingerprints = {_fingerprint(record) for record in train}
            query_fingerprints = {_fingerprint(record) for record in query_records}
            if train_ids & query_ids or train_fingerprints & query_fingerprints:
                raise RuntimeError(f"recording leakage detected while building {key}")
            train_writers = {str(record.writer_id) for record in train}
            train_words = {record.text for record in train}
            actual_writer_exposure = bool(train_writers & targets)
            actual_vocabulary_exposure = bool(train_words & query_words)
            if actual_writer_exposure != bool(writer_exposure):
                raise ValueError(f"{key} cannot satisfy its writer exposure definition")
            if actual_vocabulary_exposure != bool(vocabulary_exposure):
                raise ValueError(f"{key} cannot satisfy its vocabulary exposure definition")
            manifest = {
                "schema_version": 1,
                "protocol": "custom_writer_vocabulary_factorial",
                "cell": key,
                "writer_exposure": bool(writer_exposure),
                "vocabulary_exposure": bool(vocabulary_exposure),
                "target_writers": sorted(targets),
                "writer_overlap": sorted(train_writers & targets),
                "vocabulary_overlap": sorted(train_words & query_words),
                "training_sample_count": len(train),
                "training_sample_target": requested_budget,
                "training_samples_balanced": counts_are_balanced,
                "query_sample_count": len(query_records),
                "training_sample_ids": [_identity(record) for record in train],
                "query_sample_ids": [_identity(record) for record in query_records],
                "target_support_sample_ids": [
                    _identity(record) for record in support_records
                ] if writer_exposure else [],
                "query_words": sorted(query_words),
                "query_word_seen_in_training": {
                    word: word in train_words for word in sorted(query_words)
                },
                "source_writer_count": len(
                    {str(record.writer_id) for record in train if str(record.writer_id) not in targets}
                ),
                "target_support_count": sum(
                    str(record.writer_id) in targets for record in train
                ),
                "seed": seed,
                "identity_guarantee": (
                    "query set fixed across all four cells; split-local source IDs and exact "
                    "canonical content fingerprints checked for train/query overlap"
                ),
            }
            result[key] = FactorialCell(train, list(query_records), manifest)

    query_sets = {tuple(cell.manifest["query_sample_ids"]) for cell in result.values()}
    if len(query_sets) != 1:
        raise RuntimeError("factorial cells do not share the same query recordings")
    return result


def write_factorial_manifests(
    cells: dict[str, FactorialCell], output_dir: str | Path
) -> None:
    """Write each cell manifest and a top-level design manifest atomically."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for cell_name, cell in cells.items():
        _atomic_json(root / f"{cell_name}.json", cell.manifest)
    _atomic_json(
        root / "factorial.json",
        {
            "schema_version": 1,
            "protocol": "custom_writer_vocabulary_factorial",
            "cells": {name: f"{name}.json" for name in sorted(cells)},
            "query_sample_ids": next(iter(cells.values())).manifest["query_sample_ids"],
            "training_sample_counts": {
                name: cell.manifest["training_sample_count"]
                for name, cell in sorted(cells.items())
            },
            "training_sample_target": next(iter(cells.values())).manifest[
                "training_sample_target"
            ],
            "training_samples_balanced": all(
                cell.manifest["training_samples_balanced"] for cell in cells.values()
            ),
            "imbalance_interpretation": (
                "Training sample counts differ across factorial cells. Writer/vocabulary effects "
                "may also reflect training-data quantity; interpret causal contrasts cautiously."
                if not all(cell.manifest["training_samples_balanced"] for cell in cells.values())
                else None
            ),
            "interpretation": (
                "Custom matched experiment. Not an official Words500 WD/WI benchmark. "
                "V1 denotes word identities observed through eligible training writers."
            ),
        },
    )


def build_writer_adaptation_splits(
    records: Sequence[SequenceRecord],
    *,
    target_writer: int | str,
    query_count: int,
    support_budgets: Sequence[int] = (1, 5, 10, 25, 50, 100),
    seed: int = 1337,
) -> tuple[dict[str, WriterAdaptationSplit], dict]:
    """Build exact manifests for single-writer, zero-shot, few-shot, SSL, and M6-D.

    This constructs leakage-checked splits and method manifests; it does not
    claim that each listed adaptation strategy has an end-to-end OnHW trainer.
    """
    target = str(target_writer)
    budgets = tuple(sorted(set(int(budget) for budget in support_budgets)))
    if query_count < 1 or not budgets or any(budget < 1 for budget in budgets):
        raise ValueError("query_count and support budgets must be positive")
    identities = [_identity(record) for record in records]
    if not records or len(identities) != len(set(identities)):
        raise ValueError("records must be non-empty and have unique sample identities")
    seen_fingerprints: dict[str, str] = {}
    for record in records:
        fingerprint = _fingerprint(record)
        previous = seen_fingerprints.setdefault(fingerprint, _identity(record))
        if previous != _identity(record):
            raise ValueError("duplicate exact-content recordings occur in the protocol pool")

    target_records = [record for record in records if str(record.writer_id) == target]
    if len(target_records) <= query_count:
        raise ValueError("target writer needs disjoint query and at least one support recording")
    ordered_target = _ordered(target_records, seed)
    query_records = ordered_target[:query_count]
    support_pool = ordered_target[query_count:]
    feasible_budgets = [budget for budget in budgets if budget <= len(support_pool)]
    skipped_budgets = [budget for budget in budgets if budget > len(support_pool)]
    if not feasible_budgets:
        raise ValueError("target writer has no support budget available after query reservation")
    query_ids = {_identity(record) for record in query_records}
    query_fingerprints = {_fingerprint(record) for record in query_records}
    source_records = [record for record in records if str(record.writer_id) != target]
    if not source_records:
        raise ValueError("cross-writer protocols require at least one source writer")
    support_pool = _ordered(support_pool, seed + 1)
    source_records = _ordered(source_records, seed + 2)

    sessions_observed = all(record.session_id for record in records)
    shared_manifest = {
        "schema_version": 1,
        "protocol_family": "custom_onhw_writer_adaptation",
        "target_writer": target,
        "query_count": len(query_records),
        "query_sample_ids": [_identity(record) for record in query_records],
        "query_words": sorted({record.text for record in query_records}),
        "query_fingerprints_sha256": sorted(query_fingerprints),
        "seed": seed,
        "query_traces_excluded_from_all_training_stages": True,
        "session_metadata_available": sessions_observed,
        "session_limitation": (
            "source session identifiers are present"
            if sessions_observed
            else "OnHW source records do not expose genuine session IDs; no session split is claimed"
        ),
        "official_benchmark": False,
    }
    splits: dict[str, WriterAdaptationSplit] = {}

    def add_split(
        key: str,
        labeled_training: list[SequenceRecord],
        *,
        strategy: str,
        unlabeled: list[SequenceRecord] | None = None,
        zero_shot: bool = False,
    ) -> None:
        training_ids = {_identity(record) for record in labeled_training}
        unlabeled = unlabeled or []
        unlabeled_ids = {_identity(record) for record in unlabeled}
        if query_ids & (training_ids | unlabeled_ids):
            raise ValueError(f"query recording leakage in adaptation split {key}")
        training_fingerprints = {_fingerprint(record) for record in labeled_training + unlabeled}
        if query_fingerprints & training_fingerprints:
            raise ValueError(f"query content fingerprint leakage in adaptation split {key}")
        training_writers = {str(record.writer_id) for record in labeled_training}
        seen_writers = training_writers | {str(record.writer_id) for record in unlabeled}
        training_words = {record.text for record in labeled_training}
        query_words = {record.text for record in query_records}
        manifest = {
            **shared_manifest,
            "cell": key,
            "training_strategy": strategy,
            "training_sample_ids": [_identity(record) for record in labeled_training],
            "unlabeled_training_sample_ids": [_identity(record) for record in unlabeled],
            "query_sample_ids": [_identity(record) for record in query_records],
            "writer_overlap": target in seen_writers,
            "vocabulary_overlap": sorted(training_words & query_words),
            "vocabulary_overlap_count": len(training_words & query_words),
            "source_writer_count": len(seen_writers - {target}),
            "target_writer_support_count": sum(
                str(record.writer_id) == target for record in labeled_training
            ),
            "target_writer_unlabeled_support_count": sum(
                str(record.writer_id) == target for record in unlabeled
            ),
            "zero_shot": zero_shot,
            "adapted": not zero_shot and target in seen_writers,
            "query_traces_excluded_from_all_training_stages": True,
        }
        splits[key] = WriterAdaptationSplit(
            labeled_training, list(query_records), unlabeled, manifest
        )

    add_split(
        "P1_single_writer_scratch",
        support_pool,
        strategy="single_writer_from_scratch",
    )
    add_split(
        "P2_cross_writer_zero_shot",
        source_records,
        strategy="cross_writer_supervised_ctc",
        zero_shot=True,
    )
    add_split(
        "P3_zero_shot_pretrained",
        source_records,
        strategy="multi_writer_pretraining_zero_shot",
        zero_shot=True,
    )
    methods = ("scratch", "frozen_encoder", "partial_finetuning", "full_finetuning")
    for budget in feasible_budgets:
        support = support_pool[:budget]
        adapted = source_records + support
        add_split(
            f"P3_fewshot_{budget}",
            adapted,
            strategy="multi_writer_pretraining_plus_personalization",
        )
        for method in methods:
            add_split(
                f"P4_{budget}_{method}",
                adapted,
                strategy=method,
            )
        ssl_records = [replace(record, text="") for record in source_records + support]
        add_split(
            f"P5_ssl_{budget}",
            [],
            strategy="self_supervised_adaptation",
            unlabeled=ssl_records,
        )
        add_split(
            f"P6_contrastive_{budget}",
            adapted,
            strategy="ctc_plus_training_transcript_contrastive",
        )

    summary = {
        **shared_manifest,
        "splits": {key: f"{key}.json" for key in sorted(splits)},
        "support_budgets_requested": list(budgets),
        "support_budgets_feasible": feasible_budgets,
        "support_budgets_skipped": skipped_budgets,
        "P4_strategies": list(methods),
        "P6_hard_negative_policy": "derive negatives from labeled training transcripts only",
        "method_status": (
            "split manifests and leakage checks are available; protocol-specific OnHW "
            "training orchestration is not implemented by this builder"
        ),
    }
    return splits, summary


def write_writer_adaptation_manifests(
    splits: dict[str, WriterAdaptationSplit],
    summary: dict,
    output_dir: str | Path,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for key, split in splits.items():
        _atomic_json(root / f"{key}.json", split.manifest)
    _atomic_json(root / "adaptation_protocols.json", summary)


def _atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
