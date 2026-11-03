"""Loader for the official OnHW-Chars NumPy archive."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartpen.hwr.data import (
    ONHW_CHANNEL_NAMES,
    SequenceRecord,
    _content_fingerprint,
)


ONHW_CHARS_ALPHABETS = {
    "both": tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"),
    "lower": tuple("abcdefghijklmnopqrstuvwxyz"),
    "upper": tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
}
ONHW_CHARS_PROTOCOLS = {"dep": "writer_dependent", "indep": "writer_independent"}
ONHW_CHARS_RAW_UNIT = "raw_uncalibrated_sensor_value"


@dataclass(frozen=True)
class OnHWCharsFold:
    """One official OnHW-Chars train/test fold and its quality notes."""

    train: list[SequenceRecord]
    test: list[SequenceRecord]
    fold: int
    corpus: str
    protocol: str
    archive_name: str
    data_quality: dict


def _read_npy(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    # The published X arrays are NumPy object arrays containing variable-length
    # numeric sequences. This dataset is user-selected and its archive digest
    # is recorded by the experiment runner.
    return np.load(io.BytesIO(archive.read(member)), allow_pickle=True)


def load_onhw_chars(
    path: str | Path,
    *,
    corpus: str = "both",
    protocol: str = "indep",
    fold: int = 0,
) -> OnHWCharsFold:
    """Load one published OnHW-Chars fold without manufacturing writer IDs.

    Exact sequence duplicates are resolved by keeping the official test copy
    and removing its identical train copy. Repeated exact contents within a
    partition are deduplicated, with the first split-local row retained.
    Empty sequences are dropped and counted.
    """
    source = Path(path).resolve()
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError("OnHW-Chars must be provided as the published ZIP archive")
    if corpus not in ONHW_CHARS_ALPHABETS:
        raise ValueError(f"corpus must be one of {tuple(ONHW_CHARS_ALPHABETS)}")
    if protocol not in ONHW_CHARS_PROTOCOLS:
        raise ValueError(f"protocol must be one of {tuple(ONHW_CHARS_PROTOCOLS)}")
    if not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")

    folder = f"onhw2_{corpus}_{protocol}_{fold}"
    suffixes = {
        f"{folder}/{name}.npy": name
        for name in ("X_train", "y_train", "X_test", "y_test")
    }
    with zipfile.ZipFile(source) as archive:
        files: dict[str, str] = {}
        for member in archive.namelist():
            normalized = member.replace("\\", "/")
            if normalized.endswith("/"):
                continue
            for suffix, name in suffixes.items():
                if normalized.endswith(suffix):
                    if name in files:
                        raise ValueError(f"archive contains duplicate {name} members for {folder}")
                    files[name] = normalized
        if set(files) != set(suffixes.values()):
            missing = sorted(set(suffixes.values()) - set(files))
            raise FileNotFoundError(f"OnHW-Chars archive is missing {folder} files: {missing}")

        arrays = {name: _read_npy(archive, member) for name, member in files.items()}

    expected = set(ONHW_CHARS_ALPHABETS[corpus])
    raw_records: dict[str, list[SequenceRecord]] = {"train": [], "test": []}
    empty_counts = {"train": 0, "test": 0}
    shape_errors = {"train": 0, "test": 0}
    nonfinite_counts = {"train": 0, "test": 0}
    for partition in ("train", "test"):
        features = arrays[f"X_{partition}"]
        labels = arrays[f"y_{partition}"]
        if len(features) != len(labels):
            raise ValueError(
                f"OnHW-Chars sample/label count mismatch in {folder}/{partition}: "
                f"{len(features)} vs {len(labels)}"
            )
        if features.ndim != 1 or labels.ndim != 1:
            raise ValueError(f"OnHW-Chars arrays in {folder}/{partition} must be one-dimensional")
        for sample_index, (sequence, raw_label) in enumerate(zip(features, labels)):
            label = str(raw_label)
            if len(label) != 1 or label not in expected:
                raise ValueError(
                    f"invalid OnHW-Chars label {label!r} in {folder}/{partition} sample {sample_index}"
                )
            values = np.asarray(sequence)
            if values.ndim != 2 or values.shape[1] != 13:
                shape_errors[partition] += 1
                raise ValueError(
                    f"expected [time, 13] OnHW-Chars features in {folder}/{partition} "
                    f"sample {sample_index}, got {values.shape}"
                )
            if values.shape[0] == 0:
                empty_counts[partition] += 1
                continue
            if not np.isfinite(values).all():
                nonfinite_counts[partition] += 1
                raise ValueError(
                    f"non-finite OnHW-Chars data in {folder}/{partition} sample {sample_index}"
                )
            values = np.ascontiguousarray(values, dtype=np.float32)
            raw_records[partition].append(
                SequenceRecord(
                    features=values,
                    text=label,
                    writer_id=None,
                    sample_index=sample_index,
                    sample_id=(
                        f"onhw-chars:{source.name}:{corpus}:{protocol}:fold-{fold}:"
                        f"{partition}:{sample_index}"
                    ),
                    content_sha256=_content_fingerprint(values),
                    dataset_variant=f"onhw_chars_{corpus}_{protocol}",
                    archive_name=source.name,
                    fold=fold,
                    partition=partition,
                    channel_names=ONHW_CHANNEL_NAMES,
                    channel_units=(ONHW_CHARS_RAW_UNIT,) * 13,
                    sample_rate_hz=None,
                )
            )

    def deduplicate(records: list[SequenceRecord], partition: str):
        seen: dict[str, SequenceRecord] = {}
        unique = []
        removed = 0
        for record in records:
            fingerprint = record.content_sha256
            previous = seen.get(fingerprint)
            if previous is not None:
                if previous.text != record.text:
                    raise ValueError(
                        f"identical OnHW-Chars signals have conflicting labels in {folder}/{partition}"
                    )
                removed += 1
                continue
            seen[fingerprint] = record
            unique.append(record)
        return unique, removed, seen

    train, train_duplicates, train_hashes = deduplicate(raw_records["train"], "train")
    test, test_duplicates, test_hashes = deduplicate(raw_records["test"], "test")
    cross_partition = train_hashes.keys() & test_hashes.keys()
    for fingerprint in cross_partition:
        if train_hashes[fingerprint].text != test_hashes[fingerprint].text:
            raise ValueError(f"identical OnHW-Chars signals have conflicting train/test labels in {folder}")
    if cross_partition:
        train = [record for record in train if record.content_sha256 not in cross_partition]

    data_quality = {
        "dataset": "OnHW-Chars",
        "corpus": corpus,
        "official_protocol": ONHW_CHARS_PROTOCOLS[protocol],
        "protocol_code": protocol,
        "fold": fold,
        "writer_ids_available": False,
        "writer_overlap_verifiable": False,
        "channel_count": 13,
        "sample_rate_hz_available": False,
        "sensor_calibration_available": False,
        "channel_units": ONHW_CHARS_RAW_UNIT,
        "official_train_samples": len(arrays["y_train"]),
        "official_test_samples": len(arrays["y_test"]),
        "train_samples_after_quality_filter": len(train),
        "test_samples_after_quality_filter": len(test),
        "empty_sequences_dropped": empty_counts,
        "duplicate_sequences_deduplicated_within_split": {
            "train": train_duplicates,
            "test": test_duplicates,
        },
        "exact_train_test_duplicates_removed_from_train": len(cross_partition),
        "invalid_shape_count": shape_errors,
        "nonfinite_sequence_count": nonfinite_counts,
        "official_membership_policy": (
            "test rows retain priority for exact-content duplicates; identical duplicates "
            "within each partition keep the first split-local row"
        ),
    }
    return OnHWCharsFold(
        train=train,
        test=test,
        fold=fold,
        corpus=corpus,
        protocol=protocol,
        archive_name=source.name,
        data_quality=data_quality,
    )
