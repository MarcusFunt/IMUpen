"""OnHW-Words500 writer-independent fold loading and six-axis adaptation."""
from __future__ import annotations

import pickle
import zipfile
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


# The sequence dataset uses this 59-character order. 59 is also the padding
# token in the author-provided labels and becomes the CTC blank class.
ONHW_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß"
ONHW_BLANK_INDEX = len(ONHW_CHARSET)
ONHW_SENSOR_CHANNELS = 13
ONHW_SIX_AXIS_COLUMNS = (0, 1, 2, 6, 7, 8)  # front accel xyz, gyro xyz


@dataclass(frozen=True)
class SequenceRecord:
    features: np.ndarray
    text: str
    writer_id: int
    sample_index: int = 0


@dataclass(frozen=True)
class OnHWFold:
    train: list[SequenceRecord]
    validation: list[SequenceRecord]
    fold: int
    dropped_train: int = 0
    dropped_validation: int = 0

    @property
    def train_writers(self) -> set[int]:
        return {record.writer_id for record in self.train}

    @property
    def validation_writers(self) -> set[int]:
        return {record.writer_id for record in self.validation}


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Load the NumPy arrays/lists in official OnHW archives without globals."""

    _NUMPY_MODULES = {"numpy", "numpy.core.multiarray", "numpy._core.multiarray"}
    _NUMPY_NAMES = {"ndarray", "dtype", "_reconstruct", "scalar"}

    def find_class(self, module: str, name: str):
        if module in self._NUMPY_MODULES and name in self._NUMPY_NAMES:
            if module == "numpy":
                return getattr(np, name)
            try:
                multiarray = importlib.import_module("numpy._core.multiarray")
            except ImportError:
                multiarray = importlib.import_module("numpy.core.multiarray")
            return getattr(multiarray, name)
        raise pickle.UnpicklingError(
            f"unsupported pickle global {module}.{name}; use an official OnHW archive"
        )


def _load_pickle(stream: BinaryIO):
    return _NumpyOnlyUnpickler(stream).load()


def select_six_axis(sequence: np.ndarray) -> np.ndarray:
    """Return [front accel x/y/z, gyro x/y/z] as a contiguous float32 array.

    The OnHW sequence files contain 13 sensor channels ordered as front
    accelerometer (3), rear accelerometer (3), gyroscope (3), magnetometer
    (3), and force (1). Existing IMUpen recordings contain the six target
    channels already, in the same semantic order.
    """
    values = np.asarray(sequence)
    if values.ndim != 2:
        raise ValueError(f"expected a (timesteps, channels) array, got {values.shape}")
    if values.shape[1] == 6:
        selected = values
    elif values.shape[1] == ONHW_SENSOR_CHANNELS:
        selected = values[:, ONHW_SIX_AXIS_COLUMNS]
    else:
        raise ValueError(
            f"expected 6 recorder channels or 13 OnHW channels, got {values.shape[1]}"
        )
    return np.ascontiguousarray(selected, dtype=np.float32)


def decode_onhw_label(padded_tokens: list[int] | tuple[int, ...]) -> str:
    """Decode a published OnHW integer label and strip trailing blank padding."""
    tokens = [int(token) for token in padded_tokens]
    while tokens and tokens[-1] == ONHW_BLANK_INDEX:
        tokens.pop()
    if any(token == ONHW_BLANK_INDEX for token in tokens):
        raise ValueError("OnHW blank padding must be trailing")
    invalid = [token for token in tokens if not 0 <= token < ONHW_BLANK_INDEX]
    if invalid:
        raise ValueError(f"OnHW label token out of range: {invalid[0]}")
    return "".join(ONHW_CHARSET[token] for token in tokens)


def encode_text(text: str) -> list[int]:
    """Encode text using the OnHW-Words500 character order."""
    index = {character: token for token, character in enumerate(ONHW_CHARSET)}
    try:
        return [index[character] for character in text]
    except KeyError as exc:
        raise ValueError(f"character {exc.args[0]!r} is outside the OnHW charset") from exc


def _resolve_directory_files(root: Path, fold: int) -> tuple[dict[str, Path], str]:
    candidates = [root / str(fold), root / f"fold_{fold}"]
    if all((root / name).is_file() for name in _required_files("train")):
        candidates.insert(0, root)
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / name).is_file()
            for split in ("train", "val")
            for name in _required_files(split)
        ):
            return (
                {
                    name: candidate / name
                    for split in ("train", "val")
                    for name in _required_files(split)
                },
                candidate.name,
            )
    raise FileNotFoundError(f"could not find OnHW fold {fold} below {root}")


def _required_files(split: str) -> tuple[str, ...]:
    return (
        f"all_x_dat_{split}_imu.pkl",
        f"all_{split}_gt.pkl",
        f"{split}_ids.pkl",
    )


def _resolve_zip_members(
    archive: zipfile.ZipFile, fold: int
) -> tuple[dict[str, str], str]:
    files = {
        name.rsplit("/", 1)[-1]: name
        for name in archive.namelist()
        if name.endswith(".pkl") and f"/{fold}/" in f"/{name}"
    }
    required = {
        name
        for split in ("train", "val")
        for name in _required_files(split)
    }
    if not required.issubset(files):
        raise FileNotFoundError(
            f"archive does not contain the six required pickle files for fold {fold}"
        )
    prefix = files[_required_files("train")[0]].rsplit(f"/{fold}/", 1)[0]
    return {name: files[name] for name in required}, prefix


def _read(path: Path | None = None, *, archive=None, member: str | None = None):
    if path is not None:
        with path.open("rb") as stream:
            return _load_pickle(stream)
    with archive.open(member, "r") as stream:
        return _load_pickle(stream)


def load_onhw_words500(path: str | Path, fold: int = 0) -> OnHWFold:
    """Load one official writer-independent Words500 fold from a ZIP or extract.

    Pickle files are loaded with a restricted unpickler that permits only
    NumPy reconstruction globals; archive data is never accepted as code.
    The supplied `*_indep.zip` fold defines the held-out writer partition.
    """
    if fold not in range(5):
        raise ValueError(f"fold must be between 0 and 4, got {fold}")
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    archive = None
    if source.is_file() and zipfile.is_zipfile(source):
        archive = zipfile.ZipFile(source)
        files, _ = _resolve_zip_members(archive, fold)
        read = lambda filename: _read(archive=archive, member=files[filename])
        is_writer_independent = "indep" in source.name.lower()
    elif source.is_dir():
        files, directory = _resolve_directory_files(source, fold)
        read = lambda filename: _read(path=files[filename])
        is_writer_independent = "indep" in str(source).lower()
    else:
        raise ValueError(f"expected an extracted OnHW folder or ZIP archive: {source}")

    try:
        records: dict[str, list[SequenceRecord]] = {}
        dropped: dict[str, int] = {}
        for split in ("train", "val"):
            sequences = read(f"all_x_dat_{split}_imu.pkl")
            labels = read(f"all_{split}_gt.pkl")
            writer_ids = read(f"{split}_ids.pkl")
            if not (len(sequences) == len(labels) == len(writer_ids)):
                raise ValueError(f"sample, label, and writer counts differ in {split} split")
            split_records: list[SequenceRecord] = []
            removed = 0
            for sample_index, (sequence, label, writer_id) in enumerate(
                zip(sequences, labels, writer_ids)
            ):
                features = select_six_axis(np.asarray(sequence))
                text = decode_onhw_label(label)
                if not text:
                    raise ValueError(f"empty OnHW label in {split} split")
                if features.shape[0] == 0:
                    removed += 1
                    continue
                split_records.append(
                    SequenceRecord(features, text, int(writer_id), sample_index)
                )
            records[split] = split_records
            dropped[split] = removed

        train_writers = {record.writer_id for record in records["train"]}
        validation_writers = {record.writer_id for record in records["val"]}
        overlap = train_writers & validation_writers
        if overlap:
            kind = "writer-independent " if is_writer_independent else ""
            raise ValueError(
                f"{kind}OnHW fold {fold} leaks {len(overlap)} writers across train/val"
            )
        return OnHWFold(
            train=records["train"],
            validation=records["val"],
            fold=fold,
            dropped_train=dropped["train"],
            dropped_validation=dropped["val"],
        )
    finally:
        if archive is not None:
            archive.close()
