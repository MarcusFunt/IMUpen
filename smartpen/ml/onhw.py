"""OnHW-Words500 official-fold pickle adapter, back acceleration + gyroscope.

Original 13 columns: front accel 0:3, back accel 3:6, gyro 6:9,
magnetometer 9:12, force 12. Six-axis subset: 3,4,5,6,7,8.
Unpickling can execute code: --trust-pickle is REQUIRED for trusted archives.
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .text import ONHW_WORDS500_CHARSET, Vocabulary
from .model import required_ctc_steps, output_lengths

SIX_CHANNEL_COLUMNS = (3, 4, 5, 6, 7, 8)
REQUIRED = (
    "all_x_dat_train_imu.pkl", "all_x_dat_val_imu.pkl",
    "all_train_gt.pkl", "all_val_gt.pkl", "train_ids.pkl", "val_ids.pkl")

@dataclass(frozen=True)
class Record:
    x: np.ndarray
    text: str
    writer: str
    sample_id: str

@dataclass(frozen=True)
class Fold:
    train: list[Record]
    official_val: list[Record]
    dropped_train: int
    dropped_val: int

def _load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)  # noqa: S301; explicit --trust-pickle CLI opt-in

def _resolve_fold(root: Path, fold: int) -> Path:
    if not 0 <= fold <= 4:
        raise ValueError("OnHW fold must be in 0..4")
    for path in (root / str(fold), root / f"fold_{fold}"):
        if path.is_dir():
            return path
    matches = [p for p in root.glob(f"*/{fold}") if p.is_dir()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot find OnHW fold {fold} under {root}")

def load_words500(root: str | Path, *, fold: int = 0, trust_pickle: bool = False,
                  max_train: int | None = None, max_val: int | None = None,
                  seed: int = 42) -> Fold:
    if not trust_pickle:
        raise PermissionError("OnHW uses executable pickle; pass --trust-pickle only for trusted archives")
    location = _resolve_fold(Path(root), fold)
    for name in REQUIRED:
        if not (location / name).is_file():
            raise FileNotFoundError(f"Missing {name} in {location}")
    vocab = Vocabulary()

    def read(part: str, limit: int | None):
        if part == "train":
            xs, ys, ids = (
                _load_pickle(location / "all_x_dat_train_imu.pkl"),
                _load_pickle(location / "all_train_gt.pkl"),
                _load_pickle(location / "train_ids.pkl"))
        else:
            xs, ys, ids = (
                _load_pickle(location / "all_x_dat_val_imu.pkl"),
                _load_pickle(location / "all_val_gt.pkl"),
                _load_pickle(location / "val_ids.pkl"))
        if not len(xs) == len(ys) == len(ids):
            raise ValueError(f"Unaligned OnHW {part} arrays")
        indices = np.arange(len(xs))
        if limit is not None:
            if limit < 1:
                raise ValueError("Sample limit must be positive")
            indices = np.sort(np.random.default_rng(seed + (0 if part == "train" else 1001))
                              .choice(indices, min(len(indices), limit), replace=False))
        accepted: list[Record] = []
        dropped = 0
        for i in indices:
            signal = np.asarray(xs[int(i)], dtype=np.float32)
            raw_label = [int(token) for token in ys[int(i)]]
            while raw_label and raw_label[-1] == len(ONHW_WORDS500_CHARSET):
                raw_label.pop()
            if not raw_label or any(not 0 <= token < len(ONHW_WORDS500_CHARSET)
                                    for token in raw_label):
                dropped += 1
                continue
            if signal.ndim != 2 or signal.shape[1] != 13 or not np.isfinite(signal).all():
                dropped += 1
                continue
            text = "".join(ONHW_WORDS500_CHARSET[c] for c in raw_label)
            if len(signal) < 1 or int(output_lengths(np.asarray([len(signal)]))[0]) < required_ctc_steps(vocab.encode(text)):
                dropped += 1
                continue
            six = signal[:, SIX_CHANNEL_COLUMNS].copy()
            accepted.append(Record(six, text, str(ids[int(i)]), f"{part}-{int(i):06d}"))
        return accepted, dropped

    train, n_train = read("train", max_train)
    validation, n_val = read("val", max_val)
    if not train or not validation:
        raise ValueError("OnHW training or official validation partition is empty")
    return Fold(train, validation, n_train, n_val)

def split_training_writers(train: list[Record], *, fraction: float = 0.1,
                           seed: int = 42) -> tuple[list[Record], list[Record]]:
    writers = sorted({record.writer for record in train})
    if len(writers) < 2:
        raise ValueError("Need at least two training writers to split by writer")
    count = max(1, min(len(writers) - 1, round(len(writers) * fraction)))
    chosen = set(np.random.default_rng(seed).choice(writers, size=count, replace=False))
    fit = [r for r in train if r.writer not in chosen]
    val = [r for r in train if r.writer in chosen]
    if not fit or not val:
        raise ValueError("Writer split produced an empty partition")
    return fit, val
