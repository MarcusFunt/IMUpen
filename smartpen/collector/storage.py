"""Immutable capture storage: raw stream, parsed NPZ, metadata, JSONL manifest."""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from smartpen.protocol import Config, DeviceError, Frame, Sample, SequenceTracker


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("Identifiers must contain only letters, digits, '_' or '-'")
    return value


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RecordingStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(
        self,
        *,
        writer_id: str,
        session_id: str,
        prompt: str,
        raw_bytes: bytes,
        frames: Sequence[Frame],
        accepted: bool,
        fallback_config: Config | None = None,
        parser_crc_errors: int = 0,
        parser_header_errors: int = 0,
    ) -> dict:
        writer_id, session_id = _safe_id(writer_id), _safe_id(session_id)
        if not prompt.strip():
            raise ValueError("Prompt cannot be blank")
        samples = [f.message for f in frames if isinstance(f.message, Sample)]
        if not samples:
            raise ValueError("Recording contains no valid IMU samples")
        config = next(
            (f.message for f in reversed(frames) if isinstance(f.message, Config)),
            fallback_config,
        )
        errors = [asdict(f.message) for f in frames
                  if isinstance(f.message, DeviceError)]
        tracker = SequenceTracker()
        for sample in samples:
            tracker.add(sample)
        record_id = uuid.uuid4().hex
        raw_path = Path("raw") / session_id / f"{record_id}.bin"
        npz_path = Path("processed") / session_id / f"{record_id}.npz"
        meta_path = Path("metadata") / session_id / f"{record_id}.json"

        axes = np.asarray([s.axes for s in samples], dtype=np.int16)
        seq = np.asarray([s.sequence for s in samples], dtype=np.uint32)
        timestamps = np.asarray([s.timestamp_us for s in samples], dtype=np.uint64)
        flags = np.asarray([s.flags for s in samples], dtype=np.uint16)

        metadata = {
            "schema_version": 1,
            "sample_id": record_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "writer_id": writer_id,
            "session_id": session_id,
            "prompt": prompt,
            "accepted": bool(accepted),
            "sample_count": len(samples),
            "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
            "raw_file": raw_path.as_posix(),
            "array_file": npz_path.as_posix(),
            "config": asdict(config) if config is not None else None,
            "diagnostics": {
                **tracker.summary(),
                "crc_errors": parser_crc_errors,
                "header_errors": parser_header_errors,
                "device_error_frames": errors,
            },
            "timestamp_note": "Nominal 208-Hz FIFO reconstruction, not hardware timestamps",
        }

        # Metadata is written last; it is the commit marker for a recording.
        _atomic_bytes(self.root / raw_path, bytes(raw_bytes))
        target = self.root / npz_path
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(suffix=".npz", prefix=".tmp-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as file:
                np.savez_compressed(file, axes=axes, sequence=seq,
                                    timestamp_us=timestamps, flags=flags)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        _atomic_bytes(self.root / meta_path,
                      (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        manifest = self.root / "manifests" / f"{session_id}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        # Single-writer process only; not a multi-process concurrent append API.
        with manifest.open("a", encoding="utf-8") as file:
            file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        return metadata
