"""Recording persistence, schema, and raw byte integrity."""
import json

import numpy as np
import pytest

from smartpen.collector.recording import CaptureBuffer
from smartpen.collector.storage import RecordingStore
from smartpen.protocol import Config, DeviceError, Sample, encode_frame


def test_save_and_replay_round_trip(tmp_path):
    config = Config(208, 4, 500, 0x6A, 8, 9, 0x69, 1)
    stream = (encode_frame(config) +
              encode_frame(Sample(5, 1000, 1, 2, 3, 4, 5, 6)) +
              encode_frame(Sample(7, 10600, -1, -2, -3, -4, -5, -6)) +
              encode_frame(DeviceError(3, 1)))
    capture = CaptureBuffer()
    for i in range(0, len(stream), 7):
        capture.feed(stream[i:i + 7])
    assert capture.sample_count == 2
    store = RecordingStore(tmp_path)
    metadata = store.save(
        writer_id="p01", session_id="s01", prompt="hello world",
        raw_bytes=bytes(capture.raw), frames=capture.frames, accepted=True)
    assert (tmp_path / metadata["raw_file"]).read_bytes() == stream
    with np.load(tmp_path / metadata["array_file"]) as arrays:
        assert arrays["axes"].dtype == np.int16
        assert arrays["axes"].shape == (2, 6)
        assert arrays["axes"].tolist() == [[1, 2, 3, 4, 5, 6],
                                           [-1, -2, -3, -4, -5, -6]]
        assert arrays["sequence"].tolist() == [5, 7]
    assert metadata["diagnostics"]["missing_transmitted_samples"] == 1
    assert metadata["config"]["i2c_address"] == 0x6A
    assert metadata["accepted"] is True
    assert json.loads((tmp_path / "manifests" / "s01.jsonl")
                      .read_text(encoding="utf-8"))["sample_id"] == metadata["sample_id"]


def test_corrupt_bytes_are_preserved_but_not_counted(tmp_path):
    frame = encode_frame(Sample(0, 1, 1, 1, 1, 1, 1, 1))
    broken = bytearray(frame)
    broken[20] ^= 0x80
    capture = CaptureBuffer()
    capture.feed(bytes(broken) + frame)
    assert capture.sample_count == 1
    assert capture.parser.crc_errors == 1
    metadata = RecordingStore(tmp_path).save(
        writer_id="p01", session_id="s01", prompt="a",
        raw_bytes=bytes(capture.raw), frames=capture.frames, accepted=False,
        parser_crc_errors=capture.parser.crc_errors)
    assert metadata["accepted"] is False
    assert metadata["diagnostics"]["crc_errors"] == 1
    assert (tmp_path / metadata["raw_file"]).read_bytes() == bytes(broken) + frame


def test_invalid_ids_and_empty_capture(tmp_path):
    store = RecordingStore(tmp_path)
    value = Sample(0, 1, 0, 0, 0, 0, 0, 0)
    capture = CaptureBuffer()
    capture.feed(encode_frame(value))
    args = dict(writer_id="p01", session_id="s01", prompt="x",
                raw_bytes=bytes(capture.raw), frames=capture.frames,
                accepted=True)
    with pytest.raises(ValueError):
        store.save(**(args | {"session_id": "../escape"}))
    with pytest.raises(ValueError):
        store.save(**(args | {"prompt": "  "}))
    with pytest.raises(ValueError):
        store.save(**(args | {"frames": []}))
