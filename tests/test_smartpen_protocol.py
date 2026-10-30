"""Host-side protocol contract tests; no physical hardware needed."""
import pytest

from smartpen.protocol import (
    Config, DeviceError, FrameParser, Sample, SequenceTracker,
    crc16, encode_frame, FLAG_DISCONTINUITY,
)


def sample(sequence=17, flags=0):
    return Sample(sequence, 2**40 + sequence * 4807,
                  -100, 200, -300, 400, -500, 600, flags)


def test_crc_reference_vector():
    assert crc16(b"123456789") == 0x29B1


def test_sample_frame_exact_size_and_round_trip():
    value = sample()
    frame = encode_frame(value)
    assert frame[:2] == b"\x5a\xa5"
    assert len(frame) == 34
    parsed = FrameParser().feed(frame)
    assert len(parsed) == 1
    assert parsed[0].message == value
    assert parsed[0].raw == frame


def test_all_types_and_fragmented_stream():
    messages = [
        Config(208, 4, 500, 0x6B, 8, 9, 0x69, 1),
        sample(1), DeviceError(3, 27), sample(2, FLAG_DISCONTINUITY)
    ]
    stream = b"noise" + b"".join(encode_frame(m) for m in messages)
    decoder = FrameParser()
    frames = []
    for byte in stream:
        frames.extend(decoder.feed(bytes([byte])))
    assert [frame.message for frame in frames] == messages
    assert decoder.frames_parsed == len(messages)
    assert decoder.discarded_bytes == len(b"noise")


def test_corrupt_crc_and_bad_header_resynchronize():
    good = encode_frame(sample(5))
    damaged = bytearray(encode_frame(sample(4)))
    damaged[18] ^= 0x01
    invalid_header = bytearray(good)
    invalid_header[4] = 255  # wrong declared payload size
    decoder = FrameParser()
    decoded = decoder.feed(bytes(damaged) + bytes(invalid_header) + good)
    assert [f.message.sequence for f in decoded] == [5]
    assert decoder.crc_errors >= 1
    assert decoder.header_errors >= 1


def test_split_sync_marker():
    decoder = FrameParser()
    assert decoder.feed(b"\x00\x5a") == []
    result = decoder.feed(encode_frame(sample(8))[1:])
    assert result[0].message.sequence == 8


def test_sequence_gap_wrap_duplicate_and_out_of_order():
    tracker = SequenceTracker()
    for n in [0xFFFFFFFE, 0xFFFFFFFF, 0, 2, 2, 1]:
        tracker.add(sample(n))
    result = tracker.summary()
    assert result["missing_transmitted_samples"] == 1
    assert result["duplicates"] == 1
    assert result["out_of_order"] == 1


def test_timestamp_regression_and_discontinuity():
    tracker = SequenceTracker()
    tracker.add(Sample(1, 100, 0, 0, 0, 0, 0, 0))
    tracker.add(Sample(2, 90, 0, 0, 0, 0, 0, 0, FLAG_DISCONTINUITY))
    assert tracker.timestamp_regressions == 1
    assert tracker.discontinuities == 1


def test_encoder_rejects_unsupported_message():
    with pytest.raises(TypeError):
        encode_frame(object())
