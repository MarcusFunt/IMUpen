"""Prompt and capture behavior without serial hardware."""
from smartpen.collector.cli import DEFAULT_PROMPTS, load_prompts
from smartpen.collector.recording import CaptureBuffer
from smartpen.protocol import Sample, encode_frame


def test_default_and_custom_prompts(tmp_path):
    assert "hello world" in load_prompts(None)
    file = tmp_path / "prompts.txt"
    file.write_text("# comment\n\nhej æøå\nanother prompt\n", encoding="utf-8")
    assert load_prompts(file) == ["hej æøå", "another prompt"]


def test_capture_chunk_boundaries_and_raw_preservation():
    value = Sample(4, 500, 1, 2, 3, 4, 5, 6)
    raw = b"\x01\xff" + encode_frame(value)
    capture = CaptureBuffer()
    capture.feed(raw[:3])
    capture.feed(raw[3:8])
    capture.feed(raw[8:])
    assert bytes(capture.raw) == raw
    assert capture.sample_count == 1
    assert capture.frames[0].message == value
