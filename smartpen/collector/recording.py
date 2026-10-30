"""Capture an exact serial byte stream and its decoded valid frames."""
from __future__ import annotations

from smartpen.protocol import Frame, FrameParser


class CaptureBuffer:
    def __init__(self) -> None:
        self.raw = bytearray()
        self.parser = FrameParser()
        self.frames: list[Frame] = []

    def feed(self, data: bytes) -> None:
        # Save all bytes, including noise and corrupted frames, for audit.
        self.raw.extend(data)
        self.frames.extend(self.parser.feed(data))

    @property
    def sample_count(self) -> int:
        from smartpen.protocol import Sample
        return sum(isinstance(f.message, Sample) for f in self.frames)
