"""Small pyserial wrapper. Opening USB CDC may reboot the ESP32-S3."""
from __future__ import annotations

import serial
from smartpen.protocol import Frame, FrameParser


class SerialFrameReader:
    def __init__(self, port: str, baud: int = 921600, timeout: float = 0.1):
        self._serial = serial.Serial(port, baudrate=baud, timeout=timeout)
        self.parser = FrameParser()

    def read_chunk(self, maximum: int = 4096) -> bytes:
        return self._serial.read(maximum)

    def read_frames(self, maximum: int = 4096) -> list[Frame]:
        return self.parser.feed(self.read_chunk(maximum))

    def close(self) -> None:
        self._serial.close()

    def __enter__(self) -> "SerialFrameReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
