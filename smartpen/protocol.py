"""SmartPen binary USB protocol v1.

Frame layout: sync 5A A5, version u8, type u8, payload_len u16,
payload, CRC16-CCITT-FALSE u16. All multi-byte fields are little-endian.
CRC covers version/type/length/payload, excluding sync and CRC.
"""
from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass
from typing import Union

SYNC = b"\x5a\xa5"
VERSION = 1
TYPE_SAMPLE = 1
TYPE_CONFIG = 2
TYPE_ERROR = 3
FLAG_DISCONTINUITY = 1
FLAG_FIFO_OVERRUN = 2
FLAG_I2C_ERROR = 4

_HEADER = struct.Struct("<BBH")
_SAMPLE = struct.Struct("<IQ6hH")
_CONFIG = struct.Struct("<HHHBBBBH")
_ERROR = struct.Struct("<HI")
_CRC = struct.Struct("<H")
_SIZES = {TYPE_SAMPLE: _SAMPLE.size, TYPE_CONFIG: _CONFIG.size,
          TYPE_ERROR: _ERROR.size}
assert _SAMPLE.size == 26 and _CONFIG.size == 12 and _ERROR.size == 6


@dataclass(frozen=True)
class Sample:
    sequence: int
    timestamp_us: int
    ax: int
    ay: int
    az: int
    gx: int
    gy: int
    gz: int
    flags: int = 0

    @property
    def axes(self) -> tuple[int, int, int, int, int, int]:
        return self.ax, self.ay, self.az, self.gx, self.gy, self.gz


@dataclass(frozen=True)
class Config:
    sample_rate_hz: int
    accel_range_g: int
    gyro_range_dps: int
    i2c_address: int
    sda_pin: int
    scl_pin: int
    who_am_i: int
    flags: int


@dataclass(frozen=True)
class DeviceError:
    code: int
    cumulative_count: int


Message = Union[Sample, Config, DeviceError]


@dataclass(frozen=True)
class Frame:
    message: Message
    raw: bytes
    type: int


def crc16(data: bytes) -> int:
    """CRC16-CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection/xorout."""
    return binascii.crc_hqx(data, 0xFFFF)


def encode_frame(message: Message) -> bytes:
    """Encoder for deterministic fixtures and interoperability tests."""
    if isinstance(message, Sample):
        frame_type = TYPE_SAMPLE
        payload = _SAMPLE.pack(message.sequence, message.timestamp_us,
                               *message.axes, message.flags)
    elif isinstance(message, Config):
        frame_type = TYPE_CONFIG
        payload = _CONFIG.pack(message.sample_rate_hz, message.accel_range_g,
                               message.gyro_range_dps, message.i2c_address,
                               message.sda_pin, message.scl_pin,
                               message.who_am_i, message.flags)
    elif isinstance(message, DeviceError):
        frame_type = TYPE_ERROR
        payload = _ERROR.pack(message.code, message.cumulative_count)
    else:
        raise TypeError(f"Unsupported message: {type(message)!r}")
    body = _HEADER.pack(VERSION, frame_type, len(payload)) + payload
    return SYNC + body + _CRC.pack(crc16(body))


class FrameParser:
    """Incremental, corruption-resynchronizing parser for arbitrary chunks."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.frames_parsed = 0
        self.crc_errors = 0
        self.header_errors = 0
        self.discarded_bytes = 0

    def feed(self, chunk: bytes) -> list[Frame]:
        self._buffer.extend(chunk)
        output: list[Frame] = []
        while True:
            offset = self._buffer.find(SYNC)
            if offset < 0:
                # Keep a trailing first sync byte for a split sync marker.
                keep = int(bool(self._buffer) and self._buffer[-1] == SYNC[0])
                discard = len(self._buffer) - keep
                if discard:
                    del self._buffer[:discard]
                    self.discarded_bytes += discard
                break
            if offset:
                del self._buffer[:offset]
                self.discarded_bytes += offset
            if len(self._buffer) < 6:
                break
            version, kind, length = _HEADER.unpack_from(self._buffer, 2)
            if version != VERSION or _SIZES.get(kind) != length:
                self.header_errors += 1
                del self._buffer[0]
                self.discarded_bytes += 1
                continue
            total = 6 + length + 2
            if len(self._buffer) < total:
                break
            expected = _CRC.unpack_from(self._buffer, 6 + length)[0]
            actual = crc16(bytes(self._buffer[2:6 + length]))
            if expected != actual:
                self.crc_errors += 1
                del self._buffer[0]
                self.discarded_bytes += 1
                continue
            raw = bytes(self._buffer[:total])
            payload = bytes(self._buffer[6:6 + length])
            del self._buffer[:total]
            if kind == TYPE_SAMPLE:
                message: Message = Sample(*_SAMPLE.unpack(payload))
            elif kind == TYPE_CONFIG:
                message = Config(*_CONFIG.unpack(payload))
            else:
                message = DeviceError(*_ERROR.unpack(payload))
            output.append(Frame(message, raw, kind))
            self.frames_parsed += 1
        return output


class SequenceTracker:
    """Detect dropped transmitted samples; FIFO overflow is separate."""

    def __init__(self) -> None:
        self.last_sequence: int | None = None
        self.last_timestamp_us: int | None = None
        self.missing = 0
        self.duplicates = 0
        self.out_of_order = 0
        self.timestamp_regressions = 0
        self.discontinuities = 0

    def add(self, sample: Sample) -> None:
        if self.last_sequence is not None:
            delta = (sample.sequence - self.last_sequence) & 0xFFFFFFFF
            if delta == 0:
                self.duplicates += 1
            elif delta >= 0x80000000:
                self.out_of_order += 1
            else:
                self.missing += delta - 1
                self.last_sequence = sample.sequence
        else:
            self.last_sequence = sample.sequence
        if (self.last_timestamp_us is not None and
                sample.timestamp_us < self.last_timestamp_us):
            self.timestamp_regressions += 1
        self.last_timestamp_us = sample.timestamp_us
        if sample.flags & FLAG_DISCONTINUITY:
            self.discontinuities += 1

    def summary(self) -> dict[str, int]:
        return {
            "missing_transmitted_samples": self.missing,
            "duplicates": self.duplicates,
            "out_of_order": self.out_of_order,
            "timestamp_regressions": self.timestamp_regressions,
            "discontinuities": self.discontinuities,
        }
