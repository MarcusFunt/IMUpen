"""Tests for the IMUpen host utilities."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import xiao_mg24_ahrs_gui as gui


def build_packet(
    *,
    sequence: int,
    timestamp_ms: int = 0,
    accel: Iterable[float] = (0.0, 0.0, -1.0),
    gyro: Iterable[float] = (0.0, 0.0, 0.0),
) -> bytes:
    """Return a binary packet that mirrors the sketch output."""

    ax_g, ay_g, az_g = accel
    gx_dps, gy_dps, gz_dps = gyro
    without_checksum = gui.PACKET_STRUCT.pack(
        gui.PACKET_PREAMBLE,
        gui.PACKET_VERSION,
        sequence,
        timestamp_ms,
        float(ax_g),
        float(ay_g),
        float(az_g),
        float(gx_dps),
        float(gy_dps),
        float(gz_dps),
        0,  # placeholder checksum
    )
    checksum = gui.checksum_bytes(without_checksum[:-2])
    return gui.PACKET_STRUCT.pack(
        gui.PACKET_PREAMBLE,
        gui.PACKET_VERSION,
        sequence,
        timestamp_ms,
        float(ax_g),
        float(ay_g),
        float(az_g),
        float(gx_dps),
        float(gy_dps),
        float(gz_dps),
        checksum,
    )


def test_checksum_bytes_wraps_at_16_bits():
    assert gui.checksum_bytes(bytes(range(256))) == sum(range(256)) & 0xFFFF


def test_extract_packets_handles_noise_and_partial_data():
    buffer = bytearray(b"\x00\xFF")  # preamble noise should be discarded
    buffer.extend(build_packet(sequence=1, timestamp_ms=10))
    buffer.extend(build_packet(sequence=2, timestamp_ms=20))
    buffer.extend(b"\x01\x02")  # partial packet remains in the buffer

    packets = gui.extract_packets(buffer)

    assert [p.sequence for p in packets] == [1, 2]
    assert buffer == bytearray()  # trailing bytes without a preamble are cleared


def test_configure_history_length_resets_ring_buffers():
    gui.configure_history_length(5)

    assert gui.vec_x_hist.maxlen == 5
    assert gui.vec_y_hist.maxlen == 5
    assert gui.vec_z_hist.maxlen == 5
    assert gui.sample_time_hist.maxlen == 5

    # pushing samples should respect the configured maxlen
    for value in range(10):
        gui.vec_x_hist.append(value)
    assert gui.vec_x_hist[0] == 5
    assert gui.vec_x_hist[-1] == 9
