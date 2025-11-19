import unittest

import numpy as np

from xiao_mg24_ahrs_gui import (
    PACKET_PREAMBLE,
    PACKET_STRUCT,
    PACKET_VERSION,
    ImuPacket,
    checksum_bytes,
    compute_packet_stats,
    extract_packets,
    quat_to_direction_vector,
)


def make_packet(
    sequence: int,
    timestamp_ms: int,
    ax_g: float = 0.0,
    ay_g: float = 0.0,
    az_g: float = -1.0,
    gx_dps: float = 0.0,
    gy_dps: float = 0.0,
    gz_dps: float = 0.0,
) -> bytes:
    """Create a binary packet matching the sketch's layout for tests."""

    values = (
        PACKET_PREAMBLE,
        PACKET_VERSION,
        sequence,
        timestamp_ms,
        ax_g,
        ay_g,
        az_g,
        gx_dps,
        gy_dps,
        gz_dps,
        0,
    )
    raw = bytearray(PACKET_STRUCT.pack(*values))
    checksum = checksum_bytes(raw[:-2])
    raw[-2:] = checksum.to_bytes(2, "little")
    return bytes(raw)


class PacketParsingTests(unittest.TestCase):
    def test_checksum_bytes_simple(self):
        self.assertEqual(checksum_bytes(b"\x01\x02\x03"), 6)

    def test_extract_packets_with_noise(self):
        noise = b"\x00\x01\x02"
        packet_a = make_packet(10, 1000, ax_g=0.1)
        packet_b = make_packet(11, 1050, ax_g=0.2)
        buffer = bytearray(noise + packet_a + b"\xFF" + packet_b)
        packets = extract_packets(buffer)

        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0].sequence, 10)
        self.assertAlmostEqual(packets[1].ax_g, 0.2, places=4)
        self.assertEqual(buffer, bytearray())  # fully consumed

    def test_compute_packet_stats_detects_drops(self):
        packets = [
            ImuPacket(sequence=1, timestamp_ms=0, ax_g=0, ay_g=0, az_g=-1, gx_dps=0, gy_dps=0, gz_dps=0),
            ImuPacket(sequence=2, timestamp_ms=5, ax_g=0, ay_g=0, az_g=-1, gx_dps=0, gy_dps=0, gz_dps=0),
            ImuPacket(sequence=5, timestamp_ms=15, ax_g=0, ay_g=0, az_g=-1, gx_dps=0, gy_dps=0, gz_dps=0),
        ]
        stats = compute_packet_stats(packets)

        self.assertEqual(stats.count, 3)
        self.assertEqual(stats.dropped_packets, 2)  # sequences 3 and 4 missing
        self.assertAlmostEqual(stats.avg_interval_ms, 7.5)
        self.assertEqual(stats.first_sequence, 1)
        self.assertEqual(stats.last_sequence, 5)

    def test_quaternion_rotation_matches_expected_direction(self):
        # 180 degree rotation around X should flip Z axis direction
        quat = np.array([0.0, 1.0, 0.0, 0.0])
        vec = quat_to_direction_vector(quat)
        np.testing.assert_allclose(vec, np.array([0.0, 0.0, 1.0]), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
