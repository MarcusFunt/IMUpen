import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import dataset
import measure


def write_raw(path, sequences):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["host_elapsed_s", "session"] + measure.FIELDS)
        for seq in sequences:
            batch, index = divmod(seq, 8)
            irq = 1_000_000 + (batch * 8 + 7) * 4808
            values = [123, 0x69, seq, 0, batch, batch + 1, irq, irq + 40, irq + 240,
                      48, 0, index * 6 + 5, 100, -100, 50, 1000, -1000, 16000,
                      0, 0, 0]
            writer.writerow([f"{seq / 208:.9f}", 0, *values])


def write_canonical_sine(path, *, frequency_hz, segments=((0, 0.0, 4.0),), rate_hz=208.0):
    fields = ["time_us", "segment_id", "seq", "validity_mask", *dataset.SCALED_AXES]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for segment in segments:
            segment_id, start_s, end_s = segment[:3]
            seq_offset = segment[3] if len(segment) > 3 else 0
            count = round((end_s - start_s) * rate_hz)
            for index in range(count):
                t = start_s + index / rate_hz
                signal = math.sin(2 * math.pi * frequency_hz * t)
                writer.writerow({"time_us": round(t * 1e6), "segment_id": segment_id,
                                 "seq": index + seq_offset, "validity_mask": 63,
                                 "gx_dps": 0, "gy_dps": 0, "gz_dps": 0,
                                 "ax_g": signal, "ay_g": 0, "az_g": 0})
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps({"source_type": "synthetic", "synthetic": True}), encoding="utf-8")


class DatasetExportTests(unittest.TestCase):
    def test_canonical_manifest_preserves_raw_scaled_calibrated_and_gap_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "raw.csv"
            canonical = root / "canonical.csv"
            write_raw(capture, [*range(8), *range(10, 42)])
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps({
                "version": "bias-v1", "gyro_bias": [1.75, -1.75, 0.875],
                "accel_bias": [0.061, -0.061, 0.976],
            }), encoding="utf-8")
            metadata = root / "writer.json"
            metadata.write_text(json.dumps({"writer_id": "W-01", "prompt": "sample"}), encoding="utf-8")
            manifest = dataset.canonicalize(capture, canonical, source_type="synthetic",
                                            calibration_path=calibration,
                                            writer_metadata_path=metadata)
            with canonical.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 40)
            self.assertEqual(rows[0]["validity_mask"], "63")
            self.assertAlmostEqual(float(rows[0]["gx_dps"]), 1.75)
            self.assertAlmostEqual(float(rows[0]["ax_g"]), 0.061)
            self.assertAlmostEqual(float(rows[0]["gx_cal_dps"]), 0.0)
            self.assertAlmostEqual(float(rows[0]["ax_cal_g"]), 0.0)
            self.assertEqual(rows[8]["segment_id"], "1")
            self.assertEqual(manifest["source_type"], "synthetic")
            self.assertTrue(manifest["synthetic"])
            self.assertEqual(manifest["calibration"]["status"], "applied")
            self.assertIn("writer_id", manifest["writer_metadata"])
            self.assertEqual(manifest["timing_sources"], ["host_receive_elapsed"])
            self.assertTrue(manifest["source_capture_sha256"])

    def test_canonical_export_requires_explicit_real_or_synthetic_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "raw.csv"
            write_raw(capture, range(8))
            with self.assertRaisesRegex(ValueError, "source_type"):
                dataset.canonicalize(capture, root / "canonical.csv", source_type="unknown")

    def test_canonical_export_rejects_duplicate_column_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "raw.csv"
            fieldnames = ["host_elapsed_s", "session"] + measure.FIELDS
            values = ["0", 0, 123, 0x69, 0, 0, 0, 0, -1, 0, 0, 48, 0, 5,
                      0, 0, 0, 0, 0, 16384, 0, 0, 0]
            with capture.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(fieldnames + ["seq"])
                writer.writerow(values + [0])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                dataset.canonicalize(capture, root / "canonical.csv", source_type="real")

    def test_resampler_rejects_nonpositive_filter_width(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            canonical = root / "canonical.csv"
            write_canonical_sine(canonical, frequency_hz=10)
            for width in (0, -1):
                with self.subTest(width=width), self.assertRaisesRegex(ValueError, "half_width"):
                    dataset.resample_canonical(canonical, root / f"out-{width}.csv",
                                               half_width_samples=width)

    def test_208_to_100_resampling_passes_low_frequency_and_rejects_alias(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            low = root / "low.csv"
            low_out = root / "low-100.csv"
            write_canonical_sine(low, frequency_hz=10)
            low_manifest = dataset.resample_canonical(low, low_out, target_rate_hz=100)
            with low_out.open(newline="", encoding="utf-8") as stream:
                low_rows = list(csv.DictReader(stream))
            low_rms = math.sqrt(sum(float(row["ax_g"]) ** 2 for row in low_rows) / len(low_rows))
            self.assertAlmostEqual(low_rms, 1 / math.sqrt(2), delta=0.02)
            self.assertEqual(low_manifest["method"], "windowed_sinc_blackman_lowpass")
            self.assertEqual(low_manifest["target_rate_hz"], 100)
            self.assertTrue(low_manifest["synthetic"])

            high = root / "high.csv"
            high_out = root / "high-100.csv"
            write_canonical_sine(high, frequency_hz=70)
            dataset.resample_canonical(high, high_out, target_rate_hz=100)
            with high_out.open(newline="", encoding="utf-8") as stream:
                high_rows = list(csv.DictReader(stream))
            alias_rms = math.sqrt(sum(float(row["ax_g"]) ** 2 for row in high_rows) / len(high_rows))
            self.assertLess(alias_rms, 0.03)

    def test_resampler_filters_discontinuity_segments_independently(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "segments.csv", root / "resampled.csv"
            write_canonical_sine(source, frequency_hz=10,
                                 segments=((0, 0.0, 2.0), (0, 4.0, 6.0, 416)))
            manifest = dataset.resample_canonical(source, output, target_rate_hz=100)
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            ids = {row["source_segment_id"] for row in rows}
            chunks = {row["continuity_chunk_id"] for row in rows}
            times = [float(row["time_us"]) / 1e6 for row in rows]
        self.assertEqual(ids, {"0"})
        self.assertEqual(chunks, {"0:0", "0:1"})
        self.assertEqual(manifest["segments"], 2)
        self.assertFalse(any(2.0 < t < 4.0 for t in times))
        self.assertIn("no interpolation crosses a gap", manifest["gap_policy"])


if __name__ == "__main__":
    unittest.main()
