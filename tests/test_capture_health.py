import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import test_report_health

measure = test_report_health.measure


def device_line(seq=0):
    values = [100, 0x69, seq, 0, seq // 8, seq // 8 + 1,
              1_000_000 + (seq // 8 * 8 + 7) * 4808,
              1_000_040, 1_000_240, 48, 0, (seq % 8) * 6 + 5,
              0, 0, 0, 0, 0, 16384, 0, 0, 0]
    return (",".join(map(str, values)) + "\n").encode("ascii")


class FakeSerial:
    def __init__(self, lines=(), open_error=None):
        self.lines = list(lines)
        self.open_error = open_error
        self.dtr = self.rts = False
        self.port = None
        self.closed = False

    def open(self):
        if self.open_error:
            raise self.open_error("port unavailable")

    def close(self):
        self.closed = True

    def readline(self):
        return self.lines.pop(0) if self.lines else b""

    def read_until(self, expected=b"\n", size=None):
        line = self.lines.pop(0) if self.lines else b""
        return line[:size] if size is not None else line


def args_for(output, **kwargs):
    values = dict(port="FAKE", output=output, duration=20, baud=921600,
                  pause_at=None, pause_seconds=1, disconnect_at=None,
                  disconnect_seconds=1, logic_csv=None, sigrok_driver=None,
                  alignment=False, alignment_tolerance_ms=None,
                  startup_timeout_s=1.0, silence_timeout_s=0.5,
                  minimum_valid_rows=1, max_unplanned_recoveries=0,
                  allow_expected_faults=False, test_intent="clean", source="virtual",
                  max_line_bytes=4096, max_output_bytes=1_000_000,
                  minimum_free_bytes=0, clip_threshold=0.98,
                  nominal_odr_hz=208.0, confirmed_tr_c=False, config=None)
    values.update(kwargs)
    return SimpleNamespace(**values)


class CaptureHealthTests(unittest.TestCase):
    def run_capture(self, output, lines=(), *, open_error=None, **kwargs):
        clock = [0.0]

        def monotonic():
            clock[0] += 0.1
            return clock[0]

        def sleep(seconds):
            clock[0] += seconds

        fake_module = SimpleNamespace(Serial=lambda **_kwargs: FakeSerial(lines, open_error),
                                      SerialException=OSError)
        with mock.patch.object(measure.time, "monotonic", monotonic), \
             mock.patch.object(measure.time, "sleep", sleep), \
             contextlib.redirect_stdout(io.StringIO()):
            summary = measure.capture(args_for(output, **kwargs), serial_module=fake_module)
        return summary

    def test_header_only_device_fails_startup_and_writes_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "capture"
            summary = self.run_capture(output, startup_timeout_s=0.6)
            self.assertEqual(summary["quality_verdict"], "FAILED")
            self.assertIn("NO_SAMPLES", summary["quality_reasons"])
            self.assertIn("STARTUP_TIMEOUT", summary["quality_reasons"])
            self.assertTrue((output / "capture.csv").is_file())
            self.assertTrue((output / "events.jsonl").is_file())
            self.assertTrue((output / "report.json").is_file())

    def test_device_fatal_message_is_failed_even_without_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = self.run_capture(Path(temp) / "capture",
                                       [b"# ERROR: Unrecognized sensor\n"])
        self.assertEqual(summary["quality_verdict"], "FAILED")
        self.assertIn("DEVICE_FATAL_ERROR", summary["quality_reasons"])

    def test_valid_stream_that_goes_silent_fails_with_partial_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "capture"
            summary = self.run_capture(output, [device_line(0)], silence_timeout_s=0.4)
            self.assertGreater(summary["rows"], 0)
            self.assertEqual(summary["quality_verdict"], "FAILED")
            self.assertIn("SILENT_STREAM", summary["quality_reasons"])
            self.assertTrue((output / "report.json").is_file())

    def test_no_device_fails_without_waiting_out_full_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = self.run_capture(Path(temp) / "capture", open_error=OSError)
        self.assertEqual(summary["quality_verdict"], "FAILED")
        self.assertIn("PORT_OPEN_FAILED", summary["quality_reasons"])
        self.assertEqual(summary["events_summary"]["unplanned_recoveries"], 1)

    def test_oversized_line_is_read_with_a_hard_byte_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "capture"
            summary = self.run_capture(output, [b"x" * 800], max_line_bytes=128,
                                       startup_timeout_s=5)
            event_text = (output / "events.jsonl").read_text()
        self.assertEqual(summary["quality_verdict"], "FAILED")
        self.assertIn("INVALID_ROW_FORMAT", summary["quality_reasons"])
        events = [json.loads(line) for line in event_text.splitlines()]
        malformed = next(event for event in events if event["event"] == "malformed_line")
        self.assertEqual(malformed["byte_count"], 129)

    def test_capture_artifacts_stop_at_configured_output_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "capture"
            summary = self.run_capture(output, [device_line(seq) for seq in range(100)],
                                       max_output_bytes=1200)
            raw_size = (output / "capture.csv").stat().st_size
            event_size = (output / "events.jsonl").stat().st_size
        self.assertEqual(summary["quality_verdict"], "FAILED")
        self.assertIn("OUTPUT_SIZE_LIMIT", summary["quality_reasons"])
        self.assertLessEqual(raw_size + event_size, 1200)

    def test_capture_rejects_nonfinite_duration_before_opening_serial(self):
        with tempfile.TemporaryDirectory() as temp:
            args = args_for(Path(temp) / "capture", duration=float("inf"))
            fake_module = SimpleNamespace(Serial=mock.Mock(), SerialException=OSError)
            with self.assertRaisesRegex(SystemExit, "duration"):
                measure.capture(args, serial_module=fake_module)
            fake_module.Serial.assert_not_called()


if __name__ == "__main__":
    unittest.main()
