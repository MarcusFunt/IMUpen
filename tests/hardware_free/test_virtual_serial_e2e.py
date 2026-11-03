import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "measure.py"
SUPPORT = Path(__file__).resolve().parent


def row_line(seq):
    batch, index = divmod(seq, 8)
    irq = 1_000_000 + (batch * 8 + 7) * round(1_000_000 / 208)
    values = [123456, 0x69, seq, 0, batch, batch + 1, irq, irq + 40, irq + 240,
              48, 0, index * 6 + 5, 0, 0, 0, 0, 0, 16384, 0, 0, 0]
    return ",".join(map(str, values)) + "\n"


class VirtualSerialEndToEndTests(unittest.TestCase):
    def run_cli(self, root, events, *extra, open_error=None):
        script = root / "serial-script.json"
        script.write_text(json.dumps({"events": events, "open_error": open_error}), encoding="utf-8")
        output = root / "capture"
        env = os.environ.copy()
        env["IMUPEN_VIRTUAL_SERIAL_SCRIPT"] = str(script)
        env["PYTHONPATH"] = str(SUPPORT) + os.pathsep + env.get("PYTHONPATH", "")
        command = [sys.executable, str(TOOL), "capture", "--port", "VIRTUAL",
                   "--output", str(output), "--baud", "921600", *map(str, extra)]
        process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                 text=True, timeout=15, check=False)
        report_path = output / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
        return process, output, report

    def test_real_capture_cli_consumes_clean_virtual_firmware_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [{"data": row_line(seq), "delay_s": 0.001} for seq in range(56)]
            process, output, report = self.run_cli(
                root, events, "--duration", "0.35", "--startup-timeout-s", "0.1",
                "--silence-timeout-s", "1", "--minimum-valid-rows", "40", "--source", "virtual")
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(report["quality_verdict"], "CLEAN")
            self.assertEqual(report["rows"], 56)
            self.assertEqual(report["segments"][0]["model"]["anchors"], 7)
            with (output / "capture.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 56)
            self.assertTrue((output / "events.jsonl").is_file())

    def test_real_capture_cli_classifies_header_only_and_device_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            header, _, report = self.run_cli(
                root, [{"data": "# format=imupen_fifo_v1\n"}], "--duration", "0.5",
                "--startup-timeout-s", "0.12", "--silence-timeout-s", "0.2")
            self.assertNotEqual(header.returncode, 0)
            self.assertEqual(report["quality_verdict"], "FAILED")
            self.assertIn("NO_SAMPLES", report["quality_reasons"])

        with tempfile.TemporaryDirectory() as temp:
            error, _, report = self.run_cli(
                Path(temp), [{"data": "# ERROR: Unrecognized/unconfirmed variant\n"}],
                "--duration", "1", "--startup-timeout-s", "0.5")
            self.assertNotEqual(error.returncode, 0)
            self.assertIn("DEVICE_FATAL_ERROR", report["quality_reasons"])

    def test_real_capture_cli_marks_scheduled_reconnect_as_expected_fault(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [{"data": row_line(seq), "delay_s": 0.006} for seq in range(56)]
            process, _, report = self.run_cli(
                root, events, "--duration", "0.75", "--startup-timeout-s", "0.2",
                "--silence-timeout-s", "1", "--disconnect-at", "0.12",
                "--disconnect-seconds", "0.03", "--test-intent", "fault_injection",
                "--allow-expected-faults", "--source", "virtual")
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(report["quality_verdict"], "DEGRADED_EXPECTED")
            self.assertGreater(report["events_summary"]["planned_fault_events"], 0)
            self.assertEqual(report["events_summary"]["unplanned_recoveries"], 0)
            self.assertEqual(report["rows"], 56)

    def test_no_device_open_failure_still_publishes_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            process, output, report = self.run_cli(
                root, [], "--duration", "1", "--startup-timeout-s", "0.2",
                open_error="simulated no device")
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(report["quality_verdict"], "FAILED")
            self.assertIn("PORT_OPEN_FAILED", report["quality_reasons"])
            self.assertTrue((output / "capture.csv").is_file())
            self.assertTrue((output / "events.jsonl").is_file())

    def test_partial_serial_reads_are_reassembled_and_truncated_tail_is_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = row_line(0)
            events = [{"data": first[:13], "delay_s": 0.001},
                      {"data": first[13:], "delay_s": 0.001}]
            events.extend({"data": row_line(seq), "delay_s": 0.001} for seq in range(1, 40))
            process, _, report = self.run_cli(
                root, events, "--duration", "0.25", "--startup-timeout-s", "0.1",
                "--silence-timeout-s", "0.8", "--minimum-valid-rows", "32", "--source", "virtual")
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(report["rows"], 40)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            process, output, report = self.run_cli(
                root, [{"data": row_line(0)[:-1]}], "--duration", "0.15",
                "--startup-timeout-s", "0.1", "--silence-timeout-s", "0.1")
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(report["quality_verdict"], "FAILED")
            self.assertIn("INVALID_ROW_FORMAT", report["quality_reasons"])
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertIn("TRUNCATED_LINE", [event.get("reason") for event in events])

    def test_unplanned_transient_transport_error_is_not_accepted_as_expected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [{"raise": "one simulated read exception"}]
            events.extend({"data": row_line(seq), "delay_s": 0.001} for seq in range(40))
            process, _, report = self.run_cli(
                root, events, "--duration", "2", "--startup-timeout-s", "0.2",
                "--silence-timeout-s", "1", "--max-unplanned-recoveries", "1",
                "--test-intent", "fault_injection", "--allow-expected-faults", "--source", "virtual")
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(report["quality_verdict"], "DEGRADED")
            self.assertEqual(report["events_summary"]["unplanned_recoveries"], 1)
            self.assertEqual(report["rows"], 40)

    def test_duplicate_firmware_sequence_is_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = [{"data": row_line(0)}, {"data": row_line(0)}]
            process, _, report = self.run_cli(
                root, events, "--duration", "0.5", "--startup-timeout-s", "0.2",
                "--silence-timeout-s", "1", "--source", "virtual")
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual(report["quality_verdict"], "FAILED")
            self.assertIn("DUPLICATE_SEQUENCE", report["quality_reasons"])


if __name__ == "__main__":
    unittest.main()
