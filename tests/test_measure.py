import contextlib
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / "tools" / "measure.py"
spec = importlib.util.spec_from_file_location("measure", MODULE)
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


class MeasurementTests(unittest.TestCase):
    def test_capture_runs_pause_and_reconnect_schedule(self):
        clock = [0.0]
        seq = [0]
        class FakeSerial:
            def __init__(self, **kwargs):
                self.dtr = self.rts = False
                self.port = None
            def open(self):
                pass
            def close(self):
                pass
            def readline(self):
                n = seq[0]
                seq[0] += 1
                batch, pos = divmod(n, 8)
                irq = 1000000 + (batch * 8 + 7) * 5000
                values = [1234, 105, n, 0, batch, batch + 1, irq, irq + 40, irq + 80,
                          48, 0, pos * 6 + 5, 0, 0, 0, 0, 0, 16384, 0, 0, 0]
                return (",".join(map(str, values)) + "\n").encode()
        def monotonic():
            clock[0] += .02
            return clock[0]
        def sleep(seconds):
            clock[0] += seconds
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(port="FAKE", output=tmp, duration=2, baud=921600,
                                   pause_at=.4, pause_seconds=.1, disconnect_at=1.1,
                                   disconnect_seconds=.2, logic_csv=None, sigrok_driver=None,
                                   alignment=False, alignment_tolerance_ms=None)
            fake_module = SimpleNamespace(Serial=FakeSerial, SerialException=OSError)
            with mock.patch.dict("sys.modules", {"serial": fake_module}), \
                 mock.patch.object(measure.time, "monotonic", monotonic), \
                 mock.patch.object(measure.time, "sleep", sleep), \
                 contextlib.redirect_stdout(io.StringIO()):
                measure.capture(args)
            events = [json.loads(line)["event"] for line in (Path(tmp) / "events.jsonl").read_text().splitlines()]
            self.assertIn("read_pause_start", events)
            self.assertIn("read_pause_end", events)
            self.assertIn("port_closed", events)
            self.assertGreaterEqual(events.count("port_open"), 2)
            self.assertTrue((Path(tmp) / "report.json").exists())

    def test_report_recovers_rate_and_detects_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture.csv"
            with capture.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["host_elapsed_s", "session"] + measure.FIELDS)
                for seq in range(64):
                    if seq == 10:
                        continue
                    batch, index = divmod(seq, 8)
                    irq = 1000000 + (batch * 8 + 7) * 5000
                    writer.writerow([seq / 200, 0, 1234, 105, seq, 0, batch, batch + 1, irq,
                                     irq + 50, irq + 100, 48, 0, index * 6 + 5,
                                     0, 0, 0, 0, 0, 16384, int(seq > 10), 0, 0])
            logic = root / "logic.csv"
            logic.write_text("time_s,int1,cs\n0,0,1\n0.001,1,1\n0.001025,1,0\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                measure.analyze(capture, root / "out", str(logic))
            report = json.loads((root / "out" / "report.json").read_text())
            self.assertEqual(report["missing_host_sequence_numbers"], 1)
            self.assertEqual(report["maximum_cumulative_counters_per_session"]["queue_drops"], 1)
            self.assertAlmostEqual(report["segments"][0]["model"]["estimated_hz"], 200)
            self.assertAlmostEqual(report["logic_analyzer"]["irq_to_next_cs_falling_us"]["median"], 25)
            with (root / "out" / "reconstructed.csv").open(newline="") as f:
                row = next(csv.DictReader(f))
                self.assertAlmostEqual(float(row["estimated_sample_host_us"]), 1000000)

    def test_sigrok_conversion_preserves_sample_clock_and_channel_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "samples.csv"
            source.write_text("CS,INT1\n1,0\n1,1\n1,1\n0,1\n", encoding="utf-8")
            measure.convert_sigrok(source, root / "edges.csv", 1000000)
            result = measure.logic_metrics(root / "edges.csv")
            self.assertAlmostEqual(result["irq_to_next_cs_falling_us"]["median"], 2)

    def test_logic_does_not_pair_interrupt_when_chip_select_is_already_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "logic.csv"
            path.write_text("time_s,int1,cs\n0,0,1\n0.001,0,0\n0.002,1,0\n0.003,1,1\n0.004,1,0\n", encoding="utf-8")
            result = measure.logic_metrics(path)
            self.assertEqual(result["unmatched_irq_edges"], 1)
            self.assertIsNone(result["irq_to_next_cs_falling_us"])

    def test_rotation_check_recovers_known_gyro_delay(self):
        hz, delay = 200.0, 2
        def theta(t):
            return 0.4 * math.sin(2 * math.pi * .4 * t) + 0.2 * math.sin(2 * math.pi * .9 * t)
        def omega(t):
            return -(0.4 * 2 * math.pi * .4 * math.cos(2 * math.pi * .4 * t) +
                     0.2 * 2 * math.pi * .9 * math.cos(2 * math.pi * .9 * t))
        span = []
        for i in range(1200):
            t = i / hz
            angle = theta(t)
            span.append(([0, 0, omega((i - delay) / hz)], [math.cos(angle), math.sin(angle), 0]))
        key = (0, 0)
        result = measure.alignment_metrics({key: [span]}, {key: {"estimated_hz": hz}}, 15)[0]
        self.assertEqual(result["best_gyro_offset_samples"], delay)
        self.assertGreater(result["correlation"], .99)
        self.assertTrue(result["within_tolerance_if_assumptions_hold"])


if __name__ == "__main__":
    unittest.main()
