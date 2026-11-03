import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import measure


def sample_rows(batches=4, *, pattern=0, session=0, boot=100, queue_end=0):
    rows = []
    base_word = (5 - pattern) % 6
    period = 1_000_000 / 208
    for batch in range(batches):
        irq = round(1_000_000 + (batch * 8 + 7) * period)
        for index in range(8):
            seq = batch * 8 + index
            rows.append({
                "host_elapsed_s": f"{session * 10 + boot / 1000 + seq / 208:.6f}",
                "session": session,
                "boot_id": boot,
                "who_am_i": 0x69,
                "seq": seq,
                "epoch": 0,
                "batch": batch,
                "irq_count": batch + 1,
                "irq_us": irq,
                "read_start_us": irq + 40,
                "read_end_us": irq + 240,
                "fifo_words": 48,
                "pattern": pattern,
                "word_end": base_word + index * 6,
                "gx": index,
                "gy": -index,
                "gz": 2 * index,
                "ax": 0,
                "ay": 0,
                "az": 16384,
                "queue_drops": queue_end if index == 7 and batch == batches - 1 else 0,
                "fifo_recoveries": 0,
                "overrun_observations": 0,
            })
    return rows


def write_capture(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["host_elapsed_s", "session"] + measure.FIELDS
        if any("stationary" in row for row in rows):
            fieldnames.append("stationary")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ReportHealthTests(unittest.TestCase):
    def analyze(self, root, rows, *, events=None, **kwargs):
        capture = root / "capture.csv"
        output = root / "report"
        write_capture(capture, rows)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = measure.analyze(capture, output, events_jsonl=events, **kwargs)
        return summary, output, capture

    def test_empty_capture_has_failed_verdict_and_strict_cli_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "empty.csv"
            write_capture(capture, [])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = measure.analyze(capture, root / "report")
            self.assertEqual(summary["quality_verdict"], "FAILED")
            self.assertIn("NO_SAMPLES", summary["quality_reasons"])
            self.assertTrue((root / "report" / "report.json").exists())
            process = subprocess.run(
                [sys.executable, str(TOOLS / "measure.py"), "report", str(capture),
                 "--output", str(root / "strict-report"), "--strict"],
                capture_output=True, text=True, check=False)
            self.assertNotEqual(process.returncode, 0)
            written = json.loads((root / "strict-report" / "report.json").read_text())
            self.assertEqual(written["quality_verdict"], "FAILED")

    def test_duplicate_csv_columns_fail_instead_of_silently_overwriting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "duplicate.csv"
            fieldnames = ["host_elapsed_s", "session"] + measure.FIELDS
            row = sample_rows(1)[0]
            with capture.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(fieldnames + ["seq"])
                writer.writerow([row[name] for name in fieldnames] + [row["seq"]])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = measure.analyze(capture, root / "report")
        self.assertEqual(summary["quality_verdict"], "FAILED")
        self.assertIn("DUPLICATE_COLUMNS", summary["quality_reasons"])

    def test_report_cli_rejects_nonfinite_or_nonpositive_policy_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "capture.csv"
            write_capture(capture, sample_rows(1))
            for option, value in (("--nominal-odr-hz", "0"),
                                  ("--nominal-odr-hz", "nan"),
                                  ("--clip-threshold", "nan")):
                with self.subTest(option=option, value=value):
                    process = subprocess.run(
                        [sys.executable, str(TOOLS / "measure.py"), "report", str(capture),
                         "--output", str(root / (option[2:] + value)), option, value],
                        capture_output=True, text=True, check=False)
                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn("must be", process.stderr)

    def test_counter_totals_sum_independent_boot_sessions(self):
        rows = []
        for boot in (100, 200):
            part = sample_rows(3, boot=boot)
            for row in part:
                row["queue_drops"] = min(row["seq"], 9)
            rows.extend(part)
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        sessions = summary["per_session_counters"]
        self.assertEqual([entry["last_observed"]["queue_drops"] for entry in sessions], [9, 9])
        self.assertEqual(summary["total_observed_session_end_counters"]["queue_drops"], 18)
        self.assertTrue(all(entry["monotonic"]["queue_drops"] for entry in sessions))

    def test_counter_decrease_is_reported_without_rewriting_observed_totals(self):
        rows = sample_rows(2)
        for row in rows[8:]:
            row["queue_drops"] = 2
        rows[8]["queue_drops"] = 4
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        self.assertFalse(summary["per_session_counters"][0]["monotonic"]["queue_drops"])
        self.assertEqual(summary["total_observed_session_end_counters"]["queue_drops"], 2)
        self.assertIn("COUNTER_DECREASE", summary["quality_reasons"])

    def test_robust_fit_downweights_a_delayed_irq_outlier(self):
        period = 1_000_000 / 208
        points = [(index * 8, round(50_000 + index * 8 * period + (120_000 if index == 8 else 0)))
                  for index in range(20)]
        fit = measure.fit_robust_line(points)
        self.assertAlmostEqual(fit["estimated_hz"], 208, delta=0.1)
        self.assertGreaterEqual(fit["anchors_downweighted"], 1)
        self.assertEqual(fit["method"], "huber_irls")

    def test_backward_mcu_anchor_is_rejected_and_fit_remains_segment_local(self):
        rows = sample_rows(10)
        for row in rows[40:48]:
            row["irq_us"] += 200_000
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        segment = summary["segments"][0]
        self.assertEqual(segment["anchors_rejected_nonmonotonic"], 1)
        self.assertIn("MCU_TIME_NONMONOTONIC", summary["quality_reasons"])
        self.assertAlmostEqual(segment["model"]["estimated_hz"], 208, delta=0.1)
        self.assertEqual(summary["estimated_sample_time_clock"], "mcu_irq_us_since_boot")

    def test_too_few_irq_anchors_leave_reconstructed_timestamps_blank(self):
        with tempfile.TemporaryDirectory() as temp:
            summary, output, _ = self.analyze(Path(temp), sample_rows(1))
            self.assertIn("UNANCHORED_PATTERN", summary["quality_reasons"])
            with (output / "reconstructed.csv").open(encoding="utf-8", newline="") as stream:
                reconstructed = list(csv.DictReader(stream))
        self.assertTrue(all(row["estimated_sample_host_us"] == "" for row in reconstructed))

    def test_stationary_metrics_require_explicit_labels_and_axis_inactivity_is_reported(self):
        rows = sample_rows()
        for row in rows[:8]:
            row["stationary"] = "true"
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        metrics = summary["quality"]["stationary_metrics"]
        self.assertEqual(metrics["status"], "computed_from_explicit_labels")
        self.assertEqual(metrics["windows"][0]["samples"], 8)
        self.assertAlmostEqual(metrics["windows"][0]["accel_norm_mean_g"], 1.0, delta=0.001)
        self.assertIn("ax", summary["quality"]["inactive_axes_constant_over_recording"])

    def test_sample_rate_far_from_nominal_is_degraded(self):
        rows = sample_rows()
        for row in rows:
            batch = row["batch"]
            row["irq_us"] = 1_000_000 + (batch * 8 + 7) * 10_000
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        self.assertIn("ODR_OUT_OF_RANGE", summary["quality_reasons"])
        self.assertEqual(summary["quality_verdict"], "DEGRADED")

    def test_each_fifo_pattern_has_word_derived_timing_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for pattern in range(6):
                with self.subTest(pattern=pattern):
                    summary, _, _ = self.analyze(root, sample_rows(pattern=pattern), test_intent="clean")
                    self.assertEqual(summary["segments"][0]["model"]["anchors"], 4)
                    self.assertEqual(summary["segments"][0]["anchor_pattern_offsets"], [pattern])
                    self.assertNotIn("UNANCHORED_PATTERN", summary["quality_reasons"])

    def test_clipping_is_counted_by_axis_and_fails_clean_quality_policy(self):
        rows = sample_rows()
        rows[5]["ax"] = 32767
        rows[7]["gz"] = -32768
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows, test_intent="clean")
        self.assertEqual(summary["quality"]["clipping_by_axis"]["ax"], 1)
        self.assertEqual(summary["quality"]["clipping_by_axis"]["gz"], 1)
        self.assertIn("CLIPPING_DETECTED", summary["quality_reasons"])
        self.assertEqual(summary["quality_verdict"], "FAILED")

    def test_report_summarizes_transport_events_and_planned_fault_intent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = root / "events.jsonl"
            events.write_text("\n".join([
                json.dumps({"event": "read_pause_start", "planned": True}),
                json.dumps({"event": "port_error", "planned": True}),
                json.dumps({"event": "malformed_line", "planned": True}),
            ]) + "\n", encoding="utf-8")
            summary, _, _ = self.analyze(root, sample_rows(), events=events,
                                        test_intent="fault_injection", allow_expected_faults=True)
        self.assertEqual(summary["events_summary"]["counts"]["read_pause_start"], 1)
        self.assertEqual(summary["events_summary"]["counts"]["port_error"], 1)
        self.assertEqual(summary["events_summary"]["counts"]["malformed_line"], 1)
        self.assertEqual(summary["quality_verdict"], "DEGRADED_EXPECTED")

    def test_unplanned_transport_fault_cannot_be_accepted_as_expected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event": "port_error", "planned": False}) + "\n", encoding="utf-8")
            summary, _, _ = self.analyze(root, sample_rows(), events=events,
                                        test_intent="fault_injection", allow_expected_faults=True)
        self.assertEqual(summary["quality_verdict"], "DEGRADED")
        self.assertIn("UNPLANNED_TRANSPORT_ERRORS", summary["quality_reasons"])

    def test_invalid_raw_values_and_bad_patterns_are_reported(self):
        rows = sample_rows()
        rows[3]["ax"] = 32768
        rows[4]["pattern"] = 6
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        self.assertGreaterEqual(summary["invalid_rows"], 2)
        self.assertIn("INVALID_CAPTURE_ROWS", summary["quality_reasons"])
        self.assertEqual(summary["report_schema_version"], "2.0")
        self.assertIn("quality_policy_version", summary)

    def test_missing_required_values_are_counted_by_field(self):
        rows = sample_rows(2)
        rows[2]["gy"] = ""
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        self.assertEqual(summary["quality"]["missing_values_by_field"]["gy"], 1)
        self.assertEqual(summary["quality"]["missingness_rate_by_field"]["gy"], 1 / len(rows))
        self.assertEqual(summary["quality_verdict"], "FAILED")

    def test_batch_and_interrupt_counter_regressions_are_reported(self):
        rows = sample_rows(3)
        rows[16]["batch"] = 0
        rows[16]["irq_count"] = 1
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        self.assertIn("BATCH_INDEX_NONMONOTONIC", summary["quality_reasons"])
        self.assertIn("IRQ_COUNT_NONMONOTONIC", summary["quality_reasons"])

    def test_batch_gap_and_inconsistent_continuation_metadata_are_reported(self):
        rows = sample_rows(3)
        for row in rows[8:16]:
            row["batch"] = 2
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        self.assertIn("BATCH_INDEX_GAP", summary["quality_reasons"])

        rows = sample_rows(1)
        for row in rows:
            row["irq_us"] = -1
        rows[4]["fifo_words"] = 47
        with tempfile.TemporaryDirectory() as temp:
            summary, _, _ = self.analyze(Path(temp), rows)
        self.assertIn("INCONSISTENT_BATCH_METADATA", summary["quality_reasons"])
        self.assertEqual(summary["quality_verdict"], "FAILED")


if __name__ == "__main__":
    unittest.main()
