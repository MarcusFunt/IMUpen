import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheduler_model import SchedulerModel, make_model


class SchedulerModelTests(unittest.TestCase):
    def test_800_frame_clean_run_preserves_reference_frames(self):
        for pattern in range(6):
            with self.subTest(pattern=pattern):
                model = SchedulerModel(100 + pattern, frames=800, capacity_words=4096,
                                       max_irq_latency_us=100, output_queue_limit=1024,
                                       host_period_us=2000, pattern_offset=pattern)
                result = model.run()
                # Partial-pattern starts leave an unserviceable sub-watermark
                # tail because acquisition is threshold-triggered at 48 words.
                expected = 799 if pattern == 0 else 790
                self.assertEqual(result["accepted_frames"], expected)
                self.assertEqual(result["final_fifo_words"], 0 if pattern == 0 else 48 - pattern)
                self.assertEqual(result["queue_drops"], 0)
                self.assertEqual(result["fifo_recoveries"], 0)

    def test_one_thousand_seeded_interleavings_and_faults(self):
        artifact_dir = ROOT / "build" / "hardware-free"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        seed_count = int(os.environ.get("IMUPEN_SCHEDULER_SEED_COUNT", "1000"))
        for seed in range(seed_count):
            model = make_model(seed, frames=96)
            try:
                summaries.append(model.run())
            except Exception as exc:
                repro = {"seed": seed, "parameters": model.params, "error": repr(exc),
                         "trace_tail": model.trace[-400:]}
                path = artifact_dir / f"scheduler-repro-seed-{seed}.json"
                path.write_text(json.dumps(repro, indent=2), encoding="utf-8")
                self.fail(f"scheduler seed {seed} failed; replay trace saved to {path}: {exc}")
        summary = {"simulator": "scheduler_model_v1", "seed_start": 0,
                   "seed_count": len(summaries), "frames_per_seed": 96,
                   "accepted_frames_total": sum(item["accepted_frames"] for item in summaries),
                   "queue_drops_total": sum(item["queue_drops"] for item in summaries),
                   "fifo_recoveries_total": sum(item["fifo_recoveries"] for item in summaries),
                   "hardware_verified": False}
        (artifact_dir / "scheduler-seed-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.assertEqual(summary["seed_count"], seed_count)


if __name__ == "__main__":
    unittest.main()
