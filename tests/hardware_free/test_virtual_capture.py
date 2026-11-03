"""Synthetic report regressions against simulator ground truth."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from virtual_capture import generate
import measure

class VirtualReportTests(unittest.TestCase):
    def run_case(self,case,**kwargs):
        root=tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        base=Path(root.name)
        truth=generate(base/'capture.csv',seconds=12,scenario=case,**kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            measure.analyze(base/'capture.csv',base/'report',alignment=False)
        report=json.loads((base/'report'/'report.json').read_text())
        return truth,report,base

    def test_clean_recovers_208_hz_and_no_gaps(self):
        truth,report,_=self.run_case('clean')
        self.assertEqual(report['missing_host_sequence_numbers'],0)
        self.assertEqual(report['maximum_cumulative_counters_per_session']['queue_drops'],0)
        self.assertAlmostEqual(report['segments'][0]['model']['estimated_hz'],truth['true_hz'],delta=0.15)

    def test_120_second_clean_run_preserves_expected_rate_and_quality(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth = generate(root / 'capture.csv', seconds=120, scenario='clean', seed=120)
            with contextlib.redirect_stdout(io.StringIO()):
                measure.analyze(root / 'capture.csv', root / 'report', source='virtual', test_intent='clean')
            report = json.loads((root / 'report' / 'report.json').read_text())
        self.assertGreater(report['rows'], 24_000)
        self.assertEqual(report['missing_host_sequence_numbers'], 0)
        self.assertEqual(report['quality_verdict'], 'CLEAN')
        self.assertAlmostEqual(report['segments'][0]['model']['estimated_hz'], truth['true_hz'], delta=0.1)

    def test_queue_drop_is_reported_as_sequence_gap(self):
        truth,report,_=self.run_case('queue_drop')
        self.assertGreater(truth['true_missing_sequence_numbers'],0)
        self.assertEqual(report['missing_host_sequence_numbers'],truth['true_missing_sequence_numbers'])

    def test_reboot_creates_multiple_fitted_sessions(self):
        truth,report,_=self.run_case('reboot')
        self.assertEqual(truth['sessions'],2)
        self.assertEqual({x['session'] for x in report['segments']},{0,1})

    def test_fifo_recovery_separates_epochs(self):
        truth,report,_=self.run_case('fifo_reset')
        self.assertEqual(truth['epochs'],2)
        self.assertEqual({x['epoch'] for x in report['segments']},{0,1})
        self.assertEqual(report['maximum_cumulative_counters_per_session']['fifo_recoveries'],1)

    def test_nonzero_pattern_anchors_use_word_offset(self):
        truth,report,_=self.run_case('nonzero_pattern')
        self.assertEqual(report['segments'][0]['anchor_pattern_offsets'],[2])
        self.assertAlmostEqual(report['segments'][0]['model']['estimated_hz'],truth['true_hz'],delta=0.15)
        self.assertGreaterEqual(report['segments'][0]['model']['anchors'],3)

    def test_clipping_is_reported_as_failed_clean_quality(self):
        truth,report,_=self.run_case('clipping')
        self.assertGreater(truth['clipped_injected_frames'],0)
        self.assertGreater(report['quality']['clipping_count'],0)
        self.assertEqual(report['quality_verdict'],'FAILED')
        self.assertIn('CLIPPING_DETECTED',report['quality_reasons'])

    def test_irq_outliers_increase_fit_residual_spread(self):
        _,clean,_=self.run_case('clean',irq_jitter_us=0)
        _,bad,_=self.run_case('irq_outliers',irq_jitter_us=0)
        rms=lambda r: r['segments'][0]['model']['residual_us']['max']-r['segments'][0]['model']['residual_us']['min']
        self.assertGreater(rms(bad),rms(clean)+10000)

    def test_empty_csv_has_explicit_failed_quality_verdict(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            with (base/'empty.csv').open('w',newline='') as f:
                csv.writer(f).writerow(['host_elapsed_s','session']+measure.FIELDS)
            with contextlib.redirect_stdout(io.StringIO()):
                measure.analyze(base/'empty.csv',base/'out')
            report=json.loads((base/'out'/'report.json').read_text())
            self.assertEqual(report['rows'],0)
            self.assertEqual(report['quality_verdict'],'FAILED')
            self.assertIn('NO_SAMPLES',report['quality_reasons'])

if __name__=='__main__':unittest.main()
