#!/usr/bin/env python3
"""Generate and validate a bounded 30-minute synthetic clean recording."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "hardware_free"))
import measure
from virtual_capture import generate


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output-dir", default=str(ROOT / "build" / "endurance"))
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 14_400:
        parser.error("seconds must be between 1 and 14400")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture = output / "capture.csv"
    report_dir = output / "report"
    if capture.exists() or (report_dir / "report.json").exists():
        parser.error("endurance output already exists; choose a fresh --output-dir to preserve prior artifacts")
    truth = generate(capture, seconds=args.seconds, scenario="clean", seed=args.seed,
                     irq_jitter_us=60)
    with contextlib.redirect_stdout(io.StringIO()):
        report = measure.analyze(capture, report_dir, source="virtual", test_intent="clean",
                                 nominal_hz=208.0)
    if report["quality_verdict"] != "CLEAN":
        raise SystemExit("Endurance capture quality was not CLEAN: " +
                         ", ".join(report["quality_reasons"]))
    summary = {"schema": "imupen_endurance_v1", "source_type": "synthetic",
               "synthetic": True, "seconds": args.seconds, "seed": args.seed,
               "simulator": "virtual_capture.py", "truth": truth,
               "capture_sha256": file_hash(capture),
               "report_sha256": file_hash(report_dir / "report.json"),
               "rows": report["rows"], "quality_verdict": report["quality_verdict"],
               "quality_reasons": report["quality_reasons"],
               "estimated_hz": report["segments"][0]["model"]["estimated_hz"],
               "configuration_sha256": file_hash(ROOT / "configs" / "xiao_lsm6ds3_spi_208hz.yaml"),
               "hardware_verified": False}
    (output / "endurance-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("seconds", "seed", "rows", "quality_verdict", "estimated_hz",
                       "capture_sha256", "report_sha256", "hardware_verified")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
