"""Inspect a saved raw capture and optionally export decoded samples as CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from smartpen.protocol import Config, DeviceError, FrameParser, Sample, SequenceTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a SmartPen .bin recording")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--csv", type=Path, help="Export valid samples as CSV")
    args = parser.parse_args()
    stream = FrameParser()
    frames = stream.feed(args.recording.read_bytes())
    samples = [f.message for f in frames if isinstance(f.message, Sample)]
    configs = [f.message for f in frames if isinstance(f.message, Config)]
    errors = [f.message for f in frames if isinstance(f.message, DeviceError)]
    tracker = SequenceTracker()
    for sample in samples:
        tracker.add(sample)
    print(f"Samples: {len(samples)}, config frames: {len(configs)}, error frames: {len(errors)}")
    print(f"CRC errors: {stream.crc_errors}, invalid headers: {stream.header_errors}")
    print(f"Sequence diagnostics: {tracker.summary()}")
    if configs:
        print(f"Last device config: {configs[-1]}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["sequence", "timestamp_us", "ax", "ay", "az",
                             "gx", "gy", "gz", "flags"])
            for s in samples:
                writer.writerow([s.sequence, s.timestamp_us, *s.axes, s.flags])
        print(f"Exported: {args.csv}")


if __name__ == "__main__":
    main()
