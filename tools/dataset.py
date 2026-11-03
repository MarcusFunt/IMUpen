#!/usr/bin/env python3
"""Create canonical, traceable IMUpen datasets and optional 100 Hz products."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

ACC_G_PER_LSB = 0.000061
GYRO_DPS_PER_LSB = 0.0175
RAW_AXES = ("gx", "gy", "gz", "ax", "ay", "az")
SCALED_AXES = ("gx_dps", "gy_dps", "gz_dps", "ax_g", "ay_g", "az_g")
CAL_AXES = ("gx_cal_dps", "gy_cal_dps", "gz_cal_dps", "ax_cal_g", "ay_cal_g", "az_cal_g")
CANONICAL_FIELDS = ["time_us", "host_elapsed_s", "session", "boot_id", "epoch", "seq",
                    "batch", "segment_id", "validity_mask", *[f"{axis}_raw" for axis in RAW_AXES],
                    *SCALED_AXES, *CAL_AXES]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_calibration(calibration):
    if calibration is None:
        return None
    if not isinstance(calibration, dict) or not isinstance(calibration.get("version"), str):
        raise ValueError("calibration JSON must contain a string version")
    result = {}
    for sensor, length in (("gyro", 3), ("accel", 3)):
        bias = calibration.get(f"{sensor}_bias")
        matrix = calibration.get(f"{sensor}_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if bias is None:
            bias = [0.0] * length
        if (not isinstance(bias, list) or len(bias) != length or
                any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in bias)):
            raise ValueError(f"{sensor}_bias must contain three finite numbers")
        if (not isinstance(matrix, list) or len(matrix) != 3 or
                any(not isinstance(row, list) or len(row) != 3 or
                    any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in row)
                    for row in matrix)):
            raise ValueError(f"{sensor}_matrix must be a finite 3x3 matrix")
        result[sensor] = {"bias": [float(v) for v in bias],
                          "matrix": [[float(v) for v in row] for row in matrix]}
    result["version"] = calibration["version"]
    return result


def _apply_matrix(matrix, vector):
    return [sum(matrix[row][column] * vector[column] for column in range(3))
            for row in range(3)]


def canonicalize(capture_csv, output_csv, *, source_type, nominal_rate_hz=208.0,
                 config_path=None, calibration_path=None, writer_metadata_path=None):
    if source_type not in ("real", "synthetic"):
        raise ValueError("source_type must be explicitly real or synthetic")
    capture_csv, output_csv = Path(capture_csv), Path(output_csv)
    if not math.isfinite(nominal_rate_hz) or nominal_rate_hz <= 0:
        raise ValueError("nominal sample rate must be positive and finite")
    calibration = None
    calibration_hash = None
    if calibration_path:
        calibration_bytes = Path(calibration_path).read_bytes()
        calibration_hash = hashlib.sha256(calibration_bytes).hexdigest()
        calibration = _validate_calibration(json.loads(calibration_bytes))
    writer_metadata = None
    if writer_metadata_path:
        writer_metadata = json.loads(Path(writer_metadata_path).read_text(encoding="utf-8"))
        if not isinstance(writer_metadata, dict):
            raise ValueError("writer metadata JSON must be an object")

    rows_out = []
    invalid_rows = 0
    segment_id = -1
    prior_identity = None
    prior_seq = None
    prior_time = None
    prior_data_time = None
    timing_sources = set()
    required = ["host_elapsed_s", "session", "boot_id", "epoch", "seq", "batch", *RAW_AXES]
    with capture_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError("capture contains duplicate column names")
        missing = [field for field in required if field not in headers]
        if missing:
            raise ValueError("capture is missing required columns: " + ", ".join(missing))
        for row in reader:
            try:
                host_elapsed = float(row["host_elapsed_s"])
                if not math.isfinite(host_elapsed) or host_elapsed < 0:
                    raise ValueError("invalid host time")
                session, boot, epoch, seq, batch = (int(row[name]) for name in
                                                     ("session", "boot_id", "epoch", "seq", "batch"))
                raw = [int(row[axis]) for axis in RAW_AXES]
                if any(value < -32768 or value > 32767 for value in raw):
                    raise ValueError("raw axis outside signed int16 range")
                if any(value < 0 for value in (session, epoch, seq, batch)) or not 0 <= boot <= 0xFFFFFFFF:
                    raise ValueError("invalid session, boot, epoch, sequence, or batch")
            except (TypeError, ValueError, OverflowError):
                invalid_rows += 1
                continue

            identity = (session, boot, epoch)
            if row.get("estimated_sample_host_us") not in (None, ""):
                time_us = float(row["estimated_sample_host_us"])
                time_source = "mcu_irq_fit"
            else:
                time_us = host_elapsed * 1_000_000
                time_source = "host_receive_elapsed"
            if not math.isfinite(time_us):
                invalid_rows += 1
                continue
            contiguous = (prior_identity == identity and prior_seq is not None and
                          seq == prior_seq + 1 and prior_time is not None and host_elapsed >= prior_time and
                          prior_data_time is not None and time_us > prior_data_time)
            if not contiguous:
                segment_id += 1
            timing_sources.add(time_source)
            gyro_dps = [value * GYRO_DPS_PER_LSB for value in raw[:3]]
            accel_g = [value * ACC_G_PER_LSB for value in raw[3:]]
            if calibration:
                gyro_cal = _apply_matrix(calibration["gyro"]["matrix"],
                                         [v - b for v, b in zip(gyro_dps, calibration["gyro"]["bias"])])
                accel_cal = _apply_matrix(calibration["accel"]["matrix"],
                                          [v - b for v, b in zip(accel_g, calibration["accel"]["bias"])])
            else:
                gyro_cal = accel_cal = [None, None, None]
            output_row = {"time_us": f"{time_us:.3f}", "host_elapsed_s": f"{host_elapsed:.9f}",
                          "session": session, "boot_id": boot, "epoch": epoch, "seq": seq,
                          "batch": batch, "segment_id": segment_id, "validity_mask": 63,
                          **{f"{axis}_raw": value for axis, value in zip(RAW_AXES, raw)},
                          **{axis: value for axis, value in zip(SCALED_AXES, [*gyro_dps, *accel_g])},
                          **{axis: ("" if value is None else value)
                             for axis, value in zip(CAL_AXES, [*gyro_cal, *accel_cal])}}
            rows_out.append(output_row)
            prior_identity, prior_seq, prior_time, prior_data_time = identity, seq, host_elapsed, time_us

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(output_csv.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_csv)

    manifest_path = output_csv.with_suffix(output_csv.suffix + ".manifest.json")
    config_hash = sha256(config_path) if config_path else None
    manifest = {"dataset_schema_version": "imu_canonical_v1",
                "source_type": source_type, "synthetic": source_type == "synthetic",
                "source_capture_schema": "imupen_fifo_v1",
                "source_capture_sha256": sha256(capture_csv),
                "canonical_file": output_csv.name, "canonical_sha256": sha256(output_csv),
                "configuration_sha256": config_hash,
                "calibration": {"status": "applied" if calibration else "unknown",
                                "version": calibration["version"] if calibration else None,
                                "sha256": calibration_hash},
                "nominal_sample_rate_hz": nominal_rate_hz,
                "timing_sources": sorted(timing_sources),
                "rows": len(rows_out), "invalid_source_rows": invalid_rows,
                "segment_count": len({row["segment_id"] for row in rows_out}),
                "validity_mask": {"meaning": "bit 0..2 gyro xyz; bit 3..5 accel xyz", "all_axes_valid": 63},
                "units": {"raw": "signed int16 LSB", "gyro_scaled": "deg/s",
                          "accelerometer_scaled": "g", "time": "microseconds"},
                "source_ranges": {"raw_signed_int16": [-32768, 32767],
                                  "accelerometer_g_per_lsb": ACC_G_PER_LSB,
                                  "gyroscope_dps_per_lsb": GYRO_DPS_PER_LSB},
                "physical_verification": "not assessed by software export",
                "writer_metadata": writer_metadata}
    atomic_json(manifest_path, manifest)
    return manifest


def _sinc(x):
    return 1.0 if abs(x) < 1e-12 else math.sin(math.pi * x) / (math.pi * x)


def _lowpass_resample(times_s, values, target_rate_hz, source_rate_hz, *, half_width_samples=24):
    cutoff_hz = target_rate_hz * 0.45
    radius_s = half_width_samples / source_rate_hz
    step_s = 1.0 / target_rate_hz
    first_target = math.ceil((times_s[0] + radius_s) / step_s) * step_s
    last_target = times_s[-1] - radius_s
    if first_target > last_target:
        return []
    result = []
    target = first_target
    left = 0
    while target <= last_target + 1e-12:
        while left < len(times_s) and times_s[left] < target - radius_s:
            left += 1
        weighted = [0.0] * len(values[0])
        weight_sum = 0.0
        for index in range(left, len(times_s)):
            delta = times_s[index] - target
            if delta > radius_s:
                break
            normalized = delta / radius_s
            window = 0.42 + 0.5 * math.cos(math.pi * normalized) + 0.08 * math.cos(2 * math.pi * normalized)
            weight = (2 * cutoff_hz / source_rate_hz) * _sinc(2 * cutoff_hz * delta) * window
            weight_sum += weight
            for channel, value in enumerate(values[index]):
                weighted[channel] += weight * value
        if abs(weight_sum) > 1e-12:
            result.append((target, [value / weight_sum for value in weighted]))
        target += step_s
    return result


def resample_canonical(canonical_csv, output_csv, *, target_rate_hz=100.0, half_width_samples=24):
    canonical_csv, output_csv = Path(canonical_csv), Path(output_csv)
    if target_rate_hz <= 0 or not math.isfinite(target_rate_hz):
        raise ValueError("target_rate_hz must be positive and finite")
    if (isinstance(half_width_samples, bool) or not isinstance(half_width_samples, int) or
            half_width_samples < 1):
        raise ValueError("half_width_samples must be a positive integer")
    source_manifest_path = canonical_csv.with_suffix(canonical_csv.suffix + ".manifest.json")
    if not source_manifest_path.is_file():
        raise ValueError("canonical sidecar manifest is required before resampling")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("source_type") not in ("real", "synthetic"):
        raise ValueError("canonical manifest must explicitly label source_type as real or synthetic")
    groups = {}
    with canonical_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if int(row.get("validity_mask", "0")) != 63:
                continue
            groups.setdefault(int(row["segment_id"]), []).append(row)
    output_rows = []
    segment_rates = {}
    target_fields = ["time_us", "source_segment_id", "continuity_chunk_id",
                     "validity_mask", *SCALED_AXES]
    continuity_chunks = []
    for segment_id, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["seq"]))
        candidate_periods = [float(row["time_us"]) - float(previous["time_us"])
                             for previous, row in zip(rows, rows[1:])
                             if int(row["seq"]) == int(previous["seq"]) + 1 and
                             float(row["time_us"]) > float(previous["time_us"])]
        expected_period_us = statistics.median(candidate_periods) if candidate_periods else None
        current = []
        previous_seq = previous_time = None
        chunk_index = 0
        for row in rows:
            sequence = int(row["seq"])
            sample_time = float(row["time_us"])
            time_gap = (expected_period_us is not None and previous_time is not None and
                        sample_time - previous_time > expected_period_us * 1.5)
            if (current and (sequence != previous_seq + 1 or sample_time <= previous_time or time_gap)):
                continuity_chunks.append((f"{segment_id}:{chunk_index}", segment_id, current))
                chunk_index += 1
                current = []
            current.append(row)
            previous_seq, previous_time = sequence, sample_time
        if current:
            continuity_chunks.append((f"{segment_id}:{chunk_index}", segment_id, current))
    for continuity_id, segment_id, rows in continuity_chunks:
        times = [float(row["time_us"]) / 1_000_000 for row in rows]
        if len(times) < 3 or any(b <= a for a, b in zip(times, times[1:])):
            continue
        median_period = statistics.median(b - a for a, b in zip(times, times[1:]))
        source_rate = 1.0 / median_period
        if target_rate_hz >= source_rate / 2:
            raise ValueError(f"target rate {target_rate_hz} Hz must be below Nyquist of segment {continuity_id} ({source_rate/2:.3f} Hz)")
        values = [[float(row[axis]) for axis in SCALED_AXES] for row in rows]
        resampled = _lowpass_resample(times, values, target_rate_hz, source_rate,
                                      half_width_samples=half_width_samples)
        segment_rates[continuity_id] = source_rate
        for time_s, channels in resampled:
            output_rows.append({"time_us": f"{time_s * 1_000_000:.3f}",
                                "source_segment_id": segment_id,
                                "continuity_chunk_id": continuity_id, "validity_mask": 63,
                                **{axis: value for axis, value in zip(SCALED_AXES, channels)}})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(output_csv.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(output_rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_csv)
    manifest = {"dataset_schema_version": "imu_resampled_v1", "source_file": canonical_csv.name,
                "source_sha256": sha256(canonical_csv), "output_file": output_csv.name,
                "output_sha256": sha256(output_csv), "source_segment_rates_hz": segment_rates,
                "source_type": source_manifest["source_type"],
                "synthetic": source_manifest["source_type"] == "synthetic",
                "canonical_manifest_sha256": sha256(source_manifest_path),
                "target_rate_hz": target_rate_hz,
                "method": "windowed_sinc_blackman_lowpass", "cutoff_hz": target_rate_hz * 0.45,
                "half_width_source_samples": half_width_samples,
                "rows": len(output_rows), "segments": len(segment_rates),
                "gap_policy": "segments are filtered independently; no interpolation crosses a gap, reboot, or FIFO epoch",
                "units": {"gyro": "deg/s", "accelerometer": "g", "time": "microseconds"}}
    atomic_json(output_csv.with_suffix(output_csv.suffix + ".manifest.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    canonical = sub.add_parser("canonicalize")
    canonical.add_argument("capture_csv")
    canonical.add_argument("--output", required=True)
    canonical.add_argument("--source-type", choices=("real", "synthetic"), required=True)
    canonical.add_argument("--nominal-rate-hz", type=float, default=208.0)
    canonical.add_argument("--config")
    canonical.add_argument("--calibration-json")
    canonical.add_argument("--writer-metadata-json")
    resample = sub.add_parser("resample")
    resample.add_argument("canonical_csv")
    resample.add_argument("--output", required=True)
    resample.add_argument("--target-rate-hz", type=float, default=100.0)
    resample.add_argument("--half-width-samples", type=int, default=24)
    args = parser.parse_args(argv)
    if args.command == "canonicalize":
        result = canonicalize(args.capture_csv, args.output, source_type=args.source_type,
                              nominal_rate_hz=args.nominal_rate_hz, config_path=args.config,
                              calibration_path=args.calibration_json,
                              writer_metadata_path=args.writer_metadata_json)
    else:
        result = resample_canonical(args.canonical_csv, args.output,
                                    target_rate_hz=args.target_rate_hz,
                                    half_width_samples=args.half_width_samples)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
