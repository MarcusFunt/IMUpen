#!/usr/bin/env python3
"""Record IMUpen FIFO output and summarize timing and loss indicators."""
import argparse
from array import array
from collections import deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

FIELDS = "boot_id who_am_i seq epoch batch irq_count irq_us read_start_us read_end_us fifo_words pattern word_end gx gy gz ax ay az queue_drops fifo_recoveries overrun_observations".split()


def percentiles(values):
    values = sorted(values)
    if not values:
        return None
    def at(p):
        return values[round((len(values) - 1) * p)]
    return {"count": len(values), "min": values[0], "median": at(.5),
            "p95": at(.95), "p99": at(.99), "max": values[-1]}


def capture(args, serial_module=None):
    """Capture a device stream and always publish available partial artifacts."""
    if serial_module is None:
        try:
            import serial as serial_module
        except ImportError:
            raise SystemExit("Install pyserial: python -m pip install -r requirements.txt")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw_path, events_path = out / "capture.csv", out / "events.jsonl"
    protected_outputs = (raw_path, events_path, out / "report.json", out / "reconstructed.csv")
    if any(path.exists() for path in protected_outputs):
        raise SystemExit("Output already exists; choose a new output directory.")
    duration = float(args.duration)
    pause_at, pause_seconds = getattr(args, "pause_at", None), getattr(args, "pause_seconds", 10)
    disconnect_at = getattr(args, "disconnect_at", None)
    disconnect_seconds = getattr(args, "disconnect_seconds", 10)
    startup_timeout = float(getattr(args, "startup_timeout_s", 10.0))
    silence_timeout = float(getattr(args, "silence_timeout_s", 2.0))
    minimum_rows = int(getattr(args, "minimum_valid_rows", 1))
    max_recoveries = int(getattr(args, "max_unplanned_recoveries", 0))
    max_malformed = int(getattr(args, "max_malformed_lines", 0))
    max_line_bytes = int(getattr(args, "max_line_bytes", 4096))
    max_output_bytes = int(getattr(args, "max_output_bytes", 2 * 1024 ** 3))
    minimum_free_bytes = int(getattr(args, "minimum_free_bytes", 64 * 1024 ** 2))
    nominal_hz = float(getattr(args, "nominal_odr_hz", 208.0))
    clip_threshold = float(getattr(args, "clip_threshold", 0.98))
    if not math.isfinite(duration) or duration <= 0:
        raise SystemExit("Capture duration must be positive and finite.")
    if (not math.isfinite(startup_timeout) or startup_timeout <= 0 or
            not math.isfinite(silence_timeout) or silence_timeout <= 0 or
            minimum_rows < 1 or max_recoveries < 0 or max_malformed < 0):
        raise SystemExit("Capture health thresholds must be positive and finite (limits may be zero).")
    if (not math.isfinite(nominal_hz) or nominal_hz <= 0 or
            not math.isfinite(clip_threshold) or not 0 < clip_threshold <= 1):
        raise SystemExit("Nominal ODR must be positive and clipping threshold must be in (0, 1].")
    for name, at, interval in (("pause", pause_at, pause_seconds),
                               ("disconnect", disconnect_at, disconnect_seconds)):
        if at is not None and (not math.isfinite(at) or not math.isfinite(interval) or
                               at < 0 or interval <= 0 or at + interval >= duration):
            raise SystemExit(f"{name} must start at or after zero and finish before duration.")
    if pause_at is not None and disconnect_at is not None:
        if max(pause_at, disconnect_at) < min(pause_at + pause_seconds, disconnect_at + disconnect_seconds):
            raise SystemExit("Pause and disconnect intervals cannot overlap.")
    if max_line_bytes < 128 or max_output_bytes <= 0 or minimum_free_bytes < 0:
        raise SystemExit("Capture resource limits are invalid.")

    start = time.monotonic()
    session, previous_seq, previous_boot, port = 0, None, None, None
    pause_done = disconnect_done = False
    logic_process, logic_log, logic_attempted = None, None, False
    sigrok_raw = out / "sigrok_samples.csv"
    logic_path = getattr(args, "logic_csv", None)
    source = getattr(args, "source", "actual_device")
    test_intent = getattr(args, "test_intent", "clean")
    allow_expected = bool(getattr(args, "allow_expected_faults", False))
    failure_reasons, malformed_lines, unplanned_recoveries = set(), 0, 0
    rows, first_sample_elapsed, last_sample_elapsed = 0, None, None
    state = "INITIALIZING"
    capture_health = {}
    partial_line = bytearray()
    complete_lines = deque()

    def active_interval(start_s, end_s):
        ignored = 0.0
        for interval_start, length in ((pause_at, pause_seconds), (disconnect_at, disconnect_seconds)):
            if interval_start is not None:
                ignored += max(0.0, min(end_s, interval_start + length) - max(start_s, interval_start))
        return max(0.0, end_s - start_s - ignored)

    with raw_path.open("w", newline="", encoding="utf-8") as raw, events_path.open("w", encoding="utf-8") as events:
        writer = csv.writer(raw)
        writer.writerow(["host_elapsed_s", "session"] + FIELDS)

        def event(kind, **details):
            entry = {"elapsed_s": round(time.monotonic() - start, 6), "event": kind, **details}
            payload = json.dumps(entry, separators=(",", ":")) + "\n"
            if raw.tell() + events.tell() + len(payload.encode("utf-8")) > max_output_bytes:
                failure_reasons.add("OUTPUT_SIZE_LIMIT")
                return
            events.write(payload)
            events.flush()
            print(json.dumps(entry), flush=True)

        def transition(next_state, **details):
            nonlocal state
            if next_state != state:
                event("capture_state", previous=state, state=next_state, **details)
                state = next_state

        def open_port():
            device = serial_module.Serial(port=None, baudrate=args.baud, timeout=0.2)
            device.dtr = False
            device.rts = False
            device.port = args.port
            device.open()
            return device

        def close_port():
            nonlocal port
            if port is not None:
                try:
                    port.close()
                except (serial_module.SerialException, OSError):
                    pass
                port = None

        def resource_failure():
            try:
                usage = shutil.disk_usage(out)
            except OSError:
                return "OUTPUT_DIRECTORY_UNAVAILABLE"
            if usage.free < minimum_free_bytes:
                return "INSUFFICIENT_DISK_SPACE"
            if raw.tell() + events.tell() >= max_output_bytes:
                return "OUTPUT_SIZE_LIMIT"
            return None

        event("start", port=args.port, duration_s=duration, source=source, test_intent=test_intent)
        try:
            usage_error = resource_failure()
            if usage_error:
                failure_reasons.add(usage_error)
                event("capture_failure", reason=usage_error)
                transition("FAILED")
            else:
                while time.monotonic() - start < duration:
                    if raw.tell() + events.tell() >= max_output_bytes:
                        failure_reasons.add("OUTPUT_SIZE_LIMIT")
                        transition("FAILED")
                        break
                    elapsed = time.monotonic() - start
                    if pause_at is not None and not pause_done and elapsed >= pause_at:
                        pause_done = True
                        transition("DEGRADED", expected_fault=True)
                        event("read_pause_start", seconds=pause_seconds, planned=True)
                        time.sleep(pause_seconds)
                        event("read_pause_end", planned=True)
                    if disconnect_at is not None and not disconnect_done and elapsed >= disconnect_at:
                        disconnect_done = True
                        close_port()
                        transition("DEGRADED", expected_fault=True)
                        event("port_closed", seconds=disconnect_seconds, planned=True)
                        time.sleep(disconnect_seconds)
                        event("port_reopen_attempt", planned=True)
                    elapsed = time.monotonic() - start
                    if port is None:
                        try:
                            port = open_port()
                            event("port_open")
                        except serial_module.SerialException as exc:
                            unplanned_recoveries += 1
                            event("port_open_error", error=str(exc)[:300], planned=False)
                            if unplanned_recoveries > max_recoveries:
                                failure_reasons.add("PORT_OPEN_FAILED")
                                event("capture_failure", reason="PORT_OPEN_FAILED")
                                transition("FAILED")
                                break
                            time.sleep(min(1.0, max(0.0, duration - elapsed)))
                            continue
                    try:
                        if complete_lines:
                            raw_line = complete_lines.popleft()
                        else:
                            # pyserial's read_until supports a hard byte bound; retain
                            # readline only for lightweight injected transports.
                            bounded_reader = getattr(port, "read_until", None)
                            raw_line = (bounded_reader(expected=b"\n", size=max_line_bytes + 1)
                                        if bounded_reader else port.readline())
                    except (serial_module.SerialException, OSError) as exc:
                        unplanned_recoveries += 1
                        event("port_error", error=str(exc)[:300], planned=False)
                        close_port()
                        if unplanned_recoveries > max_recoveries:
                            failure_reasons.add("SERIAL_READ_FAILED")
                            event("capture_failure", reason="SERIAL_READ_FAILED")
                            transition("FAILED")
                            break
                        time.sleep(min(1.0, max(0.0, duration - elapsed)))
                        continue
                    now = time.monotonic()
                    elapsed = now - start
                    if not raw_line:
                        if rows == 0 and active_interval(0.0, elapsed) >= startup_timeout:
                            failure_reasons.add("NO_SAMPLES")
                            failure_reasons.add("STARTUP_TIMEOUT")
                            event("startup_timeout", timeout_s=startup_timeout)
                            transition("FAILED")
                            break
                        if rows and last_sample_elapsed is not None and active_interval(last_sample_elapsed, elapsed) >= silence_timeout:
                            failure_reasons.add("SILENT_STREAM")
                            event("silence_timeout", timeout_s=silence_timeout)
                            transition("FAILED")
                            break
                        continue
                    partial_line.extend(raw_line)
                    while (newline_index := partial_line.find(b"\n")) >= 0:
                        complete_lines.append(bytes(partial_line[:newline_index + 1]))
                        del partial_line[:newline_index + 1]
                    if len(partial_line) > max_line_bytes:
                        malformed_lines += 1
                        event("malformed_line", reason="LINE_TOO_LONG", byte_count=len(partial_line), text=bytes(partial_line[:300]).decode("ascii", errors="replace"), planned=False)
                        partial_line.clear()
                        if malformed_lines > max_malformed:
                            failure_reasons.add("INVALID_ROW_FORMAT")
                            transition("FAILED")
                            break
                        continue
                    if not complete_lines:
                        continue
                    raw_line = complete_lines.popleft()
                    if len(raw_line) > max_line_bytes + 1:
                        malformed_lines += 1
                        event("malformed_line", reason="LINE_TOO_LONG", byte_count=len(raw_line), text=raw_line[:300].decode("ascii", errors="replace"), planned=False)
                        if malformed_lines > max_malformed:
                            failure_reasons.add("INVALID_ROW_FORMAT")
                            transition("FAILED")
                            break
                        continue
                    line = raw_line.decode("ascii", errors="replace").strip()
                    if not line:
                        continue
                    if "\ufffd" in line:
                        malformed_lines += 1
                        event("malformed_line", reason="NON_ASCII", text=line[:300], planned=False)
                        if malformed_lines > max_malformed:
                            failure_reasons.add("INVALID_ROW_FORMAT")
                            transition("FAILED")
                            break
                        continue
                    if line.startswith("#"):
                        event("device_message", text=line[:300])
                        if line.startswith("# ERROR:"):
                            failure_reasons.add("DEVICE_FATAL_ERROR")
                            event("capture_failure", reason="DEVICE_FATAL_ERROR")
                            transition("FAILED")
                            break
                        if line.startswith("# WHO_AM_I=") and rows:
                            session += 1
                            previous_seq = previous_boot = None
                            event("device_boot_detected", session=session)
                        continue
                    values = line.split(",")
                    try:
                        if len(values) != len(FIELDS):
                            raise ValueError("wrong field count")
                        numbers = [int(value) for value in values]
                        row_values = dict(zip(FIELDS, numbers))
                        _parse_sample({**row_values, "host_elapsed_s": elapsed, "session": session})
                    except (ValueError, TypeError, OverflowError) as exc:
                        malformed_lines += 1
                        event("malformed_line", reason=str(exc)[:100], text=line[:300], planned=False)
                        if malformed_lines > max_malformed:
                            failure_reasons.add("INVALID_ROW_FORMAT")
                            transition("FAILED")
                            break
                        continue
                    boot, seq = numbers[0], numbers[2]
                    if previous_boot is not None and boot != previous_boot:
                        session += 1
                        event("boot_id_changed", previous=previous_boot, current=boot, planned=False)
                        previous_seq = None
                    if previous_seq is not None and seq < previous_seq:
                        session += 1
                        event("sequence_restart", previous=previous_seq, current=seq, planned=False)
                        previous_seq = None
                    elif previous_seq is not None and seq == previous_seq:
                        event("duplicate_sequence", seq=seq, planned=False)
                        failure_reasons.add("DUPLICATE_SEQUENCE")
                        transition("FAILED")
                        break
                    previous_seq, previous_boot = seq, boot
                    elapsed_text = f"{elapsed:.6f}"
                    csv_row = [elapsed_text, session] + numbers
                    projected = len((",".join(str(value) for value in csv_row) + "\r\n").encode("utf-8"))
                    if raw.tell() + events.tell() + projected > max_output_bytes:
                        failure_reasons.add("OUTPUT_SIZE_LIMIT")
                        event("capture_failure", reason="OUTPUT_SIZE_LIMIT")
                        transition("FAILED")
                        break
                    writer.writerow(csv_row)
                    raw.flush()
                    rows += 1
                    first_sample_elapsed = elapsed if first_sample_elapsed is None else first_sample_elapsed
                    last_sample_elapsed = elapsed
                    if rows == 1:
                        event("first_valid_sample", elapsed_s=round(elapsed, 6))
                        transition("ACQUIRING")
                    if getattr(args, "sigrok_driver", None) and not logic_attempted:
                        logic_attempted = True
                        logic_log = (out / "sigrok.log").open("w", encoding="utf-8")
                        command = ["sigrok-cli", "--driver", args.sigrok_driver,
                                   "--config", f"samplerate={args.logic_rate}",
                                   "--channels", f"{args.irq_channel}=INT1,{args.cs_channel}=CS",
                                   "--samples", str(round(args.logic_seconds * args.logic_rate)),
                                   "--output-format", "csv:header=false:label=channel:time=false:dedup=false",
                                   "--output-file", str(sigrok_raw)]
                        try:
                            logic_process = subprocess.Popen(command, stdout=logic_log, stderr=logic_log)
                            event("logic_capture_started", command=command)
                        except OSError as exc:
                            event("logic_capture_error", error=str(exc)[:300])
        except KeyboardInterrupt:
            failure_reasons.add("CAPTURE_INTERRUPTED")
            event("capture_interrupted")
            transition("FAILED")
        finally:
            if partial_line:
                malformed_lines += 1
                event("malformed_line", reason="TRUNCATED_LINE", byte_count=len(partial_line),
                      text=bytes(partial_line[:300]).decode("ascii", errors="replace"), planned=False)
                failure_reasons.add("INVALID_ROW_FORMAT")
            close_port()
            if logic_process:
                try:
                    code = logic_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    logic_process.terminate()
                    code = logic_process.wait(timeout=5)
                    event("logic_capture_timeout")
                if code == 0:
                    try:
                        logic_path = str(out / "logic_edges.csv")
                        convert_sigrok(sigrok_raw, logic_path, args.logic_rate)
                        event("logic_capture_complete", output=logic_path)
                    except (ValueError, OSError) as exc:
                        logic_path = None
                        event("logic_capture_error", error=str(exc)[:300])
                else:
                    event("logic_capture_error", returncode=code, log="sigrok.log")
            if logic_log:
                logic_log.close()
            if state != "FAILED":
                transition("COMPLETED")
            event("stop", state=state, rows=rows)
            capture_health = {"state": state, "valid_rows": rows,
                              "first_sample_elapsed_s": first_sample_elapsed,
                              "last_sample_elapsed_s": last_sample_elapsed,
                              "malformed_lines": malformed_lines,
                              "unplanned_recoveries": unplanned_recoveries,
                              "policy": {"startup_timeout_s": startup_timeout,
                                         "silence_timeout_s": silence_timeout,
                                         "minimum_valid_rows": minimum_rows,
                                         "max_unplanned_recoveries": max_recoveries,
                                         "max_malformed_lines": max_malformed},
                              "failure_reasons": sorted(failure_reasons)}
    return analyze(raw_path, out, logic_path, getattr(args, "alignment", False),
                   getattr(args, "alignment_tolerance_ms", None), events_jsonl=events_path,
                   test_intent=test_intent, allow_expected_faults=allow_expected,
                   source=source, nominal_hz=getattr(args, "nominal_odr_hz", 208.0),
                   clip_threshold=getattr(args, "clip_threshold", 0.98),
                   confirmed_tr_c=getattr(args, "confirmed_tr_c", False),
                   config_path=getattr(args, "config", None),
                   minimum_valid_rows=minimum_rows, capture_health=capture_health,
                   capture_failure_reasons=failure_reasons)


def fit_line(points):
    if len(points) < 3:
        return None
    xbar = statistics.fmean(x for x, _ in points)
    ybar = statistics.fmean(y for _, y in points)
    denom = sum((x - xbar) ** 2 for x, _ in points)
    if not denom:
        return None
    slope = sum((x - xbar) * (y - ybar) for x, y in points) / denom
    if slope <= 0:
        return None
    residual = [y - (ybar + slope * (x - xbar)) for x, y in points]
    return {"points": len(points), "slope_us_per_sample": slope, "estimated_hz": 1e6 / slope,
            "xbar": xbar, "ybar": ybar, "residual_us": percentiles(residual)}


def fit_robust_line(points, nominal_hz=208.0):
    """Fit sample index to MCU time, down-weighting isolated delayed IRQs."""
    points = sorted((float(x), float(y)) for x, y in points)
    if len(points) < 3:
        return None
    adjacent_slopes = [(y1 - y0) / (x1 - x0)
                       for (x0, y0), (x1, y1) in zip(points, points[1:]) if x1 > x0]
    if not adjacent_slopes:
        return None
    slope = statistics.median(adjacent_slopes)
    intercept = statistics.median(y - slope * x for x, y in points)
    weights = [1.0] * len(points)
    for _ in range(20):
        xbar = sum(w * x for (x, _), w in zip(points, weights)) / sum(weights)
        ybar = sum(w * y for (_, y), w in zip(points, weights)) / sum(weights)
        denom = sum(w * (x - xbar) ** 2 for (x, _), w in zip(points, weights))
        if denom <= 0:
            return None
        slope = sum(w * (x - xbar) * (y - ybar)
                    for (x, y), w in zip(points, weights)) / denom
        if slope <= 0:
            return None
        intercept = ybar - slope * xbar
        residual = [y - (intercept + slope * x) for x, y in points]
        center = statistics.median(residual)
        mad = statistics.median(abs(value - center) for value in residual)
        cutoff = max(1.0, 1.345 * 1.4826 * mad)
        new_weights = [min(1.0, cutoff / max(abs(value - center), 1e-12))
                       for value in residual]
        if max(abs(a - b) for a, b in zip(weights, new_weights)) < 1e-5:
            weights = new_weights
            break
        weights = new_weights

    residual = [y - (intercept + slope * x) for x, y in points]
    absolute = [abs(value) for value in residual]
    period = 1e6 / slope
    effective_hz = 1e6 / slope
    if not 1.0 <= effective_hz <= 1000.0:
        return None
    p95 = percentiles(absolute)["p95"]
    confidence = ("HIGH" if len(points) >= 8 and p95 <= period * 0.25 else
                  "MEDIUM" if p95 <= period else "LOW")
    xbar = statistics.fmean(x for x, _ in points)
    ybar = statistics.fmean(y for _, y in points)
    return {"method": "huber_irls", "points": len(points), "anchors": len(points),
            "slope_us_per_sample": slope, "estimated_hz": effective_hz,
            "nominal_drift_ppm": (effective_hz / nominal_hz - 1) * 1e6,
            "xbar": xbar, "ybar": ybar,
            "fit_sample_interval": [min(x for x, _ in points), max(x for x, _ in points)],
            "residual_us": percentiles(residual),
            "absolute_residual_us": percentiles(absolute),
            "anchors_downweighted": sum(weight < 0.5 for weight in weights),
            "timing_confidence": confidence}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_event_log(path):
    counts, planned_counts, unplanned_counts, issues, planned = {}, {}, {}, [], 0
    if path is None or not Path(path).is_file():
        return {"available": False, "counts": counts, "planned_counts": planned_counts,
                "unplanned_counts": unplanned_counts, "planned_fault_events": 0,
                "unplanned_recoveries": 0}, issues
    unplanned = 0
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if len(line) > 65536:
                    issues.append("INVALID_EVENT_LOG")
                    continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    issues.append("INVALID_EVENT_LOG")
                    continue
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    issues.append("INVALID_EVENT_LOG")
                    continue
                name = event["event"]
                counts[name] = counts.get(name, 0) + 1
                if event.get("planned") is True:
                    planned += 1
                    planned_counts[name] = planned_counts.get(name, 0) + 1
                else:
                    unplanned_counts[name] = unplanned_counts.get(name, 0) + 1
                if name in ("port_error", "port_open_error") and event.get("planned") is not True:
                    unplanned += 1
                if name == "device_message" and str(event.get("text", "")).startswith("# ERROR:"):
                    issues.append("DEVICE_FATAL_ERROR")
                if name == "startup_timeout":
                    issues.append("STARTUP_TIMEOUT")
                if name == "silence_timeout":
                    issues.append("SILENT_STREAM")
                if name == "capture_failure" and isinstance(event.get("reason"), str):
                    issues.append(event["reason"])
    except OSError:
        issues.append("INVALID_EVENT_LOG")
    if unplanned_counts.get("malformed_line", 0):
        issues.append("MALFORMED_LINES")
    if unplanned:
        issues.append("UNPLANNED_TRANSPORT_ERRORS")
    if counts.get("fifo_recovery", 0) or counts.get("fifo_recoveries", 0):
        issues.append("FIFO_RECOVERY_OBSERVED")
    if counts.get("overrun", 0) or counts.get("overrun_observations", 0):
        issues.append("FIFO_OVERRUN_OBSERVED")
    return {"available": True, "counts": counts, "planned_counts": planned_counts,
            "unplanned_counts": unplanned_counts, "planned_fault_events": planned,
            "unplanned_recoveries": unplanned}, issues


def convert_sigrok(source, output, sample_rate):
    # Deduplication and time columns are disabled; each row is one clock tick.
    with open(source, newline="", encoding="utf-8-sig") as src, open(output, "w", newline="", encoding="utf-8") as dst:
        reader, writer = csv.reader(src), csv.writer(dst)
        labels = next(reader, [])
        if set(labels) != {"INT1", "CS"} or len(labels) != 2:
            raise ValueError(f"Unexpected sigrok channel labels: {labels!r}")
        irq_index, cs_index = labels.index("INT1"), labels.index("CS")
        writer.writerow(["time_s", "int1", "cs"])
        previous, count = None, 0
        for count, row in enumerate(reader, start=1):
            if len(row) != 2 or any(x not in ("0", "1") for x in row):
                raise ValueError(f"Invalid sigrok sample row {count}: {row!r}")
            values = (row[irq_index], row[cs_index])
            if values != previous:
                writer.writerow([f"{(count - 1) / sample_rate:.9f}", *values])
                previous = values
        if count == 0:
            raise ValueError("No sigrok samples recorded.")


def logic_metrics(path):
    # Level-sampled/transition CSV on one analyzer clock: time_s,int1,cs.
    delays, missed = [], 0
    previous_int, previous_cs, pending, last_time = None, None, None, None
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            t, irq, cs = float(row["time_s"]), int(row["int1"]), int(row["cs"])
            if not math.isfinite(t) or irq not in (0, 1) or cs not in (0, 1):
                raise ValueError("Logic CSV requires finite time_s and 0/1 int1,cs values.")
            if last_time is not None and t < last_time:
                raise ValueError("Logic CSV times must be in ascending order.")
            if previous_int == 0 and irq == 1:
                if pending is not None:
                    missed += 1
                if cs == 0:
                    pending = None
                    missed += 1
                else:
                    pending = t
            if previous_cs == 1 and cs == 0 and pending is not None:
                delays.append((t - pending) * 1e6)
                pending = None
            if previous_int == 1 and irq == 0 and pending is not None:
                missed += 1
                pending = None
            previous_int, previous_cs, last_time = irq, cs, t
    if pending is not None:
        missed += 1
    return {"irq_to_next_cs_falling_us": percentiles(delays), "unmatched_irq_edges": missed,
            "note": "CS edges must belong to this IMU; a CS already low at the IRQ is not measured."}


def alignment_metrics(chunks, models, tolerance_ms):
    # For rotation about a fixed sensor, d(accel)/dt ~= -omega cross accel.
    # This comparison is invalid when translation or vibration is significant.
    results = []
    for key, spans in chunks.items():
        model = models.get(key)
        if not model:
            continue
        hz = model["estimated_hz"]
        if not 150 <= hz <= 260:
            continue
        max_lag = max(1, round(0.04 * hz))
        scores = []
        for lag in range(-max_lag, max_lag + 1):
            n = 0
            sx = sy = sxx = syy = sxy = 0.0
            for span in spans:
                for i in range(max(1, -lag), min(len(span) - 1, len(span) - lag)):
                    a0, a, a1 = span[i-1][1], span[i][1], span[i+1][1]
                    g = span[i+lag][0]
                    derivative = [(a1[j] - a0[j]) * hz / 2 for j in range(3)]
                    predicted = [a[1]*g[2] - a[2]*g[1], a[2]*g[0] - a[0]*g[2], a[0]*g[1] - a[1]*g[0]]
                    for x, y in zip(derivative, predicted):
                        n += 1; sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y
            denom = (sxx - sx*sx/max(n, 1)) * (syy - sy*sy/max(n, 1))
            score = (sxy - sx*sy/n) / math.sqrt(denom) if n >= 300 and denom > 1e-15 else None
            if score is not None:
                scores.append((score, lag, n // 3))
        if not scores:
            continue
        score, lag, n = max(scores)
        lag_ms = lag * 1000 / hz
        # Integer lag resolution and model assumptions prevent this from being
        # a general proof of physical synchronization.
        quality = score >= 0.5 and abs(lag) < max_lag
        entry = {"session": key[0], "epoch": key[-1], "sample_pairs": n,
                        "best_gyro_offset_samples": lag, "best_gyro_offset_ms": lag_ms,
                        "correlation": score, "resolution_ms": 1000/hz,
                        "peak_at_search_boundary": abs(lag) == max_lag,
                        "user_tolerance_ms": tolerance_ms,
                        "within_tolerance_if_assumptions_hold": (abs(lag_ms) <= tolerance_ms if tolerance_ms is not None and quality else None),
                        "note": "Compare only a controlled fixed-position rotation. Positive offset compares a later gyro row to the accel derivative. Translation, vibration, filter delay, and periodic ambiguity can invalidate this estimate."}
        if len(key) >= 4:
            entry["boot_id"] = key[1]
            entry["session_instance"] = key[2]
        results.append(entry)
    return results


def _weighted_line(points, weights):
    total = sum(weights)
    if total <= 0:
        return None
    xbar = sum(w * x for (x, _), w in zip(points, weights)) / total
    ybar = sum(w * y for (_, y), w in zip(points, weights)) / total
    denom = sum(w * (x - xbar) ** 2 for (x, _), w in zip(points, weights))
    if denom <= 0:
        return None
    slope = sum(w * (x - xbar) * (y - ybar)
                for (x, y), w in zip(points, weights)) / denom
    return slope, ybar - slope * xbar, xbar, ybar


def _parse_sample(row):
    if None in row or any(row.get(name) in (None, "") for name in ("host_elapsed_s", "session", *FIELDS)):
        raise ValueError("missing or extra CSV field")
    host_s = float(row["host_elapsed_s"])
    if not math.isfinite(host_s) or host_s < 0:
        raise ValueError("invalid host_elapsed_s")
    values = {name: int(row[name]) for name in FIELDS}
    session = int(row["session"])
    if session < 0:
        raise ValueError("negative session")
    for name in ("seq", "epoch", "batch", "irq_count", "read_start_us", "read_end_us",
                 "fifo_words", "queue_drops", "fifo_recoveries", "overrun_observations"):
        if values[name] < 0:
            raise ValueError(f"negative {name}")
    if values["irq_us"] < -1 or values["read_end_us"] < values["read_start_us"]:
        raise ValueError("invalid MCU timestamps")
    if not 0 <= values["fifo_words"] <= 4095:
        raise ValueError("fifo_words outside documented register range")
    pattern, word_end = values["pattern"], values["word_end"]
    if not 0 <= pattern <= 5 or not 0 <= word_end < 48:
        raise ValueError("invalid FIFO pattern or word index")
    if word_end % 6 != (5 - pattern) % 6:
        raise ValueError("word_end does not complete the reported FIFO pattern")
    if not 0 <= values["who_am_i"] <= 255 or not 0 <= values["boot_id"] <= 0xFFFFFFFF:
        raise ValueError("invalid device identity field")
    for name in ("gx", "gy", "gz", "ax", "ay", "az"):
        if not -32768 <= values[name] <= 32767:
            raise ValueError(f"{name} outside signed int16 range")
    return host_s, session, values


def _histogram_quantile(histogram, count, probability):
    target = round((count - 1) * probability)
    cumulative = 0
    for index, frequency in enumerate(histogram):
        cumulative += frequency
        if cumulative > target:
            return index - 32768
    return 32767


def _axis_statistics(histograms, counts, minima, maxima):
    result = {}
    for axis, histogram in histograms.items():
        count = counts[axis]
        result[axis] = ({"count": count, "min": minima[axis], "max": maxima[axis],
                         "p50": _histogram_quantile(histogram, count, .50),
                         "p95": _histogram_quantile(histogram, count, .95),
                         "p99": _histogram_quantile(histogram, count, .99)} if count else
                        {"count": 0, "min": None, "max": None, "p50": None, "p95": None, "p99": None})
    return result


def analyze(path, output, logic_csv=None, alignment=False, alignment_tolerance_ms=None, *,
            events_jsonl=None, strict=False, test_intent="clean", allow_expected_faults=False,
            source="unknown", nominal_hz=208.0, clip_threshold=0.98,
            confirmed_tr_c=False, config_path=None, minimum_valid_rows=1,
            capture_health=None, capture_failure_reasons=()):
    """Validate a raw capture, estimate timing, and publish an atomic report."""
    if not math.isfinite(nominal_hz) or nominal_hz <= 0:
        raise ValueError("nominal_hz must be positive and finite")
    if not math.isfinite(clip_threshold) or not 0 < clip_threshold <= 1:
        raise ValueError("clip_threshold must be finite and in (0, 1]")
    if (alignment_tolerance_ms is not None and
            (not math.isfinite(alignment_tolerance_ms) or alignment_tolerance_ms < 0)):
        raise ValueError("alignment_tolerance_ms must be nonnegative and finite")
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    required = ["host_elapsed_s", "session"] + FIELDS
    reasons, invalid_counts = set(), {}
    reasons.update(capture_failure_reasons)
    rows_seen = rows = gaps = invalid_rows = 0
    missing_values = {name: 0 for name in required}
    latency, duration, irq_intervals = [], [], []
    axes = ("gx", "gy", "gz", "ax", "ay", "az")
    histograms = {axis: array("I", [0]) * 65536 for axis in axes}
    axis_counts = {axis: 0 for axis in axes}
    axis_min = {axis: None for axis in axes}
    axis_max = {axis: None for axis in axes}
    clipping = {axis: 0 for axis in axes}
    max_abs_raw = round(32767 * clip_threshold)
    previous_frame, repeated_frames = {}, 0
    stationary_data = {}
    previous_seq, run_index, counter_data = {}, {}, {}
    last_batch_irq = {}
    batch_metadata = {}
    last_host_time, last_boot_by_host_session = {}, {}
    segments, anchor_candidates, chunks = {}, {}, {}

    def invalidate(code):
        nonlocal invalid_rows
        invalid_rows += 1
        invalid_counts[code] = invalid_counts.get(code, 0) + 1
        reasons.add("INVALID_CAPTURE_ROWS")

    try:
        source_stream = path.open(newline="", encoding="utf-8-sig")
    except OSError:
        source_stream = None
        reasons.add("CAPTURE_FILE_UNREADABLE")
    headers = []
    if source_stream is not None:
        with source_stream:
            reader = csv.DictReader(source_stream)
            headers = reader.fieldnames or []
            missing = [name for name in required if name not in headers]
            if len(headers) != len(set(headers)):
                reasons.add("DUPLICATE_COLUMNS")
            if missing:
                reasons.add("MISSING_REQUIRED_COLUMNS")
            elif "DUPLICATE_COLUMNS" not in reasons:
                for row_number, row in enumerate(reader, start=2):
                    rows_seen += 1
                    for name in required:
                        if row.get(name) in (None, ""):
                            missing_values[name] += 1
                    try:
                        host_s, host_session, value = _parse_sample(row)
                    except (TypeError, ValueError, OverflowError) as exc:
                        code = "INVALID_WORD_PATTERN" if "pattern" in str(exc) or "word index" in str(exc) else "INVALID_ROW_VALUE"
                        invalidate(code)
                        continue

                    if host_session in last_host_time and host_s < last_host_time[host_session]:
                        invalidate("HOST_TIME_NONMONOTONIC")
                        continue
                    last_host_time[host_session] = host_s
                    boot = value["boot_id"]
                    previous_boot = last_boot_by_host_session.get(host_session)
                    if previous_boot is not None and previous_boot != boot:
                        reasons.add("DEVICE_BOOT_CHANGED")
                    last_boot_by_host_session[host_session] = boot
                    base_key = (host_session, boot)
                    seq = value["seq"]
                    prior_seq = previous_seq.get(base_key)
                    current_run = run_index.get(base_key, 0)
                    if prior_seq is not None and seq < prior_seq:
                        current_run += 1
                        run_index[base_key] = current_run
                        reasons.add("SEQUENCE_RESTART")
                        prior_seq = None
                    elif prior_seq is not None and seq == prior_seq:
                        invalidate("DUPLICATE_SEQUENCE")
                        continue
                    sequence_gap = prior_seq is not None and seq > prior_seq + 1
                    if sequence_gap:
                        gaps += seq - prior_seq - 1
                        reasons.add("HOST_SEQUENCE_GAP")
                    previous_seq[base_key] = seq
                    session_key = (host_session, boot, current_run)
                    epoch = value["epoch"]
                    segment_key = session_key + (epoch,)
                    rows += 1

                    batch_state = last_batch_irq.get(session_key)
                    batch, irq_count = value["batch"], value["irq_count"]
                    if batch_state is not None:
                        old_batch, old_irq = batch_state
                        if batch < old_batch:
                            reasons.add("BATCH_INDEX_NONMONOTONIC")
                        elif batch > old_batch + 1:
                            reasons.add("BATCH_INDEX_GAP")
                        if irq_count < old_irq:
                            reasons.add("IRQ_COUNT_NONMONOTONIC")
                        if batch == old_batch and irq_count != old_irq:
                            reasons.add("INCONSISTENT_BATCH_INTERRUPT_COUNT")
                    last_batch_irq[session_key] = (batch, irq_count)
                    batch_key = (session_key, batch)
                    batch_signature = (irq_count, value["irq_us"], value["pattern"], value["fifo_words"])
                    previous_signature = batch_metadata.get(batch_key)
                    if previous_signature is not None and previous_signature != batch_signature:
                        reasons.add("INCONSISTENT_BATCH_METADATA")
                    batch_metadata[batch_key] = batch_signature

                    if value["who_am_i"] == 0x6A and not confirmed_tr_c:
                        reasons.add("UNCONFIRMED_SENSOR_ID")
                    elif value["who_am_i"] != 0x69 and not (value["who_am_i"] == 0x6A and confirmed_tr_c):
                        reasons.add("UNRECOGNIZED_SENSOR")

                    frame = tuple(value[axis] for axis in axes)
                    if previous_frame.get(session_key) == frame:
                        repeated_frames += 1
                    previous_frame[session_key] = frame
                    for axis, raw in zip(axes, frame):
                        histograms[axis][raw + 32768] += 1
                        axis_counts[axis] += 1
                        axis_min[axis] = raw if axis_min[axis] is None else min(axis_min[axis], raw)
                        axis_max[axis] = raw if axis_max[axis] is None else max(axis_max[axis], raw)
                        if abs(raw) >= max_abs_raw:
                            clipping[axis] += 1

                    stationary_label = str(row.get("stationary", "")).strip().lower()
                    if stationary_label in ("1", "true", "yes", "stationary"):
                        stat = stationary_data.setdefault(segment_key, {
                            "session": host_session, "boot_id": boot, "session_instance": current_run,
                            "epoch": epoch, "count": 0, "accel_norm_sum_g": 0.0,
                            "gyro_sum_dps": [0.0, 0.0, 0.0]})
                        stat["count"] += 1
                        stat["accel_norm_sum_g"] += math.sqrt(sum((value[name] * 0.000061) ** 2
                                                                    for name in ("ax", "ay", "az")))
                        for axis_index, name in enumerate(("gx", "gy", "gz")):
                            stat["gyro_sum_dps"][axis_index] += value[name] * 0.0175

                    state = counter_data.setdefault(session_key, {
                        "session": host_session, "boot_id": boot, "session_instance": current_run,
                        "first_observed": {}, "last_observed": {}, "monotonic": {name: True for name in
                            ("queue_drops", "fifo_recoveries", "overrun_observations")},
                        "observed_rows": 0})
                    state["observed_rows"] += 1
                    for name in state["monotonic"]:
                        counter = value[name]
                        if name not in state["first_observed"]:
                            state["first_observed"][name] = counter
                        old = state["last_observed"].get(name)
                        if old is not None and counter < old:
                            state["monotonic"][name] = False
                            reasons.add("COUNTER_DECREASE")
                        state["last_observed"][name] = counter

                    if value["fifo_recoveries"]:
                        reasons.add("FIFO_RECOVERY_OBSERVED")
                    if value["overrun_observations"]:
                        reasons.add("FIFO_OVERRUN_OBSERVED")
                    batch_key = (segment_key, value["batch"])
                    duration.append(value["read_end_us"] - value["read_start_us"])
                    if value["irq_us"] >= 0:
                        latency.append(value["read_start_us"] - value["irq_us"])
                        anchor_candidates.setdefault(batch_key, []).append((value, host_s))
                    if alignment:
                        spans = chunks.setdefault(segment_key, [])
                        if not spans or prior_seq is None or sequence_gap:
                            spans.append([])
                        gyro = [value[name] * 0.0175 * math.pi / 180 for name in ("gx", "gy", "gz")]
                        accel = [value[name] * 0.000061 for name in ("ax", "ay", "az")]
                        spans[-1].append((gyro, accel))

    raw_anchors = {}
    for (segment_key, _batch), candidates in anchor_candidates.items():
        irq_counts = {candidate[0]["irq_count"] for candidate in candidates}
        irq_times = {candidate[0]["irq_us"] for candidate in candidates}
        patterns = {candidate[0]["pattern"] for candidate in candidates}
        if len(irq_counts) != 1 or len(irq_times) != 1 or len(patterns) != 1:
            reasons.add("INCONSISTENT_BATCH_ANCHOR")
            continue
        value, _host_s = max(candidates, key=lambda item: item[0]["word_end"])
        # FIFO_PATTERN is the index of the next 16-bit word in the repeating
        # six-word Gx,Gy,Gz,Ax,Ay,Az sequence. Watermark word 47 is therefore
        # (47-word_end)/6 sample periods after this completed frame. Keep the
        # fractional frame coordinate for offsets 1..5 instead of rounding it.
        anchor_sample = value["seq"] + (47 - value["word_end"]) / 6.0
        raw_anchors.setdefault(segment_key, []).append((anchor_sample, value["irq_us"]))
        data = segments.setdefault(segment_key, {"points": [], "patterns": set(),
                                                  "anchors_rejected_nonmonotonic": 0})
        data["patterns"].add(value["pattern"])

    for segment_key, points in raw_anchors.items():
        points.sort()
        accepted = list(points)
        while len(accepted) >= 2:
            inversion = next((i for i in range(1, len(accepted))
                              if accepted[i][1] <= accepted[i - 1][1]), None)
            if inversion is None:
                break
            provisional = fit_robust_line(accepted, nominal_hz) if len(accepted) >= 3 else None
            if provisional:
                residuals = [abs(y - (provisional["ybar"] +
                                      provisional["slope_us_per_sample"] *
                                      (x - provisional["xbar"])))
                             for x, y in accepted]
                reject_index = inversion - 1 if residuals[inversion - 1] > residuals[inversion] else inversion
            else:
                reject_index = inversion
            accepted.pop(reject_index)
            segments[segment_key]["anchors_rejected_nonmonotonic"] += 1
            reasons.add("MCU_TIME_NONMONOTONIC")
        if len(accepted) >= 2 and all(b[1] > a[1] for a, b in zip(accepted, accepted[1:])):
            segments[segment_key]["points"] = accepted
            for a, b in zip(accepted, accepted[1:]):
                if abs((b[0] - a[0]) - 8) < 1e-6:
                    irq_intervals.append(b[1] - a[1])
        else:
            segments[segment_key]["points"] = []

    models = {key: fit_robust_line(data["points"], nominal_hz) for key, data in segments.items()}
    for model in models.values():
        if model and abs(model["nominal_drift_ppm"]) > 50_000:
            reasons.add("ODR_OUT_OF_RANGE")
    segment_reports = []
    model_by_key = {}
    for key, data in segments.items():
        model = models[key]
        session, boot, instance, epoch = key
        status = "ANCHORED" if model else "TOO_FEW_OR_INVALID_ANCHORS"
        if model is None:
            reasons.add("UNANCHORED_PATTERN")
        segment_reports.append({"session": session, "boot_id": boot, "session_instance": instance,
                                "epoch": epoch, "model": model,
                                "anchor_pattern_offsets": sorted(data["patterns"]),
                                "anchors_rejected_nonmonotonic": data["anchors_rejected_nonmonotonic"],
                                "timing_status": status})
        model_by_key[key] = model

    for state in counter_data.values():
        state["counter_tail_is_censored"] = True
    per_session_counters = list(counter_data.values())
    maxima = {name: max((state["last_observed"].get(name, 0) for state in per_session_counters), default=0)
              for name in ("queue_drops", "fifo_recoveries", "overrun_observations")}
    totals = {name: sum(state["last_observed"].get(name, 0) for state in per_session_counters)
              for name in maxima}

    event_path = Path(events_jsonl) if events_jsonl else path.parent / "events.jsonl"
    event_summary, event_issues = _read_event_log(event_path if event_path.is_file() else None)
    reasons.update(event_issues)
    if test_intent == "fault_injection" and event_summary["planned_fault_events"]:
        reasons.add("PLANNED_FAULT_INJECTION")
    if rows == 0:
        reasons.add("NO_SAMPLES")
    if rows < minimum_valid_rows:
        reasons.add("BELOW_MINIMUM_ROWS")
    if invalid_rows:
        reasons.add("INVALID_CAPTURE_ROWS")
    if any(missing_values.values()):
        reasons.add("MISSING_REQUIRED_VALUES")
    if any(clipping.values()):
        reasons.add("CLIPPING_DETECTED")
    if gaps:
        reasons.add("HOST_SEQUENCE_GAP")
    if test_intent not in ("clean", "fault_injection", "physical", "training"):
        reasons.add("INVALID_TEST_INTENT")
    fatal_reasons = {"NO_SAMPLES", "MISSING_REQUIRED_COLUMNS", "CAPTURE_FILE_UNREADABLE",
                     "INVALID_CAPTURE_ROWS", "DEVICE_FATAL_ERROR", "UNRECOGNIZED_SENSOR",
                     "MISSING_REQUIRED_VALUES", "BATCH_INDEX_NONMONOTONIC",
                     "INCONSISTENT_BATCH_INTERRUPT_COUNT", "INCONSISTENT_BATCH_METADATA",
                     "UNCONFIRMED_SENSOR_ID", "INVALID_EVENT_LOG", "INVALID_TEST_INTENT",
                     "DUPLICATE_COLUMNS", "CLIPPING_DETECTED", "BELOW_MINIMUM_ROWS",
                     "STARTUP_TIMEOUT", "SILENT_STREAM", "PORT_OPEN_FAILED",
                     "SERIAL_READ_FAILED", "INVALID_ROW_FORMAT", "DUPLICATE_SEQUENCE",
                     "CAPTURE_INTERRUPTED", "OUTPUT_SIZE_LIMIT", "INSUFFICIENT_DISK_SPACE",
                     "OUTPUT_DIRECTORY_UNAVAILABLE"}
    degraded_reasons = {"DEVICE_BOOT_CHANGED", "SEQUENCE_RESTART", "HOST_SEQUENCE_GAP",
                        "COUNTER_DECREASE", "FIFO_RECOVERY_OBSERVED", "FIFO_OVERRUN_OBSERVED",
                        "CLIPPING_DETECTED", "MALFORMED_LINES", "UNPLANNED_TRANSPORT_ERRORS",
                        "INCONSISTENT_BATCH_ANCHOR", "IRQ_COUNT_NONMONOTONIC",
                        "MCU_TIME_NONMONOTONIC", "UNANCHORED_PATTERN", "ODR_OUT_OF_RANGE",
                        "BATCH_INDEX_GAP", "PLANNED_FAULT_INJECTION"}
    fatal = bool(reasons & fatal_reasons)
    degraded = bool(reasons & degraded_reasons)
    if fatal:
        verdict = "FAILED"
    elif (degraded and test_intent == "fault_injection" and allow_expected_faults
          and event_summary["unplanned_recoveries"] == 0
          and event_summary["unplanned_counts"].get("malformed_line", 0) == 0):
        verdict = "DEGRADED_EXPECTED"
    elif degraded:
        verdict = "DEGRADED"
    else:
        verdict = "CLEAN"

    config_hash = None
    if config_path:
        try:
            config_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        except OSError:
            reasons.add("CONFIG_UNREADABLE")
            if verdict == "CLEAN":
                verdict = "FAILED"

    axis_stats = _axis_statistics(histograms, axis_counts, axis_min, axis_max)
    stationary_windows = []
    for stat in stationary_data.values():
        count = stat.pop("count")
        accel_mean = stat.pop("accel_norm_sum_g") / count
        gyro_means = [total / count for total in stat.pop("gyro_sum_dps")]
        stationary_windows.append({**stat, "samples": count,
                                   "accel_norm_mean_g": accel_mean,
                                   "gyro_bias_mean_dps": dict(zip(("gx", "gy", "gz"), gyro_means))})
    inactive_axes = [axis for axis, stats in axis_stats.items()
                     if stats["count"] > 0 and stats["min"] == stats["max"]]
    summary = {
        "report_schema_version": "2.0", "quality_policy_version": "1.0",
        "source": source, "test_intent": test_intent, "nominal_odr_hz": nominal_hz,
        "config_hash": config_hash, "rows": rows, "rows_seen": rows_seen,
        "invalid_rows": invalid_rows, "invalid_field_counts": invalid_counts,
        "missing_host_sequence_numbers": gaps,
        "per_session_counters": per_session_counters,
        "total_observed_session_end_counters": totals,
        "maximum_cumulative_counters_per_session": maxima,
        "irq_to_status_read_start_us": percentiles(latency),
        "status_and_fifo_read_duration_us": percentiles(duration),
        "consecutive_watermark_interval_us": percentiles(irq_intervals),
        "segments": segment_reports,
        "estimated_sample_time_clock": "mcu_irq_us_since_boot",
        "events_summary": event_summary,
        "capture_health": capture_health,
        "quality": {"raw_units": "signed_int16_lsb", "axis_statistics_raw": axis_stats,
                    "clipping_threshold_fraction": clip_threshold,
                    "clipping_by_axis": clipping,
                    "clipping_count": sum(clipping.values()),
                    "missingness_rate": (invalid_rows / rows_seen if rows_seen else 0.0),
                    "missing_values_by_field": missing_values,
                    "missingness_rate_by_field": {name: (count / rows_seen if rows_seen else 0.0)
                                                   for name, count in missing_values.items()},
                    "repeated_adjacent_frames": repeated_frames,
                    "inactive_axes_constant_over_recording": inactive_axes,
                    "calibration_status": "unknown",
                    "stationary_metrics": ({"status": "computed_from_explicit_labels",
                                            "windows": stationary_windows} if stationary_windows else
                                           {"status": "not_computed", "reason": "no_stationary_labels"}),
                    "configured_scales": {"accelerometer_g": 2, "gyroscope_dps": 500}},
        "quality_verdict": verdict, "quality_reasons": sorted(reasons),
        "limits": ["FIFO recovery creates an unknown-length gap; sequence numbers do not count overwritten sensor frames.",
                   "Estimated times are robust linear fits to watermark ISR times, not sensor timestamps.",
                   "Anchor offsets use the six-word FIFO pattern; sensor-side channel simultaneity is unverified.",
                   "Serial writes may lose data beyond the firmware queue; sequence gaps reveal only observed gaps.",
                   "Gyro/accelerometer physical alignment needs an external motion reference.",
                   "Counters are last-observed values; unreported tail losses remain unknown."]
    }
    if logic_csv:
        summary["logic_analyzer"] = logic_metrics(logic_csv)
    if alignment:
        summary["controlled_rotation_alignment"] = alignment_metrics(chunks, models, alignment_tolerance_ms)

    header = headers if headers else required
    reconstructed_tmp = output / "reconstructed.csv.tmp"
    run_tracker, prev_tracker = {}, {}
    with path.open(newline="", encoding="utf-8-sig") if path.exists() else open(os.devnull, "r") as src, \
            reconstructed_tmp.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src) if headers else None
        fieldnames = list(header) + (["estimated_sample_host_us"] if "estimated_sample_host_us" not in header else [])
        writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if reader:
            for row in reader:
                estimated = ""
                try:
                    session = int(row["session"])
                    boot = int(row["boot_id"])
                    seq = int(row["seq"])
                    epoch = int(row["epoch"])
                    base = (session, boot)
                    run = run_tracker.get(base, 0)
                    prior = prev_tracker.get(base)
                    if prior is not None and seq < prior:
                        run += 1
                        run_tracker[base] = run
                        prior = None
                    prev_tracker[base] = seq
                    model = model_by_key.get((session, boot, run, epoch))
                    if model:
                        estimated = f'{model["ybar"] + model["slope_us_per_sample"] * (seq - model["xbar"]):.1f}'
                except (KeyError, TypeError, ValueError):
                    pass
                row["estimated_sample_host_us"] = estimated
                writer.writerow(row)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(reconstructed_tmp, output / "reconstructed.csv")
    _atomic_json(output / "report.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--port", required=True)
    cap.add_argument("--output", required=True)
    cap.add_argument("--duration", type=float, default=300)
    cap.add_argument("--baud", type=int, default=921600)
    cap.add_argument("--pause-at", type=float)
    cap.add_argument("--pause-seconds", type=float, default=10)
    cap.add_argument("--disconnect-at", type=float)
    cap.add_argument("--disconnect-seconds", type=float, default=10)
    cap.add_argument("--logic-csv")
    cap.add_argument("--sigrok-driver", help="Optional sigrok-cli driver for concurrent analyzer capture")
    cap.add_argument("--irq-channel", default="0")
    cap.add_argument("--cs-channel", default="1")
    cap.add_argument("--logic-rate", type=int, default=1000000)
    cap.add_argument("--logic-seconds", type=float, default=2)
    cap.add_argument("--alignment", action="store_true")
    cap.add_argument("--alignment-tolerance-ms", type=float)
    cap.add_argument("--startup-timeout-s", type=float, default=10.0)
    cap.add_argument("--silence-timeout-s", type=float, default=2.0)
    cap.add_argument("--minimum-valid-rows", type=int, default=1)
    cap.add_argument("--max-unplanned-recoveries", type=int, default=0)
    cap.add_argument("--max-malformed-lines", type=int, default=0)
    cap.add_argument("--max-line-bytes", type=int, default=4096)
    cap.add_argument("--max-output-bytes", type=int, default=2 * 1024 ** 3)
    cap.add_argument("--minimum-free-bytes", type=int, default=64 * 1024 ** 2)
    cap.add_argument("--test-intent", choices=("clean", "fault_injection", "physical", "training"), default="clean")
    cap.add_argument("--allow-expected-faults", action="store_true")
    cap.add_argument("--source", choices=("unknown", "virtual", "actual_device"), default="actual_device")
    cap.add_argument("--nominal-odr-hz", type=float, default=208.0)
    cap.add_argument("--clip-threshold", type=float, default=0.98)
    cap.add_argument("--confirmed-tr-c", action="store_true")
    cap.add_argument("--config")
    report = sub.add_parser("report")
    report.add_argument("capture_csv")
    report.add_argument("--output", required=True)
    report.add_argument("--logic-csv")
    report.add_argument("--alignment", action="store_true")
    report.add_argument("--alignment-tolerance-ms", type=float)
    report.add_argument("--events-jsonl")
    report.add_argument("--strict", action="store_true")
    report.add_argument("--test-intent", choices=("clean", "fault_injection", "physical", "training"), default="clean")
    report.add_argument("--allow-expected-faults", action="store_true")
    report.add_argument("--source", choices=("unknown", "virtual", "actual_device"), default="unknown")
    report.add_argument("--nominal-odr-hz", type=float, default=208.0)
    report.add_argument("--clip-threshold", type=float, default=0.98)
    report.add_argument("--confirmed-tr-c", action="store_true")
    report.add_argument("--config")
    args = parser.parse_args(argv)
    if args.alignment_tolerance_ms is not None and (not args.alignment or
            not math.isfinite(args.alignment_tolerance_ms) or args.alignment_tolerance_ms < 0):
        parser.error("--alignment-tolerance-ms requires --alignment and a nonnegative finite value")
    if (not math.isfinite(args.nominal_odr_hz) or args.nominal_odr_hz <= 0 or
            not math.isfinite(args.clip_threshold) or not 0 < args.clip_threshold <= 1):
        parser.error("nominal ODR must be positive and clipping threshold must be finite and in (0, 1]")
    if args.command == "capture":
        if not math.isfinite(args.duration) or args.duration <= 0:
            parser.error("duration must be positive and finite")
        if (not math.isfinite(args.startup_timeout_s) or args.startup_timeout_s <= 0 or
                not math.isfinite(args.silence_timeout_s) or args.silence_timeout_s <= 0 or
                args.minimum_valid_rows < 1):
            parser.error("capture health thresholds must be positive and finite")
        if args.max_unplanned_recoveries < 0 or args.max_malformed_lines < 0:
            parser.error("recovery and malformed-line limits cannot be negative")
        if args.max_line_bytes < 128 or args.max_output_bytes <= 0 or args.minimum_free_bytes < 0:
            parser.error("capture resource limits are invalid")
        if not 0 < args.clip_threshold <= 1 or args.nominal_odr_hz <= 0:
            parser.error("quality threshold and nominal ODR must be positive")
        if args.logic_csv and args.sigrok_driver:
            parser.error("choose --logic-csv or --sigrok-driver")
        if args.sigrok_driver and (args.logic_rate <= 0 or not math.isfinite(args.logic_seconds) or
                                   args.logic_seconds <= 0 or args.logic_seconds > args.duration):
            parser.error("logic rate and duration must be positive and logic duration must fit within capture")
        if args.irq_channel == args.cs_channel or any(c in args.irq_channel + args.cs_channel for c in ",="):
            parser.error("logic channel names must differ and cannot contain ',' or '='")
        summary = capture(args)
        if summary["quality_verdict"] in ("CLEAN", "DEGRADED_EXPECTED"):
            if summary["quality_verdict"] == "DEGRADED_EXPECTED" and not args.allow_expected_faults:
                return 2
            return 0
        return 2
    else:
        summary = analyze(args.capture_csv, args.output, args.logic_csv,
                          args.alignment, args.alignment_tolerance_ms,
                          events_jsonl=args.events_jsonl, test_intent=args.test_intent,
                          allow_expected_faults=args.allow_expected_faults, source=args.source,
                          nominal_hz=args.nominal_odr_hz, clip_threshold=args.clip_threshold,
                          confirmed_tr_c=args.confirmed_tr_c, config_path=args.config)
        if args.strict and summary["quality_verdict"] not in ("CLEAN", "DEGRADED_EXPECTED"):
            return 2
        if args.strict and summary["quality_verdict"] == "DEGRADED_EXPECTED" and not args.allow_expected_faults:
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
