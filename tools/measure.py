#!/usr/bin/env python3
"""Record IMUpen FIFO output and summarize timing and loss indicators."""
import argparse
import csv
import json
import math
from pathlib import Path
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


def capture(args):
    try:
        import serial
    except ImportError:
        raise SystemExit("Install pyserial: python -m pip install pyserial")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw_path, events_path = out / "capture.csv", out / "events.jsonl"
    if raw_path.exists() or events_path.exists():
        raise SystemExit("Output already exists; choose a new output directory.")
    for name, at, duration in (("pause", args.pause_at, args.pause_seconds),
                               ("disconnect", args.disconnect_at, args.disconnect_seconds)):
        if at is not None and (at < 0 or duration <= 0 or at + duration >= args.duration):
            raise SystemExit(f"{name} must start after zero and finish before duration.")
    if args.pause_at is not None and args.disconnect_at is not None:
        if max(args.pause_at, args.disconnect_at) < min(args.pause_at + args.pause_seconds, args.disconnect_at + args.disconnect_seconds):
            raise SystemExit("Pause and disconnect intervals cannot overlap.")

    start = time.monotonic()
    session, previous_seq, previous_boot, port = 0, None, None, None
    pause_done = disconnect_done = False
    logic_process, logic_log, logic_attempted = None, None, False
    sigrok_raw = out / "sigrok_samples.csv"
    logic_path = args.logic_csv
    with raw_path.open("w", newline="", encoding="utf-8") as raw, events_path.open("w", encoding="utf-8") as events:
        writer = csv.writer(raw)
        writer.writerow(["host_elapsed_s", "session"] + FIELDS)
        def event(kind, **details):
            entry = {"elapsed_s": round(time.monotonic() - start, 6), "event": kind, **details}
            events.write(json.dumps(entry) + "\n")
            events.flush()
            print(json.dumps(entry), flush=True)
        def open_port():
            device = serial.Serial(port=None, baudrate=args.baud, timeout=0.2)
            device.dtr = False
            device.rts = False
            device.port = args.port
            device.open()
            return device
        event("start", port=args.port, duration_s=args.duration)
        try:
            while time.monotonic() - start < args.duration:
                elapsed = time.monotonic() - start
                if args.pause_at is not None and not pause_done and elapsed >= args.pause_at:
                    pause_done = True
                    event("read_pause_start", seconds=args.pause_seconds)
                    time.sleep(args.pause_seconds)
                    event("read_pause_end")
                if args.disconnect_at is not None and not disconnect_done and elapsed >= args.disconnect_at:
                    disconnect_done = True
                    if port:
                        port.close()
                        port = None
                    event("port_closed", seconds=args.disconnect_seconds)
                    time.sleep(args.disconnect_seconds)
                    event("port_reopen_attempt")
                if port is None:
                    try:
                        port = open_port()
                        event("port_open")
                    except serial.SerialException as exc:
                        event("port_open_error", error=str(exc))
                        time.sleep(1)
                        continue
                try:
                    line = port.readline().decode("ascii", errors="replace").strip()
                except (serial.SerialException, OSError) as exc:
                    event("port_error", error=str(exc))
                    port.close()
                    port = None
                    time.sleep(1)
                    continue
                if not line:
                    continue
                if line.startswith("#"):
                    event("device_message", text=line)
                    if line.startswith("# WHO_AM_I=") and previous_seq is not None:
                        session += 1
                        previous_seq = None
                        previous_boot = None
                        event("device_boot_detected", session=session)
                    continue
                values = line.split(",")
                if len(values) != len(FIELDS):
                    event("malformed_line", text=line[:300])
                    continue
                try:
                    numbers = [int(v) for v in values]
                except ValueError:
                    event("malformed_line", text=line[:300])
                    continue
                boot, seq = numbers[0], numbers[2]
                if previous_boot is not None and boot != previous_boot:
                    session += 1
                    event("boot_id_changed", previous=previous_boot, current=boot)
                    previous_seq = None
                if previous_seq is not None and seq <= previous_seq:
                    session += 1
                    event("sequence_restart", previous=previous_seq, current=seq)
                previous_seq = seq
                previous_boot = boot
                writer.writerow([f"{time.monotonic() - start:.6f}", session] + numbers)
                if args.sigrok_driver and not logic_attempted:
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
                        event("logic_capture_error", error=str(exc))
        finally:
            if port:
                port.close()
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
                        event("logic_capture_error", error=str(exc))
                else:
                    event("logic_capture_error", returncode=code, log="sigrok.log")
            if logic_log:
                logic_log.close()
            event("stop")
    analyze(raw_path, out, logic_path, args.alignment, args.alignment_tolerance_ms)


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
        results.append({"session": key[0], "epoch": key[1], "sample_pairs": n,
                        "best_gyro_offset_samples": lag, "best_gyro_offset_ms": lag_ms,
                        "correlation": score, "resolution_ms": 1000/hz,
                        "peak_at_search_boundary": abs(lag) == max_lag,
                        "user_tolerance_ms": tolerance_ms,
                        "within_tolerance_if_assumptions_hold": (abs(lag_ms) <= tolerance_ms if tolerance_ms is not None and quality else None),
                        "note": "Compare only a controlled fixed-position rotation. Positive offset compares a later gyro row to the accel derivative. Translation, vibration, filter delay, and periodic ambiguity can invalidate this estimate."})
    return results


def analyze(path, output, logic_csv=None, alignment=False, alignment_tolerance_ms=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    segments, seen_batches, previous, gaps, chunks = {}, set(), {}, 0, {}
    latency, duration, irq_intervals = [], [], []
    maxima = {"queue_drops": 0, "fifo_recoveries": 0, "overrun_observations": 0}
    rows = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = {key: int(row[key]) for key in FIELDS}
            session, epoch = int(row["session"]), v["epoch"]
            key = (session, epoch)
            rows += 1
            prev = previous.get(session)
            if prev is not None and v["seq"] > prev + 1:
                gaps += v["seq"] - prev - 1
            previous[session] = v["seq"]
            if alignment:
                spans = chunks.setdefault(key, [])
                if not spans or prev is None or v["seq"] != prev + 1:
                    spans.append([])
                gyro = [v[x] * 0.0175 * math.pi / 180 for x in ("gx", "gy", "gz")]
                accel = [v[x] * 0.000061 for x in ("ax", "ay", "az")]
                spans[-1].append((gyro, accel))
            for name in maxima:
                maxima[name] = max(maxima[name], v[name])
            batch_key = (session, epoch, v["batch"])
            if batch_key in seen_batches:
                continue
            seen_batches.add(batch_key)
            duration.append(v["read_end_us"] - v["read_start_us"])
            if v["irq_us"] < 0:
                continue
            latency.append(v["read_start_us"] - v["irq_us"])
            # For pattern 0, the watermark is crossed by word 47. This anchor
            # assumes the observed IRQ corresponds to that crossing.
            if v["pattern"] != 0 or v["word_end"] % 6 != 5:
                continue
            anchor_seq = v["seq"] + (47 - v["word_end"]) // 6
            segment = segments.setdefault(key, {"points": [], "last": None})
            last = segment["last"]
            if last and v["irq_count"] == last[0] + 1 and anchor_seq == last[1] + 8:
                irq_intervals.append(v["irq_us"] - last[2])
            segment["points"].append((anchor_seq, v["irq_us"]))
            segment["last"] = (v["irq_count"], anchor_seq, v["irq_us"])
    models = {key: fit_line(data["points"]) for key, data in segments.items()}
    summary = {"rows": rows, "missing_host_sequence_numbers": gaps,
               "maximum_cumulative_counters_per_session": maxima,
               "irq_to_status_read_start_us": percentiles(latency),
               "status_and_fifo_read_duration_us": percentiles(duration),
               "consecutive_watermark_interval_us": percentiles(irq_intervals),
               "segments": [{"session": k[0], "epoch": k[1], "model": v} for k, v in models.items()],
               "limits": ["FIFO recovery creates an unknown-length gap; sequence numbers do not count overwritten sensor frames.",
                          "Estimated times are a linear host-clock fit to watermark ISR times, not sensor timestamps.",
                          "Serial writes may lose data beyond the firmware queue; sequence gaps reveal only observed gaps.",
                          "Gyro/accelerometer physical alignment needs an external motion reference."]}
    if logic_csv:
        summary["logic_analyzer"] = logic_metrics(logic_csv)
    if alignment:
        summary["controlled_rotation_alignment"] = alignment_metrics(chunks, models, alignment_tolerance_ms)
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with open(path, newline="", encoding="utf-8") as src, (output / "reconstructed.csv").open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames + ["estimated_sample_host_us"])
        writer.writeheader()
        for row in reader:
            model = models.get((int(row["session"]), int(row["epoch"])))
            row["estimated_sample_host_us"] = (f'{model["ybar"] + model["slope_us_per_sample"] * (int(row["seq"]) - model["xbar"]):.1f}' if model else "")
            writer.writerow(row)
    print(json.dumps(summary, indent=2))


def main():
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
    report = sub.add_parser("report")
    report.add_argument("capture_csv")
    report.add_argument("--output", required=True)
    report.add_argument("--logic-csv")
    report.add_argument("--alignment", action="store_true")
    report.add_argument("--alignment-tolerance-ms", type=float)
    args = parser.parse_args()
    if args.alignment_tolerance_ms is not None and (not args.alignment or args.alignment_tolerance_ms < 0):
        parser.error("--alignment-tolerance-ms requires --alignment and a nonnegative value")
    if args.command == "capture":
        if args.duration <= 0:
            parser.error("duration must be positive")
        if args.logic_csv and args.sigrok_driver:
            parser.error("choose --logic-csv or --sigrok-driver")
        if args.sigrok_driver and (args.logic_rate <= 0 or args.logic_seconds <= 0 or args.logic_seconds > args.duration):
            parser.error("logic rate and duration must be positive and logic duration must fit within capture")
        if args.irq_channel == args.cs_channel or any(c in args.irq_channel + args.cs_channel for c in ",="):
            parser.error("logic channel names must differ and cannot contain ',' or '='")
        capture(args)
    else:
        analyze(args.capture_csv, args.output, args.logic_csv, args.alignment, args.alignment_tolerance_ms)


if __name__ == "__main__":
    main()
