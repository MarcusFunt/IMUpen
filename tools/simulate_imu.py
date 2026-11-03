#!/usr/bin/env python3
"""Write reproducible rigid-body, clock, and sensor-value truth JSON."""
import argparse
import json
import math
import os
from pathlib import Path

from imu_physics import (ACCEL_G_PER_LSB, G0, generate_clock_truth, quat_from_axis_angle,
                         quat_rotate, simulate_sensor)


def generate(output, *, frames=832, nominal_hz=208.0, odr_ppm=0.0,
             mcu_clock_ppm=0.0, irq_jitter_std_us=35.0,
             irq_outlier_probability=0.005, irq_outlier_us=25000.0,
             host_latency_mean_us=1200.0, host_latency_jitter_us=300.0,
             accel_noise_std_g=0.001, gyro_noise_std_dps=0.08,
             gyro_bias_dps=0.15, accel_bias_g=0.005, seed=1):
    if frames < 16 or nominal_hz <= 0:
        raise ValueError("at least 16 frames and a positive nominal rate are required")
    true_rate = nominal_hz * (1 + odr_ppm * 1e-6)
    states = []
    for index in range(frames):
        t = index / true_rate
        theta = 0.36 * math.sin(2 * math.pi * 0.43 * t) + 0.13 * math.sin(2 * math.pi * 0.81 * t)
        omega_z = (0.36 * 2 * math.pi * 0.43 * math.cos(2 * math.pi * 0.43 * t) +
                   0.13 * 2 * math.pi * 0.81 * math.cos(2 * math.pi * 0.81 * t))
        alpha_z = (-0.36 * (2 * math.pi * 0.43) ** 2 * math.sin(2 * math.pi * 0.43 * t) -
                   0.13 * (2 * math.pi * 0.81) ** 2 * math.sin(2 * math.pi * 0.81 * t))
        stroke_hz = 0.8
        x = 0.03 * math.sin(2 * math.pi * stroke_hz * t)
        ax = -0.03 * (2 * math.pi * stroke_hz) ** 2 * math.sin(2 * math.pi * stroke_hz * t)
        states.append({"seq": index, "time_s": t,
                       "position_world_m": (x, 0.0, 0.0),
                       "acceleration_world": (ax, 0.0, 0.0),
                       "orientation_body_to_world": quat_from_axis_angle((0, 0, 1), theta),
                       "angular_velocity_body_rad_s": (0.0, 0.0, omega_z),
                       "angular_acceleration_body_rad_s2": (0.0, 0.0, alpha_z),
                       "contact": math.sin(2 * math.pi * 0.5 * t) >= 0})
    sensor = simulate_sensor(states, seed=seed,
                             accel_bias_g=(accel_bias_g, 0.0, 0.0),
                             gyro_bias_dps=(0.0, 0.0, gyro_bias_dps),
                             accel_noise_std_g=accel_noise_std_g,
                             gyro_noise_std_dps=gyro_noise_std_dps,
                             sensor_offset_body_m=(0.015, 0.0, 0.0))
    clocks = generate_clock_truth(frames=frames, nominal_hz=nominal_hz,
                                  odr_ppm=odr_ppm, mcu_clock_ppm=mcu_clock_ppm,
                                  irq_jitter_std_us=irq_jitter_std_us,
                                  irq_outlier_probability=irq_outlier_probability,
                                  irq_outlier_us=irq_outlier_us,
                                  host_latency_mean_us=host_latency_mean_us,
                                  host_latency_jitter_us=host_latency_jitter_us,
                                  seed=seed)
    irq_by_batch = {row["batch"]: row["irq_time_us"] for row in clocks["rows"]
                    if row["irq_time_us"] is not None}
    samples = []
    velocity_est = [0.0, 0.0, 0.0]
    position_est = [0.0, 0.0, 0.0]
    last_drift = 0.0
    dt = 1.0 / true_rate
    for sensor_row, clock_row in zip(sensor["samples"], clocks["rows"]):
        raw_accel = [sensor_row["raw"][axis] * ACCEL_G_PER_LSB * G0
                     for axis in ("ax", "ay", "az")]
        q_wb = states[sensor_row["seq"]]["orientation_body_to_world"]
        accel_world_est = quat_rotate(q_wb, raw_accel)
        accel_world_est = [accel_world_est[i] + (0.0, 0.0, -G0)[i] for i in range(3)]
        velocity_est = [velocity_est[i] + accel_world_est[i] * dt for i in range(3)]
        position_est = [position_est[i] + velocity_est[i] * dt for i in range(3)]
        sensor_position_truth = [states[sensor_row["seq"]]["position_world_m"][i] +
                                 quat_rotate(q_wb, (0.015, 0.0, 0.0))[i] for i in range(3)]
        last_drift = math.sqrt(sum((position_est[i] - sensor_position_truth[i]) ** 2
                                   for i in range(3)))
        samples.append({"seq": sensor_row["seq"], "batch": clock_row["batch"],
                        "t_sensor_us": clock_row["sensor_time_us"],
                        "t_mcu_us": clock_row["mcu_time_us"],
                        "t_irq_us": irq_by_batch.get(clock_row["batch"]),
                        "t_host_receive_us": clock_row["host_receive_time_us"],
                        "raw": sensor_row["raw"],
                        "specific_force_g_truth": sensor_row["specific_force_g_truth"],
                        "gyro_dps_truth": sensor_row["gyro_dps_truth"],
                        "position_world_m_truth": sensor_row["position_world_m"],
                        "contact_truth_only": sensor_row["contact"]})
    document = {"simulator_version": "rigid_body_imu_v1",
                "configuration": {"frames": frames, "nominal_hz": nominal_hz,
                                  "odr_ppm": odr_ppm, "mcu_clock_ppm": mcu_clock_ppm,
                                  "irq_jitter_std_us": irq_jitter_std_us,
                                  "irq_outlier_probability": irq_outlier_probability,
                                  "irq_outlier_us": irq_outlier_us,
                                  "host_latency_mean_us": host_latency_mean_us,
                                  "host_latency_jitter_us": host_latency_jitter_us,
                                  "accel_noise_std_g": accel_noise_std_g,
                                  "gyro_noise_std_dps": gyro_noise_std_dps,
                                  "gyro_bias_dps": gyro_bias_dps,
                                  "accel_bias_g": accel_bias_g, "seed": seed},
                "source_type": "synthetic", "synthetic": True,
                "units": {"t_sensor_us": "us", "t_mcu_us": "us", "t_irq_us": "us",
                          "t_host_receive_us": "us", "accelerometer": "g",
                          "gyroscope": "deg/s", "position_world_m_truth": "m"},
                "clock_truth": {key: clocks[key] for key in
                                ("true_sensor_hz", "true_mcu_clock_rate", "apparent_mcu_hz",
                                 "clock_names")},
                "clipping_by_axis": sensor["clipping_by_axis"],
                "free_integration_drift_metric_m": last_drift,
                "free_integration_note": "accelerometer integration uses truth orientation to remove gravity; real attitude error would add drift",
                "samples": samples,
                "limitations": ["synthetic rigid-body and clock truth only",
                                "no measured optical pen-tip trajectory or physical temperature response",
                                "no actual sensor, USB, SPI electrical, or board scheduling evidence"]}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=832)
    parser.add_argument("--nominal-hz", type=float, default=208)
    parser.add_argument("--odr-ppm", type=float, default=0)
    parser.add_argument("--mcu-clock-ppm", type=float, default=0)
    parser.add_argument("--irq-jitter-std-us", type=float, default=35)
    parser.add_argument("--irq-outlier-probability", type=float, default=0.005)
    parser.add_argument("--irq-outlier-us", type=float, default=25000)
    parser.add_argument("--host-latency-mean-us", type=float, default=1200)
    parser.add_argument("--host-latency-jitter-us", type=float, default=300)
    parser.add_argument("--accel-noise-std-g", type=float, default=0.001)
    parser.add_argument("--gyro-noise-std-dps", type=float, default=0.08)
    parser.add_argument("--gyro-bias-dps", type=float, default=0.15)
    parser.add_argument("--accel-bias-g", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    generate(args.output, frames=args.frames, nominal_hz=args.nominal_hz,
             odr_ppm=args.odr_ppm, mcu_clock_ppm=args.mcu_clock_ppm,
             irq_jitter_std_us=args.irq_jitter_std_us,
             irq_outlier_probability=args.irq_outlier_probability,
             irq_outlier_us=args.irq_outlier_us,
             host_latency_mean_us=args.host_latency_mean_us,
             host_latency_jitter_us=args.host_latency_jitter_us,
             accel_noise_std_g=args.accel_noise_std_g,
             gyro_noise_std_dps=args.gyro_noise_std_dps,
             gyro_bias_dps=args.gyro_bias_dps, accel_bias_g=args.accel_bias_g,
             seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
