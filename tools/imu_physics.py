"""Hardware-free rigid-body IMU signal and clock-truth models.

Quaternion convention: (w, x, y, z) rotates sensor/body vectors into world
coordinates. Angular velocity is right-handed and expressed in body axes.
These models provide synthetic validation data; they are not measurements of a
particular sensor, breakout, filter, or pen.
"""
from __future__ import annotations

import math
import random

G0 = 9.80665
ACCEL_G_PER_LSB = 0.000061
GYRO_DPS_PER_LSB = 0.0175
RAW_MIN, RAW_MAX = -32768, 32767


def quat_normalize(q):
    norm = math.sqrt(sum(value * value for value in q))
    if norm <= 0:
        raise ValueError("zero quaternion is not an orientation")
    return tuple(value / norm for value in q)


def quat_conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw-ax*bx-ay*by-az*bz,
            aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw)


def quat_from_axis_angle(axis, angle_rad):
    norm = math.sqrt(sum(value * value for value in axis))
    if norm <= 0:
        raise ValueError("rotation axis must be nonzero")
    half = angle_rad / 2.0
    scale = math.sin(half) / norm
    return quat_normalize((math.cos(half), *(value * scale for value in axis)))


def quat_rotate(q, vector):
    q = quat_normalize(q)
    pure = (0.0, float(vector[0]), float(vector[1]), float(vector[2]))
    rotated = quat_multiply(quat_multiply(q, pure), quat_conjugate(q))
    return rotated[1:]


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def scale(vector, factor):
    return tuple(value * factor for value in vector)


def rigid_body_sample(*, acceleration_world=(0, 0, 0), orientation_body_to_world=(1, 0, 0, 0),
                      angular_velocity_body_rad_s=(0, 0, 0),
                      angular_acceleration_body_rad_s2=(0, 0, 0),
                      sensor_offset_body_m=(0, 0, 0),
                      sensor_mount_orientation_body=(1, 0, 0, 0),
                      gravity_world=(0, 0, -G0)):
    """Return sensor-frame specific force (m/s²) and gyro (rad/s).

    `sensor_offset_body_m` runs from the body origin to the sensor. Its
    rotational acceleration is alpha×r + omega×(omega×r).
    `sensor_mount_orientation_body` maps sensor coordinates into body axes.
    """
    q_wb = quat_normalize(orientation_body_to_world)
    q_bs = quat_normalize(sensor_mount_orientation_body)
    q_ws = quat_normalize(quat_multiply(q_wb, q_bs))
    omega = tuple(float(value) for value in angular_velocity_body_rad_s)
    alpha = tuple(float(value) for value in angular_acceleration_body_rad_s2)
    lever = tuple(float(value) for value in sensor_offset_body_m)
    rotational_accel_body = add(cross(alpha, lever), cross(omega, cross(omega, lever)))
    sensor_accel_world = add(tuple(float(v) for v in acceleration_world),
                             quat_rotate(q_wb, rotational_accel_body))
    specific_force_world = tuple(a - g for a, g in zip(sensor_accel_world, gravity_world))
    specific_force_sensor = quat_rotate(quat_conjugate(q_ws), specific_force_world)
    gyro_sensor = quat_rotate(quat_conjugate(q_bs), omega)
    return {"specific_force_m_s2": specific_force_sensor,
            "specific_force_g": scale(specific_force_sensor, 1.0 / G0),
            "gyro_rad_s": gyro_sensor,
            "gyro_dps": scale(gyro_sensor, 180.0 / math.pi)}


def _quantize_raw(value):
    return max(RAW_MIN, min(RAW_MAX, int(round(value))))


def simulate_sensor(states, *, seed=1, accel_bias_g=(0, 0, 0), gyro_bias_dps=(0, 0, 0),
                    accel_noise_std_g=0.0, gyro_noise_std_dps=0.0,
                    accel_bias_walk_std_g=0.0, gyro_bias_walk_std_dps=0.0,
                    accel_delay_samples=0, gyro_delay_samples=0,
                    sensor_offset_body_m=(0, 0, 0),
                    sensor_mount_orientation_body=(1, 0, 0, 0)):
    """Generate quantized LSM6DS3-scale raw samples from rigid-body states."""
    if accel_delay_samples < 0 or gyro_delay_samples < 0:
        raise ValueError("sensor delay must be a nonnegative sample count")
    states = list(states)
    rnd = random.Random(seed)
    accel_bias = list(accel_bias_g)
    gyro_bias = list(gyro_bias_dps)
    output, clipped = [], {axis: 0 for axis in ("gx", "gy", "gz", "ax", "ay", "az")}
    physical = []
    for state in states:
        physical.append(rigid_body_sample(
            acceleration_world=state.get("acceleration_world", (0, 0, 0)),
            orientation_body_to_world=state.get("orientation_body_to_world", (1, 0, 0, 0)),
            angular_velocity_body_rad_s=state.get("angular_velocity_body_rad_s", (0, 0, 0)),
            angular_acceleration_body_rad_s2=state.get("angular_acceleration_body_rad_s2", (0, 0, 0)),
            sensor_offset_body_m=sensor_offset_body_m,
            sensor_mount_orientation_body=sensor_mount_orientation_body))
    for index, state in enumerate(states):
        accel_index = max(0, index - accel_delay_samples)
        gyro_index = max(0, index - gyro_delay_samples)
        accel = physical[accel_index]["specific_force_g"]
        gyro = physical[gyro_index]["gyro_dps"]
        for axis in range(3):
            accel_bias[axis] += rnd.gauss(0, accel_bias_walk_std_g)
            gyro_bias[axis] += rnd.gauss(0, gyro_bias_walk_std_dps)
        accel_values = [accel[i] + accel_bias[i] + rnd.gauss(0, accel_noise_std_g) for i in range(3)]
        gyro_values = [gyro[i] + gyro_bias[i] + rnd.gauss(0, gyro_noise_std_dps) for i in range(3)]
        raw_values = [*gyro_values, *accel_values]
        sensitivities = [GYRO_DPS_PER_LSB] * 3 + [ACCEL_G_PER_LSB] * 3
        axes = ("gx", "gy", "gz", "ax", "ay", "az")
        raw = []
        for axis_name, value, sensitivity in zip(axes, raw_values, sensitivities):
            unbounded = int(round(value / sensitivity))
            quantized = _quantize_raw(unbounded)
            if quantized != unbounded:
                clipped[axis_name] += 1
            raw.append(quantized)
        output.append({"seq": int(state.get("seq", index)),
                       "time_s": float(state.get("time_s", index / 208.0)),
                       "raw": dict(zip(axes, raw)),
                       "specific_force_g_truth": tuple(accel),
                       "gyro_dps_truth": tuple(gyro),
                       "position_world_m": tuple(state.get("position_world_m", (0, 0, 0))),
                       "contact": state.get("contact")})
    return {"samples": output, "clipping_by_axis": clipped,
            "units": {"raw": "signed_int16_lsb", "accelerometer": "g",
                      "gyroscope": "deg/s", "time": "s"},
            "sensitivities": {"accelerometer_g_per_lsb": ACCEL_G_PER_LSB,
                              "gyroscope_dps_per_lsb": GYRO_DPS_PER_LSB},
            "seed": seed, "synthetic": True,
            "limitations": ["synthetic rigid-body output; no physical sensor/filter characterization",
                            "temperature behavior is not modeled or calibrated"]}


def generate_clock_truth(*, frames=832, nominal_hz=208.0, odr_ppm=0.0,
                         mcu_clock_ppm=0.0, irq_jitter_std_us=0.0,
                         irq_outlier_probability=0.0, irq_outlier_us=0.0,
                         host_latency_mean_us=1200.0, host_latency_jitter_us=300.0,
                         seed=1, batch_frames=8):
    """Create separate sensor, MCU, IRQ, and host times plus watermark anchors."""
    if frames < 1 or nominal_hz <= 0 or batch_frames < 1:
        raise ValueError("frames, nominal rate, and batch size must be positive")
    rnd = random.Random(seed)
    sensor_hz = nominal_hz * (1.0 + odr_ppm * 1e-6)
    mcu_rate = 1.0 + mcu_clock_ppm * 1e-6
    sensor_period_us = 1e6 / sensor_hz
    mcu_offset_us = rnd.uniform(100_000, 2_000_000)
    rows, anchors = [], []
    previous_host_us = 0.0
    for seq in range(frames):
        sensor_time_us = seq * sensor_period_us
        mcu_time_us = mcu_offset_us + sensor_time_us * mcu_rate
        batch = seq // batch_frames
        is_anchor = (seq % batch_frames == batch_frames - 1)
        irq_time_us = None
        if is_anchor:
            jitter = rnd.gauss(0.0, irq_jitter_std_us)
            if rnd.random() < irq_outlier_probability:
                jitter += math.copysign(irq_outlier_us, rnd.choice((-1.0, 1.0)))
            irq_time_us = mcu_time_us + jitter
            anchors.append({"sample_index": float(seq), "irq_time_us": irq_time_us,
                            "sensor_time_us": sensor_time_us,
                            "mcu_sample_time_us": mcu_time_us,
                            "jitter_us": jitter, "batch": batch})
        host_delay = max(0.0, host_latency_mean_us + rnd.gauss(0.0, host_latency_jitter_us))
        host_time_us = max(previous_host_us, mcu_time_us + host_delay)
        previous_host_us = host_time_us
        rows.append({"seq": seq, "batch": batch, "sensor_time_us": sensor_time_us,
                     "mcu_time_us": mcu_time_us, "irq_time_us": irq_time_us,
                     "host_receive_time_us": host_time_us})
    return {"nominal_hz": nominal_hz, "true_sensor_hz": sensor_hz,
            "true_mcu_clock_rate": mcu_rate, "apparent_mcu_hz": sensor_hz / mcu_rate,
            "odr_ppm": odr_ppm, "mcu_clock_ppm": mcu_clock_ppm,
            "seed": seed, "rows": rows, "anchors": anchors,
            "clock_names": ["sensor_time_us", "mcu_time_us", "irq_time_us", "host_receive_time_us"],
            "synthetic": True}
