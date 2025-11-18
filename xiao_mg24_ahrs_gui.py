"""Simple GUI for visualising AHRS output from the XIAO MG24 Sense board."""

from __future__ import annotations

import argparse
import logging
import math
import os
import threading
import time
from collections import deque

import dearpygui.dearpygui as dpg
import numpy as np
import serial
from ahrs.filters import Madgwick


LOGGER = logging.getLogger(__name__)


# ---- Configuration ----
DEFAULT_SERIAL_PORT = os.environ.get("IMUPEN_SERIAL_PORT", "COM6")
DEFAULT_BAUD_RATE = 115200
DEFAULT_HISTORY_LENGTH = 200  # number of samples to keep for plotting

# ---- Shared state between serial thread and GUI thread ----
HISTORY_LENGTH = DEFAULT_HISTORY_LENGTH
quat_lock = threading.Lock()
current_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
current_euler_deg = np.array([0.0, 0.0, 0.0], dtype=float)
roll_hist = deque(maxlen=HISTORY_LENGTH)
pitch_hist = deque(maxlen=HISTORY_LENGTH)
yaw_hist = deque(maxlen=HISTORY_LENGTH)
connection_status = "Not connected"
sample_index = 0

stop_event = threading.Event()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Visualise roll/pitch/yaw derived from the CSV stream emitted by "
            "the XIAO MG24 Sense sketch."
        )
    )
    parser.add_argument(
        "--serial-port",
        default=DEFAULT_SERIAL_PORT,
        help=(
            "Serial port to open (can also be provided via IMUPEN_SERIAL_PORT "
            "environment variable)."
        ),
    )
    parser.add_argument(
        "--baud-rate",
        type=int,
        default=DEFAULT_BAUD_RATE,
        help="Baud rate that matches the Arduino sketch (default: 115200).",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=DEFAULT_HISTORY_LENGTH,
        help="Number of samples to keep for the rolling plot (default: 200).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging on stdout.",
    )
    return parser.parse_args()


def configure_history_length(length: int) -> None:
    """Resize the shared history deques."""

    global HISTORY_LENGTH, roll_hist, pitch_hist, yaw_hist

    HISTORY_LENGTH = max(1, length)
    roll_hist = deque(maxlen=HISTORY_LENGTH)
    pitch_hist = deque(maxlen=HISTORY_LENGTH)
    yaw_hist = deque(maxlen=HISTORY_LENGTH)


def configure_logging(verbose: bool) -> None:
    """Initialise the logging subsystem."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def quat_to_euler_deg(q):
    """
    Convert quaternion [w, x, y, z] to roll/pitch/yaw in degrees.
    Uses aerospace sequence (roll X, pitch Y, yaw Z).
    """
    w, x, y, z = q

    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return tuple(math.degrees(v) for v in (roll_x, pitch_y, yaw_z))


def parse_imu_line(line: str) -> tuple[float, ...] | None:
    """Parse a single CSV line from the microcontroller sketch."""

    # Skip possible header from Arduino
    lowered = line.lower()
    if lowered.startswith("t_ms") or lowered.startswith("ax"):
        return None

    parts = line.split(",")
    if len(parts) != 7:
        return None

    try:
        floats = tuple(map(float, parts))
    except ValueError:
        return None
    return floats


def serial_worker(serial_port: str, baud_rate: int):
    """
    Background thread:
      - opens serial port
      - parses lines: t_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
      - runs Madgwick AHRS
      - updates shared orientation state
    """
    global connection_status, current_quat, current_euler_deg, sample_index

    madgwick = Madgwick()
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    last_t_ms = None

    while not stop_event.is_set():
        try:
            LOGGER.info("Opening serial port %s @ %d baud", serial_port, baud_rate)
            ser = serial.Serial(serial_port, baud_rate, timeout=0.1)
            with quat_lock:
                connection_status = f"Connected on {serial_port}"
        except serial.SerialException as e:
            LOGGER.error("Serial error: %s", e)
            with quat_lock:
                connection_status = f"Serial error: {e}; retrying..."
            time.sleep(1.0)
            continue

        # Clean out any junk
        ser.reset_input_buffer()

        while not stop_event.is_set():
            try:
                raw = ser.readline()
            except serial.SerialException as e:
                LOGGER.error("Serial read error: %s", e)
                with quat_lock:
                    connection_status = f"Serial error during read: {e}"
                break

            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            parsed = parse_imu_line(line)
            if parsed is None:
                continue

            t_ms, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps = parsed

            # LSM6DS3: accel in g, gyro in deg/s -> convert to SI units
            acc = np.array([ax_g, ay_g, az_g], dtype=float) * 9.80665
            gyr = np.radians(np.array([gx_dps, gy_dps, gz_dps], dtype=float))

            if last_t_ms is None:
                last_t_ms = t_ms
                continue

            dt = (t_ms - last_t_ms) / 1000.0
            if dt <= 0:
                LOGGER.debug("Ignoring non-positive dt=%s", dt)
                continue
            last_t_ms = t_ms

            # Run Madgwick update; dt passed explicitly per sample
            q = madgwick.updateIMU(q, gyr=gyr, acc=acc, dt=dt)
            if q is None:
                LOGGER.debug("Madgwick update returned None")
                continue

            roll_deg, pitch_deg, yaw_deg = quat_to_euler_deg(q)

            with quat_lock:
                current_quat = np.array(q, copy=True)
                current_euler_deg = np.array(
                    [roll_deg, pitch_deg, yaw_deg], dtype=float
                )
                roll_hist.append(roll_deg)
                pitch_hist.append(pitch_deg)
                yaw_hist.append(yaw_deg)
                sample_index += 1

        try:
            ser.close()
        except Exception:
            pass

        LOGGER.info("Reconnecting to serial port %s after short delay", serial_port)
        time.sleep(0.5)  # brief pause before trying to reconnect


def run_gui():
    """
    DearPyGui front-end:
      - displays connection status
      - shows current roll/pitch/yaw
      - plots recent history
    """

    dpg.create_context()
    dpg.create_viewport(
        title="XIAO MG24 Sense AHRS Viewer",
        width=900,
        height=600
    )

    with dpg.window(label="IMU Orientation", tag="main_window", width=880, height=580):
        dpg.add_text("Connection / Filter status:")
        dpg.add_text("Not connected", tag="status_text")
        dpg.add_separator()

        dpg.add_text("Euler angles (degrees):")
        dpg.add_text("Roll:  0.00", tag="roll_text")
        dpg.add_text("Pitch: 0.00", tag="pitch_text")
        dpg.add_text("Yaw:   0.00", tag="yaw_text")

        dpg.add_separator()
        dpg.add_text(f"History (last ~{HISTORY_LENGTH} samples)")

        with dpg.plot(label="Yaw / Pitch / Roll", height=350, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Sample idx", tag="x_axis")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Angle (deg)", tag="y_axis")
            dpg.set_axis_limits("y_axis", -180.0, 180.0)

            dpg.add_line_series([], [], label="Roll", parent=y_axis, tag="roll_series")
            dpg.add_line_series([], [], label="Pitch", parent=y_axis, tag="pitch_series")
            dpg.add_line_series([], [], label="Yaw", parent=y_axis, tag="yaw_series")

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Manual render loop so we can update each frame
    while dpg.is_dearpygui_running():
        with quat_lock:
            status = connection_status
            euler = np.array(current_euler_deg, copy=True)
            roll_vals = list(roll_hist)
            pitch_vals = list(pitch_hist)
            yaw_vals = list(yaw_hist)

        dpg.set_value("status_text", status)
        dpg.set_value("roll_text", f"Roll:  {euler[0]:6.2f}")
        dpg.set_value("pitch_text", f"Pitch: {euler[1]:6.2f}")
        dpg.set_value("yaw_text", f"Yaw:   {euler[2]:6.2f}")

        n = len(roll_vals)
        if n > 0:
            # just use sample index as x-axis
            x_vals = list(range(-n + 1, 1))
            dpg.set_value("roll_series", [x_vals, roll_vals])
            dpg.set_value("pitch_series", [x_vals, pitch_vals])
            dpg.set_value("yaw_series", [x_vals, yaw_vals])

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


def main():
    args = parse_args()
    configure_logging(args.verbose)
    configure_history_length(args.history_length)

    LOGGER.info(
        "Starting IMUpen GUI (port=%s, baud=%d, history=%d)",
        args.serial_port,
        args.baud_rate,
        HISTORY_LENGTH,
    )

    worker = threading.Thread(
        target=serial_worker,
        args=(args.serial_port, args.baud_rate),
        daemon=True,
    )
    worker.start()
    try:
        run_gui()
    finally:
        stop_event.set()
        worker.join(timeout=1.0)


if __name__ == "__main__":
    main()
