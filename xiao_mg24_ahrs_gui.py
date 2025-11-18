"""Simple GUI for visualising AHRS output from the XIAO MG24 Sense board."""

from __future__ import annotations

import argparse
import logging
import os
import struct
import threading
import time
from collections import deque
from typing import NamedTuple

import dearpygui.dearpygui as dpg
import numpy as np
import serial
from ahrs.filters import Madgwick


LOGGER = logging.getLogger(__name__)


# ---- Configuration ----
DEFAULT_SERIAL_PORT = os.environ.get("IMUPEN_SERIAL_PORT", "COM6")
DEFAULT_BAUD_RATE = 115200
DEFAULT_HISTORY_LENGTH = 200  # number of samples to keep for plotting

PACKET_PREAMBLE = 0xAA55
PACKET_VERSION = 1
PACKET_STRUCT = struct.Struct("<HHII6fH")
PACKET_SIZE = PACKET_STRUCT.size
PREAMBLE_BYTES = PACKET_PREAMBLE.to_bytes(2, "little")

# ---- Shared state between serial thread and GUI thread ----
HISTORY_LENGTH = DEFAULT_HISTORY_LENGTH
quat_lock = threading.Lock()
current_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
current_direction_vec = np.array([0.0, 0.0, -1.0], dtype=float)
vec_x_hist = deque(maxlen=HISTORY_LENGTH)
vec_y_hist = deque(maxlen=HISTORY_LENGTH)
vec_z_hist = deque(maxlen=HISTORY_LENGTH)
sample_time_hist = deque(maxlen=HISTORY_LENGTH)
connection_status = "Not connected"
sample_index = 0
first_sample_time_ms: float | None = None
last_sample_interval_ms: float | None = None

stop_event = threading.Event()


class ImuPacket(NamedTuple):
    sequence: int
    timestamp_ms: int
    ax_g: float
    ay_g: float
    az_g: float
    gx_dps: float
    gy_dps: float
    gz_dps: float


def checksum_bytes(data: bytes) -> int:
    """Return a simple 16-bit additive checksum of ``data``."""

    return sum(data) & 0xFFFF


def extract_packets(buffer: bytearray) -> list[ImuPacket]:
    """Pull as many complete packets as possible out of ``buffer``."""

    packets: list[ImuPacket] = []
    while True:
        idx = buffer.find(PREAMBLE_BYTES)
        if idx == -1:
            buffer.clear()
            break
        if idx > 0:
            del buffer[:idx]
        if len(buffer) < PACKET_SIZE:
            break

        raw_packet = bytes(buffer[:PACKET_SIZE])
        (
            preamble,
            version,
            sequence,
            timestamp_ms,
            ax_g,
            ay_g,
            az_g,
            gx_dps,
            gy_dps,
            gz_dps,
            checksum,
        ) = PACKET_STRUCT.unpack(raw_packet)

        if preamble != PACKET_PREAMBLE:
            del buffer[0]
            continue

        calc_checksum = checksum_bytes(raw_packet[:-2])
        if version != PACKET_VERSION or checksum != calc_checksum:
            del buffer[0]
            continue

        packets.append(
            ImuPacket(
                sequence=sequence,
                timestamp_ms=timestamp_ms,
                ax_g=ax_g,
                ay_g=ay_g,
                az_g=az_g,
                gx_dps=gx_dps,
                gy_dps=gy_dps,
                gz_dps=gz_dps,
            )
        )
        del buffer[:PACKET_SIZE]

    return packets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Visualise the direction vector derived from the binary stream emitted"
            " by the XIAO MG24 Sense sketch."
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

    global HISTORY_LENGTH, vec_x_hist, vec_y_hist, vec_z_hist, sample_time_hist
    global last_sample_interval_ms, first_sample_time_ms

    HISTORY_LENGTH = max(1, length)
    vec_x_hist = deque(maxlen=HISTORY_LENGTH)
    vec_y_hist = deque(maxlen=HISTORY_LENGTH)
    vec_z_hist = deque(maxlen=HISTORY_LENGTH)
    sample_time_hist = deque(maxlen=HISTORY_LENGTH)
    last_sample_interval_ms = None
    first_sample_time_ms = None


def configure_logging(verbose: bool) -> None:
    """Initialise the logging subsystem."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def quat_to_direction_vector(q, base_vector=None):
    """Rotate ``base_vector`` by quaternion ``q`` and return the resulting vector."""

    if base_vector is None:
        base_vector = np.array([0.0, 0.0, -1.0], dtype=float)

    w, x, y, z = q
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return rot @ base_vector


def serial_worker(serial_port: str, baud_rate: int):
    """
    Background thread:
      - opens serial port
      - parses binary packets: sequence, timestamp, accel (g), gyro (deg/s)
      - runs Madgwick AHRS
      - updates shared orientation state
    """
    global connection_status, current_quat, current_direction_vec, sample_index
    global sample_time_hist, first_sample_time_ms, last_sample_interval_ms

    madgwick = Madgwick()
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    last_t_ms = None
    last_sequence = None

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
        buffer = bytearray()

        while not stop_event.is_set():
            try:
                chunk = ser.read(ser.in_waiting or PACKET_SIZE)
            except serial.SerialException as e:
                LOGGER.error("Serial read error: %s", e)
                with quat_lock:
                    connection_status = f"Serial error during read: {e}"
                break

            if chunk:
                buffer.extend(chunk)

            packets = extract_packets(buffer)
            if not packets:
                continue

            for packet in packets:
                t_ms = packet.timestamp_ms
                ax_g = packet.ax_g
                ay_g = packet.ay_g
                az_g = packet.az_g
                gx_dps = packet.gx_dps
                gy_dps = packet.gy_dps
                gz_dps = packet.gz_dps

                if last_sequence is None:
                    status_suffix = ""
                elif packet.sequence == last_sequence + 1:
                    status_suffix = ""
                elif packet.sequence > last_sequence:
                    dropped = packet.sequence - last_sequence - 1
                    LOGGER.warning("Detected %d dropped packet(s)", dropped)
                    status_suffix = f"; dropped {dropped} pkt"
                else:
                    LOGGER.info("Packet sequence reset (device reboot?)")
                    status_suffix = "; sequence reset"
                last_sequence = packet.sequence
                status_message = (
                    f"Connected on {serial_port} (seq={packet.sequence}){status_suffix}"
                )

                # LSM6DS3: accel in g, gyro in deg/s -> convert to SI units
                acc = np.array([ax_g, ay_g, az_g], dtype=float) * 9.80665
                gyr = np.radians(np.array([gx_dps, gy_dps, gz_dps], dtype=float))

                if last_t_ms is None:
                    last_t_ms = t_ms
                    first_sample_time_ms = t_ms
                    continue

                dt = (t_ms - last_t_ms) / 1000.0
                if dt <= 0:
                    LOGGER.debug("Ignoring non-positive dt=%s", dt)
                    last_t_ms = t_ms
                    first_sample_time_ms = t_ms
                    continue
                last_t_ms = t_ms

                # Run Madgwick update; dt passed explicitly per sample
                q = madgwick.updateIMU(q, gyr=gyr, acc=acc, dt=dt)
                if q is None:
                    LOGGER.debug("Madgwick update returned None")
                    continue

                direction_vec = quat_to_direction_vector(q)
                elapsed_s = 0.0
                if first_sample_time_ms is not None:
                    elapsed_s = max(0.0, (t_ms - first_sample_time_ms) / 1000.0)
                last_sample_interval_ms = dt * 1000.0

                with quat_lock:
                    connection_status = status_message
                    current_quat = np.array(q, copy=True)
                    current_direction_vec = np.array(direction_vec, copy=True)
                    vec_x_hist.append(direction_vec[0])
                    vec_y_hist.append(direction_vec[1])
                    vec_z_hist.append(direction_vec[2])
                    sample_time_hist.append(elapsed_s)
                    sample_index = packet.sequence

        try:
            ser.close()
        except Exception:
            pass

        last_t_ms = None
        first_sample_time_ms = None
        last_sequence = None
        LOGGER.info("Reconnecting to serial port %s after short delay", serial_port)
        time.sleep(0.5)  # brief pause before trying to reconnect


def run_gui():
    """
    DearPyGui front-end:
      - displays connection status
      - shows the current direction vector
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

        dpg.add_text("Direction vector (device-forward axis):")
        dpg.add_text("X:  0.000", tag="vec_x_text")
        dpg.add_text("Y:  0.000", tag="vec_y_text")
        dpg.add_text("Z:  0.000", tag="vec_z_text")

        dpg.add_separator()
        dpg.add_text("Timing diagnostics:")
        dpg.add_text("Last interval: n/a", tag="sample_interval_text")

        dpg.add_separator()
        dpg.add_text(f"History (last ~{HISTORY_LENGTH} samples)")

        with dpg.plot(label="Direction vector components", height=350, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label="Elapsed time (s)",
                tag="x_axis",
            )
            y_axis = dpg.add_plot_axis(
                dpg.mvYAxis, label="Component", tag="y_axis"
            )
            dpg.set_axis_limits("y_axis", -1.2, 1.2)

            dpg.add_line_series([], [], label="X", parent=y_axis, tag="vec_x_series")
            dpg.add_line_series([], [], label="Y", parent=y_axis, tag="vec_y_series")
            dpg.add_line_series([], [], label="Z", parent=y_axis, tag="vec_z_series")

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Manual render loop so we can update each frame
    while dpg.is_dearpygui_running():
        with quat_lock:
            status = connection_status
            direction_vec = np.array(current_direction_vec, copy=True)
            vec_x_vals = list(vec_x_hist)
            vec_y_vals = list(vec_y_hist)
            vec_z_vals = list(vec_z_hist)
            time_vals = list(sample_time_hist)
            interval_ms = last_sample_interval_ms

        dpg.set_value("status_text", status)
        dpg.set_value("vec_x_text", f"X: {direction_vec[0]:7.3f}")
        dpg.set_value("vec_y_text", f"Y: {direction_vec[1]:7.3f}")
        dpg.set_value("vec_z_text", f"Z: {direction_vec[2]:7.3f}")

        if interval_ms is None or interval_ms <= 0:
            interval_text = "Last interval: n/a"
        else:
            rate_hz = 1000.0 / interval_ms if interval_ms > 0 else 0.0
            interval_text = f"Last interval: {interval_ms:6.1f} ms (~{rate_hz:5.1f} Hz)"
        dpg.set_value("sample_interval_text", interval_text)

        n = len(vec_x_vals)
        if n > 0 and len(time_vals) == n:
            dpg.set_value("vec_x_series", [time_vals, vec_x_vals])
            dpg.set_value("vec_y_series", [time_vals, vec_y_vals])
            dpg.set_value("vec_z_series", [time_vals, vec_z_vals])

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
