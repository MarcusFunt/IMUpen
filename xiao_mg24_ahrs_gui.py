"""Simple GUI for visualising AHRS output from the XIAO MG24 Sense board."""

from __future__ import annotations

import argparse
import logging
import os
import struct
import threading
import time
from collections import deque
from typing import Callable, NamedTuple

import dearpygui.dearpygui as dpg
import numpy as np
import serial
from ahrs.filters import Madgwick, UKF


LOGGER = logging.getLogger(__name__)


# ---- Configuration ----
DEFAULT_SERIAL_PORT = os.environ.get("IMUPEN_SERIAL_PORT", "COM6")
DEFAULT_BAUD_RATE = 115200
DEFAULT_HISTORY_LENGTH = 200  # number of samples to keep for plotting
DEFAULT_FILTER_NAME = "madgwick"
EXPECTED_PACKET_RATE_HZ = 100.0

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
current_euler_deg = np.array([0.0, 0.0, 0.0], dtype=float)
current_acc_g = np.array([0.0, 0.0, 0.0], dtype=float)
current_gyro_dps = np.array([0.0, 0.0, 0.0], dtype=float)
vec_x_hist = deque(maxlen=HISTORY_LENGTH)
vec_y_hist = deque(maxlen=HISTORY_LENGTH)
vec_z_hist = deque(maxlen=HISTORY_LENGTH)
sample_time_hist = deque(maxlen=HISTORY_LENGTH)
connection_status = "Not connected"
sample_index = 0
first_sample_time_ms: float | None = None
last_sample_interval_ms: float | None = None
elapsed_time_s: float = 0.0
selected_filter_name: str = DEFAULT_FILTER_NAME

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
        "--filter",
        choices=["madgwick", "ukf"],
        default=DEFAULT_FILTER_NAME,
        help="AHRS filter to run on the host (default: madgwick).",
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
    global last_sample_interval_ms, first_sample_time_ms, elapsed_time_s

    HISTORY_LENGTH = max(1, length)
    vec_x_hist = deque(maxlen=HISTORY_LENGTH)
    vec_y_hist = deque(maxlen=HISTORY_LENGTH)
    vec_z_hist = deque(maxlen=HISTORY_LENGTH)
    sample_time_hist = deque(maxlen=HISTORY_LENGTH)
    last_sample_interval_ms = None
    first_sample_time_ms = None
    elapsed_time_s = 0.0


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


def quat_to_euler_degrees(q: np.ndarray) -> np.ndarray:
    """Convert quaternion ``q`` into roll/pitch/yaw (degrees)."""

    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])


def create_filter_runner(filter_name: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray | None]:
    """Return a callable that runs the requested filter for each sample."""

    name = filter_name.lower()
    if name == "ukf":
        ukf_filter = UKF()

        def run(q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, dt: float):
            return ukf_filter.update(q=q, gyr=gyr, acc=acc, dt=dt)

        return run

    madgwick_filter = Madgwick()

    def run(q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, dt: float):
        return madgwick_filter.updateIMU(q, gyr=gyr, acc=acc, dt=dt)

    return run


def serial_worker(serial_port: str, baud_rate: int, filter_name: str):
    """
    Background thread:
      - opens serial port
      - parses binary packets: sequence, timestamp, accel (g), gyro (deg/s)
      - runs the requested AHRS filter
      - updates shared orientation state
    """
    global connection_status, current_quat, current_direction_vec, sample_index
    global sample_time_hist, first_sample_time_ms, last_sample_interval_ms
    global current_euler_deg, current_acc_g, current_gyro_dps, elapsed_time_s

    filter_runner = create_filter_runner(filter_name)
    filter_label = filter_name.upper()
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
                bytes_waiting = ser.in_waiting
                # ``ser.read(0)`` returns immediately without fetching any data which
                # would make the loop spin aggressively.  Previously we requested a
                # whole packet when nothing was waiting which meant that the read
                # call would block until an entire packet had arrived, effectively
                # delivering samples to the GUI in noticeable bursts.  By always
                # requesting at least one byte we allow the OS driver to deliver
                # data as soon as it becomes available which keeps the updates
                # flowing smoothly while still avoiding a hot loop.
                chunk_size = bytes_waiting if bytes_waiting > 0 else 1
                chunk = ser.read(chunk_size)
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
                    f"Connected on {serial_port} (seq={packet.sequence}, filter={filter_label})"
                    f"{status_suffix}"
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

                # Run AHRS filter update; dt passed explicitly per sample
                q = filter_runner(q, gyr=gyr, acc=acc, dt=dt)
                if q is None:
                    LOGGER.debug("Filter update returned None")
                    continue

                direction_vec = quat_to_direction_vector(q)
                euler_deg = quat_to_euler_degrees(q)
                elapsed_s = 0.0
                if first_sample_time_ms is not None:
                    elapsed_s = max(0.0, (t_ms - first_sample_time_ms) / 1000.0)
                last_sample_interval_ms = dt * 1000.0

                with quat_lock:
                    connection_status = status_message
                    current_quat = np.array(q, copy=True)
                    current_direction_vec = np.array(direction_vec, copy=True)
                    current_euler_deg = np.array(euler_deg, copy=True)
                    current_acc_g = np.array([ax_g, ay_g, az_g], dtype=float)
                    current_gyro_dps = np.array([gx_dps, gy_dps, gz_dps], dtype=float)
                    vec_x_hist.append(direction_vec[0])
                    vec_y_hist.append(direction_vec[1])
                    vec_z_hist.append(direction_vec[2])
                    sample_time_hist.append(elapsed_s)
                    sample_index = packet.sequence
                    elapsed_time_s = elapsed_s

        try:
            ser.close()
        except Exception:
            pass

        last_t_ms = None
        first_sample_time_ms = None
        last_sequence = None
        with quat_lock:
            elapsed_time_s = 0.0
        LOGGER.info("Reconnecting to serial port %s after short delay", serial_port)
        time.sleep(0.5)  # brief pause before trying to reconnect


def run_gui(filter_name: str):
    """
    DearPyGui front-end:
      - displays connection status
      - shows the current direction vector
      - plots recent history
    """

    dpg.create_context()

    # Typeface + colour palette shared across the widgets to give a subtle
    # "brand" identity without altering any data handling logic.
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    with dpg.font_registry():
        default_font = dpg.add_font(font_path, 16)
    dpg.bind_font(default_font)

    header_color = (0, 186, 188, 255)
    separator_color = (0, 90, 100, 255)
    plot_line_color = (255, 145, 0, 255)
    plot_fill_color = (0, 186, 188, 64)

    with dpg.theme() as branded_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(
                dpg.mvThemeCol_Header, header_color, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderActive,
                (0, 150, 155, 255),
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_HeaderHovered,
                (0, 200, 205, 255),
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_Separator,
                separator_color,
                category=dpg.mvThemeCat_Core,
            )
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotLines,
                plot_line_color,
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotLinesHovered,
                (255, 180, 80, 255),
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotBorder,
                separator_color,
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotFill,
                plot_fill_color,
                category=dpg.mvThemeCat_Plots,
            )
    dpg.bind_theme(branded_theme)
    dpg.create_viewport(
        title="XIAO MG24 Sense AHRS Viewer",
        width=900,
        height=600
    )

    with dpg.window(label="IMU Orientation", tag="main_window", width=880, height=580):
        dpg.add_text("Connection / Filter status:")
        with dpg.group(horizontal=True):
            dpg.add_text("Not connected", tag="status_text")
            dpg.add_button(
                label="",
                width=18,
                height=18,
                enabled=False,
                tag="status_indicator",
            )
        dpg.add_text(f"Filter: {filter_name.upper()}", tag="filter_text")
        dpg.add_separator()

        dpg.add_text("Direction vector (device-forward axis):")
        dpg.add_text("X:  0.000", tag="vec_x_text")
        dpg.add_text("Y:  0.000", tag="vec_y_text")
        dpg.add_text("Z:  0.000", tag="vec_z_text")

        dpg.add_separator()
        dpg.add_text("Orientation details:")
        dpg.add_text("Quat w:  1.000", tag="quat_w_text")
        dpg.add_text("Quat x:  0.000", tag="quat_x_text")
        dpg.add_text("Quat y:  0.000", tag="quat_y_text")
        dpg.add_text("Quat z:  0.000", tag="quat_z_text")
        dpg.add_text("Roll (deg):   0.0", tag="roll_text")
        dpg.add_text("Pitch (deg):  0.0", tag="pitch_text")
        dpg.add_text("Yaw (deg):    0.0", tag="yaw_text")

        dpg.add_separator()
        dpg.add_text("Raw sensor data:")
        dpg.add_text("Accel X: 0.000 g", tag="acc_x_text")
        dpg.add_text("Accel Y: 0.000 g", tag="acc_y_text")
        dpg.add_text("Accel Z: 0.000 g", tag="acc_z_text")
        dpg.add_text("Gyro X: 0.000 dps", tag="gyro_x_text")
        dpg.add_text("Gyro Y: 0.000 dps", tag="gyro_y_text")
        dpg.add_text("Gyro Z: 0.000 dps", tag="gyro_z_text")

        dpg.add_separator()
        dpg.add_text("Timing diagnostics:")
        dpg.add_text("Last interval: n/a", tag="sample_interval_text")
        dpg.add_text("Sequence: n/a", tag="sample_sequence_text")
        dpg.add_text("Elapsed time: 0.0 s", tag="elapsed_time_text")
        dpg.add_progress_bar(
            tag="packet_rate_bar",
            default_value=0.0,
            overlay="Packet rate: 0.0 Hz",
            width=300,
        )

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

            vec_x_color = (0, 186, 188, 255)
            vec_y_color = (0, 120, 155, 255)
            vec_z_color = (255, 145, 0, 255)

            dpg.add_line_series(
                [], [], label="X", parent=y_axis, tag="vec_x_series", color=vec_x_color
            )
            dpg.add_line_series(
                [], [], label="Y", parent=y_axis, tag="vec_y_series", color=vec_y_color
            )
            dpg.add_line_series(
                [], [], label="Z", parent=y_axis, tag="vec_z_series", color=vec_z_color
            )

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Theme helpers for the status indicator
    def _make_indicator_theme(tag: str, color: tuple[int, int, int, int]):
        with dpg.theme(tag=tag):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, color)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, color)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, color)

    _make_indicator_theme("status_theme_green", (0, 200, 0, 255))
    _make_indicator_theme("status_theme_yellow", (230, 180, 0, 255))
    _make_indicator_theme("status_theme_red", (200, 0, 0, 255))

    def _status_theme_for_message(message: str) -> str:
        lowered = message.lower()
        if "dropped" in lowered or "error" in lowered:
            return "status_theme_red"
        if "retry" in lowered or "sequence reset" in lowered:
            return "status_theme_yellow"
        if "connected" in lowered:
            return "status_theme_green"
        return "status_theme_red"

    # Manual render loop so we can update each frame
    while dpg.is_dearpygui_running():
        with quat_lock:
            status = connection_status
            direction_vec = np.array(current_direction_vec, copy=True)
            quat_vals = np.array(current_quat, copy=True)
            euler_vals = np.array(current_euler_deg, copy=True)
            acc_vals = np.array(current_acc_g, copy=True)
            gyro_vals = np.array(current_gyro_dps, copy=True)
            vec_x_vals = list(vec_x_hist)
            vec_y_vals = list(vec_y_hist)
            vec_z_vals = list(vec_z_hist)
            time_vals = list(sample_time_hist)
            interval_ms = last_sample_interval_ms
            seq = sample_index
            elapsed = elapsed_time_s

        dpg.set_value("status_text", status)
        dpg.bind_item_theme("status_indicator", _status_theme_for_message(status))
        dpg.set_value("vec_x_text", f"X: {direction_vec[0]:7.3f}")
        dpg.set_value("vec_y_text", f"Y: {direction_vec[1]:7.3f}")
        dpg.set_value("vec_z_text", f"Z: {direction_vec[2]:7.3f}")
        dpg.set_value("quat_w_text", f"Quat w: {quat_vals[0]:7.3f}")
        dpg.set_value("quat_x_text", f"Quat x: {quat_vals[1]:7.3f}")
        dpg.set_value("quat_y_text", f"Quat y: {quat_vals[2]:7.3f}")
        dpg.set_value("quat_z_text", f"Quat z: {quat_vals[3]:7.3f}")
        dpg.set_value("roll_text", f"Roll (deg):  {euler_vals[0]:7.2f}")
        dpg.set_value("pitch_text", f"Pitch (deg): {euler_vals[1]:7.2f}")
        dpg.set_value("yaw_text", f"Yaw (deg):   {euler_vals[2]:7.2f}")
        dpg.set_value("acc_x_text", f"Accel X: {acc_vals[0]:7.3f} g")
        dpg.set_value("acc_y_text", f"Accel Y: {acc_vals[1]:7.3f} g")
        dpg.set_value("acc_z_text", f"Accel Z: {acc_vals[2]:7.3f} g")
        dpg.set_value("gyro_x_text", f"Gyro X: {gyro_vals[0]:7.3f} dps")
        dpg.set_value("gyro_y_text", f"Gyro Y: {gyro_vals[1]:7.3f} dps")
        dpg.set_value("gyro_z_text", f"Gyro Z: {gyro_vals[2]:7.3f} dps")

        if interval_ms is None or interval_ms <= 0:
            interval_text = "Last interval: n/a"
            rate_hz = 0.0
        else:
            rate_hz = 1000.0 / interval_ms if interval_ms > 0 else 0.0
            interval_text = f"Last interval: {interval_ms:6.1f} ms (~{rate_hz:5.1f} Hz)"
        dpg.set_value("sample_interval_text", interval_text)
        if seq:
            dpg.set_value("sample_sequence_text", f"Sequence: {seq}")
        else:
            dpg.set_value("sample_sequence_text", "Sequence: n/a")
        dpg.set_value("elapsed_time_text", f"Elapsed time: {elapsed:7.2f} s")
        progress_value = min(rate_hz / EXPECTED_PACKET_RATE_HZ, 1.0) if EXPECTED_PACKET_RATE_HZ > 0 else 0.0
        dpg.set_value("packet_rate_bar", progress_value)
        dpg.configure_item("packet_rate_bar", overlay=f"Packet rate: {rate_hz:5.1f} Hz")

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
    global selected_filter_name
    selected_filter_name = args.filter

    LOGGER.info(
        "Starting IMUpen GUI (port=%s, baud=%d, history=%d)",
        args.serial_port,
        args.baud_rate,
        HISTORY_LENGTH,
    )

    worker = threading.Thread(
        target=serial_worker,
        args=(args.serial_port, args.baud_rate, selected_filter_name),
        daemon=True,
    )
    worker.start()
    try:
        run_gui(selected_filter_name)
    finally:
        stop_event.set()
        worker.join(timeout=1.0)


if __name__ == "__main__":
    main()
