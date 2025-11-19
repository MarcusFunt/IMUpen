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

import platform

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
PACKET_RATE_TARGET_HZ = 200.0

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


def resolve_default_font_path() -> str | None:
    """Return a platform-appropriate font path if one can be found."""

    candidates: list[str] = []
    system = platform.system().lower()
    if system == "windows":
        system_root = os.environ.get("SystemRoot", r"C:\\Windows")
        fonts_dir = os.path.join(system_root, "Fonts")
        candidates.extend(
            [
                os.path.join(fonts_dir, "segoeui.ttf"),
                os.path.join(fonts_dir, "arial.ttf"),
            ]
        )
    elif system == "darwin":
        candidates.extend(
            [
                "/System/Library/Fonts/SFNS.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
        )

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


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
    default_font = None
    font_path = resolve_default_font_path()
    if font_path:
        with dpg.font_registry():
            default_font = dpg.add_font(font_path, 16)
        dpg.bind_font(default_font)
    else:
        LOGGER.warning(
            "Unable to locate a default font for this platform;"
            " falling back to DearPyGui's built-in font."
        )

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
            # Plot specific colours live in the mvThemeCat_Plots category.  Use
            # the mvPlotCol_* enumerations so DearPyGui 2.x recognises them.
            dpg.add_theme_color(
                dpg.mvPlotCol_Line,
                plot_line_color,
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotLinesHovered,
                (255, 180, 80, 255),
                category=dpg.mvThemeCat_Core,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_PlotBorder,
                separator_color,
                category=dpg.mvThemeCat_Plots,
            )
            dpg.add_theme_color(
                dpg.mvPlotCol_Fill,
                plot_fill_color,
                category=dpg.mvThemeCat_Plots,
            )
    dpg.bind_theme(branded_theme)

    def make_status_theme(color: tuple[int, int, int, int]):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button, color, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    color,
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    color,
                    category=dpg.mvThemeCat_Core,
                )
        return theme

    def make_line_series_theme(color: tuple[int, int, int, int]):
        """Return a theme that applies ``color`` to a line series."""

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots
                )
        return theme

    status_themes = {
        "good": make_status_theme((0, 200, 0, 255)),
        "warn": make_status_theme((220, 180, 0, 255)),
        "bad": make_status_theme((220, 30, 30, 255)),
    }
    status_labels = {
        "good": "Connected",
        "warn": "Retrying",
        "bad": "Dropped",
    }

    dpg.create_viewport(
        title="XIAO MG24 Sense AHRS Viewer",
        width=900,
        height=600
    )

    def info_table(rows: list[tuple[str, str, str]]):
        """Helper to create a two column table for label/value rows."""

        table = dpg.add_table(
            header_row=False,
            policy=dpg.mvTable_SizingStretchProp,
            borders_innerH=True,
            borders_innerV=True,
            borders_outerH=True,
            borders_outerV=True,
        )
        dpg.add_table_column(parent=table, width_fixed=True)
        dpg.add_table_column(parent=table)
        for label, tag, default in rows:
            with dpg.table_row(parent=table):
                dpg.add_text(label)
                dpg.add_text(default, tag=tag)

    with dpg.window(label="IMU Orientation", tag="main_window", width=880, height=580):
        with dpg.group(horizontal=True):
            with dpg.child_window(label="Status", width=250, height=180):
                dpg.add_text("Status")
                info_table(
                    [
                        ("Connection", "status_text", "Not connected"),
                        ("Filter", "filter_text", f"{filter_name.upper()}"),
                    ]
                )
                dpg.add_spacer(height=4)
                dpg.add_text("Link health")
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="",
                        width=24,
                        height=24,
                        tag="status_indicator",
                        enabled=False,
                    )
                    dpg.add_text("Waiting", tag="status_indicator_label")
                dpg.add_spacer(height=4)
                dpg.add_text("Packet rate")
                dpg.add_progress_bar(
                    tag="packet_rate_bar",
                    default_value=0.0,
                    overlay="n/a",
                    width=-1,
                )

            with dpg.child_window(label="Direction", width=250, height=180):
                dpg.add_text("Direction vector")
                info_table(
                    [
                        ("X", "vec_x_text", "0.000"),
                        ("Y", "vec_y_text", "0.000"),
                        ("Z", "vec_z_text", "0.000"),
                    ]
                )

            with dpg.child_window(label="Orientation", width=340, height=180):
                dpg.add_text("Orientation details")
                info_table(
                    [
                        ("Quat w", "quat_w_text", "1.000"),
                        ("Quat x", "quat_x_text", "0.000"),
                        ("Quat y", "quat_y_text", "0.000"),
                        ("Quat z", "quat_z_text", "0.000"),
                        ("Roll (deg)", "roll_text", "0.0"),
                        ("Pitch (deg)", "pitch_text", "0.0"),
                        ("Yaw (deg)", "yaw_text", "0.0"),
                    ]
                )

        with dpg.group(horizontal=True):
            with dpg.child_window(label="Sensors", width=400, height=200):
                dpg.add_text("Raw sensor data")
                info_table(
                    [
                        ("Accel X (g)", "acc_x_text", "0.000"),
                        ("Accel Y (g)", "acc_y_text", "0.000"),
                        ("Accel Z (g)", "acc_z_text", "0.000"),
                        ("Gyro X (dps)", "gyro_x_text", "0.000"),
                        ("Gyro Y (dps)", "gyro_y_text", "0.000"),
                        ("Gyro Z (dps)", "gyro_z_text", "0.000"),
                    ]
                )

            with dpg.child_window(label="Timing", width=220, height=200):
                dpg.add_text("Timing diagnostics")
                info_table(
                    [
                        ("Last interval", "sample_interval_text", "n/a"),
                        ("Sequence", "sample_sequence_text", "n/a"),
                        ("Elapsed (s)", "elapsed_time_text", "0.0"),
                    ]
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
                [], [], label="X", parent=y_axis, tag="vec_x_series"
            )
            dpg.add_line_series(
                [], [], label="Y", parent=y_axis, tag="vec_y_series"
            )
            dpg.add_line_series(
                [], [], label="Z", parent=y_axis, tag="vec_z_series"
            )

            dpg.bind_item_theme("vec_x_series", make_line_series_theme(vec_x_color))
            dpg.bind_item_theme("vec_y_series", make_line_series_theme(vec_y_color))
            dpg.bind_item_theme("vec_z_series", make_line_series_theme(vec_z_color))

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Manual render loop so we can update each frame
    last_indicator_state = None

    def classify_status(status: str) -> str:
        text = status.lower()
        if "dropped" in text or "error" in text:
            return "bad"
        if "retrying" in text or "sequence reset" in text:
            return "warn"
        if "connected" in text:
            return "good"
        return "warn"

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
        dpg.set_value("vec_x_text", f"{direction_vec[0]:7.3f}")
        dpg.set_value("vec_y_text", f"{direction_vec[1]:7.3f}")
        dpg.set_value("vec_z_text", f"{direction_vec[2]:7.3f}")
        dpg.set_value("quat_w_text", f"{quat_vals[0]:7.3f}")
        dpg.set_value("quat_x_text", f"{quat_vals[1]:7.3f}")
        dpg.set_value("quat_y_text", f"{quat_vals[2]:7.3f}")
        dpg.set_value("quat_z_text", f"{quat_vals[3]:7.3f}")
        dpg.set_value("roll_text", f"{euler_vals[0]:7.2f}")
        dpg.set_value("pitch_text", f"{euler_vals[1]:7.2f}")
        dpg.set_value("yaw_text", f"{euler_vals[2]:7.2f}")
        dpg.set_value("acc_x_text", f"{acc_vals[0]:7.3f}")
        dpg.set_value("acc_y_text", f"{acc_vals[1]:7.3f}")
        dpg.set_value("acc_z_text", f"{acc_vals[2]:7.3f}")
        dpg.set_value("gyro_x_text", f"{gyro_vals[0]:7.3f}")
        dpg.set_value("gyro_y_text", f"{gyro_vals[1]:7.3f}")
        dpg.set_value("gyro_z_text", f"{gyro_vals[2]:7.3f}")

        if interval_ms is None or interval_ms <= 0:
            interval_text = "n/a"
        else:
            rate_hz = 1000.0 / interval_ms if interval_ms > 0 else 0.0
            interval_text = f"{interval_ms:6.1f} ms (~{rate_hz:5.1f} Hz)"
        dpg.set_value("sample_interval_text", interval_text)
        if seq:
            dpg.set_value("sample_sequence_text", str(seq))
        else:
            dpg.set_value("sample_sequence_text", "n/a")
        dpg.set_value("elapsed_time_text", f"{elapsed:7.2f}")

        indicator_state = classify_status(status)
        if indicator_state != last_indicator_state:
            dpg.bind_item_theme("status_indicator", status_themes[indicator_state])
            last_indicator_state = indicator_state
        dpg.set_value(
            "status_indicator_label",
            status_labels.get(indicator_state, "Status"),
        )

        if interval_ms is None or interval_ms <= 0:
            dpg.set_value("packet_rate_bar", 0.0)
            dpg.configure_item("packet_rate_bar", overlay="n/a")
        else:
            rate_hz = 1000.0 / interval_ms if interval_ms > 0 else 0.0
            normalized = min(rate_hz / PACKET_RATE_TARGET_HZ, 1.0)
            dpg.set_value("packet_rate_bar", normalized)
            dpg.configure_item("packet_rate_bar", overlay=f"{rate_hz:5.1f} Hz")

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
