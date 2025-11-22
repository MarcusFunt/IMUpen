"""Simple GUI for visualising AHRS output from the XIAO MG24 Sense board."""

from __future__ import annotations

import argparse
import logging
import os
import queue
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
sample_queue: queue.Queue["FilteredSample"] = queue.Queue()

# New global queue used to send commands from the GUI to the serial worker.
command_queue: queue.Queue[str] = queue.Queue()


IMU_DRAW_TAG = "imu_drawlist"
ACC_TRANSLATION_SCALE = 0.05  # how much to move the stick per g of accel
ACC_TRANSLATION_LIMIT = 0.8  # clamp translation to keep the model visible
IMU_BODY_POINTS = np.array(
    [
        [0.0, 0.0, -0.7],
        [0.0, 0.0, 0.7],
    ],
    dtype=float,
)
IMU_AXIS_POINTS = {
    "x": np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=float),
    "y": np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=float),
    "z": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], dtype=float),
}

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


class FilteredSample(NamedTuple):
    timestamp_ms: int
    elapsed_s: float
    sequence: int
    quaternion: np.ndarray
    direction_vec: np.ndarray
    euler_deg: np.ndarray
    acc_g: np.ndarray
    gyro_dps: np.ndarray
    interval_ms: float
    reset: bool = False


def checksum_bytes(data: bytes) -> int:
    """Return a simple 16-bit additive checksum of ``data``."""

    return sum(data) & 0xFFFF


def extract_packets(buffer: bytearray) -> list[ImuPacket]:
    """Pull as many complete packets as possible out of ``buffer``."""

    packets: list[ImuPacket] = []
    tail_keep = max(len(PREAMBLE_BYTES) - 1, 0)
    while True:
        idx = buffer.find(PREAMBLE_BYTES)
        if idx == -1:
            if tail_keep:
                if len(buffer) > tail_keep:
                    del buffer[:-tail_keep]
                # otherwise leave ``buffer`` as-is so the partial preamble sticks around
            else:
                buffer.clear()
            break
        if len(buffer) - idx < PACKET_SIZE:
            break
        if idx > 0:
            del buffer[:idx]

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


def flush_sample_queue() -> None:
    """Remove all pending samples from the playback queue."""

    global sample_queue

    try:
        while True:
            sample_queue.get_nowait()
    except queue.Empty:
        return


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


def sanitize_quaternion(q: np.ndarray) -> np.ndarray:
    """Return a finite, normalized quaternion with a sensible fallback."""

    cleaned = np.array(q, dtype=float)
    if not np.all(np.isfinite(cleaned)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    norm = np.linalg.norm(cleaned)
    if norm <= 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    return cleaned / norm


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix for quaternion ``q``."""

    w, x, y, z = sanitize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


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


def accelerometer_to_translation(acc_vals: np.ndarray) -> np.ndarray:
    """Convert accelerometer readings (g) to a translation vector for the model."""

    scaled = np.array(acc_vals, dtype=float) * ACC_TRANSLATION_SCALE
    return np.clip(scaled, -ACC_TRANSLATION_LIMIT, ACC_TRANSLATION_LIMIT)


def transform_model_points(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply rotation + translation to ``points`` and return transformed copy."""

    return points @ rotation.T + translation


def project_points(points: np.ndarray, width: float, height: float) -> np.ndarray:
    """Project 3D ``points`` into 2D viewport coordinates."""

    if width <= 0 or height <= 0:
        width = 300.0
        height = 200.0

    fov_rad = np.radians(60.0)
    scale = 0.5 * min(width, height) / np.tan(fov_rad / 2.0)
    projected: list[tuple[float, float]] = []
    for x, y, z in points:
        z_cam = max(z + 3.0, 0.1)
        x_proj = (x / z_cam) * scale + width / 2.0
        y_proj = height / 2.0 - (y / z_cam) * scale
        projected.append((x_proj, y_proj))
    return np.array(projected, dtype=float)


def update_imu_visual(quat_vals: np.ndarray, acc_vals: np.ndarray) -> None:
    """Update the DearPyGui drawlist that shows the IMU stick."""

    if not dpg.does_item_exist(IMU_DRAW_TAG):
        return

    width, height = dpg.get_item_rect_size(IMU_DRAW_TAG)
    if width <= 1 or height <= 1:
        parent = dpg.get_item_parent(IMU_DRAW_TAG)
        if parent:
            parent_width, parent_height = dpg.get_item_rect_size(parent)
            if parent_width > 1:
                width = parent_width
            if parent_height > 1:
                height = min(parent_height, height if height > 1 else parent_height)
        if width <= 1:
            width = 320.0
        if height <= 1:
            height = 220.0

    dpg.configure_item(IMU_DRAW_TAG, width=width, height=height)
    rotation = quat_to_rotation_matrix(quat_vals)
    translation = accelerometer_to_translation(acc_vals)

    dpg.delete_item(IMU_DRAW_TAG, children_only=True)

    dpg.draw_rectangle(
        (0.0, 0.0),
        (width, height),
        color=(80, 80, 80, 90),
        fill=(15, 15, 20, 140),
        thickness=1.0,
        parent=IMU_DRAW_TAG,
    )

    body_points = transform_model_points(IMU_BODY_POINTS, rotation, translation)
    body_projected = project_points(body_points, width, height)
    dpg.draw_line(
        body_projected[0],
        body_projected[1],
        color=(0, 186, 188, 220),
        thickness=6.0,
        parent=IMU_DRAW_TAG,
    )

    axis_colors = {
        "x": (255, 95, 95, 255),
        "y": (120, 220, 120, 255),
        "z": (90, 170, 255, 255),
    }
    for axis, pts in IMU_AXIS_POINTS.items():
        transformed = transform_model_points(pts, rotation, translation)
        projected = project_points(transformed, width, height)
        dpg.draw_line(
            projected[0],
            projected[1],
            color=axis_colors.get(axis, (255, 255, 255, 255)),
            thickness=2.0,
            parent=IMU_DRAW_TAG,
        )

    pivot = transform_model_points(np.array([[0.0, 0.0, 0.0]], dtype=float), rotation, translation)
    pivot_projected = project_points(pivot, width, height)[0]
    dpg.draw_circle(
        center=pivot_projected,
        radius=6.0,
        color=(255, 255, 255, 120),
        fill=(0, 0, 0, 60),
        parent=IMU_DRAW_TAG,
    )


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


def serial_worker(serial_port: str, baud_rate: int, filter_name: str, cmd_queue: queue.Queue[str] | None = None):
    """
    Background thread:
      - opens serial port
      - parses binary packets: sequence, timestamp, accel (g), gyro (deg/s)
      - runs the requested AHRS filter
      - pushes filtered samples into a thread-safe queue for the GUI thread
      - writes outbound control commands (sample period, buffer length) from
        ``cmd_queue`` to the device.
    """
    global connection_status, first_sample_time_ms, last_sample_interval_ms
    global sample_queue

    filter_runner = create_filter_runner(filter_name)
    filter_label = filter_name.upper()
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    last_t_ms = None
    last_sequence = None
    reset_direction = np.array([0.0, 0.0, -1.0], dtype=float)

    def push_reset_sample(timestamp_ms: int, sequence: int) -> None:
        sample_queue.put(
            FilteredSample(
                timestamp_ms=timestamp_ms,
                elapsed_s=0.0,
                sequence=sequence,
                quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                direction_vec=np.array(reset_direction, copy=True),
                euler_deg=np.array([0.0, 0.0, 0.0], dtype=float),
                acc_g=np.zeros(3, dtype=float),
                gyro_dps=np.zeros(3, dtype=float),
                interval_ms=0.0,
                reset=True,
            )
        )

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
        flush_sample_queue()

        while not stop_event.is_set():
            try:
                # send any queued commands to the device
                if cmd_queue is not None:
                    try:
                        while True:
                            cmd = cmd_queue.get_nowait()
                            # send command as ASCII followed by newline
                            if cmd:
                                ser.write(cmd.encode("utf-8") + b"\n")
                    except queue.Empty:
                        pass

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

                reset_detected = False
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
                    flush_sample_queue()
                    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                    last_t_ms = None
                    last_sample_interval_ms = None
                    reset_detected = True
                last_sequence = packet.sequence
                status_message = (
                    f"Connected on {serial_port} (seq={packet.sequence}, filter={filter_label})"
                    f"{status_suffix}"
                )

                # LSM6DS3: accel in g, gyro in deg/s -> convert to SI units
                acc = np.array([ax_g, ay_g, az_g], dtype=float) * 9.80665
                gyr = np.radians(np.array([gx_dps, gy_dps, gz_dps], dtype=float))

                if reset_detected:
                    push_reset_sample(t_ms, packet.sequence)
                    with quat_lock:
                        connection_status = status_message
                    continue

                if last_t_ms is None:
                    last_t_ms = t_ms
                    first_sample_time_ms = t_ms
                    continue

                dt = (t_ms - last_t_ms) / 1000.0
                if dt <= 0:
                    LOGGER.debug("Ignoring non-positive dt=%s", dt)
                    last_t_ms = t_ms
                    first_sample_time_ms = t_ms
                    flush_sample_queue()
                    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                    push_reset_sample(t_ms, packet.sequence)
                    continue
                last_t_ms = t_ms

                # Run AHRS filter update; dt passed explicitly per sample
                q = filter_runner(q, gyr=gyr, acc=acc, dt=dt)
                if q is None:
                    LOGGER.debug("Filter update returned None")
                    continue

                q = sanitize_quaternion(q)

                direction_vec = quat_to_direction_vector(q)
                euler_deg = quat_to_euler_degrees(q)
                elapsed_s = 0.0
                if first_sample_time_ms is not None:
                    elapsed_s = max(0.0, (t_ms - first_sample_time_ms) / 1000.0)
                last_sample_interval_ms = dt * 1000.0

                with quat_lock:
                    connection_status = status_message

                sample_queue.put(
                    FilteredSample(
                        timestamp_ms=t_ms,
                        elapsed_s=elapsed_s,
                        sequence=packet.sequence,
                        quaternion=np.array(q, copy=True),
                        direction_vec=np.array(direction_vec, copy=True),
                        euler_deg=np.array(euler_deg, copy=True),
                        acc_g=np.array([ax_g, ay_g, az_g], dtype=float),
                        gyro_dps=np.array([gx_dps, gy_dps, gz_dps], dtype=float),
                        interval_ms=dt * 1000.0,
                        reset=False,
                    )
                )

        flush_sample_queue()
        push_reset_sample(int(time.time() * 1000), 0)

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
      - provides controls for adjusting the sampling interval and history length
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

    playback_queue: deque[FilteredSample] = deque()
    playback_origin_device_ms: float | None = None
    playback_origin_host_s: float | None = None
    last_scheduled_time_ms: float | None = None
    last_applied_sample: FilteredSample | None = None

    def reset_playback_state() -> None:
        nonlocal playback_origin_device_ms, playback_origin_host_s
        nonlocal last_scheduled_time_ms, last_applied_sample

        playback_queue.clear()
        playback_origin_device_ms = None
        playback_origin_host_s = None
        last_scheduled_time_ms = None
        last_applied_sample = None

    def blend_arrays(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * a + alpha * b

    def interpolate_samples(
        s0: FilteredSample, s1: FilteredSample, alpha: float
    ) -> FilteredSample:
        blended_quat = blend_arrays(s0.quaternion, s1.quaternion, alpha)
        norm = np.linalg.norm(blended_quat)
        if norm > 0:
            blended_quat = blended_quat / norm

        return FilteredSample(
            timestamp_ms=int((1.0 - alpha) * s0.timestamp_ms + alpha * s1.timestamp_ms),
            elapsed_s=(1.0 - alpha) * s0.elapsed_s + alpha * s1.elapsed_s,
            sequence=s0.sequence if alpha < 0.5 else s1.sequence,
            quaternion=blended_quat,
            direction_vec=blend_arrays(s0.direction_vec, s1.direction_vec, alpha),
            euler_deg=blend_arrays(s0.euler_deg, s1.euler_deg, alpha),
            acc_g=blend_arrays(s0.acc_g, s1.acc_g, alpha),
            gyro_dps=blend_arrays(s0.gyro_dps, s1.gyro_dps, alpha),
            interval_ms=(1.0 - alpha) * s0.interval_ms + alpha * s1.interval_ms,
            reset=False,
        )

    def apply_scheduled_sample(target_ms: float) -> None:
        nonlocal playback_origin_device_ms, playback_origin_host_s
        nonlocal last_scheduled_time_ms, last_applied_sample

        while True:
            try:
                pending = sample_queue.get_nowait()
            except queue.Empty:
                break

            if pending.reset:
                reset_playback_state()
                playback_origin_device_ms = pending.timestamp_ms
                playback_origin_host_s = time.monotonic()

            playback_queue.append(pending)

        if playback_origin_device_ms is None and playback_queue:
            playback_origin_device_ms = playback_queue[0].timestamp_ms
            playback_origin_host_s = time.monotonic()

        if playback_origin_device_ms is None or playback_origin_host_s is None:
            return

        while playback_queue and playback_queue[0].timestamp_ms <= target_ms:
            last_applied_sample = playback_queue.popleft()

        next_sample = playback_queue[0] if playback_queue else None
        scheduled_sample = last_applied_sample or next_sample
        if scheduled_sample is None:
            return

        if last_applied_sample and next_sample:
            denom = max(
                1e-9, next_sample.timestamp_ms - last_applied_sample.timestamp_ms
            )
            alpha = (target_ms - last_applied_sample.timestamp_ms) / denom
            alpha = float(np.clip(alpha, 0.0, 1.0))
            scheduled_sample = interpolate_samples(last_applied_sample, next_sample, alpha)

        last_applied_sample = scheduled_sample

        scheduled_elapsed_ms = target_ms - playback_origin_device_ms
        scheduled_elapsed_s = max(0.0, scheduled_elapsed_ms / 1000.0)

        interval_ms = None
        if last_scheduled_time_ms is not None:
            interval_ms = max(0.0, target_ms - last_scheduled_time_ms)
        last_scheduled_time_ms = target_ms

        with quat_lock:
            global current_quat, current_direction_vec, current_euler_deg
            global current_acc_g, current_gyro_dps, sample_index, elapsed_time_s
            global last_sample_interval_ms

            current_quat = np.array(scheduled_sample.quaternion, copy=True)
            current_direction_vec = np.array(scheduled_sample.direction_vec, copy=True)
            current_euler_deg = np.array(scheduled_sample.euler_deg, copy=True)
            current_acc_g = np.array(scheduled_sample.acc_g, copy=True)
            current_gyro_dps = np.array(scheduled_sample.gyro_dps, copy=True)
            vec_x_hist.append(scheduled_sample.direction_vec[0])
            vec_y_hist.append(scheduled_sample.direction_vec[1])
            vec_z_hist.append(scheduled_sample.direction_vec[2])
            sample_time_hist.append(scheduled_elapsed_s)
            sample_index = scheduled_sample.sequence
            elapsed_time_s = scheduled_elapsed_s
            last_sample_interval_ms = interval_ms

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

            # New configuration panel: allows adjusting the sampling period and history length
            with dpg.child_window(label="Configuration", width=250, height=200):
                dpg.add_text("Controls")
                # Callback to update sampling period on the device
                def on_sample_period_change(sender, app_data, user_data):
                    try:
                        value = int(app_data)
                        if value > 0:
                            # send command to device
                            command_queue.put(f"PERIOD{value}")
                    except Exception:
                        pass
                # Callback to update history length both on host and device
                def on_history_length_change(sender, app_data, user_data):
                    try:
                        value = int(app_data)
                        if value < 1:
                            value = 1
                        configure_history_length(value)
                        # send command to device
                        command_queue.put(f"BUFFER{value}")
                    except Exception:
                        pass
                dpg.add_input_int(
                    label="Sample period (ms)",
                    default_value=int(1000.0 / PACKET_RATE_TARGET_HZ),
                    min_value=1,
                    max_value=1000,
                    callback=on_sample_period_change,
                    width=140,
                )
                dpg.add_input_int(
                    label="History length",
                    default_value=HISTORY_LENGTH,
                    min_value=1,
                    max_value=10000,
                    callback=on_history_length_change,
                    width=140,
                )

        with dpg.child_window(label="IMU 3D View", width=-1, height=260):
            dpg.add_text("IMU pose (accelerometer offset + roll/pitch/yaw)")
            dpg.add_spacer(height=4)
            dpg.add_drawlist(
                width=-1,
                height=210,
                tag=IMU_DRAW_TAG,
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
        if playback_origin_host_s is None:
            target_ms = playback_origin_device_ms or 0.0
        else:
            target_ms = (
                (playback_origin_device_ms or 0.0)
                + (time.monotonic() - playback_origin_host_s) * 1000.0
            )
        apply_scheduled_sample(target_ms)

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
        update_imu_visual(quat_vals, acc_vals)

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

    # Launch the serial worker thread and pass the command queue so outbound
    # commands issued from the GUI can reach the microcontroller.
    worker = threading.Thread(
        target=serial_worker,
        args=(args.serial_port, args.baud_rate, selected_filter_name, command_queue),
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