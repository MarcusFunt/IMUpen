"""
xiao_mg24_fifo_gui.py

Graphical user interface for collecting IMU data from the Seeed Studio
XIAO MG24 Sense board over USB.  The companion Arduino sketch
``xiao_mg24_fifo_usb.ino`` implements a simple command protocol that
allows the host to trigger recording of a block of accelerometer and
gyroscope samples using the LSM6DS3TR-C’s FIFO.  This script uses
``pyserial`` to communicate with the microcontroller and
``dearpygui`` to present a minimalistic control panel and plots.

Features
--------

* Enumerates available serial ports and lets the user select one.
* Allows configuration of the sample rate (in Hz) and number of samples
  to record.  These values are sent to the Arduino using a ``START``
  command.
* Displays status messages and a progress bar while data is being
  captured.
* Plots the accelerometer and gyroscope data for the X, Y and Z axes
  once acquisition is complete.
* Provides a button to save the most recent data set to a CSV file.

Dependencies
------------

Install the following Python packages via ``pip`` before running
``xiao_mg24_fifo_gui.py``::

    pip install pyserial dearpygui

Usage
-----

Run the script on a computer connected to the XIAO MG24 Sense via USB.
Select the appropriate serial port from the drop–down list, choose
desired recording parameters and press **Start**.  After the board
responds with the recorded data the plots will update and you can
export the CSV file by clicking **Save CSV**.

This example is released under the MIT license.  See accompanying
Arduino sketch for firmware implementation details.
"""

import threading
import time
import csv
from typing import List, Tuple

import serial
from serial.tools import list_ports

import dearpygui.dearpygui as dpg


class IMURecorder:
    """Handles communication with the Arduino based IMU logger.

    The recorder opens a serial port, sends commands to start and stop
    recordings and parses the CSV formatted output produced by the
    ``xiao_mg24_fifo_usb.ino`` firmware.  It stores the most recently
    recorded data so that the GUI can update the plots and write to
    disk.
    """

    def __init__(self):
        self.ser: serial.Serial | None = None
        self.data: List[Tuple[int, float, float, float, float, float, float]] = []
        self.thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.Lock()

    def open(self, port: str, baud: int = 115200) -> None:
        """Open the given serial port.

        If a port is already open it will be closed first.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = serial.Serial(port, baudrate=baud, timeout=0.1)
        # flush any leftover data
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def close(self) -> None:
        """Close the serial port if it is open."""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None

    def send_command(self, cmd: str) -> None:
        """Send a textual command to the board followed by a newline."""
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port not open")
        # Ensure newline termination
        line = cmd.strip() + "\n"
        self.ser.write(line.encode('ascii'))

    def start_recording(self, rate: int, num_samples: int, status_callback=None) -> None:
        """Begin a recording session in a background thread.

        :param rate: Sample rate in Hz to request from the firmware.
        :param num_samples: Number of samples to collect.
        :param status_callback: Optional callback receiving status strings.
        """
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Recording already in progress")
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Serial port not open")
        # Clear previous data
        with self.lock:
            self.data = []
        # send the command
        cmd = f"START,{rate},{num_samples}"
        self.send_command(cmd)
        # Launch thread
        self.running = True
        self.thread = threading.Thread(
            target=self._reader_thread,
            args=(status_callback,),
            daemon=True
        )
        self.thread.start()

    def stop_recording(self) -> None:
        """Abort a recording in progress."""
        if not self.ser or not self.ser.is_open:
            return
        self.send_command("STOP")
        self.running = False

    def _reader_thread(self, status_callback=None) -> None:
        """Internal method run in a separate thread to receive data.

        It waits for the ``DATA_START`` marker from the device, then
        reads lines until ``DATA_END`` is encountered.  Parsed rows are
        stored in ``self.data``.  The optional status callback is
        invoked periodically with progress updates.
        """
        assert self.ser is not None
        # Wait for DATA_START
        header_found = False
        start_time = time.time()
        # Provide feedback while waiting for data
        if status_callback:
            status_callback("Waiting for device to respond...")
        line_bytes: bytes
        # We'll treat the serial port as text lines separated by \n
        while self.running:
            line_bytes = self.ser.readline()
            if not line_bytes:
                # Timeout; update status occasionally
                if status_callback and (time.time() - start_time) > 0.5:
                    status_callback("Awaiting DATA_START marker...")
                    start_time = time.time()
                continue
            try:
                line = line_bytes.decode('ascii').strip()
            except UnicodeDecodeError:
                # skip garbage
                continue
            if line == "DATA_START":
                header_found = True
                if status_callback:
                    status_callback("Receiving data...")
                break
        if not header_found:
            # aborted
            if status_callback:
                status_callback("No data received")
            self.running = False
            return
        # Next line should be header; ignore or parse for column names
        header = self.ser.readline().decode('ascii').strip()
        # Acquire data rows
        rows: List[Tuple[int, float, float, float, float, float, float]] = []
        sample_count = 0
        while self.running:
            line_bytes = self.ser.readline()
            if not line_bytes:
                continue
            line = line_bytes.decode('ascii').strip()
            if line == "DATA_END":
                break
            # Parse comma separated values: timestamp_ms,acc_x_g,acc_y_g,acc_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps
            parts = line.split(',')
            if len(parts) != 7:
                continue
            try:
                ts = int(parts[0])
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
                gx = float(parts[4])
                gy = float(parts[5])
                gz = float(parts[6])
            except ValueError:
                # skip malformed lines
                continue
            rows.append((ts, ax, ay, az, gx, gy, gz))
            sample_count += 1
            # Provide occasional progress updates (every 10 samples)
            if status_callback and sample_count % 10 == 0:
                status_callback(f"Received {sample_count} samples...")
        # Save the data
        with self.lock:
            self.data = rows
        self.running = False
        if status_callback:
            status_callback(f"Recording complete: {len(rows)} samples")

    def get_data_arrays(self) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float], List[float]]:
        """Return the recorded data as separate lists for plotting."""
        with self.lock:
            ts = [row[0] for row in self.data]
            ax = [row[1] for row in self.data]
            ay = [row[2] for row in self.data]
            az = [row[3] for row in self.data]
            gx = [row[4] for row in self.data]
            gy = [row[5] for row in self.data]
            gz = [row[6] for row in self.data]
        return ts, ax, ay, az, gx, gy, gz

    def save_to_csv(self, path: str) -> None:
        """Write the most recently recorded data to a CSV file."""
        with self.lock:
            rows = list(self.data)
        if not rows:
            raise RuntimeError("No data available to save")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_ms', 'acc_x_g', 'acc_y_g', 'acc_z_g',
                'gyro_x_dps', 'gyro_y_dps', 'gyro_z_dps'
            ])
            writer.writerows(rows)


# Instantiate the recorder
recorder = IMURecorder()


def refresh_port_list() -> List[str]:
    """Return a list of available serial port device names."""
    return [port.device for port in list_ports.comports()]


def on_refresh_ports(sender, app_data, user_data):
    """Callback triggered when the user presses the refresh button."""
    ports = refresh_port_list()
    dpg.configure_item('port_combo', items=ports)
    if ports:
        dpg.set_value('port_combo', ports[0])


def on_connect(sender, app_data, user_data):
    """Open or close the serial port when the connect button is pressed."""
    port = dpg.get_value('port_combo')
    connected = dpg.get_value('connect_button')  # toggled value
    if connected:
        try:
            recorder.open(port)
            dpg.set_value('status_text', f"Opened {port}")
        except Exception as e:
            dpg.set_value('status_text', f"Failed to open {port}: {e}")
            dpg.set_value('connect_button', False)  # revert toggle
    else:
        recorder.close()
        dpg.set_value('status_text', "Port closed")


def on_start(sender, app_data, user_data):
    """Start recording using parameters from the UI."""
    if not recorder.ser or not recorder.ser.is_open:
        dpg.set_value('status_text', "Serial port not open")
        return
    # Validate parameters
    rate = dpg.get_value('rate_input')
    samples = dpg.get_value('samples_input')
    if rate <= 0 or samples <= 0:
        dpg.set_value('status_text', "Rate and samples must be positive")
        return
    # Disable start button to prevent duplicates
    dpg.configure_item('start_button', enabled=False)
    dpg.set_value('status_text', f"Starting recording: {samples} samples at {rate} Hz")
    dpg.set_value('progress_bar', 0.0)
    dpg.configure_item('progress_bar', show=True)

    # Launch recording; pass a status callback that updates the progress bar
    def status_callback(msg: str):
        dpg.set_value('status_text', msg)
        # update progress bar if we know approximate number of samples
        if msg.startswith("Received"):
            try:
                count = int(msg.split()[1])
                dpg.set_value('progress_bar', min(1.0, count / samples))
            except (ValueError, IndexError):
                pass
        elif msg.startswith("Recording complete"):
            # ensure progress bar is full
            dpg.set_value('progress_bar', 1.0)

    try:
        recorder.start_recording(rate, samples, status_callback=status_callback)

        # Poll until the recording thread finishes, then update UI
        def monitor_thread():
            while recorder.thread and recorder.thread.is_alive():
                time.sleep(0.1)
            # After finishing, update plots and enable UI
            update_plots()
            dpg.configure_item('start_button', enabled=True)
            dpg.configure_item('progress_bar', show=False)
            # Enable save button only if data exists
            has_data = len(recorder.data) > 0
            dpg.configure_item('save_button', enabled=has_data)

        # Start monitor in another thread to avoid blocking dearpygui event loop
        threading.Thread(target=monitor_thread, daemon=True).start()
    except Exception as e:
        dpg.set_value('status_text', f"Error: {e}")
        dpg.configure_item('start_button', enabled=True)


def update_plots() -> None:
    """Refresh the line series with the latest data."""
    ts, ax, ay, az, gx, gy, gz = recorder.get_data_arrays()
    if not ts:
        return
    # Convert timestamps from ms to seconds relative to first sample
    t0 = ts[0]
    t = [(v - t0) / 1000.0 for v in ts]
    # Update accelerometer plot
    dpg.set_value('accel_x_series', [t, ax])
    dpg.set_value('accel_y_series', [t, ay])
    dpg.set_value('accel_z_series', [t, az])
    # Update gyroscope plot
    dpg.set_value('gyro_x_series', [t, gx])
    dpg.set_value('gyro_y_series', [t, gy])
    dpg.set_value('gyro_z_series', [t, gz])


def on_save_csv(sender, app_data, user_data):
    """Open a file save dialog and write the data to CSV."""
    # Use dearpygui's built-in file dialog
    def save_callback(sender, app_data):
        # app_data contains {'file_path_name': ..., 'file_name': ..., 'directory': ..., 'raw_path': ...}
        file_path = app_data['file_path_name']
        try:
            recorder.save_to_csv(file_path)
            dpg.set_value('status_text', f"Saved data to {file_path}")
        except Exception as e:
            dpg.set_value('status_text', f"Error saving CSV: {e}")
        dpg.configure_item('save_dialog', show=False)

    dpg.show_item('save_dialog')
    dpg.configure_item('save_dialog', callback=save_callback)


def on_exit(sender, app_data, user_data):
    """Handle closing of the window; ensure serial port is closed."""
    recorder.close()


def create_gui():
    """Construct the Dear PyGui interface."""
    dpg.create_context()
    dpg.create_viewport(title="XIAO MG24 IMU Recorder", width=800, height=600)

    with dpg.window(label="IMU Recorder", width=800, height=600, on_close=on_exit):
        # Port selection
        dpg.add_text("Serial Port:")
        dpg.add_combo(refresh_port_list(), tag='port_combo', width=150)
        dpg.add_button(label="Refresh", callback=on_refresh_ports)
        # Connect/disconnect toggle
        dpg.add_checkbox(label="Connect", tag='connect_button', callback=on_connect)
        dpg.add_separator()
        # Recording parameters
        dpg.add_input_int(label="Sample Rate (Hz)", default_value=104, min_value=1, tag='rate_input')
        dpg.add_input_int(label="Number of Samples", default_value=200, min_value=1, tag='samples_input')
        dpg.add_button(label="Start Recording", tag='start_button', callback=on_start)
        dpg.add_progress_bar(tag='progress_bar', default_value=0.0, show=False, width=300)
        dpg.add_separator()
        # Plots for accelerometer and gyroscope
        dpg.add_text("Accelerometer (g)")
        with dpg.plot(label="Accelerometer", height=200, width=750, tag='accel_plot'):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag='accel_x_axis')
            dpg.add_plot_axis(dpg.mvYAxis, label="Acceleration (g)", tag='accel_y_axis')
            dpg.set_axis_limits_auto('accel_x_axis')
            dpg.set_axis_limits_auto('accel_y_axis')
            # series for x, y, z
            dpg.add_line_series([], [], label='ax', parent='accel_y_axis', tag='accel_x_series')
            dpg.add_line_series([], [], label='ay', parent='accel_y_axis', tag='accel_y_series')
            dpg.add_line_series([], [], label='az', parent='accel_y_axis', tag='accel_z_series')

        dpg.add_text("Gyroscope (°/s)")
        with dpg.plot(label="Gyroscope", height=200, width=750, tag='gyro_plot'):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag='gyro_x_axis')
            dpg.add_plot_axis(dpg.mvYAxis, label="Angular Rate (°/s)", tag='gyro_y_axis')
            dpg.set_axis_limits_auto('gyro_x_axis')
            dpg.set_axis_limits_auto('gyro_y_axis')
            dpg.add_line_series([], [], label='gx', parent='gyro_y_axis', tag='gyro_x_series')
            dpg.add_line_series([], [], label='gy', parent='gyro_y_axis', tag='gyro_y_series')
            dpg.add_line_series([], [], label='gz', parent='gyro_y_axis', tag='gyro_z_series')

        # Save button
        dpg.add_button(label="Save CSV", tag='save_button', callback=on_save_csv, enabled=False)
        # Status text
        dpg.add_text("", tag='status_text')
        # File dialog hidden by default
        with dpg.file_dialog(directory_selector=False, show=False, callback=None, tag='save_dialog', width=400, height=300):
            dpg.add_file_extension(".csv", color=(0, 255, 0, 255), custom_text="CSV files")

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == '__main__':
    create_gui()
