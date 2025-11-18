# IMUpen

Real-time AHRS viewer for the Seeed Studio XIAO MG24 Sense. The project is made
up of two pieces:

1. `xiao_mg24_imu_stream.ino` – an Arduino sketch that powers up the onboard
   LSM6DS3TR-C IMU and streams accelerometer/gyroscope samples as binary packets
   over USB.
2. `xiao_mg24_ahrs_gui.py` – a Python utility that ingests the binary stream,
   filters it with the Madgwick AHRS algorithm, and visualises the resulting
   direction vector using DearPyGui.

The repo now also contains a lightweight CI workflow and a modern `.gitignore`
to make contributing a little smoother.

## Requirements

* Python 3.10+
* Arduino IDE (or PlatformIO) for flashing the sketch
* System dependencies for [DearPyGui](https://github.com/hoffstadt/DearPyGui)

Install the Python dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Flash the `xiao_mg24_imu_stream.ino` sketch onto the XIAO MG24 Sense.
2. Identify the serial port that the board enumerated as (e.g. `/dev/ttyACM0` on
   Linux, `COM6` on Windows).
3. Run the GUI:

```bash
python xiao_mg24_ahrs_gui.py --serial-port /dev/ttyACM0
```

Command-line options:

* `--serial-port` – port name; defaults to the value of
  `IMUPEN_SERIAL_PORT` (or `COM6`).
* `--baud-rate` – serial speed; must match the sketch (default `115200`).
* `--history-length` – number of samples retained for the rolling plot.
* `-v/--verbose` – enable debug logging.

## Serial packet format

The sketch emits fixed-length, 38-byte, little-endian packets to minimise
latency and allow the host to recover from framing errors. Each packet contains:

| Field          | Type     | Notes                                        |
| -------------- | -------- | -------------------------------------------- |
| `preamble`     | `uint16` | Constant `0xAA55` for synchronisation        |
| `version`      | `uint16` | Protocol version (`1`)                       |
| `sequence`     | `uint32` | Monotonic counter for dropout detection      |
| `timestamp_ms` | `uint32` | `millis()` when the sample was captured      |
| `ax_g` … `az_g`| `float`  | Acceleration in `g`                          |
| `gx_dps` … `gz_dps` | `float` | Angular rate in degrees per second     |
| `checksum`     | `uint16` | Sum of all previous bytes modulo `2^16`      |

The Python GUI re-syncs to the preamble automatically and validates the checksum
and version before passing each sample to the Madgwick filter.

## Development

* Format and linting is intentionally lightweight; CI simply ensures that the
  Python files compile with the current interpreter.
* Run `python -m compileall xiao_mg24_ahrs_gui.py` locally before committing to
  mirror the CI check.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.
