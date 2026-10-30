# IMUpen — XIAO ESP32-S3 + LSM6DS3 SmartPen

A personalized handwriting-data collection platform for a **Seeed Studio XIAO ESP32-S3** connected to a **standalone ST LSM6DS3 six-axis IMU**. The firmware reads an accelerometer and gyroscope, streams framed binary packets over native USB CDC, and the Python application records labeled handwriting samples for later machine-learning experiments.

## Hardware connections

Connect your standalone LSM6DS3 breakout to the XIAO ESP32-S3's exposed I2C pads:

| LSM6DS3 breakout | XIAO ESP32-S3 pad | Chip GPIO |
| --- | --- | --- |
| SDA | D4 / SDA | GPIO5 |
| SCL | D5 / SCL | GPIO6 |
| VCC | 3V3 | — |
| GND | GND | — |

Use **3.3 V logic**. Check the breakout's power requirements and whether it needs pull-up resistors. If it exposes CS, follow the module's instructions for I2C mode (typically CS high). SA0/SDO controls I2C address; firmware probes 0x6A and 0x6B.

The PlatformIO target and pin assignments are set to the XIAO ESP32-S3; no board-pin edits are needed for these connections.

## Quick start

Install Python 3.10+, PlatformIO, and dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[test,gui]"
python -m pytest -q
```

Build and flash the XIAO ESP32-S3:

```bash
cd firmware
pio run
pio run -t upload
cd ..
```

Connect to its USB CDC serial port (replace COM6 with the port on your computer) and start the graphical prompted handwriting recorder:

```bash
smartpen-gui --port COM6 --writer p01 --session day01 --prompts prompts/demo.txt
```

A terminal-only recorder is also available:

```bash
smartpen-collect --port COM6 --writer p01 --session day01 --prompts prompts/demo.txt
```

Review [SMARTPEN_SPRINT1.md](SMARTPEN_SPRINT1.md) for recording and replay instructions and [firmware/README.md](firmware/README.md) for the full USB binary protocol.

## What is implemented

- PlatformIO firmware for the XIAO ESP32-S3 and standalone LSM6DS3.
- FIFO-based 208 Hz IMU acquisition, 400 kHz I2C, configurable sensor ranges in firmware.
- Versioned, CRC16-checked binary USB frames with sample sequence counters and error telemetry.
- Prompted Python terminal and optional PySide6 graphical recording tools.
- Immutable raw binary streams and decoded NumPy arrays, plus JSON metadata and JSONL manifests.
- Optional six-axis CNN-BiLSTM-CTC baseline and residual CNN-BiLSTM-CTC handwriting recognition with OnHW-Words500 training, held-out writer evaluation, and CER/WER reporting.
- Host unit tests and GitHub Actions Python/firmware builds.

**Not yet implemented:** live plots, sensor calibration or automatic pen-contact detection.

## Handwriting recognition (M4/M5)

The recognizers are an optional install so the recorder does not need the PyTorch runtime. To train or evaluate, install `python -m pip install -e ".[ml]"`. Follow [ONHW_M4.md](ONHW_M4.md) for dataset and baseline details, and [M5_RESNET_BILSTM.md](M5_RESNET_BILSTM.md) for the residual architecture and a reproducible side-by-side comparison. The OnHW dataset is separate from the MIT-licensed code and is not stored in this repository.

## Hardware validation

Continuous data capture has not yet been tested against your physical XIAO and breakout. The sensor is accepted only when WHO_AM_I is 0x69. Validate the sensor variant, USB interface, FIFO order, signal scale, clipping, and dropped-sample counters before collecting your main dataset.

The 64-bit timestamps are reconstructed from the nominal 208 Hz sample grid; they are not timestamps directly measured by the IMU.

## License

MIT; see [LICENSE](LICENSE).
