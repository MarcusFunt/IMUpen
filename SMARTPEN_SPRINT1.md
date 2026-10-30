# SmartPen AI: sprint 1 quick start

The existing XIAO MG24 Sense AHRS viewer remains unchanged. This directory
adds an **independent** ESP32-S3 + standalone LSM6DS3 handwriting capture
pipeline. It does not yet include handwriting recognition. A minimal optional
PySide6 prompt-and-record GUI is included; live plotting is future work.

## Before connecting the sensor

The generic ESP32-S3 DevKitC-1 board target and GPIO8/9 in
firmware/platformio.ini are placeholders. Confirm the exact ESP32-S3 board,
its native USB interface, and which SDA/SCL pins are wired. Connect the
breakout to 3.3-V-compatible power and logic, never assume 5-V tolerance.
For I2C, follow your breakout's CS/SA0 wiring instructions.

## Install

Python 3.10+:

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # Linux/macOS: source .venv/bin/activate
    pip install -e ".[test]"
    pytest -q

PlatformIO:

    cd firmware
    pio run
    pio run -t upload
    cd ..

The firmware sends only binary packets on the USB CDC port; do not expect
human-readable text in a serial monitor.

## Collect handwriting

    smartpen-collect --port COM6 --writer p01 --session day01 \
      --prompts prompts/demo.txt --output datasets

Replace COM6 with your serial port. The command displays each prompt, lets
you press Enter to start/stop writing on paper, then accept/reject the sample.
USB CDC may reset the board when the port opens; wait for it to start.

The capture loop runs in a background thread so terminal input does not
block serial reading. The host saves **all received raw bytes**, including
corruption, and separately parses valid frames. Each recording produces:

- datasets/raw/day01/<id>.bin: original USB byte stream.
- datasets/processed/day01/<id>.npz: int16 axes [T,6], uint32 sequence,
  uint64 nominal timestamps, uint16 flags.
- datasets/metadata/day01/<id>.json: prompt, writer/session, device
  configuration if observed, quality diagnostics, acceptance.
- datasets/manifests/day01.jsonl: searchable recording index.

Rejected recordings remain marked as rejected and must not be used for
training. The device periodically sends its configuration; if a very short
recording lacks a config packet, the collector uses the latest config seen
while idle. If no config was received, metadata records null; do not assume
sensor units without validating configuration.

## Optional graphical recorder

Install the GUI extra and launch the prompt/record/accept/reject window:

    pip install -e ".[gui]"
    smartpen-gui --port COM6 --writer p01 --session day01 \
      --prompts prompts/demo.txt --output datasets

This is a basic prompted interface. It does not yet plot the six sensor axes.

## Replay

    smartpen-replay datasets/raw/day01/<id>.bin
    smartpen-replay datasets/raw/day01/<id>.bin --csv sample.csv

## Validation limitations

The hardware has not yet been flashed or tested against your actual ESP32-S3
and standalone breakout. The firmware uses an LSM6DS3 WHO_AM_I=0x69
check, FIFO at nominal 208 Hz and 400-kHz I2C. Its sample timestamps are
**nominally reconstructed**, not sampled from a hardware time register.
FIFO overflow and I2C errors mark discontinuities; USB transmit drops
appear as gaps in sequence numbers. A USB disconnect can lose an unknown
amount of FIFO data.

Before collecting the real dataset, check sampling, orientation, FIFO order,
gyro/accel scale, signal clipping, and loss counters over a long recording.

## Next sprint

Live sensor plots and a data quality dashboard, calibration,
resampling, training/validation/test split manifests, then CNN-BiLSTM-CTC.
