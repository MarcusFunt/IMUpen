# SmartPen AI: sprint 1 quick start

The repository now targets only the **Seeed XIAO ESP32-S3 with a standalone LSM6DS3**.
This sprint adds handwriting acquisition and prompted dataset collection; it does not
yet include handwriting recognition. A minimal optional PySide6 prompt-and-record
GUI is included; live plotting is future work.

## Before connecting the sensor

The PlatformIO target is now **seeed_xiao_esp32s3** and the standard exposed
I2C pins are configured: XIAO **D4/SDA = GPIO5**, **D5/SCL = GPIO6**.
Connect the breakout's VCC and GND to 3V3 and GND, respectively. Never
assume 5-V-tolerant sensor logic; follow your breakout's CS/SA0 instructions.

## Install

Python 3.10+:

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # Linux/macOS: source .venv/bin/activate
    pip install -e ".[test,gui]"
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

The hardware has not yet been flashed or tested against your XIAO ESP32-S3
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
