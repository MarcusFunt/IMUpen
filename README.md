# IMUpen FIFO acquisition and measurements

The sketch collects gyro and accelerometer data from one XIAO ESP32-S3 and one LSM6DS3 over 4-wire SPI. The host tool records the stream, estimates timing, and can exercise serial backpressure and reconnection automatically. Physical measurements still require the wired sensor; logic-analyzer measurements require a compatible analyzer.

## Set up and execute the validation plan

Start with [START_HERE.md](START_HERE.md) and [the project setup guide](docs/PROJECT_SETUP.md). On Windows, bootstrap the isolated Python environment, run the hardware-free baseline, and compile the board target with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-hardware-free.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools/compile-firmware.ps1
```

The board compile does not upload firmware. The validation plan separates software readiness from physical bring-up and handwriting feasibility; the [baseline record](docs/BASELINE.md) identifies what has and has not been checked in this workspace.

## Desktop GUI concept

An optional PySide6 desktop UI design is available. Its Record page includes an animated 3D board and IMU-axis preview driven by synthetic demo motion; dragging orbits the camera and pauses the demo for inspection. The UI does not connect to serial hardware, read or write capture files, export datasets, or train a recognition model. Install its separate UI dependency and launch it with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-gui.ps1
```

See [the GUI design brief](docs/GUI_DESIGN.md) for the intended capture, dataset, and future training workflows.

The cross-platform verification command is `python tools/run_all.py --sanitizers` (requires a C++17 compiler). On Windows, the normal native tests use MinGW GCC; the sanitizer stages use LLVM-MinGW Clang because the WinLibs GCC package does not ship `libasan`/`libubsan`. Install the UCRT LLVM-MinGW toolchain if needed with `winget install --id MartinStorsjo.LLVM-MinGW.UCRT -e`. The runner executes the same firmware harness and FIFO decoder property test under ASan and UBSan. The nightly 30-minute synthetic check is `python tools/run_endurance.py --seconds 1800`. It writes ignored artifacts under `build/endurance/` and reports synthetic software evidence only.

## Wiring

| XIAO ESP32-S3 | GPIO | IMU breakout |
| --- | ---: | --- |
| 3V3 | — | VIN, if this breakout accepts 3.3 V on VIN |
| GND | — | GND |
| D8 / SCK | 7 | SCL |
| D10 / MOSI | 9 | SDA |
| D9 / MISO | 8 | SA0 |
| D3 | 4 | CS |
| D0 | 1 | INT1 |

Verify the breakout's power path and regulated supply, and remove any I²C address strap on SA0 or fixed CS connection. Breakout SDA and SCL are SPI signals here. The XIAO GPIO pins are not 5 V tolerant. Consult the supplied hardware specification for the remaining pins and power checks.

## Firmware

Open `firmware/IMUpen/IMUpen.ino` in Arduino IDE, select **XIAO_ESP32S3**, enable USB CDC on boot, and upload. The sketch checks `WHO_AM_I` and register readbacks. It accepts `0x69` by default. `0x6A` is shared by several devices: only after physically identifying an **LSM6DS3TR-C**, set `CONFIRMED_LSM6DS3TR_C` to `true` and rebuild. The TR-C FIFO has a different capacity from the original; neither configuration has been validated on this physical breakout.

The IMU runs at nominal 208 Hz, ±2 g and ±500 °/s. Its continuous FIFO uses an eight-frame watermark (48 16-bit words) routed to INT1. At each asserted watermark, the firmware checks FIFO status and the next-word pattern, then reads 48 words using the FIFO's SPI rounding behavior. The pattern is `gx, gy, gz, ax, ay, az`; partial patterns are carried across bursts. The first full frame after entering FIFO mode is discarded. A full FIFO, observed overrun, or unexpected pattern starts a new `epoch`, resets the FIFO, and discards another initial frame. The number of overwritten samples is unknown. The 256-entry output queue is separate from acquisition.

## Run an automated recording

Run `tools/setup.ps1` to install the pinned serial dependency into the project `.venv`, then choose the board's actual serial port:

```powershell
.\.venv\Scripts\python.exe tools/measure.py capture --port COM14 --output measurements/run01 --duration 300 --pause-at 90 --pause-seconds 10 --disconnect-at 210 --disconnect-seconds 10
```

Replace `COM14` with the XIAO's port and use a fresh output directory. The example records for five minutes, stops host reads for ten seconds, resumes, closes the serial port for ten seconds, and opens it again. The firmware continues while the host port is closed **only if the board remains powered**; closing a port is not the same as physically unplugging USB. Opening a serial port can reset some boards or drivers. The `boot_id` field and event log identify observed restarts; the random boot ID is an identifier, not a guarantee of uniqueness.

For a 30-minute run, use `--duration 1800` and select suitable pause and disconnect times. A longer pause than the output queue can hold deliberately causes output loss. The test does not create CPU, Wi-Fi, or other workload on the microcontroller; run those separately if relevant to the intended deployment.

The output directory contains:

- `capture.csv`: raw values plus host arrival time and detected session number.
- `events.jsonl`: port opens, planned interruptions, device messages, errors, and detected resets.
- `report.json`: observed sequence gaps, per-session cumulative counters, FIFO recoveries, overrun observations, read duration, interrupt-to-read latency, watermark intervals, and a robust clock fit per host session, boot, sequence run, and FIFO epoch.
- `reconstructed.csv`: the raw rows with `estimated_sample_host_us` when enough valid anchors exist. That legacy-named column is expressed in the MCU IRQ clock (`mcu_irq_us_since_boot`); `host_elapsed_s` is the separate host receive clock. Blank reconstructed values mean there is no defensible fit.

Re-run analysis with `.\.venv\Scripts\python.exe tools/measure.py report measurements/run01/capture.csv --output measurements/reanalysis`. The output schema is described by `FIELDS` in the script and by the firmware's startup header. Each row includes a random `boot_id`, sensor identity, host read sequence, FIFO epoch, batch and ISR count, interrupt and read times, FIFO state and pattern, six raw axes, and cumulative quality counters. `irq_us = -1` means that burst continued draining an asserted watermark without a new captured edge. `word_end` is the zero-based last word of that frame within the 48-word burst.

At the configured ranges, nominal conversions for the original LSM6DS3 are `accel_g = raw * 0.000061` and `gyro_dps = raw * 0.0175`. Queue drops and sequence gaps identify observed output loss. They cannot count sensor data overwritten during an overrun, bytes lost after a serial write, or data lost before the recording begins or after it ends. A counter value may only become visible when a later row is delivered.

## Logic analyzer

Connect analyzer ground to the common ground, one analyzer channel to IMU INT1/XIAO D0, and another to IMU CS/XIAO D3. Use input levels suitable for 3.3 V. Install `sigrok-cli` and its hardware driver, then add these options to the capture command:

```powershell
--sigrok-driver fx2lafw --irq-channel 0 --cs-channel 1 --logic-rate 1000000 --logic-seconds 2
```

Replace the driver and channel names with those for your analyzer. Once the first serial sample arrives, the tool starts a concurrent finite capture, saves the raw analyzer CSV and `sigrok.log`, converts transitions to `logic_edges.csv`, and reports the delay from each INT1 rising edge to the next CS falling edge. The `sigrok-cli` executable must be on PATH. The analyzer and ESP32 clocks are independent; compare their latency distributions rather than subtracting absolute timestamps across devices. At 1 MHz, edge timing is quantized to 1 µs. The analyzer CSV is uncompressed and its size grows with sample rate and capture time.

Alternatively, provide `--logic-csv path/to/edges.csv` to `capture` or `report`. This file must have the header `time_s,int1,cs` and monotonic timestamps in seconds, with digital levels 0 or 1 in each row. Include an initial state and each transition. CS must be dedicated to this IMU. An interrupt that arrives while CS is already low does not provide a corresponding next-edge latency measurement. Hardware, driver, and CSV format compatibility must be checked on the actual analyzer.

## Clock and alignment interpretation

`irq_us` is read in the GPIO interrupt handler; `read_start_us` precedes the status SPI transfer and `read_end_us` follows the FIFO transfer. These are host times, not sensor timestamps. The report fits FIFO sequence against watermark interrupt times, assuming the interrupt corresponds to the eighth frame in a pattern-aligned burst. It reports fit residuals and splits fits at observed FIFO recoveries or MCU restarts. ISR latency, delayed servicing, unknown losses, and sensor clock drift limit this estimate. A common fitted frame time does not prove that gyro and accelerometer conversions were simultaneous. Do not interpolate across epochs or treat reconstructed times as exact ground truth.

An optional exploratory alignment check is available for a **controlled rotation about a fixed sensor position**, with little translation or vibration:

```powershell
.\.venv\Scripts\python.exe tools/measure.py report measurements/run01/capture.csv --output measurements/alignment --alignment --alignment-tolerance-ms 10
```

It compares the acceleration derivative with the rotation predicted by the gyro at integer sample offsets. The result reports correlation, lag, and sample-period resolution; positive offset means the comparison used a later gyro row. A tolerance result is emitted only for a sufficiently correlated interior peak. This is a conditional signal-based estimate, not proof of synchronization: periodic movement, filter delay, translation, vibration, sensor bias, and weak motion can invalidate it. Decide the acceptable tolerance for the intended handwriting analysis and use an independently timed motion reference to verify the result.

The FIFO procedure and pattern follow the [LSM6DS3 datasheet](https://www.st.com/resource/en/datasheet/lsm6ds3.pdf) and [ST application note AN4650](https://community.st.com/ysqtg83639/attachments/ysqtg83639/mems-sensors-forum/21052/2/AN4650.PDF). Preserve native 208 Hz captures; make any 100 Hz derivative offline with anti-alias filtering.

## Canonical dataset export

Keep the raw capture unchanged. Create a versioned canonical dataset and optional 100 Hz derivative with explicit source and configuration provenance:

```powershell
python tools/dataset.py canonicalize measurements/run01/capture.csv --output measurements/run01/canonical.csv --source-type real --config configs/xiao_lsm6ds3_spi_208hz.yaml
python tools/dataset.py resample measurements/run01/canonical.csv --output measurements/run01/canonical-100hz.csv --target-rate-hz 100
```

Both outputs have manifest sidecars. Canonical files preserve raw values and add scaled/corrected columns; resampling is segment-local and never crosses missing sequence numbers, time discontinuities, sessions, or FIFO epochs. See [the data schema](docs/DATA_SCHEMA.md), [report schema migration](docs/REPORT_SCHEMA_V2.md), and [simulator boundaries](docs/SIMULATOR_LIMITATIONS.md).
