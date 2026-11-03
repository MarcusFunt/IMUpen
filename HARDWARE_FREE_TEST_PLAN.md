# IMUpen — detailed test plan without a connected sensor

This testing pack is an **additive extension** to the original IMUpen ZIP. It does not modify the existing firmware or measurement code. Real sensor properties are represented by an imperfect mock, not confirmed measurements.

## Quick start

Use Python 3.10+ and a C++17 compiler (GCC/Clang or MSYS2/WSL on Windows). Work from the directory containing `tools`, `firmware`, `tests`.

```bash
# Existing Python regression tests
python -m unittest discover -s tests -p 'test_*.py' -v

# NEW virtual measurement integration tests
python -m unittest discover -s tests/hardware_free -p 'test_*.py' -v

# Native C++ tests using the ACTUAL IMUpen.ino and a mocked ST register/FIFO model
# Linux, WSL or macOS with GCC (adapt compiler for your platform)
g++ -std=c++17 -Wall -Wextra -O2 -Itests/hardware_free/stubs tests/hardware_free/native_harness.cpp -o virtual_imu_test
./virtual_imu_test

# Generate a deterministic simulated 30-minute CSV very quickly
python tests/hardware_free/virtual_capture.py --scenario clean --seconds 1800 --output simulated/clean.csv
python tools/measure.py report simulated/clean.csv --output simulated/clean_report

# Reproduce sequence gaps, FIFO resets, imperfect timing and analysis limitations
python tests/hardware_free/virtual_capture.py --scenario queue_drop --seconds 120 --output simulated/drops.csv
python tools/measure.py report simulated/drops.csv --output simulated/drops_report
python tests/hardware_free/virtual_capture.py --scenario nonzero_pattern --seconds 30 --output simulated/pattern.csv
python tools/measure.py report simulated/pattern.csv --output simulated/pattern_report
```

`virtual_capture.py` also supports `reboot`, `fifo_reset`, `irq_outliers` and `clipping`, as well as `--odr-ppm`, `--irq-jitter-us`, `--accel-noise-mg`, `--gyro-noise-dps`, `--gyro-bias-dps`, and `--accel-bias-mg`. Every generated `.csv` has a matching `.truth.json` stating the injected defects, which provides a ground truth for assertions.

## Existing tools directly exercised

- `firmware/IMUpen/IMUpen.ino` is included **unchanged** in the native harness. Register reads/writes, SPI burst parsing, ISR timestamp hook, FIFO decode, sample queue, and recovery path execute against deterministic substitutes.
- `tools/measure.py` is imported **unchanged** by Python integration tests. Its real report-writing and clock-fitting logic process generated files.
- The existing `test_measure.py` tests continue to run.

The native mocks do **not** implement a full ESP32 FreeRTOS scheduler, USB CDC peripheral, actual SPI electrical behavior, radio, hardware timers, or the analog MEMS signal chain. In particular, tests do not prove ISR preemption safety or lossless USB streaming in real-world conditions.

## Current passing coverage

### Native MCU logic (6 scenarios)

1. Load firmware configuration into mock registers and read out 24 FIFO frames in a single delayed service; confirm correct channel order, discard behavior and continuation timestamps.
2. Fill a 10-sample bounded queue while reading 40 frames; ensure sequence advances and exactly 29 failed inserts are counted.
3. Inject FIFO overrun; confirm reset, epoch increment and first-frame discard after recovery.
4. Start the FIFO at word pattern 2; verify correct channel mapping and discard of initial partial and full frame.
5. Inject a pattern mismatch; confirm recovery without emitting corrupt frames.
6. Simulate 800 consecutive frames; assert 799 good outputs after the startup discard, no reordering or loss.

### Python report integration (8 scenarios)

1. Correctly fit 208 Hz on clean synthetic rotation.
2. Confirm queue losses are counted via missing sequence numbers.
3. Confirm reboot creates separate session models.
4. Confirm FIFO recovery creates separate epoch models.
5. **Demonstrate known omission:** pattern 2 yields zero fitted models despite valid synthetic anchors.
6. **Demonstrate known omission:** injected saturated samples are not flagged in `report.json`.
7. Large IRQ outliers increase fit-residual spread.
8. **Demonstrate known omission:** an empty CSV still receives a report without `verdict=FAILED`.

Some passing tests deliberately assert **known failures** in existing code so they remain visible to reviewers. After fixing the implementation, revise these tests to demand the correct behavior; do not interpret an all-green test run as proof those issues are fixed.

## Additional high-value work (not yet implemented)

1. **Native full-scheduler fuzzing:** use an RTOS-like discrete-event scheduler, model `Serial.printf` blocking and simultaneous interrupts, vary free-queue capacity, and assert no out-of-order or duplicated samples after any recovery.
2. **Virtual serial integration:** connect the real `capture` command to a PTY (Linux/macOS) or virtual COM pair (Windows). Inject malformed ASCII frames, partial UTF-8/ASCII, silent devices, disconnects, repeated `WHO_AM_I`, and block reads. Test that errors cause an appropriate nonzero exit status; current code does not meet that contract.
3. **Metamorphic tests:** change simulated clock drift or all timestamps by a fixed offset; sample-rate results should respond predictably while sample values remain unchanged. Add property-based testing with Hypothesis to randomize patterns, counter resets and reorderings.
4. **Firmware build CI:** install the Espressif Arduino core and compile the actual board target. Native stubs cannot validate ESP32 API compatibility, stack use, task pinning or USB CDC build flags.
5. **Gyroscope/acceleration physics simulator:** extend the fixed-position rotation generator with arbitrary 3D orientation, pen-tip constraints, translation, pressure/contact states, user-specific stroke kinematics and a known optical displacement reference. Then test reconstruction algorithms. **These signals are not substitutes for real handwritten labels.**
6. **Safety/quality gate:** fail reports with zero rows, unexpected silent intervals or unrecognized device ID, high clipping, nonmonotonic time, impossible acceleration values, and unbounded recoveries. Keep planned stress failures as separate intentionally degraded results.

## Suggested acceptance tests before buying more hardware

- All native and Python tests pass, including revised tests that assert defects are fixed rather than merely documenting them.
- Firmware compiles for XIAO ESP32-S3 in CI, independently of a physical board.
- 30-minute virtual stream with injected queue drops reports exact observable loss and per-session totals.
- Simulated resets split epochs correctly; do not bridge known-unknown time gaps during training.
- Timing reconstruction returns estimated rate within a chosen tolerance of simulated ground truth (e.g. 0.1%, with separately defined jitter budget) and reports residual uncertainty.
- A clean stationary model has approximately 1 g acceleration magnitude (plus controlled bias) and near-zero gyroscope rate; saturation injections fail the quality gate.
- Simulated backpressure and disconnected transport cause the requested failure/recovery event accounting and correct overall test verdict.

**Critical physical test that remains:** first assembled test on the real XIAO and GY-LSM6DS3 must confirm device identification, register readback, FIFO level/interrupt behavior, sample scaling, timing and noise. No simulation can certify actual board wiring, regulator safety, or handwriting recognition accuracy.
