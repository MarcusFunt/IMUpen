# IMUpen repository guidance

## Project scope

This repository contains one XIAO ESP32-S3 + LSM6DS3 SPI/FIFO acquisition prototype, a Python capture/report CLI, and hardware-free tests. The plan in `COMPLETE_IMPLEMENTATION_PLAN.md` is a roadmap; it is not evidence that a milestone or physical check has passed.

## Work safely

- Preserve the `imupen_fifo_v1` CSV field order until a versioned migration and compatibility tests are in place.
- Keep the original `.ino` usable on the board. Do not change firmware solely to satisfy mocks.
- Before claiming a software change works, run the relevant tests and record their result. Hardware-free tests do not establish electrical, sensor, or handwriting performance.
- The virtual report tests include checks that expose known defects. Passing those checks does not mean the defects are fixed; see `HARDWARE_FREE_TEST_PLAN.md`.
- Keep raw captures immutable. Treat generated captures and handwriting data as private local artifacts; do not commit them.
- Do not claim physical verification until the checks in M6 of `COMPLETE_IMPLEMENTATION_PLAN.md` have been measured on the actual board and breakout.

## Local commands

- Bootstrap Python dependencies: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup.ps1`
- Run the current hardware-free baseline: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-hardware-free.ps1`
- Compile the board sketch without uploading: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/compile-firmware.ps1`

The firmware target is `esp32:esp32:XIAO_ESP32S3`, using Espressif Arduino core `3.3.6` for the recorded setup. The configuration in `configs/` documents the intended values and is not yet consumed by firmware or analysis code.
