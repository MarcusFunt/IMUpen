# IMUpen repository guidance

## Project scope

This repository contains two related tracks: OnHW inertial handwriting recognition research and a separate XIAO ESP32-S3 + LSM6DS3 SPI/FIFO acquisition prototype with Python capture/report tools. The OnHW datasets and physical prototype are distinct data sources. Plans and software tests are not evidence that hardware, electrical behavior, or handwriting performance has been verified.

## Work safely

- Preserve the `imupen_fifo_v1` CSV field order until a versioned migration and compatibility tests are in place.
- Keep the original Arduino sketch usable on the board. Do not change firmware solely to satisfy mocks.
- Run the relevant tests before claiming a software change works. Hardware-free tests do not establish electrical, sensor, or handwriting performance.
- Virtual report tests include checks that expose known defects. Passing those checks does not mean the defects are fixed; see `HARDWARE_FREE_TEST_PLAN.md`.
- Keep raw captures immutable. Treat generated captures, handwriting data, OnHW datasets, and run artifacts as private local data; do not commit them.
- Do not claim physical verification until the checks in M6 of `COMPLETE_IMPLEMENTATION_PLAN.md` have been measured on the actual board and breakout.
- Do not commit secrets or `.env` files, and understand tests before changing or removing them.
- Do not force-push to `main` or `master`.

## Local commands

- Bootstrap acquisition Python dependencies: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup.ps1`
- Run the acquisition hardware-free baseline: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-hardware-free.ps1`
- Compile the board sketch without uploading: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/compile-firmware.ps1`
- Run the Python suite: `python -m pytest -q`
- Run the C++ harness and sanitizer stages: `python tools/run_all.py --sanitizers` (requires a C++17 compiler; Windows sanitizer stages use LLVM-MinGW Clang).

The Arduino firmware target is `esp32:esp32:XIAO_ESP32S3`, using Espressif Arduino core `3.3.6` for the recorded setup. The configuration in `configs/` documents intended values and is not yet consumed by firmware or analysis code. See `ONHW_RESEARCH.md` for research workflows and `docs/HARDWARE_ACQUISITION.md` for the acquisition prototype.
