# Project setup and execution

## Scope

This workspace is prepared to start milestone M0 and proceed through the hardware-free milestones in `COMPLETE_IMPLEMENTATION_PLAN.md`. It covers one XIAO ESP32-S3 and one LSM6DS3-family sensor. It does not cover the complete handwriting-recognition project.

The setup keeps three evidence levels distinct:

1. **Hardware-free software:** Python analysis, native decoder, virtual sensor harness, and eventually virtual serial and timing tests.
2. **Physical hardware:** voltage path, sensor identity, SPI/FIFO/INT1 behavior, calibration, and timing on the actual board.
3. **Handwriting feasibility:** labeled real recordings evaluated across separate sessions.

Only the first level can be advanced without the board and sensor. Passing a simulator does not pass the physical or recognition gates.

## Workspace setup on Windows

Requirements for the current project scripts are Python 3.10 or later, `g++` with C++17 support, and (for the board build) Arduino CLI with Espressif Arduino core 3.3.6. The workspace snapshot is recorded in `BASELINE.md`.

Run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/setup.ps1
```

This creates `.venv` and installs the pinned `requirements.txt`. It reports whether the native compiler and board toolchain are available; it does not install system-level tools.

## M0 baseline commands

After setup, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-hardware-free.ps1
```

The script runs the two Python unittest groups and builds/runs the standalone FIFO decoder and native firmware harness. Generated executables go under ignored `build/`. Some virtual tests intentionally document current report defects; their passing result is not evidence those defects are fixed. Capture the command output and update `BASELINE.md` with the actual result before starting implementation tickets.

Compile the real sketch for the installed board target without uploading it:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/compile-firmware.ps1
```

The compile is a software toolchain check. It does not identify the attached sensor or establish electrical correctness.

## Recommended sequence

1. Record the baseline and preserve the current `.ino` behavior (IMU-001).
2. Fix report health and per-session accounting with positive regression cases (IMU-002 through IMU-005).
3. Expand the native model and add virtual serial coverage (IMU-007 through IMU-009).
4. Validate timing, signal-quality, and motion models against explicit synthetic truth (IMU-010 through IMU-012).
5. Add CI and reproducibility gates (IMU-006 and IMU-013).
6. Only after software readiness, follow the physical bring-up checklist (IMU-014), then the real-data pilot (IMU-015).

Use the full plan for ticket-level acceptance criteria. Do not mark future or physical milestones complete based on repository setup alone.

## Captures and generated files

Generated builds, simulations, measurements, recordings, and artifacts are ignored by Git. Raw recordings should remain unchanged; derived data belongs in separately named outputs with its processing and quality metadata. Do not commit private handwriting recordings.
