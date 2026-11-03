# IMUpen hardware-free work tracker

## Software milestones M1–M5

- [x] M1.1 Capture state machine, fatal/silence/startup policies, partial artifacts, atomic reports, strict exit codes.
- [x] M1.2 Per-session counters, monotonicity/anomalies, observed aggregate, censored-tail notes.
- [x] M1.3 Word-pattern timing anchors with derivation and explicit unanchored reasons.
- [x] M1.4 Event integration, structural validation, clipping, missingness, ranges, duplicates, calibration status, schema migration.
- [x] M2.1 Register/SPI/FIFO semantics, 4 KiB and 8 KiB profiles, watermark/overrun behavior, and peripheral fault controls.
- [x] M2.2 Deterministic event scheduler and stepwise invariants; baseline 1,000 and extended 10,000 interleaving/fault seeds pass.
- [x] M2.3 Independent C++ reference decoder; 1,000 seeded streams and six FIFO patterns pass.
- [x] M3.1 Injectable serial transport and child-process CLI integration.
- [x] M3.2 Virtual startup/error, malformed/truncated, no-device, pause/reconnect, sequence ordering, reboot, and retry cases.
- [x] M3.3 Atomic artifacts, bounded lines/logs/output, and portable invocation.
- [x] M4.1 Four-clock ground-truth model and robust timing confidence checks.
- [x] M4.2 Rigid-body sensor model and motion sign/scale/delay tests.
- [x] M4.3 Calibration/quality summaries and explicit units.
- [x] M4.4 Canonical manifest and discontinuity-aware antialiased 100 Hz export.
- [x] M5.1 Pinned XIAO target compile and compiler-warning record.
- [x] M5.2 PR/nightly CI workflows authored with timeouts and evidence artifacts. Hosted GitHub Actions results are pending a push/runner.
- [x] M5.3 Cross-platform runner, release checklist, provenance hashes, and 30-minute synthetic endurance run.

## Verification record

- `python tools/run_all.py`: PASS on Windows 11, Python 3.13.5, MinGW g++ 15.2.0. This ran 43 unit tests, 18 virtual integration tests, the standalone decoder, 1,000-seed decoder property check, and nine firmware-native peripheral scenarios.
- `python tools/run_all.py --sanitizers`: PASS; the same stages plus firmware harness and decoder property test under LLVM-MinGW Clang 22.1.8 with ASan/UBSan (26.7 seconds total).
- `powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-hardware-free.ps1`: PASS with the same 43 + 18 Python tests and the native decoder/peripheral checks.
- `python -m pytest -q`: 61 passed. `ruff check --select F tools tests`, `actionlint .github/workflows/ci.yml`, `python -m compileall -q tools tests`, and `git diff --check` passed.
- The default Ruff style profile still reports 25 E402/E701/E702/E731 findings, mainly dynamic test imports and compact statements; the F-series correctness checks are clean.
- Extended scheduler stress: 10,000 seeded interleaving/fault cases passed in 79.1 seconds.
- 30-minute end-to-end dataset export/resample: 374,400 canonical rows produced 179,976 100 Hz rows with source hashes and no gaps. Peak Windows working set was approximately 986 MB during resampling; budget memory for longer recordings.
- New boundary regressions verified duplicate CSV headers, nonfinite/nonpositive timing policies, nonfinite capture duration, and invalid resampler widths fail safely.
- `python tools/run_endurance.py --seconds 1800 --seed 20261103`: PASS; 374,400 synthetic rows, CLEAN verdict, estimated 208.000000164 Hz. Capture SHA-256: `a9b15c8c5d8ec31ba510ffaa0bf68fad2be9bd135696a630f4d89a9a9a574738`.
- XIAO ESP32-S3 compile: PASS with Arduino CLI 1.4.1 and ESP32 core 3.3.6; no upload performed. See `docs/BUILD_BASELINE.md`.
- ASan/UBSan passed locally using LLVM-MinGW Clang 22.1.8 for the firmware harness and decoder property test; small intentional memory/overflow probes confirmed both runtimes detect faults. WinLibs GCC itself still lacks those runtimes. Hosted CI has not yet run in this session.
- All software simulations and mocks remain software-only evidence.

## Hardware-gated work

- [ ] M6 physical electrical/sensor verification — pending hardware and measurements.
- [ ] M7 real labeled handwriting pilot — pending hardware and real recordings.
