# IMUpen hardware-free completion plan

## Goal

Complete the remaining software-only milestones M1–M5 from `COMPLETE_IMPLEMENTATION_PLAN.md`. Keep the `imupen_fifo_v1` input field order and raw captures unchanged. M6 physical bring-up and M7 real handwriting evaluation remain pending hardware and data.

## Work order and acceptance

### 1. M1 — Report integrity and quality verdicts

- Add failing regressions for empty/error/silent captures, nonzero FIFO patterns, per-session counters, events, schema versions, and signal-quality checks.
- Implement policy-driven capture/report verdicts and strict exit codes; preserve partial CSV/events/reports atomically.
- Separate host sequence gaps from firmware cumulative counters and publish per-session ends plus an observed aggregate with censored-tail caveats.
- Validate input structure and values; report event summaries, missingness, raw ranges/quantiles, repeated frames, clipping, and calibration status.
- Preserve raw report fields while versioning the new report schema.

**Acceptance:** positive regression tests replace known-defect assertions; malformed, empty, or fatal inputs cannot pass strict validation; known FIFO word offsets are handled with justified anchors or explicitly unanchored status.

### 2. M2 — Peripheral model and deterministic scheduling

- Extend the native mock with register access semantics, SPI transaction checks, FIFO capacities, sensor IDs, level-triggered watermark behavior, and fault injection.
- Add an independent deterministic discrete-event scheduler and reference decoder with reproducible seeds/traces.
- Exercise interleavings, service delays, output backpressure, resets, malformed reads, and boundary seeds; keep simulator assumptions explicit.

**Acceptance:** all existing firmware-native cases pass; 1,000 seeded scheduler/fault cases satisfy sequence, channel, epoch, and bounds invariants.

### 3. M3 — Real capture CLI through virtual serial

- Make serial transport injectable without changing the production capture parser.
- Invoke the actual CLI in child processes against a deterministic serial-compatible endpoint.
- Cover clean, no-device, startup/error-only, malformed/truncated lines, silence, planned pause/reconnect, reboot, duplicate/out-of-order data, invalid options, bounded retries, and safe partial outputs.
- Add line-size, retained-message, and disk-budget protections with bounded cleanup and atomic report publication.

**Acceptance:** CLI exit status, CSV, event log, and report match the simulated ground truth; no fatal/empty stream can appear clean.

### 4. M4 — Timing, sensor quality, physical motion, and datasets

- Simulate separate sensor, MCU, IRQ, and host clocks; add robust clock fitting with confidence and ground-truth time-error metrics.
- Build a deterministic 3D rigid-body IMU model with gravity/specific force, lever arm, axis transforms, quantization, scale, bias, noise, drift, filtering, delay, and saturation.
- Add calibration/quality summaries, canonical manifests, validity/discontinuity masks, and deterministic antialiased offline 208-to-100 Hz export.

**Acceptance:** sign/scale/phase checks pass; injected model values remain traceable; fits do not cross invalid boundaries; resampling does not bridge gaps or epochs.

### 5. M5 — Repeatable automation and release evidence

- Add a cross-platform test entry point, pinned toolchains/dependencies, pull-request CI for Python/native/virtual-serial/board builds, and bounded nightly stress jobs.
- Archive test summaries, failing seeds, reports, and build logs; add a release checklist and environment/configuration hashes.
- Run the actual XIAO ESP32-S3 Arduino compile locally (software-only).

**Acceptance:** a clean CI checkout runs every software gate; generated artifacts identify their source, configuration, seed, schema, and environment; no hardware claim is implied.

## Verification sequence

For every behavior change: add the regression first, verify it fails for the expected reason, implement the smallest fix, rerun the focused test, then rerun the full hardware-free suite. At the end, run the complete test wrapper, sanitizer builds where supported, and the board-target compile. Keep the M6/M7 checklist explicitly open.
