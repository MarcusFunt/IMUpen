# IMUpen V0 — Complete Hardware-Free Validation, Acquisition Reliability, and Hardware Verification Plan

**Status:** Implementation plan, not a claim that all work is finished  
**Prepared:** 2026-11-03  
**Scope:** One Seeed Studio XIAO ESP32-S3 + one ST LSM6DS3-family IMU, 4-wire SPI, FIFO watermark on INT1, native 208 Hz sampling, USB serial recording, Python measurement analysis  
**Primary goal:** Prove as much of the acquisition *software* as possible without a physical sensor, then define narrowly scoped physical checks before trusting real recordings for handwriting experiments.  
**Not in scope for V0:** Wireless transmission, a second IMU, optical tracking, pressure electronics, machine-learning performance guarantees, a finished pen enclosure, or a full SLAM system.

---

## 0. Executive decision

**Do not rewrite the acquisition pipeline or buy additional sensing hardware yet.** Keep the present SPI/FIFO/interrupt architecture. First make the firmware compilable on the real board target, the host capture fail clearly when invalid, and every analysis result reproducible from known simulated ground truth. Add a virtual serial device and a scheduler-aware sensor model. Only then perform a small number of highly targeted electrical and mechanical checks on the existing/first physical configuration.

The V0 *software-complete* gate is not equivalent to *hardware-verified* or *handwriting-feasible*. Those are separate gates below.

### Non-negotiable principles

1. **Raw data is immutable.** Preserve original 208 Hz integer readings, device messages, acquisition events, and timestamps. Derived/calibrated/resampled versions must be separately labeled.
2. **A test passing may mean an injected failure was detected.** Separate test-result status (`passed`) from recording-quality status (`clean`, `degraded-expected`, `failed`, `unknown`).
3. **Do not invent data continuity.** Never bridge unknown FIFO loss, MCU reboots, host restarts, or unverified timing intervals when preparing training data.
4. **Observed loss is a lower bound.** Missing sequence numbers, firmware queue-drop counters, and overrun observations describe different failure modes; none measures every lost physical sensor sample.
5. **Hardware specifications are not measurements.** Datasheet electrical limits and reference noise values must be checked against the exact IC and breakout revision, and cannot prove real-board timing or calibration.
6. **Do not call a simulator a handwriting benchmark.** Synthetic motions validate code and physics; real sensor/writer recordings validate recognition feasibility.
7. **Keep one small, reproducible configuration first.** XIAO ESP32-S3 + LSM6DS3, SPI mode 3 at 4 MHz, ±2 g, ±500 °/s, FIFO at 208 Hz, 48-word watermark, 256-entry output queue.

---

## 1. Source material, baseline and provenance

The plan combines the three supplied sources:

- Uploaded `IMUpen-project (1).zip`: actual eight-file acquisition prototype, including `firmware/IMUpen/IMUpen.ino`, `FifoDecoder.h`, `tools/measure.py`, `tests/test_measure.py`, `tests/fifo_decoder_test.cpp`, and `README.md`.
- Uploaded `IMUpen-hardware-free-test-plan(1).md`: additional virtual sensor tests, known limitations, proposed acceptance criteria, and extension roadmap.
- Uploaded `IMUpen-hardware-specifications(1).md`: pinout, sensor variants, selected registers, bandwidth calculations, and unresolved board-specific electrical facts.

An earlier **separate** `IMUpen-hardware-free-validation.zip` package contains add-on files under `tests/hardware_free/`. **Those files are not present in the uploaded base project ZIP**. Integrate or copy them deliberately; do not assume they are already committed to the repository.

### Tests verified while preparing this plan

| Test set | Verified local result | What it proves |
| --- | --- | --- |
| Base `tests/test_measure.py` | 5/5 passed | Original Python unit tests execute |
| Base `tests/fifo_decoder_test.cpp` | Passed | Basic standalone C++ decoder behavior |
| Add-on `tests/hardware_free/test_virtual_capture.py` | 8/8 passed | Virtual CSV/reporting cases execute, **including tests documenting current defects** |
| Add-on `tests/hardware_free/native_harness.cpp` | 6 scenarios passed | Actual `.ino` executes against simplified virtual SPI/FIFO/RTOS stubs |
| True `arduino-cli` XIAO firmware build | **Not performed** | Cannot claim board-toolchain compile readiness |
| Physical SPI/USB/IMU measurements | **Not performed** | Cannot claim sensor or board behavior |

The 8 add-on Python tests explicitly include checks that **current shortcomings remain observable** (e.g., empty recordings not failing). They must later be converted into positive regression assertions as each defect is fixed.

### Confirmed current issues to fix

| ID | Severity | Location / behavior | Required outcome |
| --- | --- | --- | --- |
| BUG-01 | P0 | `tools/measure.py`: zero-row/silent/error-only capture can yield a report without an explicit FAILED quality verdict | Fail acquisition validation when the sensor never delivered usable data |
| BUG-02 | P1 | `analyze()` uses a global max for cumulative counters from multiple sessions | Report per-session end counters and a clearly defined aggregate, not the cross-session max |
| BUG-03 | P1 | Timing anchor only accepts `pattern == 0` | Support properly justified nonzero-pattern anchors or mark them explicitly unsupported; never invent fitted times |
| BUG-04 | P1 | No physical signal plausibility/saturation reporting | Add raw-domain quality checks and configuration-aware scale handling |
| BUG-05 | P1 | Ordinary least-squares clock fit vulnerable to delayed IRQ outliers | Add robust estimation, diagnostics, and explicit uncertainty |
| BUG-06 | P1 | Event log is not incorporated into overall quality verdict | Merge serial/device/transport events with report health criteria |
| GAP-01 | P1 | No full ESP32-S3 Arduino board build in CI | Compile actual board target in CI with pinned toolchain |
| GAP-02 | P1 | No virtual serial/real CLI end-to-end test | Exercise the *actual* `capture` command against a simulated port |
| GAP-03 | P1 | Native stubs omit preemptive scheduling/USB backpressure | Model delays, wakeups, and scheduler interleavings explicitly |
| GAP-04 | P2 | No calibration or sensor-noise acceptance workflow | Add quality and physical verification protocols |

---

## 2. Hardware contract and invariant configuration

### 2.1 Exact pin map to use

| XIAO pad | ESP32-S3 GPIO | Breakout pad | Signal | Direction (from ESP32) |
| --- | ---: | --- | --- | --- |
| `3V3` | — | `VIN` **only when verified**; otherwise documented safe 3.3 V supply input | Regulated supply | Power |
| `GND` | — | `GND` | Ground | — |
| `D8` | GPIO7 | `SCL` | SPI SCK | Output |
| `D10` | GPIO9 | `SDA` | SPI MOSI/SDI | Output |
| `D9` | GPIO8 | `SA0` | SPI MISO/SDO | Input |
| `D3` | GPIO4 | `CS` | SPI chip select, active low | Output |
| `D0` | GPIO1 | `INT1` | FIFO watermark interrupt | Input |

**Critical caveats:** `SCL`/`SDA` are pin labels reused as SPI SCK/MOSI on this breakout; they are **not I²C in this design**. In 4-wire SPI, `SA0` is sensor SDO; it must not be hard-strapped to a logic rail. `CS` must be driveable. Treat the XIAO's `3V3` pin as a regulated **output** by default; do not back-feed it. Never apply 5 V directly to XIAO GPIO or to the bare IMU. Verify the exact breakout's VIN regulator and 3V3 rail before wiring.

The optional `INT2`, `OCS`, `SCX`, `SDX` breakout pads are unused in V0. If using the XIAO **Sense** revision, check shared D8–D10 SPI use with onboard microSD peripherals before attempting concurrent access.

### 2.2 Sensor-variant and electrical table

| Property | LSM6DS3 | LSM6DS3TR-C | Action |
| --- | --- | --- | --- |
| `WHO_AM_I` | `0x69` | `0x6A` | Check device identity; `0x6A` alone is not adequate proof of an exact ST part |
| Accel axes / gyro axes | 3 + 3 | 3 + 3 | Keep six signed 16-bit raw channels |
| Accel scales | ±2/4/8/16 g | same | Start at ±2 g; check clipping |
| Gyro scales | ±125/250/500/1000/2000 dps | same | Start at ±500 dps; check clipping |
| Accel at ±2 g | 0.061 mg/LSB | same | Use correct recorded-range calibration |
| Gyro at ±500 dps | 17.5 mdps/LSB | same | Use correct recorded-range calibration |
| FIFO capacity | 8 KiB | 4 KiB | Different model parameters, do not assume same hold-up |
| Max SPI frequency | 10 MHz | 10 MHz | Current 4 MHz is under IC limit, not a guarantee for breakout wiring |
| IC analog supply | 1.71–3.6 V | 1.71–3.6 V | Check actual breakout regulator and IO supply |
| IC temperature rating | −40 to +85 °C | −40 to +85 °C | Not equivalent to calibrated accuracy over range |
| IC package | 2.5 × 3 × 0.83 mm | same | Confirm board orientation separately |

Representative ST datasheet noise densities, not verified board measurements: original LSM6DS3 accelerometer ~90 µg/√Hz and gyroscope ~7 mdps/√Hz in cited high-performance conditions; LSM6DS3TR-C gyro ~5 mdps/√Hz. Actual effective RMS noise depends on bandwidth, filtering and operating mode.

### 2.3 Current code register map

| Register | Address | Value | Meaning to model and test |
| --- | --- | --- | --- |
| `WHO_AM_I` | `0x0F` | read | Sensor identification |
| `CTRL3_C` | `0x12` | reset `0x01`, then `0x44` | Software reset, BDU, automatic address increment; verify 4-wire SPI |
| `CTRL1_XL` | `0x10` | `0x50` | Accel ODR 208 Hz, ±2 g |
| `CTRL2_G` | `0x11` | `0x54` | Gyro ODR 208 Hz, ±500 dps |
| `FIFO_CTRL1` | `0x06` | `0x30` | FIFO watermark low bits, 48 words |
| `FIFO_CTRL2` | `0x07` | `0x00` | FIFO watermark high bits etc. |
| `FIFO_CTRL3` | `0x08` | `0x09` | Gyro and accel streams enabled without decimation |
| `FIFO_CTRL4` | `0x09` | `0x00` | No auxiliary FIFO streams |
| `FIFO_CTRL5` | `0x0A` | `0x2E` | FIFO continuous operation, 208 Hz |
| `INT1_CTRL` | `0x0D` | `0x08` | Route FIFO watermark to INT1 |
| `FIFO_STATUS1..4` | `0x3A..0x3D` | read | FIFO words, status flags, pattern |
| `FIFO_DATA_OUT_L` | `0x3E` | burst-read | FIFO data in 16-bit words |

Expected word order when both sensors run at matched rates: `gx, gy, gz, ax, ay, az`, repeated. Native model must represent **word-level** patterns and nonzero starting offsets, not only complete preassembled six-axis frames.

**Variant-specific confirmation:** Verify all register/bit fields against the applicable ST datasheet and application note before claiming TR-C compatibility. Keep `CONFIRMED_LSM6DS3TR_C = false` until the exact IC is identified.

### 2.4 Derived nominal rates and buffers

| Quantity | Nominal value | Interpretation |
| --- | ---: | --- |
| Output data rate | 208 Hz | Nominal, not an exactly measured sensor oscillator rate |
| One six-axis frame | 6 × 16-bit = 12 bytes | Excludes metadata and CSV overhead |
| Raw FIFO payload rate | 2,496 bytes/s | 208 × 12 |
| Per-frame interval | 4.807692 ms | Nominal 1/208 s |
| FIFO watermark | 48 × 16-bit = 96 bytes | Eight full six-axis frames |
| Nominal watermark interval | 38.461538 ms | 8/208 s |
| 96-byte FIFO burst plus 1-byte address | ≥194 µs at 4 MHz | SPI wire time only, excluding status read, CS and software |
| Original 8 KiB FIFO theoretical depth | ~3.28 s | Ignoring FIFO overhead and configuration effects |
| TR-C 4 KiB FIFO theoretical depth | ~1.64 s | Same caveat |
| Firmware 256-entry sample queue | ~1.23 s | Separate from sensor FIFO, may overflow sooner depending on transport behavior |

**Do not add FIFO and queue durations to claim a safe USB interruption duration.** The two stages run concurrently and can fail independently.

### 2.5 Hardware facts still unknown

The following cannot be resolved conclusively from the ZIP alone: exact GY-LSM6DS3 PCB revision, regulator part number and dropout, SA0/CS pull resistors or straps, level shifting, actual `INT1` logic high, exact silicon identity/authenticity, axis alignment on the breakout, mounting rotation on the pen, oscillator tolerance, sensor-specific filter delay, and susceptibility to wiring noise. Mark these as **unverified pending physical inspection** rather than filling gaps from generic listings.

---

## 3. Proposed software architecture and repository layout

Do not remove the original `.ino` initially. Prefer extracting portable logic behind narrow interfaces *after* establishing baseline test equivalence.

```text
IMUpen/
├── firmware/
│   └── IMUpen/
│       ├── IMUpen.ino                   # actual board sketch
│       ├── FifoDecoder.h                # original, tested parser
│       ├── AcquisitionCore.h/.cpp       # PROPOSED: portable state machine
│       ├── DeviceConfig.h               # PROPOSED: versioned settings
│       └── TelemetrySchema.h            # PROPOSED: shared field definitions
├── tools/
│   ├── measure.py                       # existing CLI; retain compatibility
│   ├── capture_transport.py             # PROPOSED: live serial boundary
│   ├── report_core.py                   # PROPOSED: quality/report algorithms
│   ├── sensor_quality.py                # PROPOSED: clipping and plausibility
│   ├── synthetic_motion.py              # PROPOSED: physically consistent motion
│   └── validate_manifest.py             # PROPOSED: dataset metadata checks
├── tests/
│   ├── test_measure.py                  # existing regression tests
│   ├── fifo_decoder_test.cpp            # existing C++ tests
│   ├── hardware_free/                  # FROM separate validation pack
│   │   ├── native_harness.cpp
│   │   ├── virtual_capture.py
│   │   ├── test_virtual_capture.py
│   │   └── stubs/
│   ├── test_report_health.py            # PROPOSED
│   ├── test_timestamps.py               # PROPOSED
│   ├── test_capture_pty.py              # PROPOSED
│   ├── test_sensor_quality.py           # PROPOSED
│   └── fuzz/                            # PROPOSED
├── configs/
│   └── xiao_lsm6ds3_spi_208hz.yaml      # PROPOSED
├── docs/
│   ├── HARDWARE_SPECIFICATIONS.md
│   ├── PROTOCOL.md
│   ├── MEASUREMENT_METHOD.md
│   └── HARDWARE_BRINGUP.md
├── .github/workflows/                   # PROPOSED CI workflows
└── test_artifacts/                      # ignore bulk files in Git
```

**Avoid premature modularization:** A faithful simulator calling unchanged acquisition functions is preferable to rewriting firmware and then validating only the rewritten version. When code extraction happens, use golden tests to prove that the old and new paths emit the same frames, counters, and recovery behavior.

### Required metadata contract

Every run must save:

- Unique `run_id`, `run_start_utc`, schema version, test intent (`clean`, `fault_injection`, `physical`, `training`), and source (`virtual`, `actual_device`).
- Firmware Git SHA, configuration hash, board target/revision, detected `WHO_AM_I` (or explicitly `simulated`), sensor variant assumption, channel order, nominal ODR and full scales.
- SPI mode, configured bus frequency, FIFO watermark and mode, queue length, output transport, host OS/Python version, firmware/analysis versions.
- Raw CSV or future binary data; `events.jsonl`; all generated reports; optional `.truth.json` for simulation only.
- Calibration state, axis transform, sensor-to-tip rigid transform if known, temperature methodology and units; `unknown` values are allowed.
- A machine-readable `quality_verdict`, reasons, and policy version.

Preserve the present `imupen_fifo_v1` CSV protocol until a schema migration and compatibility tests exist. Future `v2` telemetry must be versioned, not silently re-ordered.

---

## 4. Milestones and definition of done

| Milestone | Primary output | Hardware required? | Exit gate |
| --- | --- | --- | --- |
| M0 — Freeze and characterize baseline | Reproducible test results and configuration | No | Original code/tests unmodified; baseline documented |
| M1 — Correct report and health logic | Reliable verdict and per-session statistics | No | Known BUG-01/02/03/04 addressed with positive tests |
| M2 — Virtual peripheral and scheduling | Extended native firmware test suite | No | FIFO and concurrent fault matrix passing |
| M3 — Real capture CLI over simulated serial | Full host integration | No | End-to-end fault injection is accurately classified |
| M4 — Timing, quality, and motion physics | Ground-truth-based metrics | No | Synthetic truth recovered within configured tolerances |
| M5 — CI and reproducibility | Automatic board builds and release artifacts | No | Repeatable CI gates, pinned dependencies |
| M6 — Physical sensor bring-up | Device-level validation measurements | **Yes** | Identity, signals, timing and scaling verified |
| M7 — Handwriting feasibility pilot | Real labeled writer recordings and ML evidence | **Yes** | Measurable within-writer cross-session recognition results |

M0–M5 are the **hardware-free program**. M6–M7 are explicitly future stages; completion of M5 does **not** imply M6/M7 are solved.

---

## 5. M0 — Baseline, reproducibility and safe integration (P0)

**Goal:** Know exactly what is running and avoid conflating the uploaded source ZIP with the separate virtual-validation pack.

### Tasks

- [ ] Create a clean repository checkout or branch `test/hardware-free-validation`.
- [ ] Record Git SHA (or checksum of ZIP contents), Python version, compiler version, OS, and platform-toolchain versions.
- [ ] Import the existing project unchanged; run base Python and C++ tests.
- [ ] Import **only additive** `tests/hardware_free/` files from the separately supplied validation archive; review diffs before committing.
- [ ] Run all 8 virtual Python tests and the 6 native harness scenarios.
- [ ] Mark the 3 known-defect tests as `xfail`/regression TODO or rename them clearly; never count their existing behavior as fixed.
- [ ] Save a checked-in configuration file plus reproducible test commands.
- [ ] Add `.gitignore` entries for generated simulations, built binaries, huge CSV files, virtualenvs, and captures containing private handwriting.
- [ ] Document that this stage is single IMU, not the full writer-recognition project.

### Run commands

From the `IMUpen` root (after copying the additive test pack):

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m unittest discover -s tests/hardware_free -p 'test_*.py' -v

g++ -std=c++17 -Wall -Wextra -O2 \
  -Ifirmware/IMUpen tests/fifo_decoder_test.cpp -o fifo_decoder_test
./fifo_decoder_test

g++ -std=c++17 -Wall -Wextra -O2 \
  -Itests/hardware_free/stubs \
  tests/hardware_free/native_harness.cpp -o virtual_imu_test
./virtual_imu_test
```

Windows: use WSL, MSYS2/GCC, or adapt the commands for an installed C++17 toolchain. Add PowerShell wrappers later if Windows remains the main development environment.

### Acceptance

- All baseline tests remain green.
- Exact binaries, inputs, and logs can be regenerated from documented commands.
- No modifications to `IMUpen.ino` have been made merely to accommodate mocks.

---

## 6. M1 — Fix reporting correctness, session accounting and quality verdicts (P0/P1)

### M1.1 Fix empty or silent capture success (BUG-01)

**Current failure:** `capture()` can log `# ERROR` or no samples and still run to elapsed duration, then `analyze()` writes a report without a clearly failed acquisition verdict.

**Implementation:**

- [ ] Define a capture state machine: `INITIALIZING → ACQUIRING → [DEGRADED/RECOVERING] → COMPLETED` or `FAILED`.
- [ ] Track elapsed time to first valid sample, time since last sample, malformed-line rate, device `# ERROR`, serial reopen failures, and unexpected interruption durations.
- [ ] Add policy-controlled thresholds: `startup_timeout_s`, `silence_timeout_s`, `minimum_valid_rows`, `max_unplanned_recoveries`, `allow_expected_faults`.
- [ ] Report failure reasons with stable IDs (e.g., `NO_SAMPLES`, `DEVICE_FATAL_ERROR`, `SILENT_STREAM`, `INVALID_ROW_FORMAT`).
- [ ] Exit with nonzero CLI status when a normal `capture` or `report --strict` fails; intentional fault tests should explicitly request their own policy.
- [ ] Still write all available partial data, events, and report artifacts even on failure; use atomic final report publication.

**Positive tests:** Empty stream; only headers; only `# ERROR`; valid stream stopping halfway; device not found; malformed startup; valid clean stream; deliberate scheduled disconnect with expected recovery.

### M1.2 Correct cumulative loss counters across sessions (BUG-02)

**Implementation:**

- [ ] Group counter observations by actual boot/session; store each session's first/last seen counter and observed monotonicity.
- [ ] On `boot_id` change, separate sessions, regardless of `epoch` value; avoid relying solely on a random ID for true uniqueness.
- [ ] Provide `per_session_counters` and `total_observed_session_end_counters` as distinct fields.
- [ ] Flag counter decreases within a session as an anomaly; distinguish intentional reboot/reset.
- [ ] Explicitly describe censored intervals: a final counter may be unreported if the device stops sending before publishing it.
- [ ] Do **not** add inferred missing sequence numbers to queue-drop counters without checking whether they describe the same losses (prevent double counting).

**Ground truth cases:** Two sessions with nine queue drops each must show **9 + 9 as two session reports** and an aggregate of **18 observed end-counter units**, subject to counter visibility rules. Test three sessions, counter resets, wraparound/corruption, and an end-of-session loss after final received line.

### M1.3 Correct/clarify nonzero FIFO timing alignment (BUG-03)

**Current failure:** `analyze()` excludes nonzero FIFO start patterns from anchor fitting (`pattern != 0`). The C++ decoder accepts valid partial patterns.

**Implementation:**

- [ ] Write an explicit timing/word-index derivation for each valid FIFO pattern offset; cross-check against the applicable ST FIFO semantics.
- [ ] Do not simply remove the `pattern != 0` condition. Determine which complete frame corresponds to the watermark edge and whether partial words alter the anchor or increase uncertainty.
- [ ] Validate first-frame discard and sequence advancement semantics under nonzero patterns and FIFO resets.
- [ ] If a reliable anchor cannot be established, keep sample timestamps null and add a machine-readable reason (`UNANCHORED_PATTERN`), instead of fabricating a model.
- [ ] Use original `boot_id`, `session`, `epoch`, `batch`, `irq_count`, `pattern`, `word_end`, and `seq` to build verifiable anchors.

**Tests:** pattern 0–5; burst begins mid-pattern; first full frame discard; continuation burst with `irq_us=-1`; delayed service; watermark repeated while high; reset between two incomplete frames; missing device samples.

### M1.4 Merge events into report health and add quality gates (BUG-04/06)

- [ ] Add optional `--events-jsonl` to `report` and automatically find sibling `events.jsonl` for local capture outputs; make absence explicit, not fatal for imported CSV.
- [ ] Count `device_message`, `malformed_line`, `port_error`, `port_open_error`, `boot_id_changed`, `read_pause`, `reconnect`, `fifo_recoveries`, `overrun_observations`.
- [ ] Separate planned test fault intervals from unplanned losses.
- [ ] Validate signed 16-bit range; all required columns; sensible finite host times; monotonically increasing sequence per session except acknowledged restarts; legal pattern [0..5], `word_end`, and consistent batch indexing.
- [ ] Count samples close to raw full-scale clipping (configuration-aware threshold, e.g., ≥98% of limit; make threshold configurable).
- [ ] Report stationary-window acceleration norm and gyro bias only when the segment is actually labeled as stationary; do **not** fail arbitrary handwriting because acceleration norm differs from 1 g during motion.
- [ ] Add missingness rates, per-axis min/max/quantiles, repeated-identical-frame diagnostics, invalid/unusual values, and calibration status.
- [ ] Emit `report_schema_version` and `quality_policy_version`.

### Proposed report structure

```json
{
  "report_schema_version": "2.0",
  "source": "virtual",
  "test_intent": "fault_injection",
  "quality_verdict": "DEGRADED_EXPECTED",
  "quality_reasons": ["EXPECTED_QUEUE_OVERFLOW"],
  "rows": 1000,
  "sessions": [{"session": 0, "boot_id": 1234,
                "last_observed_counters": {"queue_drops": 9}}],
  "observed_loss": {"host_sequence_gaps": 9,
                    "sensor_samples_lost_in_fifo": null},
  "clock_models": [],
  "quality": {"accel_clipping_count": 0, "gyro_clipping_count": 0},
  "events_summary": {"malformed_line": 0},
  "config_hash": "example-only"
}
```

The JSON above is a **proposed** output shape, not current `measure.py` output. Numeric values are illustrative.

### M1 acceptance

- [ ] Positive regression tests replace known-defect assertions.
- [ ] Empty and error-only recording attempts produce an explicit `FAILED` verdict and nonzero strict-mode exit code.
- [ ] Session counter example reports 9 per session and 18 as the aggregate, without confusing that number with exact total physical sample loss.
- [ ] Pattern 2 is either correctly fitted with evidenced anchoring or explicitly tagged unanchored.
- [ ] Deliberate saturation is reported and fails a clean-data policy.
- [ ] Original `report.json` consumers have a documented schema migration path.

---

## 7. M2 — Build a more faithful virtual LSM6DS3 and scheduler (P1)

### M2.1 SPI/register/FIFO device emulator

Retain and extend the existing native harness, which already compiles unchanged `IMUpen.ino` against mocked platform interfaces.

**Requirements:**

- [ ] A 256-address register space with reset defaults and explicit read/write/RO semantics for supported addresses.
- [ ] `WHO_AM_I` variants `0x69` and simulated `0x6A`, with a negative test for all unsupported IDs.
- [ ] SPI mode 3 transaction boundary, CS high/low, command/read-bit interpretation, auto-increment behavior, bursts, and FIFO register read behavior.
- [ ] Realistic FIFO word stream `Gx,Gy,Gz,Ax,Ay,Az`, including arbitrary 0–5 pattern offsets.
- [ ] Configurable physical FIFO depth (original 8 KiB vs TR-C 4 KiB), watermark and overrun/full flags.
- [ ] Continuous sampling at a configurable *actual* rate around nominal 208 Hz, not only on each MCU read.
- [ ] Watermark INT1 behavior as a level with rising edges; model IRQ asserted/deasserted as FIFO is filled/drained.
- [ ] Separate event times for sensor sample creation, IRQ edge, MCU servicing, FIFO/status SPI transfers, output queue insertion, and transport write.
- [ ] Faults: dropped/duplicated FIFO words, short reads, stuck-high/low INT1, CS transaction aborted, status value races, delayed service, register readback mismatch, reset that never clears, wrong sensor ID.
- [ ] Simulator configuration and RNG seed are recorded in machine-readable truth files.

**Do not** emulate complex physical hardware features without evidence. Maintain a list of simplified assumptions and distinguish register-faithful behavior from estimates.

### M2.2 Deterministic RTOS-like event scheduler

- [ ] Introduce separately scheduled sensor clock, watermark ISR, acquisition task, output task, serial endpoint, and host reader.
- [ ] Inject IRQ latency, high-priority task interference, queue contention, blocked `Serial.printf`, USB disconnection, and delayed resumes.
- [ ] Vary scheduling order around watermark crossing and near FIFO-full transitions.
- [ ] Simulate ISR notification coalescing and a level-triggered FIFO flag remaining asserted when more words are available.
- [ ] Assert invariants after **every** scheduled step, not only at end-of-test.
- [ ] Support deterministic replay from `seed`, event trace, and state snapshot when a test fails.
- [ ] Run randomized seeds under property-based tests, optionally with Hypothesis on Python-side models.

**Invariants:**

1. Accepted output sequence values are unique and strictly increasing within a boot/session.
2. Six-axis frames preserve channel order and sign; no axis crosses a word boundary.
3. No partial pattern is emitted as a valid complete frame.
4. A FIFO reset increments epoch/recovery counters exactly as intended and invalidates previous continuity.
5. Queue drops never silently reuse a sequence number; counters and later output make the loss observable if transport survives.
6. Reads never overrun emulator FIFO bounds; a status inconsistency produces an explicit error/recovery path.
7. `irq_us=-1` is never treated as a fresh independent clock anchor.
8. Simulated power loss, reboot, and host reconnect are represented as distinct events.

### M2.3 Fuzzing and native tooling

- [ ] Property-based frame sequences for lengths 0–100,000 with varying word offsets and signed int16 edge values.
- [ ] Compare actual firmware output to an independent simple reference decoder (not reusing the same parsing algorithm).
- [ ] Add C++ AddressSanitizer and UndefinedBehaviorSanitizer builds to native tests where toolchain permits.
- [ ] Record minimal failing seed and emit a short `repro.json` plus trace.
- [ ] Add golden regression tests for each discovered issue.

### M2 acceptance

- [ ] Deterministic clean simulated 800+ frame run preserves all expected frames except documented intentional discard(s).
- [ ] All six existing native harness cases still pass.
- [ ] Every injected FIFO overrun/reset produces a distinct epoch boundary and no false cross-boundary timestamp continuity.
- [ ] At least 1,000 seeded interleaving/fault cases complete without assertion failures in continuous integration; increase in nightly tests.
- [ ] Simulator limitations are documented and reviewed against ST datasheets.

---

## 8. M3 — Exercise the real Python capture command through virtual serial (P1)

**Objective:** Validate *the complete live host path* rather than only generating CSV files and calling `analyze()`.

### Test transport

- [ ] Create a virtual serial endpoint: Linux/macOS PTY or Windows virtual COM port pair; alternatively inject a minimal pyserial-compatible transport interface in tests.
- [ ] Stream the exact 21 firmware CSV fields plus boot/header/error lines.
- [ ] Match real firmware `Serial.printf` formatting, including signed `irq_us=-1`, unsigned counters, line endings, and optional partial reads.
- [ ] Run the real `python tools/measure.py capture ...` CLI in a child process; do not replace the capture parser with a different implementation.
- [ ] Supply ground-truth event schedule and compare `capture.csv`, `events.jsonl`, and `report.json`.

### Fault matrix

| Fault | Expected behavior |
| --- | --- |
| No serial device available | Explicit connection failure; artifacts and nonzero exit |
| Header-only device | `FAILED/NO_SAMPLES` after startup timeout |
| `# ERROR: Unrecognized/unconfirmed variant` | Fatal device error, no false success |
| Partial line, truncated line, non-ASCII byte, malformed integer | Count malformed/invalid lines, preserve debugging evidence, continue or fail by configured policy |
| Silent but still-open port | Silence warning then timeout/failure if unplanned |
| Host read pause for 10 s | Planned event window; detect and account for observable losses; classify intentionally degraded |
| Serial close/open without MCU reboot | New host connection; same boot context if device persisted |
| MCU reboot during reconnect | New device session, counter and timestamp origin reset |
| Sequence decreases without a documented reboot | Anomaly or new segment with explicit reason |
| Timestamp goes backward within one session | Reject affected anchors; report time-integrity error |
| Random transient serial exceptions | Controlled retry policy and bounded recovery |
| Host process terminated mid-write | Partial outputs detectable; safe rerun with new output path |
| Invalid `--pause-at`, overlap or negative duration | CLI rejects before opening port |
| Duplicate/out-of-order frames | Report distinct anomaly, do not confuse with missing data |

### Capture security and robustness

- [ ] Bound line lengths, message logging sizes, JSON log entries and the retained malformed snippet.
- [ ] Add maximum output directory size or available disk-space check; avoid exhausting the disk during long CSV runs.
- [ ] Keep CLI and file names portable across Windows/Linux/macOS.
- [ ] Flush/close files reliably on exceptions, Ctrl+C, planned stop, and unexpected process termination where possible.
- [ ] Do not silently overwrite previous raw captures or use truncated files as valid training data.

### M3 acceptance

- [ ] Real capture CLI passes all simulated clean and injected-fault scenarios.
- [ ] Device error, silent stream, and no usable samples cannot produce an apparently clean success.
- [ ] Planned disconnects and unexpected failures are differentiated in both JSON and CLI exit code.
- [ ] A clean 120-second virtual serial capture produces consistent expected rate and no observed loss (given simulated nonblocking conditions).

---

## 9. M4 — Clock reconstruction, sensor quality, and motion-model testing (P1)

### M4.1 Ground-truth clock and interrupt model

Represent **at least four clocks/times**, never collapse them:

1. Ideal simulated sensor sampling time `t_sensor`.
2. MCU monotonic `esp_timer_get_time()` with adjustable relative drift and offset.
3. IRQ observed time, including ISR latency and possible missing edges.
4. Host monotonic receive/process time, including serial transport delay and batching.

Add deterministic fault controls:

- [ ] ODR drift: `--odr-ppm` around nominal 208 Hz.
- [ ] IRQ jitter/outliers, rare large stalls, and rate drift during a run.
- [ ] MCU clock drift relative to sensor, `esp_timer` origin reset after simulated reboot.
- [ ] Host transfer latency distribution, backpressure, and variable line batching.
- [ ] Missing watermark edges and continuation drains where `irq_us=-1`.

### M4.2 Robust sample-time fitting

- [ ] Keep basic least squares as a baseline method.
- [ ] Add Huber regression (or another well-justified robust estimator) on independently validated anchors.
- [ ] Reject physically impossible models, backward timestamps, too-few anchors, or across-epoch fits.
- [ ] Report fitted effective ODR, drift in ppm relative to the configured *nominal* rate, median/p95/p99/max residual, number of anchors rejected, clock fit interval, and a timing-confidence grade.
- [ ] Compare reconstructed **times** with simulator ground truth, not only estimated average Hz.
- [ ] For sensor/gyro and accel acquisition offsets, distinguish simultaneous-looking output rows from confirmed common acquisition instant.
- [ ] Keep raw IRQ/read/host timing fields unchanged in output and report which estimator produced `estimated_sample_host_us`.

**Candidate acceptance targets (simulation only; refine after implementation):**

- Clean zero-jitter synthetic fit: ODR estimation within **100 ppm** of injected model after warmup.
- Noisy injected fit: mean effective ODR within **0.1%** of ground truth under a *documented* jitter/outlier profile.
- Median reconstructed time error and p99 error thresholds chosen explicitly per profile from the injected jitter budget. Never set an absolute 1 ms guarantee simply because average ODR fits.
- No fitted times at all for segments without defensible anchors; flag them.

### M4.3 Sensor-value simulator

Create a physically consistent 3D rigid-body model rather than arbitrary uncorrelated sine waves:

- [ ] Trajectory input: position, velocity, acceleration, orientation/quaternion, angular velocity and angular acceleration.
- [ ] Compute specific force at the sensor location from gravity and translational/rotational motion; include a configurable sensor-to-pen-tip lever arm and axis-mount transform.
- [ ] Generate gyro readings from angular velocity; choose and document the sign, handedness and quaternion convention.
- [ ] Apply scale factor, additive bias, Gaussian noise, optional bias random walk, saturation and quantization per channel.
- [ ] Apply configurable digital filter/group delay and sampling phase, including separate accel and gyro delay if needed.
- [ ] Model pen-tip contact state and optional ideal tip trajectory for algorithm-validation **only**; do not pretend this is a measured optical sensor.
- [ ] Include static orientations, 1D translations, fixed-point rotations, combined motions and stroke-like trajectories.

**Required motion tests:**

- Static level: ~1 g norm (within injected bias/noise) and ~0 °/s gyro.
- Static tilted in six orthogonal orientations: recover known gravity projection and axis signs.
- Constant angular velocity: correct sign and integrated orientation under zero-bias ideal model.
- Oscillating rotation with known gyro-to-accel sample delay: recover offset with a confidence score.
- Constant linear acceleration: correct specific-force interpretation; do not conflate gravity and motion.
- Fast transient: configured clipping is detected.
- Cross-session sensor mounting rotation: axis transform restores a common frame when transform is supplied.
- Drift cases: demonstrate that free integration does **not** recover accurate pen-tip position indefinitely; output a drift metric rather than a success claim.

### M4.4 Quality and calibration algorithms

- [ ] Data schema must carry units and source ranges; default conversion for current LSM6DS3 settings: `ax_g = raw * 0.000061`, `gx_dps = raw * 0.0175` (corresponding units for each axis).
- [ ] Assess clipping near signed raw limits, repeated frames, signal dropouts, axis inactivity, suspicious saturation and possible sample-rate inconsistency.
- [ ] Define static calibration protocol: six orientations, stationary gyro bias, noise vs time, repeated identical gestures, and warmup behavior.
- [ ] Treat temperature as optional real measurement; unknown physical temperature response is **not** calibrated from an invented simulator curve.
- [ ] Support raw/scaled/calibrated outputs as separate products, with calibration version and coefficients.
- [ ] Generate annotated histograms/time-series summaries for diagnostic use, but keep quality decisions machine-readable and testable.

### M4.5 Downstream handwriting pipeline compatibility

- [ ] Export a canonical recording schema expected by the broader IMUpen ML project; keep per-session/epoch validity masks.
- [ ] Preserve native 208 Hz captures and timestamps.
- [ ] Implement optional *offline* antialiased resampling to 100 Hz for OnHW-aligned experiments; record method and source/target rates.
- [ ] Never interpolate across unknown packet gaps or FIFO resets.
- [ ] Keep writer ID, recording session, word/letter prompt and contact labels separate from sensor telemetry; these are *not* present in the single-IMU sensor itself.
- [ ] Label `synthetic` versus `real` unmistakably in all dataset manifests. Synthetic recognition scores cannot substitute for real-user evidence.

### M4 acceptance

- [ ] Controlled simulated motion reproduces known sign, scale, phase and timing within explicit test tolerances.
- [ ] Deliberately biased/clipped/noisy runs are identified and their outputs remain traceable to injected truth.
- [ ] Clock fits are never constructed across reboot/epoch boundaries or invalid segments.
- [ ] 208→100 Hz conversion is antialiased, documented, deterministic, and covered by signal-level tests.

---

## 10. M5 — Build and test automation without hardware (P1)

### M5.1 Actual XIAO board build

- [ ] Install and pin `arduino-cli` and the Espressif Arduino ESP32 core.
- [ ] Discover and record the **exact installed board FQBN**, rather than guess it; use `arduino-cli board listall "XIAO ESP32S3"` and `arduino-cli board details --fqbn <confirmed-fqbn>`.
- [ ] Compile `firmware/IMUpen` for that target with configured USB CDC-on-boot option; capture options/build log.
- [ ] Repeat compilation with available supported core versions only when there is an explicit compatibility requirement.
- [ ] Turn compiler warnings into tracked issues, especially format specifiers, integer widths, RTOS types, stack allocation, and ISR signatures.
- [ ] Confirm that use of `Serial` and USB CDC interfaces matches the selected board core/options.

Example **template** (replace placeholder only after board discovery):

```bash
arduino-cli board listall "XIAO ESP32S3"
arduino-cli board details --fqbn YOUR_INSTALLED_FQBN
arduino-cli compile --fqbn YOUR_INSTALLED_FQBN firmware/IMUpen
```

### M5.2 CI jobs and cadence

| Job | Pull request | Nightly | Notes |
| --- | --- | --- | --- |
| Python unit tests | Yes | Yes | Pinned Python 3.10+ and deps |
| Existing C++ decoder test | Yes | Yes | GCC/Clang where feasible |
| Native virtual firmware harness | Yes | Yes | Add ASan/UBSan variant |
| 120-s simulated CSV/report tests | Yes | Yes | Fast deterministic scenarios |
| Virtual serial end-to-end | Yes on Linux | Yes | PTY on CI; Windows separate when available |
| Property-based 100–1,000 seeds | Yes, limited | More seeds | Print seed on failure |
| Real Arduino target compile | Yes | Yes | Requires installed board package, **no device** |
| 30-min virtual stream + faults | Optional | Yes | Streaming output; cap artifact sizes |
| 2–4 h simulated endurance | No | Optional weekly | Model assumptions must be reported |
| Actual sensor electrical/timing tests | No | No | Hardware-only manual/HIL stage |

- [ ] Pin dependency versions; save board package/compile flags in CI artifacts.
- [ ] Provide explicit CI job names and `PASS`, `FAIL`, `EXPECTED_FAILURE` states.
- [ ] Upload test report JSON, failing seeds, minimal relevant traces, and benchmark summaries on failure.
- [ ] Set finite timeouts; avoid generating giant unbounded logs in CI.
- [ ] Make all clean-path tests fail automatically if an existing known bug reappears after correction.

### M5.3 Reproducibility and release artifacts

- [ ] Add `make test` / `just test` or a cross-platform `scripts/test_all.py` wrapper.
- [ ] Add a release checklist: Git SHA, build target/core version, test suite summary, open issues, and a representative clean/fault-injection report.
- [ ] Record test seed, configuration hash, emulator version, schema version, and operating environment in reports.
- [ ] Compare generated JSON in CI using stable field assertions rather than brittle large snapshots containing local paths/times.

### M5 acceptance

- [ ] `python` tests, native C++ tests, virtual serial tests and board-target Arduino compile pass automatically.
- [ ] The results can be reproduced on a second machine or clean CI environment.
- [ ] Tests are demonstrably **hardware-free**, while reports explicitly identify unavailable physical evidence.

---

## 11. Suggested fault-injection test catalog

Use a machine-readable manifest where every case declares setup, injected fault, expected event(s), acceptable uncertainty, and expected quality verdict. The following is the minimum matrix.

| ID | Injection / scenario | Ground-truth assertion | Recording verdict |
| --- | --- | --- | --- |
| T01 | 30 s clean 208 Hz | Correct sequence, FIFO channels and rate | CLEAN |
| T02 | 120 s clean with moderate deterministic IRQ jitter | Timing model consistent with injected truth | CLEAN / bounded uncertainty |
| T03 | 30 min clean high-volume stream | No simulated queue losses, bounded memory | CLEAN |
| T04 | Queue 256 fills under serial pause | Dropped frames observable by later seq/counters | DEGRADED_EXPECTED |
| T05 | Original 8 KiB FIFO overflow | Overrun, epoch increment, lost interval unknown | DEGRADED_EXPECTED |
| T06 | TR-C-sized 4 KiB FIFO overflow | Earlier overflow than original model, if same input | DEGRADED_EXPECTED |
| T07 | FIFO starts at pattern 1..5 | Correct decoding; timestamp anchors justified | CLEAN or UNANCHORED |
| T08 | Deliberate word-order corruption | No silent plausible six-axis frame | FAILED / recovered |
| T09 | Register readback mismatch | Fatal firmware startup error | FAILED |
| T10 | `WHO_AM_I=0x00/0xFF/unknown` | No silent continuation | FAILED |
| T11 | Simulated `0x6A` without confirmed part | Device rejected until explicitly configured | FAILED |
| T12 | Interrupt edge missed, flag remains high | Scheduler handles level and draining correctly | CLEAN if no data loss |
| T13 | INT1 stuck low | Startup/silence failure | FAILED |
| T14 | INT1 stuck high | Avoid infinite unproductive acquisition loop | FAILED / bounded recovery |
| T15 | IRQ jitter 5–50 µs | Fit uncertainty within model expectations | CLEAN / bounded uncertainty |
| T16 | Rare long IRQ delay 10–100 ms | Outliers identified; no false sub-ms guarantee | DEGRADED / confidence reduced |
| T17 | MCU reboot during capture | Distinct session and time origin | DEGRADED_EXPECTED |
| T18 | Host reconnect but MCU stays running | No fictitious reboot or counter reset | CLEAN or DEGRADED_EXPECTED |
| T19 | Host receives duplicate/out-of-order frame | Anomaly flagged, no silent time fitting | FAILED / DEGRADED |
| T20 | Empty CSV / device only errors | Explicit FAILED result and strict-mode error | FAILED |
| T21 | Clipping at ±2 g or ±500 dps | Correct per-axis clipping counts | FAILED clean-data policy |
| T22 | Stationary 6-axis calibration | ~1 g gravity and near-zero gyro before injected bias | CLEAN |
| T23 | Known rotation + 2-frame gyro delay | Lag recovered with confidence/ambiguity note | CLEAN / qualified |
| T24 | Stationary biased/noisy IMU | Bias and noise stats match injected ranges | DEGRADED / calibration-needed |
| T25 | Clean 208→100 Hz conversion | Correct anti-alias response and gap masks | CLEAN |
| T26 | Distorted/missing metadata | Validation refuses ambiguous configuration | FAILED |
| T27 | Two sessions with 9 drops each | Session results [9,9], observed aggregate 18 | DEGRADED_EXPECTED |
| T28 | Last unseen drop after final serial row | Explicit 'unknown/unobserved tail' limitation | UNKNOWN for tail |
| T29 | Truncated serial line plus recovery | Parser logs malformed line and continues safely | DEGRADED |
| T30 | 3D motion with sensor-to-tip offset | Correct inertial lever-arm acceleration in ideal model | CLEAN |

**Acceptance policy warning:** `DEGRADED_EXPECTED` is a correct success result *for the fault-injection test*, never a license to add that recording to clean training data.

---

## 12. Simulation interfaces and example commands

The previous add-on pack already supports these virtual recording scenarios:

- `clean`, `queue_drop`, `fifo_reset`, `reboot`, `irq_outliers`, `nonzero_pattern`, `clipping`.
- Fault knobs: `--odr-ppm`, `--irq-jitter-us`, `--accel-noise-mg`, `--gyro-noise-dps`, `--gyro-bias-dps`, `--accel-bias-mg`.
- A generated `.truth.json` companion to each `.csv`.

After importing `tests/hardware_free/` from the earlier validation ZIP:

```bash
# Deterministic 30-minute simulated recording
python tests/hardware_free/virtual_capture.py \
  --scenario clean --seconds 1800 --output simulated/clean.csv
python tools/measure.py report simulated/clean.csv \
  --output simulated/clean_report

# Queue drops, loss accounting
python tests/hardware_free/virtual_capture.py \
  --scenario queue_drop --seconds 120 --output simulated/drops.csv
python tools/measure.py report simulated/drops.csv \
  --output simulated/drops_report

# Existing known limitation: nonzero pattern does not yet fit a model
python tests/hardware_free/virtual_capture.py \
  --scenario nonzero_pattern --seconds 30 --output simulated/pattern.csv
python tools/measure.py report simulated/pattern.csv \
  --output simulated/pattern_report

# Existing known limitation: clipping not yet a report quality failure
python tests/hardware_free/virtual_capture.py \
  --scenario clipping --seconds 30 --output simulated/clipping.csv
python tools/measure.py report simulated/clipping.csv \
  --output simulated/clipping_report
```

**Note:** The `report` CLI currently generates a report but has no new proposed `--strict` or `--events-jsonl` argument. Add those as implementation tasks; do not run them until implemented.

For future scenario implementations, prefer a simple manifest:

```yaml
scenario_id: T16-irq-stall
source: virtual
seed: 12016
sensor_variant: LSM6DS3
nominal_odr_hz: 208
run_seconds: 120
faults:
  - kind: irq_service_delay
    at_s: 40
    duration_ms: 50
expected:
  recording_verdict: DEGRADED_EXPECTED
  no_silent_reordering: true
  clock_outlier_visible: true
```

The YAML above is **proposed**, not supported by current scripts.

---

## 13. Physical verification after software readiness (M6, requires hardware)

Do these only once the actual breakout is available; do not mark any complete based on native simulations.

### M6.1 Before applying power

- [ ] Photograph PCB front/back; identify exact IC package marking, regulator, wiring, jumpers/pull-ups, supply net and orientation.
- [ ] Check for shorts from 3.3 V to ground and verify CS/SA0 aren't hard-strapped incompatibly with SPI.
- [ ] Verify the breakout's `VIN` accepts the intended supply; if not, use a *documented* safe alternative input.
- [ ] Check wiring against Section 2.1, including common ground and 3.3 V logic limits.

### M6.2 First power-on

- [ ] Measure rail voltage at the sensor supply and confirm safe levels under normal operation.
- [ ] Check WHO_AM_I and readback for all configured registers; reject unknown variant.
- [ ] Confirm FIFO count increases at 208 Hz equivalent, watermark asserts on INT1, CS and SPI work in mode 3.
- [ ] Verify the expected sign and axis mapping by controlled static orientation.
- [ ] Record power-on/reset errors and recovery behavior.

### M6.3 Real acquisition quality

- [ ] Collect 5-minute and 30-minute stationary captures; then repeat controlled handwriting motion captures.
- [ ] Measure effective output rate and jitter uncertainty with the available external reference; examine FIFO/USB gaps.
- [ ] Run read-pause/reconnect tests; compare observed loss and epoch behaviors to simulation, without expecting losslessness under a 10-s block.
- [ ] Record raw accelerometer/gyro distribution, bias, noise, static gravity magnitude, temperature evolution if supported, and clipping frequency.
- [ ] Repeat with Wi-Fi/BLE activity **only if those modes are part of the future configuration**; compare any effect on acquisition timing.
- [ ] If a logic analyzer is available: capture INT1 and dedicated CS at suitable sample rate; validate `sigrok-cli` CSV conversion and IRQ-to-SPI timing.

### M6.4 Hardware acceptance (evidence, not promises)

- [ ] Exact sensor variant and power path documented with board photographs and measurement notes.
- [ ] Register readback and SPI transactions are repeatable.
- [ ] Normal uninterrupted acquisition exhibits no *observed* queue/sequence gaps or recoveries under tested conditions.
- [ ] Stationary calibration and movement checks fall within **measured** and documented tolerances.
- [ ] Deviations between emulator and device traces become regression cases for the emulator.

---

## 14. First handwriting feasibility pilot (M7, requires real pen data)

This milestone is separate from transport and firmware correctness. Keep it narrowly scoped and writer-dependent.

- [ ] Attach the existing single-IMU configuration to a stable, repeatably oriented pen fixture; record its axis transform and approximate sensor-to-tip offset.
- [ ] Define a small labeled prompt set of Latin letters and short words, with clearly recorded start/stop boundaries.
- [ ] Collect repeated sessions on different days, including comfortable and fast writing, and a deliberate reposition/remount test.
- [ ] Use session-disjoint evaluation for personal recognition rather than random windows from the same recording.
- [ ] Train/evaluate using the existing IMUpen recognition project and OnHW work as a *separate* ML workflow; do not infer feasibility from synthetic gesture classification.
- [ ] Compare raw 208 Hz with antialiased 100 Hz as a controlled experiment.
- [ ] Quantify per-letter and per-word errors, latency, performance variability across sessions, and sensitivity to handwriting speed.
- [ ] Decide whether a second IMU, force/contact sensor, or optical displacement reference is justified by specific observed failures.

**Go/no-go decision:** Upgrade hardware only when you can describe which error mechanism additional sensing is supposed to improve and have a test capable of measuring that improvement.

---

## 15. Work packages suitable for a coding agent

### Ticket list in recommended order

| Ticket | Title | Depends on | Completion evidence |
| --- | --- | --- | --- |
| IMU-001 | Import virtual pack and freeze baseline | none | 5+8 Python tests, C++ decoder and 6 native scenarios run |
| IMU-002 | Add report schema version + quality state machine | IMU-001 | Fatal/silent/empty cases fail correctly |
| IMU-003 | Fix per-session counters and event accounting | IMU-002 | Multi-session synthetic truth assertions |
| IMU-004 | Correct timing-anchor pattern handling | IMU-001 | Tests for patterns 0–5 and uncertainty |
| IMU-005 | Add clipping/plausibility/units checks | IMU-002 | Synthetic clipping and known static cases |
| IMU-006 | Add full target Arduino compile workflow | IMU-001 | CI board-target compile log |
| IMU-007 | Extend FIFO/register simulator | IMU-001 | Fault matrix native tests pass |
| IMU-008 | Add deterministic scheduler/backpressure | IMU-007 | Seeded interleavings and queue invariants |
| IMU-009 | Implement simulated serial end-to-end | IMU-002, IMU-007 | Real capture CLI health/fault tests |
| IMU-010 | Robust clock fitting and truth comparison | IMU-004 | Rate/time error metrics under jitter |
| IMU-011 | Add 3D motion and calibration model | IMU-005 | Physics-based test suite |
| IMU-012 | Export canonical 208/100 Hz valid-mask dataset | IMU-011 | Resampling and boundary tests |
| IMU-013 | Complete CI, nightly stress and docs | IMU-006..012 | Reproducible software-only release |
| IMU-014 | Run physical electrical bring-up | IMU-013 | Scope/logic/device measurements |
| IMU-015 | Run personal handwriting pilot | IMU-014 | Session-disjoint real-data benchmark |

### Coding-agent execution rules

1. Start with a repository inventory and a baseline run; avoid unrelated refactors.
2. For every change, add a failing reproduction first, then implementation, then passing positive regression.
3. Keep code changes incremental and maintain the present CSV field order until a documented protocol migration.
4. Do not use mocks to claim properties they omit; record unmodeled hardware features.
5. Report exact files changed, tests executed, unexpected behavior, and unverified assumptions at the end of each ticket.
6. Never quietly lower a test threshold to get green CI. Document changes to thresholds and their rationale.
7. Maintain the distinction between observed samples, inferred values, and unknown lost data throughout the pipeline.

### Suggested PR boundaries

- **PR A:** Baseline import, CI smoke tests, reproducible board compile.
- **PR B:** Quality verdict, counter aggregation, events, clipping regression fixes.
- **PR C:** FIFO pattern/anchor corrections and robust timing tests.
- **PR D:** Native FIFO emulator and deterministic task scheduling.
- **PR E:** Virtual serial and complete host integration.
- **PR F:** 3D motion, calibration validation and 100 Hz exporter.
- **PR G:** Documentation, nightly endurance, software readiness review.

---

## 16. Project-wide acceptance checklist

### Hardware-free software readiness: all required

- [ ] Original five Python regression tests and decoder test remain passing.
- [ ] Existing eight virtual Python cases and six firmware-native cases remain passing after expected-defect tests are converted to fixed-behavior tests.
- [ ] Real Arduino board-target firmware compile passes and build options are recorded.
- [ ] Known silent/empty/error-only capture defect is fixed with a strict nonzero failure outcome.
- [ ] Per-session counters preserve true session boundaries and don't double-count loss.
- [ ] All FIFO starting patterns are either correctly handled or explicitly rejected from clock fitting.
- [ ] Clean and injected-clipping test runs receive appropriate, distinct recording-quality verdicts.
- [ ] Robust timing and phase-alignment metrics are verified against explicitly modeled truth.
- [ ] Native FIFO and RTOS-interleaving tests cover delayed FIFO service, overrun, output backpressure, resets and malformed transport.
- [ ] Real `capture` CLI has been exercised through a virtual serial endpoint.
- [ ] A virtual 30-minute clean stream and multiple fault streams produce reproducible reports.
- [ ] Generated datasets have manifests, units, sample rate, source type, quality flags and discontinuity masks.
- [ ] 100 Hz export uses offline anti-aliasing and never crosses unknown gaps.
- [ ] All CI jobs are repeatable on a clean machine, and no unverified physical claims appear in release notes.

### Physical readiness: requires a later separate sign-off

- [ ] Exact breakout power path and IC variant confirmed.
- [ ] SPI transactions, INT1 watermark, FIFO order, real voltage levels and clock behavior measured.
- [ ] Sensor calibration/noise/clipping observed on actual pen movement.
- [ ] Actual serial recording and stress tests meet documented criteria.

### Recognition readiness: later separate sign-off

- [ ] Labeled real pen recordings collected across independent sessions.
- [ ] Writer-dependent models evaluated on held-out sessions and prompts.
- [ ] Additional sensing considered only in response to real measured deficiencies.

---

## 17. Primary references

### Local source documents

- `README.md` in the uploaded project — firmware wiring, recording, analyzer interpretation and limitations.
- `IMUpen-hardware-free-test-plan(1).md` — virtual validation pack capabilities and known gaps.
- `IMUpen-hardware-specifications(1).md` — pinout, sensor variants, register config and electrical limitations.

### Manufacturer and developer documentation

- [ST LSM6DS3 datasheet (original)](https://docs.arduino.cc/resources/datasheets/LSM6DS3-datasheet.pdf)
- [ST LSM6DS3TR-C datasheet](https://www.st.com/resource/en/datasheet/lsm6ds3tr-c.pdf)
- [ST LSM6DS3TR-C application note AN5130](https://www.st.com/resource/en/application_note/dm00472670-lsm6ds3tr-c-always-on-3d-accelerometer-and-3d-gyroscope-stmicroelectronics.pdf)
- [Seeed XIAO ESP32-S3 getting started and pinout](https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/)
- [Seeed XIAO ESP32-S3 pin multiplexing](https://wiki.seeedstudio.com/xiao_esp32s3_pin_multiplexing/)
- [Arduino CLI documentation](https://docs.arduino.cc/arduino-cli/getting-started/)

**Reference policy:** Datasheets define documented behavior and limits. The exact breakout board, electrical measurements, actual timing distribution, handwriting quality, and end-to-end user experience remain unverified until the relevant physical tests are performed.

---

## 18. Final delivery definition

The hardware-free phase is complete when a coding agent can check out the project on a clean machine, compile the real XIAO target, run native and Python virtual tests, inject known faults through simulated FIFO and serial interfaces, generate versioned reports, and demonstrate through assertions that every declared fault was detected or explicitly marked unknown. The software must never quietly certify a recording with missing samples, invalid timing anchors, an unrecognized sensor, or an empty capture as clean.

**Only after that milestone should the first physical board and breakout be used to validate the remaining electrical and MEMS-specific unknowns.** Completing this plan makes the acquisition software trustworthy under tested assumptions; it does not demonstrate reliable handwriting recognition on a future pen.
