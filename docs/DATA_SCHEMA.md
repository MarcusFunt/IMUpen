# Canonical and resampled datasets

Raw `imupen_fifo_v1` CSV remains the immutable acquisition artifact. Convert it with:

```powershell
python tools/dataset.py canonicalize capture.csv --output canonical.csv --source-type real --config configs/xiao_lsm6ds3_spi_208hz.yaml
python tools/dataset.py resample canonical.csv --output canonical-100hz.csv --target-rate-hz 100
```

`--source-type` is required. A synthetic file must be labeled `synthetic`; the exporter never infers that it is real. Each canonical CSV has an adjacent `.manifest.json` containing source and output hashes, schema version, units and raw ranges, configuration/calibration hashes, timing source, segment count, and explicit lack of physical verification.

Canonical rows retain raw signed 16-bit values and add uncalibrated engineering units using the configured LSM6DS3 sensitivities (`0.0175 deg/s/LSB` for gyro and `0.000061 g/LSB` for accelerometer). Optional calibration JSON contains a version, a three-value `gyro_bias` in deg/s, an `accel_bias` in g, and optional 3×3 `gyro_matrix`/`accel_matrix` coefficients. Corrected values are written to separate `*_cal_*` columns; raw and scaled values are unchanged. Calibration is `unknown` when no calibration file is supplied. Temperature compensation is intentionally unsupported without physical characterization.

`segment_id` changes at session, boot, FIFO epoch, sequence gap, or timestamp discontinuity. `validity_mask` bits 0–2 identify valid gyro x/y/z and bits 3–5 valid accelerometer x/y/z; value 63 means all six axes are present. Writer ID, prompt, contact labels, and other human metadata belong in a separate metadata JSON object in the manifest, not in the sensor telemetry columns.

Optional 100 Hz output uses an offline Blackman-windowed sinc low-pass resampler with a cutoff at 45 Hz. It estimates the source rate per valid segment, rejects target rates at or above the source Nyquist limit, and discards filter-edge samples where the full kernel is unavailable. It filters each segment independently, so no value is interpolated across a gap, reboot, or FIFO epoch. The resampled sidecar points to and hashes the canonical source manifest and preserves its real/synthetic label.

## Calibration capture protocol

For future physical sessions, keep a separate calibration recording with six labeled static orientations (one dominant positive/negative axis at a time), a stationary gyro-bias interval, warm-up timestamp, duration, sample rate, and temperature only when measured by a real sensor. Record repeated placements to estimate noise and repeatability. Keep gesture trials separate from calibration and label repeated-gesture prompts in metadata. These software tools can apply supplied coefficients; they do not claim a calibration is valid until physical reference measurements are reviewed.

Synthetic motion, timing recovery, and alias rejection tests validate code behavior only. They do not substitute for M6 sensor checks or M7 handwriting recordings.

## Export resource note

A local 30-minute synthetic end-to-end export (374,400 source rows) completed and produced 179,976 rows at 100 Hz. On the Windows verification machine, the resampling process peaked at approximately 986 MB working memory because the current implementation retains segment and output rows in memory. Budget memory for longer sessions or constrained devices.
