# IMUpen Studio GUI design

## Purpose

Design a local-first workstation for recording IMUpen sessions, reviewing data quality, exporting canonical datasets, and eventually running handwriting experiments. The current repository has command-line capture, report, and dataset tools; it does not contain a GUI or recognition trainer.

The runnable desktop concept is `ui/imupen_studio.py`, built with PySide6. It demonstrates the screen layout and navigation only: it does not connect to a serial device, read session files, write captures, export datasets, or train a model. The earlier browser wireframe remains at `ui/IMUpen-Studio-Prototype.html` for quick reference.

## Intended user

A researcher or project owner collecting repeated IMU recordings and checking whether sessions are suitable for later analysis. The interface should make recording provenance, device state, and data-quality limitations visible without asking the user to interpret raw counter fields.

## Information architecture

| Section | Primary job | Initial implementation boundary |
| --- | --- | --- |
| Overview | Show project readiness and latest session status | Read-only summary from local reports |
| Record | Select a serial device, enter trial metadata, capture a raw session | Wrap `tools/measure.py capture`; preserve the raw CSV and event log |
| Sessions | Find and inspect prior captures | Read `report.json`; never rewrite raw captures |
| Dataset | Create canonical and optional 100 Hz derivatives | Wrap `tools/dataset.py`; require an explicit real/synthetic source label |
| Train & evaluate | Configure grouped splits and review model results | Future M7 work; disabled until a real recognition backend exists |

## Main recording flow

1. Confirm local workspace and device state.
2. Select an available serial port and verify the expected firmware header and sensor ID.
3. Enter a pseudonymous writer code, prompt, repetition, session, and optional notes. Keep these labels in a sidecar manifest, never in raw telemetry columns.
4. Review the target ODR, source type, sample range, and output folder.
5. Start capture. Show elapsed time, rows received, last sample age, observed sequence gaps, queue-drop counter, FIFO epoch, and event status.
6. Stop or finish the requested duration. Preserve `capture.csv`, `events.jsonl`, and `report.json` as separate artifacts.
7. Show the report verdict and its reason codes. Offer reanalysis and dataset export only after the raw artifact is closed.

## Quality and safety rules in the UI

- Clearly distinguish `CLEAN`, `DEGRADED`, `DEGRADED_EXPECTED`, and `FAILED`; do not use color alone.
- A simulator preview must carry a persistent `SYNTHETIC` label.
- Physical sensor identity and electrical checks stay `NOT VERIFIED` until measured on the actual board and breakout.
- Explain that observed sequence gaps and cumulative counters do not measure every lost physical sample.
- Keep `training` intent separate from evidence that a model exists or has been trained.
- Use session- or writer-grouped train/test splits. Never offer random row splitting as the default because adjacent IMU samples leak near-duplicate motion across partitions.
- Keep writer identity and prompt metadata separate from sensor telemetry. Default to pseudonymous IDs and local storage.

## Visual direction

- Desktop-first, responsive down to tablet; capture controls remain usable on narrow screens.
- Warm neutral workspace with a dark blue-green rail and a restrained teal accent. Amber marks unresolved checks; red is reserved for failed captures.
- Content-first tables, quality reports, and a rotatable 3D device preview; use dividers and spacing before adding cards or shadows.
- Large readable status labels, explicit units, fixed-width numeric values, keyboard-visible focus, and form labels.
- One primary action per page: `Prepare recording`, `Start capture`, `Export canonical dataset`, or `Configure evaluation`.

## Future backend seams

- `DeviceService`: list ports, open/close transport, surface sensor and firmware identity.
- `CaptureService`: invoke the existing capture state machine and stream progress/events.
- `ReportService`: render versioned quality reports without changing raw inputs.
- `DatasetService`: invoke canonical export and resampling with source/configuration manifests.
- `ExperimentService` (M7): own model configuration, grouped splits, training, evaluation, and reproducibility metadata. This service is not present in the repository yet.

## Prototype status

The PySide6 artifact is a desktop interaction mock with Overview, Record, Sessions, Dataset, and Train & evaluate views. The Record page includes an animated 3D-style board, device axes, a separate world-frame compass, and a ground grid. Dragging orbits the camera and pauses the generated demo pose so it can be inspected; it does not change the sensor pose. Demo motion is never represented as live sensor output. Navigation is functional. Device, capture, export, and training actions are intentionally unavailable until their backends are implemented. It should be run from the project virtual environment after installing `requirements-ui.txt` with `tools/run-gui.ps1`.
