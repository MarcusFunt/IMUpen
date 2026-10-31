# IMUpen

IMUpen is a research platform for inertial handwriting recognition. Its next milestone is a reproducible experimental sweep on the measured OnHW-Words500 recordings. The physical pen remains a separate prototype and data-collection track; the immediate research work does not depend on choosing new hardware.

## Research workflow

The recognition code keeps the complete OnHW sensor array and selects one declared sensor profile per experiment. It supports the official writer-dependent and writer-independent folds, variable-channel M4/M5 models, train-only normalization, checkpoint provenance, greedy CTC decoding, and machine-readable metrics.

Start with the [OnHW research guide](ONHW_RESEARCH.md) for the 13-channel schema, all 31 supported profiles, official split interpretation, experiment commands, and limitations. Existing method notes cover [M4](ONHW_M4.md), [M5](M5_RESNET_BILSTM.md), [M6-A](M6A_ONHW_PRETRAINING.md), [M6-B](M6B_IMU_AUGMENTATION.md), [M6-C](M6C_TS2VEC_PERSONAL_SSL.md), and [M6-D](M6D_ECHWR_PERSONAL_CONTRASTIVE.md).

Install the optional ML dependencies with `python -m pip install -e ".[ml]"`. A minimal JSON configuration for one M4 or M5 official-fold experiment is:

```json
{
  "data": "C:/datasets/OnHW-words500_indep.zip",
  "fold": 0,
  "sensor_profile": "front_acc_gyro6",
  "model_type": "resnet_bilstm_ctc",
  "seed": 1337,
  "epochs": 10,
  "batch_size": 32,
  "device": "auto",
  "num_threads": 4,
  "protocol": "writer_independent"
}
```

Run it with `smartpen-hwr run-experiment --config experiments/onhw_m5.json --output results/onhw`. Add `--dry-run` to validate the archive, fold, profile, and CTC-feasible sample plan. Completed runs are skipped when their artifact hashes still match; interrupted runs resume from the last completed epoch. Sample caps are available for pipeline checks, and those reports are labeled as smoke tests rather than accuracy benchmarks.

Run a configured profile × model × fold × seed screen with `smartpen-hwr run-sweep`; `experiments/onhw_primary_sensor_screen.json` covers the six prespecified sensor comparisons. Use `--dry-run` before training. The sweep preserves failures and writes descriptive summaries and paired sensor differences.

For a matched writer × vocabulary development experiment, use `smartpen-hwr run-factorial` with target writer IDs. The runner fixes the query recordings across W0V0, W0V1, W1V0, and W1V1, writes cell manifests, and reports paired macro-writer effects with writer-level bootstrap intervals. These custom runs are clearly marked and do not replace either official benchmark.

## Prototype collection track

The repository also contains firmware and Python collection tools for a Seeed Studio XIAO ESP32-S3 and standalone ST LSM6DS3. That hardware setup is not the OnHW DigiPen and its recordings do not establish the value of OnHW sensor placements. The [Sprint 1 document](SMARTPEN_SPRINT1.md) is retained as a historical collection milestone. Hardware integration and continuous-stream word segmentation remain future work.

## Development

Install test and GUI dependencies with `python -m pip install -e ".[test,gui]"`. Run `python -m pytest -q` for the host test suite. The OnHW dataset is separate from this MIT-licensed repository and must be obtained from its maintainers.

See [firmware/README.md](firmware/README.md) for the USB protocol and collector instructions. The timestamp grid is nominally 208 Hz for the prototype firmware; OnHW-Words500 is recorded at 100 Hz. Keep those data sources and rates distinct.

## License

MIT; see [LICENSE](LICENSE).
