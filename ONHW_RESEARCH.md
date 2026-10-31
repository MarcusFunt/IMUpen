# OnHW-Words500 research guide

This guide is the current research workflow for IMUpen. It supersedes the hardware-first sequencing in earlier milestone documents. The goal is to compare sensor profiles and learning methods on recordings that already exist before drawing conclusions about future hardware.

## Research questions and experiment factors

Keep these dimensions separate in experiment configurations and reports:

| Dimension | Examples | What it changes |
| --- | --- | --- |
| Sensor modality | FG, RG, FRG, FGM, FRGP | Measured input channels |
| Neural architecture | M4 CNN-BiLSTM-CTC, M5 ResNet-BiLSTM-CTC | Model structure |
| Training method | scratch CTC, supervised pretraining, augmentation, SSL, contrastive | Initialization or objective |
| Evaluation protocol | official WD, official WI, custom W × V | Writer and vocabulary exposure |

OnHW-Words500's measured channel order is `[time, 13]`:

| Group | Measurement | Channel indices | Units |
| --- | --- | --- | --- |
| F | Front accelerometer XYZ | 0–2 | m/s² |
| R | Rear accelerometer XYZ | 3–5 | m/s² |
| G | One shared gyroscope XYZ | 6–8 | deg/s |
| M | Magnetometer XYZ | 9–11 | mT |
| P | Scalar pen-tip force | 12 | N |

The registry generates every nonempty subset of those five measured groups. Profile channels keep the dataset's original order.

| Groups | Profile ID | Input channels |
| --- | --- | ---: |
| F | `front_acc3` | 3 |
| R | `rear_acc3` | 3 |
| G | `gyro3` | 3 |
| M | `mag3` | 3 |
| P | `force1` | 1 |
| FR | `dual_acc6` | 6 |
| FG | `front_acc_gyro6` | 6 |
| FM | `front_acc_mag6` | 6 |
| FP | `front_acc_force4` | 4 |
| RG | `rear_acc_shared_gyro6` | 6 |
| RM | `rear_acc_mag6` | 6 |
| RP | `rear_acc_force4` | 4 |
| GM | `shared_gyro_mag6` | 6 |
| GP | `shared_gyro_force4` | 4 |
| MP | `mag_force4` | 4 |
| FRG | `dual_acc_shared_gyro9` | 9 |
| FRM | `front_acc_rear_acc_mag9` | 9 |
| FRP | `front_acc_rear_acc_force7` | 7 |
| FGM | `front_acc_gyro_mag9` | 9 |
| FGP | `front_acc_shared_gyro_force7` | 7 |
| FMP | `front_acc_mag_force7` | 7 |
| RGM | `rear_acc_shared_gyro_mag9` | 9 |
| RGP | `rear_acc_shared_gyro_force7` | 7 |
| RMP | `rear_acc_mag_force7` | 7 |
| GMP | `shared_gyro_mag_force7` | 7 |
| FRGM | `front_acc_rear_acc_shared_gyro_mag12` | 12 |
| FRGP | `front_acc_rear_acc_shared_gyro_force10` | 10 |
| FRMP | `front_acc_rear_acc_mag_force10` | 10 |
| FGMP | `front_acc_shared_gyro_mag_force10` | 10 |
| RGMP | `rear_acc_shared_gyro_mag_force10` | 10 |
| FRGMP | `full13` | 13 |

`RG` reuses OnHW's only gyroscope; it does not mean a rear-mounted gyroscope. `FRGM` contains two accelerometers, one gyroscope, and one magnetometer. It is not two independent six-axis IMUs. The profile registry rejects unrecorded hardware channels. Derived signals belong in separate feature-engineering experiments.

The standard OnHW time base is 100 Hz. Normalization is fit on training samples only. Profile selection preserves the original 13-channel record and projects a separate model input. Profile metadata, channel units, sample rate, normalization, model dimensions, split IDs, and archive digest are written to reports and checkpoints.

## Official Words500 protocols

The loader preserves published train and validation membership and checks writer and word leakage rules:

| Protocol | Writers | Vocabulary | Interpretation |
| --- | --- | --- | --- |
| Writer-dependent (WD) | Writers occur in both partitions | Per writer, training and validation word identities are disjoint; published target is 400 training and 100 validation words | Familiar writers, held-out words |
| Writer-independent (WI) | Train and validation writers are disjoint | Both populations draw from the shared 500-word vocabulary | Unseen writers, familiar words |

Reports include observed vocabulary overlap and WD distinct-word counts so a reduced or synthetic archive is not mistaken for the full published benchmark. WD and WI scores are different tasks. Their difference is not a causal estimate of writer familiarity because word exposure changes too.

### Recording identity and duplicates

The loader does not treat writer IDs or split-local positions as globally stable recording IDs. When the consumed archive fields provide no authoritative recording ID, `source_id` remains null and `sample_id` combines archive name, split family, fold, partition, and local index. That guarantees identity within that partition namespace only. The full 13-channel array also gets a SHA-256 fingerprint of canonical float32 little-endian values. Exact fingerprints are checked across train/query partitions; exact matches can help identify unchanged recordings across folds, but they cannot reconstruct identity after signal transforms. No session identity is inferred.

## Minimal official-fold supervised experiment

Create a JSON config with `data`, `fold`, `sensor_profile`, `seed`, `epochs`, and `batch_size`; `model_type` selects M4 or M5 and defaults to M5. Optional fields are `device`, `num_threads`, sample caps, and `protocol`. Example:

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

Run:

```bash
smartpen-hwr run-experiment --config experiments/onhw_m5.json --output results/onhw
```

Use `--dry-run` to validate the archive, fold, profile, split, and CTC-feasible records. The runner fingerprints the config and dataset, saves a selected checkpoint (minimum validation WER, then validation loss), evaluates that checkpoint, writes JSON reports, resumes epoch-level training after interruption, and skips successful runs whose artifact hashes still match.

### M4/M5 sensor and fold sweeps

The sweep runner expands `sensor_profiles`, `model_types`, `folds`, and `seeds` as a Cartesian product. `experiments/onhw_primary_sensor_screen.json` registers the seven profiles needed for the six prespecified sensor contrasts. Validate every planned run first:

```bash
smartpen-hwr run-sweep --config experiments/onhw_primary_sensor_screen.json --output results/onhw --dry-run
```

Remove `--dry-run` to train. Each child run is independently resumable; the sweep manifest retains failures and the aggregation reports per-profile mean and standard deviation plus paired differences for registered sensor contrasts where both runs exist. Set `"sensor_profiles": "all"` for the exploratory 31-profile expansion. These summaries are development evidence; confirmation still requires a frozen comparison on data not used to choose the profiles.

For a pipeline check on a small sample, add `max_train_samples` and `max_validation_samples` to JSON. Such reports are labeled sample-capped smoke tests. They are not accuracy benchmarks and must not be used to choose the winning profile.

## Matched writer × vocabulary experiment

For a custom development analysis, run:

```bash
smartpen-hwr run-factorial --data OnHW-words500_indep.zip --output results/factorial --fold 0 --target-writers 12 17 --query-per-writer 10 --support-per-writer 5 --sensor-profile front_acc_gyro6 --epochs 10
```

The builder fixes target-writer query recordings across all four cells. W1 support is sampled outside that query set and held fixed across W1V0 and W1V1. V1 requires the query word identities in training through non-target writers. W0 excludes target writers; V0 excludes query words from every training label. Training counts are matched across cells when the available pool allows it. Manifests record exact query/training IDs, support counts, overlap, and seed. Runs verify query IDs and references before calculating paired macro-writer WER/CER effects and writer-level bootstrap intervals.

This command pools records from the selected official fold to build its custom train/query pools. It is a development experiment, not an official WD/WI score or untouched confirmation set. The target's exact word in support is not required for W1V1; V1 means that the query word identity was observed somewhere in the training corpus.

### Writer-transfer and personalization split manifests

Generate exact P1–P6 split manifests for one target writer with:

```bash
smartpen-hwr build-adaptation-manifests --data OnHW-words500_indep.zip --output results/writer-12 --fold 0 --target-writer 12 --query-count 10 --support-budgets 1 5 10 25 50 100
```

The manifests cover single-writer scratch, cross-writer zero-shot, multi-writer pretraining with few-shot support, the scratch/frozen/partial/full fine-tuning comparison grid, unlabeled SSL adaptation, and supervised contrastive training. They reuse one fixed target query set, list each source/support/query ID, report writer and vocabulary overlap, and exclude query traces from labeled and unlabeled training. Budgets that do not fit are recorded as skipped. These are split manifests and leakage checks; protocol-specific OnHW adaptation training is not wired into the common runner yet. The source archive does not expose genuine session IDs through the current loader, so the manifests make no session-split claim.

## Outcomes and statistical reporting

For Words500, WER is the primary recognition metric. Reports also include CER, exact-word accuracy, edit counts and denominators, per-writer metrics, macro-writer metrics, paired sample outputs, parameter count, and available timing/memory provenance. M4/M5 comparisons keep fold, samples, profile, preprocessing, optimizer settings, and seed matched. Writer-level bootstrap intervals resample writers rather than treating every recording as an independent writer observation. The five official folds are still a small number of clusters.

Prespecified primary sensor contrasts are FG vs RG, FG vs FRG, FG vs FGM, FG vs FGP, FRG vs FRGM, and FRG vs FRGP. Rank all 31 profiles as exploratory until a selected profile is confirmed on held-out data or a nested evaluation. Preserve failures and negative results. If every fold has influenced selection, report development/cross-validation evidence rather than an untouched final test.

Force may help recognize motion or contact structure within pre-segmented labeled words. Words500 does not establish online word-boundary detection in continuous handwriting. Findings concern OnHW's actual sensor placements and do not directly establish the value of an unbuilt dual-IMU pen or a different sensor placement.

## Methods and remaining scope

M4, M5, M6-A/B/C/D source implementations and their focused commands remain available. The common official runner and sweep scheduler handle M4/M5 supervised CTC; the matched factorial runner handles four matched CTC cells. OnHW single-writer/few-shot model training, broad M6 strategy sweeps, and confirmation-stage orchestration remain future work. The existing M6 personalization commands primarily exercise the prototype's personal six-axis recordings and should not be presented as full OnHW writer-generalization studies.

Hardware validation, sensor simulation, desktop GUI work, and continuous-stream segmentation are outside the current OnHW research milestone.
