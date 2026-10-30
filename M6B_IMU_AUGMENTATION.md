# Milestone M6-B: six-axis IMU augmentation ablations

M6-B adds reproducible, training-only augmentation experiments for the M5 ResNet-BiLSTM-CTC recognizer. It runs a baseline and five independent ablations on the same official OnHW writer-independent fold, sample indexes, seed, model, optimizer, epoch count, and training-only normalization. The optional combined run is scheduled after every isolated experiment.

The recorder, USB protocol, and ESP32-S3 firmware are unchanged. OnHW-Words500 is licensed separately from this repository under CC BY-NC 4.0; follow the attribution and non-commercial terms in [ONHW_M4.md](ONHW_M4.md). The dataset archive and model checkpoints are not committed.

## Ablations

| Name | Training-only transform |
| --- | --- |
| `temporal_stretch` | Resample the sequence to 90–110% of its original duration, with labels unchanged. A sampled stretch that would make CTC alignment infeasible falls back to the original record. |
| `gaussian_noise` | Add independent per-axis, per-timestep Gaussian noise with standard deviation equal to 1% of that axis's standard deviation on the selected original training samples. |
| `sensor_bias_gain` | Apply a per-axis gain sampled from 0.97–1.03 and a per-axis constant bias sampled with standard deviation 2% of the selected training-axis standard deviation. |
| `rotation_3d` | Sample roll, pitch, and yaw independently from −5° to +5°. Apply the same rigid rotation to the accelerometer and gyroscope xyz vectors. |
| `sequence_concat` | Add seeded pairs of sequences from the same writer and session in the fitting partition, with an 8-frame linear transition and concatenated labels. OnHW's alphabet has no space token, so its default transcript delimiter is empty. |

Each single transform is trained independently from a fresh initialization. Online transforms are re-sampled deterministically by seed, epoch, and record index. Concatenated records are made only from selected training records. Normalization is fitted to the original selected training records before augmentation, then held fixed across variants; the validation set stays original and is never augmented.

## Run

```bash
python -m pip install -e ".[ml,test]"

python -m smartpen.hwr.cli augment-ablate \
  --data datasets/OnHW-Words500_indep.zip \
  --output runs/m6b-onhw-fold0 \
  --fold 0 --epochs 30 --batch-size 24 --seed 2026 --device cuda
```

This runs baseline plus all five individual transforms. Add `--include-combined` to run the combined experiment after the individual variants. For an implementation smoke test on a deterministic subset, add `--max-train-samples 64 --max-validation-samples 64` and reduce epochs. The runner writes one checkpoint and metrics file per variant plus `augmentation_ablation.json`, which records selected indexes, normalization, CER/WER counts, augmentation parameters, concatenation counts, and fairness checks.

## Evaluation and limitations

The report uses the existing corpus micro-averaged character error rate (CER) and word error rate (WER). OnHW samples are single words, so WER is the fraction of held-out word samples transcribed incorrectly. The official writer-independent validation writers are never used to construct training augmentations. Sample-capped smoke runs test execution only; they are not benchmark results and do not establish that an augmentation improves recognition. A full comparison should be repeated across all five official folds before drawing a general conclusion.

This measures host-side performance on OnHW's pen and acquisition setup. It does not establish accuracy on the XIAO ESP32-S3 + LSM6DS3 recorder; evaluate on accepted personal recordings split by writer/session before making a transfer claim. No personal recordings are included in this repository.

## Evidence

The real OnHW smoke run in [`reports/onhw_m6b_augmentation_smoke.json`](reports/onhw_m6b_augmentation_smoke.json) used official fold 0, seed 2026, 16 training and 16 validation samples, one epoch, CPU, and the default 4,676,540-parameter M5 model. All six isolated/baseline variants and the optional combined variant completed. Each measured CER 100% (93/93 characters) and WER 100% (16/16 words); the selected sample is too small and training too short to infer any augmentation effect. The sequence-concatenation variants added 9 training sequences. The host's PyTorch `2.13.0+xpu` package emitted an unsupported integrated-GPU warning during CPU backward; the run completed on CPU. Synthetic CI tests cover deterministic transforms, rigid-rotation consistency, writer/session-safe concatenation, validation isolation, and ablation fairness; they do not measure handwriting accuracy.
