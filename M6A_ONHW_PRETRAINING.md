# Milestone M6-A: OnHW encoder pretraining and personalization

M6-A uses the M5 ResNet-BiLSTM encoder trained on OnHW-Words500 to initialize personal handwriting recognition. It then fine-tunes the encoder on accepted XIAO recorder recordings and compares that run with the same model trained from a fresh initialization. The comparison shares the personal session split, training-only normalization, vocabulary, seed, optimizer, batch size, epoch count, and validation metrics.

The experiment is optional host-side PyTorch code. It does not change the recorder, USB protocol, or firmware.

## 1. Pretrain on OnHW

Download the writer-independent OnHW-Words500 archive from the [Fraunhofer IIS dataset page](https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html). The adapter selects the same six channels as M5: front accelerometer x/y/z and gyroscope x/y/z.

```bash
python -m pip install -e ".[ml,test]"

python -m smartpen.hwr.cli pretrain \
  --data datasets/OnHW-Words500_indep.zip \
  --output runs/onhw-m6a-pretrain \
  --fold 0 --epochs 30 --batch-size 24 --seed 2026 --device cuda
```

This trains the M5 ResNet-BiLSTM-CTC model on the selected official writer-independent fold and saves `model.pt` plus `metrics.json`. The checkpoint is tagged `onhw_encoder_pretraining`. Sample caps are available for pipeline checks but do not establish benchmark performance.

**OnHW is licensed CC BY-NC 4.0.** Attribution is required, and the license limits use and adaptations to non-commercial purposes. Do not use this pretraining path or derived checkpoints for commercial purposes under that license. Dataset terms and citation are recorded in [ONHW_M4.md](ONHW_M4.md).

## 2. Prepare personal recordings

Use the existing prompted recorder to collect accepted samples across multiple sessions. Each accepted recording needs a correct prompt and a distinct `session_id`; rejected recordings are ignored. The M6-A loader reads `metadata/**/*.json` and the `axes` array in each referenced `processed/**/*.npz`. It uses only the six stored accelerometer/gyroscope channels, keeps raw files unchanged, and fits normalization on training sessions only.

The default character set supports `A-Z`, `a-z`, German `ÄÖÜäöüß`, Danish `ÆØÅæøå`, spaces, digits, and common punctuation. Set `--charset` to a unique string containing every character in your prompts if your text needs a different vocabulary. Its final CTC class is reserved as blank.

At least two accepted writer/session groups are required. The deterministic split keeps each writer/session group entirely in one partition; the validation partition contains whole unseen sessions. The generated `session_split.json` records every accepted sample ID and session assignment. To reuse the exact split, pass it with `--split-manifest`; that manifest must cover the current recording set exactly.

## 3. Fine-tune and compare

```bash
python -m smartpen.hwr.cli personalize \
  --recordings recordings \
  --pretrained-checkpoint runs/onhw-m6a-pretrain/model.pt \
  --output runs/m6a-personal-comparison \
  --validation-fraction 0.2 --epochs 30 --batch-size 16 \
  --seed 2026 --device cuda
```

The pretrained run copies compatible M5 CNN and BiLSTM weights, then initializes a fresh CTC classifier for the personal character set. The full network is fine-tuned. The scratch run uses the same architecture, seed, and newly initialized classifier without loading OnHW weights. The tool checks that both runs use the same personal sample IDs, vocabulary, and train-only normalization.

The output directory contains:

- `session_split.json`: persistent train/validation sample and session assignments.
- `from_scratch/` and `onhw_pretrained_encoder/`: checkpoint and per-run metrics files.
- `comparison.json`: CER/WER with edit counts, run settings, transfer source, fairness checks, and peak CUDA memory when available.

The split report contains personal sample IDs and session labels. Keep the run directory private if those identifiers are sensitive. The dataset and checkpoints are not committed by this project.

CUDA float16 mixed precision is enabled automatically when CUDA is selected. Use `--no-mixed-precision` to disable it. No CUDA accelerator is required for CPU runs.

## Evidence and current limitations

CI tests the recording format loader, rejected-sample filtering, path containment, deterministic session grouping, encoder-only checkpoint transfer, and a tiny end-to-end pretrain/fine-tune/scratch comparison. Those synthetic personal recordings test the pipeline and are not accuracy evidence.

A real OnHW fold-0 pretraining smoke run is recorded in [`reports/onhw_m6a_pretraining_smoke.json`](reports/onhw_m6a_pretraining_smoke.json): 64 training and 64 held-out validation samples, one epoch, seed 2026, CPU, 4,676,540 parameters. It measured CER 100% (318/318 characters) and WER 100% (64/64 words). This verifies that the M6 pretraining command writes a transferable checkpoint from the actual archive; the score is not a benchmark or a recognition claim. The recording host's PyTorch XPU package emitted an unsupported integrated-GPU warning during CPU backward, but the run completed on CPU.

This repository does not include accepted personal handwriting recordings. Therefore M6-A currently makes no claim about whether OnHW pretraining improves recognition on the user's pen. Run the command above on real accepted sessions before drawing that conclusion. The OnHW smoke report, when present, records only public-data pretraining and is not a personal validation benchmark.
