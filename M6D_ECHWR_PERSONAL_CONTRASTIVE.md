# Milestone M6-D: ECHWR-style personal contrastive training

M6-D implements the roadmap's A6 experiment: train the existing ResNet-BiLSTM-CTC recognizer with temporary sensor/text contrastive objectives, then compare those objectives on an identical held-out-session split. It runs three variants: CTC only, CTC plus in-batch contrastive alignment (BC), and CTC plus BC and error-based contrastive alignment (EC).

The training-only branch uses attention pooling over the BiLSTM sensor features and a small character-level Transformer for transcripts. BC applies symmetric sensor/text InfoNCE and filters duplicate transcript labels within a batch to avoid treating identical text as a negative. EC contrasts each sensor embedding against its true transcript and synthetic one-character deletion, insertion, and substitution variants. The full objective is `CTC + λBC·BC + λEC·EC`. The text branch is excluded from the recognizer's inference state. This is an ECHWR-style adaptation for the project's personal ResNet-BiLSTM-CTC pipeline, not a reproduction of the published architecture or benchmark. The paper describes the temporary auxiliary text branch and dual contrastive objective; the [official implementation](https://github.com/jindongli24/ECHWR) is MIT-licensed. See the [paper](https://arxiv.org/abs/2602.07049).

The recorder, USB protocol, and XIAO ESP32-S3 firmware are unchanged.

## Run the comparison

Collect accepted recordings across multiple sessions with the existing prompted recorder, then run:

```bash
python -m pip install -e ".[ml,test]"

python -m smartpen.hwr.cli personalize-echwr \
  --recordings recordings \
  --pretrained-checkpoint runs/onhw-m6a-pretrain/model.pt \
  --output runs/m6d-echwr \
  --validation-fraction 0.2 \
  --epochs 30 --batch-size 16 --seed 2026 --device cuda
```

`--pretrained-checkpoint` is optional. When supplied, the same compatible M6-A OnHW encoder initializes all three variants; the personal CTC classifier is freshly initialized for each. OnHW-Words500 is licensed separately under CC BY-NC 4.0, so that checkpoint and its adaptations inherit the dataset's non-commercial restriction and attribution requirement. The experiment can also compare all three objectives from scratch.

Use `--split-manifest` to reuse a saved `session_split.json`. `--max-train-samples` and `--max-validation-samples` select deterministic subsets for pipeline smoke runs. The optional `--augmentation` setting is shared across all variants; `combined` applies the M6-B transforms and generates same-session concatenations from fitting records only. Use the default `none` to isolate the contrastive loss changes.

All variants use the same labeled records, writer/session split, train-only normalization, initialization, seed, optimizer, augmentation, and training settings. The auxiliary text objectives and hard negatives use training transcripts only. Held-out sessions are used only for CTC validation and CER/WER reporting.

The output contains `session_split.json`, `comparison.json`, and a folder per variant with its checkpoint and `metrics.json`. Checkpoints keep the inference `model_state` separate from optional `training_auxiliary_state`, allowing the standard recognizer to load without the auxiliary branch. Reports include per-epoch CTC/contrastive losses, CER/WER, source sample IDs, normalization, pretraining hash when applicable, augmentation settings, and fairness checks. Session and sample identifiers are personal data; keep the run directory private.

## Evidence and interpretation

This repository does not include accepted personal recordings. Synthetic tests exercise negative generation, the contrastive losses, variable-length sensor/text encoding, the session split, and the three-variant training/checkpoint path. They are pipeline evidence only and do not establish handwriting accuracy or that either contrastive loss improves CER/WER. Use a separate later session for final evaluation after model selection; do not expose final-test transcripts or sensor traces to training, including contrastive training.
