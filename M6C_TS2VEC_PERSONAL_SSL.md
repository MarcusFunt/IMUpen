# Milestone M6-C: personal TS2Vec-style self-supervision

M6-C tests whether unlabeled personal IMU recordings help the personal ResNet-BiLSTM-CTC recognizer. It trains a TS2Vec-style encoder from unlabeled sensor data, then compares supervised fine-tuning from that encoder against the same recognizer trained from scratch. Both supervised runs use identical labeled recordings, writer/session split, seed, model, normalization, optimizer, and training settings.

The self-supervised objective follows TS2Vec's core design: two context views share an aligned temporal segment, and timestamp representations are contrasted at multiple scales using temporal and instance-wise losses. The implementation uses the existing M5 encoder, random timestamp masking on normalized sensor views, and a projection head that is discarded before CTC fine-tuning. This is a TS2Vec-style adaptation for variable-length handwriting IMU; it is not a reproduction of the original benchmark system. See the [TS2Vec paper](https://arxiv.org/abs/2106.10466) and [official implementation](https://github.com/zhihanyue/ts2vec).

The optional M6-C runner and recorder remain host-side. No firmware, USB, or acquisition changes are required.

## Prepare data

Use the prompted collector for the labeled recording root. For unlabeled pretraining, use accepted recordings in the recorder's `metadata/` and `processed/` layout. The SSL loader reads only accepted status, writer/session IDs, and six-axis arrays; it ignores any prompt field. It excludes every unlabeled record whose writer/session group is held out for labeled validation. If both CLI roots point at the same directory, labeled training samples may also be seen without their labels during SSL; validation-session samples remain excluded.

Do not include a final test session or any recordings reserved for a future final evaluation in the unlabeled root. Self-supervised learning still constitutes training exposure to those sensor traces even when their transcript is hidden.

## Run the comparison

```bash
python -m pip install -e ".[ml,test]"

python -m smartpen.hwr.cli personalize-ssl \
  --recordings data/personal-labeled \
  --unlabeled-recordings data/personal-unlabeled \
  --output runs/m6c-ts2vec \
  --validation-fraction 0.2 \
  --ssl-epochs 30 --epochs 20 --batch-size 16 \
  --seed 2026 --device cuda
```

Use `--split-manifest` to reuse a saved `session_split.json`, and `--max-unlabeled-samples` for a deterministic pipeline smoke run. The output contains the SSL checkpoint and `metrics.json`, both fine-tuned CTC checkpoints, session manifest, per-run CER/WER, and a root `comparison.json` with leakage and fairness checks. CTC fine-tuning fits normalization on labeled training sessions and shares those statistics across the scratch and TS2Vec-initialized runs.

## Data and interpretation

No personal recordings are included in this repository. Synthetic tests exercise loading, unlabeled prompt isolation, overlapping context alignment, the hierarchical loss, validation-session exclusion, encoder transfer, and both downstream training paths; they do not establish personal handwriting accuracy. Run this experiment with real collected sessions before interpreting CER/WER. Keep a later session and its transcripts untouched for a final evaluation after model selection.

The command does not download or use OnHW. Reports contain personal sample and session identifiers, so keep the run directory private. Personal recordings and checkpoints are not committed.
