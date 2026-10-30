# Milestone M5: residual CNN-BiLSTM-CTC

M5 adds a larger residual 1D CNN encoder while retaining the M4 CNN-BiLSTM-CTC baseline. Both architectures use the same six-axis OnHW adapter, character vocabulary, writer-independent fold, train-only normalization, CTC loss, greedy decoder, and CER/WER evaluator. The model is host-side PyTorch code; firmware and the XIAO ESP32-S3 recorder are unchanged.

## Architecture

The default M5 model uses:

```text
IMU sequence [time, 6]
  → 64-channel stride-2 input projection
  → residual 1D CNN blocks: 64 → 64 → 128 → 256 channels
  → total temporal downsampling ×4
  → 3-layer bidirectional LSTM, 256 hidden units per direction
  → dropout 0.2 and character classifier
  → CTC loss / greedy decoding
```

Residual blocks use two 1D convolutions and a projected skip when channel width or stride changes. Per-timestep channel layer normalization keeps padded batch positions from changing valid sample statistics. The default configuration is intended to land in the 4–8 million trainable parameter range; every run records its exact count. Channels, recurrent width, layer count, and dropout are configurable through the Python API.

Training uses the M4 AdamW settings and gradient clipping. CUDA runs enable float16 autocast by default; pass `--no-mixed-precision` to disable it or `--mixed-precision` to request it explicitly. Mixed precision is only supported on CUDA. CTC log probabilities and loss remain float32 for stability.

## Reproducible M4/M5 comparison

The comparison command trains both architectures sequentially with the same official fold, archive, selected train and validation sample indexes, seed, batch size, epoch count, optimizer settings, device, and precision setting:

```bash
python -m smartpen.hwr.cli compare \
  --data datasets/OnHW-Words500_indep.zip \
  --output runs/onhw-fold0-m4-vs-m5 \
  --fold 0 --epochs 30 --batch-size 24 --seed 2026 --device cuda
```

It writes one checkpoint and metrics file per model plus `comparison.json`, which includes fairness checks, parameter counts, and per-epoch CER/WER. Both runs select the same records deterministically. This is a same-fold model comparison, not a five-fold benchmark; publish or compare aggregate claims only after running every official fold.

For a short pipeline check, cap the data and epochs. The output is explicitly marked as a sample experiment:

```bash
python -m smartpen.hwr.cli compare \
  --data datasets/OnHW-Words500_indep.zip \
  --output runs/onhw-fold0-m4-vs-m5-sample \
  --fold 0 --epochs 1 --batch-size 16 --seed 2026 --device cpu \
  --num-threads 2 --max-train-samples 64 --max-validation-samples 64
```

The recorded real-data run is in [`reports/onhw_fold0_m4_vs_m5_sample.json`](reports/onhw_fold0_m4_vs_m5_sample.json). It used the same 64 train and 64 held-out validation samples, seed 2026, batch size 16, CPU, and one epoch for both models:

| Model | Trainable parameters | CER | WER |
| --- | ---: | ---: | ---: |
| M4 CNN-BiLSTM-CTC | 184,924 | 94.03% (299 / 318 characters) | 100% (64 / 64 words) |
| M5 ResNet-BiLSTM-CTC | 4,676,540 | 100% (318 / 318 characters) | 100% (64 / 64 words) |

This smoke comparison does **not** show an accuracy improvement from M5. One epoch over 64 training samples is insufficient to judge the architecture; these figures are measured evidence for pipeline operation only, not a benchmark. The available host had no CUDA device, so CUDA mixed precision and GPU memory use were not measured here. The local PyTorch XPU build emitted an unsupported integrated-GPU warning during CPU backward; both training runs completed on CPU.

Individual runs are also available:

```bash
python -m smartpen.hwr.cli train \
  --data datasets/OnHW-Words500_indep.zip \
  --output runs/onhw-fold0-resnet --model-type resnet_bilstm_ctc \
  --fold 0 --epochs 30 --batch-size 24 --seed 2026 --device cuda

python -m smartpen.hwr.cli evaluate \
  --data datasets/OnHW-Words500_indep.zip \
  --checkpoint runs/onhw-fold0-resnet/model.pt \
  --fold 0 --device cuda
```

The checkpoint stores the architecture name and configuration, so evaluation rebuilds the right model automatically. The training/evaluation commands use the same license-bound OnHW data described in [ONHW_M4.md](ONHW_M4.md). Do not commit the dataset, checkpoints, or derived recordings.

## Verification and scope

Automated coverage checks the default parameter budget, ×4 sequence-length mapping, padding-independent logits, and a train/save/reload/evaluate cycle through the shared pipeline. CI continues to run the Python suite and the existing firmware build.

OnHW uses a different sensor pen and recording setup from the XIAO ESP32-S3 + LSM6DS3 recorder. These experiments validate the host-side architecture and comparison pipeline; they do not establish transfer accuracy on the physical pen. Sample-capped results are implementation evidence only, not recognition benchmarks.
