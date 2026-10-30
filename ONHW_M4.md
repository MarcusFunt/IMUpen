# Milestone M4: six-axis handwriting recognition

M4 adds an optional PyTorch CNN-BiLSTM-CTC recognizer for variable-length IMU recordings. It uses the same six semantic channels as the XIAO ESP32-S3 recorder: front accelerometer x/y/z followed by gyroscope x/y/z. Firmware, the serial protocol, and recording formats remain unchanged.

## Dataset and permitted use

Use the right-handed **OnHW-Words500 writer-independent** archive from the [Fraunhofer IIS dataset page](https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html). The archive ships five writer-independent folds. Each fold provides its own training and validation partitions; this project uses those partitions as published and checks that writer IDs do not overlap.

The OnHW dataset is separate from this repository's MIT-licensed code. The dataset author's [license statement](https://huggingface.co/datasets/FelixOtt/OnHW_datasets) identifies OnHW data as **CC BY-NC 4.0**: use, sharing, and adaptation require attribution and are limited to non-commercial purposes. Include the [license](https://creativecommons.org/licenses/by-nc/4.0/) and the dataset citation in derivative research. Commercial use is not permitted by that license. Do not commit the archive, extracted recordings, or trained checkpoints to this repository.

Dataset citation:

> Felix Ott, Mohamad Wehbi, Tim Hamann, Jens Barth, Björn Eskofier, and Christopher Mutschler. “The OnHW Dataset: Online Handwriting Recognition from IMU-Enhanced Ballpoint Pens with Machine Learning.” *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies*, 4(3), article 92, 2020. [doi:10.1145/3411842](https://doi.org/10.1145/3411842).

Sequence files contain 13 channels in this order: front accelerometer (3), rear accelerometer (3), gyroscope (3), magnetometer (3), and force (1). M4 selects indexes 0–2 and 6–8. The released integer labels use the 59-character set `A-Z`, `a-z`, `ÄÖÜäöüß`; the value 59 is the archive's trailing padding and the model's CTC blank. The downloaded archive is loaded with a restricted unpickler that only permits NumPy reconstruction classes.

## Install and run

Install the optional ML extra:

```bash
python -m pip install -e ".[ml]"
```

Train on one official fold. The CLI accepts either the original ZIP archive or an extracted archive folder:

```bash
python -m smartpen.hwr.cli train \
  --data datasets/OnHW-Words500_indep.zip \
  --output runs/onhw-fold0 \
  --fold 0 --epochs 10 --batch-size 32 --seed 1337 --device cpu
```

Training writes `model.pt` and `metrics.json`. Evaluation can be repeated separately with the saved checkpoint:

```bash
python -m smartpen.hwr.cli evaluate \
  --data datasets/OnHW-Words500_indep.zip \
  --checkpoint runs/onhw-fold0/model.pt \
  --fold 0 --seed 1337 --device cpu
```

The checkpoint stores the model configuration, training-only per-axis normalization statistics, vocabulary, seed, and fold. `torch.load` uses `weights_only=True`; only load checkpoints you trust.

The model uses two stride-2 1D convolution blocks, a two-layer bidirectional LSTM, a character projection, and CTC loss. The logits and sequence lengths follow the [PyTorch CTCLoss contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html). Training sets Python, NumPy, and CPU PyTorch seeds, enables deterministic algorithms, records package versions, and uses a length-bucketed sampler. Deterministic algorithms reduce variation on the same software and hardware; PyTorch does not promise identical results across different versions or devices ([reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)).

## Metrics and split policy

The evaluator reports corpus micro-averaged **character error rate (CER)** and **word error rate (WER)**, with edit counts and denominators. On OnHW-Words500 each sample is a single word, so WER is the proportion of word samples decoded incorrectly. The official writer-independent validation writers remain entirely outside training. The fold's validation partition is used for the reported held-out evaluation; this is not an additional test set and must not be compared as a full benchmark without running the full protocol across all five folds.

This validates the host-side model pipeline on OnHW only. It does not measure transfer accuracy on the XIAO ESP32-S3 recorder: OnHW was captured with a different sensor pen and sampling setup, and no XIAO recordings are part of this run. The trained model is not firmware inference code.

Empty recordings and sequences too short for a valid CTC alignment are excluded and counted in `metrics.json`. `--max-train-samples` and `--max-validation-samples` enable a smaller deterministic run; the report records selected source indexes, writer counts, dropped samples, seed, model configuration, and archive SHA-256. Sample-capped results are explicitly marked as samples and are not benchmark results.

## Recorded sample run

The repository's `reports/onhw_fold0_sample.json` records a real one-epoch CPU training and held-out evaluation run using 64 training and 64 validation samples from official fold 0. It reports the measured CER/WER and all sample indexes and run settings. This short run verifies that the real archive loads, six-axis adaptation works, the model updates, checkpoint save/load works, and held-out evaluation completes; its score is not a performance claim. On the recording host, PyTorch `2.13.0+xpu` printed an unsupported integrated-GPU warning even though the requested and reported device was CPU; training and the separate checkpoint evaluation both completed.

M5 preserves this baseline and adds a residual CNN-BiLSTM-CTC alternative. Use `python -m smartpen.hwr.cli compare` to train both models with identical fold, seed, sample indexes, and optimization settings; see [M5_RESNET_BILSTM.md](M5_RESNET_BILSTM.md).

M6-A uses an OnHW-trained M5 encoder to initialize fine-tuning on accepted personal recorder recordings, then compares that run with personal training from scratch on the same session-held-out split. See [M6A_ONHW_PRETRAINING.md](M6A_ONHW_PRETRAINING.md) for the workflow and data requirements.

M6-B compares temporal stretching, Gaussian sensor noise, per-axis bias/gain, physically consistent 3D rotations, and controlled sequence concatenation in isolated runs before an optional combined run. See [M6B_IMU_AUGMENTATION.md](M6B_IMU_AUGMENTATION.md) for the reproducible training-only ablation workflow.

| Fold | Seed | Train / validation samples | Epochs | CER | WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0, writer-independent | 2026 | 64 / 64 | 1 | 94.03% (299 / 318 characters) | 100% (64 / 64 words) |

To reproduce that run:

```bash
python -m smartpen.hwr.cli train \
  --data datasets/OnHW-Words500_indep.zip \
  --output runs/onhw-fold0-sample \
  --fold 0 --epochs 1 --batch-size 16 --seed 2026 --device cpu \
  --num-threads 2 --max-train-samples 64 --max-validation-samples 64
```

The archive is intentionally not downloaded by CI. CI runs synthetic adapter/model/training tests and the existing firmware build; the dataset user must download separately from Fraunhofer under the license terms above.
