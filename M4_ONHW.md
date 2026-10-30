# Milestone M4: CNN-BiLSTM-CTC baseline, tested with OnHW

This is the **baseline** recognizer (not M5 ResNet). It accepts a variable-length
sequence of six IMU measurements and predicts a character sequence using
two temporal strided convolutions, a bidirectional LSTM and CTC loss.

## OnHW data source and restrictions

OnHW-Words500 by Ott et al. / Fraunhofer IIS is mirrored at
https://huggingface.co/datasets/FelixOtt/OnHW_datasets .

Its data license is **CC BY-NC 4.0**; code in IMUpen remains MIT. Do not
redistribute downloaded OnHW recordings or use the data commercially without
appropriate authorization. The downloaded archives contain **pickle**
objects; only run --trust-pickle on authentic, verified source archives.

The adapter uses original input columns 3,4,5 (back accelerometer) and
6,7,8 (gyro), discarding the other seven OnHW channels. This six-axis
representation is *not* automatically calibrated for the LSM6DS3 breakout.

## Installing and obtaining the dataset

Install the ML extra (torch installed separately on some hardware platforms):

```bash
python -m pip install -e ".[ml,test]"
python scripts/download_onhw_words500.py --dest datasets/onhw
```

The downloader looks up the original right-handed writer-independent ZIP
on the Hugging Face mirror, downloads and extracts it. It does not unpickle.

## Real-data smoke test (not a benchmark)

```bash
python -m smartpen.ml.training \
  --onhw datasets/onhw/extracted --trust-pickle --fold 0 \
  --max-train 400 --max-official-val 100 \
  --epochs 1 --batch-size 16 --channels 8 --hidden 16 --layers 1 \
  --out runs/ci-onhw
```

The subset sample selection is deterministic; a one-epoch model trained on
400 random samples should be expected to perform poorly. It is a wiring
check of real data loading -> training -> evaluation, **not** an official
OnHW research benchmark. CI records results in a downloadable artifact.

## Full single-fold experiment on your RTX 3060

```bash
python -m smartpen.ml.training \
  --onhw datasets/onhw/extracted --trust-pickle --fold 0 \
  --epochs 30 --batch-size 24 --channels 64 --hidden 128 --layers 2 \
  --out runs/m4-full
```

The official training partition is *internally* split by writer to choose
the best epoch based on character error rate (CER). The official validation
partition is evaluated only after best-epoch selection. The output includes
aggregate CER, WER, exact-string accuracy, writer IDs, example predictions,
counts and the flag marking debug subsets. Train-only mean and std are
stored in the checkpoint.

Outputs: runs/m4-full/best.pt, runs/m4-full/results.json.

```bash
python -m smartpen.ml.predict \
  --checkpoint runs/m4-full/best.pt \
  --recording datasets/processed/day01/YOUR_SAMPLE.npz
```

**Warning:** raw personal recordings are at nominal 208 Hz and contain
uncalibrated signed int16 counts, whereas this OnHW model was trained on
a different pen's 100 Hz signals. Predictions on raw personal recordings
are not yet meaningful. Sensor calibration, unit conversion, resampling,
orientation alignment, and personal fine-tuning follow in M5/M6.

## Evaluation integrity

- OnHW-Words500 is a closed vocabulary of German words; it cannot validate
  free-form unseen-sentence transcription.
- The original writer-independent OnHW fold may be checked by confirming
  disjoint writer IDs in the emitted JSON.
- Best-epoch selection uses ONLY a writer-grouped split from official training.
- One fold/seed is not equivalent to a published 5-fold averaged result.
- Time-series padded inputs are masked after each convolution and packed
  before bidirectional recurrence.
- A synthetic-format fixture test is different from a real-data test.
- Official test data must not be reused repeatedly to select hyperparameters.

## Tests

```bash
python -m pytest -q tests/test_ml_m4.py
```
