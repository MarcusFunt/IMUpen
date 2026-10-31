# OnHW-Chars data quality scan

Scanned the complete supplied `onhw-chars_2021-06-30.zip` archive on 2026-10-31. The archive digest is `e73dfd73873def95bcd626f6188975581ec18f50add682bdeae74b706196cc75`. The machine-readable fold-by-fold report is [onhw_chars_data_quality.json](onhw_chars_data_quality.json).

## Coverage and integrity

- All 30 official partitions were scanned: `both`, `lower`, and `upper`; writer-dependent and writer-independent; five folds each.
- The archive contains 466,404 training and 159,096 test sample occurrences summed across published folds. These are fold occurrences, not unique recordings.
- The loader retained 466,357 training and 159,083 test rows after dropping 47 empty training traces and 13 empty test traces.
- No exact duplicate traces were found within a split or across train/test in any scanned official fold.
- Every fold loaded with finite `[time, 13]` sensor arrays and valid labels for its alphabet; malformed shapes and non-finite values fail the scan.
- Sequence lengths range up to 3,475 time steps. Median length is 39–40 for `lower`, 43 for `both`, and 48 for `upper`.

## Interpretation limits

OnHW-Chars is a single-character classification dataset, separate from Words500. Its 13 channels are raw uncalibrated values; the archive does not supply a sample rate or writer IDs. This scan therefore does not verify writer overlap or support writer-clustered uncertainty estimates. Character accuracy, balanced accuracy, macro-F1, per-class metrics, and confusion matrices must be reported separately from Words500 word error rate.

Sources: [Fraunhofer OnHW dataset page](https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html), [Ott et al. (2020), DOI 10.1145/3411842](https://doi.org/10.1145/3411842).
