# OnHW-Chars Partial Results (Interrupted)

Generated: 2026-10-31T23:45:29.007525+01:00

## Status

This is an incomplete record of the full-data OnHW-Chars sweep: **2/70 runs completed**, 0 failed. The requested sweep was stopped before completion to move it to a stronger computer.

Dataset archive: `onhw-chars_2021-06-30.zip`  
Archive SHA-256: `e73dfd73873def95bcd626f6188975581ec18f50add682bdeae74b706196cc75`  
Sweep ID: `0ac768428579f5fd878f`  
Training and development sample caps: none.

## Completed runs

| Protocol / fold | Sensor profile | Official test accuracy | Balanced accuracy | Macro F1 | Correct / n | Selected epoch | Train seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| dep / 0 | `front_acc_gyro6` | 69.1155% | 69.1090% | 68.4256% | 5673 / 8208 | 10 | 3089.7 |
| dep / 0 | `rear_acc_shared_gyro6` | 68.0312% | 67.9065% | 67.2164% | 5584 / 8208 | 10 | 22102.7 |

Both rows are for `both / dep / fold 0`; each uses the same 8,208-sample official test partition. The `front_acc_gyro6` row reused a compatible completed artifact. The `rear_acc_shared_gyro6` row completed during this sweep.

## Limits

These two rows are preliminary single-character classification results. They do not complete the seven-profile, two-protocol, five-fold matrix. No word-recognition or WER claim is made. The remaining runs must be executed and analyzed before drawing comparative conclusions.
