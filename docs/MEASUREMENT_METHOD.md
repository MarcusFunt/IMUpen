# Measurement and timing method

## Raw capture contract

The firmware's `imupen_fifo_v1` CSV fields and order remain unchanged. Each row is one completed six-axis frame in `Gx,Gy,Gz,Ax,Ay,Az` order. `seq` is a host-visible frame sequence; it does not count samples overwritten in the sensor FIFO. `irq_us` is the MCU time recorded by the watermark ISR, while `read_start_us` and `read_end_us` bracket the status/FIFO SPI reads. Host receive time is recorded separately as `host_elapsed_s`.

## FIFO word anchor

The LSM6DS3 FIFO pattern register reports the index of the next 16-bit output word in the repeating six-word data set. This agrees with ST's FIFO pattern description in [AN4650](https://community.st.com/ysqtg83639/attachments/ysqtg83639/mems-sensors-forum/21052/2/AN4650.PDF) and the [LSM6DS3TR-C datasheet](https://www.st.com/resource/en/datasheet/lsm6ds3tr-c.pdf). For a row with FIFO `pattern = p`, the last word of its frame is `word_end`, and the FIFO watermark is 48 words, the coordinate used to relate the independent watermark IRQ to frame sequence is:

```text
anchor_sample = seq + (47 - word_end) / 6
```

The fraction preserves offsets for patterns 1–5. An IRQ is used as an anchor only when `irq_us >= 0`; continuation drains with `irq_us = -1` are not independent clock observations. Anchors are grouped by host session, boot ID, sequence run, and FIFO epoch. A robust Huber fit maps this frame coordinate to MCU IRQ time. It estimates effective ODR and describes fit residuals; it does not create a physical sensor timestamp or prove gyro/accelerometer simultaneity.

The formula relies on the firmware's 48-word watermark and valid six-word channel order. Unknown sensor variants, malformed patterns, FIFO resets, or missing valid anchors must remain explicit quality issues. Hardware timing, part-specific FIFO semantics, channel delay, and electrical behavior still require M6 measurements.

## Quality policy

Reports carry schema and policy versions, validity counts, per-session counter observations, missing field counts, per-axis raw ranges and quantiles, clipping counts, repeated adjacent frame count, event counts, calibration status, and a machine-readable verdict. `CLEAN`, `DEGRADED`, `DEGRADED_EXPECTED`, and `FAILED` describe the recording against the selected policy; `DEGRADED_EXPECTED` requires explicit fault-injection intent and explicit acceptance of expected faults. It does not make a faulted recording appropriate for training.

Counters are reported per observed device session and summed only from each session's last observed value. A final unreported device counter increment may be censored by capture stop, so these sums are not exact physical-loss totals. Sequence gaps are kept separately and are not added to device counters.
