# Report schema migration: 1.x to 2.0

The raw `imupen_fifo_v1` capture CSV has not changed. Report JSON is now explicitly versioned as `report_schema_version: "2.0"` with `quality_policy_version: "1.0"`. Consumers must branch on the schema version and should reject unknown major versions.

## Compatibility changes

- The old top-level `rows`, `missing_host_sequence_numbers`, timing summaries, `segments`, and `limits` remain available.
- `segments` now identify each fit by host session, boot ID, sequence-run instance, and FIFO epoch. Consumers must not assume `(session, epoch)` alone is unique.
- `segments[].model` is `null` when there are too few defensible IRQ anchors. The robust model names its estimator (`huber_irls`), effective rate, nominal drift, residual quantiles, down-weighted anchors, and timing confidence.
- `estimated_sample_host_us` retains its historical column name, but its values are in the MCU IRQ clock (`estimated_sample_time_clock: "mcu_irq_us_since_boot"`), not host monotonic time. `host_elapsed_s` remains the separate host receive/process clock.
- `per_session_counters` and `total_observed_session_end_counters` distinguish observed per-session counters from the aggregate of last-observed values. The aggregate is not an exact count of physical sensor loss.
- `events_summary` distinguishes planned and unplanned events and explicitly marks an absent event log as unavailable.
- `quality`, `quality_verdict`, and stable `quality_reasons` are new. Strict CLI use exits nonzero for failed data quality.
- `reconstructed.csv` preserves original columns and adds `estimated_sample_host_us`. Blank values mean no defensible fit exists for that row's session/boot/run/epoch.

## Migration steps

1. Read and require `report_schema_version` before accessing fields.
2. Continue to read the raw CSV with the existing field list; do not infer timestamps from row number alone.
3. Replace consumers that use the old segment key with the full identity `(session, boot_id, session_instance, epoch)`.
4. Treat null models and blank reconstructed timestamps as unavailable timing, not zero.
5. Use `quality_verdict` and reason IDs instead of interpreting the presence of a report file as a successful capture.
6. Treat session counter aggregates as observed lower bounds with censored tails.

Generated captures, reports, and handwriting data can contain private user recordings. Keep them local unless the project owner explicitly authorizes sharing.
