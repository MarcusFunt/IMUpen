# Hardware-free model boundaries

## Native peripheral mock

The native harness compiles the unchanged Arduino sketch against a deterministic test API. The peripheral model has a 256-address register space; known configuration registers are writable, identity/status/FIFO output registers are read-only, and unimplemented addresses read as zero. It checks CS transaction boundaries, mode 3, MSB-first transfers, auto-increment when `CTRL3_C.IF_INC` is set, FIFO data output behavior, configurable FIFO capacity, watermark status, and continuous-mode overwrite indication.

Default 4096-word capacity represents 8 KiB for the LSM6DS3 profile; the 2048-word setting represents the 4 KiB LSM6DS3TR-C capacity. Continuous FIFO overwrite behavior is based on ST's [LSM6DS3 FIFO application note](https://community.st.com/ysqtg83639/attachments/ysqtg83639/mems-sensors-forum/21052/2/AN4650.PDF) and the [TR-C application note](https://www.st.com/content/ccc/resource/technical/document/application_note/group0/90/41/09/82/24/d9/4f/29/DM00472670/files/DM00472670.pdf/jcr%3Acontent/translations/en.DM00472670.pdf). That model behavior remains software-only evidence.

Fault controls cover WHO_AM_I, readback mismatch, reset-stuck, short reads, FIFO overflow, and pattern corruption. The mock does not model analog effects, electrical timing, real interrupt pin characteristics, sensor ODR startup, USB CDC buffering, ESP32 CPU scheduling, actual FreeRTOS concurrency, or behavior of an unconfirmed sensor variant. The interrupt/task scheduler model is a separate deterministic software model and is not claimed to be cycle-accurate.

The seeded scheduler model records its seed and parameters. `tests/hardware_free/test_scheduler_model.py` runs 1,000 interleaving/fault seeds and writes `build/hardware-free/scheduler-seed-summary.json`; a failure writes a seed-specific `scheduler-repro-seed-<n>.json` trace. Run the single failing seed directly with `scheduler_model.make_model(seed).run()`.
