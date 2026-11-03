# Hardware-free build baseline

Recorded during the software-only M5 pass. This is a compile result, not a board upload or sensor check.

| Setting | Value |
| --- | --- |
| Arduino CLI | 1.4.1 |
| Espressif Arduino core | `esp32:esp32` 3.3.6 |
| Discovered target | `esp32:esp32:XIAO_ESP32S3` |
| Compile options | `USBMode=hwcdc,CDCOnBoot=default` |
| Command | `arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32S3:USBMode=hwcdc,CDCOnBoot=default --warnings all ...` |
| Result | PASS; no upload performed |
| Program storage | 293,379 / 3,342,336 bytes (8%) |
| Global memory | 21,152 / 327,680 bytes (6%) |

The sketch warning inventory includes `IMUpen.ino:44`: incrementing volatile-qualified `irqCount` is deprecated by the current compiler. The cold core build also reported missing-field-initializer warnings in bundled ESP32/TinyUSB headers. Neither warning was hidden with a source or toolchain change. Keep the ISR implementation unchanged until the concurrency semantics are reviewed and rerun under the pinned target.

The repeatable workflow is `.github/workflows/ci.yml`; its build artifacts retain the exact board properties and compiler output. The build target and memory use do not establish electrical safety, correct sensor identity, measured ODR, or handwriting performance.
