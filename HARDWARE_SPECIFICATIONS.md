# IMUpen V0 — hardware specifications and design limits

**Hardware target:** Seeed Studio XIAO ESP32-S3, single ST LSM6DS3 on a GY-LSM6DS3-style breakout, 4-wire SPI plus FIFO watermark INT1. **Status:** hardware not tested; voltages for the *exact* breakout PCB must be verified before energizing it. This document is a hardware reference, not evidence of working physical samples.

## 1. Actual firmware pin map

| XIAO pad | ESP32-S3 GPIO | LSM6DS3 breakout pin | Role |
|---|---:|---|---|
| 3V3 | regulated 3.3 V | `VIN` **only if board regulator accepts this**; alternatively `3V3` if documented as safe power input | power |
| GND | ground | GND | common reference |
| D8 | 7 | SCL | SPI SCK |
| D10 | 9 | SDA | SPI MOSI / SDI |
| D9 | 8 | SA0 | SPI MISO / SDO |
| D3 | 4 | CS | chip-select, active low |
| D0 | 1 | INT1 | watermark interrupt |

Other exposed board pads: **INT2** (second configurable interrupt); **OCS, SCX, SDX** (auxiliary/sensor-hub interface, not required here); and `3V3` (board-specific input/output; do not assume a safe back-feed path). Breakout-board variants may contain level shifters or pull-ups. SA0 is an I²C address strap **only in I²C mode**; in 4-wire SPI it must be able to drive MISO. An imposed low/high short or solder jumper can break SPI. CS must be freely driven and default high while idle. Disconnect any I²C address strap before SPI as needed.

GY-LSM6DS3 vendor reference: https://easyelecmodule.com/product/gy-lsm6ds3-6-axis-imu/

**Do not apply 5 V to the bare LSM6DS3 IC or XIAO GPIO.** A seller's claim of 5 V `VIN` compatibility only applies through that breakout's validated onboard regulator / shifters. If powering a 3.3 V-only circuit from the XIAO 3V3 pin, ensure current draw and regulator margins are adequate.

Seeed official XIAO reference: https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/

## 2. Sensor comparison — do not silently mix these variants

| Property | Original LSM6DS3 | LSM6DS3TR-C |
|---|---|---|
| Device identity (`WHO_AM_I`) | `0x69` | `0x6A` |
| Axes | 3 accelerometer + 3 gyro | same |
| FIFO nominal capacity | 8 KiB (4096 16-bit words) | 4 KiB (2048 16-bit words) |
| Acceleration full scale | ±2, ±4, ±8, ±16 g | same |
| Gyro full scale | ±125, ±250, ±500, ±1000, ±2000 °/s | same |
| Accel sensitivity at ±2 g | 0.061 mg/LSB | 0.061 mg/LSB |
| Gyro sensitivity at ±500 °/s | 17.50 mdps/LSB | 17.50 mdps/LSB |
| Acceleration noise density, ±2 g, HP | typ. 90 µg/√Hz | typ. 90 µg/√Hz |
| Gyro noise density, HP | typ. 7 mdps/√Hz | typ. 5 mdps/√Hz |
| SPI maximum | 10 MHz | 10 MHz |
| Power supply at IC | 1.71–3.6 V analog VDD | 1.71–3.6 V analog VDD |
| Operating temperature at IC | −40 to +85 °C | −40 to +85 °C |
| Package | 2.5 × 3 × 0.83 mm | 2.5 × 3 × 0.83 mm |
| Approx. combo sensor current | 0.9 mA, 208 Hz normal; 1.25 mA HP | 0.90 mA combo high-performance (datasheet) |

The exact variant must be identified before enabling the `CONFIRMED_LSM6DS3TR_C` firmware switch. Similar register addresses do not establish equivalent FIFO data and interrupt semantics. Sensor-level current excludes breakout regulator quiescent current, power LEDs, ESP32, radio, USB, etc.

Sources:
- Original ST datasheet hosted by Arduino: https://docs.arduino.cc/resources/datasheets/LSM6DS3-datasheet.pdf
- ST TR-C datasheet: https://www.st.com/resource/en/datasheet/lsm6ds3tr-c.pdf

## 3. Sample rate, FIFO and bandwidth budgets

- Current `CTRL1_XL=0x50`: accelerometer ODR 208 Hz, full-scale ±2 g.
- Current `CTRL2_G=0x54`: gyro ODR 208 Hz, full-scale ±500 °/s.
- Current `CTRL3_C=0x44`: BDU=1, automatic register increment=1, 4-wire SPI.
- Current `FIFO_CTRL1=0x30`, `FIFO_CTRL2=0x00`: watermark = 48 16-bit FIFO words = eight 6-axis frames.
- Current `FIFO_CTRL3=0x09`: accelerometer and gyro FIFO decimation = 1.
- Current `FIFO_CTRL4=0x00`: no auxiliary FIFO streams.
- Current `FIFO_CTRL5=0x2E`: continuous FIFO at 208 Hz.
- Current `INT1_CTRL=0x08`: route FIFO watermark to INT1.
- Current SPI: mode 3, MSB-first, 4 MHz. ST rates SPI to 10 MHz; this is not a guarantee about *breakout-level* margins.

Calculations at exactly nominal 208 Hz:

| Item | Value |
|---|---:|
| Period of an individual 6-axis frame | 4.8077 ms |
| Frames per watermark | 8 |
| Interval between normal watermark crossings | 38.4615 ms |
| Raw frame size (3 gyro + 3 accel 16-bit axes) | 12 bytes |
| Raw frame throughput | 2496 bytes/s = 2.496 kB/s |
| FIFO watermark burst payload | 96 bytes |
| Minimum SPI wire time for 96 bytes + address at 4 MHz | about 194 µs, excluding CS/CPU overhead and status read |
| Original 8 KiB FIFO, theoretical 6-axis hold-up | ~3.28 seconds at 208 Hz |
| TR-C 4 KiB FIFO, theoretical 6-axis hold-up | ~1.64 seconds at 208 Hz |
| MCU output queue `256 × sizeof(Sample)` | implementation-dependent memory usage (native host struct = platform-dependent) |
| MCU queue time represented by 256 samples | ~1.23 seconds at 208 Hz (separate from FIFO) |

**Crucial:** FIFO capacity is not the same as lossless serial-output buffering. In the present sketch an output queue fill can drop samples even while the hardware FIFO is working perfectly. The maximum *lossless* serial interruption depends on scheduling, USB driver behavior and whether the output task blocks; it must not be inferred by adding theoretical buffer durations without a model.

## 4. XIAO ESP32-S3 hardware

- MCU: ESP32-S3R8, dual-core Xtensa LX7 up to 240 MHz.
- Memories on XIAO ESP32-S3: 8 MB PSRAM, 8 MB flash.
- Radio: 2.4 GHz Wi-Fi, BLE 5.0.
- Native USB-C; in USB CDC mode, the user-selected serial baud rate is not necessarily a physical UART line-rate limit.
- Board dimensions: 21 × 17.8 mm.
- Operates with 3.3 V GPIO logic; do not expose GPIO to 5 V.
- On the **Sense** model, SD card and external peripherals can share D8–D10; audit pin-mux conflicts before using an SD card or changing board models.
- Radio-on current can be much larger than the IMU's current. Seeed documents example active Wi-Fi current around 100 mA under its stated conditions; do not treat this as a guaranteed peak or a battery-life prediction.

Official source: https://wiki.seeedstudio.com/xiao_esp32s3_getting_started/

## 5. Measurement accuracy and what the numbers mean

The smallest *digital code step* is not equivalent to achievable accuracy. The accelerometer's nominal ±2 g scale corresponds to 0.061 mg per count; its noise, temperature-dependent offset, nonlinearity, solder stress and mechanical alignment can cause much larger uncertainties. A 7 mdps/√Hz gyro noise density is a spectral specification; it does not mean 0.007 °/s RMS in a 208 Hz capture without accounting for effective noise bandwidth and filtering.

At room temperature, calibration should check six static orientations, gyro zero-rate bias, orientation-dependent offsets, noise with the pen still, repeated motions, and warm-up drift. Temperature-compensation models should not be trained on invented thermal responses alone.

Gyroscopes measure angular velocity, not heading. Accelerometers measure specific force (including gravity in the measured frame), not direct pen-tip displacement. Without external reference or contact information, integrating raw acceleration to infer pen position drifts quickly. Sensor fusion can improve orientation estimates but cannot eliminate fundamental observability limits through software alone.

## 6. Unresolved board-specific items

These **cannot be concluded from IC datasheets or a generic GY-LSM6DS3 listing**: actual regulator part number/dropout and whether VIN can be fed from 3.3 V; exact SA0/CS pull resistors; solder jumpers; voltage of interrupt high; IC variant authenticity; whether the chip temperature sensor is exposed correctly; oscillator tolerance on the actual sample; board mounting orientation and axis map; cable capacitance/EMI; and whether any 3V3 pin is a regulator output or a permissible input. Confirm against the exact PCB and its schematic, or measure it when hardware is available.
