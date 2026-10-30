# Seeed XIAO ESP32-S3 + standalone LSM6DS3 recorder (experimental)

## Wiring (3.3 V logic only)

- LSM6DS3 VCC -> 3V3 (check breakout voltage requirements)
- GND -> GND
- SDA -> XIAO pad **D4 (GPIO5, SDA)**
- SCL -> XIAO pad **D5 (GPIO6, SCL)**
- If your breakout exposes CS, tie it HIGH for I2C as its documentation specifies.
- SA0/SDO selects I2C address 0x6A or 0x6B; firmware scans both.

**Board configuration:** firmware/platformio.ini targets
`seeed_xiao_esp32s3` with GPIO5/GPIO6 matching the exposed D4/D5 I2C pads.
The native USB CDC port is used for the binary stream.

Build and flash with PlatformIO:
  cd firmware
  pio run
  pio run -t upload

Use the native USB CDC serial port at 921600 baud. Do not use a text serial
monitor to interpret the binary data. Run the Python collector instead.

The firmware uses LSM6DS3 FIFO in continuous mode with gyro+accelerometer
at 208 Hz, 400-kHz I2C, +/-4 g, +/-500 dps. A FIFO read cycle is six 16-bit
words (Gx,Gy,Gz,Ax,Ay,Az). FIFO overrun, I2C recovery and USB TX drops are
reported. The 64-bit sample timestamps are reconstructed from the nominal
208-Hz sample index and reset after a detected discontinuity; they are **not**
hardware-synchronized timestamps. If a gap is detected, do not interpolate
across it without marking the segment.

The WHO_AM_I check deliberately requires 0x69 (LSM6DS3). Other variants
(e.g. LSM6DS3TR-C) must be validated separately.

## Protocol v1

All integers are little-endian.

| Field | Bytes |
| --- | --- |
| Sync | 2: 5A A5 |
| Version | 1: 01 |
| Type | 1: 01 sample, 02 config, 03 error |
| Payload length | 2 |
| Payload | length bytes |
| CRC16-CCITT-FALSE | 2 |

CRC is initialized to 0xFFFF and covers version/type/length/payload only.

Sample payload (26 bytes): seq u32, timestamp_us u64, ax/ay/az/gx/gy/gz
six signed i16, flags u16.

Config payload (12 bytes): sample_rate_hz u16, accel_range_g u16,
gyro_range_dps u16, i2c_address u8, sda_pin u8, scl_pin u8,
who_am_i u8, flags u16. Config flag bit 0 means nominal ODR timestamps.

Error payload (6 bytes): error_code u16, cumulative_count u32.
Error codes: 1 FIFO overrun, 2 I2C failure, 3 USB TX drops, 4 sensor initialization failed.

Sample flags: bit 0 discontinuity, bit 1 FIFO overrun, bit 2 I2C failure.
Sequence numbers are u32 and wrap; a gap in the sequence indicates samples
captured but not transmitted. A FIFO overflow may lose additional samples
that cannot be counted exactly, so treat it as a segment boundary.

This firmware has not yet been hardware-validated on your XIAO ESP32-S3 and
standalone breakout. Verify FIFO word order, sampling, clipping, and USB behavior on
your actual hardware before collecting training data.
