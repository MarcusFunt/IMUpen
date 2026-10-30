#include <Arduino.h>
#include <esp_timer.h>
#include "imu.h"
#include "protocol.h"

#ifndef IMU_SDA_PIN
#error "Define IMU_SDA_PIN for your board in platformio.ini"
#endif
#ifndef IMU_SCL_PIN
#error "Define IMU_SCL_PIN for your board in platformio.ini"
#endif
#ifndef IMU_I2C_HZ
#define IMU_I2C_HZ 400000
#endif

using namespace smartpen;
namespace {
Lsm6ds3Fifo imu;
uint32_t sequence = 0;
uint32_t tx_drops = 0;
uint64_t epoch_us = 0;
uint32_t epoch_sequence = 0;
bool have_epoch = false;
uint32_t last_telemetry_ms = 0;
bool imu_ready = false;

void telemetry() {
  Configuration c = {kSampleRateHz, kAccelRangeG, kGyroRangeDps,
                     imu.address(), IMU_SDA_PIN, IMU_SCL_PIN,
                     imu.whoAmI(), 0x0001};
  sendConfig(c);
  sendError(kErrorFifoOverruns, imu.fifoOverruns());
  sendError(kErrorI2cFailures, imu.i2cFailures());
  sendError(kErrorTxDrops, tx_drops);
}
}  // namespace

void setup() {
  Serial.begin(921600);
  imu_ready = imu.begin(IMU_SDA_PIN, IMU_SCL_PIN, IMU_I2C_HZ);
  // No ASCII is ever printed on the binary serial channel.
}

void loop() {
  if (!imu_ready) {
    // Initialization failed: do not pretend samples are valid.
    // Report a binary diagnostic; check board, I2C pins and WHO_AM_I.
    if (millis() - last_telemetry_ms >= 2000) {
      sendError(kErrorSensorInit, 1);
      last_telemetry_ms = millis();
    }
    delay(50);
    return;
  }
  RawImuSample raw;
  uint16_t flags = 0;
  // Drain bounded batches, allowing USB telemetry to run regularly.
  for (unsigned n = 0; n < 24; ++n) {
    if (!imu.next(raw, flags)) break;
    if (!have_epoch || (flags & kFlagDiscontinuity)) {
      epoch_us = static_cast<uint64_t>(esp_timer_get_time());
      epoch_sequence = sequence;
      have_epoch = true;
    }
    // Nominal sensor clock reconstruction; not a per-sample HW timestamp.
    const uint64_t t_us = epoch_us +
        (uint64_t(sequence - epoch_sequence) * 1000000ULL) / kSampleRateHz;
    Sample sample = {sequence, t_us, raw.gx, raw.gy, raw.gz,
                     raw.ax, raw.ay, raw.az, flags};
    if (!sendSample(sample)) ++tx_drops;
    ++sequence;  // Count acquisition even when host cannot keep up.
  }
  const uint32_t now = millis();
  if (now - last_telemetry_ms >= 2000) {
    telemetry();
    last_telemetry_ms = now;
  }
  delay(1);
}
