#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <stdint.h>
#include "protocol.h"

namespace smartpen {
struct RawImuSample {
  int16_t gx, gy, gz;
  int16_t ax, ay, az;
};

class Lsm6ds3Fifo {
 public:
  bool begin(int sda_pin, int scl_pin, uint32_t i2c_hz);
  // Returns true only if a complete six-word gyro+accel sample was read.
  // Non-ready and recoverable I2C failures both return false.
  bool next(RawImuSample& sample, uint16_t& flags);
  uint8_t address() const { return address_; }
  uint8_t whoAmI() const { return who_am_i_; }
  uint32_t fifoOverruns() const { return fifo_overruns_; }
  uint32_t i2cFailures() const { return i2c_failures_; }

 private:
  bool readBytes(uint8_t reg, uint8_t* out, size_t count);
  bool writeReg(uint8_t reg, uint8_t value);
  bool readWord(int16_t& value);
  bool flushFifo();
  void recoverI2c();

  uint8_t address_ = 0;
  uint8_t who_am_i_ = 0;
  uint32_t fifo_overruns_ = 0;
  uint32_t i2c_failures_ = 0;
  uint16_t pending_flags_ = kFlagDiscontinuity;
};
}  // namespace smartpen
