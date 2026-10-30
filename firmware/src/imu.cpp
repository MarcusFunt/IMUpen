#include "imu.h"

namespace smartpen {
namespace {
constexpr uint8_t WHO_AM_I = 0x0F;
constexpr uint8_t CTRL1_XL = 0x10;
constexpr uint8_t CTRL2_G = 0x11;
constexpr uint8_t CTRL3_C = 0x12;
constexpr uint8_t FIFO_CTRL1 = 0x06;
constexpr uint8_t FIFO_CTRL2 = 0x07;
constexpr uint8_t FIFO_CTRL3 = 0x08;
constexpr uint8_t FIFO_CTRL4 = 0x09;
constexpr uint8_t FIFO_CTRL5 = 0x0A;
constexpr uint8_t FIFO_STATUS1 = 0x3A;
constexpr uint8_t FIFO_PATTERN_L = 0x3C;
constexpr uint8_t FIFO_DATA_OUT_L = 0x3E;
constexpr uint8_t FIFO_BYPASS = 0x28;  // FIFO ODR 208 Hz; bypass mode.
constexpr uint8_t FIFO_CONTINUOUS = 0x2E; // FIFO ODR 208 Hz; continuous.
}  // namespace

bool Lsm6ds3Fifo::readBytes(uint8_t reg, uint8_t* out, size_t count) {
  Wire.beginTransmission(address_);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(static_cast<int>(address_), static_cast<int>(count),
                       static_cast<int>(true)) != count) return false;
  for (size_t i = 0; i < count; ++i) {
    if (!Wire.available()) return false;
    out[i] = uint8_t(Wire.read());
  }
  return true;
}
bool Lsm6ds3Fifo::writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(address_);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}
bool Lsm6ds3Fifo::readWord(int16_t& value) {
  uint8_t p[2];
  if (!readBytes(FIFO_DATA_OUT_L, p, 2)) return false;
  value = static_cast<int16_t>(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
  return true;
}
bool Lsm6ds3Fifo::flushFifo() {
  // Reset FIFO read pattern and restart continuous capture.
  return writeReg(FIFO_CTRL5, FIFO_BYPASS) &&
         writeReg(FIFO_CTRL5, FIFO_CONTINUOUS);
}
void Lsm6ds3Fifo::recoverI2c() {
  ++i2c_failures_;
  pending_flags_ |= kFlagI2cError | kFlagDiscontinuity;
  flushFifo();
}
bool Lsm6ds3Fifo::begin(int sda_pin, int scl_pin, uint32_t i2c_hz) {
  if (!Wire.begin(sda_pin, scl_pin, i2c_hz)) return false;
  Wire.setTimeOut(15);
  // Only accept WHO_AM_I=0x69 (LSM6DS3). Do not silently use a
  // different chip with potentially incompatible FIFO behavior.
  for (uint8_t candidate : {uint8_t(0x6A), uint8_t(0x6B)}) {
    address_ = candidate;
    uint8_t id = 0;
    if (readBytes(WHO_AM_I, &id, 1) && id == 0x69) {
      who_am_i_ = id;
      break;
    }
  }
  if (!who_am_i_) return false;

  // BDU=1, IF_INC=1. Accelerometer 208 Hz +/-4g; gyro 208 Hz +/-500 dps.
  if (!writeReg(FIFO_CTRL5, FIFO_BYPASS) ||
      !writeReg(CTRL3_C, 0x44) ||
      !writeReg(CTRL1_XL, 0x58) ||
      !writeReg(CTRL2_G, 0x54) ||
      !writeReg(FIFO_CTRL1, 0x00) ||
      !writeReg(FIFO_CTRL2, 0x00) ||
      !writeReg(FIFO_CTRL3, 0x09) || // G /1, XL /1.
      !writeReg(FIFO_CTRL4, 0x00) ||
      !flushFifo()) return false;
  return true;
}

bool Lsm6ds3Fifo::next(RawImuSample& out, uint16_t& flags) {
  uint8_t status[2];
  if (!readBytes(FIFO_STATUS1, status, 2)) {
    recoverI2c();
    return false;
  }
  const uint16_t words = uint16_t(status[0]) | (uint16_t(status[1] & 0x0F) << 8);
  if (status[1] & 0x40) { // FIFO_OVR bit
    ++fifo_overruns_;
    pending_flags_ |= kFlagFifoOverrun | kFlagDiscontinuity;
    flushFifo();
    return false;
  }
  if (words < 6 || (status[1] & 0x10)) return false; // FIFO empty / partial.

  uint8_t pattern_bytes[2];
  if (!readBytes(FIFO_PATTERN_L, pattern_bytes, 2)) {
    recoverI2c();
    return false;
  }
  const uint16_t pattern = uint16_t(pattern_bytes[0]) |
                           (uint16_t(pattern_bytes[1] & 0x03) << 8);
  // With gyro and accel both decimated by 1, FIFO pattern is six words:
  // Gx,Gy,Gz,Ax,Ay,Az. If not aligned, discard partial sample.
  const unsigned offset = pattern % 6;
  if (offset != 0) {
    int16_t discarded;
    for (unsigned i = offset; i < 6; ++i) {
      if (!readWord(discarded)) { recoverI2c(); return false; }
    }
    pending_flags_ |= kFlagDiscontinuity;
    return false;
  }

  int16_t values[6];
  for (int i = 0; i < 6; ++i) {
    if (!readWord(values[i])) { recoverI2c(); return false; }
  }
  out = {values[0], values[1], values[2],
         values[3], values[4], values[5]};
  flags = pending_flags_;
  pending_flags_ = 0;
  return true;
}
}  // namespace smartpen
