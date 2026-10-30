#pragma once
#include <Arduino.h>
#include <stdint.h>

namespace smartpen {
constexpr uint8_t kProtocolVersion = 1;
constexpr uint16_t kSampleRateHz = 208;
constexpr uint16_t kAccelRangeG = 4;
constexpr uint16_t kGyroRangeDps = 500;

enum : uint8_t {
  kSampleType = 1,
  kConfigType = 2,
  kErrorType = 3
};
enum : uint16_t {
  kFlagDiscontinuity = 1u << 0,
  kFlagFifoOverrun = 1u << 1,
  kFlagI2cError = 1u << 2
};
enum : uint16_t {
  kErrorFifoOverruns = 1,
  kErrorI2cFailures = 2,
  kErrorTxDrops = 3
};

// All integers are little-endian on the wire, never memcpy'ed from structs.
// Frame: sync(5A A5), version(u8), type(u8), length(u16),
//        payload(length bytes), CRC16-CCITT-FALSE(u16).
// CRC covers version through the final payload byte, NOT the sync or CRC.
struct Sample {
  uint32_t sequence;
  uint64_t timestamp_us;  // Nominal FIFO 208-Hz grid, not sensor-hardware time.
  int16_t gx, gy, gz;
  int16_t ax, ay, az;
  uint16_t flags;
};
struct Configuration {
  uint16_t sample_rate_hz;
  uint16_t accel_range_g;
  uint16_t gyro_range_dps;
  uint8_t i2c_address;
  uint8_t sda_pin;
  uint8_t scl_pin;
  uint8_t who_am_i;
  uint16_t flags;  // bit 0: timestamps interpolated from nominal ODR.
};

uint16_t crc16(const uint8_t* bytes, size_t size);
bool sendSample(const Sample& sample);
bool sendConfig(const Configuration& config);
bool sendError(uint16_t code, uint32_t count);
}  // namespace smartpen
