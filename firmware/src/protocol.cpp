#include "protocol.h"

namespace smartpen {
namespace {
constexpr uint8_t kSync0 = 0x5A;
constexpr uint8_t kSync1 = 0xA5;
constexpr size_t kHeaderSize = 6;
constexpr size_t kMaxPayload = 32;
constexpr size_t kMaxFrame = kHeaderSize + kMaxPayload + 2;

void put16(uint8_t* p, uint16_t x) {
  p[0] = uint8_t(x); p[1] = uint8_t(x >> 8);
}
void put32(uint8_t* p, uint32_t x) {
  for (unsigned i = 0; i < 4; ++i) p[i] = uint8_t(x >> (8 * i));
}
void put64(uint8_t* p, uint64_t x) {
  for (unsigned i = 0; i < 8; ++i) p[i] = uint8_t(x >> (8 * i));
}

bool sendFrame(uint8_t type, const uint8_t* payload, uint16_t length) {
  if (length > kMaxPayload) return false;
  const size_t total = kHeaderSize + length + 2;
  if (Serial.availableForWrite() < static_cast<int>(total)) return false;
  uint8_t frame[kMaxFrame] = {};
  frame[0] = kSync0;
  frame[1] = kSync1;
  frame[2] = kProtocolVersion;
  frame[3] = type;
  put16(frame + 4, length);
  memcpy(frame + kHeaderSize, payload, length);
  put16(frame + kHeaderSize + length,
        crc16(frame + 2, 4 + length));
  return Serial.write(frame, total) == total;
}
}  // namespace

uint16_t crc16(const uint8_t* bytes, size_t size) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < size; ++i) {
    crc ^= uint16_t(bytes[i]) << 8;
    for (unsigned b = 0; b < 8; ++b)
      crc = (crc & 0x8000) ? uint16_t((crc << 1) ^ 0x1021)
                           : uint16_t(crc << 1);
  }
  return crc;
}

bool sendSample(const Sample& s) {
  uint8_t p[26] = {};
  put32(p, s.sequence);
  put64(p + 4, s.timestamp_us);
  // Wire order is accel xyz THEN gyro xyz, despite FIFO order gyro/accel.
  put16(p + 12, uint16_t(s.ax));
  put16(p + 14, uint16_t(s.ay));
  put16(p + 16, uint16_t(s.az));
  put16(p + 18, uint16_t(s.gx));
  put16(p + 20, uint16_t(s.gy));
  put16(p + 22, uint16_t(s.gz));
  put16(p + 24, s.flags);
  return sendFrame(kSampleType, p, sizeof(p));
}
bool sendConfig(const Configuration& c) {
  uint8_t p[12] = {};
  put16(p, c.sample_rate_hz);
  put16(p + 2, c.accel_range_g);
  put16(p + 4, c.gyro_range_dps);
  p[6] = c.i2c_address;
  p[7] = c.sda_pin;
  p[8] = c.scl_pin;
  p[9] = c.who_am_i;
  put16(p + 10, c.flags);
  return sendFrame(kConfigType, p, sizeof(p));
}
bool sendError(uint16_t code, uint32_t count) {
  uint8_t p[6] = {};
  put16(p, code);
  put32(p + 2, count);
  return sendFrame(kErrorType, p, sizeof(p));
}
}  // namespace smartpen
