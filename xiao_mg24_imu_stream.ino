#include <LSM6DS3.h>
#include <Wire.h>

// XIAO MG24 Sense IMU: LSM6DS3TR-C on I2C addr 0x6A
LSM6DS3 imu(I2C_MODE, 0x6A);

const uint32_t SAMPLE_PERIOD_MS = 10;   // ~100 Hz

// Binary packet framing for reliable comms ----------------------------------
const uint16_t PACKET_PREAMBLE = 0xAA55;
const uint16_t PACKET_VERSION = 1;

struct __attribute__((packed)) ImuPacket {
  uint16_t preamble;      // 0xAA55
  uint16_t version;       // protocol version
  uint32_t sequence;      // monotonically increasing packet number
  uint32_t timestamp_ms;  // millis() when the sample was captured
  float ax_g;
  float ay_g;
  float az_g;
  float gx_dps;
  float gy_dps;
  float gz_dps;
  uint16_t checksum;      // simple 16-bit sum over preceding bytes
};

uint32_t sample_sequence = 0;

uint16_t compute_checksum(const uint8_t *data, size_t len) {
  uint32_t sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += data[i];
  }
  return static_cast<uint16_t>(sum & 0xFFFF);
}

void setup() {
  Serial.begin(115200);
  // Some host OSes toggle DTR when the port is opened which resets the
  // microcontroller.  If we block forever waiting for Serial here, the sketch
  // may never make it into loop(), so only wait for a short window.
  const uint32_t serial_wait_start = millis();
  while (!Serial && (millis() - serial_wait_start) < 2000) {
    delay(10);
  }

  // On XIAO MG24 Sense the IMU power is enabled via PD5 (per Seeed's docs)
  pinMode(PD5, OUTPUT);
  digitalWrite(PD5, HIGH);
  delay(100);  // give the IMU time to power up

  if (imu.begin() != 0) {
    Serial.println("IMU init failed");
    while (1) {
      Serial.println("Check IMU / wiring / library");
      delay(1000);
    }
  }

}

void loop() {
  static uint32_t last_sample_time = 0;
  uint32_t now = millis();

  // simple timing for ~100 Hz
  if (now - last_sample_time < SAMPLE_PERIOD_MS) {
    return;
  }
  last_sample_time = now;

  ImuPacket packet;
  packet.preamble = PACKET_PREAMBLE;
  packet.version = PACKET_VERSION;
  packet.sequence = sample_sequence++;
  packet.timestamp_ms = now;
  packet.ax_g = imu.readFloatAccelX();   // units: g
  packet.ay_g = imu.readFloatAccelY();
  packet.az_g = imu.readFloatAccelZ();
  packet.gx_dps = imu.readFloatGyroX();  // units: deg/s
  packet.gy_dps = imu.readFloatGyroY();
  packet.gz_dps = imu.readFloatGyroZ();
  packet.checksum = 0;
  packet.checksum = compute_checksum(reinterpret_cast<const uint8_t*>(&packet),
                                     sizeof(packet) - sizeof(packet.checksum));

  Serial.write(reinterpret_cast<const uint8_t*>(&packet), sizeof(packet));
}
