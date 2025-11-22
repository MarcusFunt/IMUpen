#include <LSM6DS3.h>
#include <Wire.h>

// XIAO MG24 Sense IMU: LSM6DS3TR-C on I2C addr 0x6A
LSM6DS3 imu(I2C_MODE, 0x6A);

const uint32_t SAMPLE_PERIOD_MS = 10;   // ~100 Hz default

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

// -----------------------------------------------------------------------------
// New functionality: configurable sampling period and optional circular buffer.
//
// The host can send ASCII commands over the USB serial connection to adjust
// the sampling period and the size of an on-device circular buffer that
// maintains the most recent IMU packets.  Commands are newline-terminated and
// are case-sensitive:
//   "PERIOD<ms>"  – Set the interval between samples in milliseconds.  A
//                   positive integer value updates the sampling rate.
//   "BUFFER<n>"   – Set the number of entries kept in the circular buffer.  The
//                   value must be between 1 and MAX_BUFFER_CAPACITY.
//
// If no commands are received the defaults remain unchanged.  Invalid values
// are ignored.  The on-device buffer is currently used only to retain recent
// samples; it does not change the streaming behaviour.

// Maximum number of samples that can be retained in the ring buffer.  The
// buffer consumes RAM, so the maximum should remain modest.
#define MAX_BUFFER_CAPACITY 512

// Circular buffer for storing recent IMU packets on the device.  The buffer
// length may be adjusted at runtime via the BUFFER command but will never
// exceed MAX_BUFFER_CAPACITY.
static ImuPacket ringBuffer[MAX_BUFFER_CAPACITY];

// Current size of the circular buffer (initially small).  This value is
// modifiable at runtime via the BUFFER command.  Declared volatile because it
// may be updated from interrupt context in the future.
volatile uint16_t buffer_length = 128;

// Index into the circular buffer; incremented modulo ``buffer_length`` when
// storing new packets.
static volatile uint16_t ringHead = 0;

// Current sampling period in milliseconds.  This is initialised from
// SAMPLE_PERIOD_MS but can be updated via the PERIOD command.  Declared
// volatile because it may be updated asynchronously from the main loop.
volatile uint32_t current_sample_period_ms = SAMPLE_PERIOD_MS;

// Accumulate incoming command characters until a newline is seen.  We use
// Arduino's String class because commands are small and infrequent.
String command_buffer = "";

// Compute a simple additive checksum over a byte buffer.  The sum wraps at
// 16 bits.
uint16_t compute_checksum(const uint8_t *data, size_t len) {
  uint32_t sum = 0;
  for (size_t i = 0; i < len; ++i) {
    sum += data[i];
  }
  return static_cast<uint16_t>(sum & 0xFFFF);
}

// Process any complete commands received over the serial port.  Supported
// commands allow the host to update ``current_sample_period_ms`` and
// ``buffer_length``.
void process_serial_commands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    // On CR/LF we treat the accumulated buffer as a complete command.
    if (c == '\n' || c == '\r') {
      command_buffer.trim();
      if (command_buffer.length() > 0) {
        // Handle the PERIOD command.  The remainder of the string after
        // "PERIOD" is interpreted as an integer value in milliseconds.
        if (command_buffer.startsWith("PERIOD")) {
          String valueStr = command_buffer.substring(6);
          valueStr.trim();
          long value = valueStr.toInt();
          if (value > 0) {
            current_sample_period_ms = (uint32_t)value;
          }
        }
        // Handle the BUFFER command.  The remainder of the string after
        // "BUFFER" specifies the desired buffer length.
        else if (command_buffer.startsWith("BUFFER")) {
          String valueStr = command_buffer.substring(6);
          valueStr.trim();
          long value = valueStr.toInt();
          if (value > 0 && value <= MAX_BUFFER_CAPACITY) {
            buffer_length = (uint16_t)value;
          }
        }
      }
      command_buffer = "";
    } else {
      // Append printable characters to the buffer.  Avoid unbounded growth by
      // limiting the buffer length.
      if (command_buffer.length() < 64) {
        command_buffer += c;
      }
    }
  }
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
  // Check for and process any commands from the host before sampling.
  process_serial_commands();

  static uint32_t last_sample_time = 0;
  uint32_t now = millis();

  // simple timing based on the current sampling period
  if (now - last_sample_time < current_sample_period_ms) {
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

  // Store packet into circular buffer (ring).  Note that the buffer is
  // maintained separately from the streaming path.
  ringBuffer[ringHead] = packet;
  ringHead = (ringHead + 1) % buffer_length;

  // Transmit the packet immediately to the host.
  Serial.write(reinterpret_cast<const uint8_t*>(&packet), sizeof(packet));
}