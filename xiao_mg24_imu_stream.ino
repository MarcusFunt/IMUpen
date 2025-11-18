#include <LSM6DS3.h>
#include <Wire.h>

// XIAO MG24 Sense IMU: LSM6DS3TR-C on I2C addr 0x6A
LSM6DS3 imu(I2C_MODE, 0x6A);

const uint32_t SAMPLE_PERIOD_MS = 10;   // ~100 Hz

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

  // CSV header so Python can optionally skip it
  Serial.println("t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
}

void loop() {
  static uint32_t last_sample_time = 0;
  uint32_t now = millis();

  // simple timing for ~100 Hz
  if (now - last_sample_time < SAMPLE_PERIOD_MS) {
    return;
  }
  last_sample_time = now;

  float ax = imu.readFloatAccelX();   // units: g
  float ay = imu.readFloatAccelY();
  float az = imu.readFloatAccelZ();

  float gx = imu.readFloatGyroX();    // units: deg/s
  float gy = imu.readFloatGyroY();
  float gz = imu.readFloatGyroZ();

  Serial.print(now);
  Serial.print(',');
  Serial.print(ax, 6);
  Serial.print(',');
  Serial.print(ay, 6);
  Serial.print(',');
  Serial.print(az, 6);
  Serial.print(',');
  Serial.print(gx, 6);
  Serial.print(',');
  Serial.print(gy, 6);
  Serial.print(',');
  Serial.print(gz, 6);
  Serial.println();
}
