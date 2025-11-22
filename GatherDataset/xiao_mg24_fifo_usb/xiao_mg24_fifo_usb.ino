/*
  xiao_mg24_fifo_usb.ino

  Example firmware for the Seeed Studio XIAO MG24 Sense board that records
  accelerometer and gyroscope data from the on‑board LSM6DS3TR‑C IMU using
  the device’s FIFO and a shared system clock.  The sketch collects a user
  configurable number of samples at a configurable sample rate, stores the
  raw accelerometer and gyroscope measurements together with a timestamp and
  then transmits the batch to a host computer over USB using the Serial
  interface.  Communication with the host is controlled via simple ASCII
  commands:

    START,<sample_rate_hz>,<num_samples>  Begin recording
    STOP                                  Abort an in‑progress recording
    PING                                  Respond with PONG (useful for port testing)

  The host can parse the CSV formatted output between the DATA_START and
  DATA_END markers and write it to disk.  See the accompanying Python GUI
  (xiao_mg24_fifo_gui.py) for an example implementation.

  The sketch uses the SparkFun LSM6DS3 Arduino library which exposes high
  level APIs for configuring the IMU and reading from the FIFO.  Install
  version 1.1.3 or later of the “SparkFun LSM6DS3 Arduino Library” via the
  Arduino Library Manager before compiling this sketch.

  Hardware notes:
    • XIAO MG24 Sense powers the IMU via PD5.  The pin must be driven high
      before attempting to communicate with the sensor.
    • The default I²C address for the LSM6DS3TR‑C on this board is 0x6A.

  © 2025, Example code released under the MIT license.
*/

#include <SparkFunLSM6DS3.h>
#include <Wire.h>

// Instantiate the IMU.  We use I2C and leave CS undefined since the
// LSM6DS3TR‑C on XIAO MG24 Sense is only connected via I2C.  The
// constructor takes the communication mode and the chip select pin for SPI
// (unused for I2C).  Use the default address 0x6A.
LSM6DS3 imu(I2C_MODE, 0x6A);

// Maximum number of samples that can be stored in memory.  Each sample
// comprises six floating point values and a timestamp.  If you need to
// collect more than this many samples, increase the constant and ensure
// that the available RAM is sufficient (each sample uses 4 bytes * 7
// fields ≈ 28 bytes).  On the MG24 there is ~32 kB RAM; 1024 samples use
// roughly 28 kB plus overhead.
const uint16_t MAX_SAMPLES = 512;

// Buffers for recorded data
static uint32_t timestamps[MAX_SAMPLES];
static float accelData[MAX_SAMPLES][3];
static float gyroData[MAX_SAMPLES][3];

// Recording configuration
static uint16_t requestedSamples = 0;
static uint16_t sampleRate = 104; // default 104 Hz

// State flags
static bool recording = false;
static uint16_t samplesCollected = 0;

// Helper to read a line from Serial up to a newline.  Returns true if a
// complete line was read.
bool readLine(String &line) {
  while (Serial.available() > 0) {
    char c = Serial.read();
    // Filter out carriage returns to simplify line endings
    if (c == '\r') continue;
    if (c == '\n') {
      return true;
    }
    line += c;
  }
  return false;
}

// Process a command received from the host.  Commands are of the form
// “START,rate,samples”, “STOP” or “PING”.
void handleCommand(const String &cmd) {
  if (cmd.length() == 0) return;
  // Convert to uppercase for comparison without modifying the original
  String up = cmd;
  up.toUpperCase();
  if (up.startsWith("PING")) {
    Serial.println("PONG");
    return;
  }
  if (up.startsWith("STOP")) {
    recording = false;
    Serial.println("ABORTED");
    return;
  }
  // START command: parse two comma separated arguments
  if (up.startsWith("START")) {
    int firstComma = cmd.indexOf(',');
    int secondComma = cmd.indexOf(',', firstComma + 1);
    if (firstComma == -1 || secondComma == -1) {
      Serial.println("ERR,Invalid START syntax");
      return;
    }
    // Extract numeric substrings
    String rateStr = cmd.substring(firstComma + 1, secondComma);
    String samplesStr = cmd.substring(secondComma + 1);
    uint16_t rate = rateStr.toInt();
    uint16_t samples = samplesStr.toInt();
    if (rate == 0 || samples == 0) {
      Serial.println("ERR,Bad rate or sample count");
      return;
    }
    if (samples > MAX_SAMPLES) {
      Serial.print("ERR,Max samples is ");
      Serial.println(MAX_SAMPLES);
      return;
    }
    sampleRate = rate;
    requestedSamples = samples;
    // begin recording on next loop iteration
    recording = true;
    samplesCollected = 0;
    Serial.print("ACK,START,rate=");
    Serial.print(sampleRate);
    Serial.print(",samples=");
    Serial.println(requestedSamples);
    return;
  }
  // Unknown command
  Serial.println("ERR,Unknown command");
}

// Configure the IMU based on the requested sample rate and enable FIFO
void configureIMU() {
  // Stop and clear any existing FIFO configuration
  imu.fifoEnd();
  imu.fifoClear();
  // Base settings: enable both sensors
  imu.settings.gyroEnabled = 1;
  imu.settings.accelEnabled = 1;
  // Full scale ranges (can be adjusted as needed)
  imu.settings.gyroRange = 2000;    // ±2000 dps
  imu.settings.accelRange = 16;     // ±16 g
  // Apply sample rate (gyroscope and accelerometer share the same set)
  // Allowed values: 13, 26, 52, 104, 208, 416, 833, 1666 Hz (IMU supports up to 6.6 kHz)
  imu.settings.gyroSampleRate = sampleRate;
  imu.settings.accelSampleRate = sampleRate;
  // Include both sensors in FIFO; decimation of 1 (no decimation)
  imu.settings.gyroFifoEnabled = 1;
  imu.settings.accelFifoEnabled = 1;
  imu.settings.gyroFifoDecimation = 1;
  imu.settings.accelFifoDecimation = 1;
  // FIFO threshold (watermark).  Each sample uses 6 words (gyroX,Y,Z + accelX,Y,Z),
  // but the threshold is specified in words.  We set the threshold equal
  // to the number of desired samples; the library will trigger the
  // watermark flag when the FIFO reaches this depth.
  imu.settings.fifoThreshold = requestedSamples * 6;
  // FIFO sample rate (for timestamp alignment).  Use same as sensor sample rate.
  imu.settings.fifoSampleRate = sampleRate;
  // FIFO mode: 6 = continuous mode; FIFO will overwrite oldest samples when
  // full but because we stop after collecting our requested samples it will
  // not overflow.  See datasheet §5.5.2.  Other modes such as stop when
  // full (1) are also possible.
  imu.settings.fifoModeWord = 6;
  // Apply configuration by calling begin() again.  Use commMode=1 for I2C.
  imu.settings.commMode = 1;
  if (imu.begin() != 0) {
    Serial.println("ERR,IMU init failed");
    return;
  }
  // Clear FIFO and start streaming
  imu.fifoClear();
  imu.fifoBegin();
}

void setup() {
  // Start Serial for USB CDC.  115200 baud is sufficient for batch transfer.
  Serial.begin(115200);
  // Wait briefly for host to open the port.  Without this the board may
  // reset repeatedly on some hosts due to DTR toggling.
  uint32_t waitStart = millis();
  while (!Serial && (millis() - waitStart < 2000)) {
    delay(10);
  }
  // Enable IMU power via PD5 per Seeed documentation
  pinMode(PD5, OUTPUT);
  digitalWrite(PD5, HIGH);
  delay(100);
  // Initialise IMU in a safe default configuration.  This call also
  // configures internal registers and verifies communication.
  if (imu.begin() != 0) {
    Serial.println("ERR,Failed to initialise IMU");
  }
  Serial.println("READY");
}

void loop() {
  // Process commands from host
  static String inputBuffer;
  if (readLine(inputBuffer)) {
    String line = inputBuffer;
    inputBuffer = "";
    handleCommand(line);
  }
  // If not currently recording, do nothing further
  if (!recording) {
    return;
  }
  // When recording just started, configure the IMU
  if (samplesCollected == 0) {
    configureIMU();
  }
  // Read FIFO until we have all samples
  while (samplesCollected < requestedSamples) {
    // Wait for watermark flag (bit 15) or until enough data is present in FIFO
    uint16_t status = imu.fifoGetStatus();
    // Check watermark flag or full flag; bit definitions per datasheet and library:
    // bit15: FIFO watermark status, bit12: FIFO empty status
    bool watermarkReached = (status & 0x8000) != 0;
    bool fifoNotEmpty = (status & 0x1000) == 0;
    // Only attempt to read when watermark reached or there is data; if no
    // data available yield to avoid locking up the MCU
    if (!watermarkReached && !fifoNotEmpty) {
      break;
    }
    // Each sample consists of 6 consecutive 16‑bit words in order:
    // gyroX, gyroY, gyroZ, accelX, accelY, accelZ
    // Use fifoRead() to get raw values and convert them
    int16_t raw;
    // Gyroscope
    raw = (int16_t)imu.fifoRead();
    float gX = imu.calcGyro(raw);
    raw = (int16_t)imu.fifoRead();
    float gY = imu.calcGyro(raw);
    raw = (int16_t)imu.fifoRead();
    float gZ = imu.calcGyro(raw);
    // Accelerometer
    raw = (int16_t)imu.fifoRead();
    float aX = imu.calcAccel(raw);
    raw = (int16_t)imu.fifoRead();
    float aY = imu.calcAccel(raw);
    raw = (int16_t)imu.fifoRead();
    float aZ = imu.calcAccel(raw);
    // Record timestamp in milliseconds from system clock
    timestamps[samplesCollected] = millis();
    accelData[samplesCollected][0] = aX;
    accelData[samplesCollected][1] = aY;
    accelData[samplesCollected][2] = aZ;
    gyroData[samplesCollected][0] = gX;
    gyroData[samplesCollected][1] = gY;
    gyroData[samplesCollected][2] = gZ;
    samplesCollected++;
    // Break if no more FIFO data or reached sample count
    if (samplesCollected >= requestedSamples) {
      break;
    }
  }
  // When finished collecting, send data and reset
  if (samplesCollected >= requestedSamples) {
    // Stop FIFO streaming
    imu.fifoEnd();
    // Send header
    Serial.println("DATA_START");
    Serial.println("timestamp_ms,acc_x_g,acc_y_g,acc_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps");
    for (uint16_t i = 0; i < requestedSamples; i++) {
      Serial.print(timestamps[i]);
      Serial.print(',');
      Serial.print(accelData[i][0], 6);
      Serial.print(',');
      Serial.print(accelData[i][1], 6);
      Serial.print(',');
      Serial.print(accelData[i][2], 6);
      Serial.print(',');
      Serial.print(gyroData[i][0], 6);
      Serial.print(',');
      Serial.print(gyroData[i][1], 6);
      Serial.print(',');
      Serial.println(gyroData[i][2], 6);
    }
    Serial.println("DATA_END");
    // Reset recording state
    recording = false;
    samplesCollected = 0;
  }
}