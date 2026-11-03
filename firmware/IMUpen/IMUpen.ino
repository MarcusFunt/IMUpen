#include <Arduino.h>
#include <SPI.h>
#include <esp_timer.h>
#include <esp_system.h>
#include "FifoDecoder.h"

// XIAO ESP32-S3 GPIOs, connected to the breakout's SCL, SA0, SDA, CS, INT1.
constexpr int PIN_SCK = 7, PIN_MISO = 8, PIN_MOSI = 9, PIN_CS = 4, PIN_INT1 = 1;
constexpr uint8_t FIFO_CTRL1 = 0x06, FIFO_CTRL2 = 0x07, FIFO_CTRL3 = 0x08;
constexpr uint8_t FIFO_CTRL4 = 0x09, FIFO_CTRL5 = 0x0A, INT1_CTRL = 0x0D;
constexpr uint8_t WHO_AM_I = 0x0F, CTRL1_XL = 0x10, CTRL2_G = 0x11, CTRL3_C = 0x12;
constexpr uint8_t FIFO_STATUS1 = 0x3A, FIFO_DATA_OUT_L = 0x3E;
constexpr uint8_t FIFO_CONTINUOUS_208HZ = 0x2E;
constexpr uint16_t WATERMARK_WORDS = 48;  // Eight six-word patterns.
constexpr size_t QUEUE_LENGTH = 256;
// Set true only after identifying the physical 0x6A device as LSM6DS3TR-C.
constexpr bool CONFIRMED_LSM6DS3TR_C = false;
SPISettings imuSPI(4000000, MSBFIRST, SPI_MODE3);

struct Sample {
  uint64_t seq, batch, irqCount, queueDrops, recoveries, overruns;
  int64_t irqUs, readStartUs, readEndUs;
  uint32_t epoch;
  uint16_t fifoWords;
  uint8_t pattern, wordEnd;
  int16_t axes[6];
};

QueueHandle_t sampleQueue;
TaskHandle_t acquisitionTask;
portMUX_TYPE irqMux = portMUX_INITIALIZER_UNLOCKED;
volatile uint64_t irqCount = 0;
volatile int64_t lastIrqUs = -1;
uint64_t nextSeq = 0, batchNumber = 0, lastUsedIrq = 0, queueDrops = 0;
uint64_t recoveries = 0, overruns = 0;
uint32_t epoch = 0;
uint32_t bootId;
uint8_t sensorId;
FifoDecoder decoder;

void IRAM_ATTR onWatermark() {
  const int64_t now = esp_timer_get_time();
  portENTER_CRITICAL_ISR(&irqMux);
  ++irqCount;
  lastIrqUs = now;
  portEXIT_CRITICAL_ISR(&irqMux);
  BaseType_t woken = pdFALSE;
  vTaskNotifyGiveFromISR(acquisitionTask, &woken);
  if (woken) portYIELD_FROM_ISR();
}

uint8_t readReg(uint8_t address) {
  SPI.beginTransaction(imuSPI);
  digitalWrite(PIN_CS, LOW);
  SPI.transfer(address | 0x80);
  const uint8_t value = SPI.transfer(0);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
  return value;
}

void writeReg(uint8_t address, uint8_t value) {
  SPI.beginTransaction(imuSPI);
  digitalWrite(PIN_CS, LOW);
  SPI.transfer(address & 0x7F);
  SPI.transfer(value);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
}

void readBurst(uint8_t address, uint8_t *bytes, size_t count) {
  SPI.beginTransaction(imuSPI);
  digitalWrite(PIN_CS, LOW);
  SPI.transfer(address | 0x80);
  for (size_t i = 0; i < count; ++i) bytes[i] = SPI.transfer(0);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
}

void stopWithError(const char *message) {
  Serial.printf("# ERROR: %s\n", message);
  while (true) delay(1000);
}

void writeChecked(uint8_t address, uint8_t value) {
  writeReg(address, value);
  if (readReg(address) != value) {
    Serial.printf("# ERROR: register 0x%02X readback failed\n", address);
    stopWithError("Check SPI wiring and sensor variant.");
  }
}

void outputTask(void *) {
  Sample s;
  for (;;) {
    if (xQueueReceive(sampleQueue, &s, portMAX_DELAY) != pdTRUE) continue;
    Serial.printf("%lu,%u,%llu,%lu,%llu,%llu,%lld,%lld,%lld,%u,%u,%u,%d,%d,%d,%d,%d,%d,%llu,%llu,%llu\n",
      (unsigned long)bootId, (unsigned)sensorId, (unsigned long long)s.seq,
      (unsigned long)s.epoch, (unsigned long long)s.batch,
      (unsigned long long)s.irqCount, (long long)s.irqUs,
      (long long)s.readStartUs, (long long)s.readEndUs,
      (unsigned)s.fifoWords, (unsigned)s.pattern, (unsigned)s.wordEnd,
      s.axes[0], s.axes[1], s.axes[2], s.axes[3], s.axes[4], s.axes[5],
      (unsigned long long)s.queueDrops, (unsigned long long)s.recoveries,
      (unsigned long long)s.overruns);
  }
}

void resetFifo() {
  // Each reset marks an unknown gap; discard the first complete pattern after
  // switching FIFO modes as required by the datasheet.
  writeReg(FIFO_CTRL5, 0x00);
  decoder.reset();
  ++epoch;
  ++recoveries;
  writeReg(FIFO_CTRL5, FIFO_CONTINUOUS_208HZ);
}

void setup() {
  Serial.begin(921600);
  delay(1500);
  pinMode(PIN_CS, OUTPUT);
  digitalWrite(PIN_CS, HIGH);
  pinMode(PIN_INT1, INPUT);
  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI, PIN_CS);
  const uint8_t id = readReg(WHO_AM_I);
  sensorId = id;
  bootId = esp_random();
  Serial.printf("# WHO_AM_I=0x%02X\n", id);
  // 0x6A is shared by different ST parts; their FIFO maps are not interchangeable.
  if (id != 0x69 && !(id == 0x6A && CONFIRMED_LSM6DS3TR_C))
    stopWithError("Unrecognized/unconfirmed variant; identify the sensor and check the FIFO map.");
  writeReg(CTRL3_C, 0x01);
  const uint32_t start = millis();
  while (readReg(CTRL3_C) & 0x01) {
    if ((uint32_t)(millis() - start) > 1000) stopWithError("Sensor reset timed out.");
    delay(1);
  }
  writeChecked(INT1_CTRL, 0x00);
  writeChecked(FIFO_CTRL5, 0x00);
  writeChecked(CTRL3_C, 0x44);     // BDU, auto-increment, 4-wire SPI.
  writeChecked(CTRL1_XL, 0x50);    // 208 Hz, +/-2 g.
  writeChecked(CTRL2_G, 0x54);     // 208 Hz, +/-500 dps.
  writeChecked(FIFO_CTRL1, WATERMARK_WORDS);
  writeChecked(FIFO_CTRL2, 0x00);
  writeChecked(FIFO_CTRL3, 0x09);  // G and XL: no decimation.
  writeChecked(FIFO_CTRL4, 0x00);  // No auxiliary data sets.
  delay(1000); // Allow sensor startup to settle before enabling FIFO.

  sampleQueue = xQueueCreate(QUEUE_LENGTH, sizeof(Sample));
  if (!sampleQueue) stopWithError("Sample queue allocation failed.");
  acquisitionTask = xTaskGetCurrentTaskHandle();
  if (xTaskCreate(outputTask, "imu-output", 4096, nullptr, 1, nullptr) != pdPASS)
    stopWithError("Output task creation failed.");

  Serial.println("# format=imupen_fifo_v1 odr_nominal_hz=208 watermark_words=48 spi_hz=4000000");
  Serial.println("# pattern=gx,gy,gz,ax,ay,az first_frame_discarded=1 timestamps=host_us_since_boot");
  Serial.println("# boot_id,who_am_i,seq,epoch,batch,irq_count,irq_us,read_start_us,read_end_us,fifo_words,pattern,word_end,gx,gy,gz,ax,ay,az,queue_drops,fifo_recoveries,overrun_observations");
  attachInterrupt(digitalPinToInterrupt(PIN_INT1), onWatermark, RISING);
  writeChecked(INT1_CTRL, 0x08); // FIFO threshold on INT1.
  writeChecked(FIFO_CTRL5, FIFO_CONTINUOUS_208HZ);
}

void loop() {
  ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000));
  // Only read FIFO status synchronized with an asserted watermark signal.
  while (digitalRead(PIN_INT1) == HIGH) {
    uint64_t count;
    int64_t irqTime;
    portENTER_CRITICAL(&irqMux);
    count = irqCount;
    irqTime = lastIrqUs;
    portEXIT_CRITICAL(&irqMux);
    if (count == lastUsedIrq) irqTime = -1;  // Continuation, not a new edge.
    else lastUsedIrq = count;

    uint8_t status[4];
    const int64_t readStart = esp_timer_get_time();
    readBurst(FIFO_STATUS1, status, sizeof(status));
    const uint16_t words = status[0] | ((status[1] & 0x0F) << 8);
    const unsigned pattern = status[2] | ((status[3] & 0x03) << 8);
    if ((status[1] & 0x60) || words < WATERMARK_WORDS || !decoder.accepts(pattern)) {
      if (status[1] & 0x40) ++overruns;
      resetFifo();
      break;
    }
    uint8_t bytes[WATERMARK_WORDS * 2];
    readBurst(FIFO_DATA_OUT_L, bytes, sizeof(bytes));
    const int64_t readEnd = esp_timer_get_time();
    const uint64_t batch = batchNumber++;
    for (unsigned i = 0; i < WATERMARK_WORDS; ++i) {
      const unsigned slot = (pattern + i) % 6;
      const uint16_t raw = (uint16_t)bytes[i * 2] | ((uint16_t)bytes[i * 2 + 1] << 8);
      if (decoder.add(slot, static_cast<int16_t>(raw))) {
        Sample s = {};
        s.seq = nextSeq++; s.epoch = epoch; s.batch = batch;
        s.irqCount = count; s.irqUs = irqTime;
        s.readStartUs = readStart; s.readEndUs = readEnd;
        s.fifoWords = words; s.pattern = pattern; s.wordEnd = i;
        for (unsigned j = 0; j < 6; ++j) s.axes[j] = decoder.frame[j];
        s.queueDrops = queueDrops; s.recoveries = recoveries; s.overruns = overruns;
        if (xQueueSend(sampleQueue, &s, 0) != pdTRUE) ++queueDrops;
      }
    }
    decoder.finish(pattern, WATERMARK_WORDS);
  }
}
