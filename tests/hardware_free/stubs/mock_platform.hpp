#pragma once
// Simplified deterministic Arduino/ESP32/FreeRTOS and SPI stand-ins.
// TEST SIMULATOR ONLY. Does not model real CPU contention, analog noise, or electrical timing.
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <array>
#include <string>
#include <vector>

#define IRAM_ATTR
#define OUTPUT 1
#define INPUT 0
#define HIGH 1
#define LOW 0
#define RISING 1
#define MSBFIRST 1
#define SPI_MODE3 3
#define pdTRUE 1
#define pdFALSE 0
#define pdPASS 1
#define portMAX_DELAY 0xffffffffu
#define pdMS_TO_TICKS(x) (x)
#define portYIELD_FROM_ISR() do {} while (0)
#define portENTER_CRITICAL_ISR(x) do {} while (0)
#define portEXIT_CRITICAL_ISR(x) do {} while (0)
#define portENTER_CRITICAL(x) do {} while (0)
#define portEXIT_CRITICAL(x) do {} while (0)

using BaseType_t=int;
using TaskHandle_t=void*;
using QueueHandle_t=void*;
using portMUX_TYPE=int;
#define portMUX_INITIALIZER_UNLOCKED 0
using InterruptHandler=void(*)();

namespace Mock {
inline int64_t clock_us=1000000;
inline int gpio_cs=1;
inline size_t queue_limit=256;
inline size_t queue_item_size=0;
inline int fail_queue_calls=0;
inline std::vector<std::vector<unsigned char>> queued;
inline std::string serial_log;
inline std::deque<uint16_t> fifo;
inline uint8_t registers[256]={};
inline bool fifo_overrun=false;
inline unsigned fifo_pattern=0;
inline unsigned prev_output_word=0;
inline size_t fifo_capacity_words=4096; // 8 KiB LSM6DS3 FIFO; TR-C profile is 2048 words.
inline uint8_t sensor_id=0x69;
inline int readback_mismatch_address=-1;
inline bool stuck_reset=false;
inline int short_read_after_bytes=-1;
inline unsigned transfer_count=0;
inline unsigned transaction_count=0;
inline unsigned spi_errors=0;
inline unsigned dropped_oldest_words=0;
inline bool irq_level=false;
inline bool auto_irq=true;
inline int spi_mode=-1, spi_order=-1, spi_hz=0;
inline bool transaction_active=false;
inline unsigned irq_threshold_words=48;
inline std::array<bool,256> register_read_only{};
inline bool supported_register(unsigned r) {
  switch(r) {
    case 0x06: case 0x07: case 0x08: case 0x09: case 0x0A: case 0x0D:
    case 0x10: case 0x11: case 0x12: case 0x0F:
    case 0x3A: case 0x3B: case 0x3C: case 0x3D: case 0x3E: return true;
    default: return false;
  }
}
inline void set_read_only_registers() {
  register_read_only.fill(false);
  register_read_only[0x0F]=true;
  for(unsigned r=0x3A;r<=0x3E;++r)register_read_only[r]=true;
}
inline unsigned watermark_words() {
  return unsigned(registers[0x06]) | (unsigned(registers[0x07]&0x03)<<8);
}
inline bool fifo_enabled() { return (registers[0x0A]&0x07)==0x06; }
inline void refresh_irq();
inline InterruptHandler interrupt_handler=nullptr;
inline void reset() {
  clock_us=1000000; gpio_cs=1; queue_limit=256; queue_item_size=0; fail_queue_calls=0;
  queued.clear(); serial_log.clear(); fifo.clear();
  std::fill(std::begin(registers),std::end(registers),uint8_t{0});
  sensor_id=0x69; registers[0x0F]=sensor_id; fifo_overrun=false; fifo_pattern=0;
  fifo_capacity_words=4096; readback_mismatch_address=-1; stuck_reset=false;
  short_read_after_bytes=-1; transfer_count=transaction_count=spi_errors=dropped_oldest_words=0;
  irq_level=false; auto_irq=true; spi_mode=spi_order=-1; spi_hz=0;
  transaction_active=false; irq_threshold_words=48; set_read_only_registers();
  prev_output_word=0; interrupt_handler=nullptr;
}
inline uint8_t read_register(unsigned r) {
  switch(r){
    case 0x3A:return uint8_t(fifo.size() & 255u);
    case 0x3B:return uint8_t(((fifo.size() >> 8)&0x0fu) |
      (fifo.size()>=watermark_words() ? 0x80:0) |
      (fifo.size()>=fifo_capacity_words ? 0x20:0) | (fifo_overrun?0x40:0));
    case 0x3C:return uint8_t(fifo_pattern & 255u);
    case 0x3D:return uint8_t((fifo_pattern >> 8)&3u);
    default:
      if(!supported_register(r))return 0;
      if(int(r)==readback_mismatch_address)return uint8_t(registers[r]^0x01);
      if(r==0x12 && stuck_reset && (registers[r]&0x01))return registers[r];
      return registers[r];
  }
}
inline void write_register(unsigned r,uint8_t v) {
  if(!supported_register(r) || register_read_only[r])return;
  registers[r]=v;
  if(r==0x12 && (v&1) && !stuck_reset)registers[r] &= ~uint8_t{1}; // SW_RESET self-clears
  if(r==0x0A && v==0) {fifo.clear();fifo_pattern=0;fifo_overrun=false;}
  if(r==0x06 || r==0x07)irq_threshold_words=watermark_words();
  refresh_irq();
}
inline void push_word(uint16_t word) {
  if(fifo.size()>=fifo_capacity_words) {
    fifo.pop_front();
    fifo_pattern=(fifo_pattern+1)%6;
    fifo_overrun=true;
    ++dropped_oldest_words;
  }
  fifo.push_back(word);
  refresh_irq();
}
inline uint8_t fifo_next_byte() {
  if(fifo.empty())return 0;
  uint16_t w=fifo.front();
  bool high=(prev_output_word & 1u)!=0;
  uint8_t result=high?uint8_t(w>>8):uint8_t(w&255u);
  ++prev_output_word;
  if(high){fifo.pop_front(); fifo_pattern=(fifo_pattern+1)%6; refresh_irq();}
  return result;
}
inline void push_frame(int16_t gx,int16_t gy,int16_t gz,int16_t ax,int16_t ay,int16_t az){
  const int16_t axes[]={gx,gy,gz,ax,ay,az};
  for(auto x: axes)push_word(uint16_t(x));
}
inline void refresh_irq() {
  const bool next=fifo_enabled() && (registers[0x0D]&0x08) &&
                  watermark_words()>0 && fifo.size()>=watermark_words();
  const bool rising=next && !irq_level;
  irq_level=next;
  if(rising && auto_irq && interrupt_handler)interrupt_handler();
}
}

struct MockSerialType {
  void begin(int){}
  void println(const char*s){Mock::serial_log+=s;Mock::serial_log+='\n';}
  int printf(const char*fmt,...){
    char buffer[2000];va_list ap;va_start(ap,fmt);
    const int n=std::vsnprintf(buffer,sizeof(buffer),fmt,ap);va_end(ap);
    if(n>0)Mock::serial_log+=buffer;
    return n;
  }
};
inline MockSerialType Serial;
inline void delay(unsigned ms){Mock::clock_us += int64_t(ms)*1000;}
inline uint32_t millis(){return uint32_t(Mock::clock_us/1000);}
inline void pinMode(int,int){}
inline void digitalWrite(int pin,int level){if(pin==4){Mock::gpio_cs=level;if(level==HIGH)Mock::refresh_irq();}}
inline int digitalRead(int pin){return pin==1 && Mock::irq_level ? HIGH:LOW;}
inline int digitalPinToInterrupt(int pin){return pin;}
inline void attachInterrupt(int,InterruptHandler h,int){Mock::interrupt_handler=h;}
inline int64_t esp_timer_get_time(){return Mock::clock_us;}
inline uint32_t esp_random(){return 424242;}

struct SPISettings{int hz,order,mode;SPISettings(int h,int o,int m):hz(h),order(o),mode(m){}};
struct MockSPI {
  bool first=true,reading=false;
  unsigned address=0,byte_cursor=0;
  void begin(int,int,int,int){}
  void beginTransaction(SPISettings settings){
    if(Mock::transaction_active)++Mock::spi_errors;
    Mock::transaction_active=true;Mock::spi_mode=settings.mode;
    Mock::spi_order=settings.order;Mock::spi_hz=settings.hz;
    first=true;byte_cursor=0;++Mock::transaction_count;
  }
  void endTransaction(){
    if(!Mock::transaction_active || Mock::gpio_cs==LOW)++Mock::spi_errors;
    Mock::transaction_active=false;
  }
  uint8_t transfer(uint8_t val){
    Mock::clock_us += 2; // approximate 8 SPI bits / 4 MHz, not a measured transaction duration
    ++Mock::transfer_count;
    if(!Mock::transaction_active || Mock::gpio_cs!=LOW || Mock::spi_mode!=SPI_MODE3 ||
       Mock::spi_order!=MSBFIRST || Mock::spi_hz<=0){++Mock::spi_errors;return 0xFF;}
    if(Mock::short_read_after_bytes>=0 && int(byte_cursor)>=Mock::short_read_after_bytes) {
      ++byte_cursor;return 0;
    }
    if(first){first=false;reading=bool(val&0x80);address=val&0x7f;return 0;}
    if(reading){
      ++byte_cursor;
      if(address==0x3E)return Mock::fifo_next_byte();
      const uint8_t value=Mock::read_register(address);
      if(Mock::registers[0x12]&0x04)++address;
      return value;
    }
    Mock::write_register(address,val);
    if(Mock::registers[0x12]&0x04)++address;
    return 0;
  }
};
inline MockSPI SPI;

inline QueueHandle_t xQueueCreate(unsigned,unsigned item_size){Mock::queue_item_size=item_size;return reinterpret_cast<void*>(1);}
inline BaseType_t xQueueSend(QueueHandle_t,const void*item,unsigned){
  if(Mock::queued.size()>=Mock::queue_limit){++Mock::fail_queue_calls;return pdFALSE;}
  // store a byte representation; harness knows Sample after .ino is included
  const auto*ptr=static_cast<const unsigned char*>(item);
  Mock::queued.emplace_back(ptr,ptr+Mock::queue_item_size);
  return pdTRUE;
}
inline BaseType_t xQueueReceive(QueueHandle_t,void*,unsigned){return pdFALSE;}
inline TaskHandle_t xTaskGetCurrentTaskHandle(){return reinterpret_cast<void*>(1);}
inline BaseType_t xTaskCreate(void(*)(void*),const char*,unsigned,void*,unsigned,TaskHandle_t*){return pdPASS;}
inline uint32_t ulTaskNotifyTake(BaseType_t,uint32_t){return 1;}
inline void vTaskNotifyGiveFromISR(TaskHandle_t,BaseType_t* woken){*woken=pdFALSE;}
