#pragma once
// Simplified deterministic Arduino/ESP32/FreeRTOS and SPI stand-ins.
// TEST SIMULATOR ONLY. Does not model real CPU contention, analog noise, or electrical timing.
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
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
inline InterruptHandler interrupt_handler=nullptr;
inline void reset() {
  clock_us=1000000; gpio_cs=1; queue_limit=256; queue_item_size=0; fail_queue_calls=0;
  queued.clear(); serial_log.clear(); fifo.clear();
  std::fill(std::begin(registers),std::end(registers),uint8_t{0});
  registers[0x0F]=0x69; fifo_overrun=false; fifo_pattern=0;
  prev_output_word=0; interrupt_handler=nullptr;
}
inline uint8_t read_register(unsigned r) {
  switch(r){
    case 0x3A:return uint8_t(fifo.size() & 255u);
    case 0x3B:return uint8_t(((fifo.size() >> 8)&0x0fu) | (fifo.size()>=48 ? 0x80:0) | (fifo_overrun?0x40:0));
    case 0x3C:return uint8_t(fifo_pattern & 255u);
    case 0x3D:return uint8_t((fifo_pattern >> 8)&3u);
    default:return registers[r];
  }
}
inline void write_register(unsigned r,uint8_t v) {
  registers[r]=v;
  if(r==0x12 && (v&1))registers[r] &= ~uint8_t{1}; // mock auto-clearing SW_RESET
  if(r==0x0A && v==0) {fifo.clear();fifo_pattern=0;fifo_overrun=false;}
}
inline uint8_t fifo_next_byte() {
  if(fifo.empty())return 0;
  uint16_t w=fifo.front();
  bool high=(prev_output_word & 1u)!=0;
  uint8_t result=high?uint8_t(w>>8):uint8_t(w&255u);
  ++prev_output_word;
  if(high){fifo.pop_front(); fifo_pattern=(fifo_pattern+1)%6;}
  return result;
}
inline void push_frame(int16_t gx,int16_t gy,int16_t gz,int16_t ax,int16_t ay,int16_t az){
  const int16_t axes[]={gx,gy,gz,ax,ay,az};
  for(auto x: axes)fifo.push_back(uint16_t(x));
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
inline void digitalWrite(int pin,int level){if(pin==4)Mock::gpio_cs=level;}
inline int digitalRead(int pin){return pin==1 && Mock::fifo.size()>=48 && (Mock::registers[0x0D]&0x08) ? HIGH:LOW;}
inline int digitalPinToInterrupt(int pin){return pin;}
inline void attachInterrupt(int,InterruptHandler h,int){Mock::interrupt_handler=h;}
inline int64_t esp_timer_get_time(){return Mock::clock_us;}
inline uint32_t esp_random(){return 424242;}

struct SPISettings{SPISettings(int,int,int){}};
struct MockSPI {
  bool first=true,reading=false;
  unsigned address=0,byte_cursor=0;
  void begin(int,int,int,int){}
  void beginTransaction(SPISettings){first=true;byte_cursor=0;}
  void endTransaction(){}
  uint8_t transfer(uint8_t val){
    Mock::clock_us += 2; // approximate 8 SPI bits / 4 MHz, not a measured transaction duration
    if(first){first=false;reading=bool(val&0x80);address=val&0x7f;return 0;}
    if(reading){
      if(address==0x3E)return Mock::fifo_next_byte();
      return Mock::read_register(address++);
    }
    Mock::write_register(address++,val);return 0;
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
