// Compile and run the real .ino on a deterministic virtual device.
#include "mock_platform.hpp"
#include "../../firmware/IMUpen/IMUpen.ino"
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>

static_assert(sizeof(Sample) <= 128, "Increase mock queue copy size");
static Sample get_sample(size_t i){
  Sample result{};std::memcpy(&result,Mock::queued.at(i).data(),sizeof(Sample));return result;
}
void require(bool condition,const std::string& text){if(!condition)throw std::runtime_error(text);}
void push_frames(int start, int count){
  for(int i=start;i<start+count;++i)
    Mock::push_frame(i,1000+i,-1000-i,2000+i,-2000-i,15000+i);
}
void setup_device(){
  Mock::reset();
  nextSeq=0;batchNumber=0;lastUsedIrq=0;queueDrops=0;recoveries=0;overruns=0;epoch=0;
  irqCount=0;lastIrqUs=-1;decoder.reset();
  setup();
  require(Mock::registers[0x10]==0x50,"accel register mismatch");
  require(Mock::registers[0x11]==0x54,"gyro register mismatch");
  require(Mock::registers[0x08]==0x09,"FIFO decimation mismatch");
  require(Mock::registers[0x0A]==0x2E,"FIFO mode mismatch");
  require(Mock::registers[0x0D]==0x08,"watermark interrupt config mismatch");
}
void drain_once(){if(Mock::interrupt_handler)Mock::interrupt_handler();loop();}
void test_exact_frames(){
  setup_device();push_frames(0,24);drain_once();
  require(Mock::queued.size()==23,"first full frame must be discarded");
  require(Mock::fifo.empty(),"all three batches should drain");
  for(size_t i=0;i<23;++i){
    auto s=get_sample(i);
    require(s.seq==i && s.epoch==0,"sequence or epoch unexpected");
    require(s.axes[0]==int(i+1) && s.axes[3]==2000+int(i+1),"axis order corrupt");
    require(s.wordEnd==(i<7 ? i+1 : (i-7)%8)*6+5,"wordEnd unexpected");
  }
  require(get_sample(0).fifoWords==144,"first status occupancy incorrect");
  require(get_sample(7).fifoWords==96,"second status occupancy incorrect");
  require(get_sample(15).fifoWords==48,"third status occupancy incorrect");
  require(get_sample(7).irqUs==-1,"second batch must be continuation without a new edge");
  std::cout<<"PASS: actual firmware reads 3 consecutive batches with correct channels, indices and continuation timing\n";
}
void test_queue_overflow(){
  setup_device();Mock::queue_limit=10;push_frames(0,40);drain_once();
  require(Mock::queued.size()==10,"queue capacity not enforced");
  require(nextSeq==39,"sequence must advance even when queue insertion fails");
  require(queueDrops==29,"queueDrops must count failed queued samples");
  std::cout<<"PASS: bounded output queue causes 29 counted drops while seq keeps advancing\n";
}
void test_overrun_recovery(){
  setup_device();push_frames(0,16);Mock::fifo_overrun=true;drain_once();
  require(recoveries==1 && overruns==1 && epoch==1,"overrun must reset FIFO and bump epoch");
  require(Mock::fifo.empty() && Mock::queued.empty(),"invalid FIFO must not output samples");
  push_frames(100,16);drain_once();
  require(Mock::queued.size()==15,"first post-recovery full frame should be discarded");
  require(get_sample(0).epoch==1 && get_sample(0).axes[0]==101,"post-recovery data not correctly labeled");
  std::cout<<"PASS: simulated FIFO overrun creates fresh epoch and discards first new frame\n";
}
void test_nonzero_pattern(){
  setup_device();push_frames(0,24);
  // Drop two words to mimic a FIFO whose next word is gz (pattern=2).
  Mock::fifo.pop_front();Mock::fifo.pop_front();Mock::fifo_pattern=2;
  drain_once();
  require(Mock::queued.size()==14,"expected two processed batches, with initial partial frame and first full frame discarded");
  for(size_t i=0;i<14;++i){
    auto s=get_sample(i);
    require(s.axes[0]==int(i+2) && s.axes[3]==int(i+2002),"nonzero-pattern channels corrupt");
    require(s.pattern==2 && s.wordEnd==(i<6 ? i+2 : (i-6)%8)*6+3,"nonzero pattern data position incorrect");
  }
  std::cout<<"PASS: real C++ sketch handles a misaligned FIFO start pattern with proper frame boundaries\n";
}
void test_pattern_mismatch_recovery(){
  setup_device();push_frames(0,16);
  decoder.expected=3; // emulate corruption of FIFO read position after a bad transaction
  drain_once();
  require(recoveries==1 && epoch==1,"pattern desynchronization must reset FIFO");
  require(Mock::queued.empty(),"corrupt pattern must not yield usable samples");
  std::cout<<"PASS: unexpected pattern triggers recovery without outputting corrupt frames\n";
}
void test_800_frame_stress(){
  setup_device();Mock::queue_limit=2000;
  for(int b=0;b<100;++b){
    push_frames(b*8,8);drain_once();
  }
  require(Mock::queued.size()==799,"expected 799 surviving frames after startup discard");
  for(int i=0;i<799;++i){
    const auto s=get_sample(i);
    require(s.axes[0]==i+1 && s.axes[3]==2001+i,"800-frame stress payload differs from source");
    require(s.seq==unsigned(i) && s.epoch==0,"800-frame stress seq/epoch differs");
  }
  require(recoveries==0 && queueDrops==0,"healthy repeated capture should not reset or drop");
  std::cout<<"PASS: 800 consecutive synthetic frames survive real firmware decoding without loss or reordering\n";
}
int main(){
  try{
    test_exact_frames();test_queue_overflow();test_overrun_recovery();test_nonzero_pattern();
    test_pattern_mismatch_recovery();test_800_frame_stress();
  }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
  std::cout<<"All firmware-native virtual-peripheral tests passed. Hardware behavior remains unverified.\n";
  return 0;
}
