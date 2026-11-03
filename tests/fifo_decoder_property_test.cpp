#include "../firmware/IMUpen/FifoDecoder.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

struct ReferenceDecoder {
  std::array<int16_t,6> frame{};
  bool have_x=false;
  bool discard_first=true;
  bool add(unsigned slot,int16_t value) {
    if(slot==0){frame.fill(0);have_x=true;}
    if(have_x)frame.at(slot)=value;
    if(slot!=5 || !have_x)return false;
    have_x=false;
    if(discard_first){discard_first=false;return false;}
    return true;
  }
};

static void compare_case(uint32_t seed,unsigned offset,size_t word_count) {
  std::mt19937 rng(seed ^ (offset*0x9E3779B9u));
  std::uniform_int_distribution<int> value_dist(-32768,32767);
  std::uniform_int_distribution<unsigned> burst_dist(1,73);
  const size_t frame_count=(offset+word_count+5)/6+1;
  std::vector<std::array<int16_t,6>> source(frame_count);
  for(size_t f=0;f<frame_count;++f){
    for(unsigned axis=0;axis<6;++axis){
      const unsigned edge=(unsigned(f)+axis+seed)%5;
      source[f][axis]=edge==0 ? INT16_MIN : edge==1 ? INT16_MAX : int16_t(value_dist(rng));
    }
  }

  FifoDecoder actual;
  ReferenceDecoder reference;
  size_t position=0,emitted=0;
  while(position<word_count){
    const unsigned pattern=unsigned((offset+position)%6);
    const size_t burst=std::min<size_t>(burst_dist(rng),word_count-position);
    if(!actual.accepts(pattern))throw std::runtime_error("actual decoder rejected the expected continuation pattern");
    for(size_t j=0;j<burst;++j){
      const size_t absolute=offset+position;
      const unsigned slot=unsigned(absolute%6);
      const size_t frame=absolute/6;
      const int16_t value=source.at(frame).at(slot);
      const bool actual_emitted=actual.add(slot,value);
      const bool reference_emitted=reference.add(slot,value);
      if(actual_emitted!=reference_emitted)throw std::runtime_error("decoder/reference emission mismatch");
      if(actual_emitted){
        for(unsigned axis=0;axis<6;++axis){
          if(actual.frame[axis]!=reference.frame[axis])
            throw std::runtime_error("decoder/reference axis value mismatch");
        }
        ++emitted;
      }
      ++position;
    }
    actual.finish(pattern,unsigned(burst));
    if(!actual.accepts(unsigned((offset+position)%6)))
      throw std::runtime_error("decoder failed next-pattern invariant after burst boundary");
  }
  (void)emitted;
}

int main(){
  try{
    const std::array<size_t,18> boundaries={0,1,2,5,6,7,47,48,49,95,96,97,
                                            255,256,257,4095,4096,100000};
    for(unsigned offset=0;offset<6;++offset){
      for(size_t length:boundaries)compare_case(0x1A2B3C4Du+offset,offset,length);
    }
    for(uint32_t seed=0;seed<1000;++seed){
      std::mt19937 rng(seed+0xC0FFEEu);
      const size_t length=std::uniform_int_distribution<size_t>(0,100000)(rng);
      const unsigned offset=std::uniform_int_distribution<unsigned>(0,5)(rng);
      compare_case(seed,offset,length);
    }
  }catch(const std::exception& error){
    std::cerr<<"FAIL: "<<error.what()<<"\n";
    return 1;
  }
  std::cout<<"PASS: reference decoder matched 1000 seeded streams and all six patterns across 0..100000 words\n";
  return 0;
}
