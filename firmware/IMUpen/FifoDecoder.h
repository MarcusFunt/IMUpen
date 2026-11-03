#pragma once
#include <stdint.h>

// The configured FIFO pattern is Gx,Gy,Gz,Ax,Ay,Az, repeated in 16-bit words.
struct FifoDecoder {
  int expected = -1;
  bool assembling = false;
  bool discardFirst = true;
  int16_t frame[6] = {};

  void reset() {
    expected = -1;
    assembling = false;
    discardFirst = true;
  }
  bool accepts(unsigned pattern) const {
    return pattern < 6 && (expected < 0 || pattern == static_cast<unsigned>(expected));
  }
  bool add(unsigned slot, int16_t value) {
    if (slot == 0) assembling = true;
    if (assembling) frame[slot] = value;
    if (slot != 5 || !assembling) return false;
    assembling = false;
    if (discardFirst) {
      discardFirst = false;
      return false;
    }
    return true;
  }
  void finish(unsigned pattern, unsigned words) { expected = (pattern + words) % 6; }
};
