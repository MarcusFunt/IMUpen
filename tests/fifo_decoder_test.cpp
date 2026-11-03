#include "../firmware/IMUpen/FifoDecoder.h"
#include <cassert>

int main() {
  FifoDecoder d;
  assert(d.accepts(0));
  for (int i = 0; i < 6; ++i) assert(!d.add(i, i)); // First frame discarded.
  for (int i = 0; i < 5; ++i) assert(!d.add(i, 10 + i));
  assert(d.add(5, 15));
  for (int i = 0; i < 6; ++i) assert(d.frame[i] == 10 + i);
  d.finish(0, 48);
  assert(d.accepts(0) && !d.accepts(2) && !d.accepts(6));
  d.reset();
  assert(d.accepts(2));
  // Start partway through a pattern, then cross a burst boundary.
  for (int i = 0; i < 48; ++i) d.add((2 + i) % 6, i);
  d.finish(2, 48);
  assert(d.accepts(2));
  assert(!d.add(2, 100));
  assert(!d.add(3, 101));
  assert(!d.add(4, 102));
  assert(d.add(5, 103));
  assert(d.frame[0] == 46 && d.frame[1] == 47 && d.frame[2] == 100 && d.frame[5] == 103);
  d.reset();
  for (int i = 0; i < 6; ++i) assert(!d.add(i, i));
}
