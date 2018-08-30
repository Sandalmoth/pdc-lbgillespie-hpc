#ifndef __TIMERS_H__
#define __TIMERS_H__


#include <chrono>


struct IntervalTimer {
  IntervalTimer() {
    prev = std::chrono::steady_clock::now();
  }
  long long operator()() {
    auto diff = std::chrono::steady_clock::now() - prev;
    prev = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
  }
  std::chrono::steady_clock::time_point prev;
};


struct StartTimer {
  StartTimer() {
    start = std::chrono::steady_clock::now();
  }
  long long operator()() {
    auto diff = std::chrono::steady_clock::now() - start;
    return std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
  }
  std::chrono::steady_clock::time_point start;
};


#endif
