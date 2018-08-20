#ifndef __LB_GILLESPIE_H__
#define __LB_GILLESPIE_H__


#include <chrono>
#include <iostream>
#include <numeric>
#include <random>


#include "xoshiro256ss.h"


template <typename TCell, typename TRng=vigna::xoshiro256ss>
class LB {
private:
  size_t choose_event(const std::vector<double> &rates, double sum) {
    // Because the std::discrete_distribution is very slow (probably because it copies the weight vector)
    // I was forced to make my own implementation.
    double k = std::uniform_real_distribution<double>(0, sum)(rng);
    double cumsum = 0.0;
    for (size_t i = 0; i < rates.size(); ++i) {
      cumsum += rates[i];
      if (cumsum > k)
        return i;
    }
    return rates.size() - 1;
  }


public:
  LB() {
    std::random_device rd;
    rng.seed(rd());
  }


  void add_cell(const TCell &cell) {
    cells.emplace_back(cell);
  }


  void simulate(double interval) {
    double t_end = t + interval;

    struct IntervalTimer {
      IntervalTimer() {
        prev = std::chrono::steady_clock::now();
      }
      auto operator()() {
        auto diff = std::chrono::steady_clock::now() - prev;
        prev = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(diff).count();
      }
      std::chrono::steady_clock::time_point prev;
    };
    IntervalTimer interval_timer;
    auto start_timer = [start = std::chrono::steady_clock::now()]() {
      return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
    };

    std::cout.precision(3);
    std::cout << "realtime\tsteptime\ttime\tsize\n";
    std::cout << start_timer() << '\t' << interval_timer() << '\t' << t << '\t' << cells.size() << '\n';
    double record_interval = 0.1;
    double next_record = t + record_interval;


    std::vector<double> rates; // No need to keep reallocating

    double estimated_half_wait = 0.0;
    while (t < t_end) {
      std::cout << "begin " << interval_timer() << " \t";
      // Fetch birth, mutation, and death rates for all cells
      rates.resize(cells.size() * 3);
      std::cout << interval_timer() << '\t';

      double event_rate = 0.0; // Summing during calculation is more effective (register usage?)
      for (size_t i = 0; i < cells.size(); ++i) {
        // Store rates in order (birth without mutation, birth with mutation, death)
        // With birth interaction overflowing into death rate if it is bigger than the birth rate
        rates[3*i]      = cells[i].get_birth_rate(t + estimated_half_wait)
                        - (cells.size() - 1) * cells[i].get_birth_interaction();
        rates[3*i + 2]  = -std::min(rates[3*i], 0.0);
        rates[3*i]      = std::max(rates[3*i], 0.0);
        event_rate += rates[3*i];
        rates[3*i + 1]  = rates[3*i] * cells[i].get_discrete_mutation_rate();
        rates[3*i]     -= rates[3*i + 1];
        rates[3*i + 2] += cells[i].get_death_rate()
                        + (cells.size() - 1) * cells[i].get_death_interaction();
        event_rate += rates[3*i + 2];
      }
      std::cout << interval_timer() << '\t';

      // Take timestep dependent on rate of all events
      // double event_rate = std::reduce(rates.begin(), rates.end(), 0.0);
      // double event_rate = std::accumulate(rates.begin(), rates.end(), 0.0);
      std::cout << interval_timer() << '\t';
      double dt = std::exponential_distribution<double>(event_rate)(rng);
      t += dt;
      estimated_half_wait = 0.5 / event_rate;
      std::cout << interval_timer() << '\t';

      // Select an event to perform based on their rates
      size_t event = choose_event(rates, event_rate);
      std::cout << interval_timer() << '\t';
      size_t event_type = event % 3;
      size_t event_cell = event / 3;
      switch (event_type) {
      case 0: // birth without mutation
        cells.emplace_back(cells[event_cell]);
        cells[event_cell].mutate_continuous(rng);
        cells.back().mutate_continuous(rng);
        break;
      case 1: // birth with mutation
        cells.emplace_back(cells[event_cell]);
        cells[event_cell].mutate_continuous(rng);
        cells.back().mutate_discrete(rng);
        cells.back().mutate_continuous(rng);
        break;
      case 2: //death
        std::swap(cells[event_cell], cells.back());
        cells.pop_back();
        break;
      }
      std::cout << interval_timer() << std::endl;

      if (t > next_record) {
        std::cout << start_timer() << '\t' << interval_timer() << '\t' << t << '\t' << cells.size() << std::endl;
        do {
          next_record += record_interval;
        } while (next_record < t);
      }
    }
  }


private:
  TRng rng;
  std::vector<TCell> cells;

  double t = 0;

};


#endif
