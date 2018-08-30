#ifndef __LB_GILLESPIE_H__
#define __LB_GILLESPIE_H__


#include <chrono>
#include <iostream>
#include <numeric>
#include <random>

#include "timers.h"
#include "xoshiro256ss.h"


template <typename TCell, typename TRng=vigna::xoshiro256ss>
class LB {
private:
  size_t choose_event(const std::vector<float> &rates, double sum) {
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
  LB(int id)
    : id(id) {
    std::random_device rd;
    rng.seed(rd());
  }


  template <typename T>
  void seed(T seed) {
    rng.seed(seed);
  }


  void add_cell(const TCell &cell) {
    cells.emplace_back(cell);
  }


  void simulate(double interval) {
    double t_end = t + interval;

    IntervalTimer interval_timer;
    StartTimer start_timer;

    printf("%i\t%lld\t%lld\t%f\t%lu\n", id, start_timer(), interval_timer(), t, cells.size());
    double record_interval = 0.1;
    double next_record = t + record_interval;


    std::vector<float> rates; // No need to keep reallocating

    double estimated_half_wait = 0.0;
    double event_rate = 0.0;
    while (t < t_end) {
      // Fetch birth, mutation, and death rates for all cells
      rates.resize(cells.size() * 3);
      event_rate = 0.0;
      for (size_t i = 0; i < cells.size(); ++i) {
        // Store rates in order (birth without mutation, birth with mutation, death)
        // With birth interaction overflowing into death rate if it is bigger than the birth rate
        rates[3*i]      = cells[i].get_birth_rate(static_cast<float>(t + estimated_half_wait))
                        - (cells.size() - 1) * cells[i].get_birth_interaction();
        rates[3*i + 2]  = -std::min(rates[3*i], 0.0f);
        rates[3*i]      = std::max(rates[3*i], 0.0f);
        rates[3*i + 1]  = rates[3*i] * cells[i].get_discrete_mutation_rate();
        rates[3*i + 2] += cells[i].get_death_rate()
                        + (cells.size() - 1) * cells[i].get_death_interaction();
        // Calculating the sum in here reduces the number of additions
        // additionally, I think it might allocate event_rate on a register and never move it
        // because the speedup is like ~30%
        // It does seem to give a somewhat different answer though, but such is the mysteries of the floating point
        event_rate += rates[3*i] + rates[3*i + 2];
        rates[3*i]     -= rates[3*i + 1];
      }

      // Take timestep dependent on rate of all events
      double dt = std::exponential_distribution<double>(event_rate)(rng);
      t += dt;
      estimated_half_wait = 0.5 / event_rate;

      // Select an event to perform based on their rates
      size_t event = choose_event(rates, event_rate);
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
        if (cells.size() == 0) {
          // Dead population, exit gracefully
          printf("%i\t%lld\t%lld\t%f\t%lu\n", id, start_timer(), interval_timer(), t, cells.size());
          return;
        }
        break;
      }

      if (t > next_record) {
        printf("%i\t%lld\t%lld\t%f\t%lu\n", id, start_timer(), interval_timer(), t, cells.size());
        do {
          next_record += record_interval;
        } while (next_record < t);
      }
    }
  }


private:
  int id;

  TRng rng;
  std::vector<TCell> cells;

  double t = 0;

};


#endif
