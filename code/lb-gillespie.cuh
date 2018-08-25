#ifndef __LB_GILLESPIE_H__
#define __LB_GILLESPIE_H__


#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>

#include "cub/cub.cuh"

#include "xoshiro256ss.h"


#define MAX_RATES  15000000
#define MAX_CELLS   5000000
#define BLOCK_SIZE      256
#define SUM_BLOCK_SIZE  128
#define SUM_DEPTH         2


template <typename TCell>
__global__ void get_rates(TCell *cells, float *rates, size_t n, float t_est) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= n)
    return;
  // Store rates in order (birth without mutation, birth with mutation, death)
  // With birth interaction overflowing into death rate if it is bigger than the birth rate
  rates[3*i]      = cells[i].get_birth_rate(t_est)
                  - (n - 1) * cells[i].get_birth_interaction();
  rates[3*i + 2]  = -fmin(rates[3*i], 0.0f);
  rates[3*i]      = fmax(rates[3*i], 0.0f);
  rates[3*i + 1]  = rates[3*i] * cells[i].get_discrete_mutation_rate();
  rates[3*i]     -= rates[3*i + 1];
  rates[3*i + 2] += cells[i].get_death_rate()
                  + (n - 1) * cells[i].get_death_interaction();
}


__global__ void partial_sum(float *rates, float *sum, size_t n) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  typedef cub::BlockReduce<float, SUM_BLOCK_SIZE> BR;
  __shared__ typename  BR::TempStorage temp;
  float this_rate = 0;
  if (i < n)
    this_rate = rates[i];
  // printf(" (%i %i %lu %f)", threadIdx.x, i, n, this_rate);
  float result = BR(temp).Sum(this_rate, n);
  // if (threadIdx.x == 0)
    // printf(" [%i %f]", i, result);
  if (threadIdx.x == 0)
    sum[blockIdx.x] = result;
}


template<typename T1, typename T2>
T1 divup(T1 num, T2 den) {
  return (num + den - 1) / den;
}


template <typename TCell, typename TRng=vigna::xoshiro256ss>
class LB {
private:
  size_t choose_event(const std::vector<float> &rates,
                      const std::array<std::vector<float>, SUM_DEPTH> &partial_sums,
                      double sum) {
    // Because the std::discrete_distribution is very slow
    // (probably because it copies the weight vector)
    // I was forced to make my own implementation.
    // Having vectors with partial sums speeds up the process (to log(N) i think)
    // since larger steps can be taken at first
    double k = std::uniform_real_distribution<double>(0, sum)(rng);
    // std::cout << std::endl << k << "->" << sum << std::endl;
    double cumsum = 0.0;
    int j = 0;
    for (int i = SUM_DEPTH - 1; i >= 0; --i) {
      // for (auto x: partial_sums[i]) std::cout << x << ' ';
      // std::cout << std::endl;
      for (size_t u = j; u < partial_sums[i].size(); ++u) {
        cumsum += partial_sums[i][u];
        // std::cout << "  " << u << ' ' << cumsum << std::endl;
        if (cumsum > k) {
          cumsum -= partial_sums[i][u];
          break;
        }
        ++j;
      }
      j *= SUM_BLOCK_SIZE;
      // std::cout << i << ':' << j << std::endl;
    }
    // std::cout << j << ' ' << cumsum << std::endl;
    for (size_t i = j; i < rates.size(); ++i) {
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


  void simulate(float interval) {
    float t_end = t + interval;

    // Define timer constructs.
    // IntervalTimer operator() gives time since last call
    // start_timer operator() gives time since construction
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

    // Print output header
    std::cout.precision(3);
    std::cout << "realtime\tsteptime\ttime\tsize\n";
    std::cout << start_timer() << '\t' << interval_timer() << '\t' << t << '\t' << cells.size() << '\n';

    // Setup printing parameters
    float record_interval = 0.1;
    float next_record = t + record_interval;

    // Define cuda grids
    dim3 rate_grid(divup(MAX_RATES, BLOCK_SIZE));
    dim3 rate_block(BLOCK_SIZE);
    dim3 sum_grid(divup(MAX_RATES, SUM_BLOCK_SIZE));
    dim3 sum_block(SUM_BLOCK_SIZE);

    // Allocate device memory and corresponding host memory
    std::vector<float> rates; // No need to keep reallocating
    std::array<std::vector<float>, SUM_DEPTH> rate_sums;
    std::array<float *, SUM_DEPTH> d_rate_sums;
    float *d_rates = nullptr;
    TCell *d_cells = nullptr;
    rates.resize(MAX_RATES);
    cudaMalloc(&d_rates, sizeof(float) * MAX_RATES);
    cudaMalloc(&d_cells, sizeof(TCell) * MAX_CELLS);
    int sum_blocks = MAX_RATES;
    for (int i = 0; i < SUM_DEPTH; ++i) {
      sum_blocks = divup(sum_blocks, SUM_BLOCK_SIZE);
      rate_sums[i].resize(sum_blocks);
      cudaMalloc(&d_rate_sums[i], sizeof(float) * sum_blocks);
    }

    // avoid reallocation and allow flexibility by scoping here
    double estimated_half_wait = 0.0;
    double event_rate = 0.0;

    // run simulation for given interval
    while (t < t_end) {
      std::cout << "begin " << t << '/' << t_end << '\t' << interval_timer() << " \t";

      // Fetch birth, mutation, and death rates for all cells
      // Store rates in order (birth without mutation, birth with mutation, death)
      // With birth interaction overflowing into death rate if it is bigger than the birth rate

      // select cuda or sequential calculation
      if (true) {

        // std::cout << cells.size();
        std::cout << "c\t";

        rate_grid.x = divup(cells.size(), BLOCK_SIZE);

        // large population; copy to gpu and calculate rates in parallel
        cudaMemcpy(d_cells, cells.data(), cells.size() * sizeof(TCell), cudaMemcpyHostToDevice);
        std::cout << interval_timer() << '\t';
        get_rates<<<rate_grid, rate_block>>>(d_cells, d_rates, cells.size(), t + estimated_half_wait);
        cudaDeviceSynchronize();
        std::cout << interval_timer() << '\t';
        cudaMemcpy(rates.data(), d_rates, cells.size() * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << interval_timer() << '\t';
        // get partial sums of the rates
        int n_rates_prev = cells.size() * 3;
        sum_grid.x = divup(cells.size() * 3, SUM_BLOCK_SIZE);
        rate_sums[0].resize(sum_grid.x);
        // std::cout << std::endl;
        // std::cout << sum_grid.x << ' ' << rate_sums[0].size() << std::endl;
        partial_sum<<<sum_grid, sum_block>>>(d_rates, d_rate_sums[0], n_rates_prev);
        cudaMemcpy(rate_sums[0].data(), d_rate_sums[0], sum_grid.x * sizeof(float), cudaMemcpyDeviceToHost);
        // std::cout << "{ ";
        // for (auto x: rate_sums[0]) std::cout << x << ' ';
        // std::cout << '}' << std::endl;
        // std::cout << '0' << ' ' << std::accumulate(rate_sums[0].begin(),
        //                                            rate_sums[0].begin() + sum_grid.x,
        //                                            0.0) << std::endl;
        for (int i = 1; i < SUM_DEPTH; ++i) {
          n_rates_prev = sum_grid.x;
          sum_grid.x = divup(sum_grid.x, SUM_BLOCK_SIZE);
          rate_sums[i].resize(sum_grid.x);
          // std::cout << sum_grid.x << ' ' << rate_sums[i].size() << std::endl;
          partial_sum<<<sum_grid, sum_block>>>(d_rate_sums[i-1], d_rate_sums[i], n_rates_prev);
          cudaMemcpy(rate_sums[i].data(), d_rate_sums[i], sum_grid.x * sizeof(float), cudaMemcpyDeviceToHost);
          // std::cout << i << ' ' << std::accumulate(rate_sums[i].begin(),
          //                                          rate_sums[i].begin() + sum_grid.x,
          //                                          0.0) << std::endl;
        }

        std::cout << interval_timer() << '\t';
        // std::cout << std::endl;
        // for (size_t i = 0; i < cells.size() * 3; ++i) std::cout << rates[i] << ' ';
        // std::cout << std::endl;

      } else {

        std::cout << "s\t";

        // small population; run sequentiailly
        for (size_t i = 0; i < cells.size(); ++i) {
          rates[3*i]      = cells[i].get_birth_rate(t + estimated_half_wait)
                          - (cells.size() - 1) * cells[i].get_birth_interaction();
          rates[3*i + 2]  = -std::min(rates[3*i], 0.0f);
          rates[3*i]      = std::max(rates[3*i], 0.0f);
          event_rate += rates[3*i];
          rates[3*i + 1]  = rates[3*i] * cells[i].get_discrete_mutation_rate();
          rates[3*i]     -= rates[3*i + 1];
          rates[3*i + 2] += cells[i].get_death_rate()
                          + (cells.size() - 1) * cells[i].get_death_interaction();
          event_rate += rates[3*i + 2];
        }

      }

      std::cout << interval_timer() << '\t';

      // std::cout << "cuda finished" << std::endl;

      // std::cout << "rates_sum" << ' ' << std::accumulate(rates.begin(),
      //                                                    rates.begin() + cells.size()*3,
      //                                                    0.0) << std::endl;
      // for (auto it = rates.begin(); it != rates.begin() + cells.size()*3; ++it) std::cout << (*it) << ' '; std::cout << std::endl;
      // for (int i = 0; i < SUM_DEPTH; ++i) {
      //   for (auto x: rate_sums[i]) std::cout << x << ' '; std::cout << std::endl;
      // }

      // Take timestep dependent on rate of all events
      // event_rate = std::reduce(rates.begin(), rates.end(), 0.0); // not commonly supported
      event_rate = std::accumulate(rate_sums.back().begin(),
                                   rate_sums.back().begin() + sum_grid.x,
                                   0.0);
      std::cout << interval_timer() << '\t';
      // std::cout << "er" << event_rate << std::endl;
      float dt = std::exponential_distribution<float>(event_rate)(rng);
      t += dt;
      estimated_half_wait = 0.5 / event_rate;
      std::cout << interval_timer() << '\t';

      // Select an event to perform based on their rates
      size_t event = choose_event(rates, rate_sums, event_rate);
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

    cudaFree(d_rates);
    cudaFree(d_cells);
    for (int i = 0; i < SUM_DEPTH; ++i) {
      cudaFree(d_rate_sums[i]);
    }
  }


private:
  TRng rng;
  std::vector<TCell> cells;

  float t = 0;

};


#endif
