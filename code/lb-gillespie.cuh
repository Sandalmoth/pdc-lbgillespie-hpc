#ifndef __LB_GILLESPIE_H__
#define __LB_GILLESPIE_H__


#include <array>
#include <iostream>
#include <numeric>
#include <random>

#include "cub/cub.cuh"

#include "timers.h"
#include "xoshiro256ss.h"


#define MAX_RATES   6000000
#define MAX_CELLS   2000000
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


// Calculate the sum of SUM_BLOCK_SIZE consecutive elements from rates and store them in sum
__global__ void partial_sum(float *rates, float *sum, size_t n) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;

  typedef cub::BlockReduce<float, SUM_BLOCK_SIZE> BR;
  __shared__ typename  BR::TempStorage temp;
  float this_rate = 0;
  if (i < n)
    this_rate = rates[i];
  float result = BR(temp).Sum(this_rate, n);
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
  size_t choose_event(size_t n_rates,
                      double sum) {
    // Because the std::discrete_distribution is very slow
    // (probably because it copies the weight vector)
    // I was forced to make my own implementation.
    // Having vectors with partial sums speeds up the process (to log(N) i think)
    // since larger steps can be taken at first
    double k = std::uniform_real_distribution<double>(0, sum)(rng);
    double cumsum = 0.0;
    int j = 0;
    for (int i = SUM_DEPTH - 1; i >= 0; --i) {
      for (size_t u = j; u < n_sums[i]; ++u) {
        cumsum += rate_sums[i][u];
        if (cumsum > k) {
          cumsum -= rate_sums[i][u];
          break;
        }
        ++j;
      }
      j *= SUM_BLOCK_SIZE;
      // Fetch the section of rates or partial sums where the event occurs
      if (i > 0) {
        cudaMemcpyAsync(rate_sums[i - 1] + j,
                        d_rate_sums[i - 1] + j,
                        SUM_BLOCK_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
      } else {
        cudaMemcpyAsync(rates + j,
                        d_rates + j,
                        SUM_BLOCK_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
      }
      cudaStreamSynchronize(stream);
    }

    // Find the exact position in the rates vector
    for (size_t i = j; i < n_rates; ++i) {
      cumsum += rates[i];
      if (cumsum > k)
        return i;
    }
    return n_rates - 1;
  }


public:
  LB(int id)
    : id(id) {
    std::random_device rd;
    rng.seed(rd());
    cudaMallocHost(&cells, MAX_CELLS * sizeof(TCell));
  }

  ~LB() {
    cudaFreeHost(cells);
  } // TODO Rule of three


  template <typename T>
  void seed(T seed) {
    rng.seed(seed);
  }


  void add_cell(const TCell &cell) {
    cells[n_cells++] = cell;
  }


  void simulate(double interval) {
    double t_end = t + interval;

    // Define timer constructs.
    // IntervalTimer operator() gives time since last call
    // StartTimer operator() gives time since construction
    IntervalTimer interval_timer;
    StartTimer start_timer;

    // Print output header
    printf("%i\t%lld\t%lld\t%f\t%lu\n", id, start_timer(), interval_timer(), t, n_cells);

    // Setup printing parameters
    double record_interval = 0.1;
    double next_record = t + record_interval;

    // Define cuda grids
    dim3 rate_grid(divup(MAX_RATES, BLOCK_SIZE));
    dim3 rate_block(BLOCK_SIZE);
    dim3 sum_grid(divup(MAX_RATES, SUM_BLOCK_SIZE));
    dim3 sum_block(SUM_BLOCK_SIZE);

    // Create stream (allowing LB objects to simulate better in parallel
    cudaStreamCreate(&stream);

    // Allocate device memory and corresponding host memory
    cudaMalloc(&d_rates, sizeof(float) * MAX_RATES);
    cudaMalloc(&d_cells, sizeof(TCell) * MAX_CELLS);
    cudaMallocHost(&rates, sizeof(float) * MAX_RATES);
    int sum_blocks = MAX_RATES;
    for (int i = 0; i < SUM_DEPTH; ++i) {
      sum_blocks = divup(sum_blocks, SUM_BLOCK_SIZE);
      cudaMalloc(&d_rate_sums[i], sizeof(float) * sum_blocks);
      cudaMallocHost(&rate_sums[i], sizeof(float) * sum_blocks);
    }

    // avoid reallocation and allow flexibility by scoping here
    double estimated_half_wait = 0.0;
    double event_rate = 0.0;

    // Copy initial state of cells to device
    cudaMemcpyAsync(d_cells, cells, n_cells * sizeof(TCell), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // run simulation for given interval
    while (t < t_end) {
      // Fetch birth, mutation, and death rates for all cells
      // Store rates in order (birth without mutation, birth with mutation, death)
      // With birth interaction overflowing into death rate if it is bigger than the birth rate

      rate_grid.x = divup(n_cells, BLOCK_SIZE);

      // large population; copy to gpu and calculate rates in parallel
      get_rates<<<rate_grid, rate_block, 0, stream>>>(d_cells, d_rates, n_cells, static_cast<float>(t + estimated_half_wait));

      // get partial sums of the rates
      // iteratively collects sets of them by summing either the rates vector or partial_sum vectors
      int n_rates_prev = n_cells * 3;
      sum_grid.x = divup(n_cells * 3, SUM_BLOCK_SIZE);
      n_sums[0] = sum_grid.x;
      partial_sum<<<sum_grid, sum_block, 0, stream>>>(d_rates, d_rate_sums[0], n_rates_prev);
      for (int i = 1; i < SUM_DEPTH; ++i) {
        n_rates_prev = sum_grid.x;
        sum_grid.x = divup(sum_grid.x, SUM_BLOCK_SIZE);
        n_sums[i] = sum_grid.x;
        partial_sum<<<sum_grid, sum_block, 0, stream>>>(d_rate_sums[i-1], d_rate_sums[i], n_rates_prev);
      }

      // we need the deepest sum to find the total event rate, and for the first event selection step
      cudaMemcpyAsync(rate_sums[SUM_DEPTH - 1],
                      d_rate_sums[SUM_DEPTH - 1],
                      n_sums[SUM_DEPTH - 1] * sizeof(float),
                      cudaMemcpyDeviceToHost, stream);

      cudaStreamSynchronize(stream);

      // Take timestep dependent on rate of all events
      // event_rate = std::reduce(rates.begin(), rates.end(), 0.0); // not commonly supported
      event_rate = std::accumulate(rate_sums.back(),
                                   rate_sums.back() + sum_grid.x,
                                   0.0);
      double dt = std::exponential_distribution<double>(event_rate)(rng);
      t += dt;
      estimated_half_wait = 0.5 / event_rate; // used for calculating time-dependence of rates in the next iteration

      // Select an event to perform based on their rates
      // remember the layout: [birth without mutation, birth with mutation, death]
      // and ofcourse, the continuous variable always mutates
      size_t event = choose_event(n_cells * 3, event_rate);
      size_t event_type = event % 3;
      size_t event_cell = event / 3;
      switch (event_type) {
      case 0: // birth without mutation
        cells[n_cells++] = cells[event_cell];
        cells[event_cell].mutate_continuous(rng);
        cells[n_cells-1].mutate_continuous(rng);
        cudaMemcpyAsync(d_cells + event_cell, cells + event_cell, sizeof(TCell), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_cells + n_cells - 1, cells + n_cells - 1, sizeof(TCell), cudaMemcpyHostToDevice, stream);
        break;
      case 1: // birth with mutation
        cells[n_cells++] = cells[event_cell];
        cells[event_cell].mutate_continuous(rng);
        cells[n_cells-1].mutate_discrete(rng);
        cells[n_cells-1].mutate_continuous(rng);
        cudaMemcpyAsync(d_cells + event_cell, cells + event_cell, sizeof(TCell), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_cells + n_cells - 1, cells + n_cells - 1, sizeof(TCell), cudaMemcpyHostToDevice, stream);
        break;
      case 2: //death
        // swap to end and decrement size to effectively destroy cell with minimal movement
        std::swap(cells[event_cell], cells[n_cells-1]);
        cudaMemcpyAsync(d_cells + event_cell, cells + event_cell, sizeof(TCell), cudaMemcpyHostToDevice, stream);
        --n_cells;
        if (n_cells == 0) {
          // Population died out, exit gracefully
          // (causes horrible pointer errors when it happens if not checked for)
          printf("%i\t%lld\t%lld\t%f\t%lu\n", id, start_timer(), interval_timer(), t, n_cells);
          cudaFree(d_rates);
          cudaFree(d_cells);
          cudaFreeHost(rates);
          for (int i = 0; i < SUM_DEPTH; ++i) {
            cudaFree(d_rate_sums[i]);
            cudaFreeHost(rate_sums[i]);
          }
          cudaStreamDestroy(stream);
          return;
        }
        break;
      }

      // Print output
      if (t > next_record) {
          printf("%i\t%lld\t%lld\t%f\t%lu\n", id, start_timer(), interval_timer(), t, n_cells);
        do {
          next_record += record_interval;
        } while (next_record < t);
      }
    }

    // Cleanup of device and pinned host memory
    cudaFree(d_rates);
    cudaFree(d_cells);
    cudaFreeHost(rates);
    for (int i = 0; i < SUM_DEPTH; ++i) {
      cudaFree(d_rate_sums[i]);
      cudaFreeHost(rate_sums[i]);
    }

    cudaStreamDestroy(stream);
  }


private:
  int id;

  TRng rng;

  TCell *cells = nullptr;
  float *rates = nullptr;
  std::array<float *, SUM_DEPTH> rate_sums;
  std::array<float *, SUM_DEPTH> d_rate_sums;
  std::array<size_t, SUM_DEPTH> n_sums;
  float *d_rates = nullptr;
  TCell *d_cells = nullptr;

  size_t n_cells = 0;

  double t = 0;
  cudaStream_t stream;

};


#endif
