#include "lb-gillespie.h"


#define MAX_RATES  15000000
#define MAX_CELLS   5000000
#define BLOCK_SIZE      256


void LB::simulate(double interval) {
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

  dim3 grid((MAX_RATES + BLOCK_SIZE - 1)/BLOCK_SIZE);
  dim3 block(BLOCK_SIZE);

  std::vector<float> rates; // No need to keep reallocating
  float *d_rates = nullptr;
  float *d_cells = nullptr;
  rates.resize(MAX_RATES);
  cudaMalloc(&d_rates, sizeof(float) * MAX_RATES);
  cudaMalloc(&d_cells, sizeof(TCell) * MAX_CELLS);

  double estimated_half_wait = 0.0;
  double event_rate = 0.0; // Summing during calculation is more effective (register usage?)
  while (t < t_end) {
    std::cout << "begin " << interval_timer() << " \t";
    // Fetch birth, mutation, and death rates for all cells
    // Store rates in order (birth without mutation, birth with mutation, death)
    // With birth interaction overflowing into death rate if it is bigger than the birth rate

    if (true) {

      std::cout << "c\t";

      // large population; copy to gpu and calculate rates in parallel
      cudaMemcpy(d_cells, cells.data(), cells.size() * ARRAY_SIZE, cudaMemcpyHostToDevice);
      get_rates<<<grid, block>>>(d_cells, d_rates, cells.size());
      cudaMemcpy(rates.data(), d_rates, cells.size() * 3 * ARRAY_SIZE, cudaMemcpyDeviceToHost);

    } else {

      std::cout << "s\t";

      // small population; run sequentiailly
      for (size_t i = 0; i < cells.size(); ++i) {
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

    }
    
    std::cout << interval_timer() << '\t';

    // Take timestep dependent on rate of all events
    // event_rate = std::reduce(rates.begin(), rates.end(), 0.0); // not commonly supported
    event_rate = std::accumulate(rates.begin(), rates.begin() + cells.size()*3, 0.0);
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

  cudaFree(d_rates);
  cudaFree(d_cells);
}