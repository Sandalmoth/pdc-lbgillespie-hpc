#ifndef __CELL_TWOFACTOR_H__
#define __CELL_TWOFACTOR_H__


#include <cstdlib>
#include <random>


class Cell {
public:
  Cell() {
    // continuous_mutation_distribution = std::normal_distribution<float>(0.0, 0.1);
  }

  Cell(unsigned int discrete, float continuous)
    : discrete(discrete)
    , continuous(continuous) {
    // continuous_mutation_distribution = std::normal_distribution<float>(0.0, 0.1);
  }


  static size_t get_discrete_options() {
    return 2;
  }


  __host__ __device__ unsigned int get_discrete_type() {
    return discrete;
  }
  __host__ __device__ float get_continuous_type() {
    return continuous;
  }


  // Note possibility of time dependence (double t)
  // and evolutionary game theory (first, last of list of all other cells)


  __host__ __device__ float get_birth_rate(float t) {
    return 0.5 + discrete*2 + continuous;
  }

  __host__ __device__ float get_death_rate() {
    return 0.1;
  }

  __host__ __device__ float get_birth_interaction() {
    return 0.0;
  }

  __host__ __device__ float get_death_interaction() {
    return 0.00010;
  }

  __host__ __device__ float get_discrete_mutation_rate() {
    return 0.000001 * fmax(continuous, 0.0f);
  }


  // Mutate this cell
  template <typename TRng>
  void mutate_discrete(TRng rng) {
    ++discrete;
    discrete %= 2;
  }

  template <typename TRng>
  void mutate_continuous(TRng rng) {
    // continuous += continuous_mutation_distribution(rng);
    continuous += std::normal_distribution<float>(0.0, 0.1)(rng);
  }



private:
  unsigned int discrete = 0;
  float continuous = 1.0;

  // std::normal_distribution<float> continuous_mutation_distribution;

};


__global__ void get_rates(Cell *, float *, size_t, float);


#endif
