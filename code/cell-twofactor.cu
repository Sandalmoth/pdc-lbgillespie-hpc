#include "cell-twofactor.cuh"


__global__ void get_rates(Cell *cells, float *rates, size_t n, float t_est) {
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i > n)
    return;
  int d = cells[i].get_discrete_type();
  float c = cells[i].get_continuous_type();
  printf("%i %i %f\n", i, d, c);
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
  printf("%i %f %f %f\n", i, rates[3*i], rates[3*i + 1], rates[3*i + 2]);
}
