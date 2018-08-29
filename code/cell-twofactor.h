#ifndef __CELL_TWOFACTOR_H__
#define __CELL_TWOFACTOR_H__


#include <cstdlib>
#include <random>


class Cell {
public:
  Cell() { }

  Cell(unsigned int discrete, float continuous)
    : discrete(discrete)
    , continuous(continuous) { }


  static size_t get_discrete_options() {
    return 2;
  }


  size_t get_discrete_type() {
    return discrete;
  }
  float get_continuous_type() {
    return continuous;
  }


  // Note possibility of time dependence (float t)

  float get_birth_rate(float t) {
    return 0.5 + discrete*2 + continuous;
  }

  float get_death_rate() {
    return 0.1;
  }

  float get_birth_interaction() {
    return 0.00005;
  }

  float get_death_interaction() {
    return 0.00005;
  }

  float get_discrete_mutation_rate() {
    return 0.000001 * std::max(continuous, 0.0f);
  }


  // Mutate this cell
  template <typename T>
  void mutate_discrete(T rng) {
    ++discrete;
    discrete %= 2;
  }

  template <typename T>
  void mutate_continuous(T rng) {
    continuous += std::normal_distribution<float>(0.0, 0.1)(rng);
  }



private:
  unsigned int discrete = 0;
  float continuous = 1.0;

};


#endif
