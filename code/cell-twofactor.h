#ifndef __CELL_TWOFACTOR_H__
#define __CELL_TWOFACTOR_H__


#include <cstdlib>
#include <random>


class Cell {
public:
  Cell() { }

  Cell(size_t discrete, double continuous)
    : discrete(discrete)
    , continuous(continuous) { }


  static size_t get_discrete_options() {
    return 2;
  }


  size_t get_discrete_type() {
    return discrete;
  }
  double get_continuous_type() {
    return continuous;
  }


  // Note possibility of time dependence (double t)
  // and evolutionary game theory (first, last of list of all other cells)

  double get_birth_rate(double t) {
    return 0.5 + discrete*2 + continuous;
  }

  double get_death_rate() {
    return 0.1;
  }

  double get_birth_interaction() {
    return 0.00005;
  }

  double get_death_interaction() {
    return 0.00005;
  }

  double get_discrete_mutation_rate() {
    return 0.000001 * std::max(continuous, 0.0);
  }


  // Mutate this cell
  template <typename T>
  void mutate_discrete(T rng) {
    ++discrete;
    discrete %= 2;
  }

  template <typename T>
  void mutate_continuous(T rng) {
    continuous += std::normal_distribution<double>(0.0, 0.02)(rng);
  }



private:
  int discrete = 0;
  double continuous = 1.0;

};


#endif
