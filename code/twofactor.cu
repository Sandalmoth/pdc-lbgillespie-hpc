#include <cassert>
#include <iostream>
#include <string>

#include <omp.h>
#include <tclap/CmdLine.h>

#include "lb-gillespie.cuh"
#include "cell-twofactor.cuh"
#include "timers.h"


using namespace std;


const string VERSION = "0.1.0";


struct Arguments {
  int starting_population;
  double t_end;
  int num_cores;
  int num_simulations;
};


int main(int argc, char **argv) {

  Arguments a;
  try {
    TCLAP::CmdLine cmd("LB-Process simulator", ' ', VERSION);

    TCLAP::ValueArg<int> a_n0("p", "p0", "Starting cell count", false, 1, "integer", cmd);
    TCLAP::ValueArg<double> a_t("t", "t-max", "Simulation time", false, 1.0, "double", cmd);
    TCLAP::ValueArg<int> a_cores("n", "n-cores", "Number of parallel simulations", false, 1, "integer", cmd);
    TCLAP::ValueArg<int> a_sims("s", "sims", "Number of simulations", false, 1, "integer", cmd);

    cmd.parse(argc, argv);

    a.starting_population = a_n0.getValue();
    a.t_end = a_t.getValue();
    if (a_cores.isSet())
      a.num_cores = a_cores.getValue();
    else
      a.num_cores = omp_get_num_threads();
    a.num_simulations = a_sims.getValue();

    // Sanity checks
    assert(a.starting_population > 0);
    assert(a.t_end > 0);
    assert(a.num_cores > 0);
    assert(a.num_simulations > 0);

  } catch (TCLAP::ArgException &e) {
    cerr << "TCLAP Error: " << e.error() << endl << "\targ: " << e.argId() << endl;
    return 1;
  }

  cout << a.starting_population << ' ' << a.t_end << ' ' << a.num_cores << ' ' << a.num_simulations << endl;

  StartTimer start_timer;
  IntervalTimer interval_timer;

#pragma omp parallel for num_threads(a.num_cores) private(interval_timer)
  for (int j = 0; j < a.num_simulations; ++j) {
    int id = omp_get_thread_num();
    LB<Cell> lb;
    for (int i = 0; i < a.starting_population; ++i) {
      lb.add_cell(Cell());
    }
    lb.seed(j);
    lb.simulate(a.t_end);
    cout << id << ':' << interval_timer() << endl;
  }

  cout << start_timer() << endl;
}
