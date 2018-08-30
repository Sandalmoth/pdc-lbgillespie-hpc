#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <random>

#include <omp.h>
#include <tclap/CmdLine.h>

#include "lb-gillespie.cuh"
#include "cell-twofactor.cuh"
#include "timers.h"


using namespace std;


const string VERSION = "0.2.0";


struct Arguments {
  int starting_population;
  double t_end;
  int num_cores;
  int num_simulations;
  unsigned int seed;
};


struct Simdata {
  int index;
  int thread_id;
  long long time;
};


int main(int argc, char **argv) {

  mt19937 seed_rng;

  Arguments a;
  try {
    TCLAP::CmdLine cmd("LB-Process simulator", ' ', VERSION);

    TCLAP::ValueArg<int> a_n0("p", "p0", "Starting cell count", false, 1, "integer", cmd);
    TCLAP::ValueArg<double> a_t("t", "t-max", "Simulation time", false, 1.0, "double", cmd);
    TCLAP::ValueArg<int> a_cores("n", "n-cores", "Number of parallel simulations", false, 1, "integer", cmd);
    TCLAP::ValueArg<int> a_sims("s", "sims", "Number of simulations", false, 1, "integer", cmd);
    TCLAP::ValueArg<unsigned int> a_seed("r", "seed", "Random number seed", false, 0, "unsigned int", cmd);

    cmd.parse(argc, argv);

    a.starting_population = a_n0.getValue();
    a.t_end = a_t.getValue();
    if (a_cores.isSet())
      a.num_cores = a_cores.getValue();
    else
      a.num_cores = omp_get_num_threads();
    a.num_simulations = a_sims.getValue();
    if (a_seed.isSet())
      a.seed = a_seed.getValue();
    else {
      random_device rd;
      a.seed = rd();
    }

    // Sanity checks
    assert(a.starting_population > 0);
    assert(a.t_end > 0);
    assert(a.num_cores > 0);
    assert(a.num_simulations > 0);

  } catch (TCLAP::ArgException &e) {
    cerr << "TCLAP Error: " << e.error() << endl << "\targ: " << e.argId() << endl;
    return 1;
  }

  seed_rng.seed(a.seed);
  vector<unsigned int> seeds;
  for (int j = 0; j < a.num_simulations; ++j)
    seeds.emplace_back(uniform_int_distribution<unsigned int>()(seed_rng));

  StartTimer start_timer;
  IntervalTimer interval_timer;

  vector<Simdata> simdata;
  simdata.resize(a.num_simulations);

  cout << " --- TIMELINES [tsv] --- \n";
  cout << "index\trealtime\tsteptime\ttime\tsize" << endl;

#pragma omp parallel for num_threads(a.num_cores) private(interval_timer)
  for (int j = 0; j < a.num_simulations; ++j) {
    int id = omp_get_thread_num();
    LB<Cell> lb(j);
    for (int i = 0; i < a.starting_population; ++i) {
      lb.add_cell(Cell());
    }
    lb.seed(seeds[j]);
    lb.simulate(a.t_end);
    simdata[j] = Simdata{j, id, interval_timer()};
  }

  cout << endl << " --- SIMDATA [tsv] --- \n";
  cout << "index\tthread_id\ttime\tseed\n";
  for (int j = 0; j < a.num_simulations; ++j)
    cout << simdata[j].index << '\t' << simdata[j].thread_id << '\t' << simdata[j].time << '\t' << seeds[j] << '\n';

  cout << "\n --- FOOTER [toml] --- " << endl;
  cout << "starting_population = " << a.starting_population << '\n'
       << "simulation_time = " << a.t_end << '\n'
       << "num_cores = " << a.num_cores << '\n'
       << "num_simulations = " << a.num_simulations << '\n'
       << "rng_seed = " << a.seed << '\n'
       << "total_time = " << start_timer() << endl;
}
