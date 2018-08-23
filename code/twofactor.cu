#include "lb-gillespie.cuh"
#include "cell-twofactor.cuh"


int main() {
  LB<Cell> lb;
  for (int i = 0; i < 3000000; ++i) {
    lb.add_cell(Cell());
  }
  lb.simulate(20);
}
