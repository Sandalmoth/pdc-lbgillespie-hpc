#include "lb-gillespie.cuh"
#include "cell-twofactor.cuh"


int main() {
  std::cout << sizeof(Cell) << std::endl;
  LB<Cell> lb;
  for (int i = 0; i < 100; ++i) {
    lb.add_cell(Cell());
  }
  lb.simulate(0.0000001);
}
