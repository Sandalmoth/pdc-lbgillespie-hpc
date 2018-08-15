#include "lb-gillespie.h"
#include "cell-twofactor.h"


int main() {
  LB<Cell> lb;
  for (int i = 0; i < 10; ++i) {
    lb.add_cell(Cell());
  }
  lb.simulate(20);
}
