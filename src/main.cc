#include <iostream>
#include <torch/torch.h>

auto main() -> int {
  std::cout << torch::rand({3, 2});

  return 0;
}
