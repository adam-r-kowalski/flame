#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>

#include <convert.hh>

namespace py = pybind11;

using namespace py::literals;

auto main() -> int {
  const auto interpreter = py::scoped_interpreter{};
  const auto gym = py::module::import("gym");
  const auto env = gym.attr("make")("CartPole-v0");

  const auto state =
      flame::convert<torch::Tensor>(py::array(env.attr("reset")()));

  std::cout << state;

  return 0;
}
