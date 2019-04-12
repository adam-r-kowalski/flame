#pragma once

#include <pybind11/embed.h>
#include <string_view>
#include <torch/torch.h>

namespace flame {
namespace gym {
inline namespace v0 {

namespace py = pybind11;

struct Environment;

struct Gym {
  using Interpreter = std::shared_ptr<py::scoped_interpreter>;

  explicit Gym(Interpreter interpreter);
  auto make(const std::string_view &name) const -> Environment;

private:
  Interpreter interpreter_;
  py::module gym_;
};

struct Discrete;

struct Environment {
  using State = torch::Tensor;
  using Action = int;
  using Reward = float;

  auto reset() -> State;
  auto step(const Action &action) -> std::tuple<State, Reward, bool>;
  auto action_space() const -> Discrete;

private:
  explicit Environment(py::object &&environment);
  py::object environment_;
  friend struct Gym;
};

struct Discrete {
  int n;

private:
  explicit Discrete(py::object &&env);
  py::object discrete_;
  friend struct Environment;
};

} // namespace v0
} // namespace gym
} // namespace flame
