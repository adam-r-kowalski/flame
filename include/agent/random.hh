#pragma once

#include <torch/torch.h>
#include <tuple>

namespace flame {
namespace agent {
inline namespace v0 {

struct Random {
  using State = torch::Tensor;
  using Action = int;
  using Reward = float;
  using Experience = std::tuple<State, Action, Reward, State>;

  Random(int actions);
  auto operator()(const State &state) -> Action;
  auto remember(Experience experience) -> void;

private:
  int actions_;
};

} // namespace v0
} // namespace agent
} // namespace flame
