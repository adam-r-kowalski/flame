#pragma once

#include <torch/torch.h>
#include <vector>

namespace flame {
namespace agent {
inline namespace v0 {

struct PolicyGradient {
  using State = torch::Tensor;
  using Action = int;
  using Reward = float;
  using Experience = std::tuple<State, Action, Reward, State>;

  explicit PolicyGradient(int observation_space, int action_space);

  auto operator()(const State &state) -> Action;
  auto remember(Experience experience) -> void;
  auto on_episode_end(int episode, Reward reward) -> void;

private:
  torch::nn::Sequential model_;
  std::vector<torch::Tensor> log_probabilities_;
  std::vector<Reward> rewards_;
};

} // namespace v0
} // namespace agent
} // namespace flame
