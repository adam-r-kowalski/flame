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

  explicit PolicyGradient(int observation_space, int hidden_units,
                          int action_space, float gamma);

  auto operator()(const State &state) -> Action;
  auto remember(Experience experience) -> void;
  auto on_episode_end(int episode, Reward reward) -> void;

private:
  auto discount_(const torch::Tensor &rewards) const -> torch::Tensor;

  torch::nn::Sequential model_;
  torch::optim::Adam optimizer_;
  std::vector<torch::Tensor> log_probabilities_;
  std::vector<Reward> rewards_;
  float gamma_;
};

} // namespace v0
} // namespace agent
} // namespace flame
