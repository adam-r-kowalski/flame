#pragma once

#include <torch/torch.h>
#include <tuple>

namespace flame {
namespace gym {
namespace wrappers {
inline namespace v0 {

struct Box {
  const torch::Tensor shape;
};

template <class Environment> struct StackHistory {
  using State = typename Environment::State;
  using Action = typename Environment::Action;
  using Reward = typename Environment::Reward;
  using ActionSpace = typename Environment::ActionSpace;
  using ObservationSpace = Box;

  explicit StackHistory(Environment &&environment, int states);

  auto reset() -> State;
  auto step(const Action &action) -> std::tuple<State, Reward, bool>;
  auto action_space() const -> ActionSpace;
  auto observation_space() const -> ObservationSpace;
  auto render() -> void;

private:
  Environment environment_;
  int states_;
  torch::Tensor state_;
};

// IMPLEMENTATION

template <class Environment>
StackHistory<Environment>::StackHistory(Environment &&environment, int states)
    : environment_{std::move(environment)}, states_{states} {}

template <class Environment> auto StackHistory<Environment>::reset() -> State {
  const auto state = environment_.reset();
  auto states = std::vector<State>{};
  for (auto i = 0; i < states_; ++i)
    states.push_back(state);
  state_ = torch::stack(states).view({1, 1, states_, state.size(0)});
  return state_;
}

template <class Environment>
auto StackHistory<Environment>::step(const Action &action)
    -> std::tuple<State, Reward, bool> {
  const auto &[state, reward, done] = environment_.step(action);
  state_.narrow(-2, 0, states_ - 1) = state_.narrow(-2, 1, states_ - 1);
  state_.narrow(-2, states_ - 1, 1) = state;
  return std::make_tuple(state_, reward, done);
}

template <class Environment>
auto StackHistory<Environment>::action_space() const -> ActionSpace {
  return environment_.action_space();
}

template <class Environment>
auto StackHistory<Environment>::observation_space() const -> ObservationSpace {
  return Box{torch::tensor(
      {1, 1, environment_.observation_space().shape.template item<int>(),
       states_})};
}

template <class Environment> auto StackHistory<Environment>::render() -> void {
  environment_.render();
}

} // namespace v0
} // namespace wrappers
} // namespace gym
} // namespace flame
