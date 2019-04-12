#pragma once

#include <agent/random.hh>
#include <convert.hh>
#include <gym.hh>
#include <unit.hh>

namespace flame {

using State = gym::Environment::State;
using Action = gym::Environment::Action;
using Reward = gym::Environment::Reward;

template <class Agent>
auto run_simulation(gym::Environment &environment, Agent &agent,
                    int episodes = 1) -> Reward;

// IMPLEMENTATION

template <class Agent>
auto run_simulation(gym::Environment &environment, Agent &agent, int episodes)
    -> Reward {
  auto total_reward = zero<Reward>();
  for (int episode = 0; episode < episodes; ++episode) {
    auto done = false;
    auto state = environment.reset();
    while (!done) {
      const auto action = agent(state);
      const auto [next_state, reward, is_done] = environment.step(action);
      total_reward += reward;
      done = is_done;
      agent.remember(
          std::make_tuple(std::move(state), action, reward, next_state));
      state = std::move(next_state);
    }
  }
  return total_reward / episodes;
}

} // namespace flame
