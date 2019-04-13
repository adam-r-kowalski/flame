#pragma once

#include <agent/policy_gradient.hh>
#include <agent/random.hh>
#include <callback.hh>
#include <convert.hh>
#include <gym.hh>
#include <tensorboard.hh>
#include <unit.hh>

namespace flame {

using State = gym::Environment::State;
using Action = gym::Environment::Action;
using Reward = gym::Environment::Reward;

template <class Agent>
auto run_simulation(gym::Environment &environment, Agent &agent,
                    int episodes = 1,
                    callback::Callback on_episode_end = callback::empty)
    -> void;

// IMPLEMENTATION

template <class Agent>
auto run_simulation(gym::Environment &environment, Agent &agent, int episodes,
                    callback::Callback on_episode_end) -> void {
  for (auto episode = 0; episode < episodes; ++episode) {
    auto done = false;
    auto state = environment.reset();
    auto episode_reward = zero<Reward>();
    while (!done) {
      const auto action = agent(state);
      const auto [next_state, reward, is_done] = environment.step(action);
      episode_reward += reward;
      done = is_done;
      agent.remember(
          std::make_tuple(std::move(state), action, reward, next_state));
      state = std::move(next_state);
    }
    agent.on_episode_end(episode, episode_reward);
    on_episode_end(episode, episode_reward);
  }
}

} // namespace flame
