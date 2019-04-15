#pragma once

#include <vector>

#include <callback.hh>
#include <gym/gym.hh>
#include <unit.hh>

namespace flame {
inline namespace v0 {

using State = gym::Environment::State;
using Action = gym::Environment::Action;
using Reward = gym::Environment::Reward;

struct Simulation {
  explicit Simulation();
  auto episodes(int episodes) -> Simulation &;
  auto render(bool render) -> Simulation &;
  auto on_episode_end(std::vector<callback::Callback> &&on_episode_end)
      -> Simulation &;

  template <class Agent>
  auto run(gym::Environment &environemnt, Agent &agent) -> void;

private:
  int episodes_;
  bool render_;
  std::vector<callback::Callback> on_episode_end_;
};

// IMPLEMENTATION

template <class Agent>
auto Simulation::run(gym::Environment &environment, Agent &agent) -> void {
  for (auto episode = 0; episode < episodes_; ++episode) {
    auto done = false;
    auto state = environment.reset();
    auto episode_reward = zero<Reward>();
    while (!done) {
      const auto action = agent(state);
      const auto [next_state, reward, is_done] = environment.step(action);

      if (render_)
        environment.render();

      episode_reward += reward;
      done = is_done;
      agent.remember(
          std::make_tuple(std::move(state), action, reward, next_state));
      state = std::move(next_state);
    }
    agent.on_episode_end(episode, episode_reward);

    for (auto &callback : on_episode_end_)
      callback(episode, episode_reward);
  }
}

} // namespace v0
} // namespace flame
