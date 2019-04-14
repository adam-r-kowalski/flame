#pragma once

#include <vector>

#include <callback.hh>
#include <gym.hh>
#include <unit.hh>

namespace flame {
inline namespace v0 {

using State = gym::Environment::State;
using Action = gym::Environment::Action;
using Reward = gym::Environment::Reward;

template <class Agent> struct Simulation {
  explicit Simulation(gym::Environment &environment, Agent &agent);
  auto episodes(int episodes) -> Simulation &;
  auto render(bool render) -> Simulation &;
  auto on_episode_end(std::vector<callback::Callback> &&on_episode_end)
      -> Simulation &;
  auto run() -> void;

private:
  gym::Environment &environment_;
  Agent &agent_;
  int episodes_ = 1;
  bool render_ = false;
  std::vector<callback::Callback> on_episode_end_;
};

// IMPLEMENTATION

template <class Agent>
Simulation<Agent>::Simulation(gym::Environment &environment, Agent &agent)
    : environment_{environment}, agent_{agent} {}

template <class Agent>
auto Simulation<Agent>::episodes(int episodes) -> Simulation & {
  episodes_ = episodes;
  return *this;
}

template <class Agent>
auto Simulation<Agent>::render(bool render) -> Simulation & {
  render_ = render;
  return *this;
}

template <class Agent>

auto Simulation<Agent>::on_episode_end(
    std::vector<callback::Callback> &&on_episode_end) -> Simulation & {
  on_episode_end_ = std::move(on_episode_end);
  return *this;
}

template <class Agent> auto Simulation<Agent>::run() -> void {
  for (auto episode = 0; episode < episodes_; ++episode) {
    auto done = false;
    auto state = environment_.reset();
    auto episode_reward = zero<Reward>();
    while (!done) {
      const auto action = agent_(state);
      const auto [next_state, reward, is_done] = environment_.step(action);

      if (render_)
        environment_.render();

      episode_reward += reward;
      done = is_done;
      agent_.remember(
          std::make_tuple(std::move(state), action, reward, next_state));
      state = std::move(next_state);
    }
    agent_.on_episode_end(episode, episode_reward);

    for (auto &callback : on_episode_end_)
      callback(episode, episode_reward);
  }
}

} // namespace v0
} // namespace flame
