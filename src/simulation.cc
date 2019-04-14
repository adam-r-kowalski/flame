#include <simulation.hh>

namespace flame {
inline namespace v0 {

Simulation::Simulation() : episodes_{1}, render_{false} {}

auto Simulation::episodes(int episodes) -> Simulation & {
  episodes_ = episodes;
  return *this;
}

auto Simulation::render(bool render) -> Simulation & {
  render_ = render;
  return *this;
}

auto Simulation::on_episode_end(
    std::vector<callback::Callback> &&on_episode_end) -> Simulation & {
  on_episode_end_ = std::move(on_episode_end);
  return *this;
}

} // namespace v0
} // namespace flame
