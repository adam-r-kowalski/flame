#include <agent/random.hh>

namespace flame {
namespace agent {
inline namespace v0 {

Random::Random(int actions) : actions_(actions) {}

auto Random::operator()(const State &state) -> Action {
  return torch::randint(actions_, 1).item<int>();
}

auto Random::remember(Experience experience) -> void {}

auto Random::on_episode_end(int episode, Reward reward) -> void {}

} // namespace v0
} // namespace agent
} // namespace flame
