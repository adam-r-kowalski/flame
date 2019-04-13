#include <iostream>

#include <callback.hh>

namespace flame {
namespace callback {
inline namespace v0 {

auto empty(int episode, float reward) -> void {}

auto console_logger(int episode, float reward) -> void {
  std::cout << "episode " << episode << " reward = " << reward << "\n";
}

TensorboardLogger::TensorboardLogger(const Tensorboard &tensorboard)
    : summary_writer_{tensorboard.summary_writer()} {}

auto TensorboardLogger::operator()(int episode, float reward) -> void {
  summary_writer_.add_scalar("reward", reward, episode);
}

} // namespace v0
} // namespace callback
} // namespace flame
