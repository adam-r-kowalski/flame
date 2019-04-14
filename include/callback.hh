#pragma once

#include <functional>
#include <iostream>
#include <tuple>

#include <tensorboard.hh>

namespace flame {
namespace callback {
inline namespace v0 {

using Callback = std::function<void(int, float)>;

auto empty(int episode, float reward) -> void;

auto console_logger(int episode, float reward) -> void;

struct TensorboardLogger {
  explicit TensorboardLogger(const Tensorboard &tensorboard);

  auto operator()(int episode, float reward) -> void;

private:
  SummaryWriter summary_writer_;
};

} // namespace v0
} // namespace callback
} // namespace flame
