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

template <class... Callback> struct Callbacks {
  explicit Callbacks(Callback... callback);

  auto operator()(int episode, float reward) -> void;

private:
  template <size_t Index> auto apply_(int episode, float reward) -> void;

  std::tuple<Callback...> callbacks_;
};

// IMPLEMENTATION

template <class... Callback>
Callbacks<Callback...>::Callbacks(Callback... callbacks)
    : callbacks_{callbacks...} {}

template <class... Callback>
auto Callbacks<Callback...>::operator()(int episode, float reward) -> void {
  apply_<0>(episode, reward);
}

template <class... Callback>
template <size_t Index>
auto Callbacks<Callback...>::apply_(int episode, float reward) -> void {
  if constexpr (Index < sizeof...(Callback)) {
    std::get<Index>(callbacks_)(episode, reward);
    apply_<Index + 1>(episode, reward);
  }
}

} // namespace v0
} // namespace callback
} // namespace flame
