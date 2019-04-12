#include <flame.hh>

auto main() -> int {
  const auto interpreter = std::make_shared<pybind11::scoped_interpreter>();
  const auto gym = flame::gym::Gym{interpreter};
  auto env = gym.make("CartPole-v0");

  std::cout << "action_space = " << env.action_space().n << "\n";

  auto done = false;
  auto reward = 0.0;
  auto state = env.reset();
  while (!done) {
    const auto [state_, reward_, done_] = env.step(1);
    reward += reward_;
    done = done_;
    state = std::move(state_);
  }

  std::cout << "reward = " << reward << "\n";

  return 0;
}
