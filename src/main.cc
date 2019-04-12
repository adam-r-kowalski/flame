#include <flame.hh>

auto main() -> int {
  const auto interpreter = std::make_shared<pybind11::scoped_interpreter>();
  const auto gym = flame::gym::Gym{interpreter};

  auto environment = gym.make("CartPole-v0");
  auto agent = flame::agent::Random{environment.action_space().n};

  const auto reward = flame::run_simulation(environment, agent);

  std::cout << reward << "\n";

  return 0;
}
