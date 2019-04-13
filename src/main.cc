#include <flame.hh>

using namespace flame;

auto main() -> int {
  const auto interpreter = std::make_shared<pybind11::scoped_interpreter>();
  const auto gym = gym::Gym{interpreter};

  auto environment = gym.make("CartPole-v0");

  auto observation_space = environment.observation_space().shape[0].item<int>();
  auto action_space = environment.action_space().n;

  auto agent = agent::PolicyGradient{observation_space, action_space};

  std::cout << run_simulation(environment, agent) << "\n";

  return 0;
}
