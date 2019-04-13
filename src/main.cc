#include <flame.hh>

using namespace flame;

auto main() -> int {
  const auto interpreter = std::make_shared<pybind11::scoped_interpreter>();
  const auto gym = gym::Gym{interpreter};
  const auto tensorboard = Tensorboard{interpreter};

  auto environment = gym.make("CartPole-v0");

  const auto observation_space =
      environment.observation_space().shape[0].item<int>();

  const auto action_space = environment.action_space().n;

  auto agent = agent::PolicyGradient{observation_space, /*hidden units=*/20,
                                     action_space, /*gamma=*/0.9};

  run_simulation(environment, agent, /*episodes=*/1,
                 callback::TensorboardLogger{tensorboard});

  return 0;
}
