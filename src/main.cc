#include <flame.hh>

auto main() -> int {
  const auto interpreter = std::make_shared<pybind11::scoped_interpreter>();

  pybind11::module::import("sys").attr("argv").attr("append")("");

  const auto gym = flame::gym::Gym{interpreter};
  const auto tensorboard = flame::Tensorboard{interpreter};

  auto environment = gym.make("CartPole-v0");

  const auto observation_space =
      environment.observation_space().shape[0].item<int>();

  const auto action_space = environment.action_space().n;

  auto agent =
      flame::agent::PolicyGradient{observation_space, 20, action_space, 0.9};

  flame::Simulation{environment, agent}.render(true).run();

  flame::Simulation{environment, agent}
      .episodes(300)
      .on_episode_end({flame::callback::console_logger})
      .run();

  flame::Simulation{environment, agent}.render(true).run();

  return 0;
}
