#include <flame.hh>

auto main() -> int {
  const auto interpreter = flame::python_interpreter();
  const auto gym = flame::gym::Gym{interpreter};
  const auto tensorboard = flame::Tensorboard{interpreter};

  auto environment = gym.make("CartPole-v0");

  auto agent = flame::agent::PolicyGradient{
      {.observation_space =
           environment.observation_space().shape[0].item<int>(),
       .hidden_units = 20,
       .action_space = environment.action_space().n,
       .gamma = 0.9}};

  flame::Simulation{}.render(true).run(environment, agent);

  flame::Simulation{}
      .episodes(100)
      .on_episode_end({flame::callback::console_logger})
      .run(environment, agent);

  flame::Simulation{}.render(true).run(environment, agent);

  return 0;
}
