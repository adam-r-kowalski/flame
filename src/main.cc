#include <flame.hh>

auto main() -> int {
  const auto interpreter = flame::python_interpreter();
  const auto gym = flame::gym::Gym{interpreter};
  const auto tensorboard = flame::Tensorboard{interpreter};

  auto environment =
      flame::gym::wrappers::StackHistory{gym.make("CartPole-v0"), 4};

  /*


  auto agent = flame::agent::PolicyGradient{
      {.observation_space =
           environment.observation_space().shape[0].item<int>(),
       .action_space = environment.action_space().n,
       .gamma = 0.9}};

  flame::Simulation{}.episodes(3).render(true).run(environment, agent);

  for (auto i = 0; i < 3; ++i) {
    flame::Simulation{}
        .episodes(100)
        .on_episode_end({flame::callback::TensorboardLogger{tensorboard}})
        .run(environment, agent);

    flame::Simulation{}.episodes(3).render(true).run(environment, agent);
  }

  */

  return 0;
}
