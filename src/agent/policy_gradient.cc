#include <agent/policy_gradient.hh>
#include <unit.hh>

namespace flame {
namespace agent {
inline namespace v0 {

struct Softmax : torch::nn::Module {
  explicit Softmax(int dim) : dim_{dim} {}

  auto forward(const torch::Tensor &x) -> torch::Tensor {
    return torch::softmax(x, dim_);
  }

private:
  int dim_;
};

PolicyGradient::PolicyGradient(PolicyGradientOptions &&options)
    : model_{torch::nn::Sequential(
          torch::nn::Linear(options.observation_space, 20),
          torch::nn::Functional(torch::relu),
          torch::nn::Linear(20, options.action_space), Softmax(/*dim=*/0))},
      optimizer_{torch::optim::Adam{model_->parameters(), /*lr=*/1e-2}},
      gamma_{options.gamma} {}

auto PolicyGradient::operator()(const State &state) -> Action {
  const auto probabilities = model_->forward(state.to(torch::kFloat32));
  const auto action = torch::multinomial(probabilities, 1).item<Action>();
  log_probabilities_.push_back(probabilities[action].log());
  return action;
}

auto PolicyGradient::remember(Experience experience) -> void {
  rewards_.push_back(std::get<2>(experience));
}

auto normalize(const torch::Tensor &values) {
  return (values - values.mean()) / values.std();
}

auto PolicyGradient::on_episode_end(int episode, Reward reward) -> void {
  const auto log_probabilities = torch::stack(std::move(log_probabilities_));
  const auto rewards = torch::tensor(std::move(rewards_)).to(torch::kFloat32);
  const auto returns = normalize(discount_(rewards));

  optimizer_.zero_grad();
  (-log_probabilities * returns).sum().backward();
  optimizer_.step();

  log_probabilities_.clear();
  rewards_.clear();
}

auto PolicyGradient::discount_(const torch::Tensor &rewards) const
    -> torch::Tensor {
  auto discounted = torch::zeros_like(rewards);
  auto running_sum = zero<Reward>();

  for (auto i = rewards.size(0) - 1; i >= 0; --i) {
    running_sum = rewards[i].item<Reward>() + gamma_ * running_sum;
    discounted[i] = running_sum;
  }

  return discounted;
}

} // namespace v0
} // namespace agent
} // namespace flame
