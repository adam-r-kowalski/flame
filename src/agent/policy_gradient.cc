#include <agent/policy_gradient.hh>

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

PolicyGradient::PolicyGradient(int observation_space, int action_space)
    : model_{torch::nn::Sequential(torch::nn::Linear(observation_space, 20),
                                   torch::nn::Functional(torch::relu),
                                   torch::nn::Linear(20, action_space),
                                   Softmax(0))} {}

auto PolicyGradient::operator()(const State &state) -> Action {
  const auto probabilities = model_->forward(state.to(torch::kFloat32));
  const auto action = torch::multinomial(probabilities, 1).item<Action>();
  log_probabilities_.push_back(probabilities[action].log());
  return action;
}

auto PolicyGradient::remember(Experience experience) -> void {
  rewards_.push_back(std::get<2>(experience));
}

auto PolicyGradient::on_episode_end(int episode, Reward reward) -> void {
  log_probabilities_.clear();
  rewards_.clear();
}

} // namespace v0
} // namespace agent
} // namespace flame
