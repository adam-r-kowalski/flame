#include <convert.hh>
#include <gym.hh>

namespace flame {
namespace gym {
inline namespace v0 {

Gym::Gym(Interpreter interpreter)
    : interpreter_{interpreter}, gym_{py::module::import("gym")} {}

auto Gym::make(const std::string_view &name) const -> Environment {
  return Environment{gym_.attr("make")(name)};
}

Environment::Environment(py::object &&environment)
    : environment_{std::move(environment)} {}

Environment::~Environment() { environment_.attr("close")(); }

auto Environment::reset() -> State {
  return convert<torch::Tensor>(py::array(environment_.attr("reset")()));
}

auto Environment::step(const Action &action)
    -> std::tuple<State, Reward, bool> {
  py::tuple observation = environment_.attr("step")(action);
  return std::make_tuple(convert<torch::Tensor>(py::array(observation[0])),
                         py::cast<Reward>(observation[1]),
                         py::cast<bool>(observation[2]));
}

auto Environment::action_space() const -> Discrete {
  return Discrete{environment_.attr("action_space")};
}

auto Environment::observation_space() const -> Box {
  return Box{environment_.attr("observation_space")};
}

auto Environment::render() -> void { environment_.attr("render")(); }

Discrete::Discrete(py::object &&discrete)
    : n{py::cast<int>(discrete.attr("n"))}, discrete_{std::move(discrete)} {}

Box::Box(py::object &&box)
    : shape{convert<torch::Tensor>(py::array(box.attr("shape")))},
      box_{std::move(box)} {}

} // namespace v0
} // namespace gym
} // namespace flame
