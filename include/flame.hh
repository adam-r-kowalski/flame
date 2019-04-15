#pragma once

#include <agent/policy_gradient.hh>
#include <agent/random.hh>
#include <callback.hh>
#include <convert.hh>
#include <gym/gym.hh>
#include <gym/wrappers/stack_history.hh>
#include <simulation.hh>
#include <tensorboard.hh>
#include <unit.hh>

namespace flame {
inline namespace v0 {

namespace py = pybind11;

auto python_interpreter() -> std::shared_ptr<py::scoped_interpreter>;

} // namespace v0
} // namespace flame
