#include <flame.hh>

namespace flame {
inline namespace v0 {

auto python_interpreter() -> std::shared_ptr<py::scoped_interpreter> {
  auto interpreter = std::make_shared<pybind11::scoped_interpreter>();
  py::module::import("sys").attr("argv").attr("append")("");
  return interpreter;
}

} // namespace v0
} // namespace flame
