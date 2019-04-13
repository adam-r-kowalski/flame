#pragma once

#include <pybind11/embed.h>
#include <string_view>

namespace flame {
inline namespace v0 {

namespace py = pybind11;

struct SummaryWriter;

struct Tensorboard {
  using Interpreter = std::shared_ptr<py::scoped_interpreter>;

  explicit Tensorboard(Interpreter interpreter);
  auto summary_writer() const -> SummaryWriter;

private:
  Interpreter interpreter_;
  pybind11::object tensorboard_;
};

struct SummaryWriter {
  ~SummaryWriter();

  template <class T>
  auto add_scalar(const std::string_view &name, T value, int iteration) -> void;

private:
  explicit SummaryWriter(py::object &&summary_writer);
  py::object summary_writer_;
  friend struct Tensorboard;
};

// IMPLEMENTATION

template <class T>
auto SummaryWriter::add_scalar(const std::string_view &name, T value,
                               int iteration) -> void {
  summary_writer_.attr("add_scalar")(name, value, iteration);
}

} // namespace v0
} // namespace flame
