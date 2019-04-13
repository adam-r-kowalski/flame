#include <tensorboard.hh>

namespace flame {
inline namespace v0 {

Tensorboard::Tensorboard(Interpreter interpreter)
    : interpreter_{interpreter}, tensorboard_{
                                     py::module::import("tensorboardX")} {}

auto Tensorboard::summary_writer() const -> SummaryWriter {
  return SummaryWriter{tensorboard_.attr("SummaryWriter")()};
}

SummaryWriter::SummaryWriter(py::object &&summary_writer)
    : summary_writer_{std::move(summary_writer)} {}

SummaryWriter::~SummaryWriter() { summary_writer_.attr("close")(); }

} // namespace v0
} // namespace flame
