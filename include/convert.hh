#pragma once

#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <torch/torch.h>

namespace flame {
inline namespace v0 {

namespace py = pybind11;

template <class To, class From> auto convert(From from) -> To;

template <> auto convert(py::dtype dtype) -> torch::ScalarType;

template <> auto convert(py::array numpy) -> torch::Tensor;

template <> auto convert(cv::Mat mat) -> torch::Tensor;

template <> auto convert(torch::Tensor tensor) -> cv::Mat;

} // namespace v0
} // namespace flame
