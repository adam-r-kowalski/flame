#include <convert.hh>

namespace flame {
inline namespace v0 {

namespace py = pybind11;

template <> auto convert(py::dtype dtype) -> torch::ScalarType {
  if (dtype.equal(py::dtype("int8")))
    return torch::kInt8;

  else if (dtype.equal(py::dtype("int16")))
    return torch::kInt16;

  else if (dtype.equal(py::dtype("int32")))
    return torch::kInt32;

  else if (dtype.equal(py::dtype("int64")))
    return torch::kInt64;

  if (dtype.equal(py::dtype("uint8")))
    return torch::kUInt8;

  else if (dtype.equal(py::dtype("int32")))
    return torch::kInt32;

  else if (dtype.equal(py::dtype("int64")))
    return torch::kInt64;

  else if (dtype.equal(py::dtype("float16")))
    return torch::kFloat16;

  else if (dtype.equal(py::dtype("float32")))
    return torch::kFloat32;

  else if (dtype.equal(py::dtype("float64")))
    return torch::kFloat64;

  else if (dtype.equal(py::dtype("complex64")))
    return torch::kComplexFloat;

  else if (dtype.equal(py::dtype("complex128")))
    return torch::kComplexDouble;

  throw "dtype not supported";
}

template <> auto convert(py::array numpy) -> torch::Tensor {
  const auto shapes =
      std::vector<int64_t>{numpy.shape(), numpy.shape() + numpy.ndim()};
  return torch::from_blob(numpy.mutable_data(), shapes,
                          convert<torch::ScalarType>(numpy.dtype()));
}

template <> auto convert(cv::Mat mat) -> torch::Tensor {
  return torch::from_blob(mat.data, {1, mat.rows, mat.cols, 3}, at::kByte);
}

template <> auto convert(torch::Tensor tensor) -> cv::Mat {
  return cv::Mat(cv::Size(512, 512), CV_8UC3, tensor.data<uint8_t>());
}

} // namespace v0
} // namespace flame
