cmake_minimum_required(VERSION 3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_PREFIX_PATH /Users/adamkowalski/pytorch/torch)

project(flame)

find_package(Torch REQUIRED)

add_subdirectory(lib/pybind11)

add_executable(flame
  src/main.cc
  src/convert.cc
  src/gym.cc)

target_include_directories(flame PRIVATE include)

target_link_libraries(flame PRIVATE pybind11::embed)
target_link_libraries(flame PRIVATE ${TORCH_LIBRARIES})

target_compile_features(flame PRIVATE cxx_std_17)