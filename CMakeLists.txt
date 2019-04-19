cmake_minimum_required(VERSION 3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_PREFIX_PATH /Users/adamkowalski/pytorch/torch)

project(flame)

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

add_executable(flame
  src/main.cc
  src/flame.cc
  src/convert.cc
  src/callback.cc
  src/tensorboard.cc
  src/simulation.cc
  src/gym/gym.cc
  src/agent/random.cc
  src/agent/policy_gradient.cc)

target_include_directories(flame PRIVATE include)
target_include_directories(flame PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(flame PRIVATE pybind11::embed)
target_link_libraries(flame PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(flame PRIVATE ${OpenCV_LIBS})
target_link_libraries(flame PRIVATE ${Boost_LIBRARIES})

target_compile_features(flame PRIVATE cxx_std_17)
