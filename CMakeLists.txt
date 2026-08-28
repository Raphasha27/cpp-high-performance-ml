cmake_minimum_required(VERSION 3.28)
project(cpp_high_performance_ml VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_library(ml_core
    src/tensor.cpp
    src/matrix.cpp
    src/neural_network.cpp
)

target_include_directories(ml_core PUBLIC include)
target_link_libraries(ml_core PUBLIC Eigen3::Eigen)

add_executable(ml_benchmark benchmarks/benchmark.cpp)
target_link_libraries(ml_benchmark PRIVATE ml_core)

add_executable(ml_example examples/xor_problem.cpp)
target_link_libraries(ml_example PRIVATE ml_core)

enable_testing()
add_executable(ml_tests tests/test_tensor.cpp)
target_link_libraries(ml_tests PRIVATE ml_core)
add_test(NAME ml_tests COMMAND ml_tests)
