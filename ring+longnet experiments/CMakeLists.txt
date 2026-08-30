cmake_minimum_required(VERSION 3.16)
project(ring_longnet_experiments LANGUAGES CXX)

add_executable(ring_longnet_experiments main.cpp)
target_compile_features(ring_longnet_experiments PRIVATE cxx_std_17)
target_compile_options(ring_longnet_experiments PRIVATE -O3 -Wall -Wextra -Wpedantic)

if (UNIX)
  target_link_libraries(ring_longnet_experiments PRIVATE pthread)
endif()
