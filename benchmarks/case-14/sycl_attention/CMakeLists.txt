cmake_minimum_required(VERSION 3.22)
project(case14_sycl_attention LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  message(FATAL_ERROR "Use Intel oneAPI DPC++ (configure after running setvars.bat)")
endif()

add_executable(sycl_attention sycl_attention.cpp)
target_compile_features(sycl_attention PRIVATE cxx_std_17)
target_compile_options(sycl_attention PRIVATE -fsycl -O3)
target_link_options(sycl_attention PRIVATE -fsycl)

