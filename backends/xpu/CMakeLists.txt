# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Build the minimal XPU backend runtime (see backends/xpu/runtime/).
#
cmake_minimum_required(VERSION 3.19)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

include(${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  message(
    FATAL_ERROR
      "XPU backend requires the Intel oneAPI DPC++ compiler (icx/icpx) as "
      "CMAKE_CXX_COMPILER, since xpu_guard.cpp/xpu_backend.cpp need -fsycl. "
      "Got CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}."
  )
endif()

add_library(
  xpu_backend STATIC runtime/xpu_backend.cpp runtime/xpu_guard.cpp
)

target_include_directories(
  xpu_backend PUBLIC $<BUILD_INTERFACE:${EXECUTORCH_ROOT}>
                     $<BUILD_INTERFACE:${EXECUTORCH_ROOT}/..>
                     $<INSTALL_INTERFACE:include>
)

# icx needs -fsycl at both compile and link time.
target_compile_options(xpu_backend PRIVATE -fsycl)
target_link_options(xpu_backend PUBLIC -fsycl)

target_link_libraries(xpu_backend PUBLIC executorch_core ${CMAKE_DL_LIBS})

# Backend registration (the static `register_backend()` call in
# xpu_backend.cpp) must survive linking even though nothing else references
# it; see the whole-archive helper this calls into.
executorch_target_link_options_shared_lib(xpu_backend)

# Required: executorch_backends (an INTERFACE lib linking every backend added
# to _executorch_backends) is itself part of the ExecuTorchTargets export set,
# so every backend it links must be in that same export set too.
install(TARGETS xpu_backend EXPORT ExecuTorchTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})
