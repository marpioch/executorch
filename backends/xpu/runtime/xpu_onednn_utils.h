/*
 * Shared oneDNN engine/dims helpers for the XPU GEMM (xpu_gemm.cpp) and SDPA
 * (xpu_sdpa.cpp) shims. `inline` so the function-local statics in
 * onednn_engine_for resolve to a single shared instance across both.
 */
#pragma once

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/xpu/runtime/xpu_functions.h>

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include <mutex>
#include <unordered_map>

namespace executorch::backends::xpu {

using executorch::backends::aoti::Tensor;

// Lazily creates one oneDNN engine per device index, wrapping the same
// sycl::device/context the rest of the runtime (xpu_functions.cpp) already
// uses, so oneDNN memory/tensor objects alias the same USM allocations other
// shims wrote to. Mirrors GpuEngineManager (pytorch's detail/oneDNNContext.cpp),
// which we can't reuse directly since it is only reachable through libtorch.
inline dnnl::engine& onednn_engine_for(int32_t device_index) {
  static std::mutex mutex;
  static std::unordered_map<int32_t, dnnl::engine> engines;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = engines.find(device_index);
  if (it != engines.end()) {
    return it->second;
  }
  auto [new_it, inserted] = engines.emplace(
      device_index,
      dnnl::sycl_interop::make_engine(
          c10::xpu::get_raw_device(device_index),
          c10::xpu::get_device_context()));
  (void)inserted;
  return new_it->second;
}

inline dnnl::memory::dims onednn_dims_of(Tensor* t) {
  auto sizes = t->sizes();
  return dnnl::memory::dims(sizes.begin(), sizes.end());
}

inline dnnl::memory::dims onednn_strides_of(Tensor* t) {
  auto strides = t->strides();
  return dnnl::memory::dims(strides.begin(), strides.end());
}

} // namespace executorch::backends::xpu
