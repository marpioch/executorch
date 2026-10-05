/*
 * Libtorch-free re-implementation of c10::xpu::{current_device,
 * get_raw_device, get_device_context} (see c10/xpu/XPUFunctions.h in
 * pytorch/pytorch). AOTInductor's Triton-XPU kernel-launch wrapper code calls
 * these exact mangled symbols directly (not through the aoti_torch_* C shim)
 * to bridge an abstract sycl::device/context down to native Level-Zero
 * handles for zeModuleCreate/zeKernelCreate. Unlike CUDA (which talks to the
 * CUDA Driver API directly from a cudaStream_t, no abstraction bridging
 * needed), XPU/SYCL requires this extra translation step.
 */
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include <executorch/backends/aoti/export.h>

namespace c10::xpu {

using DeviceIndex = int8_t;

// dllexport needed explicitly: unlike Linux/macOS, MSVC exports nothing from
// a DLL by default, and the AOTInductor-generated blob loaded as a *separate*
// DLL calls these by exact mangled name, not through xpu_shims.dll's own code.
AOTI_SHIM_EXPORT DeviceIndex current_device();
AOTI_SHIM_EXPORT sycl::device& get_raw_device(DeviceIndex device);
AOTI_SHIM_EXPORT sycl::context& get_device_context();

// Not part of the real c10::xpu ABI; lets the rest of the XPU runtime (guard
// and stream shims) share this same current-device bookkeeping and device
// pool/context instead of keeping a second, inconsistent copy.
void set_current_device(DeviceIndex device);

} // namespace c10::xpu
