/*
 * Minimal, libtorch-free AOTI XPU guard/stream shim (see shim_xpu.h in
 * pytorch/pytorch for the real libtorch-backed contract this mirrors).
 * Backed directly by SYCL; no c10/ATen dependency.
 */
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include <executorch/backends/aoti/export.h>
#include <executorch/runtime/core/error.h>

namespace executorch::backends::xpu {

using executorch::runtime::Error;
using AOTITorchError = Error;

extern "C" {

struct XPUGuardOpaque;
using XPUGuardHandle = XPUGuardOpaque*;
struct XPUStreamGuardOpaque;
using XPUStreamGuardHandle = XPUStreamGuardOpaque*;

// Device guard: makes device_index "current" for its scope.
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_create_xpu_guard(int32_t device_index, XPUGuardHandle* ret_guard);
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_delete_xpu_guard(XPUGuardHandle guard);
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_xpu_guard_set_index(XPUGuardHandle guard, int32_t device_index);

// Stream (queue) guard: makes `stream` the current queue for device_index.
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_create_xpu_stream_guard(
    void* stream,
    int32_t device_index,
    XPUStreamGuardHandle* ret_guard);
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_delete_xpu_stream_guard(XPUStreamGuardHandle guard);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_current_xpu_stream(int32_t device_index, void** ret_stream);
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_current_xpu_device(int32_t* device_index);
// Matches torch's real shim_xpu.h signature exactly: a by-value int32_t here
// would silently corrupt the call, since extern "C" linkage means no
// mangling -- and thus no compile-time ABI check -- catches the mismatch.
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_set_current_xpu_device(const int32_t& device_index);

// Not part of the XPU guard/stream contract above, but same shape: called
// directly by AOTInductor's generated kernel-launch code to get a queue for
// the *current* device (c10::xpu::current_device()), no index parameter.
AOTI_SHIM_EXPORT AOTITorchError aoti_torch_get_current_sycl_queue(void** ret);

} // extern "C"

// Not part of the AOTI shim ABI; used directly by xpu_backend.cpp to obtain
// the queue to run a container invocation on.
sycl::queue* get_or_create_xpu_queue(int32_t device_index);

} // namespace executorch::backends::xpu
