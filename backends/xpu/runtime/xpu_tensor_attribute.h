/*
 * XPU device-type shims not covered by backends/aoti/common_shims.* (each
 * AOTI backend must supply its own aoti_torch_get_device_type; see
 * backends/apple/metal/runtime/shims/tensor_attribute.cpp and
 * backends/cuda/runtime/shims/tensor_attribute.cpp for the same per-backend
 * pattern).
 */
#pragma once

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::xpu {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

extern "C" {

AOTI_SHIM_EXPORT int32_t aoti_torch_device_type_xpu();

// Never actually meaningful for a tensor this runtime creates (always XPU),
// but xpu_shims.def lists it: some AOTI-generated code compares a tensor's
// device_type against every known constant, so the symbol must resolve for
// the DLL to load even though the XPU branch always wins.
AOTI_SHIM_EXPORT int32_t aoti_torch_device_type_cuda();

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_get_device_type(Tensor* tensor, int32_t* ret_device_type);

} // extern "C"

} // namespace executorch::backends::xpu
