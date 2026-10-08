/*
 * XPU tensor memory shims not covered by backends/aoti/common_shims.* (whose
 * versions of these throw "Not implemented" -- each AOTI backend supplies its
 * own, same pattern as backends/apple/metal/runtime/shims/memory.* and
 * backends/cuda/runtime/shims/memory.cpp). Backed by SYCL USM allocations
 * instead of libtorch/c10 tensor storage.
 *
 * Deliberately minimal: only the functions a simple elementwise model (e.g.
 * add) actually calls, per the real linker-reported symbol list in
 * xpu_shims.def. Anything else still falls back to common_shims.cpp's
 * throwing stubs until a model that needs it surfaces a load-time error.
 */
#pragma once

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::xpu {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

extern "C" {

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_create_tensor_from_blob_v2(
    void* data,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor,
    int32_t layout,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size);

AOTI_SHIM_EXPORT AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor);

// Not part of xpu_shims.def's real-linker-reported list, but needed by
// xpu_backend.cpp's execute(): copies raw bytes between a CPU tensor and an
// XPU tensor allocated via aoti_torch_empty_strided. Both live in SYCL USM
// "shared" memory or plain host memory, both host-visible, so a plain memcpy
// suffices (no real host<->device transfer like Metal/CUDA need).
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_copy_(Tensor* self, Tensor* src, int32_t non_blocking);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_delete_tensor_object(Tensor* tensor);

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_new_tensor_handle(Tensor* orig_handle, Tensor** new_handle);

// Creates a view into `self`'s existing storage with new sizes/strides/
// storage_offset (AOTInductor's generated wrapper uses this for view ops
// instead of calling empty_strided + copy).
AOTI_SHIM_EXPORT AOTITorchError aoti_torch__reinterpret_tensor(
    Tensor* self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    Tensor** ret_new_tensor);

} // extern "C"

} // namespace executorch::backends::xpu
