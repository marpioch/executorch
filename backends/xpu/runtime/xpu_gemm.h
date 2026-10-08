/*
 * GEMM shim for XPU, backed by oneDNN (DNNL_GPU_RUNTIME=SYCL). AOTInductor's
 * Windows XPU wrapper has no extern GEMM kernel of its own (unlike CUDA,
 * which links against libtorch's cuBLAS-backed aten::mm); it instead expects
 * the backend to supply aoti_torch_xpu_mm_out directly, the same way
 * backends/apple/metal/runtime/ops/op_mm.mm supplies aoti_torch_mps_mm_out.
 * Elementwise ops (bias add, relu, ...) stay on the Triton-XPU path; only the
 * matmul itself routes through oneDNN. See test_apps/test_addmm_onednn.cpp
 * for the standalone, pre-validated oneDNN/SYCL pattern this ports.
 */
#pragma once

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::xpu {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

extern "C" {

AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_xpu_mm_out(Tensor* out, Tensor* self, Tensor* mat2);

} // extern "C"

} // namespace executorch::backends::xpu
