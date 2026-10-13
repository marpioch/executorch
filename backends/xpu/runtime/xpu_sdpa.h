/*
 * SDPA shim for XPU, backed by oneDNN's Graph API (dnnl::graph), matching
 * how PyTorch's own eager XPU backend implements scaled_dot_product_attention
 * (see at::native::onednn::sdpa, pytorch/aten/src/ATen/native/mkldnn/xpu/
 * detail/Attention.cpp in the sibling pytorch checkout -- a different, higher
 * level API than the plain dnnl::matmul primitive xpu_gemm.cpp uses). Ports
 * the "overrideable" backend XPU's dispatcher always picks (flash/cudnn/
 * efficient are not implemented for XPU), trimmed to this PoC's scope:
 * float32 or bfloat16 (query/key/value must match; attn_bias/causal mask may
 * differ -- mirrors PyTorch's own mixed-dtype Add support), no dropout, no
 * logsumexp/training outputs, causal handled as an explicit additive f32 mask
 * (matching PyTorch's own fp32 fallback), and no compiled-partition caching
 * yet (recompiles every call).
 */
#pragma once

#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/aoti/export.h>

namespace executorch::backends::xpu {

using executorch::backends::aoti::AOTITorchError;
using executorch::backends::aoti::Tensor;

extern "C" {

// Matches the real aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable
// signature exactly (see pytorch/torch/csrc/inductor/aoti_torch/generated/
// c_shim_xpu.h) -- ret4/ret5 are plain int64_t* (SymInt max_q/max_k), not
// tensor handles; getting that by-value/by-pointer shape wrong would
// silently corrupt the ABI the same way a by-value vs by-reference extern
// "C" param mismatch did for aoti_torch_set_current_xpu_device.
AOTI_SHIM_EXPORT AOTITorchError
aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable(
    Tensor* query,
    Tensor* key,
    Tensor* value,
    Tensor** attn_bias,
    double dropout_p,
    int32_t is_causal,
    int32_t return_debug_mask,
    double* scale,
    Tensor** ret0,
    Tensor** ret1,
    Tensor** ret2,
    Tensor** ret3,
    int64_t* ret4,
    int64_t* ret5,
    Tensor** ret6,
    Tensor** ret7,
    Tensor** ret8);

} // extern "C"

} // namespace executorch::backends::xpu
