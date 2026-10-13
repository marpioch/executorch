// Ports at::native::onednn::sdpa (pytorch/aten/src/ATen/native/mkldnn/xpu/
// detail/Attention.cpp in the sibling pytorch checkout) onto our libtorch-free
// Tensor*/engine/stream types. Uses oneDNN's Graph API (dnnl::graph), not the
// plain dnnl::matmul primitive xpu_gemm.cpp uses -- a graph of MatMul/Multiply/
// Add/SoftMax ops is built, compiled into a single fused partition, and
// executed. Trimmed to this PoC's scope (see xpu_sdpa.h):
//   - float32 only, dropout_p must be 0, no logsumexp/training outputs.
//   - is_causal is always materialized as an explicit additive mask tensor,
//     matching PyTorch's own fp32 fallback (real oneDNN only has an implicit-
//     causal-mask ukernel for bf16/f16) -- so the implicit-causal GenIndex/
//     GreaterEqual/Select graph ops are not ported.
//   - GQA (num_head_kv < num_head_q) uses oneDNN's documented reshape trick
//     (Q split into [B, group_num, group_size, T, Dh], K/V unsqueezed with a
//     stride-0 broadcast axis) -- no physical K/V duplication needed.
//   - No compiled-partition caching yet: recompiles the graph every call.
//     Fine for correctness validation, not for decode-loop performance.
//   - No KV-cache-offset-aware causal masking yet: assumes seq_len_kv ==
//     seq_len_q (prefill-shaped calls only).
#include <executorch/backends/xpu/runtime/xpu_sdpa.h>

#include <executorch/backends/xpu/runtime/xpu_functions.h>
#include <executorch/backends/xpu/runtime/xpu_guard.h>
#include <executorch/backends/xpu/runtime/xpu_memory.h>
#include <executorch/backends/xpu/runtime/xpu_onednn_utils.h>
#include <executorch/backends/xpu/runtime/xpu_tensor_attribute.h>
#include <executorch/runtime/platform/log.h>

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace executorch::backends::xpu {

using namespace executorch::backends::aoti; // aoti_torch_dtype_float32, etc.

namespace {

using dnnl::graph::logical_tensor;
using dnnl::graph::op;
using data_type = logical_tensor::data_type;

// float32 and bf16 only (matches xpu_gemm.cpp's scope); real PyTorch also
// allows f16, left out here since nothing exercises it yet.
bool is_supported_sdpa_dtype(executorch::aten::ScalarType t) {
  return t == executorch::aten::ScalarType::Float ||
      t == executorch::aten::ScalarType::BFloat16;
}

data_type to_graph_dtype(executorch::aten::ScalarType t) {
  return t == executorch::aten::ScalarType::BFloat16 ? data_type::bf16
                                                      : data_type::f32;
}

} // namespace

extern "C" {

AOTITorchError aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable(
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
    Tensor** ret8) {
  (void)return_debug_mask; // debug mask unsupported, matches upstream's
                           // "not supported" stance for this backend.
  ET_CHECK_OR_RETURN_ERROR(
      query != nullptr && key != nullptr && value != nullptr &&
          ret0 != nullptr,
      InvalidArgument,
      "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
      "null required tensor handle");
  ET_CHECK_OR_RETURN_ERROR(
      query->dim() == 4 && key->dim() == 4 && value->dim() == 4,
      InvalidArgument,
      "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
      "query/key/value must be 4-D [B, H, T, Dh]");
  ET_CHECK_OR_RETURN_ERROR(
      is_supported_sdpa_dtype(query->scalar_type()) &&
          key->scalar_type() == query->scalar_type() &&
          value->scalar_type() == query->scalar_type(),
      InvalidArgument,
      "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
      "query/key/value must be float32 or bfloat16 (and match each other)");
  ET_CHECK_OR_RETURN_ERROR(
      dropout_p == 0.0,
      InvalidArgument,
      "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
      "dropout_p > 0 is not supported");

  const bool has_attn_bias = attn_bias != nullptr && *attn_bias != nullptr;
  ET_CHECK_OR_RETURN_ERROR(
      !(has_attn_bias && is_causal),
      InvalidArgument,
      "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
      "attn_bias cannot be combined with is_causal");

  const int64_t batch = query->sizes()[0];
  const int64_t num_head_q = query->sizes()[1];
  const int64_t seq_len_q = query->sizes()[2];
  const int64_t head_dim_qk = query->sizes()[3];
  const int64_t num_head_kv = key->sizes()[1];
  const int64_t seq_len_kv = key->sizes()[2];
  const int64_t head_dim_v = value->sizes()[3];

  ET_CHECK_OR_RETURN_ERROR(
      key->sizes()[0] == batch && value->sizes()[0] == batch &&
          key->sizes()[3] == head_dim_qk && value->sizes()[1] == num_head_kv &&
          value->sizes()[2] == seq_len_kv,
      InvalidArgument,
      "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
      "key/value shape mismatch with query");
  ET_CHECK_OR_RETURN_ERROR(
      num_head_kv > 0 && num_head_q % num_head_kv == 0,
      InvalidArgument,
      "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
      "num_heads(query) must be a multiple of num_heads(key/value)");
  if (has_attn_bias) {
    ET_CHECK_OR_RETURN_ERROR(
        is_supported_sdpa_dtype((*attn_bias)->scalar_type()),
        InvalidArgument,
        "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
        "attn_bias must be float32 or bfloat16");
  }

  const double scale_d =
      scale != nullptr ? *scale : (1.0 / std::sqrt(static_cast<double>(head_dim_qk)));
  float softmax_scale = static_cast<float>(scale_d);

  const data_type dtype = to_graph_dtype(query->scalar_type());
  const int32_t device_index = c10::xpu::current_device();
  const int32_t dtype_code = query->scalar_type() == executorch::aten::ScalarType::BFloat16
      ? aoti_torch_dtype_bfloat16()
      : aoti_torch_dtype_float32();
  const int32_t dtype_f32 = aoti_torch_dtype_float32(); // causal mask is always f32
  const int32_t device_type_xpu = aoti_torch_device_type_xpu();

  // Allocate the (contiguous) output tensor [B, H_q, T_q, Dv].
  std::array<int64_t, 4> out_sizes = {batch, num_head_q, seq_len_q, head_dim_v};
  std::array<int64_t, 4> out_strides = {
      num_head_q * seq_len_q * head_dim_v,
      seq_len_q * head_dim_v,
      head_dim_v,
      1};
  Tensor* output = nullptr;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_empty_strided(
      4,
      out_sizes.data(),
      out_strides.data(),
      dtype_code,
      device_type_xpu,
      device_index,
      &output));

  // Explicit causal mask (matches PyTorch's own fp32 fallback -- oneDNN only
  // has an implicit-causal ukernel for bf16/f16). USM "shared" memory is
  // host-visible (see xpu_memory.cpp), so we fill it directly.
  Tensor* causal_mask = nullptr;
  if (is_causal) {
    ET_CHECK_OR_RETURN_ERROR(
        seq_len_q == seq_len_kv,
        NotSupported,
        "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
        "is_causal with seq_len_q != seq_len_kv (KV-cache decode) is not "
        "yet supported");
    std::array<int64_t, 2> mask_sizes = {seq_len_q, seq_len_kv};
    std::array<int64_t, 2> mask_strides = {seq_len_kv, 1};
    ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_empty_strided(
        2,
        mask_sizes.data(),
        mask_strides.data(),
        dtype_f32,
        device_type_xpu,
        device_index,
        &causal_mask));
    float* mask_data = static_cast<float*>(causal_mask->mutable_data_ptr());
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < seq_len_q; ++i) {
      for (int64_t j = 0; j < seq_len_kv; ++j) {
        mask_data[i * seq_len_kv + j] = (j <= i) ? 0.f : neg_inf;
      }
    }
  }
  Tensor* mask_tensor = has_attn_bias ? *attn_bias : causal_mask;

  const int64_t group_num = num_head_kv;
  const int64_t group_size = num_head_q / num_head_kv;
  const bool use_gqa = group_size > 1;

  // Logical tensor ids: 0-5 are the named input/output ports, 6+ are
  // shape-inferred intermediates (mirrors SDPALogicalParams::TensorID).
  constexpr size_t kQueryId = 0, kKeyId = 1, kScaleId = 2, kMaskId = 3,
                    kValueId = 4, kOutputId = 5;
  constexpr size_t kMatmulQkOutId = 6, kScaledQkOutId = 7, kMaskedQkOutId = 8,
                    kSoftmaxOutId = 9;

  logical_tensor query_lt, key_lt, value_lt, output_lt;
  if (use_gqa) {
    // oneDNN's documented GQA pattern: split Q's head dim into
    // [group_num, group_size], and give K/V a size-1 "group_size" axis with
    // stride 0 so MatMul broadcasts the same K/V head across the group
    // instead of requiring physically duplicated data.
    auto qstr = query->strides();
    auto kstr = key->strides();
    auto vstr = value->strides();

    dnnl::graph::logical_tensor::dims q_dims = {
        batch, group_num, group_size, seq_len_q, head_dim_qk};
    dnnl::graph::logical_tensor::dims q_strides = {
        static_cast<int64_t>(qstr[0]),
        group_size * static_cast<int64_t>(qstr[1]),
        static_cast<int64_t>(qstr[1]),
        static_cast<int64_t>(qstr[2]),
        static_cast<int64_t>(qstr[3])};
    query_lt = logical_tensor(kQueryId, dtype, q_dims, q_strides);

    dnnl::graph::logical_tensor::dims k_dims = {
        batch, group_num, 1, seq_len_kv, head_dim_qk};
    dnnl::graph::logical_tensor::dims k_strides = {
        static_cast<int64_t>(kstr[0]),
        static_cast<int64_t>(kstr[1]),
        0,
        static_cast<int64_t>(kstr[2]),
        static_cast<int64_t>(kstr[3])};
    key_lt = logical_tensor(kKeyId, dtype, k_dims, k_strides);

    dnnl::graph::logical_tensor::dims v_dims = {
        batch, group_num, 1, seq_len_kv, head_dim_v};
    dnnl::graph::logical_tensor::dims v_strides = {
        static_cast<int64_t>(vstr[0]),
        static_cast<int64_t>(vstr[1]),
        0,
        static_cast<int64_t>(vstr[2]),
        static_cast<int64_t>(vstr[3])};
    value_lt = logical_tensor(kValueId, dtype, v_dims, v_strides);

    dnnl::graph::logical_tensor::dims o_dims = {
        batch, group_num, group_size, seq_len_q, head_dim_v};
    dnnl::graph::logical_tensor::dims o_strides = {
        out_strides[0],
        group_size * out_strides[1],
        out_strides[1],
        out_strides[2],
        out_strides[3]};
    output_lt = logical_tensor(kOutputId, dtype, o_dims, o_strides);
  } else {
    query_lt = logical_tensor(
        kQueryId, dtype, onednn_dims_of(query), onednn_strides_of(query));
    key_lt = logical_tensor(
        kKeyId, dtype, onednn_dims_of(key), onednn_strides_of(key));
    value_lt = logical_tensor(
        kValueId, dtype, onednn_dims_of(value), onednn_strides_of(value));
    output_lt = logical_tensor(
        kOutputId,
        dtype,
        dnnl::graph::logical_tensor::dims(out_sizes.begin(), out_sizes.end()),
        dnnl::graph::logical_tensor::dims(
            out_strides.begin(), out_strides.end()));
  }

  logical_tensor scale_lt(
      kScaleId,
      data_type::f32,
      /*ndims=*/0,
      logical_tensor::layout_type::strided,
      logical_tensor::property_type::host_scalar);

  std::optional<logical_tensor> mask_lt;
  if (mask_tensor != nullptr) {
    // Our synthetic causal mask is always f32 (added against the f32
    // scaled_qk_out intermediate below); a user-provided attn_bias keeps
    // its own dtype, matching PyTorch's mixed-dtype Add support here.
    const data_type mask_dtype =
        has_attn_bias ? to_graph_dtype(mask_tensor->scalar_type()) : data_type::f32;
    mask_lt = logical_tensor(
        kMaskId,
        mask_dtype,
        onednn_dims_of(mask_tensor),
        onednn_strides_of(mask_tensor));
  }

  logical_tensor matmul_qk_out_lt(kMatmulQkOutId, data_type::f32);
  logical_tensor scaled_qk_out_lt(kScaledQkOutId, data_type::f32);
  logical_tensor masked_qk_out_lt;
  if (mask_lt.has_value()) {
    masked_qk_out_lt = logical_tensor(kMaskedQkOutId, data_type::f32);
  }
  // Stays at the real input dtype (not forced f32) since we never enable
  // dropout -- matches PyTorch's own softmax_out{lt_id++, dtype} default.
  logical_tensor softmax_out_lt(kSoftmaxOutId, dtype);

  size_t op_id = 0;
  op matmul_qk(
      op_id++, op::kind::MatMul, {query_lt, key_lt}, {matmul_qk_out_lt}, "matmul_qk");
  matmul_qk.set_attr<bool>(op::attr::transpose_b, true);

  op scale_mul(
      op_id++,
      op::kind::Multiply,
      {matmul_qk_out_lt, scale_lt},
      {scaled_qk_out_lt},
      "scale_mul");

  std::optional<op> mask_add;
  if (mask_lt.has_value()) {
    mask_add = op(
        op_id++,
        op::kind::Add,
        {scaled_qk_out_lt, *mask_lt},
        {masked_qk_out_lt},
        "mask_add");
  }

  op softmax(op_id++, op::kind::SoftMax, "softmax");
  softmax.set_attr<int64_t>(op::attr::axis, -1);
  softmax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
  softmax.add_input(mask_lt.has_value() ? masked_qk_out_lt : scaled_qk_out_lt);
  softmax.add_output(softmax_out_lt);

  op matmul_v(
      op_id++,
      op::kind::MatMul,
      {softmax_out_lt, value_lt},
      {output_lt},
      "matmul_v");

  try {
    dnnl::graph::graph g(dnnl::engine::kind::gpu);
    g.add_op(matmul_qk);
    g.add_op(scale_mul);
    if (mask_add.has_value()) {
      g.add_op(*mask_add);
    }
    g.add_op(softmax);
    g.add_op(matmul_v);
    g.finalize();

    auto partitions = g.get_partitions();
    ET_CHECK_OR_RETURN_ERROR(
        partitions.size() == 1 && partitions[0].is_supported(),
        Internal,
        "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
        "oneDNN graph did not produce a single supported SDPA partition");

    dnnl::engine& eng = onednn_engine_for(device_index);
    sycl::queue* q = get_or_create_xpu_queue(device_index);
    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, *q);

    std::vector<logical_tensor> compile_inputs = {query_lt, key_lt, scale_lt};
    if (mask_lt.has_value()) {
      compile_inputs.push_back(*mask_lt);
    }
    compile_inputs.push_back(value_lt);
    std::vector<logical_tensor> compile_outputs = {output_lt};

    auto compiled =
        partitions[0].compile(compile_inputs, compile_outputs, eng);

    std::vector<dnnl::graph::tensor> inputs;
    inputs.emplace_back(query_lt, eng, query->mutable_data_ptr());
    inputs.emplace_back(key_lt, eng, key->mutable_data_ptr());
    inputs.push_back(
        dnnl::graph::tensor::make_scalar_tensor(scale_lt, &softmax_scale));
    if (mask_lt.has_value()) {
      inputs.emplace_back(*mask_lt, eng, mask_tensor->mutable_data_ptr());
    }
    inputs.emplace_back(value_lt, eng, value->mutable_data_ptr());

    std::vector<dnnl::graph::tensor> outputs;
    outputs.emplace_back(output_lt, eng, output->mutable_data_ptr());

    compiled.execute(strm, inputs, outputs);
    strm.wait();
  } catch (const dnnl::error& e) {
    ET_LOG(
        Error,
        "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
        "oneDNN error: %s (status=%d)",
        e.message,
        e.status);
    return Error::Internal;
  } catch (const sycl::exception& e) {
    ET_LOG(
        Error,
        "aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable: "
        "SYCL error: %s",
        e.what());
    return Error::Internal;
  }

  *ret0 = output;
  if (ret1 != nullptr) *ret1 = nullptr; // logsumexp: training-only, not computed
  if (ret2 != nullptr) *ret2 = nullptr; // cum_seq_q: unused outside nested tensors
  if (ret3 != nullptr) *ret3 = nullptr; // cum_seq_k: unused outside nested tensors
  if (ret4 != nullptr) *ret4 = seq_len_q; // max_q (SymInt)
  if (ret5 != nullptr) *ret5 = seq_len_kv; // max_k (SymInt)
  if (ret6 != nullptr) *ret6 = nullptr; // philox_seed: no dropout, unused
  if (ret7 != nullptr) *ret7 = nullptr; // philox_offset: no dropout, unused
  if (ret8 != nullptr) *ret8 = nullptr; // debug_attn_mask: not supported

  return Error::Ok;
}

} // extern "C"

} // namespace executorch::backends::xpu
