// Ports at::native::xpu::mm_out / at::native::onednn::matmul (see
// pytorch/aten/src/ATen/native/mkldnn/xpu/{Blas.cpp,detail/{Matmul,Utils}.cpp}
// in the sibling pytorch checkout) onto our libtorch-free Tensor*/engine/
// stream types, trimmed to this PoC's scope: float32 only, 2-D only, no bias
// (nn.Linear's bias+relu stay on the Triton-XPU elementwise path -- see the
// mm_out/addmm_out split discovered from the real unresolved-symbol list), no
// post-ops/broadcast/batch support.
#include <executorch/backends/xpu/runtime/xpu_gemm.h>

#include <executorch/backends/xpu/runtime/xpu_functions.h>
#include <executorch/backends/xpu/runtime/xpu_guard.h>
#include <executorch/runtime/platform/log.h>

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include <mutex>
#include <unordered_map>

namespace executorch::backends::xpu {

namespace {

std::mutex g_engines_mutex;
std::unordered_map<int32_t, dnnl::engine> g_engines;

// Lazily creates one oneDNN engine per device index, wrapping the same
// sycl::device/context the rest of the runtime (xpu_functions.cpp) already
// uses, so GEMM memory objects alias the same USM allocations other shims
// wrote to. Mirrors GpuEngineManager (detail/oneDNNContext.cpp), which we
// can't reuse directly since it is only reachable through libtorch.
dnnl::engine& engine_for(int32_t device_index) {
  std::lock_guard<std::mutex> lock(g_engines_mutex);
  auto it = g_engines.find(device_index);
  if (it != g_engines.end()) {
    return it->second;
  }
  auto [new_it, inserted] = g_engines.emplace(
      device_index,
      dnnl::sycl_interop::make_engine(
          c10::xpu::get_raw_device(device_index),
          c10::xpu::get_device_context()));
  (void)inserted;
  return new_it->second;
}

// Mirrors at::native::onednn::make_onednn_memory (detail/Utils.cpp).
dnnl::memory make_onednn_memory(
    const dnnl::memory::desc& md,
    dnnl::engine& engine,
    void* ptr) {
  return dnnl::sycl_interop::make_memory(
      md,
      engine,
      dnnl::sycl_interop::memory_kind::usm,
      ptr == nullptr ? DNNL_MEMORY_ALLOCATE : ptr);
}

dnnl::memory::dims dims_of(Tensor* t) {
  auto sizes = t->sizes();
  return dnnl::memory::dims(sizes.begin(), sizes.end());
}

dnnl::memory::dims strides_of(Tensor* t) {
  auto strides = t->strides();
  return dnnl::memory::dims(strides.begin(), strides.end());
}

// Mirrors the core requirement of at::native::onednn::is_onednn_matmul_strides
// (detail/Utils.cpp): oneDNN's matmul primitive needs at least one of the
// last two axes contiguous (stride==1). Views that don't satisfy this (e.g.
// holes from a strided slice) are out of scope for this PoC shim.
bool has_onednn_compatible_strides(Tensor* t) {
  auto strides = t->strides();
  int64_t n = static_cast<int64_t>(strides.size());
  return n < 2 || strides[n - 1] == 1 || strides[n - 2] == 1;
}

} // namespace

extern "C" {

AOTITorchError aoti_torch_xpu_mm_out(Tensor* out, Tensor* self, Tensor* mat2) {
  ET_CHECK_OR_RETURN_ERROR(
      out != nullptr && self != nullptr && mat2 != nullptr,
      InvalidArgument,
      "aoti_torch_xpu_mm_out: null tensor handle");
  ET_CHECK_OR_RETURN_ERROR(
      self->dim() == 2 && mat2->dim() == 2 && out->dim() == 2,
      InvalidArgument,
      "aoti_torch_xpu_mm_out: mm only supports 2-D tensors");
  ET_CHECK_OR_RETURN_ERROR(
      self->scalar_type() == executorch::aten::ScalarType::Float &&
          mat2->scalar_type() == executorch::aten::ScalarType::Float &&
          out->scalar_type() == executorch::aten::ScalarType::Float,
      InvalidArgument,
      "aoti_torch_xpu_mm_out: only float32 is supported");
  ET_CHECK_OR_RETURN_ERROR(
      has_onednn_compatible_strides(self) &&
          has_onednn_compatible_strides(mat2) &&
          has_onednn_compatible_strides(out),
      InvalidArgument,
      "aoti_torch_xpu_mm_out: unsupported (non-oneDNN-compatible) strides");

  try {
    int32_t device_index = c10::xpu::current_device();
    dnnl::engine& eng = engine_for(device_index);
    sycl::queue* q = get_or_create_xpu_queue(device_index);
    dnnl::stream strm = dnnl::sycl_interop::make_stream(eng, *q);

    // mat2 is already [k, n] (mm_out's m2_trans=true convention), so its own
    // strides are used as-is -- no logical transpose needed.
    dnnl::memory::desc self_md(
        dims_of(self), dnnl::memory::data_type::f32, strides_of(self));
    dnnl::memory::desc mat2_md(
        dims_of(mat2), dnnl::memory::data_type::f32, strides_of(mat2));
    dnnl::memory::desc out_md(
        dims_of(out), dnnl::memory::data_type::f32, strides_of(out));

    // Real impl uses scratchpad_mode::user (rather than the default
    // "library" mode) so the scratchpad buffer is an explicit, visible USM
    // allocation instead of one oneDNN manages internally.
    dnnl::primitive_attr pattr;
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    dnnl::matmul::primitive_desc pd(eng, self_md, mat2_md, out_md, pattr);
    dnnl::matmul matmul_p(pd);

    auto self_mem = make_onednn_memory(self_md, eng, self->mutable_data_ptr());
    auto mat2_mem = make_onednn_memory(mat2_md, eng, mat2->mutable_data_ptr());
    auto out_mem = make_onednn_memory(out_md, eng, out->mutable_data_ptr());

    size_t scratchpad_size = pd.scratchpad_desc().get_size();
    void* scratchpad_ptr = scratchpad_size > 0
        ? sycl::malloc_device(
              scratchpad_size,
              c10::xpu::get_raw_device(device_index),
              c10::xpu::get_device_context())
        : nullptr;
    auto scratchpad_mem =
        make_onednn_memory(pd.scratchpad_desc(), eng, scratchpad_ptr);

    sycl::event done = dnnl::sycl_interop::execute(
        matmul_p,
        strm,
        {
            {DNNL_ARG_SRC, self_mem},
            {DNNL_ARG_WEIGHTS, mat2_mem},
            {DNNL_ARG_DST, out_mem},
            {DNNL_ARG_SCRATCHPAD, scratchpad_mem},
        });
    done.wait();

    if (scratchpad_ptr != nullptr) {
      sycl::free(scratchpad_ptr, c10::xpu::get_device_context());
    }
  } catch (const dnnl::error& e) {
    ET_LOG(
        Error,
        "aoti_torch_xpu_mm_out: oneDNN error: %s (status=%d)",
        e.message,
        e.status);
    return Error::Internal;
  } catch (const sycl::exception& e) {
    ET_LOG(Error, "aoti_torch_xpu_mm_out: SYCL error: %s", e.what());
    return Error::Internal;
  }
  return Error::Ok;
}

} // extern "C"

} // namespace executorch::backends::xpu
