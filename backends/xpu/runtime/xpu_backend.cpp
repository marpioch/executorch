/*
 * Minimal, libtorch-free BackendInterface for AOTInductor-compiled XPU
 * methods. Loads the compiled .so, resolves the generic AOTI container
 * function pointers (see backends/aoti/aoti_delegate_handle.h), and runs it
 * on a queue obtained from xpu_guard. No weight/constant blob handling yet --
 * fine for the add/mm PoC, which has no parameters.
 */
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <executorch/backends/aoti/aoti_delegate_handle.h>
#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/xpu/runtime/xpu_allocator.h>
#include <executorch/backends/xpu/runtime/xpu_delegate_handle.h>
#include <executorch/backends/xpu/runtime/xpu_guard.h>
#include <executorch/backends/xpu/runtime/xpu_memory.h>
#include <executorch/backends/xpu/runtime/xpu_tensor_attribute.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/device_allocator.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

namespace executorch::backends::xpu {

using executorch::backends::aoti::AOTInductorModelContainerCreateWithDeviceFunc;
using executorch::backends::aoti::AOTInductorModelContainerDeleteFunc;
using executorch::backends::aoti::AOTInductorModelContainerGetNumInputsFunc;
using executorch::backends::aoti::AOTInductorModelContainerGetNumOutputsFunc;
using executorch::backends::aoti::AOTInductorModelContainerRunFunc;
using executorch::backends::aoti::resolve_blob_keys;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::NamedDataMap;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::etensor::Tensor;

namespace {

Result<void*> load_library(const std::string& path) {
#ifdef _WIN32
  HMODULE lib = LoadLibraryA(path.c_str());
  if (lib == nullptr) {
    return Error::AccessFailed;
  }
  return static_cast<void*>(lib);
#else
  void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    ET_LOG(Error, "dlopen(%s) failed: %s", path.c_str(), dlerror());
    return Error::AccessFailed;
  }
  return lib;
#endif
}

void close_library(void* lib) {
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(lib));
#else
  dlclose(lib);
#endif
}

template <typename FuncT>
Result<FuncT> get_symbol(void* lib, const char* name) {
#ifdef _WIN32
  auto* sym = GetProcAddress(static_cast<HMODULE>(lib), name);
#else
  auto* sym = dlsym(lib, name);
#endif
  if (sym == nullptr) {
    ET_LOG(Error, "Could not resolve symbol %s", name);
    return Error::AccessFailed;
  }
  return reinterpret_cast<FuncT>(sym);
}

} // namespace

class XpuBackend final : public ::executorch::runtime::BackendInterface {
 public:
  bool is_available() const override {
    return !sycl::device::get_devices(sycl::info::device_type::gpu).empty();
  }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    std::string method_name;
    for (const CompileSpec& spec : compile_specs) {
      if (std::strcmp(spec.key, "method_name") == 0) {
        method_name.assign(
            static_cast<const char*>(spec.value.buffer), spec.value.nbytes);
      }
    }

    std::string so_blob_key, weights_blob_key;
    ET_CHECK_OK_OR_RETURN_ERROR(
        resolve_blob_keys(processed, method_name, so_blob_key, weights_blob_key),
        "Malformed named-data key payload");

    const NamedDataMap* named_data_map = context.get_named_data_map();
    auto so_buffer = named_data_map->get_data(so_blob_key.c_str());
    ET_CHECK_OR_RETURN_ERROR(
        so_buffer.ok(),
        Internal,
        "Failed to get data for key %s",
        so_blob_key.c_str());

    static std::atomic<uint64_t> so_file_counter{0};
    std::filesystem::path so_path = std::filesystem::temp_directory_path() /
        ("executorch_xpu_" +
         std::to_string(so_file_counter.fetch_add(1, std::memory_order_relaxed)) +
         ".so");

    std::ofstream outfile(so_path, std::ios::binary);
    outfile.write(
        static_cast<const char*>(so_buffer->data()), so_buffer->size());
    ET_CHECK_OR_RETURN_ERROR(
        outfile.good(), AccessFailed, "Failed to write %s", so_path.string().c_str());
    outfile.close();
    so_buffer->Free();

    auto lib = load_library(so_path.string());
    if (!lib.ok()) {
      return lib.error();
    }

    auto create_fn = get_symbol<AOTInductorModelContainerCreateWithDeviceFunc>(
        lib.get(), "AOTInductorModelContainerCreateWithDevice");
    auto delete_fn = get_symbol<AOTInductorModelContainerDeleteFunc>(
        lib.get(), "AOTInductorModelContainerDelete");
    auto num_inputs_fn = get_symbol<AOTInductorModelContainerGetNumInputsFunc>(
        lib.get(), "AOTInductorModelContainerGetNumInputs");
    auto num_outputs_fn = get_symbol<AOTInductorModelContainerGetNumOutputsFunc>(
        lib.get(), "AOTInductorModelContainerGetNumOutputs");
    auto run_fn = get_symbol<AOTInductorModelContainerRunFunc>(
        lib.get(), "AOTInductorModelContainerRun");
    if (!create_fn.ok() || !delete_fn.ok() || !num_inputs_fn.ok() ||
        !num_outputs_fn.ok() || !run_fn.ok()) {
      close_library(lib.get());
      return Error::AccessFailed;
    }

    AOTInductorModelContainerHandle container = nullptr;
    Error err =
        (*create_fn.get())(&container, /*num_models=*/1, "xpu", nullptr);
    ET_CHECK_OR_RETURN_ERROR(
        err == Error::Ok, Internal, "Failed to create XPU AOTI container");

    auto* handle = new XpuDelegateHandle();
    handle->lib_handle = lib.get();
    handle->container_handle = container;
    handle->container_delete = delete_fn.get();
    handle->get_num_inputs = num_inputs_fn.get();
    handle->get_num_outputs = num_outputs_fn.get();
    handle->run = run_fn.get();
    handle->device_index = 0; // single-GPU PoC

    return handle;
  }

  Error execute(
      BackendExecutionContext& context,
      DelegateHandle* handle_,
      Span<EValue*> args) const override {
    auto* handle = static_cast<XpuDelegateHandle*>(handle_);

    size_t n_inputs = 0, n_outputs = 0;
    handle->get_num_inputs(handle->container_handle, &n_inputs);
    handle->get_num_outputs(handle->container_handle, &n_outputs);
    ET_CHECK_OR_RETURN_ERROR(
        n_inputs + n_outputs == args.size(),
        InvalidArgument,
        "expected %zu input(s) + %zu output(s), got %zu args",
        n_inputs,
        n_outputs,
        args.size());

    const int32_t xpu_device_type = aoti_torch_device_type_xpu();

    // ExecuTorch's own Tensor* args are host-resident and were never
    // registered with xpu_memory.cpp's tracking maps. AOTInductor's
    // generated code treats every handle it touches as RAII-owned (it calls
    // aoti_torch_delete_tensor_object on inputs once it's done with them, and
    // may replace our output handles with its own) -- passing the raw args
    // straight into Run() means that cleanup can't find them and aborts. So
    // wrap each input/output as a tracked XPU tensor first, mirroring
    // backends/apple/metal/runtime/metal_backend.cpp's execute().
    std::vector<Tensor*> xpu_inputs(n_inputs, nullptr);
    std::vector<Tensor*> xpu_outputs(n_outputs, nullptr);
    std::vector<Tensor*> pre_run_outputs(n_outputs, nullptr);
    bool run_called = false;

    executorch::backends::aoti::ScopeGuard cleanup([&]() noexcept {
      if (!run_called) {
        for (auto* t : xpu_inputs) {
          if (t != nullptr) {
            aoti_torch_delete_tensor_object(t);
          }
        }
      }
      for (size_t i = 0; i < xpu_outputs.size(); i++) {
        if (pre_run_outputs[i] != nullptr &&
            pre_run_outputs[i] != xpu_outputs[i]) {
          aoti_torch_delete_tensor_object(pre_run_outputs[i]);
        }
        if (xpu_outputs[i] != nullptr) {
          aoti_torch_delete_tensor_object(xpu_outputs[i]);
        }
      }
    });

    for (size_t i = 0; i < n_inputs; i++) {
      auto* cpu_tensor = &(args[i]->toTensor());
      auto sizes = cpu_tensor->sizes();
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());
      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_empty_strided(
              sizes_vec.size(),
              sizes_vec.data(),
              nullptr,
              static_cast<int32_t>(cpu_tensor->scalar_type()),
              xpu_device_type,
              handle->device_index,
              &xpu_inputs[i]),
          "Failed to create XPU tensor for input %zu",
          i);
      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_copy_(xpu_inputs[i], cpu_tensor, 0),
          "Failed to copy input %zu from CPU to XPU",
          i);
    }

    for (size_t i = 0; i < n_outputs; i++) {
      auto* cpu_tensor = &(args[n_inputs + i]->toTensor());
      auto sizes = cpu_tensor->sizes();
      std::vector<int64_t> sizes_vec(sizes.begin(), sizes.end());
      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_empty_strided(
              sizes_vec.size(),
              sizes_vec.data(),
              nullptr,
              static_cast<int32_t>(cpu_tensor->scalar_type()),
              xpu_device_type,
              handle->device_index,
              &xpu_outputs[i]),
          "Failed to create XPU tensor for output %zu",
          i);
      pre_run_outputs[i] = xpu_outputs[i];
    }

    void* stream = get_or_create_xpu_queue(handle->device_index);
    Error err = handle->run(
        handle->container_handle,
        xpu_inputs.data(),
        n_inputs,
        xpu_outputs.data(),
        n_outputs,
        stream,
        /*proxy_executor_handle=*/nullptr);
    run_called = true;
    if (err != Error::Ok) {
      return err;
    }

    // Make sure the kernel(s) actually finished before reading results back.
    static_cast<sycl::queue*>(stream)->wait();

    for (size_t i = 0; i < n_outputs; i++) {
      auto* cpu_tensor = &(args[n_inputs + i]->toTensor());
      ET_CHECK_OK_OR_RETURN_ERROR(
          executorch::runtime::resize_tensor(
              *cpu_tensor, xpu_outputs[i]->sizes()),
          "Failed to resize output %zu",
          i);
      ET_CHECK_OK_OR_RETURN_ERROR(
          aoti_torch_copy_(cpu_tensor, xpu_outputs[i], 0),
          "Failed to copy output %zu from XPU to CPU",
          i);
    }

    // ScopeGuard destructor deletes the tracked XPU input/output handles.
    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    if (handle_ == nullptr) {
      return;
    }
    auto* handle = static_cast<XpuDelegateHandle*>(handle_);
    if (handle->container_delete != nullptr) {
      handle->container_delete(handle->container_handle);
    }
    if (handle->lib_handle != nullptr) {
      close_library(handle->lib_handle);
    }
    delete handle;
  }
};

namespace {
auto cls = XpuBackend();
Backend backend{"XpuBackend", &cls};
static auto success = ::executorch::runtime::register_backend(backend);

// Auto-register the XpuAllocator so device-aware memory planning can
// allocate the delegate's XPU-tagged planned buffers whenever this backend
// library is linked (mirrors backends/cuda/runtime/cuda_backend.cpp).
static bool xpu_allocator_registered = [] {
  executorch::runtime::register_device_allocator(&XpuAllocator::instance());
  return true;
}();
} // namespace

} // namespace executorch::backends::xpu
