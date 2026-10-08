#include <executorch/backends/xpu/runtime/xpu_memory.h>

#include <executorch/backends/aoti/utils.h>
#include <executorch/backends/xpu/runtime/xpu_functions.h>
#include <executorch/backends/xpu/runtime/xpu_tensor_attribute.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/log.h>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <unordered_map>

namespace executorch::backends::xpu {

using namespace executorch::backends::aoti;

namespace {

constexpr int32_t NOT_OWN = -1;

// Maps raw Tensor* -> shared_ptr<Tensor> for O(1) lookup/deletion, and
// tracks how many live tensors alias a given data pointer (mirrors
// backends/apple/metal/runtime/shims/memory.cpp's reference-counting scheme).
std::unordered_map<Tensor*, std::shared_ptr<Tensor>> tensors;
std::unordered_map<void*, int32_t> memory_to_n_tensor;
// Pointers this file allocated via sycl::malloc_shared (freed with
// sycl::free); anything not in this set was allocated with plain malloc.
std::unordered_map<void*, bool> memory_is_device_alloc;

AOTITorchError validate_dtype(int32_t dtype) {
  if (dtype_to_scalar_type(dtype) == executorch::aten::ScalarType::Undefined) {
    return Error::InvalidArgument;
  }
  return Error::Ok;
}

// Wraps `data` in a tensor whose strides are exactly the ones given; see
// backends/apple/metal/runtime/shims/memory.cpp's make_strided_tensor for why
// this (rather than plain for_blob().make_tensor_ptr()) is needed.
std::shared_ptr<Tensor> make_strided_tensor(
    void* data,
    std::vector<executorch::aten::SizesType> sizes,
    std::vector<executorch::aten::StridesType> strides,
    executorch::aten::ScalarType scalar_type) {
  std::vector<executorch::aten::DimOrderType> dim_order(sizes.size());
  std::iota(dim_order.begin(), dim_order.end(), 0);
  std::stable_sort(
      dim_order.begin(), dim_order.end(), [&](size_t a, size_t b) {
        if (strides[a] != strides[b]) {
          return strides[a] > strides[b];
        }
        return sizes[a] != 1 && sizes[b] == 1;
      });
  return executorch::extension::for_blob(data, std::move(sizes), scalar_type)
      .dim_order(std::move(dim_order))
      .strides(std::move(strides))
      .make_tensor_ptr();
}

} // namespace

extern "C" {

AOTITorchError aoti_torch_create_tensor_from_blob_v2(
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
    int64_t opaque_metadata_size) {
  (void)device_type;
  (void)device_index;
  (void)layout;
  (void)opaque_metadata;
  (void)opaque_metadata_size;

  ET_CHECK_OR_RETURN_ERROR(
      data != nullptr, InvalidArgument, "data pointer is null");
  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "sizes_ptr is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr, InvalidArgument, "ret_new_tensor is null");
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  void* adjusted_data = static_cast<char*>(data) +
      storage_offset * dtype_to_element_size(dtype);

  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);
  auto tensor = make_strided_tensor(
      adjusted_data, sizes, strides, dtype_to_scalar_type(dtype));
  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr, InvalidArgument, "Failed to create tensor from blob");

  tensors[tensor.get()] = tensor;
  *ret_new_tensor = tensor.get();
  memory_to_n_tensor[adjusted_data] = NOT_OWN;
  return Error::Ok;
}

AOTITorchError aoti_torch_empty_strided(
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int32_t dtype,
    int32_t device_type,
    int32_t device_index,
    Tensor** ret_new_tensor) {
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));
  size_t element_size = dtype_to_element_size(dtype);
  ET_CHECK_OR_RETURN_ERROR(
      element_size != 0,
      InvalidArgument,
      "Invalid element size for dtype: %d",
      dtype);

  int64_t numel = 1;
  for (int64_t i = 0; i < ndim; i++) {
    numel *= sizes_ptr[i];
  }
  int64_t nbytes = numel * static_cast<int64_t>(element_size);

  bool is_device_alloc = (device_type == aoti_torch_device_type_xpu());
  void* ptr = nullptr;
  if (is_device_alloc) {
    int32_t idx =
        device_index >= 0 ? device_index : c10::xpu::current_device();
    ptr = sycl::malloc_shared(
        static_cast<size_t>(nbytes),
        c10::xpu::get_raw_device(idx),
        c10::xpu::get_device_context());
    ET_CHECK_OR_RETURN_ERROR(
        ptr != nullptr,
        MemoryAllocationFailed,
        "Failed to allocate %lld bytes of XPU shared USM memory",
        nbytes);
  } else {
    ptr = malloc(static_cast<size_t>(nbytes));
    ET_CHECK_OR_RETURN_ERROR(
        ptr != nullptr, MemoryAllocationFailed, "Failed to allocate memory");
  }

  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);
  auto tensor =
      make_strided_tensor(ptr, sizes, strides, dtype_to_scalar_type(dtype));

  tensors[tensor.get()] = tensor;
  *ret_new_tensor = tensor.get();
  memory_to_n_tensor[ptr] = 1;
  memory_is_device_alloc[ptr] = is_device_alloc;
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_tensor_object(Tensor* tensor) {
  if (tensor == nullptr) {
    return Error::Ok;
  }
  auto it = tensors.find(tensor);
  ET_CHECK_OR_RETURN_ERROR(
      it != tensors.end(), InvalidArgument, "Didn't find tensor %p", tensor);

  void* data_ptr = it->second->mutable_data_ptr();
  auto memory_it = memory_to_n_tensor.find(data_ptr);
  if (memory_it != memory_to_n_tensor.end()) {
    int32_t ref_count = memory_it->second;
    if (ref_count == NOT_OWN) {
      // Tensor never owned this memory (created via create_tensor_from_blob).
    } else if (ref_count == 1) {
      auto dev_it = memory_is_device_alloc.find(data_ptr);
      if (dev_it != memory_is_device_alloc.end() && dev_it->second) {
        sycl::free(data_ptr, c10::xpu::get_device_context());
      } else {
        free(data_ptr);
      }
      memory_is_device_alloc.erase(data_ptr);
      memory_to_n_tensor.erase(memory_it);
    } else {
      memory_to_n_tensor[data_ptr] = ref_count - 1;
    }
  }
  tensors.erase(it);
  return Error::Ok;
}

AOTITorchError aoti_torch_new_tensor_handle(
    Tensor* orig_handle,
    Tensor** new_handle) {
  ET_CHECK_OR_RETURN_ERROR(
      orig_handle != nullptr && new_handle != nullptr,
      InvalidArgument,
      "null handle");

  int64_t* sizes_ptr;
  int64_t* strides_ptr;
  int32_t dtype;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_sizes(orig_handle, &sizes_ptr));
  ET_CHECK_OK_OR_RETURN_ERROR(
      aoti_torch_get_strides(orig_handle, &strides_ptr));
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_dtype(orig_handle, &dtype));

  int64_t ndim = orig_handle->dim();
  void* data_ptr = orig_handle->mutable_data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "Source tensor has null data pointer");

  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked",
      data_ptr);

  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);
  auto tensor = make_strided_tensor(
      data_ptr, sizes, strides, dtype_to_scalar_type(dtype));
  tensors[tensor.get()] = tensor;
  *new_handle = tensor.get();

  if (memory_it->second != NOT_OWN) {
    memory_it->second += 1;
  }
  return Error::Ok;
}

AOTITorchError aoti_torch__reinterpret_tensor(
    Tensor* self,
    int64_t ndim,
    const int64_t* sizes_ptr,
    const int64_t* strides_ptr,
    int64_t storage_offset,
    Tensor** ret_new_tensor) {
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor: self is null");
  ET_CHECK_OR_RETURN_ERROR(
      ret_new_tensor != nullptr,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor: ret_new_tensor is null");
  ET_CHECK_OR_RETURN_ERROR(
      ndim >= 0,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor: ndim must be >= 0, got %lld",
      static_cast<long long>(ndim));
  ET_CHECK_OR_RETURN_ERROR(
      !(sizes_ptr == nullptr && ndim > 0),
      InvalidArgument,
      "aoti_torch__reinterpret_tensor: sizes_ptr is null but ndim > 0");

  int32_t dtype;
  ET_CHECK_OK_OR_RETURN_ERROR(aoti_torch_get_dtype(self, &dtype));
  ET_CHECK_OK_OR_RETURN_ERROR(validate_dtype(dtype));

  void* data_ptr = self->mutable_data_ptr();
  ET_CHECK_OR_RETURN_ERROR(
      data_ptr != nullptr,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor: source tensor has null data pointer");

  auto memory_it = memory_to_n_tensor.find(data_ptr);
  ET_CHECK_OR_RETURN_ERROR(
      memory_it != memory_to_n_tensor.end(),
      InvalidArgument,
      "Memory address %p is not being tracked by reference counting system",
      data_ptr);

  size_t element_size = dtype_to_element_size(dtype);
  void* adjusted_data =
      static_cast<char*>(data_ptr) + (storage_offset * element_size);

  auto sizes = convert_sizes_to_vector(ndim, sizes_ptr);
  auto strides = convert_strides_to_vector(ndim, sizes_ptr, strides_ptr);

  // ETensor supports arbitrary strides directly (see make_strided_tensor
  // above), so unlike Metal's MTLBuffer-backed shim this never needs to
  // materialize non-packed views into a new contiguous buffer.
  auto tensor = make_strided_tensor(
      adjusted_data, sizes, strides, dtype_to_scalar_type(dtype));
  ET_CHECK_OR_RETURN_ERROR(
      tensor != nullptr,
      InvalidArgument,
      "aoti_torch__reinterpret_tensor: failed to create tensor view");

  tensors[tensor.get()] = tensor;
  *ret_new_tensor = tensor.get();

  if (memory_it->second != NOT_OWN) {
    memory_it->second += 1;
  }
  return Error::Ok;
}

AOTITorchError aoti_torch_copy_(Tensor* self, Tensor* src, int32_t non_blocking) {
  (void)non_blocking;
  ET_CHECK_OR_RETURN_ERROR(
      self != nullptr && src != nullptr, InvalidArgument, "null tensor");
  ET_CHECK_OR_RETURN_ERROR(
      self->nbytes() == src->nbytes(),
      InvalidArgument,
      "aoti_torch_copy_ size mismatch: %zu vs %zu bytes",
      self->nbytes(),
      src->nbytes());
  std::memcpy(self->mutable_data_ptr(), src->const_data_ptr(), src->nbytes());
  return Error::Ok;
}

} // extern "C"

} // namespace executorch::backends::xpu
