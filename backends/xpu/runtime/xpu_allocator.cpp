#include <executorch/backends/xpu/runtime/xpu_allocator.h>
#include <executorch/backends/xpu/runtime/xpu_functions.h>

#include <sycl/sycl.hpp>

#include <cstring>

namespace executorch::backends::xpu {

using executorch::runtime::Error;
using executorch::runtime::Result;
using executorch::runtime::etensor::DeviceIndex;
using executorch::runtime::etensor::DeviceType;

XpuAllocator& XpuAllocator::instance() {
  static XpuAllocator inst;
  return inst;
}

Result<void*> XpuAllocator::allocate(
    size_t nbytes,
    DeviceIndex index,
    size_t alignment) {
  (void)alignment; // SYCL USM allocations are already suitably aligned.
  int32_t idx = index >= 0 ? index : c10::xpu::current_device();
  void* ptr = sycl::malloc_shared(
      nbytes, c10::xpu::get_raw_device(idx), c10::xpu::get_device_context());
  if (ptr == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  return ptr;
}

void XpuAllocator::deallocate(void* ptr, DeviceIndex index) {
  (void)index;
  sycl::free(ptr, c10::xpu::get_device_context());
}

Error XpuAllocator::copy_host_to_device(
    void* dst,
    const void* src,
    size_t nbytes,
    DeviceIndex index) {
  (void)index;
  // USM "shared" memory is host-visible, so a plain memcpy is valid.
  std::memcpy(dst, src, nbytes);
  return Error::Ok;
}

Error XpuAllocator::copy_device_to_host(
    void* dst,
    const void* src,
    size_t nbytes,
    DeviceIndex index) {
  (void)index;
  std::memcpy(dst, src, nbytes);
  return Error::Ok;
}

DeviceType XpuAllocator::device_type() const {
  return DeviceType::XPU;
}

} // namespace executorch::backends::xpu
