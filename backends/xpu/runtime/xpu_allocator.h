/*
 * DeviceAllocator for DeviceType::XPU, backed by SYCL USM shared allocations
 * (host+device accessible, safe for an integrated GPU with system-shared
 * memory). Registered as a singleton so Method::init()'s device-aware memory
 * planning can allocate the delegate's non-CPU-tagged planned buffers (see
 * runtime/core/device_allocator.h); mirrors backends/cuda/runtime/cuda_allocator.*.
 */
#pragma once

#include <executorch/runtime/core/device_allocator.h>

namespace executorch::backends::xpu {

class XpuAllocator : public executorch::runtime::DeviceAllocator {
 public:
  executorch::runtime::Result<void*> allocate(
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index,
      size_t alignment = kDefaultAlignment) override;

  void deallocate(void* ptr, executorch::runtime::etensor::DeviceIndex index)
      override;

  executorch::runtime::Error copy_host_to_device(
      void* dst,
      const void* src,
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index) override;

  executorch::runtime::Error copy_device_to_host(
      void* dst,
      const void* src,
      size_t nbytes,
      executorch::runtime::etensor::DeviceIndex index) override;

  executorch::runtime::etensor::DeviceType device_type() const override;

  static XpuAllocator& instance();
};

} // namespace executorch::backends::xpu
