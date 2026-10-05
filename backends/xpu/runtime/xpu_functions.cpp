/*
 * See xpu_functions.h. Device pool + shared context lazily created on first
 * use (thread-safe magic-static init), mirroring c10::xpu's own device pool
 * (all devices share one context; see Note [Device Management] in
 * c10/xpu/XPUFunctions.cpp). Deliberately simple: enumerates every SYCL GPU
 * device rather than replicating upstream's dGPU/iGPU platform-priority
 * logic, since this PoC only targets a single integrated GPU.
 */
#include <executorch/backends/xpu/runtime/xpu_functions.h>

#include <executorch/runtime/platform/assert.h>

#include <memory>
#include <vector>

namespace c10::xpu {

namespace {

thread_local DeviceIndex g_current_device = 0;

struct DevicePool {
  std::vector<std::unique_ptr<sycl::device>> devices;
  std::unique_ptr<sycl::context> context;
};

DevicePool init_device_pool() {
  DevicePool pool;
  auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
  ET_CHECK_MSG(!gpus.empty(), "No XPU (SYCL GPU) devices found");
  for (auto& device : gpus) {
    pool.devices.push_back(std::make_unique<sycl::device>(device));
  }
  pool.context = std::make_unique<sycl::context>(
      pool.devices[0]->get_platform().khr_get_default_context());
  return pool;
}

DevicePool& device_pool() {
  static DevicePool pool = init_device_pool();
  return pool;
}

} // namespace

DeviceIndex current_device() {
  return g_current_device;
}

void set_current_device(DeviceIndex device) {
  g_current_device = device;
}

sycl::device& get_raw_device(DeviceIndex device) {
  auto& pool = device_pool();
  ET_CHECK_MSG(
      device >= 0 && static_cast<size_t>(device) < pool.devices.size(),
      "No XPU device at index %d (found %zu)",
      static_cast<int>(device),
      pool.devices.size());
  return *pool.devices[device];
}

sycl::context& get_device_context() {
  return *device_pool().context;
}

} // namespace c10::xpu
