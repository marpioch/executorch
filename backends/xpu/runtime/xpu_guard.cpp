/*
 * See xpu_guard.h. Deliberately simple: a thread-local "current device" plus
 * a process-wide, lazily-created default sycl::queue per device index, with a
 * thread-local override (installed by the stream guard) that takes priority.
 * Current-device state and the device pool/context themselves live in
 * xpu_functions.cpp (c10::xpu::*), so this file, aoti_torch_get_current_sycl_queue,
 * and the Triton-XPU kernel-launch code calling c10::xpu::* directly all agree
 * on the same device/context.
 */
#include <executorch/backends/xpu/runtime/xpu_guard.h>
#include <executorch/backends/xpu/runtime/xpu_functions.h>

#include <executorch/runtime/platform/assert.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace executorch::backends::xpu {

namespace {

thread_local std::unordered_map<int32_t, sycl::queue*> g_stream_override;

std::mutex g_default_queues_mutex;
std::unordered_map<int32_t, sycl::queue> g_default_queues;

sycl::queue& default_queue_for(int32_t device_index) {
  std::lock_guard<std::mutex> lock(g_default_queues_mutex);
  auto it = g_default_queues.find(device_index);
  if (it != g_default_queues.end()) {
    return it->second;
  }
  auto [new_it, _] = g_default_queues.emplace(
      device_index,
      sycl::queue(
          c10::xpu::get_device_context(),
          c10::xpu::get_raw_device(device_index)));
  return new_it->second;
}

struct XPUGuardOpaqueImpl {
  int32_t original_device;
};

struct XPUStreamGuardOpaqueImpl {
  int32_t device_index;
  sycl::queue* original_stream; // nullptr if none was overridden
};

} // namespace

sycl::queue* get_or_create_xpu_queue(int32_t device_index) {
  if (device_index < 0) {
    device_index = c10::xpu::current_device();
  }
  auto override_it = g_stream_override.find(device_index);
  if (override_it != g_stream_override.end()) {
    return override_it->second;
  }
  return &default_queue_for(device_index);
}

extern "C" {

AOTITorchError aoti_torch_create_xpu_guard(
    int32_t device_index,
    XPUGuardHandle* ret_guard) {
  if (ret_guard == nullptr) {
    return Error::InvalidArgument;
  }
  auto* impl = new XPUGuardOpaqueImpl{c10::xpu::current_device()};
  c10::xpu::set_current_device(device_index);
  *ret_guard = reinterpret_cast<XPUGuardHandle>(impl);
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_xpu_guard(XPUGuardHandle guard) {
  if (guard == nullptr) {
    return Error::InvalidArgument;
  }
  auto* impl = reinterpret_cast<XPUGuardOpaqueImpl*>(guard);
  c10::xpu::set_current_device(impl->original_device);
  delete impl;
  return Error::Ok;
}

AOTITorchError aoti_torch_xpu_guard_set_index(
    XPUGuardHandle guard,
    int32_t device_index) {
  if (guard == nullptr) {
    return Error::InvalidArgument;
  }
  c10::xpu::set_current_device(device_index);
  return Error::Ok;
}

AOTITorchError aoti_torch_create_xpu_stream_guard(
    void* stream,
    int32_t device_index,
    XPUStreamGuardHandle* ret_guard) {
  if (ret_guard == nullptr || stream == nullptr) {
    return Error::InvalidArgument;
  }
  auto it = g_stream_override.find(device_index);
  sycl::queue* original =
      it != g_stream_override.end() ? it->second : nullptr;
  g_stream_override[device_index] = static_cast<sycl::queue*>(stream);
  *ret_guard = reinterpret_cast<XPUStreamGuardHandle>(
      new XPUStreamGuardOpaqueImpl{device_index, original});
  return Error::Ok;
}

AOTITorchError aoti_torch_delete_xpu_stream_guard(
    XPUStreamGuardHandle guard) {
  if (guard == nullptr) {
    return Error::InvalidArgument;
  }
  auto* impl = reinterpret_cast<XPUStreamGuardOpaqueImpl*>(guard);
  if (impl->original_stream == nullptr) {
    g_stream_override.erase(impl->device_index);
  } else {
    g_stream_override[impl->device_index] = impl->original_stream;
  }
  delete impl;
  return Error::Ok;
}

AOTITorchError aoti_torch_get_current_xpu_stream(
    int32_t device_index,
    void** ret_stream) {
  if (ret_stream == nullptr) {
    return Error::InvalidArgument;
  }
  *ret_stream = static_cast<void*>(get_or_create_xpu_queue(device_index));
  return Error::Ok;
}

AOTITorchError aoti_torch_get_current_xpu_device(int32_t* device_index) {
  if (device_index == nullptr) {
    return Error::InvalidArgument;
  }
  *device_index = c10::xpu::current_device();
  return Error::Ok;
}

AOTITorchError aoti_torch_set_current_xpu_device(const int32_t& device_index) {
  c10::xpu::set_current_device(device_index);
  return Error::Ok;
}

AOTITorchError aoti_torch_get_current_sycl_queue(void** ret) {
  if (ret == nullptr) {
    return Error::InvalidArgument;
  }
  *ret =
      static_cast<void*>(get_or_create_xpu_queue(c10::xpu::current_device()));
  return Error::Ok;
}

} // extern "C"

} // namespace executorch::backends::xpu
