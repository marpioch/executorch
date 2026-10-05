#include <executorch/backends/xpu/runtime/xpu_tensor_attribute.h>

namespace executorch::backends::xpu {

extern "C" {

int32_t aoti_torch_device_type_xpu() {
  return 12; // c10::DeviceType::XPU, see c10/core/DeviceType.h
}

int32_t aoti_torch_device_type_cuda() {
  return 1; // c10::DeviceType::CUDA; never actually matched by our tensors
}

AOTITorchError aoti_torch_get_device_type(
    Tensor* tensor,
    int32_t* ret_device_type) {
  (void)tensor;
  // Single-GPU PoC: every tensor this runtime creates/wraps is XPU.
  *ret_device_type = aoti_torch_device_type_xpu();
  return AOTITorchError::Ok;
}

} // extern "C"

} // namespace executorch::backends::xpu
