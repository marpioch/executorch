#pragma once

#include <cstdint>

#include <executorch/backends/aoti/aoti_delegate_handle.h>

namespace executorch::backends::xpu {

using executorch::backends::aoti::AOTInductorModelContainerDeleteFunc;
using executorch::backends::aoti::AOTInductorModelContainerGetNumInputsFunc;
using executorch::backends::aoti::AOTInductorModelContainerGetNumOutputsFunc;
using executorch::backends::aoti::AOTInductorModelContainerHandle;
using executorch::backends::aoti::AOTInductorModelContainerRunFunc;
using executorch::backends::aoti::AOTInductorModelUpdateConstantsFromBlobFunc;

// State carried between init() and execute()/destroy() for one delegated XPU
// method.
struct XpuDelegateHandle {
  void* lib_handle = nullptr;
  AOTInductorModelContainerHandle container_handle = nullptr;
  AOTInductorModelContainerDeleteFunc container_delete = nullptr;
  AOTInductorModelContainerGetNumInputsFunc get_num_inputs = nullptr;
  AOTInductorModelContainerGetNumOutputsFunc get_num_outputs = nullptr;
  AOTInductorModelContainerRunFunc run = nullptr;
  AOTInductorModelUpdateConstantsFromBlobFunc update_constants_from_blob = nullptr;
  int32_t device_index = 0;
};

} // namespace executorch::backends::xpu
