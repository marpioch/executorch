/*
 * True cross-call KV-cache persistence test for XpuBackend. Companion to
 * export_kv_cache_step.py (exports the .pte + reference file this reads).
 *
 * export_kv_cache.py already validated buffer-mutation (index_copy_) +
 * growing-mask SDPA WITHIN a single trace/Run() call. This drives the SAME
 * loaded Module across CACHE_LEN SEPARATE Module::forward() calls (one per
 * simulated decode step), confirming whether mutated k_cache/v_cache state
 * actually persists across calls on real XPU hardware -- something
 * executor_runner.exe cannot test (it always re-feeds the same fixed input
 * per --num_executions iteration) and something the Python
 * executorch.runtime bindings cannot currently test either (the
 * pip-installable portable_lib extension's "pybind" CMake preset hardcodes
 * the Visual Studio/ClangCL toolset, incompatible with our XPU backend's
 * hard requirement on the Intel oneAPI DPC++ compiler for -fsycl).
 */
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <gflags/gflags.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

DEFINE_string(model_path, "kv_cache_step_xpu.pte", "Path to the exported .pte.");
DEFINE_string(
    reference_path,
    "kv_cache_step_xpu_reference.txt",
    "Path to the reference file written by export_kv_cache_step.py.");

using executorch::extension::make_tensor_ptr;
using executorch::extension::Module;
using executorch::runtime::EValue;

namespace {

struct ReferenceData {
  int cache_len;
  int hidden_size;
  std::vector<std::vector<float>> tokens; // [cache_len][hidden_size]
  std::vector<float> eager_output; // flattened [cache_len * hidden_size]
};

bool load_reference(const std::string& path, ReferenceData* out) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "Failed to open reference file: " << path << std::endl;
    return false;
  }
  f >> out->cache_len >> out->hidden_size;
  out->tokens.assign(out->cache_len, std::vector<float>(out->hidden_size));
  for (int pos = 0; pos < out->cache_len; pos++) {
    for (int i = 0; i < out->hidden_size; i++) {
      f >> out->tokens[pos][i];
    }
  }
  out->eager_output.assign(out->cache_len * out->hidden_size, 0.0f);
  for (float& v : out->eager_output) {
    f >> v;
  }
  return static_cast<bool>(f) || f.eof();
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ReferenceData ref;
  if (!load_reference(FLAGS_reference_path, &ref)) {
    return 1;
  }

  Module module(FLAGS_model_path);

  std::vector<float> et_output;
  et_output.reserve(ref.cache_len * ref.hidden_size);

  for (int pos = 0; pos < ref.cache_len; pos++) {
    auto token = make_tensor_ptr<float>({1, 1, ref.hidden_size}, ref.tokens[pos]);
    auto input_pos = make_tensor_ptr<int64_t>({1}, {static_cast<int64_t>(pos)});

    auto result = module.forward({EValue(token), EValue(input_pos)});
    if (!result.ok()) {
      std::cerr << "forward() failed at pos=" << pos << " (error "
                << static_cast<int>(result.error()) << ")" << std::endl;
      return 1;
    }
    const auto& outputs = result.get();
    if (outputs.empty() || !outputs[0].isTensor()) {
      std::cerr << "forward() returned no tensor output at pos=" << pos
                << std::endl;
      return 1;
    }
    const auto& out_tensor = outputs[0].toTensor();
    const float* data = out_tensor.const_data_ptr<float>();
    for (int i = 0; i < ref.hidden_size; i++) {
      et_output.push_back(data[i]);
    }
  }

  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < et_output.size(); i++) {
    max_abs_diff =
        std::max(max_abs_diff, std::abs(et_output[i] - ref.eager_output[i]));
  }

  std::cout << "Max abs diff vs eager step-by-step reference: " << max_abs_diff
            << std::endl;
  if (max_abs_diff < 1e-3f) {
    std::cout << "PASS: cross-call KV-cache state persisted correctly on XPU."
               << std::endl;
    return 0;
  }
  std::cout << "FAIL: ExecuTorch multi-call output diverged from eager reference."
            << std::endl;
  return 1;
}
