"""Export + dump reference data for the KV-cache cross-call persistence test.

Companion to backends/xpu/test/kv_cache_persistence_runner.cpp, which is the
actual test driver: it loads the .pte this script exports and calls
Module::forward() CACHE_LEN separate times (one per simulated decode step),
confirming whether mutated k_cache/v_cache state persists across SEPARATE
execute() calls on real XPU hardware -- something executor_runner.exe cannot
test (it always re-feeds the same fixed input per --num_executions
iteration, with no way to vary input_pos) and something the Python
executorch.runtime bindings cannot currently test either (the pip-installable
`portable_lib` extension's build uses setup.py's "pybind" CMake preset, which
hardcodes the Visual Studio/ClangCL toolset -- incompatible with our XPU
backend's hard requirement on the Intel oneAPI DPC++ compiler for -fsycl).

This script only does the (already well-covered) Python-side half: export the
model, compute the eager step-by-step reference for a fixed, non-random,
exactly-reproducible-in-C++ per-position input pattern, and write both the
inputs and the reference output to a plain-text file the C++ runner reads
back verbatim -- avoiding any need to match PyTorch's RNG in C++.
"""
import torch
import torch.nn.functional as F
from executorch.backends.xpu.xpu_backend import XpuBackend
from executorch.backends.xpu.xpu_partitioner import XpuPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

N_HEADS = 8
N_KV_HEADS = 4  # GQA, group_size == 2 (matches export_sdpa_gqa.py's ratio)
HEAD_DIM = 8
HIDDEN_SIZE = N_HEADS * HEAD_DIM
CACHE_LEN = 4


class KVCacheStepModule(torch.nn.Module):
    """Same architecture as export_kv_cache.py's KVCacheAttnModule, but
    processes ONE token per forward() call (the real incremental-decode
    shape) instead of unrolling all CACHE_LEN steps internally.
    """

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(HIDDEN_SIZE, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = torch.nn.Linear(HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = torch.nn.Linear(HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = torch.nn.Linear(N_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.register_buffer(
            "k_cache",
            torch.zeros(1, N_KV_HEADS, CACHE_LEN, HEAD_DIM),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(1, N_KV_HEADS, CACHE_LEN, HEAD_DIM),
            persistent=False,
        )

    def forward(self, x, input_pos):
        # x: [1, 1, HIDDEN_SIZE], input_pos: [1] int64 (position to write/attend up to)
        q = self.q_proj(x).view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(1, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(1, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        self.k_cache.index_copy_(2, input_pos, k)
        self.v_cache.index_copy_(2, input_pos, v)

        # Data-dependent mask (pos comes from the input_pos tensor, not a
        # static Python int), so ONE compiled graph is correct for any step.
        idx = torch.arange(CACHE_LEN)
        pos = input_pos[0]
        mask = torch.where(idx <= pos, 0.0, float("-inf")).view(1, 1, 1, CACHE_LEN)

        attn_out = F.scaled_dot_product_attention(
            q, self.k_cache, self.v_cache, attn_mask=mask, enable_gqa=True
        )
        attn_out = attn_out.transpose(1, 2).reshape(1, 1, N_HEADS * HEAD_DIM)
        return self.o_proj(attn_out)


def build_model() -> KVCacheStepModule:
    # Reset the RNG before construction so every call produces a model with
    # IDENTICAL weights (buffers always start at zero regardless) -- used to
    # get two independent instances (one for eager stepping, one fresh for
    # export) that are otherwise equivalent.
    torch.manual_seed(0)
    return KVCacheStepModule().eval()


def make_tokens() -> torch.Tensor:
    # Fixed, non-random, trivially reproducible in C++: token value at
    # position `pos` is the scalar (pos + 1) * 0.1, broadcast across the
    # whole hidden dim. Distinguishable per position (unlike an all-ones
    # input, see export_sdpa.py's note), with no RNG to keep in sync across
    # languages.
    scale = (torch.arange(CACHE_LEN, dtype=torch.float32) + 1.0) * 0.1
    return scale.view(1, CACHE_LEN, 1).expand(1, CACHE_LEN, HIDDEN_SIZE).contiguous()


def eager_reference(tokens: torch.Tensor) -> torch.Tensor:
    model = build_model()
    outputs = []
    with torch.no_grad():
        for pos in range(CACHE_LEN):
            out = model(tokens[:, pos : pos + 1, :], torch.tensor([pos]))
            outputs.append(out)
    return torch.cat(outputs, dim=1)


def export_model() -> str:
    model = build_model()
    example_inputs = (
        torch.ones(1, 1, HIDDEN_SIZE),
        torch.tensor([0], dtype=torch.long),
    )
    exported_program = torch.export.export(model, example_inputs)

    partitioner = XpuPartitioner(
        [XpuBackend.generate_method_name_compile_spec("forward")]
    )
    edge_compile_config = EdgeCompileConfig(
        _check_ir_validity=False,
        _skip_dim_order=True,
    )
    et_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
        compile_config=edge_compile_config,
    ).to_executorch()

    save_pte_program(et_program, "kv_cache_step_xpu", ".")
    return "kv_cache_step_xpu.pte"


def write_reference_file(path: str, tokens: torch.Tensor, eager_output: torch.Tensor) -> None:
    with open(path, "w") as f:
        f.write(f"{CACHE_LEN} {HIDDEN_SIZE}\n")
        for pos in range(CACHE_LEN):
            values = tokens[0, pos, :].tolist()
            f.write(" ".join(f"{v:.8g}" for v in values) + "\n")
        flat = eager_output.reshape(-1).tolist()
        f.write(" ".join(f"{v:.8g}" for v in flat) + "\n")


def main() -> None:
    tokens = make_tokens()
    eager_output = eager_reference(tokens)
    print("Eager step-by-step reference output:")
    print(eager_output)

    pte_path = export_model()
    print(f"Wrote {pte_path}")

    ref_path = "kv_cache_step_xpu_reference.txt"
    write_reference_file(ref_path, tokens, eager_output)
    print(f"Wrote {ref_path}")
    print(
        "Next: build and run kv_cache_persistence_runner "
        f"--model_path={pte_path} --reference_path={ref_path}"
    )


if __name__ == "__main__":
    main()
