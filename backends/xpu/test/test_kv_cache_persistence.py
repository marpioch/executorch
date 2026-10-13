"""True cross-call KV-cache persistence test for XpuBackend -- follow-up to
export_kv_cache.py, which only validated buffer-mutation + growing-mask SDPA
WITHIN a single trace/Run() call. This drives the SAME loaded method across
multiple SEPARATE execute() calls (one per simulated decode step) via
executorch.runtime's Python bindings, confirming whether mutated k_cache/
v_cache state actually persists across calls on real XPU hardware --
something executor_runner.exe cannot test (it always re-feeds the same fixed
input per --num_executions iteration, with no way to vary input_pos).
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


def main() -> None:
    tokens = torch.ones(1, CACHE_LEN, HIDDEN_SIZE)

    eager_output = eager_reference(tokens)
    print("Eager step-by-step reference output:")
    print(eager_output)

    pte_path = export_model()
    print(f"Wrote {pte_path}")

    from executorch.runtime import Runtime, Verification

    runtime = Runtime.get()
    program = runtime.load_program(pte_path, verification=Verification.Minimal)
    method = program.load_method("forward")

    et_outputs = []
    for pos in range(CACHE_LEN):
        token = tokens[:, pos : pos + 1, :]
        input_pos = torch.tensor([pos], dtype=torch.long)
        (out,) = method.execute([token, input_pos])
        et_outputs.append(out)
    et_output = torch.cat(et_outputs, dim=1)

    print("ExecuTorch (XPU, multi-call) output:")
    print(et_output)

    max_abs_diff = (et_output - eager_output).abs().max().item()
    print(f"Max abs diff vs eager step-by-step reference: {max_abs_diff:.6g}")
    if max_abs_diff < 1e-3:
        print("PASS: cross-call KV-cache state persisted correctly on XPU.")
    else:
        print("FAIL: ExecuTorch multi-call output diverged from eager reference.")


if __name__ == "__main__":
    main()
