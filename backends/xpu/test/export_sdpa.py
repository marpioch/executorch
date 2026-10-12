"""Minimal SDPA export for XpuBackend (oneDNN Graph-backed attention). Run
this on the PTL (XPU-capable) machine. Plain multi-head attention: no causal
mask, no GQA (num_heads_q == num_heads_kv), no attn_mask -- the simplest case
to validate aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable's
basic plumbing before testing causal/GQA variants.

executor_runner always fills inputs with all-ones at runtime (export only
fixes shapes/dtypes, not data), so this module adds fixed, seeded per-element
offsets to the (all-ones) input before SDPA -- that gives genuinely
distinguishable, non-uniform Q/K/V without depending on executor_runner's
input values. The eager reference output (computed here on the same all-ones
input) is printed so it can be diffed against executor_runner's output.
"""
import torch
from executorch.backends.xpu.xpu_backend import XpuBackend
from executorch.backends.xpu.xpu_partitioner import XpuPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

BATCH = 2
NUM_HEADS = 4
SEQ_LEN = 6
HEAD_DIM = 8


class SDPAModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        shape = (BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        self.q_offset = torch.nn.Parameter(torch.randn(shape, generator=generator))
        self.k_offset = torch.nn.Parameter(torch.randn(shape, generator=generator))
        self.v_offset = torch.nn.Parameter(torch.randn(shape, generator=generator))

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q + self.q_offset, k + self.k_offset, v + self.v_offset
        )


def main() -> None:
    model = SDPAModule().eval()
    example_inputs = (
        torch.ones(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        torch.ones(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        torch.ones(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM),
    )

    with torch.no_grad():
        expected = model(*example_inputs)
    print("Eager reference output (compare against executor_runner's):")
    print(expected)

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

    save_pte_program(et_program, "sdpa_xpu", ".")
    print("Wrote sdpa_xpu.pte")


if __name__ == "__main__":
    main()
