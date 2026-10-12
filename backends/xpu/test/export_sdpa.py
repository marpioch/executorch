"""Minimal SDPA export for XpuBackend (oneDNN Graph-backed attention). Run
this on the PTL (XPU-capable) machine. Plain multi-head attention: no causal
mask, no GQA (num_heads_q == num_heads_kv), no attn_mask -- the simplest case
to validate aoti_torch_xpu__scaled_dot_product_fused_attention_overrideable's
basic plumbing before testing causal/GQA variants.
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
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def main() -> None:
    model = SDPAModule().eval()
    example_inputs = (
        torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM),
        torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM),
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

    save_pte_program(et_program, "sdpa_xpu", ".")
    print("Wrote sdpa_xpu.pte")


if __name__ == "__main__":
    main()
