"""GQA (grouped-query attention) SDPA export for XpuBackend. Isolates
num_head_kv < num_head_q (the oneDNN reshape-and-broadcast code path in
xpu_sdpa.cpp: Q split into [group_num, group_size], K/V given a stride-0
"group_size" axis so MatMul broadcasts one K/V head across the group) on top
of the already-validated plain-SDPA path (see export_sdpa.py). Still
non-causal, no attn_mask -- one new variable at a time.
"""
import torch
from executorch.backends.xpu.xpu_backend import XpuBackend
from executorch.backends.xpu.xpu_partitioner import XpuPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

BATCH = 2
NUM_HEADS_Q = 4
NUM_HEADS_KV = 2  # group_size == 2
SEQ_LEN = 6
HEAD_DIM = 8


class SDPAGqaModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        q_shape = (BATCH, NUM_HEADS_Q, SEQ_LEN, HEAD_DIM)
        kv_shape = (BATCH, NUM_HEADS_KV, SEQ_LEN, HEAD_DIM)
        self.q_offset = torch.nn.Parameter(torch.randn(q_shape, generator=generator))
        self.k_offset = torch.nn.Parameter(torch.randn(kv_shape, generator=generator))
        self.v_offset = torch.nn.Parameter(torch.randn(kv_shape, generator=generator))

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q + self.q_offset,
            k + self.k_offset,
            v + self.v_offset,
            enable_gqa=True,
        )


def main() -> None:
    model = SDPAGqaModule().eval()
    example_inputs = (
        torch.ones(BATCH, NUM_HEADS_Q, SEQ_LEN, HEAD_DIM),
        torch.ones(BATCH, NUM_HEADS_KV, SEQ_LEN, HEAD_DIM),
        torch.ones(BATCH, NUM_HEADS_KV, SEQ_LEN, HEAD_DIM),
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

    save_pte_program(et_program, "sdpa_gqa_xpu", ".")
    print("Wrote sdpa_gqa_xpu.pte")


if __name__ == "__main__":
    main()
