"""Static KV-cache export for XpuBackend -- first test of mutable-buffer
state (register_buffer + index_copy_), the last major unvalidated piece
before real incremental decode. No xpu_sdpa.cpp/xpu_gemm.cpp changes needed:
this deliberately uses an explicit position-dependent attn_mask (built by the
model itself) instead of is_causal, so it only exercises code paths already
validated (mm, SDPA via attn_bias+enable_gqa).

Isolates ONLY the new variable (mutable buffer state + growing/masked
attention) from everything else: no embedding/RMSNorm/RoPE/MLP here (those
were separately validated by export_phi_block.py).

Scope note: this unrolls CACHE_LEN sequential "decode steps" inside ONE
forward() call / one static trace (each step mutates the shared k_cache/
v_cache buffer and attends over the positions filled so far), rather than
driving CACHE_LEN separate executor_runner invocations with different
input_pos each time -- executor_runner always re-feeds the same fixed
(all-ones) input per run, so it can't vary input_pos across calls on its own.
This validates the buffer-mutation + growing-attention mechanism itself;
whether state ALSO persists correctly across genuinely separate execute()
calls (real incremental serving) is a follow-up, best tested via
executorch.runtime's Python bindings calling the same loaded method twice,
not executor_runner.
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


class KVCacheAttnModule(torch.nn.Module):
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
        # executor_runner always feeds the same all-ones input (see
        # export_sdpa.py) -- add a fixed seeded per-position offset so each
        # of the CACHE_LEN steps actually sees a distinguishable token,
        # otherwise every cache entry would be identical and this test would
        # only catch gross masking bugs (extra zero-vectors averaged in),
        # not subtler ones.
        generator = torch.Generator().manual_seed(0)
        self.register_buffer(
            "x_offset", torch.randn(1, CACHE_LEN, HIDDEN_SIZE, generator=generator)
        )

    def forward(self, x):
        x = x + self.x_offset
        outputs = []
        for pos in range(CACHE_LEN):
            token = x[:, pos : pos + 1, :]
            q = self.q_proj(token).view(1, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
            k = self.k_proj(token).view(1, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = self.v_proj(token).view(1, 1, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
            index = torch.tensor([pos])
            self.k_cache.index_copy_(2, index, k)
            self.v_cache.index_copy_(2, index, v)

            mask = torch.zeros(1, 1, 1, CACHE_LEN)
            if pos + 1 < CACHE_LEN:
                mask[:, :, :, pos + 1 :] = float("-inf")

            attn_out = F.scaled_dot_product_attention(
                q, self.k_cache, self.v_cache, attn_mask=mask, enable_gqa=True
            )
            attn_out = attn_out.transpose(1, 2).reshape(1, 1, N_HEADS * HEAD_DIM)
            outputs.append(self.o_proj(attn_out))
        return torch.cat(outputs, dim=1)


def main() -> None:
    torch.manual_seed(0)
    model = KVCacheAttnModule().eval()
    example_inputs = (torch.ones(1, CACHE_LEN, HIDDEN_SIZE),)

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

    save_pte_program(et_program, "kv_cache_xpu", ".")
    print("Wrote kv_cache_xpu.pte")


if __name__ == "__main__":
    main()
