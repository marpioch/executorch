"""Minimal "Phi-3.5-mini-shaped" decoder block export for XpuBackend --
step 3 of the LLM-enablement plan: validate embedding/RMSNorm/RoPE/SwiGLU-MLP
(previously unvalidated beyond a 2-op Linear+ReLU model) together with the
already-validated GEMM/SDPA shims, in ONE static forward pass (no KV-cache
growth/statefulness yet -- that's a separate follow-up milestone).

Not the real Phi-3.5-mini-instruct checkpoint/config (no HF download, no
optimum-executorch involved) -- a small, randomly-initialized module with the
same architectural pieces (RMSNorm, rotary embeddings, GQA attention via
F.scaled_dot_product_attention with is_causal+enable_gqa combined for the
first time, SwiGLU MLP) so the compute pipeline can be validated without
needing real weights/tokenizer.
"""
import torch
import torch.nn.functional as F
from executorch.backends.xpu.xpu_backend import XpuBackend
from executorch.backends.xpu.xpu_partitioner import XpuPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

VOCAB_SIZE = 32
HIDDEN_SIZE = 64
N_LAYERS = 2
N_HEADS = 8
N_KV_HEADS = 4  # GQA, group_size == 2 (matches export_sdpa_gqa.py's ratio)
HEAD_DIM = HIDDEN_SIZE // N_HEADS
INTERMEDIATE_SIZE = 128
SEQ_LEN = 8


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class PhiBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = RMSNorm(HIDDEN_SIZE)
        self.q_proj = torch.nn.Linear(HIDDEN_SIZE, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = torch.nn.Linear(HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = torch.nn.Linear(HIDDEN_SIZE, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = torch.nn.Linear(N_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
        self.post_attn_norm = RMSNorm(HIDDEN_SIZE)
        self.gate_up_proj = torch.nn.Linear(
            HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE, bias=False
        )
        self.down_proj = torch.nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, x, cos, sin):
        batch, seq_len, _ = x.shape
        residual = x
        h = self.input_norm(x)
        q = self.q_proj(h).view(batch, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(h).view(batch, seq_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h).view(batch, seq_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, N_HEADS * HEAD_DIM)
        x = residual + self.o_proj(attn_out)

        residual = x
        h = self.post_attn_norm(x)
        gate, up = self.gate_up_proj(h).chunk(2, dim=-1)
        x = residual + self.down_proj(F.silu(gate) * up)
        return x


class PhiMiniLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = torch.nn.ModuleList([PhiBlock() for _ in range(N_LAYERS)])
        self.final_norm = RMSNorm(HIDDEN_SIZE)
        self.lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
        )
        t = torch.arange(SEQ_LEN).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, input_ids):
        x = self.embed(input_ids)
        cos = self.cos_cached
        sin = self.sin_cached
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.final_norm(x)
        return self.lm_head(x)


def main() -> None:
    torch.manual_seed(0)
    model = PhiMiniLike().eval()
    # executor_runner fills inputs with all-ones at runtime regardless of the
    # values used here for tracing -- so use the same all-ones (token id 1)
    # input for the eager reference, matching the convention established in
    # export_sdpa.py. RoPE + causal masking still differentiate the output
    # per position despite the uniform token id.
    example_inputs = (torch.ones(1, SEQ_LEN, dtype=torch.long),)

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

    save_pte_program(et_program, "phi_block_xpu", ".")
    print("Wrote phi_block_xpu.pte")


if __name__ == "__main__":
    main()
