"""Scaled-down, REAL-shaped Phi-3.5-mini-instruct export -- de-risking step
before attempting the full 32-layer model.

Uses the REAL architecture hyperparameters and REAL LongRoPE short_factor
table from the bundled reference config
(examples/mediatek/models/llm_models/weights/phi3.5-mini-instruct/config.json):
hidden_size=3072, num_attention_heads=num_key_value_heads=32 (MHA, not GQA),
head_dim=96, intermediate_size=8192, vocab_size=32064, rms_norm_eps=1e-5,
rope_theta=10000.0. Only num_hidden_layers is reduced (32 -> 2) to validate
compile time/memory/correctness at real per-layer dimensions before scaling
to the full depth. Random weight init (no HF download yet) -- this is a
structure/scale test, not a text-generation test.

Single prefill-style forward pass over a short prompt (no KV-cache here --
that statefulness was already validated separately in export_kv_cache_step.py
at tiny dims; combining real dims + KV-cache is a later step).
"""
import torch
import torch.nn.functional as F
from executorch.backends.xpu.xpu_backend import XpuBackend
from executorch.backends.xpu.xpu_partitioner import XpuPartitioner
from executorch.examples.models.llama.rope import (
    hf_apply_rotary_emb,
    hf_precompute_freqs_cis,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.extension.export_util.utils import save_pte_program

# Real Phi-3.5-mini-instruct hyperparameters (see config.json referenced above).
VOCAB_SIZE = 32064
HIDDEN_SIZE = 3072
N_HEADS = 32
N_KV_HEADS = 32  # MHA: num_key_value_heads == num_attention_heads
HEAD_DIM = HIDDEN_SIZE // N_HEADS  # 96
INTERMEDIATE_SIZE = 8192
RMS_NORM_EPS = 1e-5
ROPE_THETA = 10000.0
ORIGINAL_MAX_POS = 4096
MAX_POS = 4096

# Scaled down from the real 32 for this de-risking step.
N_LAYERS = 2
SEQ_LEN = 16

# Real LongRoPE short_factor table (dim/2 == HEAD_DIM/2 == 48 entries), copied
# verbatim from config.json. long_factor unused here since SEQ_LEN << 4096
# always selects short_factor.
SHORT_FACTOR = [
    1.0, 1.0199999809265137, 1.0299999713897705, 1.0299999713897705,
    1.0499999523162842, 1.0499999523162842, 1.0499999523162842,
    1.0499999523162842, 1.0499999523162842, 1.0699999332427979,
    1.0999999046325684, 1.1099998950958252, 1.1599998474121094,
    1.1599998474121094, 1.1699998378753662, 1.2899998426437378,
    1.339999794960022, 1.679999828338623, 1.7899998426437378,
    1.8199998140335083, 1.8499997854232788, 1.8799997568130493,
    1.9099997282028198, 1.9399996995925903, 1.9899996519088745,
    2.0199997425079346, 2.0199997425079346, 2.0199997425079346,
    2.0199997425079346, 2.0199997425079346, 2.0199997425079346,
    2.0299997329711914, 2.0299997329711914, 2.0299997329711914,
    2.0299997329711914, 2.0299997329711914, 2.0299997329711914,
    2.0299997329711914, 2.0299997329711914, 2.0299997329711914,
    2.0799996852874756, 2.0899996757507324, 2.189999580383301,
    2.2199995517730713, 2.5899994373321533, 2.729999542236328,
    2.749999523162842, 2.8399994373321533,
]


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=RMS_NORM_EPS):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


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
        q, k = hf_apply_rotary_emb(q, k, cos, sin, unsqueeze_dim=1)
        # MHA (num_heads == num_kv_heads): same plain-causal path already
        # validated in export_sdpa_causal.py, just at real dimensions.
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, N_HEADS * HEAD_DIM)
        x = residual + self.o_proj(attn_out)

        residual = x
        h = self.post_attn_norm(x)
        gate, up = self.gate_up_proj(h).chunk(2, dim=-1)
        x = residual + self.down_proj(F.silu(gate) * up)
        return x


class Phi35RealShape(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = torch.nn.ModuleList([PhiBlock() for _ in range(N_LAYERS)])
        self.final_norm = RMSNorm(HIDDEN_SIZE)
        self.lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

        freqs_cos, freqs_sin = hf_precompute_freqs_cis(
            dim=HEAD_DIM,
            end=SEQ_LEN,
            theta=ROPE_THETA,
            short_factor=SHORT_FACTOR,
            long_factor=SHORT_FACTOR,  # unused: SEQ_LEN << ORIGINAL_MAX_POS
            original_max_pos=ORIGINAL_MAX_POS,
            max_position_embeddings=MAX_POS,
        )
        # Add a leading batch dim so hf_apply_rotary_emb's unsqueeze_dim=1
        # broadcasts correctly against q/k shaped [batch, heads, seq, head_dim].
        self.register_buffer("cos_cached", freqs_cos.unsqueeze(0))
        self.register_buffer("sin_cached", freqs_sin.unsqueeze(0))

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
    model = Phi35RealShape().eval()
    # executor_runner fills inputs with all-ones at runtime regardless of the
    # values used here for tracing (see export_phi_block.py).
    example_inputs = (torch.ones(1, SEQ_LEN, dtype=torch.long),)

    with torch.no_grad():
        expected = model(*example_inputs)
    print("Eager reference output shape:", expected.shape)
    print("Eager reference output (first row, first 8 logits):")
    print(expected[0, 0, :8])

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

    save_pte_program(et_program, "phi35_real_shape_xpu", ".")
    print("Wrote phi35_real_shape_xpu.pte")


if __name__ == "__main__":
    main()
