"""Real Phi-3.5-mini-instruct WEIGHTS (not random init), still scaled down to
2 of the real 32 layers -- next de-risking step after
export_phi35_real_shape.py (real dims, random weights, PASSED on real HW).

Downloads microsoft/Phi-3.5-mini-instruct via transformers (public, no auth
needed) with num_hidden_layers truncated to N_LAYERS BEFORE construction, so
only the first N_LAYERS worth of weights are ever materialized (the rest
show up as harmless "unexpected keys" warnings from from_pretrained).

Our own hand-rolled module (qkv_proj/gate_up_proj combined Linears, matching
HF Phi3Attention/Phi3MLP's real weight layout exactly so loading is a direct
tensor copy, no splitting) is kept SEPARATE from the real HF model -- the
real (truncated) HF model is used only as an eager ORACLE to confirm our
minimal exportable module is architecturally exact with real weights, before
ever touching torch.export/XpuPartitioner. This avoids depending on whether
torch.export can handle the real HF model's internal cache/mask machinery.
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
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

# Real Phi-3.5-mini-instruct hyperparameters (see config.json).
HIDDEN_SIZE = 3072
N_HEADS = 32
N_KV_HEADS = 32  # MHA: num_key_value_heads == num_attention_heads
HEAD_DIM = HIDDEN_SIZE // N_HEADS  # 96
INTERMEDIATE_SIZE = 8192
VOCAB_SIZE = 32064
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
    """Mirrors real Phi3Attention/Phi3MLP's combined-Linear weight layout
    (qkv_proj, gate_up_proj) so real checkpoint tensors load via direct copy,
    no splitting/concatenation."""

    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(HIDDEN_SIZE)
        q_size = N_HEADS * HEAD_DIM
        kv_size = N_KV_HEADS * HEAD_DIM
        self.qkv_proj = torch.nn.Linear(HIDDEN_SIZE, q_size + 2 * kv_size, bias=False)
        self.o_proj = torch.nn.Linear(q_size, HIDDEN_SIZE, bias=False)
        self.post_attention_layernorm = RMSNorm(HIDDEN_SIZE)
        self.gate_up_proj = torch.nn.Linear(
            HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE, bias=False
        )
        self.down_proj = torch.nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, x, cos, sin):
        batch, seq_len, _ = x.shape
        q_size = N_HEADS * HEAD_DIM
        kv_size = N_KV_HEADS * HEAD_DIM
        residual = x
        h = self.input_layernorm(x)
        q, k, v = self.qkv_proj(h).split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(batch, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(batch, seq_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(batch, seq_len, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = hf_apply_rotary_emb(q, k, cos, sin, unsqueeze_dim=1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, q_size)
        x = residual + self.o_proj(attn_out)

        residual = x
        h = self.post_attention_layernorm(x)
        gate, up = self.gate_up_proj(h).chunk(2, dim=-1)
        x = residual + self.down_proj(F.silu(gate) * up)
        return x


class Phi35RealWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.layers = torch.nn.ModuleList([PhiBlock() for _ in range(N_LAYERS)])
        self.norm = RMSNorm(HIDDEN_SIZE)
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
        self.register_buffer("cos_cached", freqs_cos.unsqueeze(0))
        self.register_buffer("sin_cached", freqs_sin.unsqueeze(0))

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        cos, sin = self.cos_cached, self.sin_cached
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)


def load_real_weights(ours: Phi35RealWeights, hf_model) -> None:
    hf_sd = hf_model.state_dict()
    with torch.no_grad():
        ours.embed_tokens.weight.copy_(hf_sd["model.embed_tokens.weight"])
        ours.norm.weight.copy_(hf_sd["model.norm.weight"])
        ours.lm_head.weight.copy_(hf_sd["lm_head.weight"])
        for i, layer in enumerate(ours.layers):
            prefix = f"model.layers.{i}."
            layer.input_layernorm.weight.copy_(hf_sd[prefix + "input_layernorm.weight"])
            layer.post_attention_layernorm.weight.copy_(
                hf_sd[prefix + "post_attention_layernorm.weight"]
            )
            layer.qkv_proj.weight.copy_(hf_sd[prefix + "self_attn.qkv_proj.weight"])
            layer.o_proj.weight.copy_(hf_sd[prefix + "self_attn.o_proj.weight"])
            layer.gate_up_proj.weight.copy_(hf_sd[prefix + "mlp.gate_up_proj.weight"])
            layer.down_proj.weight.copy_(hf_sd[prefix + "mlp.down_proj.weight"])


def main() -> None:
    torch.manual_seed(0)

    print(f"Downloading/loading {MODEL_ID} (first {N_LAYERS} layers only)...")
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = N_LAYERS
    # Only N_LAYERS worth of decoder-layer weights get materialized; the
    # checkpoint's remaining layers show up as harmless "unexpected keys".
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, torch_dtype=torch.float32
    )
    hf_model.eval()

    ours = Phi35RealWeights().eval()
    load_real_weights(ours, hf_model)

    example_inputs = (torch.ones(1, SEQ_LEN, dtype=torch.long),)

    # Cross-check against the real (truncated) HF model BEFORE export, to
    # confirm our minimal hand-rolled module is architecturally exact with
    # real weights (independent of whether torch.export could handle the
    # real HF model's own internal cache/mask machinery).
    with torch.no_grad():
        hf_out = hf_model(example_inputs[0], use_cache=False).logits
        ours_out = ours(*example_inputs)
    max_diff = (hf_out - ours_out).abs().max().item()
    print(f"Max abs diff vs real (truncated) HF model eager output: {max_diff}")
    del hf_model

    print("Eager reference output shape:", ours_out.shape)
    print("Eager reference output (first row, first 8 logits):")
    print(ours_out[0, 0, :8])

    exported_program = torch.export.export(ours, example_inputs)

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

    save_pte_program(et_program, "phi35_real_weights_xpu", ".")
    print("Wrote phi35_real_weights_xpu.pte")


if __name__ == "__main__":
    main()
