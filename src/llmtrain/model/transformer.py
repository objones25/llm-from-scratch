import torch
from torch import nn
from torch.nn import functional as F

from llmtrain.model.cache import KVCache
from llmtrain.training.config import ModelConfig


def _rotary_cos_sin(
    seq_len: int,
    head_dim: int,
    theta: float,
    position_offset: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(
        position_offset, position_offset + seq_len, device=device, dtype=torch.float32
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary position embeddings")
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim)
        self.kv_proj = nn.Linear(config.d_model, 2 * config.n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x)
        k, v = kv.split(self.n_kv_heads * self.head_dim, dim=2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = _rotary_cos_sin(
            seq_len, self.head_dim, self.rope_theta, position_offset, x.device, x.dtype
        )
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer_idx, k, v)
        # SDPA's is_causal builds a TOP-LEFT-aligned causal bias for non-square (q_len != k_len)
        # masks, not the bottom-right alignment cached decode needs. Square cases (training,
        # uncached forward, prefill into an empty cache) are unaffected. Single-token decode
        # needs no mask at all — every cached key is a past position.
        if seq_len > 1 and cache is not None and k.shape[2] != seq_len:
            raise ValueError("multi-token queries against a non-empty cache are not supported")
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(seq_len > 1),
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d_ff = int(2 / 3 * 4 * config.d_model)
        self.w_gate = nn.Linear(config.d_model, d_ff)
        self.w_up = nn.Linear(config.d_model, d_ff)
        self.w_down = nn.Linear(d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln1(x), position_offset=position_offset, cache=cache, layer_idx=layer_idx
        )
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = self.token_emb(input_ids)
        position_offset = cache.seq_len if cache is not None else 0
        for layer_idx, block in enumerate(self.blocks):
            x = block(x, position_offset=position_offset, cache=cache, layer_idx=layer_idx)
        x = self.ln_f(x)
        return self.head(x)
