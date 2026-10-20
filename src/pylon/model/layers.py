from collections.abc import Sequence
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from pylon.kv.dense import BatchedKVCache, DecodeKVCache, KVCache
from pylon.model.config import Qwen3Config
from pylon.kv.cache import PagedBatchCache


class FeedForward(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        # Gate and up projections share one GEMM; the checkpoint loads into the halves.
        self.gate_up = nn.Linear(
            config.hidden_size, 2 * config.hidden_dim, bias=False, dtype=config.dtype
        )
        self.down = nn.Linear(
            config.hidden_dim, config.hidden_size, bias=False, dtype=config.dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).split(self.hidden_dim, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * up)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.rms_norm(
            x.float(), (x.shape[-1],), self.scale, self.eps
        )
        return normalized.to(x.dtype)


def rope_parameters(config: Qwen3Config) -> tuple[torch.Tensor, torch.Tensor]:
    inverse_frequencies = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
    )
    positions = torch.arange(config.context_length, dtype=torch.float32)
    angles = positions.unsqueeze(1) * inverse_frequencies.unsqueeze(0)
    angles = torch.cat((angles, angles), dim=1)
    return torch.cos(angles), torch.sin(angles)


def rope_for_positions(
    cos: torch.Tensor,
    sin: torch.Tensor,
    tokens: int,
    dtype: torch.dtype,
    *,
    start_pos: int = 0,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather the rotary tables once per forward, shaped to broadcast over heads."""
    if position_ids is None:
        cos = cos[start_pos : start_pos + tokens].unsqueeze(0).unsqueeze(0)
        sin = sin[start_pos : start_pos + tokens].unsqueeze(0).unsqueeze(0)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    return cos.to(dtype), sin.to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("Qwen3 RoPE requires an even head dimension.")
    first, second = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    return x * cos + torch.cat((-second, first), dim=-1) * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        if config.n_heads % config.n_kv_heads:
            raise ValueError(
                "The number of Q heads must be divisible by the number of KV heads."
            )
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.group_size = config.n_heads // config.n_kv_heads
        self.head_dim = config.head_dim
        self.query_size = config.n_heads * config.head_dim
        self.kv_size = config.n_kv_heads * config.head_dim
        # Q, K, and V share one GEMM; the checkpoint loads into the three row ranges.
        self.qkv = nn.Linear(
            config.hidden_size,
            self.query_size + 2 * self.kv_size,
            bias=False,
            dtype=config.dtype,
        )
        self.output = nn.Linear(
            config.n_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            dtype=config.dtype,
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, torch.Tensor],
        *,
        is_causal: bool = False,
        cache: KVCache
        | BatchedKVCache
        | DecodeKVCache
        | PagedBatchCache
        | None = None,
        layer_index: int = 0,
        cache_slots: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, tokens, _ = x.shape
        queries, keys, values = self.qkv(x).split(
            (self.query_size, self.kv_size, self.kv_size), dim=-1
        )
        queries = queries.view(batch_size, tokens, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, tokens, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope
        queries = apply_rope(self.q_norm(queries), cos, sin)
        keys = apply_rope(self.k_norm(keys), cos, sin)
        if isinstance(cache, PagedBatchCache):
            context = cache.attend(layer_index, queries, keys, values)
            context = context.transpose(1, 2).reshape(batch_size, tokens, -1)
            return self.output(context)
        if isinstance(cache, KVCache):
            keys, values = cache.append(layer_index, keys, values)
        elif cache is not None:
            keys, values = cache.append(layer_index, keys, values, slots=cache_slots)
        force_flash = (
            mask is None
            and queries.is_cuda
            and torch.backends.cuda.is_flash_attention_available()
            and torch.cuda.get_device_capability(queries.device)[0] >= 8
        )
        kernel = (
            sdpa_kernel(SDPBackend.FLASH_ATTENTION)
            if force_flash
            else nullcontext()
        )
        with kernel:
            context = torch.nn.functional.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=is_causal,
                enable_gqa=self.group_size > 1,
            )
        context = context.transpose(1, 2).reshape(batch_size, tokens, -1)
        return self.output(context)


class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, torch.Tensor],
        *,
        is_causal: bool = False,
        cache: KVCache
        | BatchedKVCache
        | DecodeKVCache
        | PagedBatchCache
        | None = None,
        layer_index: int = 0,
        cache_slots: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.input_norm(x),
            mask,
            rope,
            is_causal=is_causal,
            cache=cache,
            layer_index=layer_index,
            cache_slots=cache_slots,
        )
        return x + self.feed_forward(self.post_attention_norm(x))
