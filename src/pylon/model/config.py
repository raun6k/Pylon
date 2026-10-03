import json
from dataclasses import dataclass
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

_REQUIRED_SHAPE = {
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "vocab_size": 151936,
}


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int = 151_936
    context_length: int = 262_144
    hidden_size: int = 2_560
    n_heads: int = 32
    n_layers: int = 36
    hidden_dim: int = 9_728
    head_dim: int = 128
    n_kv_heads: int = 8
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16


def read_model_config(path: Path, model_id: str) -> Qwen3Config:
    if model_id != MODEL_ID:
        raise ValueError(f"Pylon serves {MODEL_ID}; received {model_id}.")
    data = json.loads(Path(path).read_text())
    _reject_unless(data.get("model_type") == "qwen3", "model_type")
    _reject_unless(data.get("architectures") == ["Qwen3ForCausalLM"], "architectures")
    for key, expected in _REQUIRED_SHAPE.items():
        _reject_unless(data.get(key) == expected, key)
    _reject_unless(data.get("hidden_act") == "silu", "hidden_act")
    _reject_unless(data.get("attention_bias") is False, "attention_bias")
    _reject_unless(data.get("tie_word_embeddings") is True, "tie_word_embeddings")
    _reject_unless(data.get("rope_scaling") is None, "rope_scaling")
    _reject_unless(data.get("use_sliding_window") is False, "use_sliding_window")
    _reject_unless(data.get("sliding_window") is None, "sliding_window")
    _reject_unless(data.get("torch_dtype") == "bfloat16", "torch_dtype")
    return Qwen3Config(
        vocab_size=data["vocab_size"],
        context_length=data["max_position_embeddings"],
        hidden_size=data["hidden_size"],
        n_heads=data["num_attention_heads"],
        n_layers=data["num_hidden_layers"],
        hidden_dim=data["intermediate_size"],
        head_dim=data["head_dim"],
        n_kv_heads=data["num_key_value_heads"],
        rope_theta=float(data["rope_theta"]),
        rms_norm_eps=float(data["rms_norm_eps"]),
        dtype=torch.bfloat16,
    )


def _reject_unless(ok: bool, field: str) -> None:
    if not ok:
        raise ValueError(f"Checkpoint config field {field} is not supported.")
