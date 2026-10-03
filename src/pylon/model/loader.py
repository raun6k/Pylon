import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from pylon.config import PylonConfig
from pylon.kv.budget import CacheCapacity, MemoryChecker, MemoryReport
from pylon.model.compile import configure_compilation
from pylon.model.config import MODEL_ID, Qwen3Config, read_model_config
from pylon.model.model import Qwen3Model
from pylon.model.tokenizer import read_stop_token_ids
from pylon.model.weights import Qwen3Weights

logger = logging.getLogger("pylon")

_SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
]


@dataclass
class LoadedModel:
    model: Qwen3Model
    cache: CacheCapacity
    report: MemoryReport
    model_revision: str
    snapshot: Path
    stop_token_ids: tuple[int, ...]


class ModelLoader:
    def load(self, config: PylonConfig) -> LoadedModel:
        if config.model_id != MODEL_ID:
            raise ValueError(f"Pylon serves {MODEL_ID}; received {config.model_id}.")
        snapshot = Path(
            snapshot_download(
                repo_id=config.model_id,
                revision=config.model_revision,
                token=config.hf_token,
                allow_patterns=_SMALL_FILES,
            )
        )
        runtime = read_model_config(snapshot / "config.json", config.model_id)
        generation_path = snapshot / "generation_config.json"
        stop_ids = read_stop_token_ids(
            generation_path if generation_path.is_file() else None,
            snapshot / "config.json",
        )
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required.")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("This checkpoint requires bfloat16.")
        checker = MemoryChecker(config)
        report = checker.weights(runtime)
        if not report.fits:
            available_bytes = max(
                0,
                report.max_gpu_bytes
                - (report.total_bytes - report.free_before_load_bytes),
            )
            raise RuntimeError(
                f"Refusing to load {config.model_id}: weights and headroom need "
                f"{report.required_bytes:,} bytes; {available_bytes:,} bytes are "
                "available under the GPU utilization limit."
            )
        index = snapshot / "model.safetensors.index.json"
        snapshot_download(
            repo_id=config.model_id,
            revision=snapshot.name,
            token=config.hf_token,
            allow_patterns=["model.safetensors.index.json"],
        )
        # Shard names come from the index, not a hardcoded list.
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            shards = sorted(set(weight_map.values()))
        else:
            shards = ["model.safetensors"]
        snapshot_download(
            repo_id=config.model_id,
            revision=snapshot.name,
            token=config.hf_token,
            allow_patterns=shards,
        )
        with torch.device("cuda"):
            model = Qwen3Model(runtime)
        Qwen3Weights(snapshot).load_into(model)
        model.eval()
        configure_compilation(model, config)
        cache = checker.cache(runtime)
        report = replace(report, cache=cache)
        logger.info(
            "pylon_weights_loaded model=%s revision=%s",
            config.model_id,
            snapshot.name,
        )
        return LoadedModel(
            model, cache, report, snapshot.name, snapshot, stop_ids
        )
