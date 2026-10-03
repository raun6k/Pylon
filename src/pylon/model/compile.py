import logging
import os

import torch
from torch import nn

from pylon.config import PylonConfig

logger = logging.getLogger("uvicorn.error")


def configure_compilation(model: nn.Module, config: PylonConfig) -> None:
    if not config.torch_compile:
        return
    if config.compile_diagnostics:
        if "TORCH_LOGS" in os.environ:
            logger.info("torch_compile_diagnostics source=TORCH_LOGS")
        else:
            torch._logging.set_logs(
                dynamo=logging.INFO, graph_breaks=True, recompiles=True,
            )
    model.compile(backend="inductor", fullgraph=config.compile_fullgraph)
    logger.info(
        "torch_compile_enabled backend=inductor mode=default fullgraph=%s "
        "diagnostics=%s compilation=lazy",
        config.compile_fullgraph,
        config.compile_diagnostics,
    )
