import logging
import time
import uuid
from typing import Annotated

import torch
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from pylon.api.deps import get_generator
from pylon.api.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionPromptTokensDetails,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionTimings,
    ChatCompletionUsage,
)
from pylon.frontend import TextGenerator
from pylon.scheduler.queue import QueueFullError, SchedulerClosedError

router = APIRouter()
logger = logging.getLogger("pylon")
GeneratorDependency = Annotated[TextGenerator, Depends(get_generator)]


def _messages(payload: ChatCompletionRequest) -> list[tuple[str, str]]:
    return [(message.role, message.content) for message in payload.messages]


@router.get("/health")
async def health(generator: GeneratorDependency) -> dict[str, object]:
    return await run_in_threadpool(generator.health)


@router.get("/internal/cache")
async def prefix_cache_state(generator: GeneratorDependency) -> dict[str, object]:
    return await run_in_threadpool(generator.engine.prefix_cache_snapshot)


@router.post("/internal/profile/reset")
async def reset_profile(generator: GeneratorDependency) -> dict[str, str]:
    del generator
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return {"status": "ok"}


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    payload: ChatCompletionRequest,
    generator: GeneratorDependency,
) -> ChatCompletionResponse:
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    logger.info(
        "pylon request_received request_id=%s model=%s messages=%d max_new_tokens=%d",
        request_id,
        payload.model,
        len(payload.messages),
        payload.max_tokens,
    )
    if payload.model != generator.model_id:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{payload.model}' is not loaded. Use '{generator.model_id}'.",
        )
    try:
        result = await generator.run_chat_async(_messages(payload), payload.sampling(), request_id)
    except (QueueFullError, SchedulerClosedError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finish = "stop" if result.finish_reason == "eos" else "length"
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=generator.model_id,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(content=result.text),
                finish_reason=finish,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
            prompt_tokens_details=ChatCompletionPromptTokensDetails(
                cached_tokens=result.cached_tokens,
            ),
        ),
        timings=ChatCompletionTimings(
            tokenize_seconds=result.tokenize_seconds,
            queue_seconds=result.queue_seconds,
            prefix_lookup_seconds=result.prefix_lookup_seconds,
            restore_seconds=result.restore_seconds,
            prefill_seconds=result.prefill_seconds,
            decode_seconds=result.decode_seconds,
            decode_compute_seconds=result.decode_compute_seconds,
            decode_compute_tokens_per_second=result.decode_compute_tokens_per_second,
            inter_token_seconds=result.inter_token_seconds,
            store_seconds=result.store_seconds,
            time_to_first_token_seconds=result.time_to_first_token_seconds,
            total_seconds=result.total_seconds,
            generation_tokens_per_second=result.generation_tokens_per_second,
            prefill_tokens_per_second=result.prefill_tokens_per_second,
            decode_tokens_per_second=result.decode_tokens_per_second,
            cache_hit_rate=result.cache_hit_rate,
            accepted_tokens_per_step=result.accepted_tokens_per_step,
        ),
    )
