import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from threading import Lock

import torch

from pylon.api.types import Sampling
from pylon.config import PylonConfig
from pylon.decode.runner import Decoder
from pylon.kv.budget import MemoryChecker
from pylon.kv.pool import KVPagePool
from pylon.prefix.cache import (
    PrefixCache,
    PromptBlockView,
    describe_prompt_blocks,
)
from pylon.scheduler.admit import admit_requests
from pylon.scheduler.batch import one_prefill_chunk
from pylon.scheduler.queue import Job, Scheduler

logger = logging.getLogger("pylon")
PAGE_SIZE = 256
FULL_BATCH_PREFILL_WAIT_SECONDS = 0.1


def skip_prefill_wave(
    decoding: int, max_batch_size: int, oldest_age: float | None
) -> bool:
    if decoding != max_batch_size or oldest_age is None:
        return False
    return oldest_age < FULL_BATCH_PREFILL_WAIT_SECONDS


def oldest_prefill_age(active, now: float) -> float | None:
    started = [
        item.prefill_wait_started
        for item in active
        if item.pending_token_id is None
        and item.prompt_offset < len(item.request.input_ids)
    ]
    if not started:
        return None
    return now - min(started)


@dataclass(frozen=True)
class PrefixTrace:
    block_size: int
    prompt_blocks: tuple[PromptBlockView, ...]
    hit_tokens: int
    restored_tokens: int
    stored_blocks: int


@dataclass(frozen=True)
class GenerationResult:
    output_ids: list[int]
    finish_reason: str
    prefill_seconds: float
    inter_token_seconds: list[float]
    restore_seconds: float
    prefix_lookup_seconds: float
    store_seconds: float
    queue_seconds: float
    prefix: PrefixTrace
    first_token_at: float | None = None
    first_token_seconds: float | None = None
    token_intervals: tuple[float, ...] | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True)
class _Request:
    input_ids: list[int]
    stop_ids: frozenset[int]
    sampling: Sampling
    request_id: str


@dataclass
class _ActiveRequest:
    job: Job
    request: _Request
    cache: object
    reservation_bytes: int
    output_ids: list[int]
    prompt_offset: int
    pending_token_id: int | None
    queue_seconds: float
    prefill_seconds: float
    inter_token_seconds: list[float]
    prefix_lookup_seconds: float
    restore_seconds: float
    hit_tokens: int
    restored_tokens: int
    prompt_blocks: tuple[PromptBlockView, ...]
    prefill_wait_started: float
    first_token_at: float | None = None
    last_token_at: float | None = None
    token_intervals: list[float] = field(default_factory=list)


class Engine:
    def __init__(
        self,
        config: PylonConfig,
        loader=None,
        *,
        decoder: Decoder | None = None,
        capacity=None,
        report=None,
    ) -> None:
        self.model_id = config.model_id
        self._config = config
        self.prefix_cache_enabled = config.prefix_cache
        self._prefill_chunk_size = config.prefill_chunk_size
        self._max_batch_size = config.max_batch_size
        self._memory_checker = None
        if decoder is None:
            from pylon.model.loader import ModelLoader

            loaded = (loader or ModelLoader()).load(config)
            self.decoder = Decoder(loaded.model)
            self.capacity = loaded.cache
            self.report = loaded.report
            self.model_revision = loaded.model_revision
            self.stop_token_ids = loaded.stop_token_ids
            self.snapshot = loaded.snapshot
            self._memory_checker = MemoryChecker(config)
        else:
            self.decoder = decoder
            self.capacity = capacity
            self.report = report
            self.model_revision = "test"
            self.stop_token_ids = frozenset()
            self.snapshot = None
        self.prefix_cache = PrefixCache(
            block_size=PAGE_SIZE,
            max_memory_bytes=self.capacity.kv_budget_bytes,
            ttl_seconds=config.prefix_cache_ttl_seconds,
        )
        self._decode_events = None
        self._generation_lock = Lock()
        self._active_requests: list[_ActiveRequest] = []
        self._install_page_pool()
        self._scheduler: Scheduler = Scheduler(
            self._continuous_tick,
            max_batch_size=config.max_batch_size,
            max_queue_size=config.max_queue_size,
            batch_wait_seconds=config.batch_wait_ms / 1_000,
        )

    @property
    def kv_budget_bytes(self) -> int:
        return int(self.prefix_cache.max_memory_bytes)

    def request_cache_bytes(self, token_count: int) -> int:
        size = self.prefix_cache.block_size
        pages = (token_count + size - 1) // size
        return pages * size * self.capacity.bytes_per_token

    def _install_page_pool(self) -> None:
        page_size = self.prefix_cache.block_size
        num_pages = self.capacity.kv_budget_bytes // (
            page_size * self.capacity.bytes_per_token
        )
        if num_pages < 1:
            raise ValueError("The KV budget must hold at least one attention page.")
        device = self.decoder.device
        if device.type == "cuda" and (
            torch.cuda.get_device_capability(device)[0] < 8
            or self.decoder.model.config.dtype not in {torch.float16, torch.bfloat16}
        ):
            raise ValueError("Paged attention requires an SM80+ GPU and FP16 or BF16.")
        self.decoder.page_pool = KVPagePool(
            self.decoder.model.config,
            num_pages,
            device=device,
            page_size=page_size,
        )
        self.prefix_cache.max_memory_bytes = (
            num_pages * page_size * self.capacity.bytes_per_token
        )

    def scheduler_snapshot(self) -> dict[str, object]:
        return self._scheduler.snapshot()

    @property
    def cuda_graph_batch_sizes(self) -> tuple[int, ...]:
        graphs = getattr(self.decoder, "graphs", None)
        if graphs is None:
            return ()
        return tuple(graphs.captured_sizes())

    def prefix_cache_snapshot(self) -> dict[str, object]:
        with self._generation_lock:
            cache = self.prefix_cache
            blocks = cache.blocks()
            return {
                "block_size": cache.block_size,
                "occupied_blocks": len(blocks),
                "cached_tokens": cache.token_count,
                "memory_bytes": cache.memory_bytes,
                "blocks": [block.as_dict() for block in blocks],
            }

    def close(self) -> None:
        self._scheduler.close()
        with self._generation_lock:
            for active in self._active_requests:
                self.decoder.release_cache(active.cache)
            self._active_requests = []
            self.prefix_cache.reserve(0)
            self.prefix_cache.clear()
            if getattr(self.decoder, "graphs", None) is not None:
                self.decoder.graphs = None
            self.decoder.page_pool = None

    def enqueue(
        self,
        input_ids: list[int],
        eos_token_id: int | frozenset[int] | tuple[int, ...] | list[int],
        sampling: Sampling,
        request_id: str | None = None,
    ) -> Future:
        request_id = request_id or "internal"
        stop_ids = _stop_ids(eos_token_id)
        self._validate_request(input_ids, stop_ids, sampling)
        payload = _Request(input_ids, stop_ids, sampling, request_id)
        logger.info("request_waiting request_id=%s", request_id)
        return self._scheduler.enqueue(Job(payload=payload, request_ids=(request_id,)))

    def _continuous_tick(self, scheduler: Scheduler) -> bool:
        with self._generation_lock:
            self._drop_cancelled_active()
            admit_requests(self)
            if self._active_requests:
                try:
                    self._run_waves()
                except Exception as error:
                    logger.exception("model_batch_failed")
                    self._fail_active_requests(error)
            admit_requests(self)
            scheduler.set_active(tuple(active.job for active in self._active_requests))
            return bool(self._active_requests or scheduler.peek() is not None)

    def _run_waves(self) -> None:
        decoding = [
            active
            for active in self._active_requests
            if active.pending_token_id is not None
        ]
        if decoding:
            self._execute_batch(
                decoding,
                [[active.pending_token_id] for active in decoding],
                decode=True,
            )
        now = time.perf_counter()
        if skip_prefill_wave(
            len(decoding),
            self._max_batch_size,
            oldest_prefill_age(self._active_requests, now),
        ):
            return
        chosen = one_prefill_chunk(
            self._active_requests, self._prefill_chunk_size, now
        )
        if chosen is None:
            return
        active, tokens = chosen
        self._execute_batch([active], [tokens], decode=False)

    def _start_request(self, job: Job, reservation_bytes: int, reserved_memory_bytes: int) -> None:
        request = job.payload
        queue_seconds = time.perf_counter() - job.enqueued_at
        prefill_state = None
        active = None
        try:
            self.prefix_cache.reserve(reserved_memory_bytes)
            lookup_started = time.perf_counter()
            prefix_hit = (
                self.prefix_cache.longest_prefix(request.input_ids)
                if self.prefix_cache_enabled
                else None
            )
            prefix_lookup_seconds = time.perf_counter() - lookup_started
            prefill_state = self.decoder.begin_prefill(
                request.input_ids,
                request.sampling,
                max_total_tokens=self.capacity.max_tokens,
                prefix_hit=prefix_hit,
            )
            active = _ActiveRequest(
                job=job,
                request=request,
                cache=prefill_state.cache,
                reservation_bytes=reservation_bytes,
                output_ids=[],
                prompt_offset=prefill_state.next_token_offset,
                pending_token_id=None,
                queue_seconds=queue_seconds,
                prefill_seconds=0.0,
                inter_token_seconds=[],
                prefix_lookup_seconds=prefix_lookup_seconds,
                restore_seconds=prefill_state.restore_seconds,
                hit_tokens=0 if prefix_hit is None else prefix_hit.length,
                restored_tokens=prefill_state.restored_tokens,
                prefill_wait_started=job.enqueued_at,
                prompt_blocks=describe_prompt_blocks(
                    request.input_ids,
                    self.prefix_cache.block_size,
                    prefix_hit,
                ),
            )
            self._active_requests.append(active)
        except Exception as error:
            if active is not None:
                self._active_requests = [
                    item for item in self._active_requests if item is not active
                ]
            if prefill_state is not None:
                self.decoder.release_cache(prefill_state.cache)
            if not job.future.done():
                job.future.set_exception(error)
            self.prefix_cache.reserve(self._reserved_memory_bytes())

    def _execute_batch(
        self,
        selected: list[_ActiveRequest],
        chunks: list[list[int]],
        *,
        decode: bool,
    ) -> None:
        started = time.perf_counter()
        device = self.decoder.device
        events = None
        if device.type == "cuda":
            stream = torch.cuda.current_stream(device)
            if self._decode_events is None:
                self._decode_events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
            events = self._decode_events
            events[0].record(stream)
        if decode:
            logits = self.decoder.decode_caches(
                [active.cache for active in selected],
                [chunk[0] for chunk in chunks],
            )
        else:
            logits = self.decoder.packed_caches(
                [active.cache for active in selected],
                chunks,
            )
        if events is not None:
            events[1].record(stream)
        ready = []
        rows = []
        for row, (active, chunk) in enumerate(zip(selected, chunks, strict=True)):
            if active.pending_token_id is None:
                active.prompt_offset += len(chunk)
                active.prefill_wait_started = time.perf_counter()
                if active.prompt_offset < len(active.request.input_ids):
                    continue
            ready.append(active)
            rows.append(row)
        if not ready:
            self.decoder._synchronize()
            elapsed = (
                events[0].elapsed_time(events[1]) / 1_000
                if events is not None
                else time.perf_counter() - started
            )
            for active in selected:
                active.prefill_seconds += elapsed
            return
        logits = logits[rows]
        if all(active.request.sampling.temperature == 0 for active in ready):
            sampled = logits.argmax(dim=-1)
        elif all(
            active.request.sampling.temperature == ready[0].request.sampling.temperature
            and active.request.sampling.top_p == ready[0].request.sampling.top_p
            for active in ready
        ):
            sampled = self.decoder._sample(logits, ready[0].request.sampling).reshape(-1)
        else:
            sampled = torch.cat(
                [
                    self.decoder._sample(
                        logits[row : row + 1], active.request.sampling
                    ).reshape(-1)
                    for row, active in enumerate(ready)
                ]
            )
        token_ids = sampled.cpu().tolist()
        sampled_at = time.perf_counter()
        elapsed = (
            events[0].elapsed_time(events[1]) / 1_000
            if events is not None
            else time.perf_counter() - started
        )
        for active in selected:
            if active.pending_token_id is None:
                active.prefill_seconds += elapsed
            else:
                active.inter_token_seconds.append(elapsed)
        removed = set()
        for active, token_id in zip(ready, token_ids, strict=True):
            if not self._accept_token(active, int(token_id), active.queue_seconds, sampled_at):
                removed.add(id(active))
        self._active_requests = [
            active for active in self._active_requests if id(active) not in removed
        ]
        self.prefix_cache.reserve(self._reserved_memory_bytes())

    def _accept_token(
        self,
        active: _ActiveRequest,
        token_id: int,
        queue_seconds: float,
        sampled_at: float,
    ) -> bool:
        if active.first_token_at is None:
            active.first_token_at = sampled_at
        if token_id in active.request.stop_ids:
            self._complete_request(active, "eos", queue_seconds)
            return False
        if active.last_token_at is not None:
            active.token_intervals.append(sampled_at - active.last_token_at)
        active.last_token_at = sampled_at
        active.output_ids.append(token_id)
        if len(active.output_ids) == active.request.sampling.max_new_tokens:
            self._complete_request(active, "length", queue_seconds)
            return False
        active.pending_token_id = token_id
        return True

    def _drop_cancelled_active(self) -> None:
        surviving = []
        for active in self._active_requests:
            if active.job.future.cancelled():
                self.decoder.release_cache(active.cache)
            else:
                surviving.append(active)
        self._active_requests = surviving
        self.prefix_cache.reserve(self._reserved_memory_bytes())

    def _complete_request(
        self, active: _ActiveRequest, finish_reason: str, queue_seconds: float
    ) -> None:
        store_started = time.perf_counter()
        stored_blocks = 0
        try:
            reserved = self._reserved_memory_bytes()
            if self.prefix_cache_enabled and reserved <= self.kv_budget_bytes:
                stored_blocks = self.prefix_cache.store_completed_blocks(
                    active.request.input_ids,
                    active.cache,
                    reserved_memory_bytes=reserved,
                )
        except Exception:
            logger.exception(
                "prefix_cache_store_failed request_id=%s", active.request.request_id
            )
            stored_blocks = 0
        finally:
            self.decoder.release_cache(active.cache)
        store_seconds = time.perf_counter() - store_started
        result = GenerationResult(
            first_token_at=active.first_token_at,
            first_token_seconds=(
                None
                if active.first_token_at is None
                else active.first_token_at - active.job.enqueued_at
            ),
            token_intervals=tuple(active.token_intervals),
            elapsed_seconds=time.perf_counter() - active.job.enqueued_at,
            output_ids=active.output_ids,
            finish_reason=finish_reason,
            prefill_seconds=active.prefill_seconds,
            inter_token_seconds=active.inter_token_seconds,
            restore_seconds=active.restore_seconds,
            prefix_lookup_seconds=active.prefix_lookup_seconds,
            store_seconds=store_seconds,
            queue_seconds=queue_seconds,
            prefix=PrefixTrace(
                block_size=self.prefix_cache.block_size,
                prompt_blocks=active.prompt_blocks,
                hit_tokens=active.hit_tokens,
                restored_tokens=active.restored_tokens,
                stored_blocks=stored_blocks,
            ),
        )
        if not active.job.future.done():
            active.job.future.set_result(result)

    def _fail_active_requests(self, error: Exception) -> None:
        for active in self._active_requests:
            self.decoder.release_cache(active.cache)
            if not active.job.future.done():
                active.job.future.set_exception(error)
        self._active_requests = []
        self.prefix_cache.reserve(0)

    def _reserved_memory_bytes(self, *, extra_capacity: int = 0, excluding=None) -> int:
        active = [item for item in self._active_requests if item is not excluding]
        kv_bytes = sum(item.reservation_bytes for item in active)
        if extra_capacity:
            kv_bytes += self.request_cache_bytes(extra_capacity)
        return kv_bytes

    def _validate_request(
        self, input_ids: list[int], stop_ids: frozenset[int], sampling: Sampling
    ) -> None:
        model_config = self.decoder.model.config
        vocabulary_size = model_config.vocab_size
        if not input_ids:
            raise ValueError("A request prompt must contain at least one token.")
        token_ids = [*stop_ids, *input_ids]
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not 0 <= token_id < vocabulary_size
            for token_id in token_ids
        ):
            raise ValueError(
                f"Token IDs must be between 0 and {vocabulary_size - 1:,}."
            )
        token_count = len(input_ids) + sampling.max_new_tokens
        if token_count > model_config.context_length:
            raise ValueError(
                f"Request needs {token_count:,} cache positions, but the model supports "
                f"{model_config.context_length:,}."
            )
        max_tokens = self.kv_budget_bytes // self.capacity.bytes_per_token
        if token_count > max_tokens:
            raise ValueError(
                f"Request needs {token_count:,} KV-cache tokens, but the profiled limit "
                f"is {max_tokens:,}."
            )

    def run_warmup(
        self,
        input_ids: list[int],
        stop_ids: frozenset[int] | tuple[int, ...],
        sampling: Sampling,
        request_id: str,
    ) -> GenerationResult:
        stops = _stop_ids(stop_ids)
        self._validate_request(input_ids, stops, sampling)
        with self._generation_lock:
            started = time.perf_counter()
            request_bytes = self.request_cache_bytes(len(input_ids) + sampling.max_new_tokens)
            self.prefix_cache.reserve(request_bytes)
            prefix_hit = (
                self.prefix_cache.longest_prefix(input_ids)
                if self.prefix_cache_enabled
                else None
            )
            decoded = self.decoder.generate(
                input_ids,
                stops,
                sampling,
                max_total_tokens=self.capacity.max_tokens,
                prefix_hit=prefix_hit,
                request_id=request_id,
            )
            stored_blocks = 0
            try:
                if self.prefix_cache_enabled and request_bytes <= self.kv_budget_bytes:
                    stored_blocks = self.prefix_cache.store_completed_blocks(
                        input_ids,
                        decoded.cache,
                        reserved_memory_bytes=request_bytes,
                    )
            finally:
                self.decoder.release_cache(decoded.cache)
            return GenerationResult(
                output_ids=decoded.output_ids,
                finish_reason=decoded.finish_reason,
                prefill_seconds=decoded.prefill_seconds,
                inter_token_seconds=decoded.inter_token_seconds,
                restore_seconds=decoded.restore_seconds,
                prefix_lookup_seconds=0.0,
                store_seconds=0.0,
                queue_seconds=0.0,
                first_token_at=decoded.first_token_at,
                first_token_seconds=(
                    None
                    if decoded.first_token_at is None
                    else decoded.first_token_at - started
                ),
                token_intervals=decoded.token_intervals,
                prefix=PrefixTrace(
                    block_size=self.prefix_cache.block_size,
                    prompt_blocks=describe_prompt_blocks(
                        input_ids, self.prefix_cache.block_size, prefix_hit
                    ),
                    hit_tokens=0 if prefix_hit is None else prefix_hit.length,
                    restored_tokens=decoded.restored_tokens,
                    stored_blocks=stored_blocks,
                ),
            )

    def capture_decode_graphs(self, input_ids: list[int]) -> tuple[int, ...]:
        if not self._config.cuda_graphs or self.decoder.device.type != "cuda":
            return ()
        from pylon.decode.warmup import capture_decode_graphs

        with self._generation_lock:
            if self._active_requests:
                raise RuntimeError("Cannot capture CUDA graphs while requests are active.")
            return capture_decode_graphs(
                self.decoder,
                token_id=int(input_ids[-1]),
                max_batch_size=self._max_batch_size,
                max_tokens=self.capacity.max_tokens,
            )

    def warm_decode(self, input_ids: list[int]) -> tuple[int, tuple[int, ...]]:
        from pylon.decode.warmup import warm_decode

        with self._generation_lock:
            if self._active_requests:
                raise RuntimeError("Cannot warm decode while requests are active.")
            return warm_decode(
                self.decoder,
                input_ids,
                max_batch_size=self._max_batch_size,
                max_tokens=self.capacity.max_tokens,
                budget_tokens=self.capacity.kv_budget_bytes // self.capacity.bytes_per_token,
                prefill_chunk_size=self._prefill_chunk_size,
            )

    def update_cache_capacity(self, *, warmup_peak_bytes: int, warmup_kv_bytes: int) -> None:
        if self._memory_checker is None:
            raise RuntimeError("Cannot reprofile KV memory without a CUDA memory check.")
        with self._generation_lock:
            if self._active_requests:
                raise RuntimeError("Cannot reprofile KV memory while requests are active.")
            self.prefix_cache.clear()
            self.decoder.page_pool = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cache = self._memory_checker.cache(
                self.decoder.model.config,
                warmup_peak_bytes=warmup_peak_bytes,
                warmup_kv_bytes=warmup_kv_bytes,
            )
            self.capacity = cache
            self.report = replace(self.report, cache=cache)
            self.prefix_cache.max_memory_bytes = cache.kv_budget_bytes
            self._install_page_pool()


def _stop_ids(eos_token_id: int | frozenset[int] | tuple[int, ...] | list[int]) -> frozenset[int]:
    if isinstance(eos_token_id, bool) or isinstance(eos_token_id, int):
        return frozenset((int(eos_token_id),))
    return frozenset(int(token_id) for token_id in eos_token_id)
