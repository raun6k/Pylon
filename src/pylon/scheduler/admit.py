import logging
import time

from pylon.scheduler.queue import Scheduler

logger = logging.getLogger("pylon")

ADMIT_SKIP_WAIT_SECONDS = 0.1


def admit_requests(engine) -> None:
    scheduler: Scheduler = engine._scheduler
    if len(engine._active_requests) >= engine._max_batch_size:
        head = scheduler.peek()
        if head is not None:
            logger.info(
                "continuous_admission_blocked request_id=%s reason=slots",
                head.payload.request_id,
            )
        return
    while len(engine._active_requests) < engine._max_batch_size:
        waiting = scheduler.waiting_jobs()
        if not waiting:
            return
        selected = _select_waiting_job(engine, waiting, time.perf_counter())
        if selected is None:
            logger.info(
                "continuous_admission_blocked request_id=%s reason=memory",
                waiting[0].payload.request_id,
            )
            return
        chosen, skipped_head = selected
        job = scheduler.take_job(chosen)
        if job is None:
            continue
        request = job.payload
        capacity = len(request.input_ids) + request.sampling.max_new_tokens
        reservation_bytes = engine.request_cache_bytes(capacity)
        if reservation_bytes > engine.kv_budget_bytes:
            if not job.future.done():
                job.future.set_exception(
                    RuntimeError(
                        "The FIFO request cannot fit in the KV-cache budget."
                    )
                )
            continue
        before = len(engine._active_requests)
        reserved = engine._reserved_memory_bytes(extra_capacity=capacity)
        engine._start_request(job, reservation_bytes, reserved)
        if skipped_head and len(engine._active_requests) > before:
            engine.head_skips += 1
            logger.info(
                "continuous_admission_skipped_head request_id=%s admitted=%s head_skips=%s",
                waiting[0].payload.request_id,
                request.request_id,
                engine.head_skips,
            )


def _select_waiting_job(engine, waiting, now: float):
    head = waiting[0]
    if _over_budget(engine, head):
        return head, False
    if _fits(engine, head):
        return head, False
    if _waited_out(head, now) or engine._config.admit_skip <= 0:
        return None
    skips = 0
    for job in waiting[1:]:
        if _over_budget(engine, job):
            return job, False
        if _fits(engine, job):
            return job, True
        if _waited_out(job, now) or skips >= engine._config.admit_skip:
            return None
        skips += 1
    return None


def _over_budget(engine, job) -> bool:
    return _reservation_bytes(engine, job) > engine.kv_budget_bytes


def _fits(engine, job) -> bool:
    capacity = _capacity(job)
    reserved = engine._reserved_memory_bytes(extra_capacity=capacity)
    if reserved > engine.kv_budget_bytes:
        return False
    free_pages = (engine.kv_budget_bytes - reserved) // _page_bytes(engine)
    decoding = sum(
        1
        for active in engine._active_requests
        if active.pending_token_id is not None
    )
    return free_pages >= decoding


def _waited_out(job, now: float) -> bool:
    return now - job.enqueued_at >= ADMIT_SKIP_WAIT_SECONDS


def _capacity(job) -> int:
    request = job.payload
    return len(request.input_ids) + request.sampling.max_new_tokens


def _reservation_bytes(engine, job) -> int:
    return engine.request_cache_bytes(_capacity(job))


def _page_bytes(engine) -> int:
    return engine.prefix_cache.block_size * engine.capacity.bytes_per_token
