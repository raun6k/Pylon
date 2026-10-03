import logging

from pylon.scheduler.queue import Scheduler

logger = logging.getLogger("pylon")


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
        head = scheduler.peek()
        if head is None:
            return
        request = head.payload
        capacity = len(request.input_ids) + request.sampling.max_new_tokens
        reservation_bytes = engine.request_cache_bytes(capacity)
        if reservation_bytes > engine.kv_budget_bytes:
            job = scheduler.take(head)
            if job is not None and not job.future.done():
                job.future.set_exception(
                    RuntimeError("The FIFO request cannot fit in the KV-cache budget.")
                )
            continue
        reserved = engine._reserved_memory_bytes(extra_capacity=capacity)
        if reserved > engine.kv_budget_bytes:
            logger.info(
                "continuous_admission_blocked request_id=%s reason=memory",
                request.request_id,
            )
            return
        job = scheduler.take(head)
        if job is None:
            return
        engine._start_request(job, reservation_bytes, reserved)
