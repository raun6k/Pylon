import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Condition, Thread
from typing import Generic, TypeVar

Payload = TypeVar("Payload")
Result = TypeVar("Result")


class QueueFullError(RuntimeError):
    pass


class SchedulerClosedError(RuntimeError):
    pass


@dataclass
class Job(Generic[Payload, Result]):
    payload: Payload
    request_ids: tuple[str, ...]
    enqueued_at: float = field(default_factory=time.perf_counter)
    future: Future[Result] = field(default_factory=Future)


class Scheduler(Generic[Payload, Result]):
    def __init__(
        self,
        tick: Callable[["Scheduler[Payload, Result]"], bool],
        *,
        max_batch_size: int,
        max_queue_size: int,
        batch_wait_seconds: float,
    ) -> None:
        self._tick = tick
        self._max_batch_size = max_batch_size
        self._max_queue_size = max_queue_size
        self._batch_wait_seconds = batch_wait_seconds
        self._condition = Condition()
        self._waiting: deque[Job[Payload, Result]] = deque()
        self._active: tuple[Job[Payload, Result], ...] = ()
        self._closed = False
        self._worker = Thread(target=self._run, name="pylon-scheduler", daemon=True)
        self._worker.start()

    def enqueue(self, job: Job[Payload, Result]) -> Future[Result]:
        with self._condition:
            if self._closed:
                raise SchedulerClosedError("The generation scheduler is closed.")
            self._remove_cancelled_jobs()
            if len(self._waiting) >= self._max_queue_size:
                raise QueueFullError("The generation waiting queue is full.")
            self._waiting.append(job)
            self._condition.notify()
        return job.future

    def peek(self) -> Job[Payload, Result] | None:
        with self._condition:
            self._remove_cancelled_jobs()
            return self._waiting[0] if self._waiting else None

    def waiting_jobs(self) -> tuple[Job[Payload, Result], ...]:
        with self._condition:
            self._remove_cancelled_jobs()
            return tuple(self._waiting)

    def take_job(
        self, expected: Job[Payload, Result]
    ) -> Job[Payload, Result] | None:
        with self._condition:
            self._remove_cancelled_jobs()
            for index, job in enumerate(self._waiting):
                if job is not expected:
                    continue
                del self._waiting[index]
                if not job.future.set_running_or_notify_cancel():
                    return None
                self._active = (*self._active, job)
                return job
            return None

    def take(
        self, expected: Job[Payload, Result] | None = None
    ) -> Job[Payload, Result] | None:
        with self._condition:
            self._remove_cancelled_jobs()
            if not self._waiting:
                return None
            if expected is not None and self._waiting[0] is not expected:
                return None
            while self._waiting:
                job = self._waiting.popleft()
                if job.future.set_running_or_notify_cancel():
                    self._active = (*self._active, job)
                    return job
            return None

    def set_active(self, jobs: tuple[Job[Payload, Result], ...]) -> None:
        with self._condition:
            self._active = jobs

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "waiting": [
                    request_id
                    for job in self._waiting
                    for request_id in job.request_ids
                ],
                "active": [
                    request_id for job in self._active for request_id in job.request_ids
                ],
                "max_batch_size": self._max_batch_size,
                "max_queue_size": self._max_queue_size,
                "batch_wait_ms": self._batch_wait_seconds * 1_000,
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            jobs = (*self._waiting, *self._active)
            self._waiting.clear()
            self._condition.notify_all()
        error = SchedulerClosedError("The generation scheduler is closed.")
        for job in jobs:
            if not job.future.done():
                job.future.set_exception(error)
        self._worker.join()

    def _run(self) -> None:
        while self._wait_for_work():
            try:
                has_work = self._tick(self)
            except Exception as error:
                self._fail_active_jobs(error)
                has_work = False
            if not has_work:
                self.set_active(())

    def _wait_for_work(self) -> bool:
        with self._condition:
            self._remove_cancelled_jobs()
            while not self._waiting and not self._active and not self._closed:
                self._condition.wait()
                self._remove_cancelled_jobs()
            if self._closed:
                return False
            if not self._active:
                self._wait_for_batch()
            return True

    def _wait_for_batch(self) -> None:
        if self._max_batch_size == 1:
            return
        deadline = self._waiting[0].enqueued_at + self._batch_wait_seconds
        while len(self._waiting) < self._max_batch_size and not self._closed:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            self._condition.wait(remaining)
            self._remove_cancelled_jobs()

    def _fail_active_jobs(self, error: Exception) -> None:
        with self._condition:
            active = self._active
            self._active = ()
        for job in active:
            if not job.future.done():
                job.future.set_exception(error)

    def _remove_cancelled_jobs(self) -> None:
        self._waiting = deque(
            job for job in self._waiting if not job.future.cancelled()
        )
