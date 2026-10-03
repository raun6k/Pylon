import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pylon.api.types import Sampling
from pylon.config import PylonConfig
from pylon.engine import Engine
from pylon.kv.budget import CacheCapacity, MemoryReport
from pylon.kv.cache import PackedBatchCache, PagedBatchCache
from pylon.kv.pool import KVPagePool
from pylon.model.config import Qwen3Config
from pylon.prefix.cache import hash_token_blocks
from pylon.scheduler.batch import pack_tick


class FakeDecoder:
    def __init__(self, config: Qwen3Config) -> None:
        self.model = SimpleNamespace(config=config)
        self.device = torch.device("cpu")
        self.page_pool: KVPagePool | None = None
        self.prefill_chunk_sizes: list[int] = []
        self._planned: list[int] = []
        self._token_for_cache: dict[int, int] = {}

    def plan(self, *token_ids: int) -> None:
        self._planned.extend(token_ids)

    def begin_prefill(
        self,
        input_ids: list[int],
        sampling: Sampling,
        *,
        max_total_tokens: int,
        prefix_hit: object | None = None,
    ) -> SimpleNamespace:
        del prefix_hit
        capacity = len(input_ids) + sampling.max_new_tokens
        if capacity > max_total_tokens:
            raise ValueError(
                f"Request needs {capacity} KV-cache tokens, but the profiled "
                f"limit is {max_total_tokens}."
            )
        if self.page_pool is None:
            raise RuntimeError("Paged decode needs a page pool.")
        cache = __import__(
            "pylon.kv.cache", fromlist=["PagedKVCache"]
        ).PagedKVCache(self.page_pool, capacity)
        token = self._planned.pop(0) if self._planned else 5
        self._token_for_cache[id(cache)] = token
        return SimpleNamespace(
            cache=cache,
            next_token_offset=cache.length,
            restore_seconds=0.0,
            restored_tokens=cache.length,
        )

    def packed_caches(self, caches: list, chunks: list[list[int]]) -> torch.Tensor:
        counts = [len(chunk) for chunk in chunks]
        self.prefill_chunk_sizes.extend(counts)
        batch = PackedBatchCache(caches)
        batch.prepare_packed(counts)
        batch.advance_packed()
        return self._logits(caches)

    def decode_caches(self, caches: list, token_ids: list[int]) -> torch.Tensor:
        del token_ids
        batch = PagedBatchCache(caches)
        batch.prepare(1)
        batch.advance(1)
        return self._logits(caches)

    def release_cache(self, cache) -> None:
        cache.close()

    def _synchronize(self) -> None:
        return None

    def _logits(self, caches: list) -> torch.Tensor:
        vocab = self.model.config.vocab_size
        logits = torch.full((len(caches), vocab), -1.0e4)
        for row, cache in enumerate(caches):
            logits[row, self._token_for_cache[id(cache)]] = 10.0
        return logits


class ContinuousBatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Qwen3Config(
            vocab_size=64,
            context_length=8192,
            hidden_size=16,
            n_heads=2,
            n_layers=2,
            hidden_dim=32,
            head_dim=8,
            n_kv_heads=1,
            dtype=torch.float32,
        )
        self.decoder = FakeDecoder(self.config)

    def make_engine(
        self,
        *,
        slots: int = 8,
        budget_tokens: int = 8192,
        prefill_chunk_size: int = 256,
        prefix_cache: bool = True,
    ) -> Engine:
        element_size = torch.empty((), dtype=self.config.dtype).element_size()
        bytes_per_token = (
            self.config.n_layers
            * 2
            * self.config.n_kv_heads
            * self.config.head_dim
            * element_size
        )
        budget_bytes = budget_tokens * bytes_per_token
        capacity = CacheCapacity(
            free_bytes=budget_bytes,
            device_occupied_bytes=0,
            activation_headroom_bytes=0,
            bytes_per_token=bytes_per_token,
            max_tokens=budget_tokens,
            kv_budget_bytes=budget_bytes,
            model_occupied_bytes=0,
        )
        report = MemoryReport(
            gpu="test-cpu",
            total_bytes=budget_bytes,
            max_gpu_utilization=1.0,
            max_gpu_bytes=budget_bytes,
            free_before_load_bytes=budget_bytes,
            weight_bytes=0,
            required_bytes=0,
            fits=True,
            cache=capacity,
        )
        with patch("pylon.scheduler.queue.Thread"):
            engine = Engine(
                PylonConfig(
                    model_id="test-cpu",
                    max_batch_size=slots,
                    prefill_chunk_size=prefill_chunk_size,
                    batch_wait_ms=0,
                    prefix_cache=prefix_cache,
                ),
                decoder=self.decoder,
                capacity=capacity,
                report=report,
            )
        self.addCleanup(engine.close)
        return engine

    def enqueue(self, engine: Engine, name: str, prompt: list[int], output_limit: int, *, eos: int = 0):
        return engine.enqueue(
            prompt,
            eos,
            Sampling(temperature=0, top_p=1, max_new_tokens=output_limit),
            request_id=name,
        )

    def tick(self, engine: Engine) -> None:
        engine._continuous_tick(engine._scheduler)

    def finish(self, engine: Engine, *futures) -> None:
        for _ in range(64):
            if all(future.done() for future in futures):
                for future in futures:
                    future.result(timeout=0)
                return
            self.tick(engine)
        self.fail("requests did not finish")

    def test_prefill_stays_inside_the_chunk_bound(self) -> None:
        engine = self.make_engine(prefill_chunk_size=4)
        prompt = list(range(1, 11))
        future = self.enqueue(engine, "long", prompt, 1)
        self.tick(engine)
        active = engine._active_requests[0]
        self.assertEqual(active.prompt_offset, 4)
        self.assertIsNone(active.pending_token_id)
        self.assertFalse(future.done())
        self.tick(engine)
        self.assertEqual(engine._active_requests[0].prompt_offset, 8)
        self.finish(engine, future)
        self.assertEqual(self.decoder.prefill_chunk_sizes, [4, 4, 2])
        self.assertTrue(all(size <= 4 for size in self.decoder.prefill_chunk_sizes))
        self.assertEqual(future.result(timeout=0).output_ids, [5])
        self.assertEqual(future.result(timeout=0).finish_reason, "length")

    def test_head_of_line_waits_until_the_reserved_pages_are_free(self) -> None:
        engine = self.make_engine(slots=2, budget_tokens=256)
        prompt = [3, 4, 5, 6]
        first = self.enqueue(engine, "first", prompt, 4)
        second = self.enqueue(engine, "second", prompt, 4)
        third = self.enqueue(engine, "third", [7, 8], 1)
        self.tick(engine)
        self.assertEqual(engine.scheduler_snapshot()["active"], ["first"])
        self.assertEqual(engine.scheduler_snapshot()["waiting"], ["second", "third"])
        self.assertFalse(second.done())
        self.tick(engine)
        self.assertEqual(engine.scheduler_snapshot()["waiting"], ["second", "third"])
        self.finish(engine, first)
        self.assertEqual(engine.scheduler_snapshot()["active"], ["second"])
        self.assertEqual(engine.scheduler_snapshot()["waiting"], ["third"])
        self.finish(engine, second, third)

    def test_cancelled_waiter_does_not_block_the_next_job(self) -> None:
        engine = self.make_engine(slots=1, budget_tokens=256)
        first = self.enqueue(engine, "first", [3, 4, 5, 6], 4)
        cancelled = self.enqueue(engine, "cancelled", [7, 8, 9, 10], 4)
        last = self.enqueue(engine, "last", [11, 12, 13, 14], 2)
        self.tick(engine)
        self.assertTrue(cancelled.cancel())
        self.tick(engine)
        self.assertEqual(engine.scheduler_snapshot()["waiting"], ["last"])
        self.finish(engine, first, last)
        self.assertTrue(cancelled.cancelled())
        self.assertEqual(last.result(timeout=0).output_ids, [5, 5])

    def test_eos_frees_the_slot(self) -> None:
        self.decoder.plan(0, 5)
        engine = self.make_engine(slots=1, budget_tokens=256)
        stopped = self.enqueue(engine, "stopped", [3, 4, 5, 6], 10, eos=0)
        nxt = self.enqueue(engine, "next", [7, 8, 9, 10], 2)
        self.tick(engine)
        result = stopped.result(timeout=0)
        self.assertEqual(result.finish_reason, "eos")
        self.assertEqual(result.output_ids, [])
        self.assertEqual(engine.scheduler_snapshot()["active"], ["next"])
        self.finish(engine, nxt)
        self.assertEqual(nxt.result(timeout=0).output_ids, [5, 5])

    def test_prefix_store_runs_at_completion_and_omits_the_short_tail(self) -> None:
        engine = self.make_engine(budget_tokens=1024, prefill_chunk_size=256)
        prompt = [(index % 63) + 1 for index in range(300)]
        future = self.enqueue(engine, "tail", prompt, 1)
        self.tick(engine)
        self.assertFalse(future.done())
        self.assertEqual(engine.prefix_cache.token_count, 0)
        self.finish(engine, future)
        result = future.result(timeout=0)
        self.assertEqual(result.prefix.stored_blocks, 1)
        self.assertEqual(engine.prefix_cache.token_count, 256)

        short = self.enqueue(engine, "short", [(index % 63) + 1 for index in range(100)], 1)
        self.finish(engine, short)
        self.assertEqual(short.result(timeout=0).prefix.stored_blocks, 0)
        self.assertEqual(engine.prefix_cache.token_count, 256)

    def test_prefix_cache_flag_stores_nothing(self) -> None:
        engine = self.make_engine(
            budget_tokens=1024, prefill_chunk_size=256, prefix_cache=False
        )
        prompt = [(index % 63) + 1 for index in range(300)]
        future = self.enqueue(engine, "off", prompt, 1)
        self.finish(engine, future)
        result = future.result(timeout=0)
        self.assertEqual(result.prefix.stored_blocks, 0)
        self.assertEqual(engine.prefix_cache.token_count, 0)

    def test_aged_prefill_takes_the_next_chunk(self) -> None:
        now = 1_000.0
        decoding = SimpleNamespace(
            pending_token_id=4,
            prefill_wait_started=now,
            prompt_offset=3,
            request=SimpleNamespace(input_ids=[1, 2, 3, 4]),
        )
        fresh = SimpleNamespace(
            pending_token_id=None,
            prefill_wait_started=now - 0.01,
            prompt_offset=0,
            request=SimpleNamespace(input_ids=[1] * 20),
        )
        aged = SimpleNamespace(
            pending_token_id=None,
            prefill_wait_started=now - 0.1,
            prompt_offset=0,
            request=SimpleNamespace(input_ids=[2] * 20),
        )
        selected, chunks = pack_tick(
            [fresh, aged, decoding], prefill_chunk_size=4, now=now
        )
        self.assertEqual(selected[0], decoding)
        self.assertEqual(chunks[0], [4])
        self.assertEqual(selected[1], aged)
        self.assertEqual(chunks[1], [2, 2, 2])
        self.assertNotIn(fresh, selected)

    def test_block_hash_is_chained_sha256_of_json(self) -> None:
        blocks = hash_token_blocks([(1, 2), (3,)])
        first_payload = json.dumps((1, 2), separators=(",", ":")).encode("ascii")
        first = hashlib.sha256(first_payload).digest()
        self.assertEqual(blocks[0].hash, first.hex())
        self.assertEqual(blocks[0].parent_hash, "")
        second_payload = json.dumps((3,), separators=(",", ":")).encode("ascii")
        second = hashlib.sha256(first + second_payload).digest()
        self.assertEqual(blocks[1].hash, second.hex())
        self.assertEqual(blocks[1].parent_hash, first.hex())


if __name__ == "__main__":
    unittest.main()
