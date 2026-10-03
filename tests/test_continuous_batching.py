import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.run import result_flags, system_environ
from pylon.api.types import Sampling
from pylon.decode.runner import Decoder
from pylon.frontend import TextGenerator
from pylon.model.model import Qwen3Model
from pylon.config import PylonConfig, get_config
from pylon.engine import FULL_BATCH_PREFILL_WAIT_SECONDS, Engine, skip_prefill_wave
from pylon.kv.budget import CacheCapacity, MemoryReport
from pylon.kv.cache import PackedBatchCache, PagedBatchCache
from pylon.kv.pool import KVPagePool
from pylon.model.config import Qwen3Config
from pylon.prefix.cache import hash_token_blocks
from pylon.scheduler.admit import ADMIT_SKIP_WAIT_SECONDS
from pylon.scheduler.batch import PREFILL_MAX_WAIT_SECONDS, pack_tick


class FakeDecoder:
    def __init__(self, config: Qwen3Config) -> None:
        self.model = SimpleNamespace(config=config)
        self.device = torch.device("cpu")
        self.page_pool: KVPagePool | None = None
        self.graphs = None
        self.prefill_chunk_sizes: list[int] = []
        self.forwards: list[tuple[str, list]] = []
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
        cached_blocks = ()
        if prefix_hit is not None:
            cached_blocks = tuple(prefix_hit.blocks)
        if cached_blocks:
            cache.restore_blocks(tuple(block.snapshot for block in cached_blocks))
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
        self.forwards.append(("prefill", [list(chunk) for chunk in chunks]))
        batch = PackedBatchCache(caches)
        batch.prepare_packed(counts)
        batch.advance_packed()
        return self._logits(caches)

    def decode_caches(self, caches: list, token_ids: list[int]) -> torch.Tensor:
        self.forwards.append(("decode", [[token_id] for token_id in token_ids]))
        if self.graphs is not None:
            replayed = self.graphs.replay(caches, token_ids)
            if replayed is not None:
                batch = PagedBatchCache(caches)
                batch.prepare(1)
                batch.advance(1)
                return replayed
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


class WarmupTokenizer:
    stop_token_ids = ()

    def encode_chat(self, messages: list[tuple[str, str]]) -> list[int]:
        count = max(8, sum(len(content) for _, content in messages) // 4)
        if len(messages) > 1:
            count += 32
        return [3] * count


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
        admit_skip: int = 4,
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
                    admit_skip=admit_skip,
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

    def tick_at(self, engine: Engine, now: float) -> None:
        with patch("pylon.scheduler.admit.time.perf_counter", return_value=now):
            self.tick(engine)

    def set_wait(self, engine: Engine, name: str, seconds: float, now: float) -> None:
        for job in engine._scheduler.waiting_jobs():
            if job.payload.request_id == name:
                job.enqueued_at = now - seconds
                return
        self.fail(f"{name} is not waiting")

    def ids(self, engine: Engine, key: str) -> list[str]:
        return list(engine.scheduler_snapshot()[key])

    def finish(self, engine: Engine, *futures) -> None:
        for _ in range(64):
            if all(future.done() for future in futures):
                for future in futures:
                    future.result(timeout=0)
                return
            self.tick(engine)
        self.fail("requests did not finish")

    def activate_head(self, engine: Engine) -> None:
        head = engine._scheduler.peek()
        self.assertIsNotNone(head)
        request = head.payload
        capacity = len(request.input_ids) + request.sampling.max_new_tokens
        reservation = engine.request_cache_bytes(capacity)
        reserved = engine._reserved_memory_bytes(extra_capacity=capacity)
        job = engine._scheduler.take(head)
        self.assertIsNotNone(job)
        engine._start_request(job, reservation, reserved)

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

    def test_resident_block_is_sighted_before_finish_and_pinned_on_the_second_sight(self) -> None:
        engine = self.make_engine(budget_tokens=1024, prefill_chunk_size=256)
        prompt = [(index % 63) + 1 for index in range(300)]
        block_hash = hash_token_blocks([tuple(prompt[:256])])[0].hash
        future = self.enqueue(engine, "first", prompt, 1)
        self.tick(engine)
        self.assertFalse(future.done())
        self.assertEqual(engine.prefix_cache.sighted_hashes, frozenset({block_hash}))
        self.assertEqual(engine.prefix_cache.token_count, 0)
        self.assertEqual(engine.prefix_cache.memory_bytes, 0)
        self.finish(engine, future)
        result = future.result(timeout=0)
        self.assertEqual(result.prefix.stored_blocks, 0)
        self.assertEqual(engine.prefix_cache.token_count, 0)
        pool = engine.decoder.page_pool
        self.assertEqual(pool.free_pages, pool.num_pages)

        short = self.enqueue(engine, "short", [(index % 63) + 1 for index in range(100)], 1)
        self.finish(engine, short)
        self.assertEqual(short.result(timeout=0).prefix.stored_blocks, 0)
        self.assertEqual(engine.prefix_cache.sighted_hashes, frozenset({block_hash}))
        self.assertEqual(engine.prefix_cache.token_count, 0)

        second = self.enqueue(
            engine, "second", prompt[:256] + [(index % 63) + 2 for index in range(44)], 1
        )
        self.tick(engine)
        self.assertFalse(second.done())
        self.assertEqual(engine.prefix_cache.token_count, 256)
        self.assertGreater(engine.prefix_cache.memory_bytes, 0)
        self.finish(engine, second)
        self.assertEqual(second.result(timeout=0).prefix.stored_blocks, 1)
        self.assertEqual(second.result(timeout=0).prefix.hit_tokens, 0)
        self.assertEqual(pool.free_pages, pool.num_pages - 1)

        third = self.enqueue(engine, "third", prompt, 1)
        self.finish(engine, third)
        restored = third.result(timeout=0)
        self.assertEqual(restored.prefix.hit_tokens, 256)
        self.assertEqual(restored.prefix.restored_tokens, 256)

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
        self.assertEqual(engine.prefix_cache.sighted_hashes, frozenset())
        self.assertEqual(engine.prefix_cache.memory_bytes, 0)

    def test_warmup_finishes_with_prefix_cache_disabled(self) -> None:
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=64,
            context_length=2048,
            hidden_size=16,
            n_heads=2,
            n_layers=1,
            hidden_dim=32,
            head_dim=8,
            n_kv_heads=1,
            dtype=torch.float32,
        )
        model = Qwen3Model(config).eval()
        element_size = torch.empty((), dtype=config.dtype).element_size()
        bytes_per_token = (
            config.n_layers * 2 * config.n_kv_heads * config.head_dim * element_size
        )
        budget_tokens = 2048
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
                    max_batch_size=1,
                    prefill_chunk_size=256,
                    batch_wait_ms=0,
                    prefix_cache=False,
                ),
                decoder=Decoder(model),
                capacity=capacity,
                report=report,
            )
        self.addCleanup(engine.close)
        generator = TextGenerator(WarmupTokenizer(), engine)
        generator.warm_up()
        health = generator.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["warmup_batch_sizes"], [1])
        self.assertEqual(engine.prefix_cache.token_count, 0)
        self.assertFalse(engine.prefix_cache_enabled)

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

    def test_one_tick_prefills_a_single_chunk(self) -> None:
        engine = self.make_engine(slots=4, budget_tokens=8192, prefill_chunk_size=8)
        first = self.enqueue(engine, "first", [1, 2, 3, 4], 1)
        second = self.enqueue(engine, "second", [6, 7, 8, 9], 1)
        self.tick(engine)
        self.assertEqual(
            [active.request.request_id for active in engine._active_requests],
            ["second"],
        )
        self.assertEqual(engine._active_requests[0].prompt_offset, 0)
        self.assertEqual(self.decoder.prefill_chunk_sizes, [4])
        self.assertEqual(first.result(timeout=0).output_ids, [5])
        self.assertEqual([name for name, _ in self.decoder.forwards], ["prefill"])

    def test_prefill_chunk_is_not_shrunk_by_decode_tokens(self) -> None:
        engine = self.make_engine(slots=4, budget_tokens=8192, prefill_chunk_size=4)
        self.enqueue(engine, "decode", [1, 2, 3], 8)
        self.enqueue(engine, "prefill", [4] * 12, 1)
        self.tick(engine)
        decode = engine._active_requests[0]
        prefill = engine._active_requests[1]
        self.assertEqual(decode.request.request_id, "decode")
        self.assertIsNotNone(decode.pending_token_id)
        self.assertEqual(prefill.prompt_offset, 0)
        prefill_seconds = decode.prefill_seconds
        intervals = len(decode.inter_token_seconds)
        self.decoder.forwards.clear()
        self.tick(engine)
        self.assertEqual(
            [name for name, _ in self.decoder.forwards], ["decode", "prefill"]
        )
        self.assertEqual(self.decoder.forwards[1][1], [[4, 4, 4, 4]])
        self.assertEqual(prefill.prompt_offset, 4)
        self.assertEqual(len(decode.inter_token_seconds), intervals + 1)
        self.assertEqual(decode.prefill_seconds, prefill_seconds)
        self.assertGreater(prefill.prefill_seconds, 0)

    def test_decode_wave_replays_the_graph_and_prefill_stays_eager(self) -> None:
        engine = self.make_engine(slots=4, budget_tokens=8192, prefill_chunk_size=4)
        replayed: list[int] = []

        class Graphs:
            def replay(self, caches, token_ids):
                del token_ids
                replayed.append(len(caches))
                return None

        self.decoder.graphs = Graphs()
        self.enqueue(engine, "decode", [1, 2, 3], 8)
        self.enqueue(engine, "prefill", [4] * 12, 1)
        self.tick(engine)
        replayed.clear()
        self.decoder.forwards.clear()
        self.tick(engine)
        self.assertEqual(replayed, [1])
        self.assertEqual(
            [name for name, _ in self.decoder.forwards], ["decode", "prefill"]
        )

    def test_full_decode_batch_holds_a_young_prefill(self) -> None:
        engine = self.make_engine(slots=1, budget_tokens=4096, prefill_chunk_size=4)
        self.enqueue(engine, "decode", [1, 2, 3], 8)
        self.tick(engine)
        self.enqueue(engine, "prefill", [4] * 12, 1)
        self.activate_head(engine)
        prefill = engine._active_requests[1]
        prefill.prefill_wait_started = time.perf_counter()
        self.decoder.forwards.clear()
        self.tick(engine)
        self.assertEqual(prefill.prompt_offset, 0)
        self.assertIsNone(prefill.pending_token_id)
        self.assertEqual([name for name, _ in self.decoder.forwards], ["decode"])

    def test_full_decode_batch_prefills_a_prompt_that_waited_100ms(self) -> None:
        engine = self.make_engine(slots=1, budget_tokens=4096, prefill_chunk_size=4)
        self.enqueue(engine, "decode", [1, 2, 3], 8)
        self.tick(engine)
        self.enqueue(engine, "prefill", [4] * 12, 1)
        self.activate_head(engine)
        prefill = engine._active_requests[1]
        prefill.prefill_wait_started = (
            time.perf_counter() - FULL_BATCH_PREFILL_WAIT_SECONDS
        )
        self.decoder.forwards.clear()
        self.tick(engine)
        self.assertEqual(prefill.prompt_offset, 4)
        self.assertEqual(
            [name for name, _ in self.decoder.forwards], ["decode", "prefill"]
        )
        self.assertEqual(self.decoder.forwards[1][1], [[4, 4, 4, 4]])

    def test_young_prefill_still_runs_when_the_decode_batch_is_not_full(self) -> None:
        engine = self.make_engine(slots=2, budget_tokens=8192, prefill_chunk_size=4)
        self.enqueue(engine, "decode", [1, 2, 3], 8)
        self.enqueue(engine, "prefill", [4] * 12, 1)
        self.tick(engine)
        prefill = engine._active_requests[1]
        prefill.prefill_wait_started = time.perf_counter()
        self.decoder.forwards.clear()
        self.tick(engine)
        self.assertEqual(prefill.prompt_offset, 4)
        self.assertEqual(
            [name for name, _ in self.decoder.forwards], ["decode", "prefill"]
        )

    def test_aged_prefill_gets_the_only_chunk(self) -> None:
        engine = self.make_engine(slots=4, budget_tokens=8192, prefill_chunk_size=4)
        self.enqueue(engine, "decode", [1, 2, 3], 8)
        self.enqueue(engine, "fresh", [7] * 12, 1)
        self.enqueue(engine, "aged", [8] * 12, 1)
        self.tick(engine)
        fresh = engine._active_requests[1]
        aged = engine._active_requests[2]
        now = time.perf_counter()
        fresh.prefill_wait_started = now - 0.01
        aged.prefill_wait_started = now - PREFILL_MAX_WAIT_SECONDS
        self.decoder.forwards.clear()
        self.tick(engine)
        self.assertEqual(fresh.prompt_offset, 0)
        self.assertEqual(aged.prompt_offset, 4)
        self.assertEqual(self.decoder.forwards[0][0], "decode")
        self.assertEqual(self.decoder.forwards[1][1], [[8, 8, 8, 8]])

    def test_skip_threshold_is_under_100ms_on_a_full_batch_only(self) -> None:
        self.assertEqual(FULL_BATCH_PREFILL_WAIT_SECONDS, 0.1)
        self.assertEqual(PREFILL_MAX_WAIT_SECONDS, 0.1)
        self.assertTrue(skip_prefill_wave(8, 8, 0.099))
        self.assertFalse(skip_prefill_wave(8, 8, 0.1))
        self.assertFalse(skip_prefill_wave(8, 8, None))
        self.assertFalse(skip_prefill_wave(7, 8, 0.0))

    def test_earlier_request_sights_a_shared_block_and_the_later_request_pins(self) -> None:
        engine = self.make_engine(slots=2, budget_tokens=2048, prefill_chunk_size=256)
        prompt = [(index % 63) + 1 for index in range(256)]
        self.enqueue(engine, "earlier", prompt, 1)
        self.enqueue(engine, "later", prompt, 1)
        self.activate_head(engine)
        self.activate_head(engine)
        self.assertEqual(
            [active.request.request_id for active in engine._active_requests],
            ["earlier", "later"],
        )
        for active in engine._active_requests:
            batch = PackedBatchCache([active.cache])
            batch.prepare_packed([256])
            batch.advance_packed()
            active.prompt_offset = 256
        earlier_page = engine._active_requests[0].cache.pages[0].index
        later_page = engine._active_requests[1].cache.pages[0].index
        self.assertNotEqual(earlier_page, later_page)
        engine._publish_resident_blocks()
        block_hash = hash_token_blocks([tuple(prompt)])[0].hash
        cached = engine.prefix_cache.get(block_hash)
        self.assertIsNotNone(cached)
        self.assertEqual(tuple(page.index for page in cached.snapshot.pages), (later_page,))
        self.assertEqual(engine.prefix_cache.token_count, 256)
        self.assertEqual(engine._active_requests[0].pinned_blocks, 0)
        self.assertEqual(engine._active_requests[1].pinned_blocks, 1)
        pool = engine.decoder.page_pool
        self.assertEqual(pool.free_pages, pool.num_pages - 2)
        engine._active_requests[0].cache.close()
        self.assertEqual(pool.free_pages, pool.num_pages - 1)
        self.assertIn(earlier_page, pool._free)
        engine._active_requests[1].cache.close()
        self.assertEqual(pool.free_pages, pool.num_pages - 1)
        self.assertNotIn(later_page, pool._free)

    def engine_with_one_decode(self, pages: int, *, admit_skip: int = 4) -> Engine:
        engine = self.make_engine(
            slots=8, budget_tokens=256 * pages, admit_skip=admit_skip
        )
        self.enqueue(engine, "decode", [1, 2, 3, 4], 32)
        self.tick(engine)
        active = engine._active_requests
        self.assertEqual([item.request.request_id for item in active], ["decode"])
        self.assertIsNotNone(active[0].pending_token_id)
        self.assertEqual(engine.head_skips, 0)
        return engine

    def test_short_job_starts_while_the_longer_head_keeps_its_place(self) -> None:
        engine = self.engine_with_one_decode(4)
        long = self.enqueue(engine, "long", [2] * (3 * 256 - 1), 1)
        short = self.enqueue(engine, "short", [3] * 64, 8)
        self.tick(engine)
        self.assertEqual(self.ids(engine, "waiting"), ["long"])
        self.assertEqual(self.ids(engine, "active"), ["decode", "short"])
        self.assertEqual(engine.head_skips, 1)
        self.assertFalse(long.done())
        admitted = engine._active_requests[1]
        self.assertEqual(
            admitted.reservation_bytes, engine.request_cache_bytes(64 + 8)
        )

    def test_admit_skip_zero_leaves_the_short_job_behind_the_head(self) -> None:
        engine = self.engine_with_one_decode(4, admit_skip=0)
        self.enqueue(engine, "long", [2] * (3 * 256 - 1), 1)
        short = self.enqueue(engine, "short", [3] * 64, 8)
        self.tick(engine)
        self.assertEqual(self.ids(engine, "waiting"), ["long", "short"])
        self.assertEqual(self.ids(engine, "active"), ["decode"])
        self.assertEqual(engine.head_skips, 0)
        self.assertFalse(short.done())

    def test_four_non_fitting_jobs_can_be_passed_to_reach_a_later_fit(self) -> None:
        engine = self.engine_with_one_decode(4)
        for index in range(5):
            self.enqueue(engine, f"block-{index}", [2] * (3 * 256 - 1), 1)
        self.enqueue(engine, "short", [3] * 64, 8)
        self.tick(engine)
        self.assertEqual(
            self.ids(engine, "waiting"), [f"block-{index}" for index in range(5)]
        )
        self.assertEqual(self.ids(engine, "active"), ["decode", "short"])
        self.assertEqual(engine.head_skips, 1)

    def test_a_fifth_non_fitting_job_stops_the_scan(self) -> None:
        engine = self.engine_with_one_decode(4)
        for index in range(6):
            self.enqueue(engine, f"block-{index}", [2] * (3 * 256 - 1), 1)
        short = self.enqueue(engine, "short", [3] * 64, 8)
        self.tick(engine)
        self.assertEqual(
            self.ids(engine, "waiting"),
            [f"block-{index}" for index in range(6)] + ["short"],
        )
        self.assertEqual(self.ids(engine, "active"), ["decode"])
        self.assertEqual(engine.head_skips, 0)
        self.assertFalse(short.done())

    def test_each_admission_behind_the_head_counts_once(self) -> None:
        engine = self.engine_with_one_decode(4)
        self.enqueue(engine, "long", [2] * (3 * 256 - 1), 1)
        self.enqueue(engine, "short-a", [3] * 64, 8)
        self.enqueue(engine, "short-b", [4] * 64, 8)
        self.tick(engine)
        self.assertEqual(self.ids(engine, "waiting"), ["long"])
        self.assertEqual(self.ids(engine, "active"), ["decode", "short-a", "short-b"])
        self.assertEqual(engine.head_skips, 2)

    def test_skipped_jobs_keep_their_order(self) -> None:
        engine = self.engine_with_one_decode(4)
        self.enqueue(engine, "first-block", [2] * (3 * 256 - 1), 1)
        self.enqueue(engine, "second-block", [4] * (3 * 256 - 1), 1)
        self.enqueue(engine, "short", [3] * 64, 8)
        self.tick(engine)
        self.assertEqual(self.ids(engine, "waiting"), ["first-block", "second-block"])
        self.assertEqual(engine.head_skips, 1)

    def test_a_job_that_has_waited_100ms_is_not_skipped(self) -> None:
        self.assertEqual(ADMIT_SKIP_WAIT_SECONDS, 0.1)
        now = 1_000.0
        engine = self.engine_with_one_decode(4)
        long = self.enqueue(engine, "long", [2] * (3 * 256 - 1), 1)
        short = self.enqueue(engine, "short", [3] * 64, 8)
        self.set_wait(engine, "long", ADMIT_SKIP_WAIT_SECONDS, now)
        self.tick_at(engine, now)
        self.assertEqual(self.ids(engine, "waiting"), ["long", "short"])
        self.assertEqual(self.ids(engine, "active"), ["decode"])
        self.assertEqual(engine.head_skips, 0)
        self.assertFalse(long.done())
        self.assertFalse(short.done())

    def test_a_job_younger_than_100ms_can_be_skipped(self) -> None:
        now = 1_000.0
        engine = self.engine_with_one_decode(4)
        self.enqueue(engine, "long", [2] * (3 * 256 - 1), 1)
        self.enqueue(engine, "short", [3] * 64, 8)
        self.set_wait(engine, "long", ADMIT_SKIP_WAIT_SECONDS - 0.001, now)
        self.tick_at(engine, now)
        self.assertEqual(self.ids(engine, "waiting"), ["long"])
        self.assertEqual(self.ids(engine, "active"), ["decode", "short"])
        self.assertEqual(engine.head_skips, 1)

    def test_an_aged_job_behind_the_head_stops_the_scan(self) -> None:
        now = 1_000.0
        engine = self.engine_with_one_decode(4)
        self.enqueue(engine, "young", [2] * (3 * 256 - 1), 1)
        self.enqueue(engine, "aged", [4] * (3 * 256 - 1), 1)
        short = self.enqueue(engine, "short", [3] * 64, 8)
        self.set_wait(engine, "aged", ADMIT_SKIP_WAIT_SECONDS, now)
        self.tick_at(engine, now)
        self.assertEqual(self.ids(engine, "waiting"), ["young", "aged", "short"])
        self.assertEqual(self.ids(engine, "active"), ["decode"])
        self.assertEqual(engine.head_skips, 0)
        self.assertFalse(short.done())

    def test_admitting_the_head_does_not_count_as_a_skip(self) -> None:
        engine = self.make_engine(budget_tokens=256 * 2)
        self.enqueue(engine, "only", [1, 2, 3, 4], 8)
        self.tick(engine)
        self.assertEqual(self.ids(engine, "active"), ["only"])
        self.assertEqual(engine.head_skips, 0)

    def test_reservation_stays_the_rounded_prompt_plus_max_new_tokens(self) -> None:
        engine = self.make_engine(budget_tokens=256 * 4, prefill_chunk_size=8)
        self.enqueue(engine, "job", [4] * 64, 200)
        self.tick(engine)
        active = engine._active_requests[0]
        self.assertEqual(active.prompt_offset, 8)
        self.assertEqual(active.reservation_bytes, engine.request_cache_bytes(64 + 200))
        self.assertEqual(
            active.reservation_bytes,
            2 * engine.prefix_cache.block_size * engine.capacity.bytes_per_token,
        )

    def test_free_page_floor_blocks_a_reservation_that_fills_the_budget(self) -> None:
        engine = self.engine_with_one_decode(2)
        page_bytes = engine.prefix_cache.block_size * engine.capacity.bytes_per_token
        reserved = engine._reserved_memory_bytes(extra_capacity=64 + 8)
        self.assertLessEqual(reserved, engine.kv_budget_bytes)
        self.assertEqual((engine.kv_budget_bytes - reserved) // page_bytes, 0)
        job = self.enqueue(engine, "fills", [3] * 64, 8)
        self.tick(engine)
        self.assertEqual(self.ids(engine, "waiting"), ["fills"])
        self.assertEqual(self.ids(engine, "active"), ["decode"])
        self.assertEqual(engine.head_skips, 0)
        self.assertFalse(job.done())

    def test_a_prefill_does_not_consume_the_free_page_floor(self) -> None:
        engine = self.make_engine(budget_tokens=256 * 2, prefill_chunk_size=4)
        self.enqueue(engine, "prefill", [1] * 200, 1)
        self.tick(engine)
        active = engine._active_requests[0]
        self.assertIsNone(active.pending_token_id)
        self.assertEqual(active.prompt_offset, 4)
        self.enqueue(engine, "next", [3] * 40, 8)
        self.tick(engine)
        self.assertEqual(self.ids(engine, "active"), ["prefill", "next"])
        self.assertEqual(self.ids(engine, "waiting"), [])
        self.assertEqual(engine.head_skips, 0)

    def test_skip_applies_when_the_head_would_break_the_free_page_floor(self) -> None:
        engine = self.make_engine(slots=8, budget_tokens=256 * 5)
        self.enqueue(engine, "decode-a", [1, 2, 3, 4], 32)
        self.enqueue(engine, "decode-b", [5, 6, 7, 8], 32)
        self.tick(engine)
        self.tick(engine)
        decoding = [
            item.request.request_id
            for item in engine._active_requests
            if item.pending_token_id is not None
        ]
        self.assertEqual(decoding, ["decode-a", "decode-b"])
        page_bytes = engine.prefix_cache.block_size * engine.capacity.bytes_per_token
        wide_capacity = (2 * 256 - 1) + 1
        reserved = engine._reserved_memory_bytes(extra_capacity=wide_capacity)
        self.assertLessEqual(reserved, engine.kv_budget_bytes)
        self.assertEqual((engine.kv_budget_bytes - reserved) // page_bytes, 1)
        wide = self.enqueue(engine, "wide", [2] * (2 * 256 - 1), 1)
        self.enqueue(engine, "short", [3] * 64, 8)
        self.tick(engine)
        self.assertEqual(self.ids(engine, "waiting"), ["wide"])
        self.assertIn("short", self.ids(engine, "active"))
        self.assertEqual(engine.head_skips, 1)
        self.assertFalse(wide.done())

    def test_a_cancelled_job_in_the_window_does_not_block_a_later_fit(self) -> None:
        engine = self.engine_with_one_decode(4)
        self.enqueue(engine, "long", [2] * (3 * 256 - 1), 1)
        cancelled = self.enqueue(engine, "cancelled", [4] * (3 * 256 - 1), 1)
        self.enqueue(engine, "short", [3] * 64, 8)
        self.assertTrue(cancelled.cancel())
        self.tick(engine)
        self.assertEqual(self.ids(engine, "waiting"), ["long"])
        self.assertEqual(self.ids(engine, "active"), ["decode", "short"])
        self.assertEqual(engine.head_skips, 1)
        self.assertTrue(cancelled.cancelled())


class AdmitSkipConfigTests(unittest.TestCase):
    def test_admit_skip_defaults_to_four_and_rejects_a_negative_value(self) -> None:
        self.assertEqual(PylonConfig().admit_skip, 4)
        with (
            patch("pylon.config.load_dotenv"),
            patch.dict(os.environ, {"PYLON_ADMIT_SKIP": "0"}),
        ):
            self.assertEqual(get_config().admit_skip, 0)
        with (
            patch("pylon.config.load_dotenv"),
            patch.dict(os.environ, {"PYLON_ADMIT_SKIP": "4"}),
        ):
            self.assertEqual(get_config().admit_skip, 4)
        for raw in ("-1", "true"):
            with (
                patch("pylon.config.load_dotenv"),
                patch.dict(os.environ, {"PYLON_ADMIT_SKIP": raw}),
            ):
                with self.assertRaises(ValueError):
                    get_config()
        with self.assertRaises(ValueError):
            PylonConfig(admit_skip=-1)
        with self.assertRaises(ValueError):
            PylonConfig(admit_skip=True)

    def test_eager_and_pylon_columns_force_skip_off(self) -> None:
        base = {"PYLON_ADMIT_SKIP": "9", "PYLON_PREFIX_CACHE": "true"}
        eager = system_environ(base, "eager")
        pylon = system_environ(base, "pylon")
        self.assertEqual(eager["PYLON_ADMIT_SKIP"], "0")
        self.assertEqual(pylon["PYLON_ADMIT_SKIP"], "0")
        self.assertEqual(result_flags("eager", eager)["admit_skip"], 0)
        self.assertEqual(result_flags("eager", {"PYLON_ADMIT_SKIP": "4"})["admit_skip"], 0)
        self.assertEqual(result_flags("pylon")["admit_skip"], 0)
        self.assertEqual(result_flags("pylon", pylon)["admit_skip"], 0)
        self.assertEqual(result_flags("vllm")["admit_skip"], 0)


if __name__ == "__main__":
    unittest.main()
