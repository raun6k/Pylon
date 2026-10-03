import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.run import result_flags, system_environ
from pylon.config import PylonConfig, get_config
from pylon.decode.graph import (
    STATIC_BUFFER_NAMES,
    CapturedDecode,
    DecodeGraphSet,
    copy_buffers,
    static_decode_values,
)
from pylon.decode.runner import Decoder
from pylon.decode.warmup import capture_each_batch_size
from pylon.engine import Engine
from pylon.frontend import TextGenerator
from pylon.kv.cache import PagedBatchCache, PagedKVCache
from pylon.kv.pool import KVPagePool
from pylon.model.config import Qwen3Config
from pylon.model.model import Qwen3Model


def tiny_config(**overrides) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=64,
        context_length=512,
        hidden_size=16,
        n_heads=2,
        n_layers=2,
        hidden_dim=32,
        head_dim=8,
        n_kv_heads=1,
        dtype=torch.float32,
    )
    if overrides:
        config = replace(config, **overrides)
    return config


class CudaGraphDecodeTests(unittest.TestCase):
    def test_cuda_graphs_default_off(self) -> None:
        self.assertFalse(PylonConfig().cuda_graphs)
        self.assertFalse(PylonConfig().torch_compile)
        with (
            patch("pylon.config.load_dotenv"),
            patch.dict(os.environ, {"PYLON_CUDA_GRAPHS": "true"}),
        ):
            self.assertTrue(get_config().cuda_graphs)
        with (
            patch("pylon.config.load_dotenv"),
            patch.dict(os.environ, {"PYLON_CUDA_GRAPHS": "false"}),
        ):
            self.assertFalse(get_config().cuda_graphs)
        with (
            patch("pylon.config.load_dotenv"),
            patch.dict(os.environ, {"PYLON_CUDA_GRAPHS": "yes"}),
        ):
            with self.assertRaises(ValueError):
                get_config()

    def test_static_buffers_match_eager_decode_metadata(self) -> None:
        self.assertEqual(
            STATIC_BUFFER_NAMES,
            (
                "token_ids",
                "positions",
                "block_table",
                "write_slots",
                "cu_seq_q",
                "cu_seq_k",
                "seqused_k",
            ),
        )
        pool = KVPagePool(tiny_config(), 6, device=torch.device("cpu"))
        first = PagedKVCache(pool, 512)
        second = PagedKVCache(pool, 512)
        second.pages.extend(pool.acquire(1))
        second.length = 256
        batch = PagedBatchCache((first, second))
        batch.prepare(1)
        width = batch.block_table.shape[1]
        values = static_decode_values(
            [first, second], [4, 9], page_size=pool.page_size, max_pages=width
        )
        self.assertEqual(len(values["token_ids"]), 2)
        self.assertEqual(values["cu_seq_q"], [0, 1, 2])
        self.assertEqual(values["positions"], [[0], [256]])
        self.assertEqual(values["seqused_k"], [1, 257])
        self.assertEqual(values["cu_seq_k"], [0, 1, 258])
        self.assertEqual(values["block_table"], batch.block_table.tolist())
        self.assertEqual(values["write_slots"], batch.write_slots.tolist())
        self.assertEqual(values["token_ids"], [[4], [9]])
        self.assertEqual(batch.cu_seq_q.tolist(), values["cu_seq_q"])
        self.assertEqual(batch.cu_seq_k.tolist(), values["cu_seq_k"])
        self.assertEqual(batch.seqused_k.tolist(), values["seqused_k"])

    def test_fill_is_non_blocking_and_happens_before_replay(self) -> None:
        pool = KVPagePool(tiny_config(), 6, device=torch.device("cpu"))
        caches = [PagedKVCache(pool, 512), PagedKVCache(pool, 512)]
        caches[1].pages.extend(pool.acquire(1))
        caches[1].length = 256
        captured = CapturedDecode(
            2,
            2,
            512,
            pool.page_size,
            torch.device("cpu"),
            4,
            torch.float32,
            pin_memory=False,
        )
        seen: list[list[int]] = []

        def replay() -> None:
            seen.append(captured.token_ids[:, 0].tolist())

        captured.graph = SimpleNamespace(replay=replay)
        calls: list[bool] = []
        real_copy = torch.Tensor.copy_

        def spy(tensor, source, non_blocking=False):
            calls.append(non_blocking)
            return real_copy(tensor, source, non_blocking=non_blocking)

        with (
            patch("torch.cuda.synchronize", side_effect=AssertionError("host sync")),
            patch.object(torch.Tensor, "copy_", spy),
            patch.object(Decoder, "_sample", side_effect=AssertionError("sample")),
        ):
            captured.run(caches, [4, 9])
            captured.run(caches, [7, 8])
        self.assertEqual(seen, [[4, 9], [7, 8]])
        self.assertTrue(calls)
        self.assertTrue(all(calls))
        self.assertEqual(caches[0].length, 2)
        self.assertEqual(caches[1].length, 258)
        self.assertEqual(captured.seqused_k.tolist(), [2, 258])
        self.assertEqual(captured.positions.tolist(), [[1], [257]])

    def test_batch_of_three_uses_the_size_three_graph(self) -> None:
        graphs = DecodeGraphSet()
        graphs._by_size[8] = SimpleNamespace(
            max_k=32,
            run=lambda caches, token_ids: (_ for _ in ()).throw(
                AssertionError("padded into a larger batch")
            ),
        )
        self.assertIsNone(graphs.replay([SimpleNamespace(length=2)] * 3, [1, 1, 1]))
        seen: list[tuple[int, list[int]]] = []
        graphs._by_size[3] = SimpleNamespace(
            max_k=32,
            run=lambda caches, token_ids: seen.append((len(caches), list(token_ids)))
            or torch.zeros(3, 2),
        )
        logits = graphs.replay([SimpleNamespace(length=2)] * 3, [4, 5, 6])
        self.assertEqual(seen, [(3, [4, 5, 6])])
        self.assertEqual(tuple(logits.shape), (3, 2))
        graphs._by_size[1] = SimpleNamespace(
            max_k=4,
            run=lambda caches, token_ids: (_ for _ in ()).throw(
                AssertionError("replayed past the captured length")
            ),
        )
        self.assertIsNone(graphs.replay([SimpleNamespace(length=4)], [1]))

    def test_failed_capture_stays_eager_and_is_not_retried(self) -> None:
        graphs = DecodeGraphSet()

        def capture_one(batch_size: int) -> None:
            if batch_size == 2:
                raise RuntimeError("capture failed")
            graphs._by_size[batch_size] = object()

        with self.assertLogs("pylon", level="ERROR") as logged:
            sizes = capture_each_batch_size(graphs, (1, 2, 3), capture_one)
        self.assertEqual(sizes, (1, 3))
        self.assertTrue(any("batch_size=2" in line for line in logged.output))
        self.assertFalse(graphs.capture_allowed(2))
        again: list[int] = []
        capture_each_batch_size(graphs, (1, 2, 3), again.append)
        self.assertEqual(again, [])
        self.assertIsNone(graphs.replay([SimpleNamespace(length=1)] * 2, [1, 1]))

    def test_health_lists_captured_sizes(self) -> None:
        graphs = DecodeGraphSet()
        graphs._by_size[1] = object()
        graphs.mark_failed(2)
        graphs._by_size[3] = object()
        engine = Engine.__new__(Engine)
        engine.decoder = SimpleNamespace(graphs=graphs)
        self.assertEqual(engine.cuda_graph_batch_sizes, (1, 3))

        frontend = TextGenerator.__new__(TextGenerator)
        frontend.engine = SimpleNamespace(
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            model_revision="rev",
            report=None,
            scheduler_snapshot=lambda: {"waiting": [], "active": []},
            cuda_graph_batch_sizes=engine.cuda_graph_batch_sizes,
        )
        frontend._decode_warmup_batch_sizes = (1, 2, 3)
        body = frontend.health()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["warmup_batch_sizes"], [1, 2, 3])
        self.assertEqual(body["cuda_graph_batch_sizes"], [1, 3])

        skipped = Engine.__new__(Engine)
        skipped._config = PylonConfig(cuda_graphs=False)
        skipped.decoder = SimpleNamespace(device=torch.device("cuda"))
        self.assertEqual(Engine.capture_decode_graphs(skipped, [1, 2]), ())
        enabled = Engine.__new__(Engine)
        enabled._config = PylonConfig(cuda_graphs=True)
        enabled.decoder = SimpleNamespace(device=torch.device("cpu"), graphs=None)
        self.assertEqual(Engine.capture_decode_graphs(enabled, [1, 2]), ())

    def test_decode_replay_returns_logits_and_prefill_stays_eager(self) -> None:
        config = tiny_config()
        model = Qwen3Model(config).eval()
        decoder = Decoder(model)
        pool = KVPagePool(config, 8, device=torch.device("cpu"))
        decoder.page_pool = pool
        sentinel = torch.arange(3 * config.vocab_size, dtype=torch.float32).reshape(
            3, config.vocab_size
        )

        class Graphs:
            def replay(self, caches, token_ids):
                self.size = len(caches)
                self.tokens = list(token_ids)
                return sentinel

            def capture(self, *args, **kwargs):
                raise AssertionError("capture retried on the request path")

        graphs = Graphs()
        decoder.graphs = graphs
        caches = [PagedKVCache(pool, 256) for _ in range(3)]
        with patch.object(Decoder, "_sample", side_effect=AssertionError("sample")):
            logits = decoder.decode_caches(caches, [1, 2, 3])
        self.assertIs(logits, sentinel)
        self.assertEqual(graphs.size, 3)
        self.assertEqual(graphs.tokens, [1, 2, 3])

        class EagerFallback:
            def __init__(self) -> None:
                self.sizes: list[int] = []

            def replay(self, caches, token_ids):
                del token_ids
                self.sizes.append(len(caches))
                return None

        fallback = EagerFallback()
        decoder.graphs = fallback
        one = PagedKVCache(pool, 256)
        eager = decoder.decode_caches([one], [3])
        self.assertEqual(fallback.sizes, [1])
        self.assertEqual(tuple(eager.shape), (1, config.vocab_size))
        self.assertEqual(one.length, 1)

        class Boom:
            def replay(self, *args, **kwargs):
                raise AssertionError("prefill used a decode graph")

        decoder.graphs = Boom()
        prompt = PagedKVCache(pool, 256)
        decoder.prefill_chunk(prompt, [1, 2, 3])
        other = PagedKVCache(pool, 256)
        packed = decoder.packed_caches([prompt, other], [[4], [5, 6]])
        self.assertEqual(tuple(packed.shape), (2, config.vocab_size))

    def test_static_forward_skips_metadata_rebuild(self) -> None:
        config = tiny_config()
        model = Qwen3Model(config).eval()
        pool = KVPagePool(config, 4, device=torch.device("cpu"))
        cache = PagedKVCache(pool, 256)
        batch = PagedBatchCache((cache,))
        batch.prepare(1)
        captured = CapturedDecode(
            1,
            batch.block_table.shape[1],
            256,
            pool.page_size,
            torch.device("cpu"),
            config.vocab_size,
            config.dtype,
            pin_memory=False,
        )
        values = static_decode_values(
            [cache], [3], page_size=pool.page_size, max_pages=captured.max_pages
        )
        copy_buffers(captured.buffers, captured.host, values)
        batch.graph_static = True
        batch.max_q = 1
        batch.max_k = captured.max_k
        batch.block_table = captured.block_table
        batch.write_slots = captured.write_slots
        batch.cu_seq_q = captured.cu_seq_q
        batch.cu_seq_k = captured.cu_seq_k
        batch.seqused_k = captured.seqused_k
        captured.logits.fill_(float("nan"))
        with (
            patch.object(PagedBatchCache, "slot_lengths", side_effect=AssertionError("sync")),
            patch.object(Decoder, "_sample", side_effect=AssertionError("sample")),
            patch("torch.cuda.synchronize", side_effect=AssertionError("host sync")),
        ):
            captured.device_forward(model, batch, (0,))
        self.assertEqual(cache.length, 1)
        self.assertTrue(torch.isfinite(captured.logits).all())

        eager_cache = PagedKVCache(pool, 256)
        eager_batch = PagedBatchCache((eager_cache,))
        eager_batch.prepare(1)
        calls: list[int] = []
        real = PagedBatchCache.slot_lengths

        def wrapped(self, slots):
            calls.append(1)
            return real(self, slots)

        with patch.object(PagedBatchCache, "slot_lengths", wrapped):
            model(
                torch.tensor([[3]]),
                cache=eager_batch,
                position_ids=torch.tensor([[0]]),
                cache_slots=(0,),
            )
        self.assertTrue(calls)

    def test_harness_enables_graphs_only_for_pylon(self) -> None:
        base = {"PYLON_PREFIX_CACHE": "true"}
        eager = system_environ(base, "eager")
        pylon = system_environ(base, "pylon")
        vllm = system_environ(base, "vllm")
        self.assertEqual(eager["PYLON_CUDA_GRAPHS"], "false")
        self.assertEqual(eager["PYLON_PREFIX_CACHE"], "false")
        self.assertEqual(pylon["PYLON_CUDA_GRAPHS"], "true")
        self.assertEqual(pylon["PYLON_PREFIX_CACHE"], "false")
        self.assertEqual(vllm, base)
        self.assertFalse(result_flags("eager")["cuda_graphs"])
        self.assertTrue(result_flags("pylon")["cuda_graphs"])
        self.assertFalse(result_flags("pylon")["prefix_cache"])
        self.assertFalse(result_flags("pylon")["speculation"])
        self.assertEqual(result_flags("pylon")["admit_skip"], 0)
        self.assertTrue(result_flags("vllm")["prefix_cache"])

    def test_replay_matches_eager_decode_on_cuda(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("CUDA graph decode requires NVIDIA SM80 or newer")
        from pylon.decode.warmup import capture_decode_graphs

        torch.manual_seed(0)
        config = tiny_config(dtype=torch.float16)
        device = torch.device("cuda")
        model = Qwen3Model(config).to(device).eval()
        decoder = Decoder(model)
        pool = KVPagePool(config, 16, device=device)
        decoder.page_pool = pool
        capture_decode_graphs(decoder, token_id=3, max_batch_size=2, max_tokens=512)
        self.assertEqual(decoder.graphs.captured_sizes(), (1, 2))

        def fresh(lengths: list[int]) -> list[PagedKVCache]:
            caches = [PagedKVCache(pool, 512) for _ in lengths]
            for cache, length in zip(caches, lengths, strict=True):
                decoder.prefill_chunk(cache, [3] * length)
            return caches

        saved = decoder.graphs
        calls: list[tuple[int, bool]] = []
        original = saved.replay

        def counting(caches, token_ids):
            result = original(caches, token_ids)
            calls.append((len(caches), result is not None))
            return result

        saved.replay = counting
        try:
            for size, lengths in ((1, [4]), (2, [3, 6])):
                calls.clear()
                eager_caches = fresh(lengths)
                graph_caches = fresh(lengths)
                tokens = [7] * size
                decoder.graphs = None
                eager_logits = decoder.decode_caches(eager_caches, tokens).clone()
                decoder.graphs = saved
                graph_logits = decoder.decode_caches(graph_caches, tokens).clone()
                torch.testing.assert_close(graph_logits, eager_logits, rtol=1e-2, atol=1e-2)
                self.assertEqual(calls, [(size, True)])
                self.assertEqual(tuple(graph_logits.shape), (size, config.vocab_size))
                calls.clear()
                tokens = [8] * size
                decoder.graphs = None
                eager_next = decoder.decode_caches(eager_caches, tokens).clone()
                decoder.graphs = saved
                graph_next = decoder.decode_caches(graph_caches, tokens).clone()
                torch.testing.assert_close(graph_next, eager_next, rtol=1e-2, atol=1e-2)
                self.assertEqual(calls, [(size, True)])
        finally:
            decoder.graphs = None


if __name__ == "__main__":
    unittest.main()
