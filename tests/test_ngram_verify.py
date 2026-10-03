import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

from benchmarks.run import _print_summary, result_flags, summarize, system_environ
from pylon.api.types import Sampling
from pylon.config import PylonConfig, get_config
from pylon.decode.runner import Decoder
from pylon.engine import Engine
from pylon.kv.budget import CacheCapacity, MemoryReport
from pylon.kv.cache import PackedBatchCache, PagedKVCache
from pylon.kv.pool import KVPagePool
from pylon.model.config import Qwen3Config
from pylon.model.model import Qwen3Model
from pylon.spec.ngram import (
    NGRAM_KEY_LENGTH,
    NGRAM_ORDER,
    PromptNgram,
    accept_greedy,
    verification_tokens,
)
from test_continuous_batching import FakeDecoder


PROMPT = [2, 3, 1, 9, 5, 2, 3]
DRAFTS = (9, 5, 2, 3)
PARTIAL = [1, 2, 3, 1, 7, 2, 3, 1, 2]


class RecordingGraphs:
    def __init__(self) -> None:
        self.sizes: list[int] = []

    def replay(self, caches, token_ids) -> None:
        del token_ids
        self.sizes.append(len(caches))
        return None


class DraftDecoder(FakeDecoder):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.verify_argmax: list[list[list[int]]] = []
        self.graphs = RecordingGraphs()

    def verify_drafts(self, caches, chunks: list[list[int]]) -> list[torch.Tensor]:
        self.forwards.append(("verify", [list(chunk) for chunk in chunks]))
        batch = PackedBatchCache(caches)
        batch.prepare_packed([len(chunk) for chunk in chunks])
        batch.advance_packed()
        planned = self.verify_argmax.pop(0)
        return self._position_logits(planned)

    def _position_logits(self, rows: list[list[int]]) -> list[torch.Tensor]:
        vocab = self.model.config.vocab_size
        tensors = []
        for tokens in rows:
            logits = torch.full((len(tokens), vocab), -1.0e4)
            for index, token_id in enumerate(tokens):
                logits[index, token_id] = 10.0
            tensors.append(logits)
        return tensors

    def _sample(self, logits: torch.Tensor, sampling: Sampling) -> torch.Tensor:
        del sampling
        return logits.argmax(dim=-1, keepdim=True)


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        context_length=128,
        hidden_size=16,
        n_heads=2,
        n_layers=2,
        hidden_dim=32,
        head_dim=8,
        n_kv_heads=1,
        dtype=torch.float32,
    )


class PromptNgramTests(unittest.TestCase):
    def test_order_is_four_and_the_key_is_the_last_three_tokens(self) -> None:
        self.assertEqual(NGRAM_ORDER, 4)
        self.assertEqual(NGRAM_KEY_LENGTH, 3)
        table = PromptNgram(PROMPT)
        self.assertEqual(table.propose([5, 2, 3, 1], 4), DRAFTS)
        self.assertEqual(table.propose([8, 8, 1], 4), ())
        self.assertEqual(table.propose([2, 3], 4), ())
        self.assertEqual(verification_tokens(1, DRAFTS), [1, 9, 5, 2])

    def test_rightmost_copy_wins_and_there_is_no_shorter_key(self) -> None:
        table = PromptNgram([1, 2, 3, 4, 1, 2, 3, 9])
        self.assertEqual(table.propose([0, 1, 2, 3], 4), (9,))
        self.assertEqual(table.propose([9, 2, 3], 4), ())

    def test_a_hit_chains_on_the_new_suffix_up_to_four(self) -> None:
        table = PromptNgram([1, 2, 3, 4, 5])
        self.assertEqual(table.propose([0, 1, 2, 3], 4), (4, 5))
        self.assertEqual(table.propose([0, 1, 2, 3], 5), (4, 5))
        self.assertEqual(len(table.propose([5, 2, 3, 1], 8)), 0)
        chained = PromptNgram(PROMPT)
        self.assertEqual(chained.propose([9, 5, 2, 3, 1], 8), DRAFTS)

    def test_generated_tokens_are_not_inserted_and_tables_are_not_shared(self) -> None:
        table = PromptNgram(PROMPT)
        other = PromptNgram([1, 2, 3, 8])
        self.assertEqual(table.propose([5, 2, 3], 4), ())
        extended = PromptNgram([*PROMPT, 1])
        self.assertEqual(extended.propose([5, 2, 3], 1), (1,))
        self.assertEqual(table.propose([5, 2, 3], 1), ())
        self.assertEqual(other.propose([1, 2, 3], 1), (8,))
        self.assertEqual(table.propose([1, 2, 3], 1), ())
        self.assertIsNot(table, other)

    def test_first_mismatch_is_the_bonus_and_a_full_match_has_no_fifth(self) -> None:
        self.assertEqual(accept_greedy([9, 5, 8, 0], DRAFTS), (9, 5, 8))
        self.assertEqual(accept_greedy([7, 1, 1, 1], DRAFTS), (7,))
        matched = accept_greedy(list(DRAFTS), DRAFTS)
        self.assertEqual(matched, DRAFTS)
        self.assertEqual(len(matched), 4)

    def test_speculate_k_defaults_to_one_and_rejects_values_past_four(self) -> None:
        self.assertEqual(PylonConfig().speculate_k, 1)
        with (
            patch("pylon.config.load_dotenv"),
            patch.dict(os.environ, {"PYLON_SPECULATE_K": "4"}),
        ):
            self.assertEqual(get_config().speculate_k, 4)
        for raw in ("0", "5", "-1", "true"):
            with (
                patch("pylon.config.load_dotenv"),
                patch.dict(os.environ, {"PYLON_SPECULATE_K": raw}),
            ):
                with self.assertRaises(ValueError):
                    get_config()

    def test_eager_column_forces_speculation_off(self) -> None:
        base = {"PYLON_SPECULATE_K": "4", "PYLON_PREFIX_CACHE": "true"}
        eager = system_environ(base, "eager")
        pylon = system_environ(base, "pylon")
        self.assertEqual(eager["PYLON_SPECULATE_K"], "1")
        self.assertEqual(pylon["PYLON_SPECULATE_K"], "4")
        self.assertFalse(result_flags("eager", eager)["speculation"])
        self.assertTrue(result_flags("pylon", pylon)["speculation"])
        self.assertFalse(result_flags("pylon")["speculation"])
        self.assertFalse(result_flags("vllm", {})["speculation"])

    def test_acceptance_is_mean_tokens_per_step_and_empty_when_off(self) -> None:
        samples = [
            {
                "ttft": 1.0,
                "inter_token": 0.2,
                "completion_tokens": 5,
                "prompt_tokens": 7,
                "cached_tokens": 0,
                "accepted_tokens_per_step": [4],
            },
            {
                "ttft": 1.0,
                "inter_token": 0.2,
                "completion_tokens": 3,
                "prompt_tokens": 7,
                "cached_tokens": 0,
                "accepted_tokens_per_step": [1, 1],
            },
        ]
        summary = summarize(samples, elapsed_seconds=2.0)
        self.assertEqual(summary["mean_accepted_tokens"], 2.0)
        quiet = summarize(
            [
                {
                    "ttft": 1.0,
                    "inter_token": None,
                    "completion_tokens": 1,
                    "prompt_tokens": 4,
                    "cached_tokens": 0,
                }
            ],
            elapsed_seconds=1.0,
        )
        self.assertIsNone(quiet["mean_accepted_tokens"])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _print_summary("pylon", "decode_heavy", 1, summary, None)
            _print_summary("eager", "decode_heavy", 1, quiet, None)
        printed = buffer.getvalue()
        self.assertIn("output_tokens_per_second=", printed)
        self.assertIn("mean_accepted_tokens=2.0000", printed)
        self.assertEqual(printed.count("mean_accepted_tokens="), 1)


class NgramWaveTests(unittest.TestCase):
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
        self.decoder = DraftDecoder(self.config)

    def make_engine(self, *, speculate_k: int = 4, slots: int = 4) -> Engine:
        element_size = torch.empty((), dtype=self.config.dtype).element_size()
        bytes_per_token = (
            self.config.n_layers
            * 2
            * self.config.n_kv_heads
            * self.config.head_dim
            * element_size
        )
        budget_tokens = 8192
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
                    prefill_chunk_size=256,
                    batch_wait_ms=0,
                    prefix_cache=False,
                    speculate_k=speculate_k,
                ),
                decoder=self.decoder,
                capacity=capacity,
                report=report,
            )
        self.addCleanup(engine.close)
        return engine

    def enqueue(
        self,
        engine: Engine,
        name: str,
        prompt: list[int],
        output_limit: int,
        *,
        temperature: float = 0,
        eos: int = 0,
    ):
        return engine.enqueue(
            prompt,
            eos,
            Sampling(temperature=temperature, top_p=1, max_new_tokens=output_limit),
            request_id=name,
        )

    def tick(self, engine: Engine) -> None:
        engine._continuous_tick(engine._scheduler)

    def test_a_miss_is_one_ordinary_decode(self) -> None:
        self.decoder.plan(7)
        engine = self.make_engine()
        future = self.enqueue(engine, "miss", [1, 2, 3, 4], 4)
        self.tick(engine)
        self.decoder.forwards.clear()
        self.decoder.graphs.sizes.clear()
        self.tick(engine)
        self.assertEqual([name for name, _chunks in self.decoder.forwards], ["decode"])
        self.assertEqual(self.decoder.forwards[0][1], [[7]])
        self.assertEqual(self.decoder.graphs.sizes, [1])
        self.assertFalse(future.done())

    def test_k1_stays_on_the_decode_graph_even_when_the_prompt_would_hit(self) -> None:
        self.decoder.plan(1)
        engine = self.make_engine(speculate_k=1)
        self.enqueue(engine, "off", PROMPT, 8)
        self.tick(engine)
        self.assertIsNone(engine._active_requests[0].prompt_ngram)
        self.decoder.forwards.clear()
        self.decoder.graphs.sizes.clear()
        self.tick(engine)
        self.assertEqual([name for name, _chunks in self.decoder.forwards], ["decode"])
        self.assertEqual(self.decoder.graphs.sizes, [1])

    def test_decode_wave_verifies_four_drafts_in_one_eager_forward(self) -> None:
        self.decoder.plan(1)
        self.decoder.verify_argmax.append([list(DRAFTS)])
        engine = self.make_engine()
        future = self.enqueue(engine, "hit", PROMPT, 8)
        self.tick(engine)
        self.assertEqual([name for name, _chunks in self.decoder.forwards], ["prefill"])
        self.decoder.forwards.clear()
        self.decoder.graphs.sizes.clear()
        self.tick(engine)
        active = engine._active_requests[0]
        self.assertEqual(self.decoder.forwards, [("verify", [[1, 9, 5, 2]])])
        self.assertEqual(self.decoder.graphs.sizes, [])
        self.assertEqual(active.output_ids, [1, *DRAFTS])
        self.assertEqual(active.pending_token_id, DRAFTS[-1])
        self.assertEqual(active.cache.length, len(PROMPT) + 4)
        self.assertEqual(active.accepted_tokens_per_step, [4])
        self.assertEqual(active.prompt_ngram.propose([5, 2, 3], 4), ())
        self.assertFalse(future.done())

    def test_a_shorter_chain_is_not_padded_to_four(self) -> None:
        self.decoder.plan(3)
        self.decoder.verify_argmax.append([[1, 2]])
        engine = self.make_engine()
        self.enqueue(engine, "partial", PARTIAL, 8)
        self.tick(engine)
        self.decoder.forwards.clear()
        self.tick(engine)
        active = engine._active_requests[0]
        self.assertEqual(self.decoder.forwards, [("verify", [[3, 1]])])
        self.assertEqual(active.output_ids, [3, 1, 2])
        self.assertEqual(active.cache.length, len(PARTIAL) + 2)
        self.assertEqual(active.accepted_tokens_per_step, [2])

    def test_remaining_tokens_cap_the_draft_run(self) -> None:
        self.decoder.plan(1)
        self.decoder.verify_argmax.append([[9]])
        engine = self.make_engine()
        future = self.enqueue(engine, "short", PROMPT, 2)
        self.tick(engine)
        self.decoder.forwards.clear()
        self.tick(engine)
        self.assertEqual(self.decoder.forwards, [("verify", [[1]])])
        result = future.result(timeout=0)
        self.assertEqual(result.output_ids, [1, 9])
        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.accepted_tokens_per_step, (1,))

    def test_first_mismatch_keeps_the_bonus_and_rewinds_the_tail(self) -> None:
        self.decoder.plan(1)
        self.decoder.verify_argmax.append([[9, 5, 8, 0]])
        engine = self.make_engine()
        self.enqueue(engine, "bonus", PROMPT, 8)
        self.tick(engine)
        self.decoder.forwards.clear()
        self.decoder.graphs.sizes.clear()
        self.tick(engine)
        active = engine._active_requests[0]
        self.assertEqual(self.decoder.graphs.sizes, [])
        self.assertEqual(active.output_ids, [1, 9, 5, 8])
        self.assertEqual(active.pending_token_id, 8)
        self.assertNotIn(2, active.output_ids[1:])
        self.assertEqual(active.cache.length, len(PROMPT) + 3)
        self.assertEqual(active.accepted_tokens_per_step, [3])

    def test_an_eos_bonus_stops_without_appending_the_stop_token(self) -> None:
        self.decoder.plan(1)
        self.decoder.verify_argmax.append([[0, 1, 1, 1]])
        engine = self.make_engine()
        future = self.enqueue(engine, "eos", PROMPT, 8, eos=0)
        self.tick(engine)
        self.tick(engine)
        result = future.result(timeout=0)
        self.assertEqual(result.finish_reason, "eos")
        self.assertEqual(result.output_ids, [1])
        self.assertEqual(result.accepted_tokens_per_step, (0,))

    def test_nonzero_temperature_does_not_verify(self) -> None:
        self.decoder.plan(1)
        engine = self.make_engine()
        self.enqueue(engine, "sample", PROMPT, 4, temperature=0.2)
        self.tick(engine)
        self.decoder.forwards.clear()
        self.decoder.graphs.sizes.clear()
        self.tick(engine)
        self.assertEqual([name for name, _chunks in self.decoder.forwards], ["decode"])
        self.assertEqual(self.decoder.graphs.sizes, [1])
        self.assertEqual(self.decoder.verify_argmax, [])

    def test_requests_do_not_share_a_prompt_table(self) -> None:
        self.decoder.plan(1, 1)
        self.decoder.verify_argmax.append([list(DRAFTS)])
        engine = self.make_engine(slots=2)
        self.enqueue(engine, "alpha", PROMPT, 8)
        self.enqueue(engine, "beta", [8, 8, 2, 3], 8)
        self.tick(engine)
        alpha, beta = engine._active_requests
        self.assertIsNot(alpha.prompt_ngram, beta.prompt_ngram)
        self.assertEqual(alpha.prompt_ngram.propose([2, 3, 1], 1), (9,))
        self.assertEqual(beta.prompt_ngram.propose([2, 3, 1], 1), ())
        self.tick(engine)
        self.decoder.forwards.clear()
        self.decoder.graphs.sizes.clear()
        self.tick(engine)
        self.assertEqual(self.decoder.forwards, [("decode", [[3], [1]])])
        self.assertEqual(self.decoder.graphs.sizes, [2])


class VerifyForwardTests(unittest.TestCase):
    def test_rewind_drops_pages_that_are_past_the_new_length(self) -> None:
        config = Qwen3Config(
            vocab_size=32,
            context_length=1024,
            hidden_size=16,
            n_heads=2,
            n_layers=2,
            hidden_dim=32,
            head_dim=8,
            n_kv_heads=1,
            dtype=torch.float32,
        )
        pool = KVPagePool(config, 4, device=torch.device("cpu"))
        cache = PagedKVCache(pool, 1024)
        cache.pages.extend(pool.acquire(2))
        cache.length = 257
        cache.rewind(1)
        self.assertEqual(cache.length, 256)
        self.assertEqual(len(cache.pages), 1)
        self.assertEqual(pool.free_pages, 3)
        cache.rewind(256)
        self.assertEqual(cache.length, 0)
        self.assertEqual(cache.pages, [])
        self.assertEqual(pool.free_pages, 4)

    def test_verify_logits_match_each_draft_position_and_skip_the_graph(self) -> None:
        torch.manual_seed(0)
        config = tiny_config()
        model = Qwen3Model(config).eval()
        decoder = Decoder(model)
        pool = KVPagePool(config, 16, device=torch.device("cpu"))
        decoder.page_pool = pool
        prompt = [1, 2, 3, 4, 5, 6]
        query = [7, 8, 9, 10]

        def prefill() -> PagedKVCache:
            cache = PagedKVCache(pool, 64)
            decoder.prefill_chunk(cache, prompt)
            return cache

        stepwise = prefill()
        verified_cache = prefill()
        packed_cache = prefill()
        left = prefill()
        right = prefill()
        left_alone = prefill()
        right_alone = prefill()
        rewound = prefill()
        reference = prefill()

        stepwise_rows = [
            decoder.decode_caches([stepwise], [token]).squeeze(0) for token in query
        ]
        verified = decoder.verify_drafts([verified_cache], [query])
        self.assertEqual(tuple(verified[0].shape), (4, config.vocab_size))
        for row, expected in zip(verified[0], stepwise_rows, strict=True):
            torch.testing.assert_close(row, expected, rtol=1e-4, atol=1e-5)
        self.assertEqual(verified_cache.length, len(prompt) + 4)

        packed = decoder.packed_caches([packed_cache], [query])
        self.assertEqual(tuple(packed.shape), (1, config.vocab_size))
        torch.testing.assert_close(packed[0], verified[0][-1], rtol=1e-4, atol=1e-5)

        batched = decoder.verify_drafts([left, right], [query, [3, 4]])
        alone_left = decoder.verify_drafts([left_alone], [query])
        alone_right = decoder.verify_drafts([right_alone], [[3, 4]])
        torch.testing.assert_close(batched[0], alone_left[0], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(batched[1], alone_right[0], rtol=1e-4, atol=1e-5)

        decoder.verify_drafts([rewound], [query])
        rewound.rewind(2)
        decoder.decode_caches([reference], [query[0]])
        decoder.decode_caches([reference], [query[1]])
        bonus = decoder.decode_caches([rewound], [11])
        expected_bonus = decoder.decode_caches([reference], [11])
        torch.testing.assert_close(bonus, expected_bonus, rtol=1e-4, atol=1e-5)

        class Bomb:
            def replay(self, caches, token_ids):
                raise AssertionError("verify replayed a CUDA graph")

        fresh = prefill()
        decoder.graphs = Bomb()
        decoder.verify_drafts([fresh], [query])
        with self.assertRaises(AssertionError):
            decoder.decode_caches([fresh], [11])


if __name__ == "__main__":
    unittest.main()
