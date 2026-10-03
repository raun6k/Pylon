import unittest
from unittest.mock import patch

import torch

from pylon.model.config import Qwen3Config
from pylon.decode.runner import Decoder
from pylon.model.model import Qwen3Model
from pylon.api.types import Sampling
from pylon.decode.warmup import DECODE_WARMUP_STEPS, warm_decode


class DenseDecodeTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            self.model = Qwen3Model(
                Qwen3Config(
                    vocab_size=64,
                    context_length=128,
                    hidden_size=16,
                    n_heads=2,
                    n_layers=1,
                    hidden_dim=32,
                    head_dim=8,
                    n_kv_heads=1,
                    dtype=torch.float32,
                )
            ).eval()
        self.sampling = Sampling(temperature=0, top_p=1, max_new_tokens=12)

    def reference_decode(self, caches, token=4):
        with torch.inference_mode():
            return torch.cat([
                self.model(torch.tensor([[token]], device=self.model.token_embedding.weight.device), cache=cache)[:, -1, :]
                for cache in caches
            ])

    def test_dense_logits_match_independent_requests(self):
        decoder = Decoder(self.model)
        for batch_size in (1, 2, 3, 8):
            prompts = [[3] * (5 + row * 2) for row in range(batch_size)]
            reference = [decoder.prefill(p, self.sampling, max_total_tokens=128).cache for p in prompts]
            caches = [decoder.prefill(p, self.sampling, max_total_tokens=128).cache for p in prompts]
            for _ in range(4):
                expected = self.reference_decode(reference)
                actual = decoder.decode_caches(caches, [4] * batch_size)
                torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)
            for a, b in zip(reference, caches, strict=True):
                self.assertEqual(a.length, b.length)
                for x, y in zip(a._layers, b._layers, strict=True):
                    torch.testing.assert_close(x.keys, y.keys, atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(x.values, y.values, atol=2e-5, rtol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_dense_logits_match_independent_requests(self):
        self.model.cuda()
        self.test_dense_logits_match_independent_requests()

    def test_dense_storage_survives_decode_and_membership_changes(self):
        eager = Decoder(self.model)
        dense = Decoder(self.model)
        prompts = ([3, 4], [3, 5, 6, 7], [8, 9, 10])
        reference = [eager.prefill(p, self.sampling, max_total_tokens=128).cache for p in prompts]
        caches = [dense.prefill(p, self.sampling, max_total_tokens=128).cache for p in prompts]
        previous_batch = None
        for members in ([0, 1], [0, 1], [1, 0], [1], [1, 2], [1, 2]):
            selected = [caches[i] for i in members]
            before = selected[0]._decode_batch
            same_members = before is not None and before.matches(selected)
            expected = self.reference_decode([reference[i] for i in members])
            actual = dense.decode_caches(selected, [4] * len(members))
            torch.testing.assert_close(expected, actual)
            batch = selected[0]._decode_batch
            if same_members:
                self.assertIs(batch, before)
                self.assertEqual(previous_batch.layers[0][0].data_ptr(), batch.layers[0][0].data_ptr())
            elif before is not None:
                self.assertIsNot(batch, before)
            for i in members:
                for a, b in zip(reference[i]._layers, caches[i]._layers, strict=True):
                    torch.testing.assert_close(a.keys, b.keys)
                    torch.testing.assert_close(a.values, b.values)
            previous_batch = batch

    def test_dense_release_and_reservation_include_retained_rows(self):
        import weakref

        decoder = Decoder(self.model)
        a = decoder.prefill([3] * 20, self.sampling, max_total_tokens=128).cache
        b = decoder.prefill([4] * 2, self.sampling, max_total_tokens=128).cache
        per_token = a.memory_bytes_per_token
        self.assertEqual(decoder.dense_reservation_bytes([a, b]),
                         (a.capacity + b.capacity + 2 * a.capacity) * per_token)
        decoder.decode_caches([a, b], [4, 4])
        owner = weakref.ref(a._decode_batch)
        self.assertEqual(decoder.dense_reservation_bytes([a, b]), 2 * a.capacity * per_token)
        decoder.release_cache(a)
        self.assertEqual(a._layers, [])
        self.assertEqual(decoder.dense_reservation_bytes([b]),
                         (2 * a.capacity + b.capacity) * per_token)
        decoder.decode_caches([b], [4])
        self.assertIsNone(owner())
        owner = weakref.ref(b._decode_batch)
        decoder.release_cache(b)
        self.assertIsNone(owner())
        self.assertEqual(decoder.dense_reservation_bytes([]), 0)

    def test_warmup_fixed_steps_covers_every_batch_and_cleans_failure(self):
        decoder = Decoder(self.model)
        with patch.object(
            decoder, "decode_caches", wraps=decoder.decode_caches
        ) as decode:
            peak, sizes = warm_decode(
                decoder, [3] * 12, max_batch_size=3, max_tokens=128, budget_tokens=128
            )
            self.assertEqual(peak, 0)
            self.assertEqual(sizes, (1, 2, 3))
            counts = {size: 0 for size in sizes}
            for call in decode.call_args_list:
                counts[len(call.args[1])] += 1
            self.assertEqual(set(counts.values()), {4 * DECODE_WARMUP_STEPS})
        with patch.object(
            decoder, "decode_caches", side_effect=RuntimeError("injected")
        ), patch.object(
            decoder, "release_cache", wraps=decoder.release_cache
        ) as release:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                warm_decode(
                    decoder,
                    [3] * 12,
                    max_batch_size=3,
                    max_tokens=128,
                    budget_tokens=128,
                )
            self.assertEqual(release.call_count, 1)
        with self.assertRaisesRegex(RuntimeError, "Reduce PYLON_MAX_BATCH_SIZE"):
            warm_decode(
                decoder, [3] * 12, max_batch_size=3, max_tokens=128, budget_tokens=7
            )
