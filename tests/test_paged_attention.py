import gc
import unittest
import torch

from pylon.kv.dense import KVCache
from pylon.model.config import Qwen3Config
from pylon.model.model import Qwen3Model
from pylon.kv.cache import PagedBatchCache, PagedKVCache
from pylon.kv.pool import KVPagePool


def tiny_config(*, context_length: int = 512) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=64,
        context_length=context_length,
        hidden_size=16,
        n_heads=2,
        n_layers=2,
        hidden_dim=32,
        head_dim=8,
        n_kv_heads=1,
        dtype=torch.float32,
    )


def attend(
    caches: list[PagedKVCache],
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    layer: int = 0,
) -> torch.Tensor:
    batch = PagedBatchCache(caches)
    batch.prepare(queries.shape[2])
    output = batch.attend(layer, queries, keys, values)
    batch.advance(queries.shape[2])
    return output


class PagedKVCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tiny_config()
        self.pool = KVPagePool(self.config, 6, device=torch.device("cpu"))

    def tensors(self, batch_size: int, tokens: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.randn(batch_size, self.config.n_heads, tokens, self.config.head_dim),
            torch.randn(batch_size, self.config.n_kv_heads, tokens, self.config.head_dim),
            torch.randn(batch_size, self.config.n_kv_heads, tokens, self.config.head_dim),
        )

    def test_page_boundary_allocates_the_next_256_token_page(self) -> None:
        cache = PagedKVCache(self.pool, 512)
        query, key, value = self.tensors(1, 256)
        attend([cache], query, key, value)

        self.assertEqual(cache.length, 256)
        self.assertEqual(len(cache.pages), 1)
        first_page = cache.pages[0].index

        query, key, value = self.tensors(1, 1)
        attend([cache], query, key, value)

        self.assertEqual(cache.length, 257)
        self.assertEqual(len(cache.pages), 2)
        self.assertEqual(cache.pages[0].index, first_page)
        self.assertNotEqual(cache.pages[1].index, first_page)

    def test_released_page_is_reused_in_a_noncontiguous_batch(self) -> None:
        first = PagedKVCache(self.pool, 512)
        survivor = PagedKVCache(self.pool, 512)
        query, key, value = self.tensors(2, 1)
        attend([first, survivor], query, key, value)
        released = first.pages[0].index
        survivor_page = survivor.pages[0].index
        first.close()

        replacement = PagedKVCache(self.pool, 512)
        query, key, value = self.tensors(1, 1)
        attend([replacement], query, key, value)

        self.assertEqual(replacement.pages[0].index, released)
        self.assertNotEqual(survivor_page, released)
        batch = PagedBatchCache([survivor, replacement])
        self.assertCountEqual(
            [cache.pages[0].index for cache in batch.caches],
            [survivor_page, released],
        )

    def test_mixed_lengths_attend_without_padding_or_cross_request_reads(self) -> None:
        short = PagedKVCache(self.pool, 512)
        long = PagedKVCache(self.pool, 512)
        for cache, tokens in ((short, 3), (long, 7)):
            query, key, value = self.tensors(1, tokens)
            attend([cache], query, key, value)

        query, key, value = self.tensors(2, 1)
        batch = PagedBatchCache([short, long])
        batch.prepare(1)
        self.assertEqual(batch.block_table.dtype, torch.int32)
        self.assertEqual(batch.seqused_k.dtype, torch.int32)
        self.assertEqual(batch.cu_seq_q.dtype, torch.int32)
        self.assertEqual(batch.cu_seq_k.dtype, torch.int32)
        self.assertEqual(batch.seqused_k.tolist(), [4, 8])
        self.assertEqual(batch.cu_seq_q.tolist(), [0, 1, 2])
        self.assertEqual(batch.cu_seq_k.tolist(), [0, 4, 12])
        output = batch.attend(0, query, key, value)
        batch.advance(1)

        self.assertEqual(output.shape, (2, self.config.n_heads, 1, self.config.head_dim))
        self.assertEqual((short.length, long.length), (4, 8))

    def test_full_page_snapshot_survives_request_close_and_restores(self) -> None:
        cache = PagedKVCache(self.pool, 512)
        query, key, value = self.tensors(1, 256)
        attend([cache], query, key, value)
        snapshot = cache.snapshot_block(0, 256)
        page_index = snapshot.pages[0].index
        cache.close()

        self.assertEqual(self.pool.free_pages, self.pool.num_pages - 1)
        restored = PagedKVCache(self.pool, 512)
        restored.restore_blocks([snapshot])
        self.assertEqual(restored.length, 256)
        self.assertEqual(restored.pages[0].index, page_index)

        sibling = PagedKVCache(self.pool, 512)
        sibling.restore_blocks([snapshot])
        self.assertIs(restored.pages[0], sibling.pages[0])
        self.assertEqual(self.pool.free_pages, self.pool.num_pages - 1)
        prefix_keys = self.pool.keys[0][page_index].clone()
        prefix_values = self.pool.values[0][page_index].clone()
        query, key, value = self.tensors(2, 1)
        attend([restored, sibling], query, key, value)
        self.assertNotEqual(restored.pages[1].index, sibling.pages[1].index)
        torch.testing.assert_close(self.pool.keys[0][page_index], prefix_keys)
        torch.testing.assert_close(self.pool.values[0][page_index], prefix_values)

        del snapshot
        restored.close()
        self.assertEqual(self.pool.free_pages, self.pool.num_pages - 2)
        sibling.close()
        gc.collect()
        self.assertEqual(self.pool.free_pages, self.pool.num_pages)

    def test_restored_page_prefix_matches_dense_gqa_continuation(self) -> None:
        torch.manual_seed(4)
        model = Qwen3Model(self.config).eval()
        dense = KVCache(self.config, 300, device=torch.device("cpu"))
        paged = PagedKVCache(self.pool, 300)
        prefix = torch.randint(0, self.config.vocab_size, (1, 256))

        with torch.inference_mode():
            model(prefix, cache=dense)
            initial = PagedBatchCache([paged])
            initial.prepare(prefix.shape[1])
            model(prefix, cache=initial)

        snapshot = paged.snapshot_block(0, 256)
        paged.close()
        restored = PagedKVCache(self.pool, 300)
        restored.restore_blocks([snapshot])
        next_token = torch.tensor([[7]])

        with torch.inference_mode():
            expected = model(next_token, cache=dense)
            continuation = PagedBatchCache([restored])
            continuation.prepare(1)
            actual = model(next_token, cache=continuation)

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        self.assertEqual(restored.length, 257)

class NativePagedAttentionTests(unittest.TestCase):
    def test_cuda_varlen_matches_dense_sdpa_for_prefill_and_decode(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("native paged varlen attention requires NVIDIA SM80 or newer")

        device = torch.device("cuda")
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                config = tiny_config()
                config = Qwen3Config(**{**config.__dict__, "dtype": dtype})
                pool = KVPagePool(config, 4, device=device)
                caches = [PagedKVCache(pool, 512), PagedKVCache(pool, 512)]

                prefill_q = torch.randn(2, config.n_heads, 2, config.head_dim, device=device, dtype=dtype)
                prefill_k = torch.randn(2, config.n_kv_heads, 2, config.head_dim, device=device, dtype=dtype)
                prefill_v = torch.randn_like(prefill_k)
                prefill = PagedBatchCache(caches)
                prefill.prepare(2)
                prefill_actual = prefill.attend(0, prefill_q, prefill_k, prefill_v)
                prefill.advance(2)
                self.assertEqual(prefill.cu_seq_q.dtype, torch.int32)
                self.assertEqual(prefill.cu_seq_k.dtype, torch.int32)
                self.assertEqual(prefill.cu_seq_q.tolist(), [0, 2, 4])
                self.assertEqual(prefill.cu_seq_k.tolist(), [0, 2, 4])
                self._assert_dense_match(pool, caches, prefill_q, prefill_actual)

                decode_q = torch.randn(2, config.n_heads, 1, config.head_dim, device=device, dtype=dtype)
                decode_k = torch.randn(2, config.n_kv_heads, 1, config.head_dim, device=device, dtype=dtype)
                decode_v = torch.randn_like(decode_k)
                decode = PagedBatchCache(caches)
                decode.prepare(1)
                decode_actual = decode.attend(0, decode_q, decode_k, decode_v)
                decode.advance(1)
                self._assert_dense_match(pool, caches, decode_q, decode_actual)

    def _assert_dense_match(
        self,
        pool: KVPagePool,
        caches: list[PagedKVCache],
        query: torch.Tensor,
        actual: torch.Tensor,
    ) -> None:
        expected_rows = []
        for row, cache in enumerate(caches):
            tokens = query.shape[2]
            start = cache.length - tokens
            packed_keys = torch.cat([pool.keys[0][page.index] for page in cache.pages])[
                : cache.length
            ]
            packed_values = torch.cat([pool.values[0][page.index] for page in cache.pages])[
                : cache.length
            ]
            dense_key = packed_keys.permute(1, 0, 2).unsqueeze(0).repeat_interleave(
                query.shape[1], dim=1
            )
            dense_value = packed_values.permute(1, 0, 2).unsqueeze(0).repeat_interleave(
                query.shape[1], dim=1
            )
            positions = torch.arange(cache.length, device=query.device)
            query_positions = start + torch.arange(tokens, device=query.device)
            mask = positions[None, None, None, :] <= query_positions[None, None, :, None]
            expected_rows.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query[row : row + 1],
                    dense_key,
                    dense_value,
                    attn_mask=mask,
                    dropout_p=0.0,
                )
            )
        torch.testing.assert_close(
            actual, torch.cat(expected_rows), rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
