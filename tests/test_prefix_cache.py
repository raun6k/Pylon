import unittest

import torch

from pylon.kv.dense import KVCache
from pylon.model.config import Qwen3Config
from pylon.prefix.cache import PrefixCache, ResidentPrompt, hash_token_blocks

_FNV_OFFSET_BASIS = 14695981039346656037
_FNV_PRIME = 1099511628211
_U64_MASK = (1 << 64) - 1


def _fnv1a_block(parent_hash: int, tokens: tuple[int, ...]) -> int:
    value = _FNV_OFFSET_BASIS
    payload = parent_hash.to_bytes(8, "little")
    for token in tokens:
        payload += token.to_bytes(4, "little")
    for byte in payload:
        value ^= byte
        value = (value * _FNV_PRIME) & _U64_MASK
    return value


class PrefixCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Qwen3Config(
            vocab_size=64,
            context_length=128,
            hidden_size=16,
            n_heads=2,
            n_layers=1,
            hidden_dim=32,
            head_dim=4,
            n_kv_heads=1,
            dtype=torch.float32,
        )
        self.now = 0.0

    def cache(self) -> PrefixCache:
        return PrefixCache(
            block_size=4,
            max_memory_bytes=10**12,
            ttl_seconds=300,
            clock=lambda: self.now,
        )

    def filled(self, tokens: list[int]) -> KVCache:
        cache = KVCache(
            self.config, capacity=max(16, len(tokens)), device=torch.device("cpu")
        )
        cache.advance(len(tokens))
        return cache

    def test_block_hash_is_chained_fnv1a_uint64(self) -> None:
        blocks = hash_token_blocks([(1, 2), (3,)])
        first = _fnv1a_block(0, (1, 2))
        second = _fnv1a_block(first, (3,))
        self.assertEqual(blocks[0].parent_hash, 0)
        self.assertEqual(blocks[0].hash, first)
        self.assertEqual(blocks[1].parent_hash, first)
        self.assertEqual(blocks[1].hash, second)
        self.assertIsInstance(blocks[0].hash, int)
        self.assertLess(blocks[0].hash, 1 << 64)
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            hash_token_blocks([(-1,)])
        with self.assertRaises(ValueError):
            hash_token_blocks([(1 << 32,)])

    def test_first_sight_records_the_hash_and_pins_nothing(self) -> None:
        prefix = self.cache()
        tokens = [1, 2, 3, 4, 5, 6]
        block_hash = hash_token_blocks([tuple(tokens[:4])])[0].hash
        admission = prefix.admit_resident_blocks(
            [ResidentPrompt(tokens, self.filled(tokens))]
        )[0]
        self.assertEqual(admission.pinned_blocks, 0)
        self.assertEqual(admission.admitted_blocks, 1)
        self.assertEqual(prefix.sighted_hashes, frozenset({block_hash}))
        self.assertTrue(all(isinstance(item, int) for item in prefix.sighted_hashes))
        self.assertEqual(prefix.token_count, 0)
        self.assertEqual(prefix.memory_bytes, 0)
        self.assertIsNone(prefix.longest_prefix(tokens))

    def test_second_sight_in_the_same_call_pins_the_later_cache(self) -> None:
        prefix = self.cache()
        tokens = [1, 2, 3, 4]
        earlier = self.filled(tokens)
        later = self.filled(tokens)
        earlier._layers[0].keys.fill_(1)
        later._layers[0].keys.fill_(7)
        admissions = prefix.admit_resident_blocks(
            [ResidentPrompt(tokens, earlier), ResidentPrompt(tokens, later)]
        )
        self.assertEqual([item.pinned_blocks for item in admissions], [0, 1])
        block_hash = hash_token_blocks([tuple(tokens)])[0].hash
        stored = prefix.get(block_hash)
        self.assertIsNotNone(stored)
        torch.testing.assert_close(
            stored.snapshot.layers[0].keys, later._layers[0].keys[:, :, :4]
        )
        self.assertFalse(
            torch.equal(stored.snapshot.layers[0].keys, earlier._layers[0].keys[:, :, :4])
        )
        self.assertEqual(prefix.token_count, 4)
        self.assertGreater(prefix.memory_bytes, 0)

    def test_the_same_request_does_not_pin_its_own_block(self) -> None:
        prefix = self.cache()
        tokens = [4, 5, 6, 7]
        cache = self.filled(tokens)
        first = prefix.admit_resident_blocks([ResidentPrompt(tokens, cache)])[0]
        second = prefix.admit_resident_blocks(
            [ResidentPrompt(tokens, cache, admitted_blocks=first.admitted_blocks)]
        )[0]
        self.assertEqual(first.pinned_blocks, 0)
        self.assertEqual(second.pinned_blocks, 0)
        self.assertEqual(prefix.memory_bytes, 0)

    def test_second_request_pins_both_blocks_of_a_chain(self) -> None:
        prefix = self.cache()
        tokens = list(range(1, 9))
        blocks = hash_token_blocks([tuple(tokens[:4]), tuple(tokens[4:])])
        self.assertEqual(blocks[0].parent_hash, 0)
        self.assertEqual(blocks[1].parent_hash, blocks[0].hash)
        first = prefix.admit_resident_blocks(
            [ResidentPrompt(tokens, self.filled(tokens))]
        )[0]
        self.assertEqual(first.pinned_blocks, 0)
        self.assertEqual(
            prefix.sighted_hashes, frozenset({blocks[0].hash, blocks[1].hash})
        )
        second = prefix.admit_resident_blocks(
            [ResidentPrompt(tokens, self.filled(tokens))]
        )[0]
        self.assertEqual(second.pinned_blocks, 2)
        self.assertEqual(prefix.token_count, 8)
        self.assertEqual(len(prefix), 2)

    def test_ttl_drops_pinned_pages_and_keeps_the_hash(self) -> None:
        prefix = self.cache()
        tokens = [1, 2, 3, 4]
        block_hash = hash_token_blocks([tuple(tokens)])[0].hash
        prefix.admit_resident_blocks([ResidentPrompt(tokens, self.filled(tokens))])
        prefix.admit_resident_blocks([ResidentPrompt(tokens, self.filled(tokens))])
        self.assertEqual(prefix.token_count, 4)
        self.now = 100
        self.assertIsNotNone(prefix.longest_prefix(tokens))
        self.now = 300
        self.assertEqual(len(prefix), 1)
        self.now = 400
        self.assertEqual(prefix.token_count, 0)
        self.assertEqual(prefix.memory_bytes, 0)
        self.assertIn(block_hash, prefix.sighted_hashes)
        pinned = prefix.admit_resident_blocks(
            [ResidentPrompt(tokens, self.filled(tokens))]
        )[0]
        self.assertEqual(pinned.pinned_blocks, 1)
        self.assertEqual(prefix.token_count, 4)

    def test_clear_drops_pinned_pages_and_keeps_sighted_hashes(self) -> None:
        prefix = self.cache()
        tokens = [8, 7, 6, 5]
        block_hash = hash_token_blocks([tuple(tokens)])[0].hash
        prefix.admit_resident_blocks([ResidentPrompt(tokens, self.filled(tokens))])
        prefix.admit_resident_blocks([ResidentPrompt(tokens, self.filled(tokens))])
        prefix.clear()
        self.assertEqual(prefix.token_count, 0)
        self.assertEqual(prefix.memory_bytes, 0)
        self.assertIn(block_hash, prefix.sighted_hashes)
        pinned = prefix.admit_resident_blocks(
            [ResidentPrompt(tokens, self.filled(tokens))]
        )[0]
        self.assertEqual(pinned.pinned_blocks, 1)


if __name__ == "__main__":
    unittest.main()
