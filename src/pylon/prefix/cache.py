import math
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pylon.kv.dense import KVBlockSnapshot, KVCache
from pylon.kv.cache import PagedKVBlockSnapshot, PagedKVCache

_FNV_OFFSET_BASIS = 14695981039346656037
_FNV_PRIME = 1099511628211
_U64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class TokenBlock:
    tokens: tuple[int, ...]
    hash: int
    parent_hash: int


@dataclass
class CachedBlock:
    tokens: tuple[int, ...]
    snapshot: KVBlockSnapshot | PagedKVBlockSnapshot
    hash: int = 0
    parent_hash: int = 0
    hit_count: int = 0
    expires_at: float = 0.0


@dataclass(frozen=True)
class PrefixCacheHit:
    blocks: tuple[CachedBlock, ...]

    @property
    def length(self) -> int:
        return sum(block.snapshot.length for block in self.blocks)


@dataclass(frozen=True)
class ResidentPrompt:
    token_ids: Sequence[int]
    cache: KVCache | PagedKVCache
    admitted_blocks: int = 0
    slot: int = 0


@dataclass(frozen=True)
class PrefixAdmission:
    pinned_blocks: int
    admitted_blocks: int


@dataclass(frozen=True)
class PromptBlockView:
    index: int
    token_count: int
    hash: int
    parent_hash: int
    status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "token_count": self.token_count,
            "hash": self.hash,
            "parent_hash": self.parent_hash,
            "status": self.status,
        }


@dataclass(frozen=True)
class PrefixBlockInfo:
    hash: int
    parent_hash: int
    token_count: int
    hit_count: int
    memory_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "hash": self.hash,
            "parent_hash": self.parent_hash,
            "token_count": self.token_count,
            "hit_count": self.hit_count,
            "memory_bytes": self.memory_bytes,
        }


def split_token_stream(
    token_ids: Sequence[int], block_size: int
) -> list[tuple[int, ...]]:
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")

    return [
        tuple(token_ids[start : start + block_size])
        for start in range(0, len(token_ids), block_size)
    ]


def _fnv1a_block(parent_hash: int, tokens: tuple[int, ...]) -> int:
    value = _FNV_OFFSET_BASIS
    for byte in parent_hash.to_bytes(8, "little"):
        value ^= byte
        value = (value * _FNV_PRIME) & _U64_MASK
    for token in tokens:
        for byte in token.to_bytes(4, "little"):
            value ^= byte
            value = (value * _FNV_PRIME) & _U64_MASK
    return value


def hash_token_blocks(blocks: Sequence[Sequence[int]]) -> list[TokenBlock]:
    hashed_blocks: list[TokenBlock] = []
    parent_hash = 0

    for block in blocks:
        tokens = tuple(block)
        if not tokens:
            raise ValueError("Token blocks must not be empty.")
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in tokens
        ):
            raise ValueError("Token IDs must be non-negative integers.")
        if any(token >= 1 << 32 for token in tokens):
            raise ValueError("Token IDs must fit in an unsigned 32-bit integer.")

        digest = _fnv1a_block(parent_hash, tokens)
        hashed_blocks.append(
            TokenBlock(tokens=tokens, hash=digest, parent_hash=parent_hash)
        )
        parent_hash = digest

    return hashed_blocks


def build_token_blocks(token_ids: Sequence[int], block_size: int) -> list[TokenBlock]:
    return hash_token_blocks(split_token_stream(token_ids, block_size))


def describe_prompt_blocks(
    token_ids: Sequence[int],
    block_size: int,
    hit: PrefixCacheHit | None,
) -> tuple[PromptBlockView, ...]:
    hit_blocks = 0 if hit is None else len(hit.blocks)
    views: list[PromptBlockView] = []
    for index, block in enumerate(build_token_blocks(token_ids, block_size)):
        if len(block.tokens) != block_size:
            status = "partial"
        elif index < hit_blocks:
            status = "hit"
        else:
            status = "miss"
        views.append(
            PromptBlockView(
                index=index,
                token_count=len(block.tokens),
                hash=block.hash,
                parent_hash=block.parent_hash,
                status=status,
            )
        )
    return tuple(views)


class PrefixCache:
    def __init__(
        self,
        block_size: int,
        max_memory_bytes: int | None = None,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1.")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and greater than 0.")
        if max_memory_bytes is not None and (
            not isinstance(max_memory_bytes, int)
            or isinstance(max_memory_bytes, bool)
            or max_memory_bytes < 0
        ):
            raise ValueError("max_memory_bytes must be a non-negative integer.")
        self.block_size = block_size
        self.max_memory_bytes = max_memory_bytes
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._blocks: OrderedDict[int, CachedBlock] = OrderedDict()
        self._sighted: set[int] = set()
        self._memory_bytes = 0
        self._reclaimed_memory_bytes = 0

    def __len__(self) -> int:
        self._purge_expired(self._clock())
        return len(self._blocks)

    @property
    def token_count(self) -> int:
        self._purge_expired(self._clock())
        return sum(len(block.tokens) for block in self._blocks.values())

    @property
    def memory_bytes(self) -> int:
        self._purge_expired(self._clock())
        return self._memory_bytes

    @property
    def sighted_hashes(self) -> frozenset[int]:
        return frozenset(self._sighted)

    def clear(self) -> None:
        self._reclaimed_memory_bytes += self._memory_bytes
        self._blocks.clear()
        self._memory_bytes = 0

    def get(self, block_hash: int) -> CachedBlock | None:
        self._purge_expired(self._clock())
        block = self._blocks.get(block_hash)
        if block is not None:
            self._blocks.move_to_end(block_hash)
        return block

    def reserve(self, memory_bytes: int) -> int:
        if self.max_memory_bytes is None:
            return 0
        if (
            not isinstance(memory_bytes, int)
            or isinstance(memory_bytes, bool)
            or not 0 <= memory_bytes <= self.max_memory_bytes
        ):
            raise ValueError("Reserved memory must fit within the KV-cache budget.")
        self._purge_expired(self._clock())
        self._evict_to(self.max_memory_bytes - memory_bytes)
        reclaimed = self._reclaimed_memory_bytes
        self._reclaimed_memory_bytes = 0
        return reclaimed

    def longest_prefix(self, token_ids: Sequence[int]) -> PrefixCacheHit | None:
        now = self._clock()
        self._purge_expired(now)
        matched: list[CachedBlock] = []
        for block in build_token_blocks(token_ids, self.block_size):
            if len(block.tokens) != self.block_size:
                break
            cached = self._blocks.get(block.hash)
            if cached is None:
                break
            cached.hit_count += 1
            cached.expires_at = now + self.ttl_seconds
            self._blocks.move_to_end(block.hash)
            matched.append(cached)
        return PrefixCacheHit(tuple(matched)) if matched else None

    def admit_resident_blocks(
        self,
        prompts: Sequence[ResidentPrompt],
        *,
        reserved_memory_bytes: int = 0,
    ) -> list[PrefixAdmission]:
        now = self._clock()
        self._purge_expired(now)
        return [
            self._observe_prompt(
                prompt, reserved_memory_bytes=reserved_memory_bytes, now=now
            )
            for prompt in prompts
        ]

    def _observe_prompt(
        self,
        prompt: ResidentPrompt,
        *,
        reserved_memory_bytes: int,
        now: float,
    ) -> PrefixAdmission:
        completed_length = len(prompt.token_ids) // self.block_size * self.block_size
        if completed_length < 1:
            return PrefixAdmission(0, prompt.admitted_blocks)
        cache = prompt.cache
        cache_length = (
            cache.length if isinstance(cache, KVCache) else cache.slot_length(prompt.slot)
        )
        if completed_length > cache_length:
            raise ValueError("KV cache has not processed every completed token block.")

        blocks = build_token_blocks(prompt.token_ids[:completed_length], self.block_size)
        available_bytes = self.max_memory_bytes
        if available_bytes is not None:
            available_bytes -= reserved_memory_bytes
            if available_bytes < 0:
                raise ValueError("Reserved memory must fit within the KV-cache budget.")
        snapshot_bytes = cache.memory_bytes_per_token * self.block_size
        can_pin = available_bytes is None or snapshot_bytes <= available_bytes
        watermark = prompt.admitted_blocks
        pinned = 0
        protected_hashes: set[int] = set()
        for index, block in enumerate(blocks):
            if index < prompt.admitted_blocks:
                if block.hash in self._blocks:
                    protected_hashes.add(block.hash)
                continue
            if block.hash not in self._sighted:
                self._sighted.add(block.hash)
                watermark = index + 1
                continue
            if block.hash in self._blocks:
                protected_hashes.add(block.hash)
                watermark = index + 1
                continue
            watermark = index + 1
            if not can_pin or (index > 0 and block.parent_hash not in self._blocks):
                continue
            if available_bytes is not None and not self._evict_to(
                available_bytes - snapshot_bytes,
                protected_hashes=protected_hashes,
            ):
                continue
            if index > 0 and block.parent_hash not in self._blocks:
                continue
            start = index * self.block_size
            snapshot = (
                cache.snapshot_block(start, start + self.block_size)
                if isinstance(cache, KVCache)
                else cache.snapshot_block_slot(prompt.slot, start, start + self.block_size)
            )
            self._blocks[block.hash] = CachedBlock(
                tokens=block.tokens,
                snapshot=snapshot,
                hash=block.hash,
                parent_hash=block.parent_hash,
                expires_at=now + self.ttl_seconds,
            )
            self._memory_bytes += snapshot_bytes
            protected_hashes.add(block.hash)
            pinned += 1
        return PrefixAdmission(pinned_blocks=pinned, admitted_blocks=watermark)

    def blocks(self) -> list[PrefixBlockInfo]:
        self._purge_expired(self._clock())
        return [
            PrefixBlockInfo(
                hash=block.hash,
                parent_hash=block.parent_hash,
                token_count=len(block.tokens),
                hit_count=block.hit_count,
                memory_bytes=self._snapshot_bytes(block.snapshot),
            )
            for block in self._blocks.values()
        ]

    def _purge_expired(self, now: float) -> None:
        expired = [
            block_hash
            for block_hash, block in self._blocks.items()
            if block.expires_at <= now
        ]
        for block_hash in expired:
            self._remove(block_hash)

    def _evict_to(
        self,
        target_bytes: int,
        *,
        protected_hashes: set[int] | None = None,
    ) -> bool:
        protected_hashes = protected_hashes or set()
        while self._memory_bytes > target_bytes:
            parent_hashes = {
                block.parent_hash
                for block in self._blocks.values()
                if block.parent_hash in self._blocks
            }
            for block_hash in self._blocks:
                if block_hash not in protected_hashes and block_hash not in parent_hashes:
                    break
            else:
                return False
            self._remove(block_hash)
        return True

    def _remove(self, block_hash: int) -> None:
        block = self._blocks.pop(block_hash)
        memory_bytes = self._snapshot_bytes(block.snapshot)
        self._memory_bytes -= memory_bytes
        self._reclaimed_memory_bytes += memory_bytes

    @staticmethod
    def _snapshot_bytes(snapshot: KVBlockSnapshot | PagedKVBlockSnapshot) -> int:
        if isinstance(snapshot, PagedKVBlockSnapshot):
            return snapshot.memory_bytes
        return sum(
            tensor.numel() * tensor.element_size()
            for layer in snapshot.layers
            for tensor in (layer.keys, layer.values)
        )
