from __future__ import annotations

import weakref

import torch

from pylon.model.config import Qwen3Config


class KVPage:
    __slots__ = ("__weakref__", "_finalizer", "index", "pool")

    def __init__(self, pool: KVPagePool, index: int) -> None:
        self.index = index
        self.pool = pool
        self._finalizer = weakref.finalize(self, pool._release, index)


class KVPagePool:
    def __init__(
        self,
        config: Qwen3Config,
        num_pages: int,
        *,
        device: torch.device,
        page_size: int = 256,
    ) -> None:
        if num_pages < 1:
            raise ValueError("KV page pool must contain at least one page.")
        if page_size < 1 or page_size % 256:
            raise ValueError("KV page size must be a positive multiple of 256.")
        self.config = config
        self.num_pages = num_pages
        self.page_size = page_size
        self.device = device
        shape = (num_pages, page_size, config.n_kv_heads, config.head_dim)
        self.keys = tuple(
            torch.empty(shape, dtype=config.dtype, device=device)
            for _ in range(config.n_layers)
        )
        self.values = tuple(
            torch.empty(shape, dtype=config.dtype, device=device)
            for _ in range(config.n_layers)
        )
        self._free = list(range(num_pages - 1, -1, -1))

    @property
    def bytes_per_token(self) -> int:
        return (
            self.config.n_layers
            * 2
            * self.config.n_kv_heads
            * self.config.head_dim
            * self.keys[0].element_size()
        )

    @property
    def bytes_per_page(self) -> int:
        return self.bytes_per_token * self.page_size

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def acquire(self, count: int) -> list[KVPage]:
        if count < 0:
            raise ValueError("Requested page count must not be negative.")
        if count > len(self._free):
            raise RuntimeError(
                f"KV page pool needs {count} pages but only {len(self._free)} are free."
            )
        return [KVPage(self, self._free.pop()) for _ in range(count)]

    def _release(self, index: int) -> None:
        self._free.append(index)


def _pages_for(tokens: int, page_size: int) -> int:
    return (tokens + page_size - 1) // page_size
