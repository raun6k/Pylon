import torch

from pylon.kv.cache import PagedBatchCache, PagedKVCache
from pylon.kv.pool import _pages_for

STATIC_BUFFER_NAMES = (
    "token_ids",
    "positions",
    "block_table",
    "write_slots",
    "cu_seq_q",
    "cu_seq_k",
    "seqused_k",
)


def static_decode_values(
    caches: list[PagedKVCache],
    token_ids: list[int],
    *,
    page_size: int,
    max_pages: int,
) -> dict[str, list]:
    if not caches or len(caches) != len(token_ids):
        raise ValueError("Each decode row needs one token id.")
    rows: list[list[int]] = []
    ends: list[int] = []
    positions: list[list[int]] = []
    slots: list[int] = []
    for cache, token_id in zip(caches, token_ids, strict=True):
        if cache._pending_tokens != 1:
            raise RuntimeError("Decode graph buffers need one reserved token.")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError("Decode token ids must be integers.")
        position = cache.length
        end = position + 1
        page_ids = [page.index for page in cache.pages]
        if len(page_ids) > max_pages or end > max_pages * page_size:
            raise ValueError("Decode block table is wider than the captured graph.")
        rows.append(page_ids + [0] * (max_pages - len(page_ids)))
        ends.append(end)
        positions.append([position])
        page = cache.pages[position // page_size]
        slots.append(page.index * page_size + position % page_size)
    offsets = [0]
    total = 0
    for end in ends:
        total += end
        offsets.append(total)
    return {
        "token_ids": [[token_id] for token_id in token_ids],
        "positions": positions,
        "block_table": rows,
        "write_slots": slots,
        "cu_seq_q": list(range(len(caches) + 1)),
        "cu_seq_k": offsets,
        "seqused_k": ends,
    }


def copy_buffers(
    device_buffers: dict[str, torch.Tensor],
    host_buffers: dict[str, torch.Tensor],
    values: dict[str, list],
) -> None:
    for name in STATIC_BUFFER_NAMES:
        host_buffers[name].numpy()[...] = values[name]
        device_buffers[name].copy_(host_buffers[name], non_blocking=True)


class CapturedDecode:
    def __init__(
        self,
        batch_size: int,
        max_pages: int,
        max_tokens: int,
        page_size: int,
        device: torch.device,
        vocab: int,
        dtype: torch.dtype,
        *,
        pin_memory: bool,
    ) -> None:
        self.batch_size = batch_size
        self.max_pages = max_pages
        self.max_k = max_tokens
        self.page_size = page_size
        self.graph = None
        self.stream = None
        self.token_ids = torch.empty((batch_size, 1), dtype=torch.long, device=device)
        self.positions = torch.empty((batch_size, 1), dtype=torch.long, device=device)
        self.block_table = torch.empty(
            (batch_size, max_pages), dtype=torch.int32, device=device
        )
        self.write_slots = torch.empty((batch_size,), dtype=torch.long, device=device)
        self.cu_seq_q = torch.empty((batch_size + 1,), dtype=torch.int32, device=device)
        self.cu_seq_k = torch.empty((batch_size + 1,), dtype=torch.int32, device=device)
        self.seqused_k = torch.empty((batch_size,), dtype=torch.int32, device=device)
        self.logits = torch.empty((batch_size, vocab), dtype=dtype, device=device)
        self.buffers = {
            "token_ids": self.token_ids,
            "positions": self.positions,
            "block_table": self.block_table,
            "write_slots": self.write_slots,
            "cu_seq_q": self.cu_seq_q,
            "cu_seq_k": self.cu_seq_k,
            "seqused_k": self.seqused_k,
        }
        self.host = {
            name: torch.empty(
                tensor.shape, dtype=tensor.dtype, pin_memory=pin_memory
            )
            for name, tensor in self.buffers.items()
        }

    def run(self, caches: list[PagedKVCache], token_ids: list[int]) -> torch.Tensor:
        if len(caches) != self.batch_size or len(token_ids) != self.batch_size:
            raise ValueError("Decode graph batch size does not match the captured graph.")
        batch = PagedBatchCache(caches)
        batch.reserve(1)
        values = static_decode_values(
            caches,
            token_ids,
            page_size=self.page_size,
            max_pages=self.max_pages,
        )
        self._launch(values)
        batch.advance(1)
        return self.logits

    def _launch(self, values: dict[str, list]) -> None:
        stream = self.stream
        if stream is None:
            copy_buffers(self.buffers, self.host, values)
            self.graph.replay()
            return
        current = torch.cuda.current_stream(self.token_ids.device)
        if current != stream:
            stream.wait_stream(current)
        with torch.cuda.stream(stream):
            copy_buffers(self.buffers, self.host, values)
            self.graph.replay()
        if current != stream:
            current.wait_stream(stream)

    def device_forward(self, model, batch: PagedBatchCache, slots: tuple[int, ...]) -> None:
        with torch.no_grad():
            hidden = model(
                self.token_ids,
                cache=batch,
                position_ids=self.positions,
                cache_slots=slots,
            )
            self.logits.copy_(hidden[:, -1, :])


class DecodeGraphSet:
    def __init__(self) -> None:
        self._by_size: dict[int, CapturedDecode] = {}
        self._failed: set[int] = set()
        self._pool = None

    def captured_sizes(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_size))

    def capture_allowed(self, batch_size: int) -> bool:
        return batch_size not in self._failed and batch_size not in self._by_size

    def mark_failed(self, batch_size: int) -> None:
        self._failed.add(batch_size)
        self._by_size.pop(batch_size, None)

    def replay(
        self, caches: list[PagedKVCache], token_ids: list[int]
    ) -> torch.Tensor | None:
        size = len(caches)
        captured = self._by_size.get(size)
        if captured is None or size in self._failed:
            return None
        if any(cache.length + 1 > captured.max_k for cache in caches):
            return None
        return captured.run(caches, token_ids)

    def capture(self, decoder, caches: list[PagedKVCache], *, token_id: int, max_tokens: int) -> None:
        batch_size = len(caches)
        if batch_size < 1:
            raise ValueError("CUDA graph capture needs a positive batch size.")
        pool = caches[0].pool
        device = pool.device
        if device.type != "cuda":
            raise RuntimeError("CUDA graph capture requires a CUDA page pool.")
        # max_k is the profiled token cap. It is a launch upper bound. seqused_k
        # carries the live length, and the batch is not padded to a larger size.
        max_pages = _pages_for(max_tokens, pool.page_size)
        captured = CapturedDecode(
            batch_size,
            max_pages,
            max_tokens,
            pool.page_size,
            device,
            decoder.model.config.vocab_size,
            decoder.model.config.dtype,
            pin_memory=True,
        )
        batch = PagedBatchCache(caches)
        batch.graph_static = True
        batch.max_q = 1
        batch.max_k = captured.max_k
        batch.block_table = captured.block_table
        batch.write_slots = captured.write_slots
        batch.cu_seq_q = captured.cu_seq_q
        batch.cu_seq_k = captured.cu_seq_k
        batch.seqused_k = captured.seqused_k
        slots = tuple(range(batch_size))
        tokens = [token_id] * batch_size

        def fill() -> None:
            batch.reserve(1)
            copy_buffers(
                captured.buffers,
                captured.host,
                static_decode_values(
                    caches,
                    tokens,
                    page_size=pool.page_size,
                    max_pages=captured.max_pages,
                ),
            )

        decoder.model.eval()
        fill()
        captured.device_forward(decoder.model, batch, slots)
        torch.cuda.synchronize(device)
        fill()
        captured.stream = torch.cuda.current_stream(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            captured.device_forward(decoder.model, batch, slots)
        if self._pool is None:
            self._pool = graph.pool()
        captured.graph = graph
        self._by_size[batch_size] = captured
