import weakref
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from pylon.model.config import Qwen3Config


@dataclass
class LayerKV:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class LayerKVSnapshot:
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class KVBlockSnapshot:
    length: int
    layers: tuple[LayerKVSnapshot, ...]


class KVCache:
    def __init__(
        self,
        config: Qwen3Config,
        capacity: int,
        *,
        device: torch.device,
    ) -> None:
        if not 1 <= capacity <= config.context_length:
            raise ValueError(
                f"KV-cache capacity must be between 1 and {config.context_length:,} tokens."
            )
        self.capacity = capacity
        self.length = 0
        self._decode_batch: DenseDecodeBatch | None = None
        self._device = device
        self._layers = [
            LayerKV(
                keys=torch.zeros(
                    1,
                    config.n_kv_heads,
                    capacity,
                    config.head_dim,
                    device=device,
                    dtype=config.dtype,
                ),
                values=torch.zeros(
                    1,
                    config.n_kv_heads,
                    capacity,
                    config.head_dim,
                    device=device,
                    dtype=config.dtype,
                ),
            )
            for _ in range(config.n_layers)
        ]

    @property
    def memory_bytes_per_token(self) -> int:
        return sum(
            tensor[:, :, :1].numel() * tensor.element_size()
            for layer in self._layers
            for tensor in (layer.keys, layer.values)
        )

    def append(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keys.shape != values.shape:
            raise ValueError("Key and value tensors must have identical shapes.")
        entry = self._layers[layer]
        if (
            keys.ndim != 4
            or keys.shape[1] != entry.keys.shape[1]
            or keys.shape[-1] != entry.keys.shape[-1]
        ):
            raise ValueError("KV-cache tensor shape does not match this Qwen3 model.")
        if keys.shape[0] != 1:
            raise ValueError("KV-cache rows must match the key/value batch size.")
        start = self.length
        end = start + keys.shape[2]
        if end > self.capacity:
            raise ValueError(
                f"KV cache overflow: capacity is {self.capacity:,} tokens."
            )
        entry.keys[:, :, start:end].copy_(keys)
        entry.values[:, :, start:end].copy_(values)
        return entry.keys[:, :, :end], entry.values[:, :, :end]

    def advance(self, tokens: int) -> None:
        if tokens < 1:
            raise ValueError("Cannot advance the KV cache by fewer than one token.")
        end = self.length + tokens
        if end > self.capacity:
            raise ValueError("Cannot advance the KV cache beyond its capacity.")
        self.length = end

    def snapshot_block(self, start: int, end: int) -> KVBlockSnapshot:
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= self.length
        ):
            raise ValueError(
                f"Block bounds must satisfy 0 <= start < end <= {self.length:,}."
            )
        return KVBlockSnapshot(
            length=end - start,
            layers=tuple(
                LayerKVSnapshot(
                    keys=layer.keys[:, :, start:end].detach().clone(),
                    values=layer.values[:, :, start:end].detach().clone(),
                )
                for layer in self._layers
            ),
        )

    def restore_blocks(self, blocks: Sequence[KVBlockSnapshot]) -> None:
        total_length = 0
        for block in blocks:
            self._validate_block(block)
            total_length += block.length
            if total_length > self.capacity:
                raise ValueError(
                    f"KV blocks must fit within the {self.capacity:,}-token cache."
                )
        offset = 0
        for block in blocks:
            end = offset + block.length
            for destination, source in zip(self._layers, block.layers, strict=True):
                destination.keys[:, :, offset:end].copy_(source.keys)
                destination.values[:, :, offset:end].copy_(source.values)
            offset = end
        self.length = total_length

    def _validate_block(self, block: KVBlockSnapshot) -> None:
        if (
            not isinstance(block.length, int)
            or isinstance(block.length, bool)
            or block.length < 1
        ):
            raise ValueError("KV block length must be a positive integer.")
        if len(block.layers) != len(self._layers):
            raise ValueError("KV block layer count does not match this Qwen3 model.")
        for destination, source in zip(self._layers, block.layers, strict=True):
            self._validate_layer(destination, source, block.length)

    @staticmethod
    def _validate_layer(
        destination: LayerKV, source: LayerKVSnapshot, length: int
    ) -> None:
        expected_shape = (
            1,
            destination.keys.shape[1],
            length,
            destination.keys.shape[3],
        )
        if source.keys.shape != expected_shape or source.values.shape != expected_shape:
            raise ValueError("Snapshot tensor shape does not match this KV cache.")
        if (
            source.keys.dtype != destination.keys.dtype
            or source.values.dtype != destination.values.dtype
        ):
            raise ValueError("Snapshot tensor dtype does not match this KV cache.")
        if (
            source.keys.device != destination.keys.device
            or source.values.device != destination.values.device
        ):
            raise ValueError("Snapshot tensor device does not match this KV cache.")


class DenseDecodeBatch:
    def __init__(self, caches: Sequence[KVCache]) -> None:
        BatchedKVCache(caches)
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("Dense decode needs distinct request caches.")
        self.members = tuple(weakref.ref(cache) for cache in caches)
        self.capacity = max(cache.capacity for cache in caches)
        self.token_capacity = len(caches) * self.capacity
        self.layers = []
        for layer_index in range(len(caches[0]._layers)):
            reference = caches[0]._layers[layer_index]
            if len(caches) == 1 and caches[0]._decode_batch is None:
                self.layers.append((reference.keys, reference.values))
                continue
            keys = reference.keys.new_zeros(
                len(caches), reference.keys.shape[1], self.capacity,
                reference.keys.shape[3],
            )
            values = torch.zeros_like(keys)
            for row, cache in enumerate(caches):
                entry = cache._layers[layer_index]
                keys[row, :, :cache.length].copy_(entry.keys[0, :, :cache.length])
                values[row, :, :cache.length].copy_(entry.values[0, :, :cache.length])
            self.layers.append((keys, values))
        for row, cache in enumerate(caches):
            cache._layers = [
                LayerKV(keys[row:row + 1, :, :cache.capacity],
                        values[row:row + 1, :, :cache.capacity])
                for keys, values in self.layers
            ]
            cache._decode_batch = self

    def matches(self, caches: Sequence[KVCache]) -> bool:
        return len(caches) == len(self.members) and all(
            member() is cache for member, cache in zip(self.members, caches, strict=True)
        )

class BatchedKVCache:
    def __init__(self, caches: Sequence[KVCache]) -> None:
        if not caches:
            raise ValueError("Batched KV cache needs at least one request cache.")
        reference = caches[0]
        reference_layers = tuple(
            (
                layer.keys.shape[1],
                layer.keys.shape[3],
                layer.keys.dtype,
                layer.values.dtype,
                layer.keys.device,
                layer.values.device,
            )
            for layer in reference._layers
        )
        for cache in caches:
            layers = tuple(
                (
                    layer.keys.shape[1],
                    layer.keys.shape[3],
                    layer.keys.dtype,
                    layer.values.dtype,
                    layer.keys.device,
                    layer.values.device,
                )
                for layer in cache._layers
            )
            if (
                cache._device != reference._device
                or layers != reference_layers
            ):
                raise ValueError(
                    "Batched KV caches must be single-request caches with matching "
                    "device, layer shape, and dtype."
                )
        self._caches = tuple(caches)
        self.batch_size = len(caches)
        self._device = reference._device

    def slot_ids(self, slots: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
        if isinstance(slots, torch.Tensor):
            if slots.ndim != 1:
                raise ValueError("Batched KV-cache slots must be one-dimensional.")
            rows = tuple(slots.cpu().tolist())
        else:
            rows = tuple(slots)
        if not rows or len(set(rows)) != len(rows) or any(
            not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < self.batch_size
            for row in rows
        ):
            raise ValueError("Batched KV-cache slots must be distinct valid row indexes.")
        return rows

    def slot_length(self, slot: int) -> int:
        return self._caches[self.slot_ids((slot,))[0]].length

    def slot_lengths(self, slots: Sequence[int] | torch.Tensor) -> torch.Tensor:
        rows = self.slot_ids(slots)
        return torch.tensor(
            [self._caches[row].length for row in rows],
            dtype=torch.long,
            device=self._device,
        )

    def append(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        slots: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if slots is None:
            raise ValueError("Batched KV cache requires request row indexes.")
        rows = self.slot_ids(slots)
        if keys.shape[0] != len(rows) or values.shape != keys.shape:
            raise ValueError("Batched KV rows must match the key/value batch size.")
        appended = [
            self._caches[row].append(
                layer, keys[index : index + 1], values[index : index + 1]
            )
            for index, row in enumerate(rows)
        ]
        key_length = max(key.shape[2] for key, _ in appended)
        template = appended[0][0]
        batched_keys = template.new_zeros(
            len(rows), template.shape[1], key_length, template.shape[3]
        )
        batched_values = torch.zeros_like(batched_keys)
        for index, (key, value) in enumerate(appended):
            batched_keys[index, :, : key.shape[2]].copy_(key[0])
            batched_values[index, :, : value.shape[2]].copy_(value[0])
        return batched_keys, batched_values

    def advance(
        self,
        tokens: int,
        *,
        slots: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        if slots is None:
            raise ValueError("Batched KV cache requires request row indexes.")
        for row in self.slot_ids(slots):
            self._caches[row].advance(tokens)


class DecodeKVCache:
    def __init__(
        self,
        layers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        positions: torch.Tensor,
    ) -> None:
        self.layers = layers
        self.positions = positions

    def append(
        self,
        layer: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        slots: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stored_keys, stored_values = self.layers[layer]
        indices = self.positions[:, None, :, None].expand_as(keys)
        stored_keys.scatter_(2, indices, keys)
        stored_values.scatter_(2, indices, values)
        return stored_keys, stored_values
