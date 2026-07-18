# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from heapq import nsmallest

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadEvictionMetadata, OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class CohortAwareLRUCachePolicy(CachePolicy):
    """LRU policy that retains valuable shared prefixes under pressure."""

    def __init__(self, cache_capacity: int):
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self.eviction_metadata: dict[OffloadKey, OffloadEvictionMetadata] = {}
        self.secondary_backed: set[OffloadKey] = set()

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable_blocks[key] = None

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)
        self.eviction_metadata.pop(key, None)
        self.secondary_backed.discard(key)

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)

    @override
    def clear(self) -> None:
        self.evictable_blocks.clear()
        self.blocks.clear()
        self.eviction_metadata.clear()
        self.secondary_backed.clear()

    def _eviction_priority(
        self,
        key: OffloadKey,
        lru_rank: int,
    ) -> tuple[int, float, int, int, int, int, int]:
        # Smaller values are evicted first: inactive suffixes, recoverable,
        # low-reuse, private, low-residency, then the ordinary LRU order.
        metadata = self.eviction_metadata.get(key, OffloadEvictionMetadata())
        return (
            metadata.lifecycle_value,
            -metadata.prefix_position,
            int(key not in self.secondary_backed),
            metadata.reuse_score,
            metadata.fanout,
            metadata.residency_value,
            lru_rank,
        )

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        candidates = [
            (key, self.blocks[key], lru_rank)
            for lru_rank, key in enumerate(self.evictable_blocks)
            if key not in protected
        ]
        if len(candidates) < n:
            return None

        selected = nsmallest(
            n,
            candidates,
            key=lambda item: self._eviction_priority(item[0], item[2]),
        )
        evicted = [(key, block) for key, block, _ in selected]
        for key, _ in evicted:
            del self.evictable_blocks[key]
            del self.blocks[key]
            self.eviction_metadata.pop(key, None)
            self.secondary_backed.discard(key)
        return evicted

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self.evictable_blocks[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        del self.evictable_blocks[key]

    @override
    def update_eviction_metadata(
        self,
        metadata: Mapping[OffloadKey, OffloadEvictionMetadata],
        *,
        replace: bool = False,
    ) -> None:
        if replace:
            self.eviction_metadata = dict(metadata)
        else:
            self.eviction_metadata.update(metadata)

    @override
    def mark_secondary_backed(self, keys: Iterable[OffloadKey]) -> None:
        self.secondary_backed.update(keys)
