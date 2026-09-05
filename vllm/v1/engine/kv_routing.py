# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded GPU-prefix hints reconstructed from the existing KV event stream."""

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import msgspec

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreRequest

CacheKey = tuple[int, int]
CacheEvent = BlockStored | BlockRemoved | AllBlocksCleared
logger = init_logger(__name__)


class GPUCacheRoutingIndex:
    """Track contiguous plain-token GPU prefixes without scheduler RPCs.

    Event hashes identify nodes; frontend-local token fingerprints match new
    requests to those nodes. Every prefix block must remain present, so a
    surviving descendant never hides an eviction earlier in the chain.
    """

    def __init__(self, num_ranks: int, block_size: int, max_blocks: int) -> None:
        if min(num_ranks, block_size, max_blocks) <= 0:
            raise ValueError("routing index dimensions must be positive")
        self.block_size = block_size
        self.max_blocks = max_blocks
        self._nodes: list[OrderedDict[bytes | int, CacheKey]] = [
            OrderedDict() for _ in range(num_ranks)
        ]
        self._resident: list[set[CacheKey]] = [set() for _ in range(num_ranks)]
        self._ready = [False] * num_ranks
        self._supported = [True] * num_ranks
        self._decoder = msgspec.msgpack.Decoder(list[CacheEvent])

    @staticmethod
    def supports(request: "EngineCoreRequest") -> bool:
        """Use physical hints only for identities fully represented by this index."""
        return (
            request.prompt_token_ids is not None
            and request.prompt_embeds is None
            and request.prompt_is_token_ids is None
            and not request.mm_features
            and request.lora_request is None
            and request.cache_salt is None
        )

    def update(self, rank: int, payload: bytes) -> None:
        """Apply one ordered engine batch; unsupported cache layouts use fallback."""
        nodes, resident = self._nodes[rank], self._resident[rank]
        if not self._ready[rank]:
            logger.info("GPU cache events ready for DP rank %d", rank)
        self._ready[rank] = True
        for event in self._decoder.decode(payload):
            if isinstance(event, AllBlocksCleared):
                nodes.clear()
                resident.clear()
                continue
            if event.medium != MEDIUM_GPU:
                continue
            if event.group_idx not in (None, 0) or (
                isinstance(event, BlockStored) and event.block_size != self.block_size
            ):
                self._supported[rank] = False
                nodes.clear()
                resident.clear()
            if not self._supported[rank]:
                continue
            if isinstance(event, BlockRemoved):
                for block_hash in event.block_hashes:
                    if (key := nodes.pop(block_hash, None)) is not None:
                        resident.discard(key)
                continue
            self._store(nodes, resident, event)

    def lookup(self, token_ids: Sequence[int]) -> list[int | None]:
        """Return resident prefix lengths in blocks, or None for unavailable ranks."""
        matches: list[int | None] = [
            0 if ready and supported else None
            for ready, supported in zip(self._ready, self._supported)
        ]
        active = [rank for rank, value in enumerate(matches) if value is not None]
        chain = 0
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            if not active:
                break
            depth = start // self.block_size + 1
            chain = hash((chain, tuple(token_ids[start : start + self.block_size])))
            key = (depth, chain)
            active = [rank for rank in active if key in self._resident[rank]]
            for rank in active:
                matches[rank] = depth
        return matches

    def clear(self) -> None:
        """Forget GPU hints after all engines acknowledge a cache reset."""
        for nodes, resident in zip(self._nodes, self._resident):
            nodes.clear()
            resident.clear()

    def _store(
        self,
        nodes: OrderedDict[bytes | int, CacheKey],
        resident: set[CacheKey],
        event: BlockStored,
    ) -> None:
        if (
            event.lora_name is not None
            or any(event.extra_keys or ())
            or len(event.token_ids) != len(event.block_hashes) * self.block_size
        ):
            return
        parent = (
            (0, 0)
            if event.parent_block_hash is None
            else nodes.get(event.parent_block_hash)
        )
        if parent is None:
            return
        depth, chain = parent
        for offset, block_hash in enumerate(event.block_hashes):
            start = offset * self.block_size
            depth += 1
            chain = hash(
                (chain, tuple(event.token_ids[start : start + self.block_size]))
            )
            key = (depth, chain)
            previous = nodes.pop(block_hash, None)
            if previous is not None:
                resident.discard(previous)
            nodes[block_hash] = key
            resident.add(key)
            if len(nodes) > self.max_blocks:
                _, expired = nodes.popitem(last=False)
                resident.discard(expired)
