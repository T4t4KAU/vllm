# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto

from vllm.v1.kv_offload.base import OffloadKey


class FanoutChunkState(Enum):
    GPU_ONLY = auto()
    OFFLOADING = auto()
    GPU_AND_CPU = auto()
    PROMOTING = auto()
    STAGED = auto()


@dataclass(frozen=True, slots=True)
class FanoutBlock:
    request_id: str
    group_idx: int
    logical_block_idx: int
    physical_block_id: int
    offload_key: OffloadKey
    fanout: int
    prefix_position: float
    last_access_time: float
    state: FanoutChunkState = FanoutChunkState.GPU_ONLY
    is_sealed: bool = True
    is_active_tail: bool = False
    in_flight: bool = False

    @property
    def is_offload_candidate(self) -> bool:
        return (
            self.physical_block_id != 0
            and self.fanout > 0
            and self.state is FanoutChunkState.GPU_ONLY
            and self.is_sealed
            and not self.is_active_tail
            and not self.in_flight
        )


@dataclass(frozen=True, slots=True)
class FanoutChunk:
    request_id: str
    group_idx: int
    blocks: tuple[FanoutBlock, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("FanoutChunk must contain at least one block")

    @property
    def logical_start_block(self) -> int:
        return self.blocks[0].logical_block_idx

    @property
    def logical_end_block(self) -> int:
        return self.blocks[-1].logical_block_idx + 1

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def fanout(self) -> int:
        return self.blocks[0].fanout

    @property
    def prefix_position(self) -> float:
        return self.blocks[-1].prefix_position

    @property
    def last_access_time(self) -> float:
        return min(block.last_access_time for block in self.blocks)

    @property
    def offload_keys(self) -> tuple[OffloadKey, ...]:
        return tuple(block.offload_key for block in self.blocks)

    @property
    def physical_block_ids(self) -> tuple[int, ...]:
        return tuple(block.physical_block_id for block in self.blocks)

    @property
    def reuse_score(self) -> int:
        return self.fanout * self.num_blocks

    @property
    def priority(self) -> tuple[int, float, int, float]:
        """Priority for shared-prefix GPU reuse. Lower is better.

        This is intentionally not an eviction-victim priority. The planner
        admits high-reuse prefix chunks first so their contributing queries can
        be grouped for one GPU ForkAttention pass, minimizing repeated prefix
        loads and QK/PV work.
        """
        return (
            -self.reuse_score,
            self.prefix_position,
            -self.fanout,
            self.last_access_time,
        )


@dataclass(frozen=True, slots=True)
class FanoutOffloadPlan:
    chunks: tuple[FanoutChunk, ...]
    num_blocks: int


class FanoutChunkPlanner:
    def __init__(
        self,
        max_blocks_per_chunk: int | None = None,
        min_fanout: int = 1,
    ) -> None:
        if max_blocks_per_chunk is not None and max_blocks_per_chunk <= 0:
            raise ValueError("max_blocks_per_chunk must be positive")
        if min_fanout <= 0:
            raise ValueError("min_fanout must be positive")
        self.max_blocks_per_chunk = max_blocks_per_chunk
        self.min_fanout = min_fanout

    def build_chunks(self, blocks: Iterable[FanoutBlock]) -> list[FanoutChunk]:
        ordered = sorted(
            (
                block
                for block in blocks
                if block.is_offload_candidate and block.fanout >= self.min_fanout
            ),
            key=lambda block: (
                block.request_id,
                block.group_idx,
                block.logical_block_idx,
            ),
        )
        chunks: list[FanoutChunk] = []
        current: list[FanoutBlock] = []

        for block in ordered:
            if current and not self._can_merge(current, block):
                chunks.append(self._make_chunk(current))
                current = []
            current.append(block)

        if current:
            chunks.append(self._make_chunk(current))
        return chunks

    def select(
        self,
        blocks: Iterable[FanoutBlock],
        budget_blocks: int,
    ) -> FanoutOffloadPlan:
        if budget_blocks < 0:
            raise ValueError("budget_blocks must be non-negative")
        if budget_blocks == 0:
            return FanoutOffloadPlan((), 0)

        chunks = sorted(
            self._deduplicate(self.build_chunks(blocks)),
            key=lambda chunk: chunk.priority,
        )
        selected: list[FanoutChunk] = []
        selected_blocks = 0
        for chunk in chunks:
            if selected_blocks >= budget_blocks:
                break
            if selected_blocks + chunk.num_blocks > budget_blocks:
                continue
            selected.append(chunk)
            selected_blocks += chunk.num_blocks
        return FanoutOffloadPlan(tuple(selected), selected_blocks)

    def _can_merge(
        self,
        current: Sequence[FanoutBlock],
        block: FanoutBlock,
    ) -> bool:
        previous = current[-1]
        return (
            previous.request_id == block.request_id
            and previous.group_idx == block.group_idx
            and previous.logical_block_idx + 1 == block.logical_block_idx
            and previous.fanout == block.fanout
            and previous.state is block.state
            and (
                self.max_blocks_per_chunk is None
                or len(current) < self.max_blocks_per_chunk
            )
        )

    @staticmethod
    def _make_chunk(blocks: Sequence[FanoutBlock]) -> FanoutChunk:
        first = blocks[0]
        return FanoutChunk(
            request_id=first.request_id,
            group_idx=first.group_idx,
            blocks=tuple(blocks),
        )

    @staticmethod
    def _deduplicate(chunks: Iterable[FanoutChunk]) -> list[FanoutChunk]:
        unique: dict[tuple[OffloadKey, ...], FanoutChunk] = {}
        for chunk in chunks:
            previous = unique.get(chunk.offload_keys)
            if previous is None or chunk.priority < previous.priority:
                unique[chunk.offload_keys] = chunk
        return list(unique.values())
