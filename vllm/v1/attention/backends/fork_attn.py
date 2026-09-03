# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ForkAttention backend with FlashAttention fallback."""

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config.cache import CacheDType
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.utils.torch_utils import (
    PIN_MEMORY,
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.utils import CpuGpuBuffer

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

_PREFIX_CHUNK_TOKENS = 2048
_MAX_SPLITS_PER_QUERY = 32
_MAX_FORK_WORKSPACE_BYTES = 256 * 1024 * 1024
_MAX_FORK_CUDAGRAPH_CTAS = 512
_MIN_FORK_CUDAGRAPH_CTAS = 2
# Bound the Python work needed to copy block IDs and build the prefix trie.
# This still covers the profiled 64-request, ~4K-token decode workload.
_MAX_FORK_PLAN_BLOCK_VISITS = 20_000


@dataclass
class ForkAttentionMetadata(FlashAttentionMetadata):
    fork_enabled: bool = False
    fork_num_split_per_seq: torch.Tensor | None = None
    fork_query_tables: list[torch.Tensor] | None = None
    fork_block_tables: list[torch.Tensor] | None = None
    fork_num_seqs_per_ctas: list[torch.Tensor] | None = None
    fork_cta_ranks: list[torch.Tensor] | None = None
    fork_kv_in_ctas: list[torch.Tensor] | None = None
    fork_mnw: list[int] | None = None
    fork_max_split_per_seq: int = 0
    fork_max_block_id: int = -1
    fork_softmax_lse: torch.Tensor | None = None
    fork_split_out: torch.Tensor | None = None
    fork_split_lse: torch.Tensor | None = None


@dataclass
class _PrefixTrieNode:
    query_ids: list[int] = field(default_factory=list)
    terminal_query_ids: list[int] = field(default_factory=list)
    children: dict[int, "_PrefixTrieNode"] = field(default_factory=dict)


@dataclass(frozen=True)
class _ForkSegment:
    query_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    rank: int
    num_kv_tokens: int
    tile_num_queries: int


@dataclass(frozen=True)
class _ForkPlan:
    segments: tuple[_ForkSegment, ...]
    num_splits_per_query: tuple[int, ...]
    max_block_id: int


class _ForkPlanError(ValueError):
    pass


class _ForkWorkspaceLimitError(RuntimeError):
    pass


@dataclass
class _ForkGroupBuffers:
    cta_capacity: int
    block_capacity: int
    query_width: int
    query_table: CpuGpuBuffer
    block_table: CpuGpuBuffer
    num_seqs: CpuGpuBuffer
    cta_ranks: CpuGpuBuffer
    kv_tokens: CpuGpuBuffer


def _kernel_head_ratio(head_ratio: int) -> int:
    if head_ratio >= 4:
        return 4
    if head_ratio >= 2:
        return 2
    return 1


class _ForkBufferPool:
    """High-water buffers reused by one persistent metadata builder."""

    def __init__(
        self,
        *,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        self.num_heads_q = num_heads_q
        self.head_dim = head_dim
        self.device = device
        self.pin_memory = pin_memory
        self.kernel_head_ratio = _kernel_head_ratio(num_heads_q // num_heads_kv)
        self.groups: dict[tuple[int, int, int], _ForkGroupBuffers] = {}
        self.workspace_query_capacity = 0
        self.workspace_split_capacity = 0
        self.num_splits: CpuGpuBuffer | None = None
        self.softmax_lse: torch.Tensor | None = None
        self.split_out: torch.Tensor | None = None
        self.split_lse: torch.Tensor | None = None

    def _copy_to_device(
        self,
        buffer: CpuGpuBuffer,
        rows: int,
    ) -> torch.Tensor:
        return buffer.copy_to_gpu(rows)

    def _make_int_buffer(self, *shape: int) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *shape,
            dtype=torch.int32,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def pack_group(
        self,
        tile: tuple[int, int, int],
        segments: list[_ForkSegment],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_ctas = len(segments)
        query_width = tile[0] // self.kernel_head_ratio
        required_blocks = max(len(segment.block_ids) for segment in segments)
        cta_capacity = next_power_of_2(num_ctas)
        block_capacity = next_power_of_2(required_blocks)
        buffers = self.groups.get(tile)
        if (
            buffers is None
            or buffers.cta_capacity < num_ctas
            or buffers.block_capacity < required_blocks
        ):
            if buffers is not None:
                cta_capacity = max(cta_capacity, buffers.cta_capacity)
                block_capacity = max(block_capacity, buffers.block_capacity)
            buffers = _ForkGroupBuffers(
                cta_capacity=cta_capacity,
                block_capacity=block_capacity,
                query_width=query_width,
                query_table=self._make_int_buffer(cta_capacity, query_width),
                block_table=self._make_int_buffer(cta_capacity, block_capacity),
                num_seqs=self._make_int_buffer(cta_capacity),
                cta_ranks=self._make_int_buffer(cta_capacity),
                kv_tokens=self._make_int_buffer(cta_capacity),
            )
            self.groups[tile] = buffers

        query_array = buffers.query_table.np
        block_array = buffers.block_table.np
        num_seqs_array = buffers.num_seqs.np
        ranks_array = buffers.cta_ranks.np
        kv_tokens_array = buffers.kv_tokens.np
        query_array[:num_ctas].fill(0)
        block_array[:num_ctas].fill(0)
        for cta_id, segment in enumerate(segments):
            if len(segment.query_ids) > query_width:
                raise _ForkPlanError("CTA query table exceeds its kernel tile")
            query_array[cta_id, : len(segment.query_ids)] = segment.query_ids
            block_array[cta_id, : len(segment.block_ids)] = segment.block_ids
            num_seqs_array[cta_id] = len(segment.query_ids)
            ranks_array[cta_id] = segment.rank
            kv_tokens_array[cta_id] = segment.num_kv_tokens

        return (
            self._copy_to_device(buffers.query_table, num_ctas),
            self._copy_to_device(buffers.block_table, num_ctas),
            self._copy_to_device(buffers.num_seqs, num_ctas),
            self._copy_to_device(buffers.cta_ranks, num_ctas),
            self._copy_to_device(buffers.kv_tokens, num_ctas),
        )

    def pack_workspace(
        self,
        num_splits_per_query: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_queries = len(num_splits_per_query)
        max_splits = max(num_splits_per_query)
        requested_query_capacity = next_power_of_2(num_queries)
        requested_split_capacity = next_power_of_2(max_splits)

        def get_workspace_bytes(query_capacity: int, split_capacity: int) -> int:
            return 4 * (
                query_capacity
                + query_capacity * self.num_heads_q
                + query_capacity * self.num_heads_q * split_capacity
                + query_capacity * self.num_heads_q * split_capacity * self.head_dim
            )

        query_capacity = max(
            self.workspace_query_capacity,
            requested_query_capacity,
        )
        split_capacity = max(
            self.workspace_split_capacity,
            requested_split_capacity,
        )
        workspace_bytes = get_workspace_bytes(query_capacity, split_capacity)
        if workspace_bytes > _MAX_FORK_WORKSPACE_BYTES:
            # Avoid an unnecessarily large Cartesian product when separate
            # batches established the query and split high-water marks.
            query_capacity = requested_query_capacity
            split_capacity = requested_split_capacity
            workspace_bytes = get_workspace_bytes(query_capacity, split_capacity)
        if workspace_bytes > _MAX_FORK_WORKSPACE_BYTES:
            raise _ForkWorkspaceLimitError(
                "ForkAttention workspace exceeds the 256 MiB safety limit"
            )

        if (
            self.num_splits is None
            or query_capacity != self.workspace_query_capacity
            or split_capacity != self.workspace_split_capacity
        ):
            self.num_splits = self._make_int_buffer(query_capacity)
            self.softmax_lse = torch.empty(
                (query_capacity, self.num_heads_q, 1),
                dtype=torch.float32,
                device=self.device,
            )
            self.split_out = torch.empty(
                (
                    query_capacity,
                    self.num_heads_q,
                    split_capacity,
                    self.head_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            self.split_lse = torch.empty(
                (query_capacity, self.num_heads_q, split_capacity),
                dtype=torch.float32,
                device=self.device,
            )
            self.workspace_query_capacity = query_capacity
            self.workspace_split_capacity = split_capacity

        assert self.num_splits is not None
        assert self.softmax_lse is not None
        assert self.split_out is not None
        assert self.split_lse is not None
        self.num_splits.np[:num_queries] = num_splits_per_query
        return (
            self._copy_to_device(self.num_splits, num_queries),
            self.softmax_lse[:num_queries],
            self.split_out[:num_queries, :, :max_splits],
            self.split_lse[:num_queries, :, :max_splits],
        )


def _fork_workspace_bytes(
    num_queries: int,
    num_heads_q: int,
    head_dim: int,
    num_splits: int = _MAX_SPLITS_PER_QUERY,
) -> int:
    return 4 * (
        num_queries
        + num_queries * num_heads_q
        + num_queries * num_heads_q * num_splits
        + num_queries * num_heads_q * num_splits * head_dim
    )


def _fork_cudagraph_metadata_bytes(
    *,
    max_model_len: int,
    block_size: int,
    head_ratio: int,
    cta_capacity: int,
) -> int:
    max_complete_blocks = max(1, max_model_len // block_size)
    block_capacity = _prefix_chunk_blocks(block_size, max_complete_blocks)
    kernel_ratio = _kernel_head_ratio(head_ratio)
    capacities = _cudagraph_group_capacities(cta_capacity)
    int_elements = 0
    for tile_m in (32, 16):
        # query IDs, block IDs, num_seqs, CTA ranks, and KV lengths.
        int_elements += capacities[tile_m] * (
            tile_m // kernel_ratio + block_capacity + 3
        )
    return int_elements * torch.int32.itemsize


def _cudagraph_group_capacities(capacity: int) -> dict[int, int]:
    return {32: max(1, capacity // 4), 16: capacity}


def _fork_cudagraph_cta_capacity(num_queries: int) -> int | None:
    if num_queries <= 1:
        return None
    capacity = max(16, next_power_of_2(2 * num_queries))
    return capacity if capacity <= _MAX_FORK_CUDAGRAPH_CTAS else None


def _iter_cudagraph_segments(
    plan: _ForkPlan,
    *,
    head_ratio: int,
    head_dim: int,
    block_size: int,
) -> Iterator[tuple[int, _ForkSegment]]:
    """Map an eager forest to the fixed M32/M16 CUDA graph kernels."""
    kernel_ratio = _kernel_head_ratio(head_ratio)
    m32_query_width = 32 // kernel_ratio
    for segment in plan.segments:
        tile_m = _get_mnw(
            segment.tile_num_queries,
            head_ratio,
            segment.num_kv_tokens,
            head_dim,
            block_size,
        )[0]
        if tile_m != 64:
            yield tile_m, segment
            continue

        # M64 cohorts are uncommon outside a wide shared prefix. Splitting
        # them into M32 cohorts preserves each query's rank and KV coverage,
        # while avoiding a third fixed kernel launch in every graph layer.
        for start in range(0, len(segment.query_ids), m32_query_width):
            yield (
                32,
                _ForkSegment(
                    query_ids=segment.query_ids[start : start + m32_query_width],
                    block_ids=segment.block_ids,
                    rank=segment.rank,
                    num_kv_tokens=segment.num_kv_tokens,
                    tile_num_queries=m32_query_width,
                ),
            )


class _ForkCUDAGraphWorkspace:
    """Fixed-address metadata and reduction workspace for one forest graph.

    Every graph uses the same two M-tile groups. Runtime plans only update
    their contents and select a prefix of the CTA rows, so graph replay never
    depends on Python list topology or newly allocated tensors.
    """

    def __init__(
        self,
        *,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        block_size: int,
        max_model_len: int,
        max_queries: int,
        max_ctas: int,
        max_splits: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        self.num_heads_q = num_heads_q
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_queries = max_queries
        self.max_ctas = max_ctas
        self.max_splits = max_splits
        self.device = device
        self.pin_memory = pin_memory
        self.kernel_head_ratio = _kernel_head_ratio(num_heads_q // num_heads_kv)
        max_complete_blocks = max(1, max_model_len // block_size)
        self.block_capacity = _prefix_chunk_blocks(block_size, max_complete_blocks)

        workspace_bytes = _fork_workspace_bytes(
            max_queries, num_heads_q, head_dim, max_splits
        ) + _fork_cudagraph_metadata_bytes(
            max_model_len=max_model_len,
            block_size=block_size,
            head_ratio=num_heads_q // num_heads_kv,
            cta_capacity=max_ctas,
        )
        if workspace_bytes > _MAX_FORK_WORKSPACE_BYTES:
            raise _ForkWorkspaceLimitError(
                "ForkAttention CUDA graph workspace exceeds the 256 MiB safety limit"
            )

        self.tiles: list[tuple[int, int, int]] = []
        self.groups: dict[int, _ForkGroupBuffers] = {}
        head_ratio = num_heads_q // num_heads_kv
        max_group_capacities = _cudagraph_group_capacities(max_ctas)
        for tile_m in (32, 16):
            graph_kv_tokens = max_model_len if tile_m == 32 else 1
            tile = _get_mnw(
                tile_m // self.kernel_head_ratio,
                head_ratio,
                graph_kv_tokens,
                head_dim,
                block_size,
            )
            if tile[0] != tile_m:
                raise _ForkPlanError("invalid CUDA graph tile configuration")
            self.tiles.append(tile)
            query_width = tile_m // self.kernel_head_ratio
            self.groups[tile_m] = _ForkGroupBuffers(
                cta_capacity=max_group_capacities[tile_m],
                block_capacity=self.block_capacity,
                query_width=query_width,
                query_table=self._make_int_buffer(
                    max_group_capacities[tile_m], query_width
                ),
                block_table=self._make_int_buffer(
                    max_group_capacities[tile_m], self.block_capacity
                ),
                num_seqs=self._make_int_buffer(max_group_capacities[tile_m]),
                cta_ranks=self._make_int_buffer(max_group_capacities[tile_m]),
                kv_tokens=self._make_int_buffer(max_group_capacities[tile_m]),
            )

        self.num_splits = self._make_int_buffer(max_queries)
        self.softmax_lse = torch.empty(
            (max_queries, num_heads_q, 1),
            dtype=torch.float32,
            device=device,
        )
        self.split_out = torch.empty(
            (max_queries, num_heads_q, max_splits, head_dim),
            dtype=torch.float32,
            device=device,
        )
        self.split_lse = torch.empty(
            (max_queries, num_heads_q, max_splits),
            dtype=torch.float32,
            device=device,
        )

    def _make_int_buffer(self, *shape: int) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *shape,
            dtype=torch.int32,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def _empty_group_arrays(
        self, buffers: _ForkGroupBuffers, cta_capacity: int
    ) -> None:
        buffers.query_table.np[:cta_capacity].fill(0)
        buffers.block_table.np[:cta_capacity].fill(0)
        buffers.num_seqs.np[:cta_capacity].fill(0)
        buffers.cta_ranks.np[:cta_capacity].fill(0)
        buffers.kv_tokens.np[:cta_capacity].fill(0)

    def pack(
        self,
        plan: _ForkPlan | None,
        *,
        query_capacity: int,
        cta_capacity: int,
        split_capacity: int,
    ) -> dict[str, Any]:
        if query_capacity <= 0 or query_capacity > self.max_queries:
            raise _ForkWorkspaceLimitError(
                "CUDA graph query capacity exceeds its fixed workspace"
            )
        if cta_capacity < _MIN_FORK_CUDAGRAPH_CTAS or cta_capacity > self.max_ctas:
            raise _ForkWorkspaceLimitError(
                "CUDA graph CTA capacity is outside its fixed workspace"
            )
        if split_capacity <= 0 or split_capacity > self.max_splits:
            raise _ForkWorkspaceLimitError(
                "CUDA graph split capacity is outside its fixed workspace"
            )

        grouped: defaultdict[int, list[_ForkSegment]] = defaultdict(list)
        max_block_id = 0
        if plan is not None:
            if len(plan.num_splits_per_query) > query_capacity:
                raise _ForkWorkspaceLimitError(
                    "CUDA graph plan has more queries than its captured graph"
                )
            max_block_id = plan.max_block_id
            if max(plan.num_splits_per_query) > split_capacity:
                raise _ForkWorkspaceLimitError(
                    "CUDA graph plan exceeds its captured split capacity"
                )
            for tile_m, segment in _iter_cudagraph_segments(
                plan,
                head_ratio=self.kernel_head_ratio,
                head_dim=self.head_dim,
                block_size=self.block_size,
            ):
                grouped[tile_m].append(segment)

        query_tables: list[torch.Tensor] = []
        block_tables: list[torch.Tensor] = []
        num_seqs_per_ctas: list[torch.Tensor] = []
        cta_ranks: list[torch.Tensor] = []
        kv_in_ctas: list[torch.Tensor] = []
        mnw: list[int] = []
        group_capacities = _cudagraph_group_capacities(cta_capacity)
        for tile in self.tiles:
            tile_m = tile[0]
            buffers = self.groups[tile_m]
            segments = grouped[tile_m]
            group_capacity = group_capacities[tile_m]
            if len(segments) > group_capacity:
                raise _ForkWorkspaceLimitError(
                    "CUDA graph plan exceeds its captured CTA capacity"
                )
            self._empty_group_arrays(buffers, group_capacity)
            for cta_id, segment in enumerate(segments):
                if len(segment.query_ids) > buffers.query_width:
                    raise _ForkPlanError("CTA query table exceeds its kernel tile")
                if len(segment.block_ids) > self.block_capacity:
                    raise _ForkWorkspaceLimitError(
                        "CUDA graph block table exceeds its fixed width"
                    )
                buffers.query_table.np[cta_id, : len(segment.query_ids)] = (
                    segment.query_ids
                )
                buffers.block_table.np[cta_id, : len(segment.block_ids)] = (
                    segment.block_ids
                )
                buffers.num_seqs.np[cta_id] = len(segment.query_ids)
                buffers.cta_ranks.np[cta_id] = segment.rank
                buffers.kv_tokens.np[cta_id] = segment.num_kv_tokens

            query_tables.append(buffers.query_table.copy_to_gpu(group_capacity))
            block_tables.append(buffers.block_table.copy_to_gpu(group_capacity))
            num_seqs_per_ctas.append(buffers.num_seqs.copy_to_gpu(group_capacity))
            cta_ranks.append(buffers.cta_ranks.copy_to_gpu(group_capacity))
            kv_in_ctas.append(buffers.kv_tokens.copy_to_gpu(group_capacity))
            mnw.extend(tile)

        self.num_splits.np[:query_capacity].fill(0)
        if plan is not None:
            self.num_splits.np[: len(plan.num_splits_per_query)] = (
                plan.num_splits_per_query
            )
        return {
            "fork_enabled": True,
            "fork_num_split_per_seq": self.num_splits.copy_to_gpu(query_capacity),
            "fork_query_tables": query_tables,
            "fork_block_tables": block_tables,
            "fork_num_seqs_per_ctas": num_seqs_per_ctas,
            "fork_cta_ranks": cta_ranks,
            "fork_kv_in_ctas": kv_in_ctas,
            "fork_mnw": mnw,
            "fork_max_split_per_seq": split_capacity,
            "fork_max_block_id": max_block_id,
            "fork_softmax_lse": self.softmax_lse[:query_capacity],
            "fork_split_out": self.split_out[:query_capacity, :, :split_capacity],
            "fork_split_lse": self.split_lse[:query_capacity, :, :split_capacity],
        }


def _flash_metadata_kwargs(metadata: FlashAttentionMetadata) -> dict[str, Any]:
    return {
        metadata_field.name: getattr(metadata, metadata_field.name)
        for metadata_field in fields(FlashAttentionMetadata)
    }


def _get_mnw(
    num_queries: int,
    head_ratio: int,
    num_kv_tokens: int,
    head_dim: int,
    page_block_size: int,
) -> tuple[int, int, int]:
    m = num_queries * _kernel_head_ratio(head_ratio)
    if m > 32:
        tile_m, warps = 64, 4
    elif m > 16:
        tile_m, warps = 32, 2
    else:
        tile_m, warps = 16, 1

    if num_kv_tokens < 32:
        tile_n = 16
    elif num_kv_tokens < 64:
        tile_n = 32
    elif num_kv_tokens < 128:
        tile_n = 64
    else:
        tile_n = 128
    if tile_m == 64:
        tile_n = max(32, tile_n)
    if head_dim == 256:
        tile_n = min(tile_n, 64)

    rows_per_thread = tile_n // (warps * 4)
    while tile_n > 16 and (
        rows_per_thread > page_block_size or page_block_size % rows_per_thread != 0
    ):
        tile_n //= 2
        rows_per_thread = tile_n // (warps * 4)
    return tile_m, tile_n, warps


def _prefix_chunk_blocks(block_size: int, max_complete_blocks: int) -> int:
    requested = max(1, cdiv(_PREFIX_CHUNK_TOKENS, block_size))
    bounded = max(1, cdiv(max_complete_blocks, 8))
    return max(requested, bounded)


def _add_trie_path(
    root: _PrefixTrieNode,
    query_id: int,
    block_ids: list[int],
) -> None:
    root.query_ids.append(query_id)
    node = root
    for block_id in block_ids:
        node = node.children.setdefault(block_id, _PrefixTrieNode())
        node.query_ids.append(query_id)
    node.terminal_query_ids.append(query_id)


def _append_segments(
    segments: list[_ForkSegment],
    query_ids: list[int],
    block_ids: list[int],
    num_kv_tokens: int,
    rank_by_query: list[int],
    max_queries_per_cta: int,
) -> None:
    if not query_ids or not block_ids or num_kv_tokens <= 0:
        return

    rank = rank_by_query[query_ids[0]]
    if any(rank_by_query[query_id] != rank for query_id in query_ids):
        raise _ForkPlanError("queries sharing a CTA must have the same next rank")

    tile_num_queries = min(len(query_ids), max_queries_per_cta)
    for start in range(0, len(query_ids), max_queries_per_cta):
        cohort = query_ids[start : start + max_queries_per_cta]
        segments.append(
            _ForkSegment(
                tuple(cohort),
                tuple(block_ids),
                rank,
                num_kv_tokens,
                tile_num_queries,
            )
        )
    for query_id in query_ids:
        rank_by_query[query_id] += 1


def _append_complete_block_segments(
    segments: list[_ForkSegment],
    query_ids: list[int],
    block_ids: list[int],
    rank_by_query: list[int],
    block_size: int,
    chunk_blocks: int,
    max_queries_per_cta: int,
) -> None:
    for start in range(0, len(block_ids), chunk_blocks):
        chunk = block_ids[start : start + chunk_blocks]
        _append_segments(
            segments,
            query_ids,
            chunk,
            len(chunk) * block_size,
            rank_by_query,
            max_queries_per_cta,
        )


def _emit_forest_segments(
    node: _PrefixTrieNode,
    segments: list[_ForkSegment],
    rank_by_query: list[int],
    block_size: int,
    chunk_blocks: int,
    max_queries_per_cta: int,
) -> None:
    for block_id, child in sorted(node.children.items()):
        block_ids = [block_id]
        current = child
        while len(current.children) == 1 and not current.terminal_query_ids:
            next_block_id, next_child = next(iter(current.children.items()))
            block_ids.append(next_block_id)
            current = next_child

        query_ids = sorted(current.query_ids)
        _append_complete_block_segments(
            segments,
            query_ids,
            block_ids,
            rank_by_query,
            block_size,
            chunk_blocks,
            max_queries_per_cta,
        )
        _emit_forest_segments(
            current,
            segments,
            rank_by_query,
            block_size,
            chunk_blocks,
            max_queries_per_cta,
        )


def _validate_fork_plan(
    segments: list[_ForkSegment],
    num_splits_per_query: list[int],
    expected_block_ids: dict[int, list[int]],
    expected_seq_lens: dict[int, int],
    num_actual_tokens: int,
    block_size: int,
) -> None:
    if len(num_splits_per_query) != num_actual_tokens:
        raise _ForkPlanError("num_split_per_seq must cover every query row")
    if expected_block_ids.keys() != expected_seq_lens.keys():
        raise _ForkPlanError("expected block and sequence metadata must agree")

    ranks_by_query: list[list[int]] = [[] for _ in range(num_actual_tokens)]
    segments_by_query: list[dict[int, _ForkSegment]] = [
        {} for _ in range(num_actual_tokens)
    ]
    for segment in segments:
        if not segment.query_ids:
            raise _ForkPlanError("every CTA must contain at least one query")
        if len(set(segment.query_ids)) != len(segment.query_ids):
            raise _ForkPlanError("a query may occur at most once in a CTA")
        if segment.tile_num_queries < len(segment.query_ids):
            raise _ForkPlanError("CTA tile capacity cannot be smaller than its queries")
        if not segment.block_ids or any(block_id < 0 for block_id in segment.block_ids):
            raise _ForkPlanError("every block ID must be non-negative")
        if segment.rank < 0:
            raise _ForkPlanError("CTA ranks must be non-negative")
        required_blocks = cdiv(segment.num_kv_tokens, block_size)
        if required_blocks != len(segment.block_ids):
            raise _ForkPlanError("CTA KV length does not match its block IDs")

        for query_id in segment.query_ids:
            if query_id < 0 or query_id >= num_actual_tokens:
                raise _ForkPlanError("query ID is outside the query tensor")
            if query_id not in expected_block_ids:
                raise _ForkPlanError("inactive query ID occurs in a CTA")
            if segment.rank in segments_by_query[query_id]:
                raise _ForkPlanError("a query has duplicate CTA ranks")
            ranks_by_query[query_id].append(segment.rank)
            segments_by_query[query_id][segment.rank] = segment

    for query_id in range(num_actual_tokens):
        split_count = num_splits_per_query[query_id]
        if split_count < 0 or split_count > _MAX_SPLITS_PER_QUERY:
            raise _ForkPlanError("num_split_per_seq is outside the kernel range")
        expected_ranks = list(range(split_count))
        if len(ranks_by_query[query_id]) != split_count:
            raise _ForkPlanError(
                "num_split_per_seq does not match the emitted CTA count"
            )
        if sorted(ranks_by_query[query_id]) != expected_ranks:
            raise _ForkPlanError(
                "each query must have continuous, unique ranks starting at zero"
            )

        if query_id not in expected_block_ids:
            if split_count != 0:
                raise _ForkPlanError("inactive queries must have zero splits")
            continue

        planned_blocks: list[int] = []
        planned_kv_tokens = 0
        for rank in expected_ranks:
            segment = segments_by_query[query_id][rank]
            planned_blocks.extend(segment.block_ids)
            planned_kv_tokens += segment.num_kv_tokens
        if planned_blocks != expected_block_ids[query_id]:
            raise _ForkPlanError(
                "CTA block IDs must exactly cover the active request block table"
            )
        if planned_kv_tokens != expected_seq_lens[query_id]:
            raise _ForkPlanError(
                "CTA KV lengths must exactly cover the request sequence"
            )


def _build_fork_plan(
    *,
    query_start_locs: list[int],
    seq_lens: list[int],
    block_rows: list[list[int]] | np.ndarray,
    block_row_indices: np.ndarray | None = None,
    num_actual_tokens: int,
    block_size: int,
    head_ratio: int,
    require_shared: bool,
) -> _ForkPlan | None:
    num_reqs = len(seq_lens)
    if num_actual_tokens <= 0 or block_size <= 0 or head_ratio <= 0:
        raise _ForkPlanError("plan dimensions must be positive")
    if len(query_start_locs) != num_reqs + 1:
        raise _ForkPlanError("request metadata has inconsistent row counts")
    if block_row_indices is None:
        if len(block_rows) != num_reqs:
            raise _ForkPlanError("request metadata has inconsistent row counts")
    elif len(block_row_indices) < num_reqs:
        raise _ForkPlanError("block-table row mapping is too short")
    if not query_start_locs or query_start_locs[0] != 0:
        raise _ForkPlanError("query start locations must begin at zero")
    if any(end < start for start, end in zip(query_start_locs, query_start_locs[1:])):
        raise _ForkPlanError("query start locations must be monotonic")
    if query_start_locs[-1] != num_actual_tokens:
        raise _ForkPlanError(
            "query start locations must exactly cover the query tensor"
        )

    # Estimating the plan cost is O(batch), unlike copying every active block
    # ID and rebuilding the trie. Long-context batches conservatively use the
    # inherited FlashAttention path until the plan can be updated incrementally.
    block_visits = sum(
        cdiv(seq_lens[req_id], block_size)
        for req_id, (query_start, query_end) in enumerate(
            zip(query_start_locs, query_start_locs[1:])
        )
        if query_end > query_start
    )
    if block_visits > _MAX_FORK_PLAN_BLOCK_VISITS:
        return None

    kernel_ratio = _kernel_head_ratio(head_ratio)
    max_queries_per_cta = max(1, 64 // kernel_ratio)
    root = _PrefixTrieNode()
    partial_segments: list[tuple[int, int, int]] = []
    expected_block_ids: dict[int, list[int]] = {}
    expected_seq_lens: dict[int, int] = {}
    max_complete_blocks = 0

    for req_id, seq_len in enumerate(seq_lens):
        query_start = query_start_locs[req_id]
        query_end = query_start_locs[req_id + 1]
        query_len = query_end - query_start
        if query_len == 0:
            continue
        if query_len != 1:
            return None
        query_id = query_start
        if seq_len <= 0:
            raise _ForkPlanError("active requests must have a positive sequence length")

        block_row_id = (
            req_id if block_row_indices is None else int(block_row_indices[req_id])
        )
        if block_row_id < 0 or block_row_id >= len(block_rows):
            raise _ForkPlanError("block-table row mapping is out of range")
        required_blocks = cdiv(seq_len, block_size)
        if len(block_rows[block_row_id]) < required_blocks:
            raise _ForkPlanError("an active block-table row is too short")
        active_blocks = [
            int(block_id) for block_id in block_rows[block_row_id][:required_blocks]
        ]
        if any(block_id < 0 for block_id in active_blocks):
            raise _ForkPlanError("active block IDs must be non-negative")
        expected_block_ids[query_id] = active_blocks
        expected_seq_lens[query_id] = seq_len

        complete_blocks = seq_len // block_size
        partial_tokens = seq_len % block_size
        max_complete_blocks = max(max_complete_blocks, complete_blocks)
        if complete_blocks:
            _add_trie_path(root, query_id, active_blocks[:complete_blocks])
        if partial_tokens:
            partial_segments.append(
                (query_id, active_blocks[complete_blocks], partial_tokens)
            )

    if not expected_block_ids:
        return None

    rank_by_query = [0] * num_actual_tokens
    segments: list[_ForkSegment] = []
    chunk_blocks = _prefix_chunk_blocks(block_size, max(1, max_complete_blocks))
    _emit_forest_segments(
        root,
        segments,
        rank_by_query,
        block_size,
        chunk_blocks,
        max_queries_per_cta,
    )
    for query_id, block_id, partial_tokens in partial_segments:
        _append_segments(
            segments,
            [query_id],
            [block_id],
            partial_tokens,
            rank_by_query,
            max_queries_per_cta,
        )

    if require_shared and not any(len(segment.query_ids) > 1 for segment in segments):
        return None
    if max(rank_by_query, default=0) > _MAX_SPLITS_PER_QUERY:
        return None

    _validate_fork_plan(
        segments,
        rank_by_query,
        expected_block_ids,
        expected_seq_lens,
        num_actual_tokens,
        block_size,
    )
    return _ForkPlan(
        tuple(segments),
        tuple(rank_by_query),
        max(block_id for blocks in expected_block_ids.values() for block_id in blocks),
    )


def _pack_fork_plan(
    plan: _ForkPlan,
    *,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    page_block_size: int,
    device: torch.device,
    buffer_pool: _ForkBufferPool | None = None,
) -> dict[str, Any]:
    if buffer_pool is None:
        if device.type != "cpu":
            raise ValueError("CUDA packing requires a persistent buffer pool")
        buffer_pool = _ForkBufferPool(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            device=device,
        )
    head_ratio = num_heads_q // num_heads_kv
    grouped: defaultdict[tuple[int, int, int], list[_ForkSegment]] = defaultdict(list)
    for segment in plan.segments:
        grouped[
            _get_mnw(
                segment.tile_num_queries,
                head_ratio,
                segment.num_kv_tokens,
                head_dim,
                page_block_size,
            )
        ].append(segment)

    query_tables: list[torch.Tensor] = []
    block_tables: list[torch.Tensor] = []
    num_seqs_per_ctas: list[torch.Tensor] = []
    cta_ranks: list[torch.Tensor] = []
    kv_in_ctas: list[torch.Tensor] = []
    mnw: list[int] = []
    for tile, segments in sorted(grouped.items(), reverse=True):
        (
            query_table,
            block_table,
            num_seqs_per_cta,
            cta_rank,
            kv_in_cta,
        ) = buffer_pool.pack_group(tile, segments)
        query_tables.append(query_table)
        block_tables.append(block_table)
        num_seqs_per_ctas.append(num_seqs_per_cta)
        cta_ranks.append(cta_rank)
        kv_in_ctas.append(kv_in_cta)
        mnw.extend(tile)

    (
        num_split_per_seq,
        softmax_lse,
        split_out,
        split_lse,
    ) = buffer_pool.pack_workspace(plan.num_splits_per_query)
    max_splits = max(plan.num_splits_per_query)
    return {
        "fork_enabled": True,
        "fork_num_split_per_seq": num_split_per_seq,
        "fork_query_tables": query_tables,
        "fork_block_tables": block_tables,
        "fork_num_seqs_per_ctas": num_seqs_per_ctas,
        "fork_cta_ranks": cta_ranks,
        "fork_kv_in_ctas": kv_in_ctas,
        "fork_mnw": mnw,
        "fork_max_split_per_seq": max_splits,
        "fork_max_block_id": plan.max_block_id,
        "fork_softmax_lse": softmax_lse,
        "fork_split_out": split_out,
        "fork_split_lse": split_lse,
    }


def _get_plan_cudagraph_requirements(
    plan: _ForkPlan,
    *,
    head_ratio: int,
    head_dim: int,
    block_size: int,
) -> tuple[int, int]:
    """Return independent CTA and split buckets for graph dispatch."""
    counts: defaultdict[int, int] = defaultdict(int)
    for tile_m, _ in _iter_cudagraph_segments(
        plan,
        head_ratio=head_ratio,
        head_dim=head_dim,
        block_size=block_size,
    ):
        counts[tile_m] += 1
    cta_requirement = max(counts[16], 4 * counts[32])
    split_requirement = next_power_of_2(max(plan.num_splits_per_query))
    return cta_requirement, split_requirement


class ForkAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    supports_update_block_table: bool = False
    _fork_cpu_metadata: tuple[torch.Tensor, np.ndarray, np.ndarray | None] | None = None
    _fork_prebuilt_plan: _ForkPlan | None = None
    _fork_cudagraph_capacity: int | None = None
    _fork_cudagraph_max_splits: int | None = None
    _fork_cudagraph_force_flash: bool = False
    _fork_cudagraph_workspace_limits: tuple[int, int, int] | None = None

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        # The legacy runner cannot inject an exact forest plan before graph
        # dispatch. Keep its existing eager ForkAttention path unchanged.
        if vllm_config is None or not vllm_config.use_v2_model_runner:
            return AttentionCGSupport.NEVER
        if envs.VLLM_BATCH_INVARIANT:
            return AttentionCGSupport.NEVER
        return cls._cudagraph_support

    def _get_cudagraph_capture_limits(self) -> tuple[int, int, int]:
        """Return the largest captured (queries, CTAs, splits) workspace."""
        if self.dcp_world_size > 1:
            return 0, 0, 0
        if self.vllm_config.model_config.is_mm_prefix_lm:
            return 0, 0, 0
        if getattr(self.kv_cache_spec, "sliding_window", None) is not None:
            return 0, 0, 0
        if self.vllm_config.scheduler_config.async_scheduling:
            return 0, 0, 0

        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        capture_sizes = getattr(compilation_config, "cudagraph_capture_sizes", None)
        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
        if (
            not capture_sizes
            or cudagraph_mode is None
            or not cudagraph_mode.separate_routine()
            or cudagraph_mode.decode_mode() != CUDAGraphMode.FULL
        ):
            return 0, 0, 0

        max_blocks = cdiv(self.vllm_config.model_config.max_model_len, self.block_size)
        chunk_blocks = _prefix_chunk_blocks(self.block_size, max_blocks)
        # Reserve two ranks for the branch and final partial block. Forests
        # with more branch points safely fall back to the Flash graph.
        expected_splits = cdiv(max_blocks, chunk_blocks) + 2
        max_splits = min(
            _MAX_SPLITS_PER_QUERY,
            next_power_of_2(expected_splits),
        )

        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        candidate_sizes = sorted(
            {int(size) for size in capture_sizes if 1 < int(size) <= max_num_seqs},
            reverse=True,
        )
        for max_queries in candidate_sizes:
            max_ctas = _fork_cudagraph_cta_capacity(max_queries)
            if max_ctas is None:
                continue
            workspace_bytes = _fork_workspace_bytes(
                max_queries,
                self.num_heads_q,
                self.headdim,
                max_splits,
            ) + _fork_cudagraph_metadata_bytes(
                max_model_len=self.vllm_config.model_config.max_model_len,
                block_size=self.block_size,
                head_ratio=self.num_heads_q // self.num_heads_kv,
                cta_capacity=max_ctas,
            )
            if workspace_bytes <= _MAX_FORK_WORKSPACE_BYTES:
                return max_queries, max_ctas, max_splits
        return 0, 0, 0

    def get_cudagraph_max_reqs(self) -> int:
        return self._get_cudagraph_capture_limits()[0]

    def get_cudagraph_max_splits(self) -> int:
        return self._get_cudagraph_capture_limits()[2]

    def set_cudagraph_workspace_limits(
        self,
        max_queries: int,
        max_splits: int,
    ) -> None:
        """Apply the common capture limits shared by all attention groups."""
        if (max_queries == 0) != (max_splits == 0):
            raise _ForkWorkspaceLimitError(
                "ForkAttention CUDA graph workspace limits are incomplete"
            )
        if max_queries == 0:
            self._fork_cudagraph_workspace_limits = (0, 0, 0)
            return

        local_queries, _, local_splits = self._get_cudagraph_capture_limits()
        max_ctas = _fork_cudagraph_cta_capacity(max_queries)
        if max_ctas is None or max_queries > local_queries or max_splits > local_splits:
            raise _ForkWorkspaceLimitError(
                "ForkAttention CUDA graph workspace limits exceed builder capacity"
            )
        self._fork_cudagraph_workspace_limits = (
            max_queries,
            max_ctas,
            max_splits,
        )

    def set_cudagraph_plan(
        self,
        capacity: int | None,
        max_splits: int | None,
        *,
        force_flash: bool,
    ) -> None:
        if (capacity is None) != (max_splits is None):
            raise _ForkWorkspaceLimitError(
                "ForkAttention CUDA graph plan is incomplete"
            )
        if capacity is not None and (
            capacity < _MIN_FORK_CUDAGRAPH_CTAS or capacity > _MAX_FORK_CUDAGRAPH_CTAS
        ):
            raise _ForkWorkspaceLimitError(
                "ForkAttention CUDA graph CTA capacity is unsupported"
            )
        if max_splits is not None and not (0 < max_splits <= _MAX_SPLITS_PER_QUERY):
            raise _ForkWorkspaceLimitError(
                "ForkAttention CUDA graph split capacity is unsupported"
            )
        self._fork_cudagraph_capacity = capacity
        self._fork_cudagraph_max_splits = max_splits
        self._fork_cudagraph_force_flash = force_flash

    def clear_prebuilt_plan(self) -> None:
        self._fork_prebuilt_plan = None

    def prepare_cudagraph_plan(
        self,
        seq_lens: torch.Tensor | np.ndarray,
        block_table: np.ndarray,
        block_table_indices: np.ndarray,
        num_kv_blocks: int,
    ) -> tuple[int, int] | None:
        """Build the exact forest once, before CUDA graph dispatch."""
        self._fork_prebuilt_plan = None
        num_reqs = len(block_table_indices)
        if num_reqs <= 1 or self.dcp_world_size > 1:
            return None
        if self.vllm_config.scheduler_config.async_scheduling:
            return None
        if envs.VLLM_BATCH_INVARIANT:
            return None

        if isinstance(seq_lens, torch.Tensor):
            if seq_lens.device.type != "cpu":
                raise _ForkPlanError(
                    "ForkAttention CPU sequence lengths must be on CPU"
                )
            seq_lens_list = [int(value) for value in seq_lens[:num_reqs].tolist()]
        else:
            seq_lens_list = [int(value) for value in seq_lens[:num_reqs]]
        plan = _build_fork_plan(
            query_start_locs=list(range(num_reqs + 1)),
            seq_lens=seq_lens_list,
            block_rows=block_table,
            block_row_indices=block_table_indices,
            num_actual_tokens=num_reqs,
            block_size=self.block_size,
            head_ratio=self.num_heads_q // self.num_heads_kv,
            require_shared=True,
        )
        if plan is None:
            return None
        if plan.max_block_id >= num_kv_blocks:
            raise _ForkPlanError(
                "ForkAttention CUDA graph plan contains an invalid block ID"
            )
        self._fork_prebuilt_plan = plan
        return _get_plan_cudagraph_requirements(
            plan,
            head_ratio=self.num_heads_q // self.num_heads_kv,
            head_dim=self.headdim,
            block_size=self.block_size,
        )

    def set_cpu_metadata(
        self,
        seq_lens: torch.Tensor,
        block_table: np.ndarray,
        block_table_indices: np.ndarray | None = None,
    ) -> None:
        self._fork_cpu_metadata = (seq_lens, block_table, block_table_indices)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ForkAttentionMetadata:
        # Consume runner-owned metadata before any operation that may fail.
        # Builders persist across batches, so metadata must never remain attached.
        cpu_metadata = getattr(self, "_fork_cpu_metadata", None)
        self._fork_cpu_metadata = None
        prebuilt_plan = getattr(self, "_fork_prebuilt_plan", None)
        self._fork_prebuilt_plan = None
        base_metadata = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        fork_kwargs = self._build_fork_kwargs(
            base_metadata,
            common_attn_metadata.query_start_loc_cpu,
            common_attn_metadata.num_reqs,
            cpu_metadata,
            prebuilt_plan,
        )
        return ForkAttentionMetadata(
            **_flash_metadata_kwargs(base_metadata),
            **fork_kwargs,
        )

    def _build_fork_kwargs(
        self,
        metadata: FlashAttentionMetadata,
        query_start_loc_cpu: torch.Tensor,
        num_reqs: int,
        cpu_metadata: tuple[torch.Tensor, np.ndarray, np.ndarray | None] | None,
        prebuilt_plan: _ForkPlan | None = None,
    ) -> dict[str, Any]:
        graph_capacity = getattr(self, "_fork_cudagraph_capacity", None)
        graph_max_splits = getattr(self, "_fork_cudagraph_max_splits", None)
        if getattr(self, "_fork_cudagraph_force_flash", False):
            return {}
        if not self._can_build_fork(metadata, num_reqs):
            if graph_capacity is not None:
                raise _ForkPlanError(
                    "captured ForkAttention graph received incompatible metadata"
                )
            return {}

        plan: _ForkPlan | None
        if prebuilt_plan is not None:
            plan = prebuilt_plan
        elif cpu_metadata is None:
            if graph_capacity is not None:
                raise _ForkPlanError(
                    "captured ForkAttention graph is missing its runtime plan"
                )
            return {}
        else:
            seq_lens_cpu, block_table_cpu, block_table_indices = cpu_metadata
            if seq_lens_cpu.device.type != "cpu":
                raise _ForkPlanError(
                    "ForkAttention CPU sequence lengths must be on CPU"
                )
            query_start_locs = [
                int(value) for value in query_start_loc_cpu[: num_reqs + 1].tolist()
            ]
            seq_lens = [int(value) for value in seq_lens_cpu[:num_reqs].tolist()]
            max_blocks = max(cdiv(seq_len, self.block_size) for seq_len in seq_lens)
            if (
                metadata.block_table.shape[1] < max_blocks
                or block_table_cpu.shape[1] < max_blocks
            ):
                raise _ForkPlanError("block table cannot cover the active sequences")
            if block_table_indices is None:
                if block_table_cpu.shape[0] < num_reqs:
                    raise _ForkPlanError("block table cannot cover the active requests")
                block_rows = block_table_cpu[:num_reqs]
            else:
                if len(block_table_indices) < num_reqs:
                    raise _ForkPlanError("block-table row mapping is too short")
                block_rows = block_table_cpu

            plan = _build_fork_plan(
                query_start_locs=query_start_locs,
                seq_lens=seq_lens,
                block_rows=block_rows,
                block_row_indices=block_table_indices,
                num_actual_tokens=metadata.num_actual_tokens,
                block_size=self.block_size,
                head_ratio=self.num_heads_q // self.num_heads_kv,
                require_shared=True,
            )
        if plan is None:
            if graph_capacity is not None:
                raise _ForkPlanError(
                    "captured ForkAttention graph received an empty runtime plan"
                )
            return {}
        if graph_capacity is not None:
            assert graph_max_splits is not None
            workspace = self._get_fork_cudagraph_workspace(metadata.block_table.device)
            return workspace.pack(
                plan,
                query_capacity=metadata.num_actual_tokens,
                cta_capacity=graph_capacity,
                split_capacity=graph_max_splits,
            )

        buffer_pool = getattr(self, "_fork_buffer_pool", None)
        if buffer_pool is None:
            buffer_pool = _ForkBufferPool(
                num_heads_q=self.num_heads_q,
                num_heads_kv=self.num_heads_kv,
                head_dim=self.headdim,
                device=metadata.block_table.device,
                pin_memory=(not self.vllm_config.use_v2_model_runner and PIN_MEMORY),
            )
            self._fork_buffer_pool = buffer_pool
        try:
            return _pack_fork_plan(
                plan,
                num_heads_q=self.num_heads_q,
                num_heads_kv=self.num_heads_kv,
                head_dim=self.headdim,
                page_block_size=self.block_size,
                device=metadata.block_table.device,
                buffer_pool=buffer_pool,
            )
        except _ForkWorkspaceLimitError:
            return {}

    def _get_fork_cudagraph_workspace(
        self, device: torch.device
    ) -> _ForkCUDAGraphWorkspace:
        workspace = getattr(self, "_fork_cudagraph_workspace", None)
        if workspace is None:
            limits = getattr(self, "_fork_cudagraph_workspace_limits", None)
            if limits is None:
                limits = self._get_cudagraph_capture_limits()
            max_queries, max_ctas, max_splits = limits
            if max_queries == 0:
                raise _ForkWorkspaceLimitError(
                    "ForkAttention has no eligible CUDA graph capture size"
                )
            workspace = _ForkCUDAGraphWorkspace(
                num_heads_q=self.num_heads_q,
                num_heads_kv=self.num_heads_kv,
                head_dim=self.headdim,
                block_size=self.block_size,
                max_model_len=self.vllm_config.model_config.max_model_len,
                max_queries=max_queries,
                max_ctas=max_ctas,
                max_splits=max_splits,
                device=device,
                pin_memory=PIN_MEMORY,
            )
            self._fork_cudagraph_workspace = workspace
        return workspace

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> ForkAttentionMetadata:
        self._fork_cpu_metadata = None
        self._fork_prebuilt_plan = None
        base_metadata = FlashAttentionMetadataBuilder.build(
            self,
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        graph_capacity = getattr(self, "_fork_cudagraph_capacity", None)
        graph_max_splits = getattr(self, "_fork_cudagraph_max_splits", None)
        fork_kwargs: dict[str, Any] = {}
        if graph_capacity is not None:
            assert graph_max_splits is not None
            workspace = self._get_fork_cudagraph_workspace(
                base_metadata.block_table.device
            )
            fork_kwargs = workspace.pack(
                None,
                query_capacity=base_metadata.num_actual_tokens,
                cta_capacity=graph_capacity,
                split_capacity=graph_max_splits,
            )
        return ForkAttentionMetadata(
            **_flash_metadata_kwargs(base_metadata),
            **fork_kwargs,
        )

    def _can_build_fork(
        self,
        metadata: FlashAttentionMetadata,
        num_reqs: int,
    ) -> bool:
        if self.vllm_config.scheduler_config.async_scheduling:
            return False
        if envs.VLLM_BATCH_INVARIANT:
            return False
        if num_reqs <= 1 or metadata.num_actual_tokens <= 1:
            return False
        if metadata.max_query_len != 1:
            return False
        if metadata.causal is not True:
            return False
        if metadata.dcp_context_kv_lens is not None:
            return False
        if metadata.mm_prefix_range_tensor is not None:
            return False
        if metadata.rswa_prefix_lens is not None:
            return False
        if self.headdim not in (64, 128, 256):
            return False
        if self.block_size % 16 != 0:
            return False
        if self.kv_cache_dtype not in (
            "auto",
            "float16",
            "bfloat16",
            torch.float16,
            torch.bfloat16,
        ):
            return False
        return self.num_heads_kv > 0 and self.num_heads_q % self.num_heads_kv == 0


class ForkAttentionBackend(FlashAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in cls.get_supported_head_sizes()

    @staticmethod
    def get_name() -> str:
        return "FORK_ATTN"

    @staticmethod
    def get_impl_cls() -> type["ForkAttentionImpl"]:
        return ForkAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["ForkAttentionMetadataBuilder"]:
        return ForkAttentionMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return current_platform.is_cuda() and capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False


class ForkAttentionImpl(FlashAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._can_run_fork(
            attn_metadata,
            query,
            kv_cache,
            output,
            output_scale,
            output_block_scale,
        ):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        metadata = cast(ForkAttentionMetadata, attn_metadata)
        required = (
            metadata.fork_num_split_per_seq,
            metadata.fork_query_tables,
            metadata.fork_block_tables,
            metadata.fork_num_seqs_per_ctas,
            metadata.fork_cta_ranks,
            metadata.fork_kv_in_ctas,
            metadata.fork_mnw,
            metadata.fork_softmax_lse,
            metadata.fork_split_out,
            metadata.fork_split_lse,
        )
        if any(value is None for value in required):
            raise RuntimeError("ForkAttention metadata is incomplete")

        key_cache, value_cache = kv_cache.unbind(1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if (
            metadata.fork_max_block_id < 0
            or metadata.fork_max_block_id >= key_cache.shape[0]
        ):
            raise RuntimeError("ForkAttention metadata contains an invalid block ID")

        num_tokens = metadata.num_actual_tokens
        fork_num_split_per_seq = cast(torch.Tensor, metadata.fork_num_split_per_seq)
        if fork_num_split_per_seq.shape[0] < num_tokens:
            raise RuntimeError(
                "ForkAttention split metadata does not cover all queries"
            )

        q = query[:num_tokens].view(num_tokens, 1, self.num_heads, self.head_size)
        out = output[:num_tokens].view(
            num_tokens,
            1,
            self.num_heads,
            self.head_size,
        )
        ops.fork_attention(
            out,
            cast(torch.Tensor, metadata.fork_softmax_lse),
            cast(torch.Tensor, metadata.fork_split_out),
            cast(torch.Tensor, metadata.fork_split_lse),
            q,
            key_cache,
            value_cache,
            fork_num_split_per_seq,
            cast(list[torch.Tensor], metadata.fork_query_tables),
            cast(list[torch.Tensor], metadata.fork_block_tables),
            cast(list[torch.Tensor], metadata.fork_num_seqs_per_ctas),
            cast(list[torch.Tensor], metadata.fork_cta_ranks),
            cast(list[torch.Tensor], metadata.fork_kv_in_ctas),
            cast(list[int], metadata.fork_mnw),
            metadata.fork_max_split_per_seq,
            self.scale,
        )
        return output

    def _can_run_fork(
        self,
        attn_metadata: FlashAttentionMetadata | None,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        if (
            not isinstance(attn_metadata, ForkAttentionMetadata)
            or not attn_metadata.fork_enabled
        ):
            return False
        if output_scale is not None or output_block_scale is not None:
            return False
        if query.dtype not in (torch.float16, torch.bfloat16):
            return False
        if kv_cache.dtype != query.dtype or output.dtype != query.dtype:
            return False
        if self.attn_type != AttentionType.DECODER:
            return False
        if is_quantized_kv_cache(self.kv_cache_dtype):
            return False
        if self.alibi_slopes is not None:
            return False
        if self.sliding_window != (-1, -1):
            return False
        if self.logits_soft_cap != 0:
            return False
        return self.sinks is None
