# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ForkAttention backend.

This backend uses FlashAttention as the general fallback and routes eligible
single-token decode batches with a shared prefix to the ForkAttention CUDA
kernel.
"""

import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, cast

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import (
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
from vllm.v1.kv_cache_interface import AttentionSpec, get_block_table_num_blocks

logger = init_logger(__name__)


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
    fork_softmax_lse: torch.Tensor | None = None
    fork_split_out: torch.Tensor | None = None
    fork_split_lse: torch.Tensor | None = None


@dataclass
class _ForkCUDAGraphWorkspace:
    max_reqs: int
    max_active_reqs: int
    max_blocks: int
    prefix_chunk_capacity_blocks: int
    max_prefix_chunks: int
    prefix_queries_per_cta: int
    prefix_cohorts: int
    prefix_ctas: int
    query_tables: list[torch.Tensor]
    block_tables: list[torch.Tensor]
    num_seqs_per_ctas: list[torch.Tensor]
    cta_ranks: list[torch.Tensor]
    kv_in_ctas: list[torch.Tensor]
    num_split_per_seq: torch.Tensor
    softmax_lse: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    mnw: list[int]


@dataclass
class _ForkForestCUDAGraphWorkspace:
    max_reqs: int
    max_active_reqs: int
    max_blocks: int
    max_ctas: int
    chunk_blocks: int
    max_split_per_seq: int
    max_q_per_cta_by_group: list[int]
    query_tables: list[torch.Tensor]
    block_tables: list[torch.Tensor]
    num_seqs_per_ctas: list[torch.Tensor]
    cta_ranks: list[torch.Tensor]
    kv_in_ctas: list[torch.Tensor]
    query_tables_cpu: list[torch.Tensor]
    block_tables_cpu: list[torch.Tensor]
    num_seqs_per_ctas_cpu: list[torch.Tensor]
    cta_ranks_cpu: list[torch.Tensor]
    kv_in_ctas_cpu: list[torch.Tensor]
    num_split_per_seq_cpu: torch.Tensor
    num_split_per_seq: torch.Tensor
    softmax_lse: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    mnw: list[int]


@dataclass
class _PrefixTrieNode:
    req_ids: list[int] = field(default_factory=list)
    terminal_reqs: list[int] = field(default_factory=list)
    children: dict[int, "_PrefixTrieNode"] = field(default_factory=dict)


@dataclass(frozen=True)
class _ForkSegmentBox:
    q_ids: list[int]
    blocks: list[int]
    rank: int
    kv_len: int


def _flash_metadata_kwargs(
    metadata: FlashAttentionMetadata,
) -> dict[str, Any]:
    return {
        field.name: getattr(metadata, field.name)
        for field in fields(FlashAttentionMetadata)
    }


def _get_mnw(
    num_seqs: int,
    hratio: int,
    kv_len: int,
    page_block_size: int | None = None,
) -> tuple[int, int, int]:
    m_val = num_seqs * hratio
    if m_val > 32:
        tile_m, warps = 64, 4
    elif m_val > 16:
        tile_m, warps = 32, 2
    else:
        tile_m, warps = 16, 1

    if kv_len < 32:
        tile_n = 16
    elif kv_len < 64:
        tile_n = 32
    elif kv_len < 128:
        tile_n = 64
    else:
        tile_n = 128
    if tile_m == 64:
        tile_n = max(32, tile_n)
    if page_block_size is not None:
        while tile_n > 16 and tile_n // (warps * 32 // 8) > page_block_size:
            tile_n //= 2
    tail_tile_n = envs.VLLM_FORK_ATTN_TAIL_TILE_N
    if tail_tile_n not in (0, 16, 32, 64, 128):
        raise ValueError("VLLM_FORK_ATTN_TAIL_TILE_N must be one of 0, 16, 32, 64, 128")
    if tile_m == 16 and tail_tile_n > 0 and kv_len >= 128:
        tile_n = min(tile_n, tail_tile_n)
    return tile_m, tile_n, warps


def _is_supported_fork_kv_cache_dtype(kv_cache_dtype: object) -> bool:
    return kv_cache_dtype in (
        "auto",
        "float16",
        "bfloat16",
        torch.float16,
        torch.bfloat16,
    )


def _fork_profile_enabled(vllm_config: VllmConfig | None) -> bool:
    if os.environ.get("PROFILE_FORK") == "1":
        return True
    if vllm_config is None:
        return False
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", {})
    if isinstance(extra_config, dict) and extra_config.get("fanout_profile", False):
        return True
    profiler_config = getattr(vllm_config, "profiler_config", None)
    return getattr(profiler_config, "profiler", None) is not None


def _get_prefix_chunk_blocks(block_size: int, max_blocks: int) -> int:
    requested_tokens = envs.VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE
    if requested_tokens <= 0:
        raise ValueError("VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE must be positive")
    requested_blocks = max(1, (requested_tokens + block_size - 1) // block_size)
    # Bound the graph topology while keeping 2K chunks for prefixes up to 16K.
    min_blocks = max(1, (max_blocks + 7) // 8)
    return max(requested_blocks, min_blocks)


def _get_adaptive_prefix_chunk_blocks(
    *,
    block_size: int,
    prefix_blocks: int,
    base_chunk_blocks: int,
    num_reqs: int,
    num_kv_heads: int,
    num_sms: int,
    prefix_cohorts: int,
    max_prefix_chunks: int,
) -> int:
    target_waves = envs.VLLM_FORK_ATTN_TARGET_CTA_WAVES
    min_tokens = envs.VLLM_FORK_ATTN_ADAPTIVE_SPLIT_MIN_TOKENS
    if target_waves < 0:
        raise ValueError("VLLM_FORK_ATTN_TARGET_CTA_WAVES must be non-negative")
    if min_tokens < 0:
        raise ValueError(
            "VLLM_FORK_ATTN_ADAPTIVE_SPLIT_MIN_TOKENS must be non-negative"
        )
    if (
        target_waves == 0
        or prefix_blocks * block_size < min_tokens
        or num_reqs <= 0
        or num_kv_heads <= 0
        or num_sms <= 0
        or prefix_cohorts <= 0
        or max_prefix_chunks <= 0
    ):
        return base_chunk_blocks

    base_chunks = (prefix_blocks + base_chunk_blocks - 1) // base_chunk_blocks
    target_plan_ctas = (target_waves * num_sms + num_kv_heads - 1) // num_kv_heads
    required_prefix_ctas = max(0, target_plan_ctas - num_reqs)
    target_chunks = (required_prefix_ctas + prefix_cohorts - 1) // prefix_cohorts
    target_chunks = min(max_prefix_chunks, max(base_chunks, target_chunks))
    if target_chunks <= base_chunks:
        return base_chunk_blocks
    return max(1, (prefix_blocks + target_chunks - 1) // target_chunks)


def _get_default_prefix_chunk_bucket(block_size: int, max_model_len: int) -> int:
    max_blocks = get_block_table_num_blocks(max_model_len, block_size)
    prefix_chunk_blocks = _get_prefix_chunk_blocks(block_size, max_blocks)
    return (max_blocks + prefix_chunk_blocks - 1) // prefix_chunk_blocks


def _get_default_forest_max_split_per_seq(
    block_size: int,
    max_model_len: int,
) -> int:
    max_split_per_seq = envs.VLLM_FORK_ATTN_FOREST_MAX_SPLITS
    if max_split_per_seq > 0:
        if max_split_per_seq > 32:
            raise ValueError("VLLM_FORK_ATTN_FOREST_MAX_SPLITS must be <= 32")
        return max_split_per_seq
    # Branch points add splits independently of sequence length. Reserve the
    # full gather-kernel range so a valid captured plan cannot under-allocate
    # its per-request split workspace.
    return 32


def _add_trie_path(root: _PrefixTrieNode, req_id: int, blocks: list[int]) -> None:
    root.req_ids.append(req_id)
    node = root
    for block in blocks:
        node = node.children.setdefault(block, _PrefixTrieNode())
        node.req_ids.append(req_id)
    node.terminal_reqs.append(req_id)


def _append_segment_boxes(
    boxes: list[_ForkSegmentBox],
    q_ids: list[int],
    blocks: list[int],
    kv_len: int,
    rank_by_req: list[int],
    max_q_per_cta: int,
) -> None:
    if not q_ids or not blocks or kv_len <= 0:
        return
    rank = rank_by_req[q_ids[0]]
    for q_id in q_ids:
        assert rank_by_req[q_id] == rank
    for start in range(0, len(q_ids), max_q_per_cta):
        cohort = q_ids[start : start + max_q_per_cta]
        boxes.append(_ForkSegmentBox(cohort, blocks, rank, kv_len))
    for q_id in q_ids:
        rank_by_req[q_id] += 1


def _append_complete_block_segments(
    boxes: list[_ForkSegmentBox],
    q_ids: list[int],
    blocks: list[int],
    rank_by_req: list[int],
    block_size: int,
    chunk_blocks: int,
    max_q_per_cta: int,
) -> None:
    for start in range(0, len(blocks), chunk_blocks):
        chunk = blocks[start : start + chunk_blocks]
        _append_segment_boxes(
            boxes,
            q_ids,
            chunk,
            len(chunk) * block_size,
            rank_by_req,
            max_q_per_cta,
        )


def _emit_forest_segments(
    node: _PrefixTrieNode,
    boxes: list[_ForkSegmentBox],
    rank_by_req: list[int],
    block_size: int,
    chunk_blocks: int,
    max_q_per_cta: int,
) -> None:
    for block, child in sorted(node.children.items()):
        blocks = [block]
        cur = child
        while len(cur.children) == 1 and not cur.terminal_reqs:
            next_block, next_child = next(iter(cur.children.items()))
            blocks.append(next_block)
            cur = next_child

        q_ids = sorted(cur.req_ids)
        _append_complete_block_segments(
            boxes,
            q_ids,
            blocks,
            rank_by_req,
            block_size,
            chunk_blocks,
            max_q_per_cta,
        )
        _emit_forest_segments(
            cur,
            boxes,
            rank_by_req,
            block_size,
            chunk_blocks,
            max_q_per_cta,
        )


def _pack_fork_segment_boxes(
    boxes: list[_ForkSegmentBox],
    num_split_per_seq: list[int],
    *,
    hratio: int,
    device: torch.device,
    page_block_size: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[int],
    int,
]:
    grouped: defaultdict[tuple[int, int, int], list[_ForkSegmentBox]]
    grouped = defaultdict(list)
    for box in boxes:
        grouped[
            _get_mnw(
                len(box.q_ids),
                hratio,
                box.kv_len,
                page_block_size,
            )
        ].append(box)

    q_tables: list[torch.Tensor] = []
    block_tables: list[torch.Tensor] = []
    num_seqs_per_ctas: list[torch.Tensor] = []
    cta_ranks: list[torch.Tensor] = []
    kv_in_ctas: list[torch.Tensor] = []
    mnw: list[int] = []

    for tile, group in sorted(grouped.items(), reverse=True):
        max_q = max(len(box.q_ids) for box in group)
        max_blocks = max(len(box.blocks) for box in group)
        q_table = []
        block_table = []
        num_seqs = []
        ranks = []
        kv_lens = []
        for box in group:
            q_table.append(box.q_ids + [0] * (max_q - len(box.q_ids)))
            block_table.append(box.blocks + [0] * (max_blocks - len(box.blocks)))
            num_seqs.append(len(box.q_ids))
            ranks.append(box.rank)
            kv_lens.append(box.kv_len)

        q_tables.append(torch.tensor(q_table, dtype=torch.int32, device=device))
        block_tables.append(torch.tensor(block_table, dtype=torch.int32, device=device))
        num_seqs_per_ctas.append(
            torch.tensor(num_seqs, dtype=torch.int32, device=device)
        )
        cta_ranks.append(torch.tensor(ranks, dtype=torch.int32, device=device))
        kv_in_ctas.append(torch.tensor(kv_lens, dtype=torch.int32, device=device))
        mnw.extend(tile)

    max_split_per_seq = max(num_split_per_seq)
    if len(q_tables) > 1:
        max_split_per_seq = max(max_split_per_seq, 2)

    return (
        torch.tensor(num_split_per_seq, dtype=torch.int32, device=device),
        q_tables,
        block_tables,
        num_seqs_per_ctas,
        cta_ranks,
        kv_in_ctas,
        mnw,
        max_split_per_seq,
    )


class ForkAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    supports_update_block_table: bool = False

    def _get_fork_num_sms(self) -> int:
        num_sms = getattr(self, "_fork_num_sms", None)
        if num_sms is None:
            try:
                num_sms = current_platform.num_compute_units()
            except NotImplementedError:
                num_sms = 1
            num_sms = max(1, int(num_sms))
            self._fork_num_sms = num_sms
        return num_sms

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> ForkAttentionMetadata:
        base_metadata = FlashAttentionMetadataBuilder.build(
            self,
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        prefix_chunk_bucket = self._get_cudagraph_prefix_chunk_bucket()
        forest_cta_bucket = self._get_cudagraph_forest_cta_bucket()
        if prefix_chunk_bucket is not None:
            prefix_workspace = self._get_cudagraph_workspace(prefix_chunk_bucket)
            self._clear_cudagraph_workspace(prefix_workspace)
            return ForkAttentionMetadata(
                **_flash_metadata_kwargs(base_metadata),
                **self._workspace_kwargs(prefix_workspace),
            )
        if forest_cta_bucket is None:
            return ForkAttentionMetadata(**_flash_metadata_kwargs(base_metadata))
        forest_workspace = self._get_cudagraph_forest_workspace(forest_cta_bucket)
        self._clear_cudagraph_workspace(forest_workspace)
        return ForkAttentionMetadata(
            **_flash_metadata_kwargs(base_metadata),
            **self._forest_workspace_kwargs(forest_workspace),
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ForkAttentionMetadata:
        base_metadata = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        fork_kwargs = self._build_fork_kwargs(
            base_metadata,
            common_attn_metadata.num_active_reqs(),
        )
        return ForkAttentionMetadata(
            **_flash_metadata_kwargs(base_metadata),
            **fork_kwargs,
        )

    def _build_fork_kwargs(
        self,
        metadata: FlashAttentionMetadata,
        num_active_reqs: int | None = None,
    ) -> dict[str, Any]:
        self._fork_last_execution_stats = ("base", 0, 0, 0, 0)
        if num_active_reqs is None:
            num_active_reqs = metadata.num_actual_tokens
        cudagraph_bucket = self._get_cudagraph_prefix_chunk_bucket()
        if cudagraph_bucket is not None:
            prefix_workspace = self._get_cudagraph_workspace(cudagraph_bucket)
            if not self._can_use_fork(metadata):
                self._clear_cudagraph_workspace(prefix_workspace)
                raise RuntimeError(
                    "ForkAttention prefix graph was selected without an "
                    "eligible common prefix"
                )
            profile_metadata = _fork_profile_enabled(getattr(self, "vllm_config", None))
            started_at = time.perf_counter() if profile_metadata else None
            kwargs = self._update_cudagraph_workspace(
                metadata,
                prefix_workspace,
                num_active_reqs,
            )
            metadata_ms = (
                (time.perf_counter() - started_at) * 1000
                if started_at is not None
                else None
            )
            if not kwargs:
                raise RuntimeError(
                    "ForkAttention prefix graph workspace does not match "
                    "the selected graph bucket"
                )
            self._profile_fork_metadata(
                metadata,
                enabled=bool(kwargs.get("fork_enabled", False)),
                reason="cudagraph",
                metadata_ms=metadata_ms,
            )
            return kwargs

        forest_cta_bucket = self._get_cudagraph_forest_cta_bucket()
        if forest_cta_bucket is not None:
            forest_workspace = self._get_cudagraph_forest_workspace(forest_cta_bucket)
            if not self._can_use_fork_decode(metadata):
                self._clear_cudagraph_workspace(forest_workspace)
                raise RuntimeError(
                    "ForkAttention forest graph was selected without an "
                    "eligible decode batch"
                )
            profile_metadata = _fork_profile_enabled(getattr(self, "vllm_config", None))
            started_at = time.perf_counter() if profile_metadata else None
            kwargs = self._update_cudagraph_forest_workspace(
                metadata,
                forest_workspace,
                num_active_reqs,
            )
            metadata_ms = (
                (time.perf_counter() - started_at) * 1000
                if started_at is not None
                else None
            )
            if not kwargs:
                mismatch = getattr(
                    self,
                    "_fork_forest_graph_mismatch_reason",
                    "unknown mismatch",
                )
                raise RuntimeError(
                    "ForkAttention forest graph workspace does not match "
                    f"the selected graph bucket: {mismatch}"
                )
            self._profile_fork_metadata(
                metadata,
                enabled=bool(kwargs.get("fork_enabled", False)),
                reason="cudagraph_forest",
                metadata_ms=metadata_ms,
            )
            return kwargs

        fallback_reason = self._fork_decode_fallback_reason(metadata)
        if fallback_reason is not None:
            self._profile_fork_metadata(
                metadata,
                enabled=False,
                reason=f"fallback_{fallback_reason}",
            )
            return {}

        profile_metadata = _fork_profile_enabled(getattr(self, "vllm_config", None))
        started_at = time.perf_counter() if profile_metadata else None
        forest_kwargs = self._build_fork_forest_kwargs(metadata, num_active_reqs)
        metadata_ms = (
            (time.perf_counter() - started_at) * 1000
            if started_at is not None
            else None
        )
        if forest_kwargs:
            self._profile_fork_metadata(
                metadata,
                enabled=True,
                reason="eager_forest",
                metadata_ms=metadata_ms,
            )
            return forest_kwargs

        self._profile_fork_metadata(
            metadata,
            enabled=False,
            reason=f"fallback_{self._fork_forest_fallback_reason or 'forest'}",
        )
        return {}

    def _build_fork_forest_kwargs(
        self,
        metadata: FlashAttentionMetadata,
        num_reqs: int,
    ) -> dict[str, Any]:
        forest = self._build_fork_forest_boxes(
            metadata,
            num_reqs,
            require_shared=True,
        )
        if forest is None:
            return {}
        boxes, num_split_per_seq = forest
        self._record_fork_execution_stats("eager", 0, boxes)
        hratio = self.num_heads_q // self.num_heads_kv
        device = metadata.block_table.device

        (
            fork_num_split_per_seq,
            fork_query_tables,
            fork_block_tables,
            fork_num_seqs_per_ctas,
            fork_cta_ranks,
            fork_kv_in_ctas,
            fork_mnw,
            fork_max_split_per_seq,
        ) = _pack_fork_segment_boxes(
            boxes,
            num_split_per_seq,
            hratio=hratio,
            device=device,
            page_block_size=self.block_size,
        )

        softmax_lse = torch.empty(
            (num_reqs, self.num_heads_q, 1),
            dtype=torch.float32,
            device=device,
        )
        split_out = torch.empty(
            (num_reqs, self.num_heads_q, fork_max_split_per_seq, self.headdim),
            dtype=torch.float32,
            device=device,
        )
        split_lse = torch.empty(
            (num_reqs, self.num_heads_q, fork_max_split_per_seq),
            dtype=torch.float32,
            device=device,
        )

        return {
            "fork_enabled": True,
            "fork_num_split_per_seq": fork_num_split_per_seq,
            "fork_query_tables": fork_query_tables,
            "fork_block_tables": fork_block_tables,
            "fork_num_seqs_per_ctas": fork_num_seqs_per_ctas,
            "fork_cta_ranks": fork_cta_ranks,
            "fork_kv_in_ctas": fork_kv_in_ctas,
            "fork_mnw": fork_mnw,
            "fork_max_split_per_seq": fork_max_split_per_seq,
            "fork_softmax_lse": softmax_lse,
            "fork_split_out": split_out,
            "fork_split_lse": split_lse,
        }

    def _record_fork_execution_stats(
        self,
        kind: str,
        capacity: int,
        boxes: list[_ForkSegmentBox],
    ) -> None:
        shared_ctas = sum(len(box.q_ids) > 1 for box in boxes)
        singleton_ctas = len(boxes) - shared_ctas
        self._fork_last_execution_stats = (
            kind,
            capacity,
            len(boxes),
            shared_ctas,
            singleton_ctas,
        )

    def _build_fork_forest_boxes(
        self,
        metadata: FlashAttentionMetadata,
        num_reqs: int,
        *,
        require_shared: bool,
    ) -> tuple[list[_ForkSegmentBox], list[int]] | None:
        self._fork_forest_fallback_reason = None
        if num_reqs <= 1:
            self._fork_forest_fallback_reason = "single_request"
            return None
        if metadata.block_table.shape[0] < num_reqs:
            self._fork_forest_fallback_reason = "block_table_rows"
            return None

        hratio = self.num_heads_q // self.num_heads_kv
        max_q_per_cta = max(1, 32 // hratio)
        seq_lens_source = getattr(self, "_fork_seq_lens_cpu", None)
        if seq_lens_source is None:
            seq_lens_cpu = [
                int(seq_len)
                for seq_len in metadata.seq_lens[:num_reqs].detach().cpu().tolist()
            ]
        else:
            seq_lens_cpu = [int(seq_len) for seq_len in seq_lens_source[:num_reqs]]
        if any(seq_len <= 0 for seq_len in seq_lens_cpu):
            self._fork_forest_fallback_reason = "sequence_length"
            return None

        block_counts = [
            (seq_len + self.block_size - 1) // self.block_size
            for seq_len in seq_lens_cpu
        ]
        max_blocks = max(block_counts)
        if metadata.block_table.shape[1] < max_blocks:
            self._fork_forest_fallback_reason = "block_table_columns"
            return None

        partial_segments: list[tuple[int, int, int]] = []
        max_complete_blocks = 0
        block_table_source = getattr(self, "_fork_block_table_cpu", None)
        if block_table_source is None:
            block_rows = (
                metadata.block_table[:num_reqs, :max_blocks].detach().cpu().tolist()
            )
        else:
            block_rows = block_table_source[:num_reqs, :max_blocks].tolist()

        for req_id, seq_len in enumerate(seq_lens_cpu):
            complete_blocks = seq_len // self.block_size
            partial_tokens = seq_len % self.block_size
            max_complete_blocks = max(max_complete_blocks, complete_blocks)
            if partial_tokens > 0:
                if block_table_source is not None:
                    partial_block = int(block_table_source[req_id, complete_blocks])
                else:
                    assert block_rows is not None
                    if complete_blocks >= len(block_rows[req_id]):
                        self._fork_forest_fallback_reason = "partial_block"
                        return None
                    partial_block = int(block_rows[req_id][complete_blocks])
                partial_segments.append((req_id, partial_block, partial_tokens))

        if max_complete_blocks <= 0 and not partial_segments:
            self._fork_forest_fallback_reason = "empty_kv"
            return None

        root = _PrefixTrieNode()
        for req_id, seq_len in enumerate(seq_lens_cpu):
            complete_blocks = seq_len // self.block_size
            if complete_blocks > 0:
                _add_trie_path(root, req_id, block_rows[req_id][:complete_blocks])
        base_chunk_blocks = _get_prefix_chunk_blocks(
            self.block_size,
            max(1, max_complete_blocks),
        )

        def build_boxes(chunk_blocks: int) -> tuple[list[_ForkSegmentBox], list[int]]:
            boxes: list[_ForkSegmentBox] = []
            rank_by_req = [0] * num_reqs
            _emit_forest_segments(
                root,
                boxes,
                rank_by_req,
                self.block_size,
                chunk_blocks,
                max_q_per_cta,
            )
            for req_id, block, partial_tokens in partial_segments:
                _append_segment_boxes(
                    boxes,
                    [req_id],
                    [block],
                    partial_tokens,
                    rank_by_req,
                    max_q_per_cta,
                )
            return boxes, rank_by_req

        boxes, rank_by_req = build_boxes(base_chunk_blocks)
        has_shared = any(len(box.q_ids) > 1 for box in boxes)
        if has_shared:
            target_waves = envs.VLLM_FORK_ATTN_TARGET_CTA_WAVES
            target_ctas = target_waves * self._get_fork_num_sms()
            target_plan_ctas = (
                target_ctas + self.num_heads_kv - 1
            ) // self.num_heads_kv
            missing_ctas = max(0, target_plan_ctas - len(boxes))
            base_chunks = (
                max_complete_blocks + base_chunk_blocks - 1
            ) // base_chunk_blocks
            target_chunks = min(31, base_chunks + missing_ctas)
            adaptive_chunk_blocks = _get_adaptive_prefix_chunk_blocks(
                block_size=self.block_size,
                prefix_blocks=max_complete_blocks,
                base_chunk_blocks=base_chunk_blocks,
                num_reqs=num_reqs,
                num_kv_heads=self.num_heads_kv,
                num_sms=self._get_fork_num_sms(),
                prefix_cohorts=1,
                max_prefix_chunks=target_chunks,
            )
            if adaptive_chunk_blocks < base_chunk_blocks:
                adaptive_boxes, adaptive_ranks = build_boxes(adaptive_chunk_blocks)
                if max(adaptive_ranks) <= 32:
                    boxes, rank_by_req = adaptive_boxes, adaptive_ranks

        if require_shared and not any(len(box.q_ids) > 1 for box in boxes):
            self._fork_forest_fallback_reason = "no_shared_kv"
            return None
        num_split_per_seq = rank_by_req
        if any(split <= 0 for split in num_split_per_seq):
            self._fork_forest_fallback_reason = "empty_split"
            return None
        return boxes, num_split_per_seq

    def _fork_decode_fallback_reason(
        self, metadata: FlashAttentionMetadata
    ) -> str | None:
        if envs.VLLM_BATCH_INVARIANT:
            return "batch_invariant"
        if metadata.max_query_len != 1:
            return "non_decode"
        if metadata.num_actual_tokens > metadata.seq_lens.shape[0]:
            return "token_count"
        if metadata.causal is not True:
            return "non_causal"
        if metadata.mm_prefix_range_tensor is not None:
            return "multimodal_prefix"
        if metadata.rswa_prefix_lens is not None:
            return "restricted_window"
        if self.headdim not in (64, 128):
            return "head_dimension"
        if self.block_size % 16 != 0:
            return "block_size"
        if not _is_supported_fork_kv_cache_dtype(self.kv_cache_dtype):
            return "kv_dtype"
        if self.num_heads_q % self.num_heads_kv != 0:
            return "head_ratio"
        return None

    def _can_use_fork_decode(self, metadata: FlashAttentionMetadata) -> bool:
        return self._fork_decode_fallback_reason(metadata) is None

    def _can_use_fork(self, metadata: FlashAttentionMetadata) -> bool:
        return (
            self._can_use_fork_decode(metadata)
            and metadata.use_cascade
            and metadata.common_prefix_len > 0
        )

    def _profile_fork_metadata(
        self,
        metadata: FlashAttentionMetadata,
        *,
        enabled: bool,
        reason: str,
        prefix_chunks: int | None = None,
        suffix_blocks: int | None = None,
        metadata_ms: float | None = None,
    ) -> None:
        if not _fork_profile_enabled(getattr(self, "vllm_config", None)):
            return
        counters = getattr(self, "_fork_profile_counters", None)
        if counters is None:
            counters = Counter()
            self._fork_profile_counters = counters
        key = f"{reason}:{'enabled' if enabled else 'fallback'}"
        first_for_path = counters[key] == 0
        counters[key] += 1
        timings: defaultdict[str, float] | None = getattr(
            self, "_fork_profile_metadata_ms", None
        )
        if timings is None:
            timings = defaultdict(float)
            self._fork_profile_metadata_ms = timings
        if metadata_ms is not None:
            timings[key] += metadata_ms
        total = sum(counters.values())
        if first_for_path or total <= 16 or total % 128 == 0:
            logger.info(
                "ForkAttention profile: total=%d path=%s num_reqs=%d "
                "common_prefix_len=%d use_cascade=%s prefix_chunks=%s "
                "suffix_blocks=%s metadata_ms=%.3f avg_metadata_ms=%.3f counters=%s",
                total,
                key,
                metadata.num_actual_tokens,
                metadata.common_prefix_len,
                metadata.use_cascade,
                prefix_chunks,
                suffix_blocks,
                metadata_ms or 0.0,
                timings[key] / counters[key],
                dict(counters),
            )

    def _get_cudagraph_prefix_chunk_bucket(self) -> int | None:
        plan = getattr(self, "_fork_cudagraph_plan", None)
        if plan is None or getattr(plan, "kind", None) != "common":
            return None
        return int(plan.capacity)

    def _get_cudagraph_forest_cta_bucket(self) -> int | None:
        plan = getattr(self, "_fork_cudagraph_plan", None)
        if plan is None or getattr(plan, "kind", None) != "forest":
            return None
        return int(plan.capacity)

    def _get_cudagraph_workspace(
        self,
        max_prefix_chunks: int | None = None,
    ) -> _ForkCUDAGraphWorkspace:
        if max_prefix_chunks is None:
            max_prefix_chunks = _get_default_prefix_chunk_bucket(
                self.block_size,
                self.model_config.max_model_len,
            )
        workspaces = getattr(self, "_fork_cudagraph_workspaces", None)
        if workspaces is None:
            workspaces = {}
            self._fork_cudagraph_workspaces = workspaces
        workspace = workspaces.get(max_prefix_chunks)
        if workspace is not None:
            return workspace

        hratio = self.num_heads_q // self.num_heads_kv
        max_active_reqs = self.vllm_config.scheduler_config.max_num_seqs
        max_reqs = max(
            max_active_reqs,
            self.compilation_config.max_cudagraph_capture_size or 0,
        )
        max_blocks = get_block_table_num_blocks(
            self.model_config.max_model_len,
            self.block_size,
        )
        prefix_chunk_capacity_blocks = _get_prefix_chunk_blocks(
            self.block_size, max_blocks
        )
        prefix_queries_per_cta = max(1, 32 // hratio)
        prefix_cohorts = (max_active_reqs + prefix_queries_per_cta - 1) // (
            prefix_queries_per_cta
        )
        prefix_ctas = max_prefix_chunks * prefix_cohorts

        prefix_query_cohorts = torch.zeros(
            (prefix_cohorts, prefix_queries_per_cta),
            dtype=torch.int32,
            device=self.device,
        )
        for cohort_id in range(prefix_cohorts):
            start = cohort_id * prefix_queries_per_cta
            end = min(start + prefix_queries_per_cta, max_active_reqs)
            if start < end:
                prefix_query_cohorts[cohort_id, : end - start] = torch.arange(
                    start,
                    end,
                    dtype=torch.int32,
                    device=self.device,
                )
        prefix_query_table = prefix_query_cohorts.repeat(max_prefix_chunks, 1)
        suffix_query_table = torch.arange(
            max_active_reqs,
            dtype=torch.int32,
            device=self.device,
        ).view(max_active_reqs, 1)

        workspace = _ForkCUDAGraphWorkspace(
            max_reqs=max_reqs,
            max_active_reqs=max_active_reqs,
            max_blocks=max_blocks,
            prefix_chunk_capacity_blocks=prefix_chunk_capacity_blocks,
            max_prefix_chunks=max_prefix_chunks,
            prefix_queries_per_cta=prefix_queries_per_cta,
            prefix_cohorts=prefix_cohorts,
            prefix_ctas=prefix_ctas,
            query_tables=[prefix_query_table, suffix_query_table],
            block_tables=[
                torch.zeros(
                    (prefix_ctas, prefix_chunk_capacity_blocks),
                    dtype=torch.int32,
                    device=self.device,
                ),
                torch.zeros(
                    (max_active_reqs, max_blocks),
                    dtype=torch.int32,
                    device=self.device,
                ),
            ],
            num_seqs_per_ctas=[
                torch.zeros(prefix_ctas, dtype=torch.int32, device=self.device),
                torch.zeros(max_active_reqs, dtype=torch.int32, device=self.device),
            ],
            cta_ranks=[
                torch.arange(
                    max_prefix_chunks,
                    dtype=torch.int32,
                    device=self.device,
                ).repeat_interleave(prefix_cohorts),
                torch.zeros(max_active_reqs, dtype=torch.int32, device=self.device),
            ],
            kv_in_ctas=[
                torch.zeros(prefix_ctas, dtype=torch.int32, device=self.device),
                torch.zeros(max_active_reqs, dtype=torch.int32, device=self.device),
            ],
            num_split_per_seq=torch.zeros(
                max_reqs, dtype=torch.int32, device=self.device
            ),
            softmax_lse=torch.empty(
                (max_reqs, self.num_heads_q, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            split_out=torch.empty(
                (
                    max_reqs,
                    self.num_heads_q,
                    max_prefix_chunks + 1,
                    self.headdim,
                ),
                dtype=torch.float32,
                device=self.device,
            ),
            split_lse=torch.empty(
                (max_reqs, self.num_heads_q, max_prefix_chunks + 1),
                dtype=torch.float32,
                device=self.device,
            ),
            mnw=[
                32,
                128,
                2,
                *_get_mnw(1, hratio, self.model_config.max_model_len, self.block_size),
            ],
        )
        workspaces[max_prefix_chunks] = workspace
        return workspace

    def _get_cudagraph_forest_workspace(
        self,
        max_ctas: int,
    ) -> _ForkForestCUDAGraphWorkspace:
        workspaces = getattr(self, "_fork_cudagraph_forest_workspaces", None)
        if workspaces is None:
            workspaces = {}
            self._fork_cudagraph_forest_workspaces = workspaces
        workspace = workspaces.get(max_ctas)
        if workspace is not None:
            return workspace

        hratio = self.num_heads_q // self.num_heads_kv
        max_active_reqs = self.vllm_config.scheduler_config.max_num_seqs
        max_reqs = max(
            max_active_reqs,
            self.compilation_config.max_cudagraph_capture_size or 0,
        )
        max_blocks = get_block_table_num_blocks(
            self.model_config.max_model_len,
            self.block_size,
        )
        chunk_blocks = _get_prefix_chunk_blocks(self.block_size, max_blocks)
        max_split_per_seq = _get_default_forest_max_split_per_seq(
            self.block_size,
            self.model_config.max_model_len,
        )
        graph_tiles = [
            _get_mnw(
                tile_m // hratio,
                hratio,
                self.model_config.max_model_len,
                self.block_size,
            )
            for tile_m in (32, 16)
        ]
        max_q_per_cta_by_group = [
            max(1, tile_m // hratio) for tile_m, _, _ in graph_tiles
        ]

        workspace = _ForkForestCUDAGraphWorkspace(
            max_reqs=max_reqs,
            max_active_reqs=max_active_reqs,
            max_blocks=max_blocks,
            max_ctas=max_ctas,
            chunk_blocks=chunk_blocks,
            max_split_per_seq=max_split_per_seq,
            max_q_per_cta_by_group=max_q_per_cta_by_group,
            query_tables=[
                torch.zeros(
                    (max_ctas, max_q_per_cta),
                    dtype=torch.int32,
                    device=self.device,
                )
                for max_q_per_cta in max_q_per_cta_by_group
            ],
            block_tables=[
                torch.zeros(
                    (max_ctas, chunk_blocks),
                    dtype=torch.int32,
                    device=self.device,
                )
                for _ in graph_tiles
            ],
            num_seqs_per_ctas=[
                torch.zeros(max_ctas, dtype=torch.int32, device=self.device)
                for _ in graph_tiles
            ],
            cta_ranks=[
                torch.zeros(max_ctas, dtype=torch.int32, device=self.device)
                for _ in graph_tiles
            ],
            kv_in_ctas=[
                torch.zeros(max_ctas, dtype=torch.int32, device=self.device)
                for _ in graph_tiles
            ],
            query_tables_cpu=[
                torch.empty(
                    (max_ctas, max_q_per_cta),
                    dtype=torch.int32,
                    pin_memory=True,
                )
                for max_q_per_cta in max_q_per_cta_by_group
            ],
            block_tables_cpu=[
                torch.empty(
                    (max_ctas, chunk_blocks),
                    dtype=torch.int32,
                    pin_memory=True,
                )
                for _ in graph_tiles
            ],
            num_seqs_per_ctas_cpu=[
                torch.empty(max_ctas, dtype=torch.int32, pin_memory=True)
                for _ in graph_tiles
            ],
            cta_ranks_cpu=[
                torch.empty(max_ctas, dtype=torch.int32, pin_memory=True)
                for _ in graph_tiles
            ],
            kv_in_ctas_cpu=[
                torch.empty(max_ctas, dtype=torch.int32, pin_memory=True)
                for _ in graph_tiles
            ],
            num_split_per_seq_cpu=torch.empty(
                max_reqs,
                dtype=torch.int32,
                pin_memory=True,
            ),
            num_split_per_seq=torch.zeros(
                max_reqs, dtype=torch.int32, device=self.device
            ),
            softmax_lse=torch.empty(
                (max_reqs, self.num_heads_q, 1),
                dtype=torch.float32,
                device=self.device,
            ),
            split_out=torch.empty(
                (
                    max_reqs,
                    self.num_heads_q,
                    max_split_per_seq,
                    self.headdim,
                ),
                dtype=torch.float32,
                device=self.device,
            ),
            split_lse=torch.empty(
                (max_reqs, self.num_heads_q, max_split_per_seq),
                dtype=torch.float32,
                device=self.device,
            ),
            mnw=[value for tile in graph_tiles for value in tile],
        )
        workspaces[max_ctas] = workspace
        return workspace

    def _clear_cudagraph_workspace(
        self,
        workspace: _ForkCUDAGraphWorkspace | _ForkForestCUDAGraphWorkspace,
    ) -> None:
        workspace.num_split_per_seq.zero_()
        for tensor in workspace.num_seqs_per_ctas:
            tensor.zero_()

    def _workspace_kwargs(
        self,
        workspace: _ForkCUDAGraphWorkspace,
    ) -> dict[str, Any]:
        return {
            "fork_enabled": True,
            "fork_num_split_per_seq": workspace.num_split_per_seq,
            "fork_query_tables": workspace.query_tables,
            "fork_block_tables": workspace.block_tables,
            "fork_num_seqs_per_ctas": workspace.num_seqs_per_ctas,
            "fork_cta_ranks": workspace.cta_ranks,
            "fork_kv_in_ctas": workspace.kv_in_ctas,
            "fork_mnw": workspace.mnw,
            "fork_max_split_per_seq": workspace.max_prefix_chunks + 1,
            "fork_softmax_lse": workspace.softmax_lse,
            "fork_split_out": workspace.split_out,
            "fork_split_lse": workspace.split_lse,
        }

    def _forest_workspace_kwargs(
        self,
        workspace: _ForkForestCUDAGraphWorkspace,
    ) -> dict[str, Any]:
        return {
            "fork_enabled": True,
            "fork_num_split_per_seq": workspace.num_split_per_seq,
            "fork_query_tables": workspace.query_tables,
            "fork_block_tables": workspace.block_tables,
            "fork_num_seqs_per_ctas": workspace.num_seqs_per_ctas,
            "fork_cta_ranks": workspace.cta_ranks,
            "fork_kv_in_ctas": workspace.kv_in_ctas,
            "fork_mnw": workspace.mnw,
            "fork_max_split_per_seq": workspace.max_split_per_seq,
            "fork_softmax_lse": workspace.softmax_lse,
            "fork_split_out": workspace.split_out,
            "fork_split_lse": workspace.split_lse,
        }

    def _update_cudagraph_workspace(
        self,
        metadata: FlashAttentionMetadata,
        workspace: _ForkCUDAGraphWorkspace,
        num_active_reqs: int | None = None,
    ) -> dict[str, Any]:
        num_reqs = (
            metadata.num_actual_tokens if num_active_reqs is None else num_active_reqs
        )
        if (
            num_reqs > workspace.max_active_reqs
            or metadata.block_table.shape[1] > workspace.max_blocks
        ):
            return {}

        self._clear_cudagraph_workspace(workspace)

        prefix_len = max(metadata.common_prefix_len, 0)
        prefix_blocks = min(prefix_len // self.block_size, workspace.max_blocks)
        total_blocks = min(metadata.block_table.shape[1], workspace.max_blocks)
        has_prefix = prefix_blocks > 0 and metadata.use_cascade
        if not has_prefix:
            prefix_len = 0
            prefix_blocks = 0
        prefix_cohorts = (
            (num_reqs + workspace.prefix_queries_per_cta - 1)
            // workspace.prefix_queries_per_cta
            if has_prefix
            else 0
        )
        prefix_chunk_blocks = workspace.prefix_chunk_capacity_blocks
        if has_prefix:
            prefix_chunk_blocks = _get_adaptive_prefix_chunk_blocks(
                block_size=self.block_size,
                prefix_blocks=prefix_blocks,
                base_chunk_blocks=prefix_chunk_blocks,
                num_reqs=num_reqs,
                num_kv_heads=self.num_heads_kv,
                num_sms=self._get_fork_num_sms(),
                prefix_cohorts=prefix_cohorts,
                max_prefix_chunks=workspace.max_prefix_chunks,
            )
        num_prefix_chunks = (
            (prefix_blocks + prefix_chunk_blocks - 1) // prefix_chunk_blocks
            if has_prefix
            else 0
        )
        if num_prefix_chunks > workspace.max_prefix_chunks:
            return {}
        suffix_start = prefix_blocks
        suffix_blocks = max(0, total_blocks - suffix_start)

        if has_prefix:
            for chunk_id in range(num_prefix_chunks):
                block_start = chunk_id * prefix_chunk_blocks
                block_count = min(
                    prefix_chunk_blocks,
                    prefix_blocks - block_start,
                )
                cta_start = chunk_id * workspace.prefix_cohorts
                cta_end = cta_start + prefix_cohorts
                workspace.block_tables[0][cta_start:cta_end, :block_count].copy_(
                    metadata.block_table[
                        :1, block_start : block_start + block_count
                    ].expand(prefix_cohorts, block_count)
                )
                workspace.kv_in_ctas[0][cta_start:cta_end].fill_(
                    min(
                        prefix_chunk_blocks * self.block_size,
                        prefix_len - block_start * self.block_size,
                    )
                )
            for cohort_id in range(prefix_cohorts):
                start = cohort_id * workspace.prefix_queries_per_cta
                count = min(workspace.prefix_queries_per_cta, num_reqs - start)
                workspace.num_seqs_per_ctas[0][
                    cohort_id : num_prefix_chunks
                    * workspace.prefix_cohorts : workspace.prefix_cohorts
                ] = count
        workspace.cta_ranks[1][:num_reqs].fill_(num_prefix_chunks)

        if suffix_blocks > 0:
            workspace.block_tables[1][:num_reqs, :suffix_blocks].copy_(
                metadata.block_table[
                    :num_reqs, suffix_start : suffix_start + suffix_blocks
                ]
            )
        torch.sub(
            metadata.seq_lens[:num_reqs],
            prefix_len,
            out=workspace.kv_in_ctas[1][:num_reqs],
        )
        workspace.kv_in_ctas[1][:num_reqs].clamp_(min=0)
        torch.sign(
            workspace.kv_in_ctas[1][:num_reqs],
            out=workspace.num_seqs_per_ctas[1][:num_reqs],
        )
        workspace.num_split_per_seq[:num_reqs].copy_(
            workspace.num_seqs_per_ctas[1][:num_reqs]
        )
        workspace.num_split_per_seq[:num_reqs].mul_(num_prefix_chunks + 1)
        shared_ctas = num_prefix_chunks * (prefix_cohorts if has_prefix else 0)
        singleton_ctas = num_reqs if suffix_blocks > 0 else 0
        self._fork_last_execution_stats = (
            "common",
            workspace.max_prefix_chunks,
            shared_ctas + singleton_ctas,
            shared_ctas,
            singleton_ctas,
        )
        return self._workspace_kwargs(workspace)

    def _update_cudagraph_forest_workspace(
        self,
        metadata: FlashAttentionMetadata,
        workspace: _ForkForestCUDAGraphWorkspace,
        num_active_reqs: int | None = None,
    ) -> dict[str, Any]:
        num_reqs = (
            metadata.num_actual_tokens if num_active_reqs is None else num_active_reqs
        )
        self._fork_forest_graph_mismatch_reason = None
        if (
            num_reqs > workspace.max_active_reqs
            or metadata.block_table.shape[1] > workspace.max_blocks
        ):
            self._fork_forest_graph_mismatch_reason = (
                f"requests={num_reqs}/{workspace.max_active_reqs}, "
                f"block_columns={metadata.block_table.shape[1]}/"
                f"{workspace.max_blocks}"
            )
            return {}

        forest = self._build_fork_forest_boxes(
            metadata,
            num_reqs,
            require_shared=False,
        )
        if forest is None:
            self._fork_forest_graph_mismatch_reason = "forest planner returned no boxes"
            self._clear_cudagraph_workspace(workspace)
            return {}
        boxes, num_split_per_seq = forest
        if (
            len(boxes) > workspace.max_ctas
            or max(num_split_per_seq) > workspace.max_split_per_seq
        ):
            self._fork_forest_graph_mismatch_reason = (
                f"ctas={len(boxes)}/{workspace.max_ctas}, "
                f"splits={max(num_split_per_seq)}/"
                f"{workspace.max_split_per_seq}"
            )
            self._clear_cudagraph_workspace(workspace)
            return {}

        self._record_fork_execution_stats("forest", workspace.max_ctas, boxes)

        self._clear_cudagraph_workspace(workspace)
        hratio = self.num_heads_q // self.num_heads_kv
        grouped_boxes: list[list[_ForkSegmentBox]] = [
            [],
            [],
        ]
        for box in boxes:
            tile_m, _, _ = _get_mnw(
                len(box.q_ids),
                hratio,
                box.kv_len,
                self.block_size,
            )
            group_idx = 0 if tile_m > 16 else 1
            max_q_per_cta = workspace.max_q_per_cta_by_group[group_idx]
            if (
                len(box.q_ids) > max_q_per_cta
                or len(box.blocks) > workspace.chunk_blocks
            ):
                self._fork_forest_graph_mismatch_reason = (
                    f"queries_per_cta={len(box.q_ids)}/{max_q_per_cta}, "
                    f"blocks_per_cta={len(box.blocks)}/"
                    f"{workspace.chunk_blocks}"
                )
                self._clear_cudagraph_workspace(workspace)
                return {}
            grouped_boxes[group_idx].append(box)

        for group_idx, group in enumerate(grouped_boxes):
            num_ctas = len(group)
            if num_ctas > workspace.max_ctas:
                self._fork_forest_graph_mismatch_reason = (
                    f"group_{group_idx}_ctas={num_ctas}/{workspace.max_ctas}"
                )
                self._clear_cudagraph_workspace(workspace)
                return {}
            if num_ctas == 0:
                continue

            query_cpu = workspace.query_tables_cpu[group_idx]
            block_cpu = workspace.block_tables_cpu[group_idx]
            num_seqs_cpu = workspace.num_seqs_per_ctas_cpu[group_idx]
            cta_ranks_cpu = workspace.cta_ranks_cpu[group_idx]
            kv_in_ctas_cpu = workspace.kv_in_ctas_cpu[group_idx]
            query_cpu[:num_ctas].zero_()
            block_cpu[:num_ctas].zero_()
            query_np = query_cpu.numpy()
            block_np = block_cpu.numpy()
            num_seqs_np = num_seqs_cpu.numpy()
            cta_ranks_np = cta_ranks_cpu.numpy()
            kv_in_ctas_np = kv_in_ctas_cpu.numpy()

            for row, box in enumerate(group):
                query_np[row, : len(box.q_ids)] = box.q_ids
                block_np[row, : len(box.blocks)] = box.blocks
                num_seqs_np[row] = len(box.q_ids)
                cta_ranks_np[row] = box.rank
                kv_in_ctas_np[row] = box.kv_len

            workspace.query_tables[group_idx][:num_ctas].copy_(
                query_cpu[:num_ctas],
                non_blocking=True,
            )
            workspace.block_tables[group_idx][:num_ctas].copy_(
                block_cpu[:num_ctas],
                non_blocking=True,
            )
            workspace.num_seqs_per_ctas[group_idx][:num_ctas].copy_(
                num_seqs_cpu[:num_ctas],
                non_blocking=True,
            )
            workspace.cta_ranks[group_idx][:num_ctas].copy_(
                cta_ranks_cpu[:num_ctas],
                non_blocking=True,
            )
            workspace.kv_in_ctas[group_idx][:num_ctas].copy_(
                kv_in_ctas_cpu[:num_ctas],
                non_blocking=True,
            )
        num_split_cpu = workspace.num_split_per_seq_cpu
        num_split_cpu.numpy()[:num_reqs] = num_split_per_seq
        workspace.num_split_per_seq[:num_reqs].copy_(
            num_split_cpu[:num_reqs],
            non_blocking=True,
        )
        return self._forest_workspace_kwargs(workspace)


class ForkAttentionBackend(FlashAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "FORK_ATTN"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type["ForkAttentionImpl"]:
        return ForkAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["ForkAttentionMetadataBuilder"]:
        return ForkAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in (64, 128)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
        **kwargs,
    ) -> str | None:
        if use_mla:
            return "FORK attention does not support MLA"
        if use_sparse:
            return "FORK attention does not support sparse attention"
        if use_mm_prefix:
            return "FORK attention does not support mm_prefix"
        if has_sink:
            return "FORK attention does not support sinks"
        if device_capability < DeviceCapability(8, 0):
            return "FORK attention requires compute capability >= 8.0"
        if dtype not in cls.supported_dtypes:
            return "dtype not supported"
        if not cls.supports_head_size(head_size):
            return "head_size not supported"
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            return "kv_cache_dtype not supported"
        if not cls.supports_block_size(block_size):
            return "block_size not supported"
        return None


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
        if self._can_run_fork(attn_metadata, output_scale, output_block_scale):
            attn_metadata = cast(ForkAttentionMetadata, attn_metadata)
            assert attn_metadata.fork_num_split_per_seq is not None
            assert attn_metadata.fork_query_tables is not None
            assert attn_metadata.fork_block_tables is not None
            assert attn_metadata.fork_num_seqs_per_ctas is not None
            assert attn_metadata.fork_cta_ranks is not None
            assert attn_metadata.fork_kv_in_ctas is not None
            assert attn_metadata.fork_mnw is not None
            assert attn_metadata.fork_softmax_lse is not None
            assert attn_metadata.fork_split_out is not None
            assert attn_metadata.fork_split_lse is not None

            num_actual_tokens = attn_metadata.num_actual_tokens
            key_cache, value_cache = kv_cache.unbind(1)
            key_cache = canonicalize_singleton_dim_strides(key_cache)
            value_cache = canonicalize_singleton_dim_strides(value_cache)
            q = query[:num_actual_tokens].view(
                num_actual_tokens, 1, self.num_heads, self.head_size
            )
            out = output[:num_actual_tokens].view(
                num_actual_tokens, 1, self.num_heads, self.head_size
            )
            ops.fork_attention(
                out,
                attn_metadata.fork_softmax_lse,
                attn_metadata.fork_split_out,
                attn_metadata.fork_split_lse,
                q,
                key_cache,
                value_cache,
                attn_metadata.fork_num_split_per_seq,
                attn_metadata.fork_query_tables,
                attn_metadata.fork_block_tables,
                attn_metadata.fork_num_seqs_per_ctas,
                attn_metadata.fork_cta_ranks,
                attn_metadata.fork_kv_in_ctas,
                attn_metadata.fork_mnw,
                attn_metadata.fork_max_split_per_seq,
                self.scale,
            )
            return output

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

    def _can_run_fork(
        self,
        attn_metadata: FlashAttentionMetadata | None,
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
