# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, NamedTuple

from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    DEFAULT_FANOUT_LAYERWISE_LOAD_THRESHOLD_BYTES,
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
    fanout_profiling_enabled,
    resolve_fanout_layerwise_load,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _TransferMetricName,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    OffloadEvictionMetadata,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    make_offload_key,
)
from vllm.v1.kv_offload.fanout_planner import (
    FanoutBlock,
    FanoutChunkPlanner,
    FanoutLifecycleState,
    FanoutPressureLevel,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)

DEFAULT_FANOUT_CHUNK_TOKENS = 2048


def _resolve_fanout_chunk_blocks(
    extra_config: dict[str, Any],
    *,
    min_offloaded_block_size: int,
    fanout_budget_blocks: int,
) -> int:
    if "fanout_chunk_blocks" in extra_config:
        chunk_blocks = int(extra_config["fanout_chunk_blocks"])
        if chunk_blocks <= 0:
            raise ValueError("fanout_chunk_blocks must be positive")
        if 0 < fanout_budget_blocks < chunk_blocks:
            raise ValueError("fanout_chunk_blocks must be <= fanout_budget_blocks")
        return chunk_blocks

    chunk_tokens = int(
        extra_config.get("fanout_chunk_tokens", DEFAULT_FANOUT_CHUNK_TOKENS)
    )
    if chunk_tokens <= 0:
        raise ValueError("fanout_chunk_tokens must be positive")

    chunk_blocks = max(1, cdiv(chunk_tokens, min_offloaded_block_size))
    if fanout_budget_blocks > 0:
        chunk_blocks = min(chunk_blocks, fanout_budget_blocks)
    return chunk_blocks


def _resolve_fanout_hot_prefix_config(
    extra_config: dict[str, Any],
    *,
    fanout_min_fanout: int,
) -> tuple[int, float, int, int, int, bool]:
    hot_prefix_min_fanout = int(
        extra_config.get(
            "fanout_hot_prefix_min_fanout",
            max(4, fanout_min_fanout + 1),
        )
    )
    hot_prefix_max_position = float(
        extra_config.get("fanout_hot_prefix_max_position", 1.0)
    )
    hot_prefix_min_reuse_blocks = int(
        extra_config.get("fanout_hot_prefix_min_reuse_blocks", 128)
    )
    hot_prefix_min_residency_steps = int(
        extra_config.get("fanout_hot_prefix_min_residency_steps", 4)
    )
    hot_prefix_cooldown_steps = int(
        extra_config.get("fanout_hot_prefix_cooldown_steps", 16)
    )
    allow_hot_prefix_backup = bool(
        extra_config.get("fanout_allow_hot_prefix_backup", True)
    )
    if hot_prefix_min_fanout < 0:
        raise ValueError("fanout_hot_prefix_min_fanout must be non-negative")
    if hot_prefix_max_position <= 0 or hot_prefix_max_position > 1:
        raise ValueError("fanout_hot_prefix_max_position must be in (0, 1]")
    if hot_prefix_min_reuse_blocks < 0:
        raise ValueError("fanout_hot_prefix_min_reuse_blocks must be non-negative")
    if hot_prefix_min_residency_steps < 0:
        raise ValueError("fanout_hot_prefix_min_residency_steps must be non-negative")
    if hot_prefix_cooldown_steps < 0:
        raise ValueError("fanout_hot_prefix_cooldown_steps must be non-negative")
    return (
        hot_prefix_min_fanout,
        hot_prefix_max_position,
        hot_prefix_min_reuse_blocks,
        hot_prefix_min_residency_steps,
        hot_prefix_cooldown_steps,
        allow_hot_prefix_backup,
    )


def _resolve_fanout_pressure_config(
    extra_config: dict[str, Any],
) -> tuple[float, float, float, float]:
    high_threshold = float(extra_config.get("fanout_high_pressure_threshold", 0.90))
    critical_threshold = float(
        extra_config.get("fanout_critical_pressure_threshold", 0.97)
    )
    high_exit_threshold = float(
        extra_config.get(
            "fanout_high_pressure_exit_threshold",
            round(max(0.0, high_threshold - 0.05), 6),
        )
    )
    critical_exit_threshold = float(
        extra_config.get(
            "fanout_critical_pressure_exit_threshold",
            round(max(high_threshold, critical_threshold - 0.04), 6),
        )
    )
    if high_threshold < 0 or high_threshold > 1:
        raise ValueError("fanout_high_pressure_threshold must be in [0, 1]")
    if critical_threshold < 0 or critical_threshold > 1:
        raise ValueError("fanout_critical_pressure_threshold must be in [0, 1]")
    if high_threshold > critical_threshold:
        raise ValueError(
            "fanout_high_pressure_threshold must be <= "
            "fanout_critical_pressure_threshold"
        )
    if not 0 <= high_exit_threshold <= high_threshold:
        raise ValueError(
            "fanout_high_pressure_exit_threshold must be in [0, high threshold]"
        )
    if not high_threshold <= critical_exit_threshold <= critical_threshold:
        raise ValueError(
            "fanout_critical_pressure_exit_threshold must be between the high "
            "and critical thresholds"
        )
    return (
        high_threshold,
        critical_threshold,
        high_exit_threshold,
        critical_exit_threshold,
    )


def _is_hot_shared_prefix(
    *,
    fanout: int,
    prefix_position: float,
    min_fanout: int,
    max_prefix_position: float,
    reuse_score: int = 0,
    min_reuse_score: int = 0,
) -> bool:
    return (
        min_fanout > 0
        and fanout >= min_fanout
        and prefix_position <= max_prefix_position
        and reuse_score >= min_reuse_score
    )


def _make_fanout_eviction_metadata(
    candidate: FanoutBlock,
    observation: "FanoutCandidateObservation",
) -> OffloadEvictionMetadata:
    return OffloadEvictionMetadata(
        lifecycle_value=candidate.lifecycle_state.value,
        reuse_score=observation.reuse_score,
        fanout=max(candidate.fanout, candidate.historical_max_fanout),
        residency_value=candidate.residency_value,
        prefix_position=candidate.prefix_position,
    )


@dataclass(slots=True)
class FanoutLifecycle:
    historical_max_fanout: int = 1
    historical_max_reuse_score: int = 0
    recent_access_count: int = 0
    last_access_step: int = 0
    hot_since_step: int | None = None
    min_resident_until_step: int = 0
    last_hot_step: int = 0
    last_observed_step: int = 0


@dataclass(slots=True)
class FanoutCandidateObservation:
    request_id: str
    group_idx: int
    logical_block_idx: int
    physical_block_id: int
    offload_key: OffloadKey
    fanout: int
    prefix_position: float
    reuse_score: int
    is_active_tail: bool
    needs_backup: bool = True

    def merge(self, other: "FanoutCandidateObservation") -> None:
        self.fanout = max(self.fanout, other.fanout)
        self.prefix_position = min(self.prefix_position, other.prefix_position)
        self.reuse_score = max(self.reuse_score, other.reuse_score)
        self.is_active_tail = self.is_active_tail or other.is_active_tail
        self.needs_backup = self.needs_backup or other.needs_backup


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store src block IDs whose ref_cnt protects them while the request
    # runs. Only registered in _block_id_to_pending_jobs on request_finished.
    non_sliding_window_block_ids: list[int] | None = None
    # Store src block IDs that may be freed before the request finishes.
    # Registered in _block_id_to_pending_jobs at store creation time.
    sliding_window_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    gpu_block_size: int
    offloaded_block_size: int
    hash_block_size_factor: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_blocks: int | None
    # Number of this group's offloaded blocks per full-attention alignment
    # segment. Used to skip storing SWA blocks that can never serve a load
    # hit (e.g. DeepSeek V4 where SWA groups have much smaller block sizes
    # than the MLA full-attention group).
    # None for full-attention groups or when the optimization doesn't apply.
    alignment_block_count: int | None = None
    # True for EAGLE/MTP draft-model attention groups. The trailing block
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False


def get_sliding_window_size_in_blocks(
    kv_cache_spec: KVCacheSpec, offloaded_block_size: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return cdiv(kv_cache_spec.sliding_window, offloaded_block_size)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def resolve_mamba_align_size(spec: "OffloadingSpec") -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" cache mode the hit window must be rounded
    down to a multiple of the offloaded block size. Asserts that all such
    groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for idx, gpu_block_size in enumerate(spec.gpu_block_size):
        kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            offload_block_size = gpu_block_size * spec.block_size_factor
            assert mamba_align_size is None or mamba_align_size == offload_block_size
            mamba_align_size = offload_block_size
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    block_size_factor: int
    num_workers: int
    offload_prompt_only: bool
    fanout_offload: bool
    fanout_chunk_blocks: int
    fanout_budget_blocks: int
    fanout_min_fanout: int
    fanout_recent_tail_blocks: int
    fanout_hot_prefix_min_fanout: int
    fanout_hot_prefix_max_position: float
    fanout_hot_prefix_min_reuse_blocks: int
    fanout_hot_prefix_min_residency_steps: int
    fanout_hot_prefix_cooldown_steps: int
    fanout_allow_hot_prefix_backup: bool
    fanout_high_pressure_threshold: float
    fanout_critical_pressure_threshold: float
    fanout_high_pressure_exit_threshold: float
    fanout_critical_pressure_exit_threshold: float
    fanout_layerwise_load: bool
    fanout_profile: bool

    @classmethod
    def from_spec(cls, spec: OffloadingSpec) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the offloaded_block_size of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_offloaded_block_sizes: set[int] = set()
        for idx, gpu_block_size in enumerate(spec.gpu_block_size):
            kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec
            sw = get_sliding_window_size_in_blocks(
                kv_spec, gpu_block_size * spec.block_size_factor
            )
            if sw is None:
                full_attn_offloaded_block_sizes.add(
                    gpu_block_size * spec.block_size_factor
                )

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.
        alignment_tokens: int | None = None
        if len(full_attn_offloaded_block_sizes) == 1:
            alignment_tokens = full_attn_offloaded_block_sizes.pop()

        def _alignment_block_count(
            offloaded_block_size: int,
            sliding_window_size_in_blocks: int | None,
        ) -> int | None:
            if alignment_tokens is None or sliding_window_size_in_blocks is None:
                return None
            if alignment_tokens <= offloaded_block_size:
                return None
            per_segment = alignment_tokens // offloaded_block_size
            if sliding_window_size_in_blocks >= per_segment:
                return None
            return per_segment

        eagle_groups = {
            idx
            for idx, g in enumerate(spec.kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }

        use_eagle = (
            spec.vllm_config.speculative_config is not None
            and spec.vllm_config.speculative_config.use_eagle()
        )
        if use_eagle and not eagle_groups:
            eagle_groups = set(range(len(spec.kv_cache_config.kv_cache_groups)))

        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
                "detected. The trailing block of these groups will be "
                "excluded from offloading due to volatility.",
                sorted(eagle_groups),
            )

        backend = spec.vllm_config.attention_config.backend
        fanout_offload = bool(
            spec.extra_config.get(
                "fanout_offload",
                backend is not None and backend.name == "FORK_ATTN",
            )
        )
        fanout_budget_blocks = int(spec.extra_config.get("fanout_budget_blocks", 64))
        min_offloaded_block_size = min(
            gpu_block_size * spec.block_size_factor
            for gpu_block_size in spec.gpu_block_size
        )
        fanout_chunk_blocks = _resolve_fanout_chunk_blocks(
            spec.extra_config,
            min_offloaded_block_size=min_offloaded_block_size,
            fanout_budget_blocks=fanout_budget_blocks,
        )
        fanout_min_fanout = int(spec.extra_config.get("fanout_min_fanout", 1))
        fanout_recent_tail_blocks = int(
            spec.extra_config.get("fanout_recent_tail_blocks", 1)
        )
        (
            fanout_hot_prefix_min_fanout,
            fanout_hot_prefix_max_position,
            fanout_hot_prefix_min_reuse_blocks,
            fanout_hot_prefix_min_residency_steps,
            fanout_hot_prefix_cooldown_steps,
            fanout_allow_hot_prefix_backup,
        ) = _resolve_fanout_hot_prefix_config(
            spec.extra_config,
            fanout_min_fanout=fanout_min_fanout,
        )
        (
            fanout_high_pressure_threshold,
            fanout_critical_pressure_threshold,
            fanout_high_pressure_exit_threshold,
            fanout_critical_pressure_exit_threshold,
        ) = _resolve_fanout_pressure_config(spec.extra_config)
        fanout_profile = fanout_profiling_enabled(
            spec.extra_config,
            spec.vllm_config,
        )
        if fanout_budget_blocks < 0:
            raise ValueError("fanout_budget_blocks must be non-negative")
        if fanout_min_fanout <= 0:
            raise ValueError("fanout_min_fanout must be positive")
        if fanout_recent_tail_blocks < 0:
            raise ValueError("fanout_recent_tail_blocks must be non-negative")
        layerwise_threshold_bytes = int(
            spec.extra_config.get(
                "fanout_layerwise_load_threshold_bytes",
                DEFAULT_FANOUT_LAYERWISE_LOAD_THRESHOLD_BYTES,
            )
        )
        estimated_load_bytes = fanout_budget_blocks * int(
            getattr(spec, "kv_bytes_per_offloaded_block", 0) or 0
        )
        fanout_layerwise_load = resolve_fanout_layerwise_load(
            spec.extra_config.get("fanout_layerwise_load", "auto"),
            fanout_offload=fanout_offload,
            has_full_cudagraphs=(
                spec.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
            ),
            estimated_load_bytes=estimated_load_bytes,
            threshold_bytes=layerwise_threshold_bytes,
        )
        if fanout_layerwise_load and not fanout_offload:
            raise ValueError("fanout_layerwise_load requires fanout_offload")
        if fanout_offload and spec.block_size_factor != 1:
            raise ValueError(
                "Fanout chunk offloading currently requires offload block_size "
                "to match the GPU block size"
            )
        if fanout_offload:
            logger.info(
                "Fanout KV offload enabled: layerwise_load=%s, "
                "chunk_blocks=%d, budget_blocks=%d, min_fanout=%d, "
                "hot_prefix_min_fanout=%d, hot_prefix_max_position=%.3f, "
                "hot_prefix_min_reuse_blocks=%d, "
                "hot_prefix_min_residency_steps=%d, "
                "hot_prefix_cooldown_steps=%d, "
                "allow_hot_prefix_backup=%s, "
                "pressure_thresholds=(%.3f, %.3f), "
                "pressure_exit_thresholds=(%.3f, %.3f), "
                "estimated_load_bytes=%d, "
                "threshold_bytes=%d, profile=%s",
                fanout_layerwise_load,
                fanout_chunk_blocks,
                fanout_budget_blocks,
                fanout_min_fanout,
                fanout_hot_prefix_min_fanout,
                fanout_hot_prefix_max_position,
                fanout_hot_prefix_min_reuse_blocks,
                fanout_hot_prefix_min_residency_steps,
                fanout_hot_prefix_cooldown_steps,
                fanout_allow_hot_prefix_backup,
                fanout_high_pressure_threshold,
                fanout_critical_pressure_threshold,
                fanout_high_pressure_exit_threshold,
                fanout_critical_pressure_exit_threshold,
                estimated_load_bytes,
                layerwise_threshold_bytes,
                fanout_profile,
            )

        return cls(
            num_workers=spec.vllm_config.parallel_config.world_size,
            kv_group_configs=tuple(
                GroupOffloadConfig(
                    group_idx=idx,
                    gpu_block_size=gpu_block_size,
                    offloaded_block_size=gpu_block_size * spec.block_size_factor,
                    hash_block_size_factor=(
                        (gpu_block_size * spec.block_size_factor)
                        // spec.hash_block_size
                    ),
                    sliding_window_size_in_blocks=(
                        sw := get_sliding_window_size_in_blocks(
                            spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec,
                            gpu_block_size * spec.block_size_factor,
                        )
                    ),
                    alignment_block_count=_alignment_block_count(
                        gpu_block_size * spec.block_size_factor, sw
                    ),
                    kv_event_group_spec=get_offloading_event_group_spec(
                        spec.kv_cache_config.kv_cache_groups[idx]
                    ),
                    is_eagle_group=idx in eagle_groups,
                )
                for idx, gpu_block_size in enumerate(spec.gpu_block_size)
            ),
            block_size_factor=spec.block_size_factor,
            offload_prompt_only=spec.offload_prompt_only,
            fanout_offload=fanout_offload,
            fanout_chunk_blocks=fanout_chunk_blocks,
            fanout_budget_blocks=fanout_budget_blocks,
            fanout_min_fanout=fanout_min_fanout,
            fanout_recent_tail_blocks=fanout_recent_tail_blocks,
            fanout_hot_prefix_min_fanout=fanout_hot_prefix_min_fanout,
            fanout_hot_prefix_max_position=fanout_hot_prefix_max_position,
            fanout_hot_prefix_min_reuse_blocks=(fanout_hot_prefix_min_reuse_blocks),
            fanout_hot_prefix_min_residency_steps=(
                fanout_hot_prefix_min_residency_steps
            ),
            fanout_hot_prefix_cooldown_steps=fanout_hot_prefix_cooldown_steps,
            fanout_allow_hot_prefix_backup=fanout_allow_hot_prefix_backup,
            fanout_high_pressure_threshold=fanout_high_pressure_threshold,
            fanout_critical_pressure_threshold=fanout_critical_pressure_threshold,
            fanout_high_pressure_exit_threshold=(fanout_high_pressure_exit_threshold),
            fanout_critical_pressure_exit_threshold=(
                fanout_critical_pressure_exit_threshold
            ),
            fanout_layerwise_load=fanout_layerwise_load,
            fanout_profile=fanout_profile,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # index of next block (of size offloaded_block_size) to offload
    next_stored_block_idx: int = 0
    # number of offloaded blocks hit (including GPU prefix cache)
    # when the request first started
    num_hit_blocks: int = 0
    # Logical offload-block indices considered by the fanout admission policy.
    fanout_admitted_block_indices: set[int] = field(default_factory=set)


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hash_block_size_factor * len(group_state.offload_keys)
                + group_config.hash_block_size_factor
                - 1,
                None,
                group_config.hash_block_size_factor,
            ):
                group_state.offload_keys.append(
                    make_offload_key(req_block_hash, group_config.group_idx)
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        assert len(new_block_id_groups) == len(self.group_states)
        for group_state, new_blocks in zip(self.group_states, new_block_id_groups):
            group_state.block_ids.extend(new_blocks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
            group_state.next_stored_block_idx = num_blocks

    def update_num_hit_blocks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_blocks = (
                num_cached_tokens // group_config.offloaded_block_size
            )


def _create_req_context(req: Request) -> ReqContext:
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=req.kv_transfer_params,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        spec: OffloadingSpec,
    ):
        self.config = SchedulerOffloadConfig.from_spec(spec)
        self.manager: OffloadingManager = spec.get_manager()
        self._connector_stats: OffloadingConnectorStats | None = None

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for group_config in self.config.kv_group_configs:
            if group_config.sliding_window_size_in_blocks is None:
                full_attention_groups.append(group_config.group_idx)
            else:
                sliding_window_groups.append(group_config.group_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_blocks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = tuple(full_attention_groups) + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(spec)

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # track loaded blocks to avoid redundant loads
        self._blocks_being_loaded: set[OffloadKey] | None = (
            set() if spec.vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)
        self._fanout_planner = (
            FanoutChunkPlanner(
                self.config.fanout_chunk_blocks,
                min_fanout=self.config.fanout_min_fanout,
                allow_hot_shared_prefix_backup=(
                    self.config.fanout_allow_hot_prefix_backup
                ),
            )
            if self.config.fanout_offload
            else None
        )
        self._fanout_profile_steps = 0
        self._fanout_step = 0
        self._fanout_pressure_state = FanoutPressureLevel.NORMAL
        self._fanout_lifecycle: dict[OffloadKey, FanoutLifecycle] = {}
        self._fanout_admitted_keys: set[OffloadKey] = set()
        self._fanout_admitted_locations: dict[
            OffloadKey, set[tuple[ReqId, int, int]]
        ] = {}

    def _record_fanout_admission(
        self,
        offload_key: OffloadKey,
        req_id: ReqId,
        group_idx: int,
        group_state: RequestGroupState,
        logical_idx: int,
    ) -> None:
        self._fanout_admitted_keys.add(offload_key)
        group_state.fanout_admitted_block_indices.add(logical_idx)
        self._fanout_admitted_locations.setdefault(offload_key, set()).add(
            (req_id, group_idx, logical_idx)
        )

    def _reuse_fanout_admission(
        self,
        offload_key: OffloadKey,
        req_id: ReqId,
        group_idx: int,
        group_state: RequestGroupState,
        logical_idx: int,
    ) -> bool:
        """Reuse one CPU copy for every request sharing the same KV key."""
        if offload_key not in self._fanout_admitted_keys:
            return False
        self._record_fanout_admission(
            offload_key,
            req_id,
            group_idx,
            group_state,
            logical_idx,
        )
        return True

    def _invalidate_fanout_admissions(self, evicted_keys: Iterable[OffloadKey]) -> None:
        """Make CPU-evicted blocks eligible for another fanout backup."""
        for offload_key in evicted_keys:
            self._fanout_admitted_keys.discard(offload_key)
            locations = self._fanout_admitted_locations.pop(offload_key, ())
            for req_id, group_idx, logical_idx in locations:
                req_status = self._req_status.get(req_id)
                if req_status is None:
                    continue
                req_status.group_states[
                    group_idx
                ].fanout_admitted_block_indices.discard(logical_idx)

    def _forget_fanout_request(
        self, req_id: ReqId, req_status: RequestOffloadState
    ) -> None:
        for group_idx, group_state in enumerate(req_status.group_states):
            for logical_idx in group_state.fanout_admitted_block_indices:
                if logical_idx >= len(group_state.offload_keys):
                    continue
                offload_key = group_state.offload_keys[logical_idx]
                locations = self._fanout_admitted_locations.get(offload_key)
                if locations is None:
                    continue
                locations.discard((req_id, group_idx, logical_idx))
                if not locations:
                    del self._fanout_admitted_locations[offload_key]

    def _profile_fanout(
        self,
        *,
        kv_cache_usage: float,
        pressure_level: FanoutPressureLevel,
        candidates: int,
        hot_shared_candidates: int,
        lifecycle_hot_candidates: int,
        lifecycle_cooling_candidates: int,
        lifecycle_cold_candidates: int,
        selected_chunks: int,
        selected_blocks: int,
        protected_hot_shared_chunks: int,
        protected_hot_shared_blocks: int,
        selected_hot_shared_chunks: int,
        selected_hot_shared_blocks: int,
    ) -> None:
        if not self.config.fanout_profile:
            return
        self._fanout_profile_steps += 1
        logger.info(
            "Fanout offload profile: step=%d candidates=%d "
            "kv_cache_usage=%.4f pressure=%s "
            "hot_shared_candidates=%d selected_chunks=%d "
            "lifecycle_hot_candidates=%d lifecycle_cooling_candidates=%d "
            "lifecycle_cold_candidates=%d "
            "selected_blocks=%d protected_hot_shared_chunks=%d "
            "protected_hot_shared_blocks=%d selected_hot_shared_chunks=%d "
            "selected_hot_shared_blocks=%d layerwise_load=%s",
            self._fanout_profile_steps,
            candidates,
            kv_cache_usage,
            pressure_level.name.lower(),
            hot_shared_candidates,
            selected_chunks,
            lifecycle_hot_candidates,
            lifecycle_cooling_candidates,
            lifecycle_cold_candidates,
            selected_blocks,
            protected_hot_shared_chunks,
            protected_hot_shared_blocks,
            selected_hot_shared_chunks,
            selected_hot_shared_blocks,
            self.config.fanout_layerwise_load,
        )

    def _fanout_pressure_level(self, kv_cache_usage: float) -> FanoutPressureLevel:
        state = getattr(
            self,
            "_fanout_pressure_state",
            FanoutPressureLevel.NORMAL,
        )
        if state is FanoutPressureLevel.NORMAL:
            if kv_cache_usage >= self.config.fanout_critical_pressure_threshold:
                state = FanoutPressureLevel.CRITICAL
            elif kv_cache_usage >= self.config.fanout_high_pressure_threshold:
                state = FanoutPressureLevel.HIGH
        elif state is FanoutPressureLevel.HIGH:
            if kv_cache_usage >= self.config.fanout_critical_pressure_threshold:
                state = FanoutPressureLevel.CRITICAL
            elif kv_cache_usage < self.config.fanout_high_pressure_exit_threshold:
                state = FanoutPressureLevel.NORMAL
        elif kv_cache_usage < self.config.fanout_high_pressure_exit_threshold:
            state = FanoutPressureLevel.NORMAL
        elif kv_cache_usage < self.config.fanout_critical_pressure_exit_threshold:
            state = FanoutPressureLevel.HIGH

        self._fanout_pressure_state = state
        return state

    def _update_fanout_lifecycle(
        self,
        key: OffloadKey,
        *,
        fanout: int,
        reuse_score: int,
        base_hot: bool,
    ) -> tuple[FanoutLifecycleState, FanoutLifecycle]:
        step = self._fanout_step
        lifecycle = self._fanout_lifecycle.setdefault(key, FanoutLifecycle())
        lifecycle.historical_max_fanout = max(lifecycle.historical_max_fanout, fanout)
        lifecycle.historical_max_reuse_score = max(
            lifecycle.historical_max_reuse_score, reuse_score
        )
        if lifecycle.last_observed_step != step:
            if lifecycle.last_observed_step == step - 1:
                lifecycle.recent_access_count += 1
            else:
                lifecycle.recent_access_count = 1
            lifecycle.last_access_step = step
            lifecycle.last_observed_step = step

        if base_hot:
            if lifecycle.hot_since_step is None:
                lifecycle.hot_since_step = step
                lifecycle.min_resident_until_step = (
                    step + self.config.fanout_hot_prefix_min_residency_steps
                )
            lifecycle.last_hot_step = step
            return FanoutLifecycleState.HOT, lifecycle

        lifecycle.hot_since_step = None
        cooling_until = max(
            lifecycle.min_resident_until_step,
            lifecycle.last_hot_step + self.config.fanout_hot_prefix_cooldown_steps,
        )
        if lifecycle.last_hot_step > 0 and step <= cooling_until:
            return FanoutLifecycleState.COOLING, lifecycle
        return FanoutLifecycleState.COLD, lifecycle

    def _prune_fanout_lifecycle(self) -> None:
        if self._fanout_step % 64:
            return
        retention_steps = max(
            256,
            self.config.fanout_hot_prefix_cooldown_steps * 8,
        )
        stale_before = self._fanout_step - retention_steps
        self._fanout_lifecycle = {
            key: lifecycle
            for key, lifecycle in self._fanout_lifecycle.items()
            if lifecycle.last_observed_step >= stale_before
        }

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _maximal_prefix_lookup(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> int | None:
        """Return the number of consecutive offloaded blocks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for key in keys:
            match self.manager.lookup(key, req_context):
                case LookupResult.HIT:
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1
                case LookupResult.RETRY:
                    # Don't break: keep scanning to let manager kick off
                    # async lookups (until a miss is detected).
                    defer_lookup = True
                case LookupResult.MISS:
                    break
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        consecutive_hits = 0
        for idx in range(len(keys) - 1, -1, -1):
            match self.manager.lookup(keys[idx], req_context):
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    defer_lookup = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
                case LookupResult.MISS:
                    consecutive_hits = 0
            if consecutive_hits == sliding_window_size:
                return idx + sliding_window_size if not defer_lookup else None
        return consecutive_hits if not defer_lookup else None

    def _touch(self, req_status: RequestOffloadState):
        for group_config, group_state in zip(
            self.config.kv_group_configs, req_status.group_states
        ):
            if group_config.sliding_window_size_in_blocks is None:
                self.manager.touch(group_state.offload_keys, req_status.req_context)
            else:
                # we aim to keep just blocks that are necessary to hit
                # the original request (+ decoded blocks)
                blocks_to_skip = max(
                    0,
                    group_state.num_hit_blocks
                    - group_config.sliding_window_size_in_blocks,
                )
                self.manager.touch(
                    group_state.offload_keys[blocks_to_skip:],
                    req_status.req_context,
                )

    def _lookup(self, req_status: RequestOffloadState) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        max_hit_size_tokens: int = req_status.req.num_tokens
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing block
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                offloaded_block_size = group_config.offloaded_block_size
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys)
                    >= req_status.req.num_tokens // offloaded_block_size
                )

                is_eagle_unverified = (
                    group_config.is_eagle_group and group_idx not in eagle_verified
                )

                # Constrain to block-aligned boundary for this group
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * offloaded_block_size
                )
                if max_hit_size_tokens - num_computed_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )

                # For eagle groups, query one extra block that will be popped.
                # We only need to increase the query size for sliding window groups.
                query_max = max_hit_size_tokens
                if is_eagle_unverified and sliding_window_size_in_blocks is not None:
                    query_max = min(
                        max_hit_size_tokens + offloaded_block_size,
                        len(offload_keys) * offloaded_block_size,
                    )

                num_blocks = min(
                    cdiv(query_max, offloaded_block_size), len(offload_keys)
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]

                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_blocks: int | None
                if sliding_window_size_in_blocks is None:
                    num_hit_blocks = self._maximal_prefix_lookup(
                        offload_keys, req_status.req_context
                    )
                else:
                    required_window = sliding_window_size_in_blocks
                    if is_eagle_unverified:
                        required_window += 1
                    num_hit_blocks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                    )
                if num_hit_blocks == 0:
                    return 0

                if num_hit_blocks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        num_hit_blocks -= 1
                        eagle_verified.add(group_idx)

                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        offloaded_block_size * (start_block_idx + num_hit_blocks),
                    )

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < offloaded_block_size:
                    # we can only load less than a block, better skip
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_blocks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # possibly delay request if any of the hit blocks is already being loaded
        if self._blocks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                offloaded_block_size = group_config.offloaded_block_size
                sliding_window_size_in_blocks = (
                    group_config.sliding_window_size_in_blocks
                )
                offload_keys = group_state.offload_keys
                num_blocks = cdiv(
                    num_computed_tokens + num_hit_tokens, offloaded_block_size
                )
                start_block_idx = num_computed_tokens // offloaded_block_size
                offload_keys = offload_keys[start_block_idx:num_blocks]
                if sliding_window_size_in_blocks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_blocks:]
                if any(key in self._blocks_being_loaded for key in offload_keys):
                    # hit blocks are being loaded, delay request
                    logger.debug(
                        "Delaying request %s since some of its"
                        " blocks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).
        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
            num_hit_tokens = self._lookup(req_status)
        req_status.update_num_hit_blocks(num_computed_tokens + (num_hit_tokens or 0))

        self._touch(req_status)

        load_async = bool(num_hit_tokens) and not self.config.fanout_layerwise_load
        return num_hit_tokens, load_async

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens

        keys_to_load: list[OffloadKey] = []
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state, group_blocks in zip(
            self.config.kv_group_configs,
            req_status.group_states,
            blocks.blocks,
        ):
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            gpu_block_size = group_config.gpu_block_size
            offloaded_block_size = group_config.offloaded_block_size
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, gpu_block_size)

            assert len(group_blocks) >= num_gpu_blocks
            if self.config.fanout_layerwise_load:
                num_locally_computed_gpu_blocks = cdiv(
                    num_locally_computed_tokens,
                    gpu_block_size,
                )
            else:
                num_locally_computed_gpu_blocks = num_gpu_blocks
                # Skip null placeholders used for sliding window or mamba padding.
                for i, block in enumerate(group_blocks[:num_gpu_blocks]):
                    if not block.is_null and block.block_hash is None:
                        num_locally_computed_gpu_blocks = i
                        break

            assert (
                num_locally_computed_tokens
                <= num_locally_computed_gpu_blocks * gpu_block_size
            )
            num_pending_gpu_blocks = num_gpu_blocks - num_locally_computed_gpu_blocks

            if group_config.sliding_window_size_in_blocks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_blocks
                    * self.config.block_size_factor
                )

            num_blocks = cdiv(num_cached_tokens, offloaded_block_size)
            assert len(offload_keys) >= num_blocks
            if num_pending_gpu_blocks:
                start_block_idx = (
                    num_locally_computed_gpu_blocks // self.config.block_size_factor
                )
                keys_to_load.extend(offload_keys[start_block_idx:num_blocks])

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[
                    num_locally_computed_gpu_blocks:num_gpu_blocks
                ]
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(num_locally_computed_gpu_blocks)

            # Skip prefix-hit blocks for block-level policy; for
            # request-level, next_stored_block_idx stays at 0 so all
            # blocks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.BLOCK_LEVEL:
                group_state.next_stored_block_idx = num_blocks

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )

        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.update(keys_to_load)

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update request states from the Scheduler's output.
        """

        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for new_blocks in new_block_id_groups:
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_block_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            block_size_factor = self.config.block_size_factor
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_block_idx * block_size_factor
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _get_num_offloadable_tokens(
        self,
        req_status: RequestOffloadState,
        num_scheduled_tokens: int,
    ) -> int:
        req = req_status.req
        num_tokens = min(
            req.num_computed_tokens + num_scheduled_tokens,
            req.num_tokens,
        )
        if req_status.max_offload_tokens is not None:
            num_tokens = min(num_tokens, req_status.max_offload_tokens)
        if self.config.offload_prompt_only:
            num_tokens = min(num_tokens, req.num_prompt_tokens)
        return num_tokens

    def _select_fanout_offload_keys(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, set[OffloadKey]] | None:
        planner = self._fanout_planner
        if planner is None:
            return None

        self._fanout_step += 1
        kv_cache_usage = float(getattr(scheduler_output, "kv_cache_usage", 0.0))
        pressure_level = self._fanout_pressure_level(kv_cache_usage)

        fanout: Counter[tuple[int, int]] = Counter()
        for tracked_req_status in self._req_status.values():
            for group_config, group_state in zip(
                self.config.kv_group_configs,
                tracked_req_status.group_states,
            ):
                for block_id in group_state.block_ids:
                    if block_id != 0:
                        fanout[(group_config.group_idx, block_id)] += 1

        waiting_demand = scheduler_output.fanout_waiting_demand or {}
        observations: dict[OffloadKey, FanoutCandidateObservation] = {}
        for (
            req_id,
            num_scheduled_tokens,
        ) in scheduler_output.num_scheduled_tokens.items():
            candidate_req_status = self._req_status.get(req_id)
            if candidate_req_status is None or candidate_req_status.transfer_jobs:
                continue
            num_offloadable_tokens = self._get_num_offloadable_tokens(
                candidate_req_status,
                num_scheduled_tokens,
            )
            for group_config, group_state in zip(
                self.config.kv_group_configs,
                candidate_req_status.group_states,
            ):
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                if group_config.is_eagle_group:
                    num_blocks = max(0, num_blocks - 1)
                tail_start = max(
                    0,
                    num_blocks - self.config.fanout_recent_tail_blocks,
                )
                block_data: list[tuple[int, int, OffloadKey, int, bool]] = []
                for logical_idx in range(num_blocks):
                    physical_idx = logical_idx * self.config.block_size_factor
                    if physical_idx >= len(group_state.block_ids):
                        break
                    block_id = group_state.block_ids[physical_idx]
                    if logical_idx >= len(group_state.offload_keys):
                        break
                    offload_key = group_state.offload_keys[logical_idx]
                    already_admitted = self._reuse_fanout_admission(
                        offload_key,
                        req_id,
                        group_config.group_idx,
                        group_state,
                        logical_idx,
                    )
                    block_fanout = fanout[(group_config.group_idx, block_id)]
                    block_fanout += waiting_demand.get(
                        get_offload_block_hash(offload_key),
                        0,
                    )
                    block_data.append(
                        (
                            logical_idx,
                            block_id,
                            offload_key,
                            block_fanout,
                            not already_admitted,
                        )
                    )

                reuse_scores: dict[int, int] = {}
                run_start = 0
                while run_start < len(block_data):
                    run_fanout = block_data[run_start][3]
                    run_end = run_start + 1
                    while (
                        run_end < len(block_data)
                        and block_data[run_end][0] == block_data[run_end - 1][0] + 1
                        and block_data[run_end][3] == run_fanout
                    ):
                        run_end += 1
                    run_reuse_score = max(0, run_fanout - 1) * (run_end - run_start)
                    for run_idx in range(run_start, run_end):
                        reuse_scores[block_data[run_idx][0]] = run_reuse_score
                    run_start = run_end

                for (
                    logical_idx,
                    block_id,
                    offload_key,
                    block_fanout,
                    needs_backup,
                ) in block_data:
                    if block_id == 0:
                        continue
                    prefix_position = (logical_idx + 1) / max(num_blocks, 1)
                    reuse_score = reuse_scores[logical_idx]
                    observation = FanoutCandidateObservation(
                        request_id=req_id,
                        group_idx=group_config.group_idx,
                        logical_block_idx=logical_idx,
                        physical_block_id=block_id,
                        offload_key=offload_key,
                        fanout=block_fanout,
                        prefix_position=prefix_position,
                        reuse_score=reuse_score,
                        is_active_tail=logical_idx >= tail_start,
                        needs_backup=needs_backup,
                    )
                    previous = observations.get(offload_key)
                    if previous is None:
                        observations[offload_key] = observation
                    else:
                        previous.merge(observation)

        candidates: list[FanoutBlock] = []
        for observation in observations.values():
            base_hot = _is_hot_shared_prefix(
                fanout=observation.fanout,
                prefix_position=observation.prefix_position,
                min_fanout=self.config.fanout_hot_prefix_min_fanout,
                max_prefix_position=self.config.fanout_hot_prefix_max_position,
                reuse_score=observation.reuse_score,
                min_reuse_score=self.config.fanout_hot_prefix_min_reuse_blocks,
            )
            lifecycle_state, lifecycle = self._update_fanout_lifecycle(
                observation.offload_key,
                fanout=observation.fanout,
                reuse_score=observation.reuse_score,
                base_hot=base_hot,
            )
            residency_value = (
                observation.reuse_score * 16
                + lifecycle.historical_max_reuse_score * 4
                + min(lifecycle.recent_access_count, 8)
                * max(1, lifecycle.historical_max_fanout)
            )
            candidates.append(
                FanoutBlock(
                    request_id=observation.request_id,
                    group_idx=observation.group_idx,
                    logical_block_idx=observation.logical_block_idx,
                    physical_block_id=observation.physical_block_id,
                    offload_key=observation.offload_key,
                    fanout=observation.fanout,
                    prefix_position=observation.prefix_position,
                    last_access_time=float(lifecycle.last_access_step),
                    is_active_tail=observation.is_active_tail,
                    lifecycle_state=lifecycle_state,
                    historical_max_fanout=lifecycle.historical_max_fanout,
                    recent_access_count=lifecycle.recent_access_count,
                    residency_value=residency_value,
                    needs_backup=observation.needs_backup,
                )
            )

        eviction_metadata = {
            candidate.offload_key: _make_fanout_eviction_metadata(
                candidate,
                observations[candidate.offload_key],
            )
            for candidate in candidates
        }
        observed_keys = set(eviction_metadata)
        for offload_key, locations in self._fanout_admitted_locations.items():
            if offload_key in observed_keys:
                continue
            active_locations = [
                (req_id, group_idx, logical_idx)
                for req_id, group_idx, logical_idx in locations
                if req_id in self._req_status
            ]
            if not active_locations:
                continue
            fanout_value = len({location[0] for location in active_locations})
            fanout_value += waiting_demand.get(
                get_offload_block_hash(offload_key),
                0,
            )
            prefix_position = 1.0
            for req_id, group_idx, logical_idx in active_locations:
                group_state = self._req_status[req_id].group_states[group_idx]
                prefix_position = min(
                    prefix_position,
                    (logical_idx + 1) / max(len(group_state.offload_keys), 1),
                )
            previous_lifecycle = self._fanout_lifecycle.get(offload_key)
            historical_reuse = (
                previous_lifecycle.historical_max_reuse_score
                if previous_lifecycle is not None
                else 0
            )
            reuse_score = max(0, fanout_value - 1)
            base_hot = _is_hot_shared_prefix(
                fanout=fanout_value,
                prefix_position=prefix_position,
                min_fanout=self.config.fanout_hot_prefix_min_fanout,
                max_prefix_position=self.config.fanout_hot_prefix_max_position,
                reuse_score=max(reuse_score, historical_reuse),
                min_reuse_score=self.config.fanout_hot_prefix_min_reuse_blocks,
            )
            lifecycle_state, lifecycle = self._update_fanout_lifecycle(
                offload_key,
                fanout=fanout_value,
                reuse_score=reuse_score,
                base_hot=base_hot,
            )
            residency_value = (
                reuse_score * 16
                + lifecycle.historical_max_reuse_score * 4
                + min(lifecycle.recent_access_count, 8)
                * max(1, lifecycle.historical_max_fanout)
            )
            eviction_metadata[offload_key] = OffloadEvictionMetadata(
                lifecycle_value=lifecycle_state.value,
                reuse_score=reuse_score,
                fanout=max(fanout_value, lifecycle.historical_max_fanout),
                residency_value=residency_value,
                prefix_position=prefix_position,
            )
        self.manager.update_eviction_metadata(eviction_metadata, replace=True)

        self._prune_fanout_lifecycle()
        plan = planner.select(
            (candidate for candidate in candidates if candidate.needs_backup),
            self.config.fanout_budget_blocks,
            pressure_level,
        )
        selected: dict[str, set[OffloadKey]] = {}
        for chunk in plan.chunks:
            selected.setdefault(chunk.request_id, set()).update(chunk.offload_keys)
        if self.config.fanout_profile:
            lifecycle_counts = Counter(
                candidate.lifecycle_state for candidate in candidates
            )
            selected_hot_chunks = [
                chunk for chunk in plan.chunks if chunk.is_hot_shared_prefix
            ]
            self._profile_fanout(
                kv_cache_usage=kv_cache_usage,
                pressure_level=pressure_level,
                candidates=len(candidates),
                hot_shared_candidates=(
                    lifecycle_counts[FanoutLifecycleState.HOT]
                    + lifecycle_counts[FanoutLifecycleState.COOLING]
                ),
                lifecycle_hot_candidates=lifecycle_counts[FanoutLifecycleState.HOT],
                lifecycle_cooling_candidates=lifecycle_counts[
                    FanoutLifecycleState.COOLING
                ],
                lifecycle_cold_candidates=lifecycle_counts[FanoutLifecycleState.COLD],
                selected_chunks=len(plan.chunks),
                selected_blocks=plan.num_blocks,
                protected_hot_shared_chunks=len(plan.protected_hot_shared_chunks),
                protected_hot_shared_blocks=plan.num_protected_hot_shared_blocks,
                selected_hot_shared_chunks=len(selected_hot_chunks),
                selected_hot_shared_blocks=sum(
                    chunk.num_blocks for chunk in selected_hot_chunks
                ),
            )
        return selected

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        block_size_factor = self.config.block_size_factor
        store_jobs: dict[int, TransferJob] = {}
        fanout_selected_keys = self._select_fanout_offload_keys(scheduler_output)
        for req_id in scheduler_output.num_scheduled_tokens:
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            if req_status.transfer_jobs:
                any_job_id = next(iter(req_status.transfer_jobs))
                if not self._jobs[any_job_id].is_store:
                    continue
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_offloadable_tokens = self._get_num_offloadable_tokens(
                req_status,
                num_scheduled_tokens,
            )
            req = req_status.req

            # Filter out blocks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            fanout_key_indices: dict[
                OffloadKey, tuple[int, RequestGroupState, int]
            ] = {}
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                if group_config.is_eagle_group:
                    num_blocks = max(0, num_blocks - 1)

                start_block_idx = (
                    0
                    if fanout_selected_keys is not None
                    else group_state.next_stored_block_idx
                )
                if num_blocks <= start_block_idx:
                    continue
                offload_keys = group_state.offload_keys[start_block_idx:num_blocks]
                # For each block to offload, take the last corresponding GPU block.
                # e.g. if block size factor is 3 and GPU block IDs are
                # 1 5 6 7 2 4 9 3 8 then we'll take blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                offload_block_ids = group_state.block_ids[
                    start_block_idx * block_size_factor
                    + block_size_factor
                    - 1 : num_blocks * block_size_factor : block_size_factor
                ]
                assert len(offload_keys) == len(offload_block_ids)

                alignment_block_count = group_config.alignment_block_count
                tail = group_config.sliding_window_size_in_blocks

                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    logical_idx = start_block_idx + key_idx
                    if (
                        fanout_selected_keys is not None
                        and offload_key not in fanout_selected_keys.get(req_id, set())
                    ):
                        continue
                    if block_id == 0:
                        continue
                    # Skip SWA blocks that can never serve a load hit:
                    # within each full-attention alignment segment, only the
                    # trailing `tail` blocks are reachable by
                    # _sliding_window_lookup. For DeepSeek V4 with 100K
                    # tokens this reduces SWA stores by ~78%.
                    if alignment_block_count is not None:
                        assert tail is not None
                        abs_block_idx = logical_idx
                        pos_in_segment = abs_block_idx % alignment_block_count
                        if pos_in_segment < alignment_block_count - tail:
                            continue
                    new_offload_keys.append(offload_key)
                    if fanout_selected_keys is not None:
                        fanout_key_indices[offload_key] = (
                            group_config.group_idx,
                            group_state,
                            logical_idx,
                        )

            if not new_offload_keys:
                if fanout_selected_keys is None:
                    req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            store_output = self.manager.prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                logger.warning("Request %s: cannot store blocks", req_id)
                continue

            self._invalidate_fanout_admissions(store_output.evicted_keys)

            for offload_key in new_offload_keys:
                location = fanout_key_indices.get(offload_key)
                if location is not None:
                    group_idx, group_state, logical_idx = location
                    self._record_fanout_admission(
                        offload_key,
                        req_id,
                        group_idx,
                        group_state,
                        logical_idx,
                    )

            if not store_output.keys_to_store:
                if fanout_selected_keys is None:
                    req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            self._touch(req_status)

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            sliding_window_block_ids: list[int] = []
            non_sliding_window_block_ids: list[int] = []
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_blocks is not None
                )
                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size
                start_block_idx = (
                    0
                    if fanout_selected_keys is not None
                    else group_state.next_stored_block_idx
                )
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_block_idx:num_blocks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    offloaded_block_idx = start_block_idx + idx

                    self._events_tracker.record_store(
                        req, group_config, offloaded_block_idx, offload_key
                    )

                    gpu_block_idx = offloaded_block_idx * block_size_factor
                    for i in range(block_size_factor):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            sliding_window_block_ids.append(block_id)
                        else:
                            non_sliding_window_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                if fanout_selected_keys is None:
                    group_state.next_stored_block_idx = num_blocks

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                non_sliding_window_block_ids=non_sliding_window_block_ids,
                sliding_window_block_ids=sliding_window_block_ids or None,
            )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )

            logger.debug(
                "Request %s offloading %s blocks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        self.manager.on_schedule_end()

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=self._build_store_jobs(scheduler_output),
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.
        """
        return bool(self._jobs) or self.manager.has_pending_work()

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            if self._connector_stats is None:
                self._connector_stats = transfer_stats
            else:
                self._connector_stats.aggregate(transfer_stats)

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if self._blocks_being_loaded:
                    self._blocks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.sliding_window_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.non_sliding_window_block_ids
                    )

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if not req_status.transfer_jobs and req_status.req.is_finished():
                self._forget_fanout_request(job_status.req_id, req_status)
                del self._req_status[job_status.req_id]

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = self._connector_stats
        self._connector_stats = None

        manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # TODO(orozery): possibly kickoff offload for last block
        # which may have been deferred due to async scheduling
        req_status = self._req_status.get(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            req_context = _create_req_context(request)
            self.manager.on_new_request(req_context)
            self.manager.on_request_finished(req_context)
            return False, None

        self.manager.on_request_finished(req_status.req_context)

        if not req_status.transfer_jobs:
            # No in-flight jobs: no later complete_store()/complete_load() calls
            # need this request's state.
            self._forget_fanout_request(request.request_id, req_status)
            del self._req_status[request.request_id]
            return False, None

        # In-flight jobs remain after the request stopped. Their completion may
        # still call manager.complete_store()/complete_load(), so keep req_status.
        # Pending stores outlive the request's block ownership; register them so
        # future reuse of those blocks triggers a flush.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.non_sliding_window_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.
        """
        yield from self._events_tracker.take_events(self.manager.take_events())

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored blocks."""

        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                self._forget_fanout_request(req_id, status)
                del self._req_status[req_id]

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from block 0
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_block_idx = 0
                group_state.fanout_admitted_block_indices.clear()
        self._fanout_lifecycle.clear()
        self._fanout_admitted_keys.clear()
        self._fanout_admitted_locations.clear()
        self._fanout_step = 0

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._blocks_being_loaded is not None:
            self._blocks_being_loaded.clear()

    def shutdown(self) -> None:
        self.manager.shutdown()
