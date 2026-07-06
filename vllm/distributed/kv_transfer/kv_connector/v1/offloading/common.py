# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.base import LoadStoreSpec

ReqId = str

FANOUT_LAYERWISE_LOAD_AUTO = "auto"
DEFAULT_FANOUT_LAYERWISE_LOAD_THRESHOLD_BYTES = 256 * 1024 * 1024


def parse_fanout_layerwise_load(value: object) -> bool | None:
    """Parse fanout_layerwise_load.

    Returns:
        True/False for explicit modes, or None for auto.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == FANOUT_LAYERWISE_LOAD_AUTO:
            return None
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    raise ValueError(
        f"fanout_layerwise_load must be a boolean or 'auto', got {value!r}"
    )


def resolve_fanout_layerwise_load(
    value: object,
    *,
    fanout_offload: bool,
    has_full_cudagraphs: bool,
    estimated_load_bytes: int = 0,
    threshold_bytes: int = DEFAULT_FANOUT_LAYERWISE_LOAD_THRESHOLD_BYTES,
) -> bool:
    parsed = parse_fanout_layerwise_load(value)
    if parsed is not None:
        return parsed
    if not fanout_offload:
        return False
    # FULL cudagraph replay was faster than layerwise overlap for the long
    # fanout workload we profiled. Keep it unless the user explicitly asks for
    # layerwise load.
    if has_full_cudagraphs:
        return False
    if threshold_bytes <= 0:
        return True
    return estimated_load_bytes >= threshold_bytes


def fanout_profiling_enabled(
    extra_config: dict[str, Any],
    vllm_config: Any,
) -> bool:
    if bool(extra_config.get("fanout_profile", False)):
        return True
    profiler_config = getattr(vllm_config, "profiler_config", None)
    return getattr(profiler_config, "profiler", None) is not None


@dataclass(slots=True)
class DirectionalTransferStats:
    bytes: int = 0
    time: float = 0.0
    sizes: list[int | float] = field(default_factory=list)

    def aggregate(
        self, other: "DirectionalTransferStats"
    ) -> "DirectionalTransferStats":
        return DirectionalTransferStats(
            bytes=self.bytes + other.bytes,
            time=self.time + other.time,
            sizes=[*self.sizes, *other.sizes],
        )

    def record(self, num_bytes: int, time: float) -> None:
        self.bytes += num_bytes
        self.time += time
        self.sizes.append(num_bytes)

    def is_empty(self) -> bool:
        return self.bytes == 0 and self.time == 0.0 and not self.sizes


@dataclass(slots=True)
class TransferStats:
    load: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)
    store: DirectionalTransferStats = field(default_factory=DirectionalTransferStats)

    def aggregate(self, other: "TransferStats") -> "TransferStats":
        return TransferStats(
            load=self.load.aggregate(other.load),
            store=self.store.aggregate(other.store),
        )

    def is_empty(self) -> bool:
        return self.load.is_empty() and self.store.is_empty()


@dataclass
class TransferJob:
    """A transfer job bundling request context with transfer spec.

    Used for both loads and stores, keyed by scheduler-assigned job ID.
    The worker reports the job ID back when the transfer finishes,
    and the scheduler processes the completion.
    """

    req_id: ReqId
    src_spec: LoadStoreSpec
    dst_spec: LoadStoreSpec


@dataclass
class OffloadingConnectorMetadata(KVConnectorMetadata):
    # Keyed by scheduler-assigned job IDs.
    load_jobs: dict[int, TransferJob]
    store_jobs: dict[int, TransferJob]
    jobs_to_flush: set[int] | None = None


@dataclass
class OffloadingWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed transfer jobs.

    Each worker reports {job_id: 1} for newly completed transfer jobs
    (load or store). aggregate() sums counts across workers within a step.
    The scheduler accumulates across steps and processes
    a transfer completion only when count reaches num_workers.
    """

    completed_jobs: dict[int, int] = field(default_factory=dict)
    transfer_stats: TransferStats = field(default_factory=TransferStats)

    def mark_completed(self, job_id: int) -> None:
        """Record a transfer job completion from this worker."""
        self.completed_jobs[job_id] = 1

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, OffloadingWorkerMetadata)

        merged = dict(self.completed_jobs)
        for job_id, v in other.completed_jobs.items():
            merged[job_id] = merged.get(job_id, 0) + v

        return OffloadingWorkerMetadata(
            completed_jobs=merged,
            transfer_stats=self.transfer_stats.aggregate(other.transfer_stats),
        )
