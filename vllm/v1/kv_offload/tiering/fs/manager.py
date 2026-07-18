# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)
"""

import functools
import json
import os
import threading
import time
from collections import Counter
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FileSystemTierMetrics:
    """Prometheus names for physical secondary-tier activity."""

    SUBMITTED_JOBS = "vllm:kv_offload_secondary_submitted_jobs"
    SUBMITTED_BLOCKS = "vllm:kv_offload_secondary_submitted_blocks"
    SUBMITTED_BYTES = "vllm:kv_offload_secondary_submitted_bytes"
    TRANSFERRED_BLOCKS = "vllm:kv_offload_secondary_transferred_blocks"
    TRANSFERRED_BYTES = "vllm:kv_offload_secondary_transferred_bytes"
    DEDUP_SKIPPED_BLOCKS = "vllm:kv_offload_secondary_dedup_skipped_blocks"
    DEDUP_SKIPPED_BYTES = "vllm:kv_offload_secondary_dedup_skipped_bytes"
    COMPLETED_JOBS = "vllm:kv_offload_secondary_completed_jobs"
    FAILED_JOBS = "vllm:kv_offload_secondary_failed_jobs"
    JOB_LATENCY = "vllm:kv_offload_secondary_job_latency_seconds"
    LOOKUPS = "vllm:kv_offload_secondary_lookups"
    INFLIGHT_JOBS = "vllm:kv_offload_secondary_inflight_jobs"


class FsAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for FileSystemTierManager."""

    def __init__(
        self,
        tier: "FileSystemTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        return (os.path.exists(self._tier.file_mapper.get_file_name(k)) for k in keys)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    Cross-process sharing:
        In order to enable KV cache sharing between multiple vLLM instances
        using the same ``root_dir`` (e.g., via a shared PVC) the environment
        variable ``PYTHONHASHSEED`` must be set to the same fixed value
        (e.g., "0") on all instances. Without this, each process initializes
        ``NONE_HASH`` (the chain-hash seed for block content hashes) with
        random bytes, producing different block filenames for identical token
        content.
    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
            parallel_agnostic=True,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)
        self._metrics_lock = threading.Lock()
        self._metric_counters: Counter[tuple[str, str]] = Counter()
        self._lookup_counters: Counter[str] = Counter()
        self._job_latencies: list[tuple[str, float]] = []
        self._inflight_jobs: Counter[str] = Counter()
        self._job_observations: dict[int, tuple[str, float]] = {}

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        direction_labels = ("tier", "direction")
        definitions: dict[str, OffloadingMetricMetadata] = {}
        for name, description in (
            (FileSystemTierMetrics.SUBMITTED_JOBS, "Submitted secondary-tier jobs."),
            (
                FileSystemTierMetrics.SUBMITTED_BLOCKS,
                "KV blocks submitted to the secondary tier.",
            ),
            (
                FileSystemTierMetrics.SUBMITTED_BYTES,
                "Logical KV bytes submitted to the secondary tier.",
            ),
            (
                FileSystemTierMetrics.TRANSFERRED_BLOCKS,
                "KV blocks physically transferred by the secondary tier.",
            ),
            (
                FileSystemTierMetrics.TRANSFERRED_BYTES,
                "KV bytes physically transferred by the secondary tier.",
            ),
            (
                FileSystemTierMetrics.DEDUP_SKIPPED_BLOCKS,
                "Store blocks skipped because the physical file already exists.",
            ),
            (
                FileSystemTierMetrics.DEDUP_SKIPPED_BYTES,
                "Store bytes skipped because the physical file already exists.",
            ),
            (
                FileSystemTierMetrics.COMPLETED_JOBS,
                "Successfully completed secondary-tier jobs.",
            ),
            (FileSystemTierMetrics.FAILED_JOBS, "Failed secondary-tier jobs."),
        ):
            definitions[name] = OffloadingCounterMetadata(
                documentation=description,
                labelnames=direction_labels,
            )
        definitions[FileSystemTierMetrics.JOB_LATENCY] = (
            OffloadingHistogramMetadata(
                documentation=(
                    "Secondary-tier job latency from submission to completion."
                ),
                labelnames=direction_labels,
                buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            )
        )
        definitions[FileSystemTierMetrics.LOOKUPS] = OffloadingCounterMetadata(
            documentation="Resolved secondary-tier lookup observations.",
            labelnames=("tier", "result"),
        )
        definitions[FileSystemTierMetrics.INFLIGHT_JOBS] = OffloadingGaugeMetadata(
            documentation="Currently in-flight secondary-tier jobs.",
            labelnames=direction_labels,
        )
        return definitions

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            with self._metrics_lock:
                self._lookup_counters["retry"] += 1
            return LookupResult.RETRY
        outcome = "hit" if result else "miss"
        with self._metrics_lock:
            self._lookup_counters[outcome] += 1
        return LookupResult.HIT if result else LookupResult.MISS

    def _record_submitted_job(self, job_metadata: JobMetadata, direction: str) -> None:
        num_blocks = len(job_metadata.keys)
        with self._metrics_lock:
            self._metric_counters[
                (FileSystemTierMetrics.SUBMITTED_JOBS, direction)
            ] += 1
            self._metric_counters[
                (FileSystemTierMetrics.SUBMITTED_BLOCKS, direction)
            ] += num_blocks
            self._metric_counters[
                (FileSystemTierMetrics.SUBMITTED_BYTES, direction)
            ] += num_blocks * self._block_size
            self._inflight_jobs[direction] += 1
            self._job_observations[job_metadata.job_id] = (
                direction,
                time.monotonic(),
            )

    def _store_block(self, path: str, offset: int) -> None:
        bytes_written = store_block(
            path,
            self._primary_kv_view,
            offset,
            self._block_size,
        )
        with self._metrics_lock:
            if bytes_written:
                self._metric_counters[
                    (FileSystemTierMetrics.TRANSFERRED_BLOCKS, "store")
                ] += 1
                self._metric_counters[
                    (FileSystemTierMetrics.TRANSFERRED_BYTES, "store")
                ] += bytes_written
            else:
                self._metric_counters[
                    (FileSystemTierMetrics.DEDUP_SKIPPED_BLOCKS, "store")
                ] += 1
                self._metric_counters[
                    (FileSystemTierMetrics.DEDUP_SKIPPED_BYTES, "store")
                ] += self._block_size

    def _load_block(self, path: str, offset: int) -> None:
        bytes_read = load_block(
            path,
            self._primary_kv_view,
            offset,
            self._block_size,
        )
        with self._metrics_lock:
            self._metric_counters[
                (FileSystemTierMetrics.TRANSFERRED_BLOCKS, "load")
            ] += 1
            self._metric_counters[
                (FileSystemTierMetrics.TRANSFERRED_BYTES, "load")
            ] += bytes_read

    @override
    def submit_store(self, job_metadata: JobMetadata) -> None:
        self._record_submitted_job(job_metadata, "store")
        tasks = (
            functools.partial(
                self._store_block,
                self.file_mapper.get_file_name(key),
                int(bid) * self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def submit_load(self, job_metadata: JobMetadata) -> None:
        self._record_submitted_job(job_metadata, "load")
        tasks = (
            functools.partial(
                self._load_block,
                self.file_mapper.get_file_name(key),
                int(bid) * self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    @override
    def submit_load_batch(self, jobs: Collection[JobMetadata]) -> None:
        queued_jobs = []
        for job_metadata in jobs:
            self._record_submitted_job(job_metadata, "load")
            tasks = (
                functools.partial(
                    self._load_block,
                    self.file_mapper.get_file_name(key),
                    int(bid) * self._block_size,
                )
                for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
            )
            queued_jobs.append(
                (job_metadata.job_id, len(job_metadata.keys), tasks)
            )
        self._pool.enqueue_load_batch(queued_jobs)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = []
        now = time.monotonic()
        for job_id, success in self._pool.get_finished():
            with self._metrics_lock:
                direction, submitted_at = self._job_observations.pop(job_id)
                self._inflight_jobs[direction] -= 1
                metric = (
                    FileSystemTierMetrics.COMPLETED_JOBS
                    if success
                    else FileSystemTierMetrics.FAILED_JOBS
                )
                self._metric_counters[(metric, direction)] += 1
                self._job_latencies.append((direction, now - submitted_at))
            results.append(JobResult(job_id=job_id, success=success))
        return results

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()
        with self._metrics_lock:
            for (metric, direction), value in self._metric_counters.items():
                stats.increase_counter(metric, value, (self.tier_type, direction))
            for result, value in self._lookup_counters.items():
                stats.increase_counter(
                    FileSystemTierMetrics.LOOKUPS,
                    value,
                    (self.tier_type, result),
                )
            for direction, latency in self._job_latencies:
                stats.observe_histogram(
                    FileSystemTierMetrics.JOB_LATENCY,
                    latency,
                    (self.tier_type, direction),
                )
            for direction in ("load", "store"):
                stats.set_gauge(
                    FileSystemTierMetrics.INFLIGHT_JOBS,
                    self._inflight_jobs[direction],
                    (self.tier_type, direction),
                )
            self._metric_counters.clear()
            self._lookup_counters.clear()
            self._job_latencies.clear()
        return stats

    @override
    def drain_jobs(self) -> None:
        """Block until all in-flight transfers in the threadpool finish."""
        self._pool.wait_idle()

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    @override
    def on_schedule_end(self) -> None:
        self._lookup_manager.flush()

    @override
    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the lookup manager and the thread pool,
        clearing pending tasks and waiting for active threads to complete.
        """
        self._lookup_manager.shutdown()
        self._pool.shutdown(wait=True)
