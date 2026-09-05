# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.kv_residency import (
    KVBlockState,
    KVResidencyIndex,
    KVResidencyTier,
    create_kv_residency_index,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash() -> None:
    init_none_hash(sha256)


def _index(
    *,
    num_blocks: int = 8,
    warm_seconds: float = 10.0,
    cooling_seconds: float = 100.0,
    shared_reuse_threshold: int = 2,
    aging_budget: int = 256,
) -> KVResidencyIndex:
    return KVResidencyIndex(
        num_blocks,
        null_block_id=0,
        warm_seconds=warm_seconds,
        cooling_seconds=cooling_seconds,
        shared_reuse_threshold=shared_reuse_threshold,
        aging_budget=aging_budget,
        now_ns=1_000_000_000,
    )


def test_lifecycle_ages_through_ordered_segments() -> None:
    index = _index()

    generation = index.on_allocated(1, 1)
    assert generation == 1
    assert index.get_entry(1).state == KVBlockState.UNHASHED

    index.on_cache_inserted(1)
    assert index.get_entry(1).state == KVBlockState.ACTIVE

    index.on_released(1, 0)
    assert index.get_entry(1).state == KVBlockState.WARM

    index.advance(now_ns=11_000_000_000)
    assert index.get_entry(1).state == KVBlockState.COOLING

    index.advance(now_ns=101_000_000_000)
    assert index.get_entry(1).state == KVBlockState.COLD
    assert index.snapshot().aging_transitions == 2
    index.check_consistency()


def test_reuse_promotes_prefix_to_shared_segments() -> None:
    index = _index(shared_reuse_threshold=2)
    index.on_allocated(1, 1)
    index.on_cache_inserted(1)
    index.on_released(1, 0)

    index.on_cache_hit(1, 1)
    index.on_released(1, 0)
    assert index.get_entry(1).state == KVBlockState.WARM

    index.on_cache_hit(1, 1)
    assert index.get_entry(1).state == KVBlockState.ACTIVE_SHARED
    index.on_released(1, 0)
    assert index.get_entry(1).state == KVBlockState.WARM_SHARED

    stats = index.snapshot()
    assert stats.shared_blocks == 1
    assert stats.warm_shared_blocks == 1
    index.check_consistency()


def test_concurrent_fanout_marks_shared_immediately() -> None:
    index = _index(shared_reuse_threshold=100)
    index.on_allocated(1, 1)
    index.on_cache_inserted(1)

    index.on_cache_hit(1, 2)

    entry = index.get_entry(1)
    assert entry.state == KVBlockState.ACTIVE_SHARED
    assert entry.peak_fanout == 2


def test_internal_pin_does_not_count_as_prefix_reuse() -> None:
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    index = KVResidencyIndex(
        pool.num_gpu_blocks,
        null_block_id=pool.null_block.block_id,
        now_ns=1_000_000_000,
    )
    pool.set_observer(index)
    block = pool.get_new_blocks(1)[0]
    block_hash = make_block_hash_with_group_id(BlockHash(b"p" * 32), 0)
    pool._insert_block_hash(block_hash, block, num_tokens=16)

    pool.pin([block])
    entry = index.get_entry(block.block_id)
    assert entry.ref_count == 2
    assert entry.request_ref_count == 1
    assert entry.reuse_count == 0
    assert entry.peak_fanout == 1
    assert entry.state == KVBlockState.ACTIVE

    pool.unpin([block])
    entry = index.get_entry(block.block_id)
    assert entry.ref_count == 1
    assert entry.request_ref_count == 1
    assert entry.state == KVBlockState.ACTIVE


def test_internal_pin_restores_idle_age_order() -> None:
    index = _index()
    for block_id, now_ns in ((1, 1_000_000_000), (2, 2_000_000_000)):
        index.advance(now_ns=now_ns)
        index.on_allocated(block_id, 1)
        index.on_cache_inserted(block_id)
        index.on_released(block_id, 0)

    index.on_pinned(1, 1)
    index.on_unpinned(1, 0)

    assert index.oldest_block_ids(KVBlockState.WARM, 2) == [1, 2]
    assert index.get_entry(1).last_access_ns == 1_000_000_000
    index.check_consistency()


def test_internal_pin_restores_block_pool_lru_position() -> None:
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    first, second = pool.get_new_blocks(2)
    pool._insert_block_hash(
        make_block_hash_with_group_id(BlockHash(b"a" * 32), 0),
        first,
        num_tokens=16,
    )
    pool._insert_block_hash(
        make_block_hash_with_group_id(BlockHash(b"b" * 32), 0),
        second,
        num_tokens=16,
    )
    pool.free_blocks([first, second])

    pool.pin([first])
    pool.unpin([first])

    assert [block.block_id for block in pool.get_new_blocks(3)] == [3, 1, 2]


def test_internal_pin_rejects_reused_idle_anchors() -> None:
    index = _index(num_blocks=4)
    for block_id in (1, 2, 3):
        index.on_allocated(block_id, 1)
        index.on_cache_inserted(block_id)
        index.on_released(block_id, 0)
    index.on_pinned(2, 1)

    index.advance(now_ns=2_000_000_000)
    for block_id in (1, 3):
        index.on_cache_removed(block_id, 1)
        index.on_allocated(block_id, 1)
        index.on_cache_inserted(block_id)
        index.on_released(block_id, 0)
    index.on_unpinned(2, 0)

    assert index.oldest_block_ids(KVBlockState.WARM, 3) == [1, 3, 2]
    assert index.get_entry(2).last_access_ns == 2_000_000_000
    index.check_consistency()


def test_generation_change_resets_shared_history() -> None:
    index = _index(shared_reuse_threshold=100)
    generation = index.on_allocated(1, 1)
    index.on_cache_inserted(1)
    index.on_cache_hit(1, 2)
    index.on_released(1, 1)
    assert index.get_entry(1).peak_fanout == 2

    index.on_cache_removed(1, 1)
    entry = index.get_entry(1)
    assert entry.generation == generation + 1
    assert entry.reuse_count == 0
    assert entry.peak_fanout == 1

    index.on_cache_inserted(1)
    assert index.get_entry(1).state == KVBlockState.ACTIVE


def test_aging_work_is_bounded_per_step() -> None:
    budget = 7
    index = _index(num_blocks=101, aging_budget=budget)
    for block_id in range(1, 101):
        index.on_allocated(block_id, 1)
        index.on_cache_inserted(block_id)
        index.on_released(block_id, 0)

    transitioned = index.advance(now_ns=101_000_000_000)

    assert transitioned == budget
    assert index.snapshot().aging_transitions == budget
    assert index.snapshot().warm_blocks == 100 - budget
    index.check_consistency()


def test_generation_rejects_stale_backup_ack() -> None:
    index = _index()
    generation = index.on_allocated(1, 1)
    index.on_cache_inserted(1)
    assert index.start_backup(1, generation, KVResidencyTier.CPU, 10)

    index.on_cache_removed(1, 1)
    assert not index.finish_backup(
        1,
        generation,
        KVResidencyTier.CPU,
        10,
        success=True,
    )

    entry = index.get_entry(1)
    assert entry.generation != generation
    assert entry.residency == KVResidencyTier.GPU
    stats = index.snapshot()
    assert stats.backing_up_blocks == 0
    assert stats.backed_up_blocks == 0
    assert stats.stale_updates == 1


def test_backup_residency_is_counted_once_across_tiers() -> None:
    index = _index()
    generation = index.on_allocated(1, 1)
    index.on_cache_inserted(1)

    assert index.start_backup(1, generation, KVResidencyTier.CPU, 10)
    assert index.finish_backup(1, generation, KVResidencyTier.CPU, 10, success=True)
    assert index.start_backup(1, generation, KVResidencyTier.REMOTE, 11)
    assert index.finish_backup(1, generation, KVResidencyTier.REMOTE, 11, success=True)
    assert index.snapshot().backed_up_blocks == 1

    assert index.drop_backup(1, generation, KVResidencyTier.CPU)
    assert index.snapshot().backed_up_blocks == 1
    assert index.drop_backup(1, generation, KVResidencyTier.REMOTE)
    assert index.snapshot().backed_up_blocks == 0


def test_restore_records_only_the_current_cached_generation() -> None:
    index = _index()
    generation = index.on_allocated(1, 1)
    index.on_cache_inserted(1)

    assert index.record_restore(1, generation, KVResidencyTier.CPU)
    assert index.get_entry(1).residency & KVResidencyTier.CPU
    assert index.snapshot().backed_up_blocks == 1
    assert index.record_restore(1, generation, KVResidencyTier.CPU)
    assert index.snapshot().backed_up_blocks == 1

    index.on_cache_removed(1, 1)
    index.on_cache_inserted(1)
    assert not index.record_restore(1, generation, KVResidencyTier.CPU)
    assert not index.get_entry(1).residency & KVResidencyTier.CPU


def test_concurrent_backup_acks_are_matched_by_tier_and_operation() -> None:
    index = _index()
    generation = index.on_allocated(1, 1)
    index.on_cache_inserted(1)

    assert index.start_backup(1, generation, KVResidencyTier.CPU, 10)
    assert index.start_backup(1, generation, KVResidencyTier.CPU, 11)
    assert index.start_backup(1, generation, KVResidencyTier.REMOTE, 10)
    entry = index.get_entry(1)
    assert entry.backing_up
    assert entry.backing_up_tiers == (KVResidencyTier.CPU | KVResidencyTier.REMOTE)
    assert index.snapshot().backing_up_blocks == 1

    assert index.finish_backup(1, generation, KVResidencyTier.CPU, 10, success=True)
    assert index.get_entry(1).backing_up
    assert not index.finish_backup(1, generation, KVResidencyTier.CPU, 99, success=True)
    assert index.get_entry(1).residency & KVResidencyTier.CPU

    assert index.finish_backup(1, generation, KVResidencyTier.REMOTE, 10, success=True)
    assert index.get_entry(1).backing_up
    assert index.finish_backup(1, generation, KVResidencyTier.CPU, 11, success=False)
    assert not index.get_entry(1).backing_up
    assert index.snapshot().backing_up_blocks == 0
    index.check_consistency()


def test_oldest_only_walks_requested_segment() -> None:
    index = _index()
    for block_id in (1, 2, 3):
        index.on_allocated(block_id, 1)
        index.on_cache_inserted(block_id)
        index.on_released(block_id, 0)

    assert [entry.block_id for entry in index.oldest(KVBlockState.WARM, 2)] == [
        1,
        2,
    ]


def test_backup_metadata_and_cache_occupancy_are_constant_size() -> None:
    index = _index(num_blocks=5)
    assert index.cache_occupancy() == 0.0
    generation = index.on_allocated(1, 1)
    index.on_cache_inserted(1)
    index.on_released(1, 0)

    assert index.cache_occupancy() == 0.25
    assert index.backup_metadata(1) == (
        generation,
        True,
        False,
        False,
        False,
    )


def test_block_pool_observer_tracks_real_transitions_without_changing_lru() -> None:
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    index = KVResidencyIndex(
        pool.num_gpu_blocks,
        null_block_id=pool.null_block.block_id,
        now_ns=1_000_000_000,
    )
    pool.set_observer(index)

    block = pool.get_new_blocks(1)[0]
    block_hash = make_block_hash_with_group_id(BlockHash(b"a" * 32), 0)
    pool._insert_block_hash(block_hash, block, num_tokens=16)
    pool.free_blocks([block])

    assert index.get_entry(block.block_id).state == KVBlockState.WARM
    assert pool.get_cached_block(BlockHash(b"a" * 32), [0]) == [block]

    pool.touch([block])
    assert index.get_entry(block.block_id).state == KVBlockState.ACTIVE
    pool.free_blocks([block])

    # The cached block remains at the LRU tail, exactly as without an observer.
    allocated = pool.get_new_blocks(3)
    assert allocated[-1] is block
    entry = index.get_entry(block.block_id)
    assert entry.state == KVBlockState.UNHASHED
    assert entry.hash_count == 0
    assert index.snapshot().cache_removals == 1
    assert index.snapshot().cache_evictions == 1
    assert index.snapshot().unbacked_cache_evictions == 1
    index.check_consistency()


def test_actual_lru_eviction_audits_shared_prefix_loss() -> None:
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    index = KVResidencyIndex(
        pool.num_gpu_blocks,
        null_block_id=pool.null_block.block_id,
        now_ns=1_000_000_000,
    )
    pool.set_observer(index)
    block = pool.get_new_blocks(1)[0]
    block_hash = make_block_hash_with_group_id(BlockHash(b"s" * 32), 0)
    pool._insert_block_hash(block_hash, block, num_tokens=16)
    pool.free_blocks([block])
    for _ in range(2):
        pool.touch([block])
        pool.free_blocks([block])

    pool.get_new_blocks(2)

    stats = index.snapshot()
    assert stats.cache_evictions == 1
    assert stats.shared_cache_evictions == 1
    assert stats.unbacked_cache_evictions == 1


def test_reset_clears_residency_without_detaching_observer() -> None:
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    index = KVResidencyIndex(
        pool.num_gpu_blocks,
        null_block_id=pool.null_block.block_id,
        now_ns=1_000_000_000,
    )
    pool.set_observer(index)
    block = pool.get_new_blocks(1)[0]
    block_hash = make_block_hash_with_group_id(BlockHash(b"b" * 32), 0)
    pool._insert_block_hash(block_hash, block, num_tokens=16)
    pool.free_blocks([block])

    assert pool.reset_prefix_cache()
    stats = index.snapshot()
    assert stats.free_blocks == 2
    assert stats.shared_blocks == 0
    assert pool.observer is index
    index.check_consistency()


def test_factory_keeps_shadow_mode_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    monkeypatch.setenv("VLLM_AGENTRIX_KV_RESIDENCY_SHADOW", "0")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_PLACEMENT_SHADOW", "0")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_PROACTIVE_BACKUP", "0")
    assert create_kv_residency_index(pool) is None

    monkeypatch.setenv("VLLM_AGENTRIX_KV_PROACTIVE_BACKUP", "1")
    assert create_kv_residency_index(pool) is not None
    monkeypatch.setenv("VLLM_AGENTRIX_KV_PROACTIVE_BACKUP", "0")

    monkeypatch.setenv("VLLM_AGENTRIX_KV_PLACEMENT_SHADOW", "1")
    assert create_kv_residency_index(pool) is not None

    monkeypatch.setenv("VLLM_AGENTRIX_KV_RESIDENCY_SHADOW", "1")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_PLACEMENT_SHADOW", "0")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_WARM_SECONDS", "3")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_COOLING_SECONDS", "9")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_SHARED_REUSE_THRESHOLD", "4")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_AGING_BUDGET", "17")
    index = create_kv_residency_index(pool)

    assert index is not None
    assert index.warm_ns == 3_000_000_000
    assert index.cooling_ns == 9_000_000_000
    assert index.shared_reuse_threshold == 4
    assert index.aging_budget == 17
