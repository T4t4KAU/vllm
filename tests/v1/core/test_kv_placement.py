# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.kv_placement import (
    KVPlacementAction,
    KVPlacementPlanner,
    create_kv_placement_planner,
)
from vllm.v1.core.kv_residency import (
    KVBlockState,
    KVResidencyIndex,
    KVResidencyTier,
)

pytestmark = pytest.mark.cpu_test


def _index(num_blocks: int = 8) -> KVResidencyIndex:
    return KVResidencyIndex(
        num_blocks,
        null_block_id=0,
        warm_seconds=10.0,
        cooling_seconds=100.0,
        now_ns=1_000_000_000,
    )


def _cache_and_release(index: KVResidencyIndex, block_id: int) -> int:
    generation = index.on_allocated(block_id, 1)
    index.on_cache_inserted(block_id)
    index.on_released(block_id, 0)
    return generation


def _cache_pool_block(pool: BlockPool, expected_block_id: int) -> KVCacheBlock:
    block = pool.get_new_blocks(1)[0]
    assert block.block_id == expected_block_id
    block_hash = make_block_hash_with_group_id(
        BlockHash(str(block.block_id).encode()),
        0,
    )
    pool._insert_block_hash(block_hash, block, num_tokens=1)
    pool.free_blocks([block])
    return block


def test_plan_protects_shared_prefixes_and_selects_tier_actions() -> None:
    index = _index()
    _cache_and_release(index, 1)

    _cache_and_release(index, 2)
    index.on_cache_hit(2, 1)
    index.on_released(2, 0)

    backed_generation = _cache_and_release(index, 3)
    assert index.start_backup(3, backed_generation, KVResidencyTier.CPU, 30)
    assert index.finish_backup(
        3,
        backed_generation,
        KVResidencyTier.CPU,
        30,
        success=True,
    )

    _cache_and_release(index, 4)
    for _ in range(2):
        index.on_cache_hit(4, 1)
        index.on_released(4, 0)

    index.advance(now_ns=101_000_000_000)
    index.advance(now_ns=101_000_000_000)

    planner = KVPlacementPlanner(scan_budget=8)
    plan = planner.plan(index, requested_blocks=4)

    actions = {decision.block_id: decision.action for decision in plan.decisions}
    assert actions == {
        1: KVPlacementAction.DISCARD,
        2: KVPlacementAction.BACKUP_TO_CPU,
        3: KVPlacementAction.RELEASE_GPU,
    }
    assert 4 not in actions
    assert plan.protected_shared_blocks == 1
    assert plan.protected_shortfall_blocks == 1
    assert plan.unresolved_blocks == 1
    assert not plan.scan_budget_exhausted

    stats = planner.snapshot()
    assert stats.planned_blocks == 3
    assert stats.backup_to_cpu_blocks == 1
    assert stats.release_gpu_blocks == 1
    assert stats.discarded_blocks == 1
    assert stats.protected_shortfall_blocks == 1


def test_plan_scan_and_output_are_bounded() -> None:
    index = _index(num_blocks=101)
    generations = {}
    for block_id in range(1, 101):
        generations[block_id] = _cache_and_release(index, block_id)
    index.advance(now_ns=101_000_000_000)
    index.advance(now_ns=101_000_000_000)
    assert index.start_backup(1, generations[1], KVResidencyTier.CPU, 1)

    planner = KVPlacementPlanner(scan_budget=7)
    plan = planner.plan(index, requested_blocks=100)

    assert plan.scanned_blocks == 7
    assert len(plan.decisions) == 6
    assert plan.deferred_blocks == 1
    assert plan.unresolved_blocks == 94
    assert plan.scan_budget_exhausted
    assert planner.snapshot().budget_exhaustions == 1

    second_plan = planner.plan(index, requested_blocks=100)
    assert second_plan.scanned_blocks == 0
    assert second_plan.scan_budget_exhausted

    index.advance(now_ns=102_000_000_000)
    next_step_plan = planner.plan(index, requested_blocks=100)
    assert next_step_plan.scanned_blocks == 7


def test_allocation_plan_only_covers_cached_recycling() -> None:
    index = _index()
    planner = KVPlacementPlanner()

    assert planner.plan_for_allocation(index, num_blocks_to_allocate=7) is None

    _cache_and_release(index, 1)
    plan = planner.plan_for_allocation(index, num_blocks_to_allocate=7)

    assert plan is not None
    assert plan.requested_blocks == 1
    assert [decision.block_id for decision in plan.decisions] == [1]


def test_active_plan_applies_tier_priority_and_protects_shared_prefix() -> None:
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=1)
    index = _index(num_blocks=5)
    pool.set_observer(index)
    shared = _cache_pool_block(pool, 1)
    fallback = _cache_pool_block(pool, 2)
    backed = _cache_pool_block(pool, 3)
    for _ in range(2):
        pool.touch([shared])
        pool.free_blocks([shared])
    generation = index.placement_metadata(backed.block_id)[0]
    assert index.start_backup(backed.block_id, generation, KVResidencyTier.CPU, 1)
    assert index.finish_backup(
        backed.block_id,
        generation,
        KVResidencyTier.CPU,
        1,
        success=True,
    )

    planner = KVPlacementPlanner(scan_budget=8, active=True)
    plan = planner.plan_for_allocation(index, num_blocks_to_allocate=3)

    assert plan is not None
    assert [decision.block_id for decision in plan.decisions] == [3, 2]
    assert [decision.action for decision in plan.decisions] == [
        KVPlacementAction.RELEASE_GPU,
        KVPlacementAction.DISCARD_UNBACKED,
    ]
    assert shared.block_id not in {decision.block_id for decision in plan.decisions}
    assert planner.apply(plan, index, pool)

    allocated = pool.get_new_blocks(3)
    assert {block.block_id for block in allocated} == {2, 3, 4}
    assert shared.block_hash is not None
    assert fallback.block_hash is None
    assert backed.block_hash is None
    residency_stats = index.snapshot()
    assert residency_stats.cache_evictions == 2
    assert residency_stats.shared_cache_evictions == 0
    assert residency_stats.unbacked_cache_evictions == 1
    stats = planner.snapshot()
    assert stats.applied_plans == 1
    assert stats.applied_blocks == 2
    assert stats.unbacked_discarded_blocks == 1


def test_active_plan_prefers_cpu_over_remote_backup() -> None:
    index = _index(num_blocks=4)
    remote_generation = _cache_and_release(index, 1)
    cpu_generation = _cache_and_release(index, 2)
    assert index.start_backup(1, remote_generation, KVResidencyTier.REMOTE, 1)
    assert index.finish_backup(
        1,
        remote_generation,
        KVResidencyTier.REMOTE,
        1,
        success=True,
    )
    assert index.start_backup(2, cpu_generation, KVResidencyTier.CPU, 2)
    assert index.finish_backup(
        2,
        cpu_generation,
        KVResidencyTier.CPU,
        2,
        success=True,
    )
    index.advance(now_ns=101_000_000_000)
    index.advance(now_ns=101_000_000_000)

    plan = KVPlacementPlanner(scan_budget=3, active=True).plan(index, 2)

    assert [decision.block_id for decision in plan.decisions] == [2, 1]
    assert all(
        decision.action == KVPlacementAction.RELEASE_GPU for decision in plan.decisions
    )


def test_active_plan_prefers_unreused_warm_block_over_reused_cold_block() -> None:
    index = _index(num_blocks=4)
    _cache_and_release(index, 1)
    index.on_cache_hit(1, 1)
    index.on_released(1, 0)
    index.advance(now_ns=101_000_000_000)
    index.advance(now_ns=101_000_000_000)
    _cache_and_release(index, 2)

    plan = KVPlacementPlanner(scan_budget=3, active=True).plan(index, 1)

    assert plan.decisions[0].block_id == 2
    assert plan.decisions[0].state == KVBlockState.WARM


def test_active_plan_never_reclaims_shared_prefixes() -> None:
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=1)
    index = _index(num_blocks=3)
    pool.set_observer(index)
    blocks = [_cache_pool_block(pool, block_id) for block_id in range(1, 3)]
    for block in blocks:
        for _ in range(2):
            pool.touch([block])
            pool.free_blocks([block])

    planner = KVPlacementPlanner(scan_budget=2, active=True)

    assert not planner.plan_and_apply_for_allocation(index, 1, pool)
    assert all(block.block_hash is not None for block in blocks)
    assert index.snapshot().shared_cache_evictions == 0


def test_active_plan_excludes_current_cache_hits() -> None:
    index = _index(num_blocks=5)
    _cache_and_release(index, 1)
    generation = _cache_and_release(index, 2)
    assert index.start_backup(2, generation, KVResidencyTier.CPU, 1)
    assert index.finish_backup(
        2,
        generation,
        KVResidencyTier.CPU,
        1,
        success=True,
    )
    planner = KVPlacementPlanner(scan_budget=4, active=True)

    plan = planner.plan(index, requested_blocks=1, excluded_block_ids={2})

    assert [decision.block_id for decision in plan.decisions] == [1]
    assert plan.excluded_blocks == 1


def test_active_plan_scales_with_allocation_above_scan_budget() -> None:
    index = _index(num_blocks=10)
    for block_id in range(1, 10):
        _cache_and_release(index, block_id)
    planner = KVPlacementPlanner(scan_budget=2, active=True)

    plan = planner.plan(index, requested_blocks=5)

    assert plan.scanned_blocks == 5
    assert len(plan.decisions) == 5
    assert not plan.scan_budget_exhausted


def test_active_hot_path_preserves_diagnostic_plan() -> None:
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=1)
    index = _index(num_blocks=4)
    pool.set_observer(index)
    first = _cache_pool_block(pool, 1)
    _cache_pool_block(pool, 2)
    planner = KVPlacementPlanner(scan_budget=2, active=True)

    assert planner.plan_and_apply(index, requested_blocks=1, block_pool=pool)

    plan = planner.last_plan
    assert plan is not None
    assert plan.requested_blocks == 1
    assert [decision.block_id for decision in plan.decisions] == [first.block_id]
    assert plan.decisions[0].action == KVPlacementAction.DISCARD_UNBACKED
    assert planner.last_plan is plan
    stats = planner.snapshot()
    assert stats.applied_plans == 1
    assert stats.applied_blocks == 1

    for _ in range(2):
        pool.touch([first])
        pool.free_blocks([first])
    assert planner.plan_and_apply(index, requested_blocks=1, block_pool=pool)
    next_plan = planner.last_plan
    assert next_plan is not None
    assert next_plan.decisions[0].block_id != first.block_id
    assert plan.decisions[0].block_id == first.block_id


def test_active_hot_path_rejects_shadow_planner() -> None:
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=1)
    index = _index(num_blocks=2)
    planner = KVPlacementPlanner(scan_budget=1)

    with pytest.raises(RuntimeError, match="requires active placement"):
        planner.plan_and_apply(index, requested_blocks=1, block_pool=pool)


def test_prioritize_eviction_blocks_is_atomic() -> None:
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=1)
    _cache_pool_block(pool, 1)
    _cache_pool_block(pool, 2)
    original_order = pool.free_block_queue.get_all_free_blocks()

    assert not pool.prioritize_eviction_blocks([1, pool.null_block.block_id])

    assert pool.free_block_queue.get_all_free_blocks() == original_order


def test_prioritize_eviction_blocks_recovers_after_duplicate_input() -> None:
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=1)
    _cache_pool_block(pool, 1)
    _cache_pool_block(pool, 2)

    assert not pool.prioritize_eviction_blocks([1, 1])
    assert pool.prioritize_eviction_blocks([2, 1])

    assert [block.block_id for block in pool.get_new_blocks(2)] == [2, 1]


def test_active_plan_revalidates_generation_before_queue_mutation() -> None:
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=1)
    index = _index(num_blocks=3)
    pool.set_observer(index)
    block = _cache_pool_block(pool, 1)
    planner = KVPlacementPlanner(scan_budget=2, active=True)
    plan = planner.plan(index, requested_blocks=1)
    original_order = pool.free_block_queue.get_all_free_blocks()
    index.on_cache_removed(block.block_id, 1)
    index.on_cache_inserted(block.block_id)

    assert not planner.apply(plan, index, pool)

    assert pool.free_block_queue.get_all_free_blocks() == original_order
    assert planner.snapshot().apply_failures == 1


def test_active_plan_defers_inflight_backup_without_queue_mutation() -> None:
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=1)
    index = _index(num_blocks=3)
    pool.set_observer(index)
    block = _cache_pool_block(pool, 1)
    generation = index.placement_metadata(block.block_id)[0]
    assert index.start_backup(block.block_id, generation, KVResidencyTier.CPU, 1)
    planner = KVPlacementPlanner(scan_budget=2, active=True)
    original_order = pool.free_block_queue.get_all_free_blocks()

    plan = planner.plan(index, requested_blocks=1)

    assert plan.decisions == ()
    assert plan.deferred_blocks == 1
    assert not planner.apply(plan, index, pool)
    assert pool.free_block_queue.get_all_free_blocks() == original_order


def test_planner_factory_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_AGENTRIX_KV_PLACEMENT_SHADOW", "0")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_PLACEMENT_ACTIVE", "0")
    assert create_kv_placement_planner() is None

    monkeypatch.setenv("VLLM_AGENTRIX_KV_PLACEMENT_SHADOW", "1")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_PLACEMENT_SCAN_BUDGET", "17")
    planner = create_kv_placement_planner()

    assert planner is not None
    assert planner.scan_budget == 17
    assert not planner.active

    monkeypatch.setenv("VLLM_AGENTRIX_KV_PLACEMENT_SHADOW", "0")
    monkeypatch.setenv("VLLM_AGENTRIX_KV_PLACEMENT_ACTIVE", "1")
    planner = create_kv_placement_planner()

    assert planner is not None
    assert planner.active
