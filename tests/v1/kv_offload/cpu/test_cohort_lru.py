# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadEvictionMetadata,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

_CTX = ReqContext(req_id="test")


def _key(value: int) -> OffloadKey:
    return make_offload_key(str(value).encode(), 0)


def _store(manager: CPUOffloadingManager, values: list[int]) -> None:
    keys = [_key(value) for value in values]
    result = manager.prepare_store(keys, _CTX)
    assert result is not None
    manager.complete_store(keys, _CTX)


def _metadata(
    *,
    lifecycle: int = 0,
    reuse: int = 0,
    fanout: int = 1,
    residency: int = 0,
    position: float = 1.0,
) -> OffloadEvictionMetadata:
    return OffloadEvictionMetadata(
        lifecycle_value=lifecycle,
        reuse_score=reuse,
        fanout=fanout,
        residency_value=residency,
        prefix_position=position,
    )


def test_cohort_lru_falls_back_to_lru_without_value_metadata() -> None:
    manager = CPUOffloadingManager(num_blocks=2, cache_policy="cohort_lru")
    _store(manager, [1, 2])
    manager.touch([_key(1)], _CTX)

    result = manager.prepare_store([_key(3)], _CTX)

    assert result is not None
    assert result.evicted_keys == [_key(2)]


def test_cohort_lru_retains_hot_shared_prefix_over_older_block() -> None:
    manager = CPUOffloadingManager(num_blocks=2, cache_policy="cohort_lru")
    _store(manager, [1, 2])
    manager.update_eviction_metadata(
        {
            _key(1): _metadata(
                lifecycle=3,
                reuse=1024,
                fanout=16,
                residency=4096,
                position=0.25,
            ),
            _key(2): _metadata(lifecycle=1),
        },
        replace=True,
    )

    result = manager.prepare_store([_key(3)], _CTX)

    assert result is not None
    assert result.evicted_keys == [_key(2)]
    assert manager.lookup(_key(1), _CTX) is LookupResult.HIT


def test_cohort_lru_prefers_secondary_backed_victim_within_lifecycle() -> None:
    manager = CPUOffloadingManager(num_blocks=2, cache_policy="cohort_lru")
    _store(manager, [1, 2])
    manager.update_eviction_metadata(
        {_key(1): _metadata(lifecycle=1), _key(2): _metadata(lifecycle=1)},
        replace=True,
    )
    manager.mark_secondary_backed([_key(2)])

    result = manager.prepare_store([_key(3)], _CTX)

    assert result is not None
    assert result.evicted_keys == [_key(2)]


def test_cohort_lru_preserves_prefix_continuity_before_backup_status() -> None:
    manager = CPUOffloadingManager(num_blocks=2, cache_policy="cohort_lru")
    _store(manager, [1, 2])
    manager.update_eviction_metadata(
        {
            _key(1): _metadata(lifecycle=1, position=0.1),
            _key(2): _metadata(lifecycle=1, position=0.9),
        },
        replace=True,
    )
    manager.mark_secondary_backed([_key(1)])

    result = manager.prepare_store([_key(3)], _CTX)

    assert result is not None
    assert result.evicted_keys == [_key(2)]


def test_cohort_lru_uses_reuse_and_prefix_position_before_lru() -> None:
    manager = CPUOffloadingManager(num_blocks=3, cache_policy="cohort_lru")
    _store(manager, [1, 2, 3])
    manager.update_eviction_metadata(
        {
            _key(1): _metadata(lifecycle=1, reuse=8, position=0.2),
            _key(2): _metadata(lifecycle=1, reuse=2, position=0.2),
            _key(3): _metadata(lifecycle=1, reuse=2, position=0.8),
        },
        replace=True,
    )

    result = manager.prepare_store([_key(4)], _CTX)

    assert result is not None
    assert result.evicted_keys == [_key(3)]


def test_cohort_lru_never_evicts_inflight_block() -> None:
    manager = CPUOffloadingManager(num_blocks=2, cache_policy="cohort_lru")
    _store(manager, [1, 2])
    manager.prepare_load([_key(2)], _CTX)
    manager.update_eviction_metadata(
        {
            _key(1): _metadata(lifecycle=3, reuse=1024, fanout=16),
            _key(2): _metadata(lifecycle=0),
        },
        replace=True,
    )

    result = manager.prepare_store([_key(3)], _CTX)

    assert result is not None
    assert result.evicted_keys == [_key(1)]
