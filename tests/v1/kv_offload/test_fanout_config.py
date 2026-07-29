# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    parse_fanout_layerwise_load,
    resolve_fanout_layerwise_load,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    FanoutCandidateObservation,
    FanoutLifecycle,
    OffloadingConnectorScheduler,
    _is_hot_shared_prefix,
    _make_fanout_eviction_metadata,
    _resolve_fanout_chunk_blocks,
    _resolve_fanout_hot_prefix_config,
    _resolve_fanout_pressure_config,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_offload.base import OffloadEvictionMetadata
from vllm.v1.kv_offload.fanout_planner import (
    FanoutBlock,
    FanoutLifecycleState,
    FanoutPressureLevel,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("auto", None),
        (None, None),
    ],
)
def test_parse_fanout_layerwise_load(value: object, expected: bool | None) -> None:
    assert parse_fanout_layerwise_load(value) is expected


def test_resolve_fanout_layerwise_load_keeps_full_graph() -> None:
    assert not resolve_fanout_layerwise_load(
        "auto",
        fanout_offload=True,
        has_full_cudagraphs=True,
        estimated_load_bytes=1 << 40,
        threshold_bytes=1,
    )


def test_resolve_fanout_layerwise_load_uses_threshold_without_full_graph() -> None:
    assert resolve_fanout_layerwise_load(
        "auto",
        fanout_offload=True,
        has_full_cudagraphs=False,
        estimated_load_bytes=1024,
        threshold_bytes=512,
    )
    assert not resolve_fanout_layerwise_load(
        "auto",
        fanout_offload=True,
        has_full_cudagraphs=False,
        estimated_load_bytes=256,
        threshold_bytes=512,
    )


def test_parse_fanout_layerwise_load_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="fanout_layerwise_load"):
        parse_fanout_layerwise_load("sometimes")


def test_resolve_fanout_chunk_blocks_defaults_from_tokens() -> None:
    assert (
        _resolve_fanout_chunk_blocks(
            {},
            min_offloaded_block_size=16,
            fanout_budget_blocks=64,
        )
        == 64
    )


def test_resolve_fanout_chunk_blocks_honors_explicit_blocks() -> None:
    assert (
        _resolve_fanout_chunk_blocks(
            {"fanout_chunk_blocks": 128},
            min_offloaded_block_size=16,
            fanout_budget_blocks=256,
        )
        == 128
    )


def test_resolve_fanout_chunk_blocks_rejects_explicit_blocks_over_budget() -> None:
    with pytest.raises(ValueError, match="fanout_chunk_blocks"):
        _resolve_fanout_chunk_blocks(
            {"fanout_chunk_blocks": 128},
            min_offloaded_block_size=16,
            fanout_budget_blocks=64,
        )


def test_resolve_fanout_chunk_blocks_rejects_invalid_tokens() -> None:
    with pytest.raises(ValueError, match="fanout_chunk_tokens"):
        _resolve_fanout_chunk_blocks(
            {"fanout_chunk_tokens": 0},
            min_offloaded_block_size=16,
            fanout_budget_blocks=64,
        )


def test_resolve_fanout_hot_prefix_config_defaults_above_min_fanout() -> None:
    assert _resolve_fanout_hot_prefix_config({}, fanout_min_fanout=2) == (
        4,
        1.0,
        128,
        4,
        16,
        True,
    )
    assert _resolve_fanout_hot_prefix_config({}, fanout_min_fanout=8) == (
        9,
        1.0,
        128,
        4,
        16,
        True,
    )


def test_resolve_fanout_hot_prefix_config_honors_explicit_values() -> None:
    assert _resolve_fanout_hot_prefix_config(
        {
            "fanout_hot_prefix_min_fanout": 6,
            "fanout_hot_prefix_max_position": 0.5,
            "fanout_hot_prefix_min_reuse_blocks": 256,
            "fanout_hot_prefix_min_residency_steps": 8,
            "fanout_hot_prefix_cooldown_steps": 32,
            "fanout_allow_hot_prefix_backup": True,
        },
        fanout_min_fanout=2,
    ) == (6, 0.5, 256, 8, 32, True)


def test_resolve_fanout_hot_prefix_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="fanout_hot_prefix_min_fanout"):
        _resolve_fanout_hot_prefix_config(
            {"fanout_hot_prefix_min_fanout": -1},
            fanout_min_fanout=2,
        )
    with pytest.raises(ValueError, match="fanout_hot_prefix_max_position"):
        _resolve_fanout_hot_prefix_config(
            {"fanout_hot_prefix_max_position": 0},
            fanout_min_fanout=2,
        )
    with pytest.raises(ValueError, match="fanout_hot_prefix_min_reuse_blocks"):
        _resolve_fanout_hot_prefix_config(
            {"fanout_hot_prefix_min_reuse_blocks": -1},
            fanout_min_fanout=2,
        )
    with pytest.raises(ValueError, match="fanout_hot_prefix_min_residency_steps"):
        _resolve_fanout_hot_prefix_config(
            {"fanout_hot_prefix_min_residency_steps": -1},
            fanout_min_fanout=2,
        )
    with pytest.raises(ValueError, match="fanout_hot_prefix_cooldown_steps"):
        _resolve_fanout_hot_prefix_config(
            {"fanout_hot_prefix_cooldown_steps": -1},
            fanout_min_fanout=2,
        )


def test_resolve_fanout_pressure_config() -> None:
    assert _resolve_fanout_pressure_config({}) == (0.90, 0.97, 0.85, 0.93)
    assert _resolve_fanout_pressure_config(
        {
            "fanout_high_pressure_threshold": 0.75,
            "fanout_critical_pressure_threshold": 0.9,
        }
    ) == (0.75, 0.9, 0.70, 0.86)


@pytest.mark.parametrize(
    "config",
    [
        {"fanout_high_pressure_threshold": -0.1},
        {"fanout_critical_pressure_threshold": 1.1},
        {
            "fanout_high_pressure_threshold": 0.95,
            "fanout_critical_pressure_threshold": 0.90,
        },
        {"fanout_high_pressure_exit_threshold": 0.91},
        {"fanout_critical_pressure_exit_threshold": 0.89},
        {"fanout_critical_pressure_exit_threshold": 0.98},
    ],
)
def test_resolve_fanout_pressure_config_rejects_invalid_values(
    config: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="pressure"):
        _resolve_fanout_pressure_config(config)


def test_is_hot_shared_prefix_can_be_disabled() -> None:
    assert not _is_hot_shared_prefix(
        fanout=8,
        prefix_position=0.25,
        min_fanout=0,
        max_prefix_position=1.0,
    )


def test_is_hot_shared_prefix_uses_fanout_and_position_thresholds() -> None:
    assert _is_hot_shared_prefix(
        fanout=8,
        prefix_position=0.25,
        min_fanout=4,
        max_prefix_position=0.5,
        reuse_score=256,
        min_reuse_score=128,
    )
    assert not _is_hot_shared_prefix(
        fanout=3,
        prefix_position=0.25,
        min_fanout=4,
        max_prefix_position=0.5,
        reuse_score=256,
        min_reuse_score=128,
    )
    assert not _is_hot_shared_prefix(
        fanout=8,
        prefix_position=0.75,
        min_fanout=4,
        max_prefix_position=0.5,
        reuse_score=256,
        min_reuse_score=128,
    )
    assert not _is_hot_shared_prefix(
        fanout=8,
        prefix_position=0.25,
        min_fanout=4,
        max_prefix_position=0.5,
        reuse_score=64,
        min_reuse_score=128,
    )


def test_fanout_lifecycle_cools_before_becoming_cold() -> None:
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(
        fanout_hot_prefix_min_residency_steps=4,
        fanout_hot_prefix_cooldown_steps=16,
    )
    scheduler._fanout_lifecycle = {}
    key = b"shared-key"

    scheduler._fanout_step = 1
    state, lifecycle = scheduler._update_fanout_lifecycle(
        key,
        fanout=8,
        reuse_score=512,
        base_hot=True,
    )
    assert state is FanoutLifecycleState.HOT
    assert lifecycle.historical_max_fanout == 8

    scheduler._fanout_step = 2
    state, _ = scheduler._update_fanout_lifecycle(
        key,
        fanout=1,
        reuse_score=0,
        base_hot=False,
    )
    assert state is FanoutLifecycleState.COOLING

    scheduler._fanout_step = 18
    state, lifecycle = scheduler._update_fanout_lifecycle(
        key,
        fanout=1,
        reuse_score=0,
        base_hot=False,
    )
    assert state is FanoutLifecycleState.COLD
    assert lifecycle.historical_max_reuse_score == 512


def test_fanout_candidate_observation_merge_is_order_independent() -> None:
    short = FanoutCandidateObservation(
        request_id="short",
        group_idx=0,
        logical_block_idx=3,
        physical_block_id=7,
        offload_key=b"key",
        fanout=2,
        prefix_position=0.5,
        reuse_score=64,
        is_active_tail=True,
    )
    long = FanoutCandidateObservation(
        request_id="long",
        group_idx=0,
        logical_block_idx=3,
        physical_block_id=9,
        offload_key=b"key",
        fanout=8,
        prefix_position=0.125,
        reuse_score=512,
        is_active_tail=False,
    )

    short.merge(long)

    assert short.fanout == 8
    assert short.prefix_position == 0.125
    assert short.reuse_score == 512
    assert short.is_active_tail
    assert short.physical_block_ids == {7, 9}


def test_fanout_candidate_builds_cpu_eviction_metadata() -> None:
    observation = FanoutCandidateObservation(
        request_id="request",
        group_idx=0,
        logical_block_idx=3,
        physical_block_id=7,
        offload_key=b"key",
        fanout=8,
        prefix_position=0.25,
        reuse_score=448,
        is_active_tail=False,
    )
    candidate = FanoutBlock(
        request_id="request",
        group_idx=0,
        logical_block_idx=3,
        physical_block_id=7,
        offload_key=b"key",
        fanout=8,
        prefix_position=0.25,
        last_access_time=4,
        lifecycle_state=FanoutLifecycleState.HOT,
        historical_max_fanout=16,
        residency_value=2048,
    )

    metadata = _make_fanout_eviction_metadata(candidate, observation)

    assert metadata.lifecycle_value == FanoutLifecycleState.HOT.value
    assert metadata.reuse_score == 448
    assert metadata.fanout == 16
    assert metadata.residency_value == 2048
    assert metadata.prefix_position == 0.25


def test_gpu_lifecycle_survives_request_release_and_cools() -> None:
    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=True,
        hash_block_size=16,
    )
    block = pool.blocks[1]
    block_hash = make_block_hash_with_group_id(BlockHash(b"shared"), 0)
    block.set_block_hash(block_hash)
    pool.cached_block_hash_to_block.insert(block_hash, block)

    offload_key = b"shared-key"
    observation = FanoutCandidateObservation(
        request_id="request",
        group_idx=0,
        logical_block_idx=0,
        physical_block_id=block.block_id,
        offload_key=offload_key,
        fanout=8,
        prefix_position=0.25,
        reuse_score=256,
        is_active_tail=False,
    )
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler._gpu_block_pool = pool
    scheduler._fanout_step = 1
    scheduler.config = SimpleNamespace(fanout_hot_prefix_cooldown_steps=2)
    scheduler._fanout_lifecycle = {
        offload_key: FanoutLifecycle(
            historical_max_fanout=8,
            historical_max_reuse_score=256,
            hot_since_step=1,
            min_resident_until_step=1,
            last_hot_step=1,
            last_observed_step=1,
            lifecycle_state=FanoutLifecycleState.HOT,
        )
    }
    scheduler._fanout_admitted_locations = {}
    scheduler._req_status = {}
    scheduler._gpu_lifecycle_locations = {}
    scheduler._gpu_lifecycle_metadata = {}

    scheduler._publish_gpu_eviction_metadata(
        {
            offload_key: OffloadEvictionMetadata(
                lifecycle_value=FanoutLifecycleState.HOT.value,
                reuse_score=256,
                fanout=8,
                residency_value=1024,
                prefix_position=0.25,
            )
        },
        {offload_key: observation},
    )
    assert (
        pool._gpu_eviction_metadata[block.block_id].lifecycle_value
        == FanoutLifecycleState.HOT.value
    )

    scheduler._fanout_step = 2
    scheduler._publish_gpu_eviction_metadata({}, {})
    assert (
        pool._gpu_eviction_metadata[block.block_id].lifecycle_value
        == FanoutLifecycleState.COOLING.value
    )

    scheduler._fanout_step = 4
    scheduler._publish_gpu_eviction_metadata({}, {})
    assert (
        pool._gpu_eviction_metadata[block.block_id].lifecycle_value
        == FanoutLifecycleState.COLD.value
    )

    pool.evict_blocks({block.block_id})
    scheduler._fanout_step = 5
    scheduler._publish_gpu_eviction_metadata({}, {})
    assert offload_key not in scheduler._gpu_lifecycle_locations
    assert block.block_id not in pool._gpu_eviction_metadata


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (0.50, FanoutPressureLevel.NORMAL),
        (0.90, FanoutPressureLevel.HIGH),
        (0.97, FanoutPressureLevel.CRITICAL),
    ],
)
def test_fanout_pressure_level_uses_real_kv_cache_usage(
    usage: float,
    expected: FanoutPressureLevel,
) -> None:
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(
        fanout_high_pressure_threshold=0.90,
        fanout_critical_pressure_threshold=0.97,
        fanout_high_pressure_exit_threshold=0.85,
        fanout_critical_pressure_exit_threshold=0.93,
    )

    assert scheduler._fanout_pressure_level(usage) is expected


def test_fanout_pressure_level_uses_hysteresis() -> None:
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.config = SimpleNamespace(
        fanout_high_pressure_threshold=0.90,
        fanout_critical_pressure_threshold=0.97,
        fanout_high_pressure_exit_threshold=0.85,
        fanout_critical_pressure_exit_threshold=0.93,
    )
    scheduler._fanout_pressure_state = FanoutPressureLevel.NORMAL

    assert scheduler._fanout_pressure_level(0.91) is FanoutPressureLevel.HIGH
    assert scheduler._fanout_pressure_level(0.88) is FanoutPressureLevel.HIGH
    assert scheduler._fanout_pressure_level(0.98) is FanoutPressureLevel.CRITICAL
    assert scheduler._fanout_pressure_level(0.95) is FanoutPressureLevel.CRITICAL
    assert scheduler._fanout_pressure_level(0.92) is FanoutPressureLevel.HIGH
    assert scheduler._fanout_pressure_level(0.84) is FanoutPressureLevel.NORMAL
