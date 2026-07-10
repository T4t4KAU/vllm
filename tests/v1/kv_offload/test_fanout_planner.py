# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.kv_offload.fanout_planner import (
    FanoutBlock,
    FanoutChunkPlanner,
    FanoutChunkState,
    FanoutLifecycleState,
    FanoutPressureLevel,
)


def make_block(
    logical_idx: int,
    *,
    request_id: str = "request",
    fanout: int = 1,
    last_access_time: float = 0,
    state: FanoutChunkState = FanoutChunkState.GPU_ONLY,
    is_sealed: bool = True,
    is_active_tail: bool = False,
    is_hot_shared_prefix: bool = False,
    lifecycle_state: FanoutLifecycleState | None = None,
    residency_value: int = 0,
    in_flight: bool = False,
) -> FanoutBlock:
    if lifecycle_state is None:
        lifecycle_state = (
            FanoutLifecycleState.HOT
            if is_hot_shared_prefix
            else FanoutLifecycleState.COLD
        )
    return FanoutBlock(
        request_id=request_id,
        group_idx=0,
        logical_block_idx=logical_idx,
        physical_block_id=logical_idx + 1,
        offload_key=(bytes([logical_idx]), 0),
        fanout=fanout,
        prefix_position=(logical_idx + 1) / 16,
        last_access_time=last_access_time,
        state=state,
        is_sealed=is_sealed,
        is_active_tail=is_active_tail,
        lifecycle_state=lifecycle_state,
        historical_max_fanout=fanout,
        residency_value=residency_value,
        in_flight=in_flight,
    )


def test_build_chunks_splits_on_fanout_and_gaps() -> None:
    planner = FanoutChunkPlanner()
    chunks = planner.build_chunks(
        [
            make_block(0, fanout=8),
            make_block(1, fanout=8),
            make_block(3, fanout=1),
            make_block(4, fanout=1),
            make_block(5, fanout=2),
        ]
    )

    assert [
        (chunk.logical_start_block, chunk.logical_end_block, chunk.fanout)
        for chunk in chunks
    ] == [(0, 2, 8), (3, 5, 1), (5, 6, 2)]


def test_build_chunks_splits_on_hot_shared_prefix_boundary() -> None:
    planner = FanoutChunkPlanner()
    chunks = planner.build_chunks(
        [
            make_block(0, fanout=8, is_hot_shared_prefix=True),
            make_block(1, fanout=8, is_hot_shared_prefix=False),
        ]
    )

    assert [
        (
            chunk.logical_start_block,
            chunk.logical_end_block,
            chunk.is_hot_shared_prefix,
        )
        for chunk in chunks
    ] == [(0, 1, True), (1, 2, False)]


def test_select_prioritizes_shared_work_position_fanout_and_age() -> None:
    planner = FanoutChunkPlanner()
    blocks = [
        make_block(0, request_id="valuable", fanout=8, residency_value=128),
        make_block(1, request_id="valuable", fanout=8, residency_value=128),
        make_block(8, request_id="newer", fanout=1, last_access_time=20),
        make_block(12, request_id="older", fanout=1, last_access_time=10),
        make_block(13, request_id="older", fanout=1, last_access_time=10),
    ]

    plan = planner.select(blocks, budget_blocks=2)

    assert [chunk.request_id for chunk in plan.chunks] == ["valuable"]
    assert plan.num_blocks == 2


def test_select_prefers_more_shared_work() -> None:
    planner = FanoutChunkPlanner()
    blocks = [
        make_block(0, request_id="tiny_hot", fanout=8),
        make_block(8, request_id="long_shared", fanout=4),
        make_block(9, request_id="long_shared", fanout=4),
        make_block(10, request_id="long_shared", fanout=4),
    ]

    plan = planner.select(blocks, budget_blocks=3)

    assert [chunk.request_id for chunk in plan.chunks] == ["long_shared"]
    assert plan.num_blocks == 3


def test_select_defers_hot_shared_prefix_under_cpu_budget_pressure() -> None:
    planner = FanoutChunkPlanner()
    blocks = [
        make_block(0, request_id="hot_shared", fanout=8, is_hot_shared_prefix=True),
        make_block(1, request_id="hot_shared", fanout=8, is_hot_shared_prefix=True),
        make_block(8, request_id="cooler_shared", fanout=2),
        make_block(9, request_id="cooler_shared", fanout=2),
    ]

    plan = planner.select(blocks, budget_blocks=2)

    assert [chunk.request_id for chunk in plan.chunks] == ["cooler_shared"]
    assert plan.num_blocks == 2
    assert [chunk.request_id for chunk in plan.protected_hot_shared_chunks] == [
        "hot_shared"
    ]
    assert plan.num_protected_hot_shared_blocks == 2
    assert plan.protected_hot_shared_keys == (
        (bytes([0]), 0),
        (bytes([1]), 0),
    )


def test_select_skips_hot_shared_prefix_by_default() -> None:
    planner = FanoutChunkPlanner()
    blocks = [
        make_block(0, request_id="hot_shared", fanout=8, is_hot_shared_prefix=True),
        make_block(1, request_id="hot_shared", fanout=8, is_hot_shared_prefix=True),
    ]

    plan = planner.select(blocks, budget_blocks=2)

    assert plan.chunks == ()
    assert plan.num_blocks == 0
    assert [chunk.request_id for chunk in plan.protected_hot_shared_chunks] == [
        "hot_shared"
    ]
    assert plan.num_protected_hot_shared_blocks == 2


def test_select_can_backup_hot_shared_prefix_when_explicitly_allowed() -> None:
    planner = FanoutChunkPlanner(allow_hot_shared_prefix_backup=True)
    blocks = [
        make_block(0, request_id="hot_shared", fanout=8, is_hot_shared_prefix=True),
        make_block(1, request_id="hot_shared", fanout=8, is_hot_shared_prefix=True),
    ]

    plan = planner.select(
        blocks,
        budget_blocks=2,
        pressure_level=FanoutPressureLevel.CRITICAL,
    )

    assert [chunk.request_id for chunk in plan.chunks] == ["hot_shared"]
    assert plan.num_blocks == 2
    assert plan.protected_hot_shared_chunks == ()
    assert plan.num_protected_hot_shared_blocks == 0


def test_select_only_releases_cooling_prefix_under_pressure() -> None:
    planner = FanoutChunkPlanner()
    blocks = [
        make_block(
            0,
            request_id="cooling",
            fanout=4,
            lifecycle_state=FanoutLifecycleState.COOLING,
        ),
        make_block(
            1,
            request_id="cooling",
            fanout=4,
            lifecycle_state=FanoutLifecycleState.COOLING,
        ),
    ]

    assert planner.select(blocks, budget_blocks=2).chunks == ()
    high_pressure_plan = planner.select(
        blocks,
        budget_blocks=2,
        pressure_level=FanoutPressureLevel.HIGH,
    )

    assert [chunk.request_id for chunk in high_pressure_plan.chunks] == ["cooling"]


def test_select_defers_hot_prefix_until_critical_pressure() -> None:
    planner = FanoutChunkPlanner(allow_hot_shared_prefix_backup=True)
    blocks = [
        make_block(0, fanout=8, is_hot_shared_prefix=True),
        make_block(1, fanout=8, is_hot_shared_prefix=True),
    ]

    high_pressure_plan = planner.select(
        blocks,
        budget_blocks=2,
        pressure_level=FanoutPressureLevel.HIGH,
    )
    critical_pressure_plan = planner.select(
        blocks,
        budget_blocks=2,
        pressure_level=FanoutPressureLevel.CRITICAL,
    )

    assert high_pressure_plan.chunks == ()
    assert [chunk.request_id for chunk in critical_pressure_plan.chunks] == ["request"]


def test_select_reports_protected_hot_shared_prefix_with_zero_store_budget() -> None:
    planner = FanoutChunkPlanner()
    blocks = [
        make_block(0, request_id="hot_shared", fanout=8, is_hot_shared_prefix=True),
        make_block(1, request_id="hot_shared", fanout=8, is_hot_shared_prefix=True),
    ]

    plan = planner.select(blocks, budget_blocks=0)

    assert plan.chunks == ()
    assert plan.num_blocks == 0
    assert [chunk.request_id for chunk in plan.protected_hot_shared_chunks] == [
        "hot_shared"
    ]
    assert plan.num_protected_hot_shared_blocks == 2


def test_select_uses_position_fanout_and_age_as_tie_breakers() -> None:
    planner = FanoutChunkPlanner()
    blocks = [
        make_block(0, request_id="short_early", fanout=4, last_access_time=1),
        make_block(8, request_id="newer_late", fanout=4, last_access_time=20),
        make_block(12, request_id="older_late", fanout=2, last_access_time=10),
        make_block(13, request_id="older_late", fanout=2, last_access_time=10),
    ]

    plan = planner.select(blocks, budget_blocks=2)

    assert [chunk.request_id for chunk in plan.chunks] == [
        "newer_late",
        "short_early",
    ]
    assert plan.num_blocks == 2


def test_select_respects_budget() -> None:
    planner = FanoutChunkPlanner(max_blocks_per_chunk=3)
    blocks = [make_block(idx, fanout=8) for idx in range(3)]

    plan = planner.select(blocks, budget_blocks=2)

    assert plan.chunks == ()
    assert plan.num_blocks == 0


def test_min_fanout_filters_private_suffix_chunks() -> None:
    planner = FanoutChunkPlanner(min_fanout=2)

    chunks = planner.build_chunks(
        [
            make_block(0, request_id="shared", fanout=8),
            make_block(1, request_id="shared", fanout=8),
            make_block(8, request_id="private", fanout=1),
        ]
    )

    assert [(chunk.request_id, chunk.fanout) for chunk in chunks] == [("shared", 8)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"state": FanoutChunkState.GPU_AND_CPU},
        {"is_sealed": False},
        {"is_active_tail": True},
        {"in_flight": True},
    ],
)
def test_build_chunks_excludes_ineligible_blocks(
    kwargs: dict[str, object],
) -> None:
    planner = FanoutChunkPlanner()

    assert planner.build_chunks([make_block(0, **kwargs)]) == []


def test_max_blocks_per_chunk_bounds_transfer_granularity() -> None:
    planner = FanoutChunkPlanner(max_blocks_per_chunk=2)
    chunks = planner.build_chunks(make_block(idx) for idx in range(5))

    assert [chunk.num_blocks for chunk in chunks] == [2, 2, 1]


def test_select_deduplicates_shared_chunk_keys() -> None:
    planner = FanoutChunkPlanner()
    first = [make_block(idx, request_id="a", fanout=2) for idx in range(2)]
    second = [
        FanoutBlock(
            **{
                field: getattr(block, field)
                for field in block.__dataclass_fields__
                if field != "request_id"
            },
            request_id="b",
        )
        for block in first
    ]

    plan = planner.select([*first, *second], budget_blocks=8)

    assert len(plan.chunks) == 1
    assert plan.num_blocks == 2
