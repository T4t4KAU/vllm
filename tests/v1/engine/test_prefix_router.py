# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, FinishReason
from vllm.v1.engine.core_client import DPAsyncMPClient, DPLBAsyncMPClient
from vllm.v1.engine.prefix_router import PrefixAwareDPRouter


def _request(
    request_id: str,
    token_ids: list[int],
    *,
    cache_salt: str | None = None,
    turn: int | None = None,
    history_tokens: int | None = None,
    session_id: str | None = None,
    beam_step: int | None = None,
):
    extra_args: dict[str, int | str] = {}
    if turn is not None:
        extra_args["agentrix_turn"] = turn
    if history_tokens is not None:
        extra_args["agentrix_history_tokens"] = history_tokens
    if session_id is not None:
        extra_args["agentrix_session_id"] = session_id
    if beam_step is not None:
        extra_args["agentrix_beam_step"] = beam_step
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=token_ids,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        lora_request=None,
        mm_features=None,
        cache_salt=cache_salt,
        sampling_params=SimpleNamespace(max_tokens=8, extra_args=extra_args),
        pooling_params=None,
        data_parallel_rank=None,
        resumable=False,
    )


def _router(**kwargs) -> PrefixAwareDPRouter:
    params = {
        "num_ranks": 2,
        "block_size": 4,
        "load_slack": 4,
        "warm_ttl_s": 30,
        "min_prefix_blocks": 1,
        "checkpoint_stride_blocks": 1,
    }
    params.update(kwargs)
    return PrefixAwareDPRouter(**params)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_session_overload_ratio_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="session_overload_ratio must be finite"):
        _router(session_overload_ratio=value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_session_hit_ratio_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="session_hit_ratio must be finite"):
        _router(session_hit_ratio=value)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("session_overload_ratio", 1.0),
        ("session_hit_ratio", 0.0),
        ("session_hit_ratio", 1.0),
    ],
)
def test_session_ratio_accepts_valid_boundaries(name: str, value: float) -> None:
    router = _router(**{name: value})

    assert getattr(router, name) == value


def test_prefix_affinity_wins_within_load_slack() -> None:
    router = _router()
    first = _request("first", list(range(16)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
    )

    matching = _request("matching", list(range(16)) + [20])
    assert router.choose_rank(matching, [[1, 0], [0, 0]], 1) == 0
    assert router.last_affinity_blocks == 4


def test_load_bound_prevents_hot_rank_overload() -> None:
    router = _router()
    first = _request("first", list(range(16)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
    )

    matching = _request("matching", list(range(16)))
    assert router.choose_rank(matching, [[3, 0], [0, 0]], 1) == 1


def test_session_first_turn_uses_native_load_balance() -> None:
    router = _router(routing_policy="session_aware")
    shared = list(range(16))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="warm", new_token_ids=[], events=None)],
        {"warm"},
    )

    first_turn = _request("first-turn", shared, turn=0)
    assert router.choose_rank(first_turn, [[0, 0], [0, 0]], 1) == 1
    assert router.first_turn_balance_count == 1
    assert router.affinity_route_count == 0


def test_session_first_turn_beam_followup_uses_prefix_affinity() -> None:
    router = _router(routing_policy="session_aware")
    shared = list(range(16))
    first_step = _request(
        "first-step",
        shared,
        turn=0,
        session_id="session-a",
        beam_step=0,
    )
    first_rank = router.choose_rank(first_step, [[0, 0], [0, 0]], 0)
    router.add_request(first_step, rank=first_rank)
    router.observe_outputs(
        [SimpleNamespace(request_id="first-step", new_token_ids=[], events=None)],
        {"first-step"},
    )

    next_step = _request(
        "next-step",
        shared + [20],
        turn=0,
        session_id="session-a",
        beam_step=1,
    )

    assert router.choose_rank(next_step, [[0, 0], [0, 0]], 1) == 0
    assert router.first_turn_balance_count == 1
    assert router.followup_affinity_count == 1
    assert router.session_id_route_count == 0
    assert router.session_prefix_fallback_count == 1


def test_beam_followup_prefers_longest_prefix_over_session_mapping() -> None:
    router = _router(routing_policy="session_aware")
    shared = list(range(32))
    mapped = _request(
        "mapped",
        shared[:16],
        turn=0,
        session_id="session-a",
        beam_step=0,
    )
    router.add_request(mapped, rank=1)
    router.observe_outputs(
        [SimpleNamespace(request_id="mapped", new_token_ids=[], events=None)],
        {"mapped"},
    )
    longest = _request("longest", shared)
    router.add_request(longest, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="longest", new_token_ids=[], events=None)],
        {"longest"},
    )
    branch = _request(
        "branch",
        shared,
        turn=0,
        session_id="session-a",
        beam_step=1,
    )
    prefix = router._build_prefix(branch)
    assert prefix is not None
    assert [
        router._deepest_match(prefix.keys, resident)[0] for resident in router._resident
    ] == [8, 4]
    assert router._session_routes["session-a"][1] == 1

    assert router.choose_rank(branch, [[0, 0], [0, 0]], 1) == 0
    assert router.last_affinity_blocks == 8
    assert router.session_id_route_count == 0
    assert router.session_prefix_fallback_count == 1

    router.add_request(branch, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="branch", new_token_ids=[], events=None)],
        {"branch"},
    )
    assert "session-a" not in router._session_routes

    next_turn = _request(
        "next-turn",
        shared + [40],
        turn=1,
        session_id="session-a",
    )
    assert router.choose_rank(next_turn, [[0, 0], [0, 0]], 1) == 0
    assert router.session_id_route_count == 0
    assert router.session_prefix_fallback_count == 2


@pytest.mark.parametrize(
    "finish_order",
    [("branch-a", "branch-b"), ("branch-b", "branch-a")],
)
def test_beam_followups_clear_session_mapping_without_restoring_it(
    finish_order: tuple[str, str],
) -> None:
    router = _router(routing_policy="session_aware")
    first_step = _request(
        "first-step",
        list(range(16)),
        turn=0,
        session_id="session-a",
        beam_step=0,
    )
    router.add_request(first_step, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first-step", new_token_ids=[], events=None)],
        {"first-step"},
    )
    assert router._session_routes["session-a"][1] == 0
    branches = {
        request_id: _request(
            request_id,
            list(range(16)) + [token_id],
            turn=0,
            session_id="session-a",
            beam_step=1,
        )
        for request_id, token_id in (("branch-a", 20), ("branch-b", 21))
    }

    router.add_request(branches["branch-a"], rank=0)
    router.add_request(branches["branch-b"], rank=1)
    assert "session-a" not in router._session_routes
    for request_id in finish_order:
        router.observe_outputs(
            [SimpleNamespace(request_id=request_id, new_token_ids=[], events=None)],
            {request_id},
        )

    assert "session-a" not in router._session_routes


def test_session_followup_uses_prefix_affinity() -> None:
    router = _router(routing_policy="session_aware")
    shared = list(range(16))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="warm", new_token_ids=[], events=None)],
        {"warm"},
    )

    followup = _request("followup", shared + [20], turn=1, history_tokens=16)
    assert router.choose_rank(followup, [[0, 0], [0, 0]], 1) == 0
    assert router.followup_affinity_count == 1
    assert router.followup_rebalance_count == 0


def test_session_id_preserves_affinity_when_prefix_exists_on_both_ranks() -> None:
    router = _router(routing_policy="session_aware")
    shared = list(range(16))
    first = _request("first", shared, turn=0, session_id="session-a")
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)],
        {"first"},
    )
    other = _request("other", shared)
    router.add_request(other, rank=1)
    router.observe_outputs(
        [SimpleNamespace(request_id="other", new_token_ids=[], events=None)],
        {"other"},
    )

    followup = _request(
        "followup", shared, turn=1, history_tokens=16, session_id="session-a"
    )
    assert router.choose_rank(followup, [[0, 0], [0, 0]], 1) == 0
    assert router.session_id_route_count == 1


def test_session_mapping_ttl_starts_when_long_request_finishes() -> None:
    now = [100.0]
    router = _router(routing_policy="session_aware", warm_ttl_s=5, clock=lambda: now[0])
    shared = list(range(16))
    first = _request("first", shared, turn=0, session_id="session-a")
    router.add_request(first, rank=0)
    now[0] = 106.0
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)],
        {"first"},
    )

    other = _request("other", shared)
    router.add_request(other, rank=1)
    router.observe_outputs(
        [SimpleNamespace(request_id="other", new_token_ids=[], events=None)],
        {"other"},
    )
    followup = _request(
        "followup", shared, turn=1, history_tokens=16, session_id="session-a"
    )

    assert router.choose_rank(followup, [[0, 0], [0, 0]], 1, start_index=1) == 0


def test_session_followup_rebalances_when_affine_rank_is_overloaded() -> None:
    router = _router(routing_policy="session_aware", session_overload_ratio=1.5)
    shared = list(range(16))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="warm", new_token_ids=[], events=None)],
        {"warm"},
    )

    followup = _request("followup", shared, turn=1, history_tokens=16)
    assert router.choose_rank(followup, [[3, 0], [0, 0]], 1) == 1
    assert router.session_overload_rebalance_count == 1


def test_session_work_overload_rebalances_when_baseline_is_affine_rank() -> None:
    router = _router(routing_policy="session_aware", work_slack_tokens=0)
    shared = list(range(16))
    warm = _request("warm", shared, session_id="session-a")
    router.add_request(warm, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="warm", new_token_ids=[], events=None)],
        {"warm"},
    )

    unrelated = _request("unrelated", list(range(100, 300)))
    unrelated.sampling_params.max_tokens = 1024
    router.add_request(unrelated, rank=0)
    assert router._rank_work == [16584, 0]

    followup = _request(
        "followup", shared, turn=1, history_tokens=16, session_id="session-a"
    )
    assert router.choose_rank(followup, [[0, 0], [0, 0]], 0) == 1
    assert router.followup_rebalance_count == 1
    assert router.session_overload_rebalance_count == 1
    assert router.rank_route_counts == [0, 1]


def test_session_followup_rebalances_when_expected_history_was_evicted() -> None:
    router = _router(routing_policy="session_aware", session_hit_ratio=0.75)
    warm = _request("warm", list(range(8)))
    router.add_request(warm, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="warm", new_token_ids=[], events=None)],
        {"warm"},
    )

    followup = _request("followup", list(range(32)), turn=1, history_tokens=32)
    assert router.choose_rank(followup, [[0, 0], [0, 0]], 1) == 1
    assert router.session_cache_miss_rebalance_count == 1


def test_session_policy_without_turn_preserves_prefix_aware_fallback() -> None:
    router = _router(routing_policy="session_aware")
    shared = list(range(16))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="warm", new_token_ids=[], events=None)],
        {"warm"},
    )

    unknown = _request("unknown", shared)
    assert router.choose_rank(unknown, [[0, 0], [0, 0]], 1) == 0
    assert router.unknown_turn_route_count == 1


def test_unconfirmed_prefix_preserves_native_distribution() -> None:
    router = _router()
    router.add_request(_request("pending", list(range(16))), rank=0)

    matching = _request("matching", list(range(16)))
    assert router.choose_rank(matching, [[1, 0], [0, 0]], 1) == 1
    assert router.affinity_route_count == 0


def test_finished_request_retains_only_observed_prefix() -> None:
    router = _router()
    observed = _request("observed", list(range(8)))
    router.add_request(observed, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="observed", new_token_ids=[], events=None)],
        {"observed"},
    )

    aborted = _request("aborted", list(range(100, 108)))
    router.add_request(aborted, rank=1)
    router.finish_request("aborted")

    assert (
        router.choose_rank(_request("warm", list(range(8))), [[1, 0], [0, 0]], 1) == 0
    )
    assert (
        router.choose_rank(
            _request("not-warm", list(range(100, 108))), [[1, 0], [0, 0]], 1
        )
        == 1
    )


def test_abort_output_does_not_confirm_unexecuted_prefix() -> None:
    router = _router()
    aborted = _request("aborted", list(range(8)))
    router.add_request(aborted, rank=0)
    router.observe_outputs(
        [EngineCoreOutput("aborted", [], finish_reason=FinishReason.ABORT)],
        {"aborted"},
    )

    matching = _request("matching", list(range(8)))
    assert router.choose_rank(matching, [[1, 0], [0, 0]], 1) == 1
    assert not router._resident[0]


def test_cache_salt_isolates_prefix_namespaces() -> None:
    router = _router()
    router.add_request(_request("first", list(range(8)), cache_salt="tenant-a"), rank=0)

    isolated = _request("isolated", list(range(8)), cache_salt="tenant-b")
    assert router.choose_rank(isolated, [[1, 0], [0, 0]], 1) == 1
    assert router.last_affinity_blocks == 0


def test_warm_affinity_expires() -> None:
    now = [100.0]
    router = _router(warm_ttl_s=5, clock=lambda: now[0])
    first = _request("first", list(range(8)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)],
        {"first"},
    )
    now[0] = 106.0

    assert (
        router.choose_rank(_request("next", list(range(8))), [[1, 0], [0, 0]], 1) == 1
    )


def test_warm_checkpoint_budget_evicts_oldest_prefix() -> None:
    router = _router(max_warm_checkpoints=2)
    for request_id, start in (("old", 0), ("new", 100)):
        request = _request(request_id, list(range(start, start + 8)))
        router.add_request(request, rank=0)
        router.observe_outputs(
            [SimpleNamespace(request_id=request_id, new_token_ids=[], events=None)],
            {request_id},
        )

    old = _request("old-match", list(range(8)))
    new = _request("new-match", list(range(100, 108)))
    assert router.choose_rank(old, [[1, 0], [0, 0]], 1) == 1
    assert router.choose_rank(new, [[1, 0], [0, 0]], 1) == 0


def test_cache_reset_invalidates_warm_residency() -> None:
    router = _router()
    first = _request("first", list(range(8)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)],
        {"first"},
    )
    router.invalidate_residency()

    assert (
        router.choose_rank(_request("next", list(range(8))), [[1, 0], [0, 0]], 1) == 1
    )


def test_short_and_prompt_embed_requests_bypass_hashing() -> None:
    router = PrefixAwareDPRouter(2, 4, 4, 30, min_prefix_blocks=2)
    assert not router.should_route(_request("short", list(range(4))))

    prompt_embed = _request("embed", list(range(8)))
    prompt_embed.prompt_embeds = SimpleNamespace()
    assert not router.should_route(prompt_embed)


def test_requests_that_skip_prefix_cache_preserve_native_routing() -> None:
    router = _router()
    first = _request("first", list(range(16)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
    )

    sampling = _request("sampling", list(range(16)))
    sampling.sampling_params.skip_reading_prefix_cache = True
    assert not router.should_route(sampling)

    pooling = _request("pooling", list(range(16)))
    pooling.sampling_params = None
    pooling.pooling_params = SimpleNamespace(skip_reading_prefix_cache=True)
    assert not router.should_route(pooling)

    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.core_engines = [b"0", b"1"]
    client.lb_engines = [[1, 0], [0, 0]]
    client.eng_start_index = 0
    client.prefix_router = router
    assert client.get_core_engine_for_request(sampling) == b"1"


def test_resumable_requests_preserve_native_routing() -> None:
    router = _router()
    first = _request("first", list(range(16)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
    )

    resumable = _request("stream", list(range(16)))
    resumable.resumable = True
    assert not router.should_route(resumable)

    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.core_engines = [b"0", b"1"]
    client.lb_engines = [[1, 0], [0, 0]]
    client.eng_start_index = 0
    client.prefix_router = router
    assert client.get_core_engine_for_request(resumable) == b"1"


def test_generated_full_blocks_extend_affinity() -> None:
    router = _router()
    first = _request("first", list(range(6)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[6, 7, 8, 9], events=None)],
        {"first"},
    )

    assert (
        router.choose_rank(_request("next", list(range(10))), [[1, 0], [0, 0]], 1) == 0
    )
    assert router.last_affinity_blocks == 2


def test_final_sampled_token_is_not_marked_resident() -> None:
    router = _router()
    first = _request("first", list(range(7)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[7], events=None)],
        {"first"},
    )

    assert max(key[0] for key in router._resident[0]) == 1


def test_generated_checkpoints_are_sparse_and_bounded() -> None:
    router = _router(
        checkpoint_stride_blocks=2,
        max_checkpoints_per_request=3,
    )
    first = _request("first", list(range(4)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
    )
    for token_id in range(4, 44):
        router.observe_outputs(
            [
                SimpleNamespace(
                    request_id="first", new_token_ids=[token_id], events=None
                )
            ],
            None,
        )

    keys = router._requests["first"].prefix.keys
    assert len(keys) == 3
    assert keys[0][0] == router.min_prefix_blocks
    assert all(
        depth == router.min_prefix_blocks
        or depth % router.checkpoint_stride_blocks == 0
        for depth, _ in keys
    )
    assert set(router._resident[0]) == set(keys)


def test_preemption_invalidates_active_residency() -> None:
    router = _router()
    first = _request("first", list(range(8)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
    )
    router.observe_outputs(
        [
            SimpleNamespace(
                request_id="first",
                new_token_ids=[],
                events=[
                    SimpleNamespace(type=EngineCoreEventType.PREEMPTED),
                    SimpleNamespace(type=EngineCoreEventType.SCHEDULED),
                ],
            )
        ],
        None,
    )

    matching = _request("matching", list(range(8)))
    assert router.choose_rank(matching, [[1, 0], [0, 0]], 1) == 1
    assert not router._resident[0]


def test_resumed_model_output_restores_and_warms_residency() -> None:
    router = _router()
    first = _request("first", list(range(8)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
    )
    router.observe_outputs([], None, {"first"})

    router.observe_outputs(
        [
            SimpleNamespace(
                request_id="first",
                new_token_ids=[8],
                events=[
                    SimpleNamespace(type=EngineCoreEventType.PREEMPTED),
                    SimpleNamespace(type=EngineCoreEventType.SCHEDULED),
                ],
            )
        ],
        {"first"},
    )

    assert "first" not in router._requests
    assert router._resident[0]
    assert len(router._warm_requests) == 1


def test_preemption_signal_invalidates_residency_without_stats() -> None:
    async def run() -> None:
        client = object.__new__(DPLBAsyncMPClient)
        client.reqs_in_flight = {}
        client.prefix_router = _router()
        first = _request("first", list(range(8)))
        client.prefix_router.add_request(first, rank=0)
        client.prefix_router.observe_outputs(
            [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
        )

        outputs = SimpleNamespace(
            outputs=[],
            finished_requests=None,
            preempted_requests={"first"},
        )
        await DPLBAsyncMPClient.process_engine_outputs(client, outputs)
        assert not client.prefix_router._resident[0]

    asyncio.run(run())


def test_preempted_output_with_token_cannot_restore_or_warm_residency() -> None:
    router = _router()
    first = _request("first", list(range(8)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
    )

    router.observe_outputs(
        [
            SimpleNamespace(
                request_id="first",
                new_token_ids=[8],
                events=[
                    SimpleNamespace(type=EngineCoreEventType.PREEMPTED),
                    SimpleNamespace(type=EngineCoreEventType.SCHEDULED),
                ],
            )
        ],
        None,
        {"first"},
    )

    record = router._requests["first"]
    assert not record.resident
    assert record.uncomputed_token_ids == [8]
    assert not router._resident[0]

    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[9], events=None)],
        None,
    )
    assert record.resident
    assert record.uncomputed_token_ids == [9]

    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[10], events=None)],
        {"first"},
        {"first"},
    )

    assert "first" not in router._requests
    assert not router._resident[0]
    assert not router._warm_requests


def test_minimum_prefix_depth_is_kept_with_sparse_checkpoints() -> None:
    router = PrefixAwareDPRouter(
        2,
        4,
        4,
        30,
        min_prefix_blocks=5,
        checkpoint_stride_blocks=4,
    )
    first = _request("first", list(range(20)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[], events=None)], None
    )

    matching = _request("matching", list(range(24)))
    assert router.choose_rank(matching, [[1, 0], [0, 0]], 1) == 0
    assert router.last_affinity_blocks == 5


def test_work_bound_prevents_affinity_from_overloading_rank() -> None:
    router = _router(work_slack_tokens=0)
    shared = list(range(16))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="warm", new_token_ids=[], events=None)],
        {"warm"},
    )

    unrelated = _request("unrelated", list(range(100, 300)))
    unrelated.sampling_params.max_tokens = 1024
    router.add_request(unrelated, rank=0)

    matching = _request("matching", shared)
    assert router.choose_rank(matching, [[0, 0], [0, 0]], 1) == 1


def test_unidentified_multimodal_request_bypasses_routing() -> None:
    router = _router()
    request = _request("mm", list(range(16)))
    request.mm_features = [
        SimpleNamespace(
            identifier=None,
            modality="image",
            mm_position=SimpleNamespace(offset=0, length=4),
        )
    ]

    assert not router.should_route(request)


def test_dplb_client_preserves_native_short_request_path() -> None:
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.core_engines = [b"0", b"1"]
    client.lb_engines = [[1, 0], [0, 0]]
    client.eng_start_index = 0
    client.prefix_router = PrefixAwareDPRouter(2, 4, 4, 30, min_prefix_blocks=2)

    request = _request("short", list(range(4)))
    assert client.get_core_engine_for_request(request) == b"1"
    assert client.prefix_router.route_count == 0


def test_dplb_client_without_router_preserves_existing_behavior() -> None:
    client = object.__new__(DPLBAsyncMPClient)
    client.client_count = 1
    client.reqs_in_flight = {}
    client.core_engines = [b"0", b"1"]
    client.lb_engines = [[1, 0], [0, 0]]
    client.eng_start_index = 0

    request = _request("native", list(range(16)))
    assert client.get_core_engine_for_request(request) == b"1"


def test_dplb_client_observes_request_lifecycle() -> None:
    async def run() -> None:
        client = object.__new__(DPLBAsyncMPClient)
        client.client_count = 1
        client.reqs_in_flight = {}
        client.core_engines = [b"0", b"1"]
        client.lb_engines = [[0, 0], [0, 0]]
        client.eng_start_index = 0
        client.prefix_router = _router()

        request = _request("first", list(range(8)))
        assert client.get_core_engine_for_request(request) == b"0"
        outputs = SimpleNamespace(
            outputs=[
                SimpleNamespace(request_id="first", new_token_ids=[], events=None)
            ],
            finished_requests={"first"},
        )
        await DPLBAsyncMPClient.process_engine_outputs(client, outputs)

        matching = _request("matching", list(range(8)))
        assert client.get_core_engine_for_request(matching) == b"0"

    asyncio.run(run())


def test_dplb_client_invalidates_hints_when_cache_is_cleared() -> None:
    async def utility_succeeded(*args, **kwargs):
        return True

    async def run() -> None:
        client = object.__new__(DPLBAsyncMPClient)
        client.prefix_router = _router()

        def add_warm(request_id: str) -> None:
            request = _request(request_id, list(range(8)))
            client.prefix_router.add_request(request, rank=0)
            client.prefix_router.observe_outputs(
                [SimpleNamespace(request_id=request_id, new_token_ids=[], events=None)],
                {request_id},
            )
            assert client.prefix_router._resident[0]

        add_warm("reset")
        await client.reset_prefix_cache_async()
        assert not client.prefix_router._resident[0]

        add_warm("pause")
        await client.pause_scheduler_async(clear_cache=True)
        assert not client.prefix_router._resident[0]

        add_warm("sleep")
        await client.sleep_async(level=1)
        assert not client.prefix_router._resident[0]

    with (
        patch.object(DPAsyncMPClient, "reset_prefix_cache_async", utility_succeeded),
        patch.object(DPAsyncMPClient, "pause_scheduler_async", utility_succeeded),
        patch.object(DPAsyncMPClient, "sleep_async", utility_succeeded),
    ):
        asyncio.run(run())
