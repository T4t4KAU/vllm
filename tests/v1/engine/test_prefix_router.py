# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing decisions and work accounting without a model or GPU allocation."""

import asyncio

import pytest

from vllm import SamplingParams
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.prefix_router import PrefixAwareDPRouter


def make_request(request_id, tokens=None, max_tokens=128):
    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(range(8192)) if tokens is None else tokens,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@pytest.fixture
def router():
    return PrefixAwareDPRouter(
        num_ranks=2,
        block_size=16,
        load_slack=4,
        warm_ttl_s=300,
        min_prefix_blocks=4,
        work_slack_tokens=0,
    )


def warm(router, rank, count=1):
    for i in range(count):
        req = make_request(f"warm-{rank}-{i}")
        router.add_request(req, rank)
        router.observe_outputs(
            [EngineCoreOutput(req.request_id, [7])], {req.request_id}
        )


def test_equal_prefixes_balance_despite_unequal_historical_visits(router):
    """Repeated visits are not additional cache savings on a replicated prefix."""
    router.work_slack_tokens = 8192
    warm(router, 0, count=8)
    warm(router, 1)
    ranks = []
    for i in range(8):
        req = make_request(f"burst-{i}")
        rank = router.choose_rank(req, [0, 0], 1, start_index=1)
        router.add_request(req, rank)
        ranks.append(rank)
    assert ranks == [1, 0] * 4


def test_first_token_releases_completed_prefill_work(router):
    warm(router, 0)
    req = make_request("long-prefill", list(range(32768)), max_tokens=256)
    router.add_request(req, 0)
    router.observe_outputs([EngineCoreOutput(req.request_id, [7])], None)

    # Only 255 decode tokens remain; reusing 8192 prompt tokens is worthwhile.
    followup = make_request("followup", max_tokens=1)
    assert router.choose_rank(followup, [1, 0], 1) == 0


def test_decode_chunks_release_remaining_work(router):
    warm(router, 0)
    warm(router, 1)
    for rank, budget in ((0, 128), (1, 64)):
        req = make_request(f"decode-{rank}", max_tokens=budget)
        router.add_request(req, rank)
        router.observe_outputs([EngineCoreOutput(req.request_id, [7])], None)

    probe = make_request("before", max_tokens=1)
    assert router.choose_rank(probe, [1, 1], 0) == 1
    router.observe_outputs([EngineCoreOutput("decode-0", [8] * 80)], None)
    probe = make_request("after", max_tokens=1)
    assert router.choose_rank(probe, [1, 1], 1) == 0


@pytest.mark.parametrize("out_of_band", [False, True])
def test_preemption_restores_recompute_cost_until_model_output(router, out_of_band):
    warm(router, 0)
    req = make_request("preempt", list(range(32768)), max_tokens=256)
    router.add_request(req, 0)
    router.observe_outputs([EngineCoreOutput(req.request_id, [7])], None)
    preempt = EngineCoreEvent.new_event(EngineCoreEventType.PREEMPTED)
    output = EngineCoreOutput(req.request_id, [])
    output.events = [] if out_of_band else [preempt]
    router.observe_outputs([output], None, {req.request_id} if out_of_band else None)
    assert router.choose_rank(make_request("blocked", max_tokens=1), [1, 0], 1) == 1

    scheduled = EngineCoreEvent.new_event(EngineCoreEventType.SCHEDULED)
    router.observe_outputs(
        [EngineCoreOutput(req.request_id, [], events=[preempt, scheduled])], None
    )
    assert router.choose_rank(make_request("queued", max_tokens=1), [1, 0], 1) == 1
    router.observe_outputs([EngineCoreOutput(req.request_id, [8])], None)
    assert router.choose_rank(make_request("resumed", max_tokens=1), [1, 0], 1) == 0


@pytest.mark.parametrize("reason", [FinishReason.STOP, FinishReason.ABORT])
def test_finish_releases_only_remaining_work(router, reason):
    warm(router, 0)
    warm(router, 1)
    req = make_request("finish", max_tokens=128)
    router.add_request(req, 0)
    router.observe_outputs([EngineCoreOutput(req.request_id, [7] * 16)], None)
    router.observe_outputs(
        [EngineCoreOutput(req.request_id, [], finish_reason=reason)],
        {req.request_id},
    )
    assert router._rank_work == [0, 0]
    assert router.choose_rank(make_request("balanced"), [0, 0], 1, 1) == 1


def test_reset_does_not_prevent_work_progress_or_restore_cache_hints(router):
    req = make_request("reset", max_tokens=128)
    router.add_request(req, 0)
    router.invalidate_residency()
    router.observe_outputs([EngineCoreOutput(req.request_id, [7] * 16)], None)
    assert router._rank_work == [112 * router.decode_token_weight, 0]
    assert router.choose_rank(make_request("cold"), [0, 0], 1) == 1
    router.observe_outputs([], {req.request_id})
    assert router._rank_work == [0, 0]


def test_cold_burst_preserves_native_and_overload_overrides_affinity(router):
    assert router.choose_rank(make_request("cold"), [0, 0], 1) == 1
    warm(router, 0)
    assert router.choose_rank(make_request("overloaded"), [5, 0], 1) == 1


def test_generated_token_at_block_boundary_is_not_yet_cached(router):
    req = make_request("boundary", list(range(8191)))
    router.add_request(req, 0)
    router.observe_outputs([EngineCoreOutput(req.request_id, [42])], None)
    next_req = make_request("next", list(range(8191)) + [42, 43])
    router.choose_rank(next_req, [0, 0], 1)
    assert router.last_affinity_blocks < 512


@pytest.mark.parametrize("successes", [(True, True), (True, False), (False, True)])
def test_dp_cache_reset_requires_every_replica_to_succeed(router, successes):
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    warm(router, 0)
    client = object.__new__(DPLBAsyncMPClient)
    client.prefix_router = router
    client.core_engines = [b"0", b"1"]
    called = []

    async def utility(method, *, engine):
        called.append(engine)
        return successes[client.core_engines.index(engine)]

    client._call_utility_async = utility
    success = asyncio.run(client.call_utility_async("reset_prefix_cache"))
    assert success is all(successes)
    assert set(called) == set(client.core_engines)
    assert router.choose_rank(make_request("after-reset"), [0, 0], 1) == 1
