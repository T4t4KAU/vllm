# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from types import SimpleNamespace

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    DPReloadEvent,
    DPReloadEventType,
    DPReloadRequest,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
)
from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient, _DPReloadPlacement
from vllm.v1.engine.prefix_router import PrefixAwareDPRouter
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder


def _request(
    request_id: str,
    token_ids: list[int],
    *,
    cache_salt: str | None = None,
):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=token_ids,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        lora_request=None,
        mm_features=None,
        cache_salt=cache_salt,
    )


def test_prefix_affinity_wins_within_load_slack() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=4,
        warm_ttl_s=30,
        min_prefix_blocks=1,
    )
    first = _request("first", list(range(16)))
    router.add_request(first, rank=0)

    matching = _request("matching", list(range(16)) + [20])
    assert router.choose_rank(matching, [[1, 0], [0, 0]]) == 0
    assert router.last_affinity_blocks == 4


def test_load_bound_prevents_hot_rank_overload() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=4,
        warm_ttl_s=30,
        min_prefix_blocks=1,
    )
    first = _request("first", list(range(16)))
    router.add_request(first, rank=0)

    matching = _request("matching", list(range(16)))
    assert router.choose_rank(matching, [[3, 0], [0, 0]]) == 1


def test_finished_request_retains_warm_affinity() -> None:
    now = 100.0
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=4,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        clock=lambda: now,
    )
    first = _request("first", list(range(8)))
    router.add_request(first, rank=0)
    router.observe_outputs(
        [SimpleNamespace(request_id="first", new_token_ids=[8, 9, 10, 11])],
        {"first"},
    )

    continuation = _request("continuation", list(range(12)))
    assert router.choose_rank(continuation, [[1, 0], [0, 0]]) == 0
    assert router.last_affinity_blocks == 3


def test_cache_salt_isolates_prefix_namespaces() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=4,
        warm_ttl_s=30,
        min_prefix_blocks=1,
    )
    first = _request("first", list(range(8)), cache_salt="tenant-a")
    router.add_request(first, rank=0)

    isolated = _request("isolated", list(range(8)), cache_salt="tenant-b")
    assert router.choose_rank(isolated, [[1, 0], [0, 0]]) == 1
    assert router.last_affinity_blocks == 0


def test_warm_affinity_expires() -> None:
    now = [100.0]
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=4,
        warm_ttl_s=5,
        min_prefix_blocks=1,
        clock=lambda: now[0],
    )
    first = _request("first", list(range(8)))
    router.add_request(first, rank=0)
    router.finish_request("first")
    now[0] = 106.0

    matching = _request("matching", list(range(8)))
    assert router.choose_rank(matching, [[1, 0], [0, 0]]) == 1
    assert router.last_affinity_blocks == 0


def test_graph_bound_overrides_affinity_when_bucket_would_grow() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        graph_buckets=(8, 16),
        graph_slack_buckets=0,
        prefix_chunk_blocks=4,
    )
    first = _request("first", list(range(16)))
    router.add_request(first, rank=0)
    router.update_engine_telemetry(
        [
            (("forest", 16, 15, 10, 5), 0.5),
            (("forest", 8, 1, 0, 1), 0.5),
        ]
    )

    matching = _request("matching", list(range(16)))
    assert router.choose_rank(matching, [[0, 0], [0, 0]]) == 1
    assert router.graph_bound_route_count == 1


def test_graph_bound_never_overrides_load_balance() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=0,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        graph_buckets=(8, 16),
        graph_slack_buckets=0,
        prefix_chunk_blocks=4,
    )
    first = _request("first", list(range(16)))
    router.add_request(first, rank=0)
    router.update_engine_telemetry(
        [
            (("forest", 8, 1, 1, 0), 0.5),
            (("forest", 16, 15, 0, 15), 0.5),
        ]
    )

    matching = _request("matching", list(range(16)))
    assert router.choose_rank(matching, [[0, 1], [0, 0]]) == 1
    assert router.graph_bound_route_count == 1


def test_zero_work_slack_balances_before_prefix_affinity() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        work_slack_tokens=0,
    )
    shared = list(range(16))
    router.add_request(_request("shared", shared), rank=0)
    router.add_request(_request("private", list(range(100, 180))), rank=0)

    matching = _request("matching", shared)
    assert router.choose_rank(matching, [[0, 0], [0, 0]]) == 1


def test_work_balance_can_keep_a_cheaper_warm_prefix_owner() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        work_slack_tokens=0,
    )
    shared = list(range(64))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.finish_request(warm.request_id)

    matching = _request("matching", shared)
    assert router.choose_rank(matching, [[1, 1], [0, 1]]) == 0


def test_short_prefix_bypasses_to_ordinary_load_balance() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        work_slack_tokens=0,
        kv_capacity_blocks=32,
        replication_relax_ratio=0.20,
    )
    shared = list(range(16))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.finish_request(warm.request_id)

    matching = _request("matching", shared)
    assert router.choose_rank(matching, [[1, 0], [0, 0]]) == 1
    assert router.replication_relaxed_route_count == 0
    assert router.ordinary_bypass_route_count == 1


def test_native_ordinary_route_accounting_does_not_track_prefix() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        kv_capacity_blocks=32,
        replication_relax_ratio=0.20,
    )
    short = _request("short", list(range(16)))
    assert not router.should_coalesce(short)

    router.record_ordinary_route(rank=1)
    assert router.route_count == 1
    assert router.ordinary_bypass_route_count == 1
    assert router.rank_route_counts == [0, 1]
    assert short.request_id not in router._pending_prefixes
    assert short.request_id not in router._requests


def test_expensive_prefix_relaxes_request_count_but_balances_work() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        work_slack_tokens=0,
        kv_capacity_blocks=32,
        replication_relax_ratio=0.20,
    )
    shared = list(range(32))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.finish_request(warm.request_id)

    matching = _request("matching", shared)
    assert router.choose_rank(matching, [[1, 0], [0, 0]]) == 0
    assert router.replication_relaxed_route_count == 1
    assert router.ordinary_bypass_route_count == 0


def test_only_expensive_resident_prefixes_coalesce() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        kv_capacity_blocks=32,
        replication_relax_ratio=0.20,
    )
    short = list(range(16))
    long = list(range(100, 132))
    for request_id, tokens in (("short-warm", short), ("long-warm", long)):
        warm = _request(request_id, tokens)
        router.add_request(warm, rank=0)
        router.finish_request(warm.request_id)

    assert not router.should_coalesce(_request("short", short))
    assert router.should_coalesce(_request("long", long))


def test_long_prefix_coalesces_to_bootstrap_owners() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        kv_capacity_blocks=32,
        replication_relax_ratio=0.20,
    )
    long = list(range(32))
    request = _request("long", long)

    assert router.should_coalesce(request)
    assert router.choose_rank(request, [[0, 0], [0, 0]]) == 0
    assert router.long_prefix_bootstrap_route_count == 1
    assert router.replication_relaxed_route_count == 0


def test_shallow_template_match_does_not_skew_long_request() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        work_slack_tokens=0,
        kv_capacity_blocks=32,
        replication_relax_ratio=0.20,
    )
    template = list(range(16))
    warm = _request("warm-template", template)
    router.add_request(warm, rank=0)
    router.finish_request(warm.request_id)

    long = _request("long", template + list(range(100, 212)))
    assert router.should_coalesce(long)
    assert router.choose_rank(long, [[1, 0], [0, 0]]) == 1
    assert router.affinity_route_count == 0
    assert router.replication_relaxed_route_count == 0
    assert router.long_prefix_bootstrap_route_count == 1


def test_deep_prefix_reuse_allows_bounded_work_imbalance() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        work_slack_tokens=0,
        kv_capacity_blocks=32,
        replication_relax_ratio=0.20,
    )
    shared = list(range(128))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.finish_request(warm.request_id)
    router.add_request(_request("rank-zero-work", list(range(1000, 1200))), rank=0)

    matching = _request("matching", shared)
    assert router.choose_rank(matching, [[0, 0], [0, 0]]) == 0
    assert router.affinity_route_count == 1


def test_deep_prefix_allows_capacity_bounded_queue_skew() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=0,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        work_slack_tokens=0,
        kv_capacity_blocks=128,
        max_num_seqs=64,
        replication_relax_ratio=0.20,
    )
    shared = list(range(128))
    warm = _request("warm", shared)
    router.add_request(warm, rank=0)
    router.finish_request(warm.request_id)

    within_bound = _request("within-bound", shared)
    assert router.choose_rank(within_bound, [[100, 0], [0, 0]]) == 0
    assert router.cohort_locked_route_count == 1

    router.rank_route_counts[:] = [16, 0]
    beyond_bound = _request("beyond-bound", shared)
    assert router.choose_rank(beyond_bound, [[17, 0], [0, 0]]) == 1


def test_deeper_warm_prefix_beats_shallow_active_prefix() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        work_slack_tokens=0,
        kv_capacity_blocks=32,
        max_num_seqs=64,
        replication_relax_ratio=0.20,
    )
    shallow = list(range(32))
    deep = shallow + list(range(100, 164))
    active = _request("active-shallow", shallow)
    router.add_request(active, rank=0)
    router.set_resident(active.request_id, rank=0, resident=True)
    warm = _request("warm-deep", deep)
    router.add_request(warm, rank=1)
    router.finish_request(warm.request_id)

    matching = _request("matching", deep)
    assert router.choose_rank(matching, [[0, 0], [0, 0]]) == 1
    assert router.cohort_locked_route_count == 1


def test_graph_bound_uses_logical_forest_before_telemetry_arrives() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
        graph_buckets=(2, 4),
        graph_slack_buckets=0,
        prefix_chunk_blocks=1,
    )
    router.add_request(_request("existing", list(range(8))), rank=0)

    private = _request("private", list(range(8, 16)))
    assert router.choose_rank(private, [[0, 0], [0, 0]]) == 1
    assert router.graph_bound_route_count == 1


def test_missing_coordinator_sample_preserves_direct_engine_telemetry() -> None:
    router = PrefixAwareDPRouter(2, 4, 32, 30, 1)
    router.update_rank_telemetry(1, ("forest", 64, 24, 8, 16), 0.625)

    router.update_engine_telemetry([(None, 0.0), (None, 0.5)])

    assert router.telemetry_snapshot[1] == ("forest", 64, 24, 8, 16, 0.5)


def test_dense_dp_output_updates_rank_telemetry_directly() -> None:
    async def run() -> None:
        client = object.__new__(DPLBAsyncMPClient)
        client.prefix_router = PrefixAwareDPRouter(2, 4, 32, 30, 1)
        client.engine_ranks_managed = [0, 1]
        client.reqs_in_flight = {}
        outputs = SimpleNamespace(
            engine_index=1,
            scheduler_stats=SimpleNamespace(
                fork_execution_stats=("forest", 64, 24, 8, 16),
                kv_cache_usage=0.625,
            ),
            outputs=[],
            finished_requests=None,
        )

        await DPLBAsyncMPClient.process_engine_outputs(client, outputs)

        assert client.prefix_router.telemetry_snapshot[1] == (
            "forest",
            64,
            24,
            8,
            16,
            0.625,
        )

    asyncio.run(run())


def test_arrival_wave_orders_deepest_shared_subtree_first() -> None:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=4,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=1,
    )
    requests = [
        _request("a0", [1] * 12 + [2, 3, 4, 5]),
        _request("b0", [9] * 8 + [2, 3, 4, 5]),
        _request("a1", [1] * 12 + [6, 7, 8, 9]),
        _request("b1", [9] * 8 + [6, 7, 8, 9]),
    ]

    assert router.order_arrival_wave(requests) == [0, 2, 1, 3]
    assert router.arrival_wave_count == 1


def test_dplb_client_coalesces_concurrent_prefix_wave() -> None:
    async def run() -> None:
        client = object.__new__(DPLBAsyncMPClient)
        client.prefix_router = PrefixAwareDPRouter(2, 4, 32, 30, 1)
        client._prefix_arrival_wave_s = 0.001
        client._prefix_wave_pending = []
        client._prefix_wave_inflight = []
        client._prefix_wave_task = None
        client.current_wave = 0
        client.client_index = 0
        client.lb_engines = [[0, 0], [0, 0]]
        client.core_engines = [b"0", b"1"]
        client.reqs_in_flight = {}
        client.eng_start_index = 0
        client.client_count = 1
        client.engines_running = True
        client._ensure_stats_update_task = lambda: None
        client._ensure_output_queue_task = lambda: None
        client._send_input = lambda *args: asyncio.sleep(0)

        requests = [
            _request("a0", [1] * 16),
            _request("b0", [9] * 16),
            _request("a1", [1] * 16),
            _request("b1", [9] * 16),
        ]
        for request in requests:
            request.pooling_params = None
            request.data_parallel_rank = None
            request.sampling_params = SimpleNamespace(max_tokens=8)
            request.current_wave = 0
            request.client_index = 0

        await asyncio.gather(
            *(client.add_request_async(request) for request in requests)
        )

        assert client.prefix_router.arrival_wave_count == 1
        assert client.prefix_router.rank_route_counts == [2, 2]
        assert len(client.reqs_in_flight) == 4

    asyncio.run(run())


def test_dp_core_publishes_physical_telemetry_when_counts_are_unchanged() -> None:
    published: list[tuple[int, EngineCoreOutputs]] = []
    execution_stats = ("forest", 64, 24, 8, 16)
    core = SimpleNamespace(
        publish_dp_lb_stats=True,
        publish_fork_dp_telemetry=True,
        last_counts=(2, 3),
        last_fork_telemetry=(None, 0.0),
        step_counter=7,
        current_wave=4,
        scheduler=SimpleNamespace(
            get_request_counts=lambda: (2, 3),
            fork_execution_stats=execution_stats,
            kv_cache_manager=SimpleNamespace(usage=0.6254),
        ),
        output_queue=SimpleNamespace(put_nowait=published.append),
    )

    DPEngineCoreProc._maybe_publish_request_counts(core)

    assert len(published) == 1
    client_index, outputs = published[0]
    assert client_index == -1
    assert outputs.scheduler_stats.fork_execution_stats == execution_stats
    assert outputs.scheduler_stats.kv_cache_usage == 0.625
    assert core.last_counts == (2, 3)
    assert core.last_fork_telemetry == (execution_stats, 0.625)


def test_residency_updates_are_idempotent_across_preemption() -> None:
    router = PrefixAwareDPRouter(2, 4, 32, 30, 1)
    request = _request("request", list(range(16)))
    router.add_request(request, rank=0)

    router.set_resident("request", 0, True)
    router.set_resident("request", 0, True)
    assert all(count == 1 for count in router._active[0].values())

    router.set_resident("request", 0, False)
    router.set_resident("request", 0, False)
    assert not router._active[0]


def test_dp_reload_request_msgpack_round_trip() -> None:
    request = EngineCoreRequest(
        request_id="reload",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8, temperature=0),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    payload = DPReloadRequest(
        request=request,
        source_rank=0,
        ownership_epoch=2,
        num_preemptions=3,
        source_local_tokens=128,
        source_external_tokens=256,
        output_token_ids=[4, 5],
    )

    frames = MsgpackEncoder().encode(payload)
    decoded = MsgpackDecoder(DPReloadRequest).decode(frames)

    assert decoded.request.request_id == "reload"
    assert decoded.request.prompt_token_ids == [1, 2, 3]
    assert decoded.ownership_epoch == 2
    assert decoded.output_token_ids == [4, 5]


def test_reload_rebalance_chooses_dominant_active_anchor() -> None:
    router = PrefixAwareDPRouter(
        2,
        4,
        32,
        30,
        1,
        reload_min_fanout_gain=1,
        reload_min_prefix_gain_blocks=1,
    )
    tokens = list(range(16))
    source = _request("source", tokens)
    router.add_request(source, rank=0)
    router.set_resident("source", 0, True)
    for request_id in ("target-a", "target-b"):
        request = _request(request_id, tokens)
        router.add_request(request, rank=1)
        router.set_resident(request_id, 1, True)

    assert router.detach_for_reload("source") == 0
    assert router.choose_reload_rank("source", 0, [[1, 0], [0, 2]]) == 1
    assert router.attach_after_reload("source", 1)
    router.set_resident("source", 1, True)

    record = router._requests["source"]
    assert record.rank == 1
    assert record.active
    assert router.reload_rebalanced_count == 1
    assert router.reload_predicted_saved_blocks == 4


def test_reload_rebalance_stays_local_without_active_anchor() -> None:
    router = PrefixAwareDPRouter(2, 4, 32, 30, 1)
    request = _request("source", list(range(16)))
    router.add_request(request, rank=0)

    assert router.detach_for_reload("source") == 0
    assert router.choose_reload_rank("source", 0, [[1, 0], [0, 0]]) == 0
    assert router.attach_after_reload("source", 0)
    assert router.reload_local_count == 1


def test_reload_rebalance_considers_recent_warm_anchor() -> None:
    router = PrefixAwareDPRouter(
        2,
        4,
        32,
        30,
        1,
        reload_min_fanout_gain=1,
        reload_min_prefix_gain_blocks=1,
    )
    tokens = list(range(16))
    source = _request("source", tokens)
    target = _request("target", tokens)
    router.add_request(source, rank=0)
    router.add_request(target, rank=1)
    router.set_resident(target.request_id, 1, True)
    router.finish_request(target.request_id)

    assert router.detach_for_reload(source.request_id) == 0
    assert router.choose_reload_rank(source.request_id, 0, [[1, 0], [0, 0]]) == 1


def test_dplb_client_commits_reload_only_after_target_prepares(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_FORK_ATTN_DP_RELOAD_MIN_EXTERNAL_TOKENS", "4")

    async def run() -> None:
        client = object.__new__(DPLBAsyncMPClient)
        client.prefix_router = PrefixAwareDPRouter(
            2,
            4,
            32,
            30,
            1,
            reload_min_prefix_gain_blocks=1,
        )
        client.dp_reload_rebalance_enabled = True
        client.engine_ranks_managed = [0, 1]
        client.core_engines = [b"rank-0", b"rank-1"]
        client.lb_engines = [[1, 0], [0, 2]]
        client.eng_start_index = 0
        client.reqs_in_flight = {"source": b"rank-0"}
        initial_request = _request("source", list(range(16)))
        client._dp_reload_placements = {
            "source": _DPReloadPlacement(
                initial_request,
                b"rank-0",
                output_token_ids=[17, 18],
            )
        }
        sent = []

        async def send(request_type, payload, engine):
            sent.append((request_type, payload, engine))

        client._send_input = send
        client.prefix_router.add_request(initial_request, rank=0)
        client.prefix_router.set_resident("source", 0, True)
        for request_id in ("target-a", "target-b"):
            request = _request(request_id, list(range(16)))
            client.prefix_router.add_request(request, rank=1)
            client.prefix_router.set_resident(request_id, 1, True)

        await client._handle_dp_reload_event(
            DPReloadEvent(
                DPReloadEventType.INTENT,
                "source",
                0,
                0,
                num_preemptions=1,
                local_tokens=0,
            )
        )
        assert [item[0] for item in sent] == [EngineCoreRequestType.PREPARE_DP_RELOAD]
        assert sent[0][1].output_token_ids == [17, 18]
        assert client.reqs_in_flight["source"] == b"rank-0"

        await client._handle_dp_reload_event(
            DPReloadEvent(
                DPReloadEventType.PREPARED,
                "source",
                1,
                1,
                local_tokens=12,
            )
        )
        await asyncio.sleep(0)

        assert [item[0] for item in sent] == [
            EngineCoreRequestType.PREPARE_DP_RELOAD,
            EngineCoreRequestType.COMMIT_DP_RELOAD,
            EngineCoreRequestType.DROP_DP_RELOAD_SOURCE,
        ]
        assert client.reqs_in_flight["source"] == b"rank-1"
        assert client.prefix_router._requests["source"].rank == 1

        lookup_event = DPReloadEvent(
            DPReloadEventType.TARGET_LOOKUP,
            "source",
            1,
            1,
            local_tokens=12,
            external_tokens=4,
        )
        await client._handle_dp_reload_event(lookup_event)
        await client._handle_dp_reload_event(lookup_event)
        assert client.prefix_router.reload_committed_count == 1
        assert client.prefix_router.reload_source_tokens == 18
        assert client.prefix_router.reload_target_local_tokens == 12
        assert client.prefix_router.reload_saved_tokens == 12
        assert client.prefix_router.reload_source_external_tokens == 0
        assert client.prefix_router.reload_target_external_tokens == 4
        assert client.prefix_router.reload_saved_external_tokens == 0

    asyncio.run(run())


def test_dplb_client_rejects_reload_without_physical_prefix_gain(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_FORK_ATTN_DP_RELOAD_MIN_EXTERNAL_TOKENS", "4")

    async def run() -> None:
        client = object.__new__(DPLBAsyncMPClient)
        client.prefix_router = PrefixAwareDPRouter(2, 4, 32, 30, 1)
        client.engine_ranks_managed = [0, 1]
        client.reqs_in_flight = {"source": b"rank-0"}
        request = _request("source", list(range(16)))
        client.prefix_router.add_request(request, rank=0)
        assert client.prefix_router.detach_for_reload("source") == 0
        placement = _DPReloadPlacement(request, b"rank-0")
        placement.state = "PREPARING"
        placement.source_engine = b"rank-0"
        placement.source_rank = 0
        placement.source_epoch = 0
        placement.source_local_tokens = 12
        placement.target_engine = b"rank-1"
        placement.target_rank = 1
        placement.target_epoch = 1
        client._dp_reload_placements = {"source": placement}
        sent = []

        async def send(request_type, payload, engine):
            sent.append((request_type, payload, engine))

        client._send_input = send
        await client._handle_dp_reload_prepared(
            DPReloadEvent(
                DPReloadEventType.PREPARED,
                "source",
                1,
                1,
                local_tokens=12,
            )
        )

        assert [item[0] for item in sent] == [
            EngineCoreRequestType.CANCEL_DP_RELOAD,
            EngineCoreRequestType.RESUME_DP_RELOAD,
        ]
        assert client.reqs_in_flight["source"] == b"rank-0"
        assert client.prefix_router._requests["source"].rank == 0
        assert client.prefix_router.reload_failed_count == 1

    asyncio.run(run())
