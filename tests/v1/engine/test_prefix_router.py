import asyncio
from types import SimpleNamespace

from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient
from vllm.v1.engine.prefix_router import PrefixAwareDPRouter


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
    published = []
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
