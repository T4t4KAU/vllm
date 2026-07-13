# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm import envs
from vllm.config import KVTransferConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def _model_output(req_ids: list[str]) -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
        sampled_token_ids=[[0] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _make_scheduler_for_admission(*, fork: bool):
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=256,
        enable_prefix_caching=True,
        block_size=16,
    )
    if fork:
        scheduler.vllm_config.attention_config.backend = AttentionBackendEnum.FORK_ATTN
        scheduler.fork_fanout_admission_window = 4
        scheduler.fork_fanout_admission_max_bypasses = 8
        scheduler.fork_fanout_preemption_window = 4
        scheduler.fork_fanout_preemption_min_fanout = 2
        scheduler.fork_fanout_gpu_hotset_enabled = True
        scheduler.fork_fanout_gpu_hotset_min_fanout = 2
        scheduler.fork_fanout_gpu_hotset_min_reuse_blocks = 0
        scheduler.fork_fanout_gpu_hotset_min_usage = 0.0
        scheduler.fork_fanout_gpu_hotset_budget_blocks = 128
        scheduler.fork_fanout_reserved_blocks = []
    return scheduler


def _mark_resident(*requests, num_computed_tokens: int = 32) -> None:
    for request in requests:
        request.num_computed_tokens = num_computed_tokens


def test_fork_fanout_admission_prefers_shared_prefix_over_fcfs() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    active, shared = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["active", "shared"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]

    scheduler.add_request(active)
    first_output = scheduler.schedule()
    scheduler.update_from_output(first_output, _model_output(["active"]))

    scheduler.add_request(private)
    scheduler.add_request(shared)

    second_output = scheduler.schedule()

    assert [req.req_id for req in second_output.scheduled_new_reqs] == ["shared"]
    assert [request.request_id for request in scheduler.waiting] == ["private"]


def test_fork_fanout_admission_clusters_waiting_shared_prefix_group() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    shared_1, shared_2 = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]

    scheduler.add_request(private)
    scheduler.add_request(shared_1)
    scheduler.add_request(shared_2)

    output = scheduler.schedule()

    assert [req.req_id for req in output.scheduled_new_reqs] == [
        "shared_1",
        "shared_2",
    ]
    assert [request.request_id for request in scheduler.waiting] == ["private"]


def test_fork_fanout_admission_promotes_waiting_shared_prefix_cohort() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    scheduler.fork_fanout_admission_window = 8
    scheduler.fork_fanout_admission_max_bypasses = 8
    shared_1, shared_2, shared_3 = create_requests(
        3,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2", "shared_3"],
    )
    private, other = create_requests(
        3,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private", "other"],
    )[1:]
    for request in (private, shared_1, other, shared_2, shared_3):
        scheduler.waiting.add_request(request)

    scheduler._promote_fanout_waiting_request()

    assert [request.request_id for request in scheduler.waiting] == [
        "shared_1",
        "shared_2",
        "shared_3",
        "private",
        "other",
    ]
    assert scheduler.fork_fanout_admission_bypass_counts["private"] == 3


def test_fanout_hash_count_score_matches_pairwise_score() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    shared_1, shared_2, shared_3 = create_requests(
        3,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2", "shared_3"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]
    peers = [shared_2.block_hashes, shared_3.block_hashes, private.block_hashes]
    counts = Counter(shared_1.block_hashes)
    for peer in peers:
        counts.update(peer)

    assert scheduler._fanout_score_from_hash_counts(
        shared_1.block_hashes,
        counts,
    ) == scheduler._fanout_admission_score(shared_1.block_hashes, peers)


def test_fanout_waiting_demand_summarizes_window() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    scheduler.connector = object()
    shared_1, shared_2 = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2"],
    )
    scheduler.waiting.add_request(shared_1)
    scheduler.waiting.add_request(shared_2)

    demand = scheduler._build_fanout_waiting_demand()

    assert demand is not None
    assert all(demand[block_hash] == 2 for block_hash in shared_1.block_hashes)


def test_fork_fanout_admission_cohort_respects_bypass_limit() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    scheduler.fork_fanout_admission_max_bypasses = 2
    shared_1, shared_2, shared_3 = create_requests(
        3,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2", "shared_3"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]
    for request in (private, shared_1, shared_2, shared_3):
        scheduler.waiting.add_request(request)

    scheduler._promote_fanout_waiting_request()

    assert [request.request_id for request in scheduler.waiting] == [
        "shared_1",
        "shared_2",
        "private",
        "shared_3",
    ]
    assert scheduler.fork_fanout_admission_bypass_counts["private"] == 2


def test_fork_fanout_admission_head_shared_pulls_later_cohort_member() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    shared_1, shared_2 = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]
    for request in (shared_1, private, shared_2):
        scheduler.waiting.add_request(request)

    scheduler._promote_fanout_waiting_request()

    assert [request.request_id for request in scheduler.waiting] == [
        "shared_1",
        "shared_2",
        "private",
    ]
    assert not scheduler.fork_fanout_admission_bypass_counts


def test_fork_fanout_admission_reserves_cached_gpu_hotset(monkeypatch) -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    shared_1, shared_2 = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2"],
    )
    scheduler.waiting.add_request(shared_1)
    scheduler.waiting.add_request(shared_2)
    reservations: list[tuple[str, int]] = []
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "reserve_cached_prefix",
        lambda request, blocks: (
            reservations.append((request.request_id, blocks)) or [object(), object()]
        ),
    )

    scheduler._promote_fanout_waiting_request()

    assert len(reservations) == 1
    assert reservations[0][1] == 2


def test_fork_fanout_gpu_hotset_respects_budget(monkeypatch) -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    scheduler.fork_fanout_gpu_hotset_budget_blocks = 1
    shared_1, shared_2 = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2"],
    )
    scheduler.waiting.add_request(shared_1)
    scheduler.waiting.add_request(shared_2)
    reservations: list[int] = []
    monkeypatch.setattr(
        scheduler.kv_cache_manager,
        "reserve_cached_prefix",
        lambda _request, blocks: reservations.append(blocks) or [object()],
    )

    scheduler._promote_fanout_waiting_request()

    assert reservations == [1]


def test_fanout_admission_keeps_fcfs_for_non_fork_backend() -> None:
    scheduler = _make_scheduler_for_admission(fork=False)
    active, shared = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["active", "shared"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]

    scheduler.add_request(active)
    first_output = scheduler.schedule()
    scheduler.update_from_output(first_output, _model_output(["active"]))

    scheduler.add_request(private)
    scheduler.add_request(shared)

    second_output = scheduler.schedule()

    assert [req.req_id for req in second_output.scheduled_new_reqs] == ["private"]
    assert [request.request_id for request in scheduler.waiting] == ["shared"]


def test_fork_fanout_admission_uses_kv_connector_extra_config() -> None:
    scheduler = _make_scheduler_for_admission(fork=False)
    scheduler.vllm_config.attention_config.backend = AttentionBackendEnum.FORK_ATTN
    scheduler.vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "fanout_admission_window": 7,
            "fanout_admission_max_bypasses": 3,
        },
    )

    assert scheduler._init_fork_fanout_admission_config() == (7, 3)


def test_fork_fanout_preemption_uses_kv_connector_extra_config() -> None:
    scheduler = _make_scheduler_for_admission(fork=False)
    scheduler.vllm_config.attention_config.backend = AttentionBackendEnum.FORK_ATTN
    scheduler.vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "fanout_preemption_window": 5,
            "fanout_preemption_min_fanout": 4,
        },
    )

    assert scheduler._init_fork_fanout_preemption_config() == (5, 4)


def test_fork_fanout_gpu_hotset_uses_kv_connector_extra_config() -> None:
    scheduler = _make_scheduler_for_admission(fork=False)
    scheduler.vllm_config.attention_config.backend = AttentionBackendEnum.FORK_ATTN
    scheduler.vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "fanout_gpu_hotset_enabled": True,
            "fanout_gpu_hotset_min_fanout": 6,
            "fanout_gpu_hotset_min_reuse_blocks": 512,
            "fanout_gpu_hotset_min_usage": 0.75,
        },
    )

    assert scheduler._init_fork_fanout_gpu_hotset_config() == (
        True,
        6,
        512,
        0.75,
        512,
    )


def test_fork_fanout_policy_switch_disables_all_scheduler_policies(
    monkeypatch,
) -> None:
    scheduler = _make_scheduler_for_admission(fork=False)
    scheduler.vllm_config.attention_config.backend = AttentionBackendEnum.FORK_ATTN
    scheduler.vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="ExampleConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "fanout_admission_window": 7,
            "fanout_preemption_window": 5,
            "fanout_gpu_hotset_enabled": True,
        },
    )
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED", False)

    assert scheduler._init_fork_fanout_admission_config() == (0, 0)
    assert scheduler._init_fork_fanout_preemption_config() == (0, 0)
    assert scheduler._init_fork_fanout_gpu_hotset_config() == (
        False,
        0,
        0,
        1.0,
        0,
    )


def test_fork_fanout_preemption_prefers_private_running_victim() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    shared_1, shared_2 = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]
    _mark_resident(shared_1, shared_2, private)
    scheduler.running = [shared_1, shared_2, private]

    victim = scheduler._select_fanout_preemption_victim(shared_1)

    assert victim is private


def test_fork_fanout_preemption_uses_waiting_branches_as_signal() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    shared_running, shared_waiting = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_running", "shared_waiting"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]
    _mark_resident(shared_running, private)
    scheduler.running = [shared_running, private]
    scheduler.waiting.add_request(shared_waiting)

    victim = scheduler._select_fanout_preemption_victim(shared_running)

    assert victim is private


def test_fork_fanout_preemption_keeps_lifo_without_reuse_signal() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    private_1, private_2 = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["private_1", "private_2"],
    )
    _mark_resident(private_1, private_2)
    scheduler.running = [private_1, private_2]

    assert scheduler._select_fanout_preemption_victim(private_1) is None


def test_fork_fanout_preemption_is_used_on_allocation_failure(monkeypatch) -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    shared_1, shared_2 = create_requests(
        2,
        num_tokens=32,
        same_prompt=True,
        req_ids=["shared_1", "shared_2"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]
    for request in (shared_1, shared_2, private):
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = 32
        request.append_output_token_ids(0)
        scheduler.requests[request.request_id] = request
    scheduler.running = [shared_1, shared_2, private]

    allocate_calls = 0

    def allocate_slots(*args, **kwargs):
        nonlocal allocate_calls
        allocate_calls += 1
        if allocate_calls == 1:
            return None
        return scheduler.kv_cache_manager.empty_kv_cache_blocks

    monkeypatch.setattr(scheduler.kv_cache_manager, "allocate_slots", allocate_slots)

    scheduler.schedule()

    assert private.status == RequestStatus.PREEMPTED
    assert [request.request_id for request in scheduler.running] == [
        "shared_1",
        "shared_2",
    ]
    assert [request.request_id for request in scheduler.waiting] == ["private"]


def test_fork_fanout_admission_respects_bypass_limit() -> None:
    scheduler = _make_scheduler_for_admission(fork=True)
    scheduler.fork_fanout_admission_max_bypasses = 1
    active, shared_1, shared_2 = create_requests(
        3,
        num_tokens=32,
        same_prompt=True,
        req_ids=["active", "shared_1", "shared_2"],
    )
    private = create_requests(
        2,
        num_tokens=32,
        same_prompt=False,
        req_ids=["unused", "private"],
    )[1]
    scheduler.running.append(active)
    scheduler.waiting.add_request(private)
    scheduler.waiting.add_request(shared_1)

    scheduler._promote_fanout_waiting_request()

    assert [request.request_id for request in scheduler.waiting] == [
        "shared_1",
        "private",
    ]

    assert scheduler.waiting.pop_request().request_id == "shared_1"
    scheduler.waiting.add_request(shared_2)

    scheduler._promote_fanout_waiting_request()

    assert [request.request_id for request in scheduler.waiting] == [
        "private",
        "shared_2",
    ]
