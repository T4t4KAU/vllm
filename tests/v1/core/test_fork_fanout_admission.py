# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config import KVTransferConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.outputs import ModelRunnerOutput

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
    return scheduler


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
