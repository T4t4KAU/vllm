# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestGroupState,
)


def test_cpu_eviction_makes_fanout_block_eligible_again() -> None:
    scheduler = object.__new__(OffloadingConnectorScheduler)
    key = (b"shared", 0)
    group_state = RequestGroupState(offload_keys=[key])
    req_status = SimpleNamespace(group_states=(group_state,))
    scheduler._req_status = {"request": req_status}
    scheduler._fanout_admitted_keys = set()
    scheduler._fanout_admitted_locations = {}

    scheduler._record_fanout_admission(key, "request", 0, group_state, logical_idx=0)
    assert group_state.fanout_admitted_block_indices == {0}

    scheduler._invalidate_fanout_admissions([key])

    assert group_state.fanout_admitted_block_indices == set()
    assert scheduler._fanout_admitted_keys == set()
    assert scheduler._fanout_admitted_locations == {}


def test_finished_request_removes_reverse_admission_state() -> None:
    scheduler = object.__new__(OffloadingConnectorScheduler)
    key = (b"shared", 0)
    group_state = RequestGroupState(offload_keys=[key])
    req_status = SimpleNamespace(group_states=(group_state,))
    scheduler._req_status = {"request": req_status}
    scheduler._fanout_admitted_keys = set()
    scheduler._fanout_admitted_locations = {}
    scheduler._record_fanout_admission(key, "request", 0, group_state, logical_idx=0)

    scheduler._forget_fanout_request("request", req_status)

    assert scheduler._fanout_admitted_locations == {}
    assert scheduler._fanout_admitted_keys == {key}


def test_shared_requests_reuse_one_cpu_admission() -> None:
    scheduler = object.__new__(OffloadingConnectorScheduler)
    key = (b"shared", 0)
    first_group = RequestGroupState(offload_keys=[key])
    second_group = RequestGroupState(offload_keys=[key])
    scheduler._req_status = {
        "first": SimpleNamespace(group_states=(first_group,)),
        "second": SimpleNamespace(group_states=(second_group,)),
    }
    scheduler._fanout_admitted_keys = set()
    scheduler._fanout_admitted_locations = {}
    scheduler._record_fanout_admission(key, "first", 0, first_group, logical_idx=0)

    reused = scheduler._reuse_fanout_admission(
        key, "second", 0, second_group, logical_idx=0
    )

    assert reused
    assert second_group.fanout_admitted_block_indices == {0}
    assert scheduler._fanout_admitted_locations[key] == {
        ("first", 0, 0),
        ("second", 0, 0),
    }
