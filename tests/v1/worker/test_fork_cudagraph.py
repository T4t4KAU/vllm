# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    ForkGraphPlan,
    _get_fork_graph_capture_plan,
    _is_compatible,
)


def test_fork_cudagraph_has_one_forest_plan_with_two_shape_buckets() -> None:
    assert _get_fork_graph_capture_plan(1, 256, 4) is None
    assert _get_fork_graph_capture_plan(2, 256, 4) == ForkGraphPlan(16, 4)
    assert _get_fork_graph_capture_plan(32, 256, 4) == ForkGraphPlan(64, 4)
    assert _get_fork_graph_capture_plan(256, 256, 4) == ForkGraphPlan(512, 4)
    assert _get_fork_graph_capture_plan(257, 256, 4) is None
    assert _get_fork_graph_capture_plan(2, 256, 0) is None
    assert not hasattr(ForkGraphPlan(2, 4), "kind")


def test_fork_cudagraph_capacity_compatibility_is_one_way() -> None:
    captured = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=64,
        num_reqs=64,
        uniform_token_count=1,
        fork_plan=ForkGraphPlan(128, 8),
    )

    assert _is_compatible(captured, 32, 32, 1, 0, ForkGraphPlan(64, 4))
    assert not _is_compatible(captured, 32, 32, 1, 0, ForkGraphPlan(256, 4))
    assert not _is_compatible(captured, 32, 32, 1, 0, ForkGraphPlan(64, 16))
    assert not _is_compatible(captured, 32, 32, 1, 0, None)


def test_flash_cudagraph_does_not_match_a_fork_plan() -> None:
    captured = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=64,
        num_reqs=64,
        uniform_token_count=1,
    )

    assert _is_compatible(captured, 32, 32, 1, 0)
    assert not _is_compatible(captured, 32, 32, 1, 0, ForkGraphPlan(32, 4))


def _mock_dp_reduce(
    monkeypatch: pytest.MonkeyPatch,
    remote_column: list[int],
) -> None:
    monkeypatch.setattr(
        dp_utils,
        "get_dp_group",
        lambda: SimpleNamespace(cpu_group=object()),
    )

    def all_reduce(tensor: torch.Tensor, group: object) -> None:
        tensor[:, 1] = torch.tensor(remote_column, dtype=torch.int32)

    monkeypatch.setattr(dp_utils.dist, "all_reduce", all_reduce)


def test_dp_mixed_fork_and_flash_graphs_fall_back_to_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_dp_reduce(monkeypatch, [8, CUDAGraphMode.FULL.value, 1, 0, 0])
    desired = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=8,
        uniform_token_count=1,
        fork_plan=ForkGraphPlan(16, 4),
    )

    synced, _ = dp_utils.sync_cudagraph_and_dp_padding(
        MagicMock(),
        desired,
        num_tokens=8,
        num_reqs=8,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        fork_plan=desired.fork_plan,
    )

    assert synced.cg_mode == CUDAGraphMode.NONE
    assert synced.fork_plan is None


def test_dp_fork_graph_uses_the_largest_shape_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_dp_reduce(monkeypatch, [8, CUDAGraphMode.FULL.value, 1, 32, 8])
    manager = MagicMock()
    expected = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=8,
        uniform_token_count=1,
        fork_plan=ForkGraphPlan(32, 8),
    )
    manager.dispatch.return_value = expected
    desired = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=8,
        uniform_token_count=1,
        fork_plan=ForkGraphPlan(16, 4),
    )

    synced, _ = dp_utils.sync_cudagraph_and_dp_padding(
        manager,
        desired,
        num_tokens=8,
        num_reqs=8,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        fork_plan=desired.fork_plan,
    )

    assert synced == expected
    manager.dispatch.assert_called_once()
    args = manager.dispatch.call_args.args
    kwargs = manager.dispatch.call_args.kwargs
    assert args[:2] == (8, 8)
    assert int(args[2]) == 1
    assert kwargs == {
        "num_active_loras": 0,
        "fork_plan": ForkGraphPlan(32, 8),
    }
