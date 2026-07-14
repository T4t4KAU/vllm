# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    get_block_table_num_blocks,
)
from vllm.v1.worker.gpu.attn_utils import (
    _compute_fork_common_prefix_len,
    should_use_fork_dynamic_forest,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    CudaGraphManager,
    ForkGraphPlan,
    _get_fork_forest_max_splits,
)
from vllm.v1.worker.gpu.dp_utils import _resolve_synced_fork_plan


def _make_manager() -> CudaGraphManager:
    manager = CudaGraphManager.__new__(CudaGraphManager)
    manager.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=32768),
        attention_config=SimpleNamespace(backend=SimpleNamespace(name="FORK_ATTN")),
    )
    manager.compilation_config = SimpleNamespace(cudagraph_capture_sizes=[16])
    manager.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    manager.max_num_reqs = 16
    manager.decode_query_len = 1
    manager.lora_capture_cases = [0]
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    manager._uses_fork_attention = True
    manager._fork_prefix_chunk_buckets = (2, 4)
    manager._fork_forest_cta_buckets = (64, 128, 256, 384, 512)
    manager._fork_capture_plans = (
        ForkGraphPlan("common", 2),
        ForkGraphPlan("common", 4),
        ForkGraphPlan("forest", 64),
        ForkGraphPlan("forest", 128),
        ForkGraphPlan("forest", 256),
        ForkGraphPlan("forest", 384),
        ForkGraphPlan("forest", 512),
    )
    manager._candidates = {}
    manager._capture_descs = {}
    manager._graphs_captured = True
    manager._fork_dispatch_stats = defaultdict(int)
    return manager


def test_fork_graph_uses_aligned_block_table_capacity() -> None:
    assert get_block_table_num_blocks(max_model_len=11424, block_size=16) == 720


def test_fork_forest_graph_default_reserves_all_gather_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_FOREST_MAX_SPLITS", 0)

    assert _get_fork_forest_max_splits(block_size=16, max_model_len=11424) == 32


def test_fork_cudagraph_dispatch_has_flash_and_fork_decode_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH", True)
    manager = _make_manager()

    manager._init_candidates()

    flash_desc = manager.dispatch(
        num_reqs=16,
        num_tokens=16,
        uniform_token_count=1,
        num_active_loras=0,
        fork_plan=None,
    )
    assert flash_desc.cg_mode == CUDAGraphMode.FULL
    assert flash_desc.fork_plan is None

    fork_desc = manager.dispatch(
        num_reqs=16,
        num_tokens=16,
        uniform_token_count=1,
        num_active_loras=0,
        fork_plan=ForkGraphPlan("common", 2),
    )
    assert fork_desc.cg_mode == CUDAGraphMode.FULL
    assert fork_desc.fork_plan == ForkGraphPlan("common", 2)

    forest_desc = manager.dispatch(
        num_reqs=16,
        num_tokens=16,
        uniform_token_count=1,
        num_active_loras=0,
        fork_plan=ForkGraphPlan("forest", 512),
    )
    assert forest_desc.cg_mode == CUDAGraphMode.FULL
    assert forest_desc.fork_plan == ForkGraphPlan("forest", 512)


def test_fork_forest_graph_uses_smallest_cta_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH", True)
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 2048)
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_FOREST_MAX_SPLITS", 0)
    manager = _make_manager()

    assert (
        manager.get_fork_forest_cta_bucket(
            num_reqs=8,
            block_size=16,
            seq_lens=[4097] * 8,
        )
        == 64
    )
    assert (
        manager.get_fork_forest_cta_bucket(
            num_reqs=40,
            block_size=16,
            seq_lens=[8193] * 40,
        )
        == 384
    )


def test_fork_cudagraph_dispatch_pads_active_requests() -> None:
    manager = _make_manager()

    manager._init_candidates()

    fork_desc = manager.dispatch(
        num_reqs=13,
        num_tokens=13,
        uniform_token_count=1,
        num_active_loras=0,
        fork_plan=ForkGraphPlan("common", 2),
    )

    assert fork_desc.cg_mode == CUDAGraphMode.FULL
    assert fork_desc.num_reqs == 16
    assert fork_desc.num_tokens == 16
    assert fork_desc.fork_plan == ForkGraphPlan("common", 2)


def test_fork_cudagraph_dispatch_records_hits_and_misses() -> None:
    manager = _make_manager()
    manager._init_candidates()

    manager.dispatch(16, 16, 1, 0, ForkGraphPlan("common", 2))
    manager.dispatch(16, 16, 1, 0, ForkGraphPlan("common", 8))

    assert manager.get_fork_dispatch_stats() == {
        "hit:common": 1,
        "miss:common": 1,
        "miss_reason:plan_capacity": 1,
    }


def test_fork_cudagraph_dispatch_uses_larger_compatible_plan() -> None:
    manager = _make_manager()
    manager._fork_capture_plans = (ForkGraphPlan("common", 4),)
    manager._init_candidates()

    desc = manager.dispatch(16, 16, 1, 0, ForkGraphPlan("common", 3))

    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.fork_plan == ForkGraphPlan("common", 4)


def test_fork_cudagraph_capture_plan_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager()
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH", True)
    monkeypatch.setattr(
        envs,
        "VLLM_FORK_ATTN_CUDAGRAPH_CAPTURE_BUCKETS",
        "common:4,8;forest:256,512",
    )

    assert manager._init_fork_capture_plans() == (
        ForkGraphPlan("common", 4),
        ForkGraphPlan("common", 8),
        ForkGraphPlan("forest", 256),
        ForkGraphPlan("forest", 512),
    )


def test_fork_dp_plan_uses_max_capacity_for_matching_kind() -> None:
    matches, plan = _resolve_synced_fork_plan(
        torch.tensor([2, 2], dtype=torch.int32),
        torch.tensor([256, 512], dtype=torch.int32),
    )
    assert matches
    assert plan == ForkGraphPlan("forest", 512)


def test_fork_dp_plan_rejects_different_kinds() -> None:
    matches, plan = _resolve_synced_fork_plan(
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([4, 256], dtype=torch.int32),
    )
    assert not matches
    assert plan is None


def test_fork_common_prefix_ignores_cudagraph_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_reqs = 13
    padded_reqs = 16
    prefix_blocks = 455
    block_size = 16
    query_start_loc = torch.cat(
        (
            torch.arange(active_reqs + 1, dtype=torch.int32),
            torch.full(
                (padded_reqs - active_reqs,),
                active_reqs,
                dtype=torch.int32,
            ),
        )
    )
    seq_lens = torch.cat(
        (
            torch.full((active_reqs,), 8193, dtype=torch.int32),
            torch.zeros(padded_reqs - active_reqs, dtype=torch.int32),
        )
    )
    metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        num_reqs=padded_reqs,
        num_actual_tokens=padded_reqs,
        max_query_len=1,
        max_seq_len=8193,
        block_table_tensor=torch.zeros(
            (padded_reqs, 512),
            dtype=torch.int32,
        ),
        slot_mapping=torch.zeros(padded_reqs, dtype=torch.int64),
        seq_lens_cpu_upper_bound=seq_lens,
    )
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.float16,
    )
    builder = SimpleNamespace(
        num_heads_q=16,
        use_cascade_attention=lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.attn_utils.current_platform.num_compute_units",
        lambda: 48,
    )

    prefix_len = _compute_fork_common_prefix_len(
        prefix_blocks,
        metadata,
        spec,
        builder,
    )

    assert metadata.num_active_reqs() == active_reqs
    assert prefix_len == prefix_blocks * block_size


def test_fork_prefix_bucket_returns_none_when_prefix_exceeds_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager()
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 2048)

    assert (
        manager.get_fork_prefix_chunk_bucket(
            prefix_blocks=32768 // 16,
            block_size=16,
        )
        is None
    )


def test_fork_dynamic_forest_uses_dynamic_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _make_manager()
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_ENABLE_FOREST", True)

    assert should_use_fork_dynamic_forest(
        manager.vllm_config,
        num_reqs=8,
        uniform_token_count=1,
        fork_plan=None,
    )
    assert not should_use_fork_dynamic_forest(
        manager.vllm_config,
        num_reqs=8,
        uniform_token_count=1,
        fork_plan=ForkGraphPlan("common", 2),
    )
    assert not should_use_fork_dynamic_forest(
        manager.vllm_config,
        num_reqs=8,
        uniform_token_count=1,
        fork_plan=ForkGraphPlan("forest", 512),
    )
    assert not should_use_fork_dynamic_forest(
        manager.vllm_config,
        num_reqs=1,
        uniform_token_count=1,
        fork_plan=None,
    )
