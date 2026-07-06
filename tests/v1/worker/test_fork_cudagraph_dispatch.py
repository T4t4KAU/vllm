# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.attn_utils import (
    _compute_fork_common_prefix_len,
    should_use_fork_dynamic_forest,
)
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


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
    manager._candidates = {}
    manager._capture_descs = {}
    manager._graphs_captured = True
    return manager


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
        fork_prefix_chunk_bucket=None,
    )
    assert flash_desc.cg_mode == CUDAGraphMode.FULL
    assert flash_desc.fork_prefix_chunk_bucket is None

    fork_desc = manager.dispatch(
        num_reqs=16,
        num_tokens=16,
        uniform_token_count=1,
        num_active_loras=0,
        fork_prefix_chunk_bucket=2,
    )
    assert fork_desc.cg_mode == CUDAGraphMode.FULL
    assert fork_desc.fork_prefix_chunk_bucket == 2
    assert fork_desc.fork_forest_cta_bucket is None

    forest_desc = manager.dispatch(
        num_reqs=16,
        num_tokens=16,
        uniform_token_count=1,
        num_active_loras=0,
        fork_forest_cta_bucket=512,
    )
    assert forest_desc.cg_mode == CUDAGraphMode.FULL
    assert forest_desc.fork_prefix_chunk_bucket is None
    assert forest_desc.fork_forest_cta_bucket == 512


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
        fork_prefix_chunk_bucket=2,
    )

    assert fork_desc.cg_mode == CUDAGraphMode.FULL
    assert fork_desc.num_reqs == 16
    assert fork_desc.num_tokens == 16
    assert fork_desc.fork_prefix_chunk_bucket == 2


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
        fork_prefix_chunk_bucket=None,
    )
    assert not should_use_fork_dynamic_forest(
        manager.vllm_config,
        num_reqs=8,
        uniform_token_count=1,
        fork_prefix_chunk_bucket=2,
    )
    assert not should_use_fork_dynamic_forest(
        manager.vllm_config,
        num_reqs=8,
        uniform_token_count=1,
        fork_prefix_chunk_bucket=None,
        fork_forest_cta_bucket=512,
    )
    assert not should_use_fork_dynamic_forest(
        manager.vllm_config,
        num_reqs=1,
        uniform_token_count=1,
        fork_prefix_chunk_bucket=None,
    )
