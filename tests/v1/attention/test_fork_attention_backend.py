# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config.attention import AttentionConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends import fork_attn as fork_attn_backend
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.fork_attn import (
    ForkAttentionBackend,
    ForkAttentionImpl,
    ForkAttentionMetadata,
    ForkAttentionMetadataBuilder,
    _build_fork_plan,
    _flash_metadata_kwargs,
    _ForkCUDAGraphWorkspace,
    _ForkPlanError,
    _ForkSegment,
    _get_plan_cudagraph_requirements,
    _pack_fork_plan,
    _validate_fork_plan,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.attn_utils import (
    configure_fork_attention_cudagraph,
    prepare_fork_attention_cudagraph_plan,
)
from vllm.v1.worker.utils import AttentionGroup


def test_fork_backend_is_registered_without_full_cudagraph_support() -> None:
    assert AttentionBackendEnum.FORK_ATTN.get_class().get_name() == "FORK_ATTN"
    assert (
        ForkAttentionMetadataBuilder.get_cudagraph_support(None, None)
        == AttentionCGSupport.NEVER
    )
    assert ForkAttentionBackend.get_supported_head_sizes() == [64, 128, 256]
    assert ForkAttentionBackend.supports_head_size(64)
    assert not ForkAttentionBackend.supports_head_size(80)
    assert ForkAttentionBackend.supports_block_size(16)
    assert ForkAttentionBackend.supports_block_size(32)
    assert not ForkAttentionBackend.supports_block_size(8)
    assert ForkAttentionBackend.supports_attn_type(AttentionType.DECODER)
    assert not ForkAttentionBackend.supports_attn_type(AttentionType.ENCODER)


def test_fork_backend_enables_only_v2_single_token_cudagraphs() -> None:
    v2_config = SimpleNamespace(use_v2_model_runner=True)
    legacy_config = SimpleNamespace(use_v2_model_runner=False)

    assert (
        ForkAttentionMetadataBuilder.get_cudagraph_support(v2_config, None)
        == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )
    assert (
        ForkAttentionMetadataBuilder.get_cudagraph_support(legacy_config, None)
        == AttentionCGSupport.NEVER
    )


@pytest.mark.parametrize(
    "capability, is_cuda, adaptive_verification, expected_reason",
    [
        pytest.param(
            DeviceCapability(9, 0),
            False,
            False,
            "compute capability not supported",
            id="gfx90a",
        ),
        pytest.param(
            DeviceCapability(9, 4),
            False,
            False,
            "compute capability not supported",
            id="gfx942",
        ),
        pytest.param(
            DeviceCapability(11, 0),
            False,
            False,
            "compute capability not supported",
            id="gfx1100",
        ),
        pytest.param(
            DeviceCapability(12, 0),
            True,
            True,
            "device-cpu query lens mismatch not supported, "
            "this is needed for adaptive verification",
            id="adaptive-verification",
        ),
    ],
)
def test_fork_backend_rejects_incompatible_configuration(
    capability: DeviceCapability,
    is_cuda: bool,
    adaptive_verification: bool,
    expected_reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fork_attn_backend.current_platform, "is_cuda", lambda: is_cuda)

    invalid_reasons = ForkAttentionBackend.validate_configuration(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        device_capability=capability,
        attn_type=AttentionType.DECODER,
        use_adaptive_verification=adaptive_verification,
    )

    assert invalid_reasons == [expected_reason]


def test_fork_plan_preserves_hierarchical_prefixes_and_partial_blocks() -> None:
    plan = _build_fork_plan(
        query_start_locs=[0, 1, 2, 3],
        seq_lens=[33, 33, 33],
        block_rows=[
            [10, 11, 20, 999],
            [10, 11, 21, 999],
            [10, 12, 22, 999],
        ],
        num_actual_tokens=3,
        block_size=16,
        head_ratio=4,
        require_shared=True,
    )

    assert plan is not None
    assert plan.num_splits_per_query == (3, 3, 3)
    assert plan.max_block_id == 22
    by_query: list[dict[int, _ForkSegment]] = [{}, {}, {}]
    for segment in plan.segments:
        for query_id in segment.query_ids:
            assert 0 <= query_id < 3
            assert segment.rank not in by_query[query_id]
            by_query[query_id][segment.rank] = segment
    assert [sorted(segments) for segments in by_query] == [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
    ]
    assert [
        [block for rank in sorted(segments) for block in segments[rank].block_ids]
        for segments in by_query
    ] == [[10, 11, 20], [10, 11, 21], [10, 12, 22]]


def test_fork_plan_uses_query_offsets_instead_of_request_indices() -> None:
    plan = _build_fork_plan(
        query_start_locs=[0, 1, 1, 2],
        seq_lens=[33, 0, 33],
        block_rows=[[10, 11, 20], [], [10, 11, 21]],
        num_actual_tokens=2,
        block_size=16,
        head_ratio=4,
        require_shared=True,
    )

    assert plan is not None
    assert plan.num_splits_per_query == (2, 2)
    assert {
        query_id for segment in plan.segments for query_id in segment.query_ids
    } == {0, 1}


def test_fork_plan_falls_back_without_shared_blocks() -> None:
    plan = _build_fork_plan(
        query_start_locs=[0, 1, 2],
        seq_lens=[16, 16],
        block_rows=[[10], [11]],
        num_actual_tokens=2,
        block_size=16,
        head_ratio=4,
        require_shared=True,
    )
    assert plan is None


def test_fork_plan_falls_back_for_non_decode_queries() -> None:
    plan = _build_fork_plan(
        query_start_locs=[0, 1, 3],
        seq_lens=[16, 17],
        block_rows=[[10], [11, 12]],
        num_actual_tokens=3,
        block_size=16,
        head_ratio=4,
        require_shared=False,
    )
    assert plan is None


def test_fork_plan_rejects_invalid_active_block_id() -> None:
    with pytest.raises(_ForkPlanError, match="non-negative"):
        _build_fork_plan(
            query_start_locs=[0, 1, 2],
            seq_lens=[16, 16],
            block_rows=[[-1], [-1]],
            num_actual_tokens=2,
            block_size=16,
            head_ratio=4,
            require_shared=True,
        )


@pytest.mark.parametrize(
    ("segments", "num_splits", "error"),
    [
        (
            [_ForkSegment((1,), (10,), 0, 16, 1)],
            [1],
            "outside the query tensor",
        ),
        (
            [
                _ForkSegment((0,), (10,), 0, 16, 1),
                _ForkSegment((0,), (11,), 0, 16, 1),
            ],
            [2],
            "duplicate CTA ranks",
        ),
        (
            [_ForkSegment((0,), (10,), 0, 16, 1)],
            [2],
            "does not match the emitted CTA count",
        ),
        (
            [_ForkSegment((0,), (10,), 1, 16, 1)],
            [1],
            "continuous, unique ranks",
        ),
    ],
)
def test_fork_plan_validator_rejects_broken_query_rank_invariants(
    segments: list[_ForkSegment],
    num_splits: list[int],
    error: str,
) -> None:
    with pytest.raises(_ForkPlanError, match=error):
        _validate_fork_plan(
            segments,
            num_splits,
            {0: [10]},
            {0: 16},
            num_actual_tokens=1,
            block_size=16,
        )


def test_fork_plan_keeps_a_tail_cohort_in_the_full_tile_group() -> None:
    batch_size = 20
    plan = _build_fork_plan(
        query_start_locs=list(range(batch_size + 1)),
        seq_lens=[16] * batch_size,
        block_rows=[[7] for _ in range(batch_size)],
        num_actual_tokens=batch_size,
        block_size=16,
        head_ratio=4,
        require_shared=True,
    )
    assert plan is not None
    assert [len(segment.query_ids) for segment in plan.segments] == [16, 4]
    assert [segment.tile_num_queries for segment in plan.segments] == [16, 16]

    metadata = _pack_fork_plan(
        plan,
        num_heads_q=32,
        num_heads_kv=8,
        head_dim=128,
        page_block_size=16,
        device=torch.device("cpu"),
    )
    assert metadata["fork_mnw"] == [64, 32, 4]
    assert metadata["fork_query_tables"][0].shape == (2, 16)
    assert metadata["fork_num_seqs_per_ctas"][0].tolist() == [16, 4]
    packed_ranks: list[list[int]] = [[] for _ in range(batch_size)]
    for query_table, num_seqs, cta_ranks in zip(
        metadata["fork_query_tables"],
        metadata["fork_num_seqs_per_ctas"],
        metadata["fork_cta_ranks"],
    ):
        for cta_id, num_queries in enumerate(num_seqs.tolist()):
            rank = int(cta_ranks[cta_id])
            for query_id in query_table[cta_id, :num_queries].tolist():
                packed_ranks[query_id].append(rank)
    assert packed_ranks == [[0] for _ in range(batch_size)]


def test_metadata_builder_uses_exact_seq_lens_and_active_block_table() -> None:
    # GPU metadata deliberately differs from the runner-owned CPU snapshot.
    # Planning must use the CPU values without copying GPU tensors to the host.
    block_table = torch.tensor([[90, 91, 92], [90, 93, 94]], dtype=torch.int32)
    seq_lens = torch.full((2,), 33, dtype=torch.int32)
    metadata = _make_base_metadata(block_table, seq_lens)
    block_table_cpu = torch.tensor(
        [[10, 12, 81], [10, 11, 80]], dtype=torch.int32
    ).numpy()
    block_table_indices = np.array([1, 0], dtype=np.int32)
    seq_lens_cpu = torch.full((2,), 17, dtype=torch.int32)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    fork_kwargs = builder._build_fork_kwargs(
        metadata,
        torch.arange(3, dtype=torch.int32),
        2,
        (seq_lens_cpu, block_table_cpu, block_table_indices),
    )

    assert fork_kwargs["fork_enabled"]
    assert fork_kwargs["fork_max_block_id"] == 12
    assert sorted(
        kv_tokens
        for group in fork_kwargs["fork_kv_in_ctas"]
        for kv_tokens in group.tolist()
    ) == [1, 1, 16]


def test_metadata_builder_reuses_persistent_buffers() -> None:
    block_table = torch.tensor([[10, 11], [10, 12]], dtype=torch.int32)
    seq_lens = torch.full((2,), 32, dtype=torch.int32)
    metadata = _make_base_metadata(block_table, seq_lens)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )

    first = builder._build_fork_kwargs(
        metadata,
        torch.arange(3, dtype=torch.int32),
        2,
        _cpu_metadata(block_table, seq_lens),
    )
    second = builder._build_fork_kwargs(
        metadata,
        torch.arange(3, dtype=torch.int32),
        2,
        _cpu_metadata(block_table, seq_lens),
    )

    for key in (
        "fork_num_split_per_seq",
        "fork_softmax_lse",
        "fork_split_out",
        "fork_split_lse",
    ):
        assert first[key].data_ptr() == second[key].data_ptr()
    for key in (
        "fork_query_tables",
        "fork_block_tables",
        "fork_num_seqs_per_ctas",
        "fork_cta_ranks",
        "fork_kv_in_ctas",
    ):
        assert [tensor.data_ptr() for tensor in first[key]] == [
            tensor.data_ptr() for tensor in second[key]
        ]


def test_cudagraph_workspace_uses_one_fixed_forest_layout() -> None:
    def make_plan(length=33, private=20):
        plan = _build_fork_plan(
            query_start_locs=[0, 1, 2],
            seq_lens=[length, length],
            block_rows=[[10, 11, private], [10, 11, 21]],
            num_actual_tokens=2,
            block_size=16,
            head_ratio=4,
            require_shared=True,
        )
        assert plan is not None
        return plan

    def new_workspace():
        return _ForkCUDAGraphWorkspace(
            num_heads_q=16,
            num_heads_kv=4,
            head_dim=128,
            block_size=16,
            max_model_len=64,
            max_queries=8,
            max_ctas=4,
            max_splits=4,
            device=torch.device("cpu"),
        )

    plan = make_plan()
    workspace = new_workspace()

    empty = workspace.pack(None, query_capacity=4, cta_capacity=4, split_capacity=4)
    packed = workspace.pack(plan, query_capacity=4, cta_capacity=4, split_capacity=4)

    assert empty["fork_mnw"] == packed["fork_mnw"]
    assert len(packed["fork_query_tables"]) == 2
    assert [tensor.shape[0] for tensor in packed["fork_query_tables"]] == [1, 4]
    assert [tensor.data_ptr() for tensor in empty["fork_query_tables"]] == [
        tensor.data_ptr() for tensor in packed["fork_query_tables"]
    ]
    assert packed["fork_num_split_per_seq"].tolist() == [2, 2, 0, 0]
    assert sum(
        int((num_seqs > 0).sum()) for num_seqs in packed["fork_num_seqs_per_ctas"]
    ) == len(plan.segments)
    cta_requirement, split_requirement = _get_plan_cudagraph_requirements(
        plan,
        head_ratio=4,
        head_dim=128,
        block_size=16,
    )
    assert cta_requirement <= 4
    assert split_requirement == 2

    def check_repack(plan, capacity=4):
        kwargs = dict(query_capacity=4, cta_capacity=capacity, split_capacity=4)
        actual = workspace.pack(plan, **kwargs)
        expected = new_workspace().pack(plan, **kwargs)
        for key in (
            "fork_query_tables",
            "fork_block_tables",
            "fork_num_seqs_per_ctas",
            "fork_cta_ranks",
            "fork_kv_in_ctas",
        ):
            for result, reference in zip(actual[key], expected[key]):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            actual["fork_num_split_per_seq"],
            expected["fork_num_split_per_seq"],
            rtol=0,
            atol=0,
        )

    # Reused rows must match a fresh pack through length changes, remapping,
    # shrinking, capture resets, and capacity changes.
    check_repack(plan)
    check_repack(make_plan(34))
    check_repack(make_plan(34, private=22))
    check_repack(make_plan(16))
    check_repack(make_plan(16), capacity=2)
    check_repack(None, capacity=2)
    check_repack(plan)
    bad = replace(
        plan,
        segments=(
            replace(plan.segments[0], block_ids=(30, 31)),
            replace(plan.segments[1], block_ids=tuple(range(33)), num_kv_tokens=528),
            *plan.segments[2:],
        ),
    )
    with pytest.raises(fork_attn_backend._ForkWorkspaceLimitError, match="block table"):
        workspace.pack(bad, query_capacity=4, cta_capacity=4, split_capacity=4)
    # A partially written failed pack cannot leave a valid cache snapshot.
    check_repack(plan)


def test_cudagraph_workspace_rejects_insufficient_split_capacity() -> None:
    plan = _build_fork_plan(
        query_start_locs=[0, 1, 2],
        seq_lens=[33, 33],
        block_rows=[[10, 11, 20], [10, 11, 21]],
        num_actual_tokens=2,
        block_size=16,
        head_ratio=4,
        require_shared=True,
    )
    assert plan is not None
    workspace = _ForkCUDAGraphWorkspace(
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
        block_size=16,
        max_model_len=64,
        max_queries=8,
        max_ctas=4,
        max_splits=4,
        device=torch.device("cpu"),
    )

    with pytest.raises(
        fork_attn_backend._ForkWorkspaceLimitError,
        match="split capacity",
    ):
        workspace.pack(
            plan,
            query_capacity=4,
            cta_capacity=4,
            split_capacity=1,
        )


@pytest.mark.parametrize("shared", [True, False])
def test_metadata_builder_consumes_prebuilt_decision_without_replanning(
    shared: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_table = torch.tensor(
        [[10, 11], [10 if shared else 13, 12]], dtype=torch.int32
    )
    seq_lens = torch.full((2,), 32, dtype=torch.int32)
    metadata = _make_base_metadata(block_table, seq_lens)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    requirement = builder.prepare_cudagraph_plan(
        seq_lens,
        block_table.numpy(),
        np.arange(2, dtype=np.int32),
        32,
    )
    assert (requirement is not None) == shared
    assert builder._fork_prebuilt_result is not None

    def fail_rebuild(*args, **kwargs):
        pytest.fail("the exact forest must not be rebuilt after dispatch")

    monkeypatch.setattr(fork_attn_backend, "_build_fork_plan", fail_rebuild)
    monkeypatch.setattr(
        FlashAttentionMetadataBuilder, "build", lambda *a, **kw: metadata
    )
    builder.set_cpu_metadata(*_cpu_metadata(block_table, seq_lens))
    result = builder.build(
        0, SimpleNamespace(query_start_loc_cpu=torch.arange(3), num_reqs=2)
    )
    assert result.fork_enabled == shared
    assert builder._fork_prebuilt_result is None
    assert builder._fork_cpu_metadata is None


@pytest.mark.parametrize("first_group_shared", [True, False])
def test_group_planning_keeps_decisions_and_checks_kernel_page_bounds(
    first_group_shared,
):
    builders = [
        _make_builder(block_size=size, num_heads_q=16, num_heads_kv=4, head_dim=128)
        for size in (16, 32)
    ]
    groups = [
        [
            AttentionGroup(
                ForkAttentionBackend, [str(i)], builder.kv_cache_spec, i, [builder]
            )
        ]
        for i, builder in enumerate(builders)
    ]
    rows = [
        np.array([[10, 11], [10 if first_group_shared else 12, 13]], dtype=np.int32),
        np.array([[6], [6]], dtype=np.int32),
    ]
    lengths = np.full(2, 32, dtype=np.int32)
    indices = np.arange(2, dtype=np.int32)
    # Eight manager blocks: group 0 splits each into two kernel pages; group 1
    # keeps one. Pages 10..13 are valid only in the first group.
    requirement = prepare_fork_attention_cudagraph_plan(
        groups, lengths, rows, indices, [16, 8]
    )
    assert (requirement is not None) == first_group_shared
    configure_fork_attention_cudagraph(
        groups, None, None, discard_prebuilt_plan=False, force_flash=False
    )
    for builder, eligible in zip(builders, [first_group_shared, True]):
        assert builder._fork_prebuilt_result is not None
        assert (builder._fork_prebuilt_result.plan is not None) == eligible
    configure_fork_attention_cudagraph(
        groups, None, None, discard_prebuilt_plan=True, force_flash=True
    )
    assert all(builder._fork_prebuilt_result is None for builder in builders)
    rows[1][:] = 8
    with pytest.raises(_ForkPlanError, match="invalid block ID"):
        prepare_fork_attention_cudagraph_plan(groups, lengths, rows, indices, [16, 8])


def test_cudagraph_plan_rejects_block_ids_outside_the_kv_cache() -> None:
    block_table = np.array([[10, 11], [10, 12]], dtype=np.int32)
    seq_lens = np.full(2, 32, dtype=np.int32)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )

    with pytest.raises(_ForkPlanError, match="invalid block ID"):
        builder.prepare_cudagraph_plan(
            seq_lens,
            block_table,
            np.arange(2, dtype=np.int32),
            num_kernel_blocks=12,
        )

    assert builder._fork_prebuilt_result is None


def test_flash_cudagraph_fallback_does_not_rebuild_the_forest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_table = torch.tensor([[10, 11], [10, 12]], dtype=torch.int32)
    seq_lens = torch.full((2,), 32, dtype=torch.int32)
    metadata = _make_base_metadata(block_table, seq_lens)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    builder.set_cudagraph_plan(None, None, force_flash=True)

    def fail_rebuild(*args, **kwargs):
        pytest.fail("a Flash CUDA graph must not rebuild the forest")

    monkeypatch.setattr(fork_attn_backend, "_build_fork_plan", fail_rebuild)
    result = builder._build_fork_kwargs(
        metadata,
        torch.arange(3, dtype=torch.int32),
        2,
        _cpu_metadata(block_table, seq_lens),
    )

    assert result == {}


def test_metadata_builder_falls_back_above_workspace_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_table = torch.tensor([[10, 11], [10, 12]], dtype=torch.int32)
    seq_lens = torch.full((2,), 32, dtype=torch.int32)
    metadata = _make_base_metadata(block_table, seq_lens)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    monkeypatch.setattr(fork_attn_backend, "_MAX_FORK_WORKSPACE_BYTES", 1)

    assert (
        builder._build_fork_kwargs(
            metadata,
            torch.arange(3, dtype=torch.int32),
            2,
            _cpu_metadata(block_table, seq_lens),
        )
        == {}
    )


def test_cudagraph_is_disabled_above_workspace_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    monkeypatch.setattr(fork_attn_backend, "_MAX_FORK_WORKSPACE_BYTES", 1)

    assert builder.get_cudagraph_max_reqs() == 0


def test_cudagraph_workspace_matches_actual_capture_sizes() -> None:
    builder = _make_builder(
        block_size=16,
        num_heads_q=64,
        num_heads_kv=8,
        head_dim=256,
    )
    builder.vllm_config.model_config.max_model_len = 4096
    builder.vllm_config.scheduler_config.max_num_seqs = 127
    builder.vllm_config.compilation_config.cudagraph_capture_sizes = [16]

    assert builder._get_cudagraph_capture_limits() == (16, 128, 16)

    workspace = builder._get_fork_cudagraph_workspace(torch.device("cpu"))
    assert workspace.max_queries == 16
    assert workspace.max_ctas == 128
    assert workspace.max_splits == 16
    assert workspace.split_out.shape == (16, 64, 16, 256)
    assert [workspace.groups[tile].cta_capacity for tile in (32, 16)] == [32, 128]


@pytest.mark.parametrize(
    "max_len,max_seqs,min_prefix,min_queries,eligible",
    [
        (4096, 64, 16384, 8, False),
        (32768, 4, 16384, 8, False),
        (16385, 64, 16385, 8, False),
        (64, 64, 0, 2, True),
        (32768, 8, 16384, 8, True),
    ],
)
def test_fork_capture_requires_reachable_admission(
    max_len, max_seqs, min_prefix, min_queries, eligible
):
    builder = _make_builder(block_size=16, num_heads_q=32, num_heads_kv=8, head_dim=128)
    builder.vllm_config.attention_config = AttentionConfig(
        fork_min_shared_tokens=min_prefix, fork_min_queries=min_queries
    )
    builder.vllm_config.model_config.max_model_len = max_len
    builder.vllm_config.scheduler_config.max_num_seqs = max_seqs
    assert (builder.get_cudagraph_max_reqs() > 0) == eligible
    assert (builder.get_cudagraph_max_splits() > 0) == eligible


def test_cudagraph_workspace_uses_common_attention_group_limits() -> None:
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    builder.set_cudagraph_workspace_limits(max_queries=4, max_splits=2)

    workspace = builder._get_fork_cudagraph_workspace(torch.device("cpu"))
    assert workspace.max_queries == 4
    assert workspace.max_ctas == 16
    assert workspace.max_splits == 2
    assert workspace.split_out.shape == (4, 16, 2, 128)


def test_cudagraph_workspace_keeps_only_capture_sizes_within_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    builder.vllm_config.compilation_config.cudagraph_capture_sizes = [2, 16]
    size_for_two_queries = fork_attn_backend._fork_workspace_bytes(
        2, 16, 128, 4
    ) + fork_attn_backend._fork_cudagraph_metadata_bytes(
        max_model_len=64,
        block_size=16,
        head_ratio=4,
        cta_capacity=16,
    )
    monkeypatch.setattr(
        fork_attn_backend,
        "_MAX_FORK_WORKSPACE_BYTES",
        size_for_two_queries,
    )

    assert builder._get_cudagraph_capture_limits() == (2, 16, 4)


def test_metadata_builder_consumes_cpu_metadata_before_base_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    block_table = torch.tensor([[10], [10]], dtype=torch.int32)
    seq_lens = torch.full((2,), 16, dtype=torch.int32)
    builder.set_cpu_metadata(*_cpu_metadata(block_table, seq_lens))

    def fail_base_build(*args, **kwargs):
        raise RuntimeError("base build failed")

    monkeypatch.setattr(FlashAttentionMetadataBuilder, "build", fail_base_build)
    with pytest.raises(RuntimeError, match="base build failed"):
        builder.build(0, SimpleNamespace())

    assert builder._fork_cpu_metadata is None


@pytest.mark.parametrize("mode", ["async_spec_decode", "mm_prefix"])
def test_metadata_builder_falls_back_for_incompatible_decode_metadata(mode) -> None:
    block_table = torch.tensor([[10], [10]], dtype=torch.int32)
    seq_lens = torch.full((2,), 16, dtype=torch.int32)
    metadata = _make_base_metadata(block_table, seq_lens)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    if mode == "async_spec_decode":
        builder.vllm_config.scheduler_config.async_scheduling = True
        builder.vllm_config.speculative_config = object()
    else:
        metadata.mm_prefix_query_range_tensor = torch.tensor(
            [[0, 16], [0, 16]], dtype=torch.int32
        )

    assert (
        builder._build_fork_kwargs(
            metadata,
            torch.arange(3, dtype=torch.int32),
            2,
            _cpu_metadata(block_table, seq_lens),
        )
        == {}
    )


def test_metadata_builder_respects_platform_pin_memory_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_table = torch.tensor([[10], [10]], dtype=torch.int32)
    seq_lens = torch.full((2,), 16, dtype=torch.int32)
    metadata = _make_base_metadata(block_table, seq_lens)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    builder.vllm_config.use_v2_model_runner = False
    monkeypatch.setattr(fork_attn_backend, "PIN_MEMORY", False)

    result = builder._build_fork_kwargs(
        metadata,
        torch.arange(3, dtype=torch.int32),
        2,
        _cpu_metadata(block_table, seq_lens),
    )

    assert result["fork_enabled"]
    assert builder._fork_buffer_pool.pin_memory is False


def _make_base_metadata(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> FlashAttentionMetadata:
    batch_size = block_table.shape[0]
    device = block_table.device
    return FlashAttentionMetadata(
        num_actual_tokens=batch_size,
        max_query_len=1,
        query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
        max_seq_len=int(seq_lens.max().item()),
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(batch_size, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )


def _cpu_metadata(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, np.ndarray, None]:
    return seq_lens.detach().cpu(), block_table.detach().cpu().numpy(), None


def _make_builder(
    *,
    block_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> ForkAttentionMetadataBuilder:
    builder = ForkAttentionMetadataBuilder.__new__(ForkAttentionMetadataBuilder)
    builder.block_size = block_size
    builder.num_heads_q = num_heads_q
    builder.num_heads_kv = num_heads_kv
    builder.headdim = head_dim
    builder.kv_cache_dtype = torch.float16
    builder.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        attention_config=AttentionConfig(fork_min_shared_tokens=0, fork_min_queries=2),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64],
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            max_num_seqs=64,
        ),
        speculative_config=None,
        model_config=SimpleNamespace(
            max_model_len=64,
            is_mm_prefix_lm=False,
        ),
    )
    builder.dcp_world_size = 1
    builder.kv_cache_spec = SimpleNamespace(sliding_window=None)
    return builder


def _make_impl(
    *,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
) -> ForkAttentionImpl:
    return ForkAttentionImpl(
        num_heads=num_heads_q,
        head_size=head_dim,
        scale=1.0 / math.sqrt(head_dim),
        num_kv_heads=num_heads_kv,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
    )


def test_fork_backend_falls_back_for_unsupported_tensor_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_table = torch.tensor([[0], [0]], dtype=torch.int32)
    seq_lens = torch.full((2,), 16, dtype=torch.int32)
    base_metadata = _make_base_metadata(block_table, seq_lens)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    metadata = ForkAttentionMetadata(
        **_flash_metadata_kwargs(base_metadata),
        **builder._build_fork_kwargs(
            base_metadata,
            torch.arange(3, dtype=torch.int32),
            2,
            _cpu_metadata(block_table, seq_lens),
        ),
    )
    query = torch.empty(2, 16, 128, dtype=torch.float32)
    kv_cache = torch.empty(1, 4, 16, 256, dtype=torch.float32)
    output = torch.empty_like(query)
    fallback_called = False

    def fake_flash_forward(*args, **kwargs):
        nonlocal fallback_called
        fallback_called = True
        return output

    monkeypatch.setattr(FlashAttentionImpl, "forward", fake_flash_forward)
    result = _make_impl(num_heads_q=16, num_heads_kv=4, head_dim=128).forward(
        SimpleNamespace(),
        query,
        query,
        query,
        kv_cache,
        metadata,
        output,
    )

    assert fallback_called
    assert result is output


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    reason="ForkAttention requires CUDA SM80+",
)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cache_layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "batch_kind", ["decode", "mixed_contiguous", "mixed_interleaved"]
)
def test_fork_backend_forward_matches_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
    dtype: torch.dtype,
    cache_layout: str,
    batch_kind: str,
) -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch_size = 8
    block_size = 32
    num_heads_q = 32 if head_dim == 128 else 16
    num_heads_kv = 8 if head_dim == 128 else 4
    prefix_blocks = [0, 1, 2, 3]
    suffix_blocks = 2
    seq_len = (len(prefix_blocks) + suffix_blocks) * block_size
    num_blocks = len(prefix_blocks) + batch_size * suffix_blocks
    block_rows = []
    for query_id in range(batch_size):
        suffix_start = len(prefix_blocks) + query_id * suffix_blocks
        block_rows.append(
            prefix_blocks + list(range(suffix_start, suffix_start + suffix_blocks))
        )
    query_lens = [1] * batch_size
    if batch_kind != "decode":
        # One unrelated decode stays on Flash even beside a qualifying cohort.
        block_rows[6] = list(range(4, 4 + len(block_rows[6])))
        query_lens = (
            [1, 1, 1, 1, 1, 1, 1, 3]
            if batch_kind == "mixed_contiguous"
            else [3, 1, 1, 2, 1, 1, 1, 1]
        )
    starts = torch.tensor([0] + list(np.cumsum(query_lens)), dtype=torch.int32)
    num_tokens = int(starts[-1])
    block_table = torch.tensor(block_rows, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    base_metadata = _make_base_metadata(block_table, seq_lens)
    base_metadata = replace(
        base_metadata,
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens),
        query_start_loc=starts.to(device),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device=device),
    )
    builder = _make_builder(
        block_size=block_size,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
    )
    builder.vllm_config.attention_config.fork_min_queries = 4
    builder.vllm_config.attention_config.fork_min_shared_tokens = 128
    # The runner's physical CPU row order can differ from the scheduled order.
    row_mapping = np.arange(batch_size)[::-1].copy()
    cpu_metadata = (
        seq_lens.cpu(),
        np.array(block_rows, dtype=np.int32)[row_mapping],
        row_mapping,
    )
    fork_metadata = ForkAttentionMetadata(
        **_flash_metadata_kwargs(base_metadata),
        **builder._build_fork_kwargs(
            base_metadata,
            starts,
            batch_size,
            cpu_metadata,
        ),
    )
    assert fork_metadata.fork_enabled
    mixed = fork_metadata.fork_mixed_batch
    if batch_kind == "decode":
        assert mixed is None
    else:
        assert mixed is not None
        expected = [
            int(starts[i]) for i in range(batch_size) if query_lens[i] == 1 and i != 6
        ]
        assert (
            torch.arange(num_tokens, device=device)[mixed.fork_tokens].tolist()
            == expected
        )
        assert mixed.num_fork_queries == len(expected)
        assert isinstance(mixed.fork_tokens, slice) == (
            batch_kind == "mixed_contiguous"
        )

    query = torch.randn(
        num_tokens + 3,
        num_heads_q,
        head_dim,
        dtype=dtype,
        device=device,
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_heads_kv,
        head_dim,
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    kv_cache = torch.cat((key_cache, value_cache), dim=-1).transpose(1, 2)
    if cache_layout == "HND":
        kv_cache = kv_cache.contiguous()
    output = torch.full_like(query, -123)
    layer = SimpleNamespace(
        _q_scale=torch.ones(1, device=device),
        _k_scale=torch.ones(1, device=device),
        _v_scale=torch.ones(1, device=device),
    )

    fork_called = False
    use_triton = head_dim == 128 and current_platform.is_device_capability(120)
    original_fork_attention = (
        fork_attn_backend.triton_fork_attention if use_triton else ops.fork_attention
    )

    def wrapped_fork_attention(*args, **kwargs):
        nonlocal fork_called
        fork_called = True
        return original_fork_attention(*args, **kwargs)

    if use_triton:
        monkeypatch.setattr(
            fork_attn_backend, "triton_fork_attention", wrapped_fork_attention
        )
    else:
        monkeypatch.setattr(ops, "fork_attention", wrapped_fork_attention)
    _make_impl(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
    ).forward(
        layer,
        query,
        query,
        query,
        kv_cache,
        fork_metadata,
        output,
    )
    assert fork_called

    reference = torch.full_like(query, -123)
    flash_attn_varlen_func(
        q=query[:num_tokens],
        k=key_cache,
        v=value_cache,
        out=reference[:num_tokens],
        cu_seqlens_q=base_metadata.query_start_loc,
        max_seqlen_q=max(query_lens),
        seqused_k=seq_lens,
        max_seqlen_k=seq_len,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        block_table=block_table,
        num_splits=0,
        return_softmax_lse=True,
    )
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    reason="ForkAttention requires CUDA SM80+",
)
def test_fork_backend_rejects_block_id_outside_physical_cache() -> None:
    device = torch.device("cuda")
    block_table = torch.tensor([[0], [0]], dtype=torch.int32, device=device)
    seq_lens = torch.full((2,), 16, dtype=torch.int32, device=device)
    base_metadata = _make_base_metadata(block_table, seq_lens)
    builder = _make_builder(
        block_size=16,
        num_heads_q=16,
        num_heads_kv=4,
        head_dim=128,
    )
    metadata = ForkAttentionMetadata(
        **_flash_metadata_kwargs(base_metadata),
        **builder._build_fork_kwargs(
            base_metadata,
            torch.arange(3, dtype=torch.int32),
            2,
            _cpu_metadata(block_table, seq_lens),
        ),
    )
    metadata.fork_max_block_id = 1
    query = torch.randn(2, 16, 128, dtype=torch.float16, device=device)
    kv_cache = torch.randn(1, 4, 16, 256, dtype=torch.float16, device=device)
    output = torch.empty_like(query)

    with pytest.raises(RuntimeError, match="invalid block ID"):
        _make_impl(num_heads_q=16, num_heads_kv=4, head_dim=128).forward(
            SimpleNamespace(),
            query,
            query,
            query,
            kv_cache,
            metadata,
            output,
        )


def test_fork_plan_cache_checks_physical_pages_and_partial_boundaries(monkeypatch):
    cache = fork_attn_backend._ForkPlanCache()
    rows = np.array([[1, 2, 3], [1, 4, 5]], dtype=np.int32)
    indices = np.array([0, 1], dtype=np.int32)
    lengths = [17, 18]
    original = fork_attn_backend._build_fork_plan
    calls = 0

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(fork_attn_backend, "_build_fork_plan", counted)

    def check():
        actual = cache.build(
            seq_lens=lengths,
            block_rows=rows,
            block_row_indices=indices,
            block_size=16,
            head_ratio=4,
        )
        expected = original(
            query_start_locs=list(range(len(lengths) + 1)),
            seq_lens=lengths,
            block_rows=rows,
            block_row_indices=indices,
            num_actual_tokens=len(lengths),
            block_size=16,
            head_ratio=4,
            require_shared=True,
        )
        assert actual == expected
        return actual

    check()
    lengths[:] = [18, 19]
    check()
    assert calls == 1
    # A private-page remapping must invalidate even when sequence lengths match.
    rows[0, 1] = 7
    check()
    assert calls == 2
    indices[:] = [1, 0]
    check()
    assert calls == 3
    lengths[:] = [32, 33]
    check()
    assert calls == 4
    rows[0, 0] = 9
    assert check() is None
    assert calls == 5
    # A rejected forest remains rejected while pages and boundaries are unchanged.
    assert check() is None
    assert calls == 5
    rows[0, 0] = 1
    check()
    assert calls == 6
    rows[0, 0] = -1
    with pytest.raises(_ForkPlanError, match="non-negative"):
        check()


@pytest.mark.parametrize("branches,cohort", [(8, 4), (8, 8), (17, 17)])
def test_decode_cache_extends_private_tails_and_invalidates_changed_requests(
    branches, cohort, monkeypatch
):
    cache = fork_attn_backend._ForkPlanCache()
    rows = np.arange(branches * 80, dtype=np.int32).reshape(branches, 80) + 100
    for i in range(branches):
        rows[i, :4] = [20 + i // cohort, 1, 2, 40 + i // 8]
    indices = np.arange(branches, dtype=np.int32)
    lengths = [81 + i % 3 for i in range(branches)]
    original = fork_attn_backend._build_fork_plan
    calls = 0

    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(fork_attn_backend, "_build_fork_plan", counted)

    def check():
        kwargs = dict(
            seq_lens=lengths,
            block_rows=rows,
            block_row_indices=indices,
            block_size=16,
            head_ratio=4,
            min_shared_tokens=32,
            min_queries=8,
        )
        actual = cache.build(**kwargs)
        expected = original(
            **kwargs,
            query_start_locs=list(range(len(lengths) + 1)),
            num_actual_tokens=len(lengths),
            require_shared=True,
        )
        assert (actual is not None) == (expected is not None)
        if actual is not None:
            _validate_fork_plan(
                list(actual.segments),
                list(actual.num_splits_per_query),
                {
                    q: rows[i, : math.ceil(length / 16)].tolist()
                    for q, (i, length) in enumerate(zip(indices, lengths))
                },
                dict(enumerate(lengths)),
                len(lengths),
                16,
            )
            assert actual.max_block_id == expected.max_block_id
        return actual

    # Staggered tails cross several full/partial boundaries, including a
    # singleton remainder of a shared 17-query CTA and a rejected 4+4 cohort.
    for _ in range(65):
        check()
        lengths = [length + 1 for length in lengths]
    # Freeze completed private chunks as the continuation grows longer.
    for offset in (600, 1100, 1101):
        lengths = [81 + i % 3 + offset for i in range(branches)]
        check()
    assert calls == 1
    # Check a page first introduced by incremental growth, not just the seed.
    rows[0, 6] = 9999
    check()
    assert calls == 2
    rows[0, 0] = 777
    check()
    assert calls == 3
    indices = indices[::-1].copy()
    check()
    assert calls == 4
    lengths = [length - 17 for length in lengths]
    check()
    assert calls == 5
    indices = indices[:-1]
    lengths = lengths[:-1]
    check()
    # Admission must be reconsidered when a branch leaves the scheduled batch.
    assert calls == (6 if branches == 17 else 5)


@pytest.mark.parametrize("prefix", [16384, 32768, 65536])
@pytest.mark.parametrize("batch", [8, 32, 64])
def test_fork_graph_long_prefix_has_capacity_for_parallel_segments(
    prefix, batch, monkeypatch
):
    builder = _make_builder(block_size=16, num_heads_q=32, num_heads_kv=8, head_dim=128)
    builder.vllm_config.attention_config = AttentionConfig()
    builder.vllm_config.model_config.max_model_len = prefix + 512
    builder.vllm_config.compilation_config.cudagraph_capture_sizes = [batch]
    queries, ctas, splits = builder._get_cudagraph_capture_limits()
    shared = prefix // 16
    rows = np.array(
        [list(range(shared)) + [shared + i] for i in range(batch)], dtype=np.int32
    )
    indices = np.arange(batch, dtype=np.int32)
    lengths = np.full(batch, prefix + 1, dtype=np.int32)
    requirement = builder.prepare_cudagraph_plan(lengths, rows, indices, shared + batch)
    assert requirement is not None
    required_ctas, required_splits = requirement
    assert queries == batch and ctas >= required_ctas and splits >= required_splits
    from vllm.v1.worker.gpu.cudagraph_utils import _get_fork_graph_capture_plan

    capture = _get_fork_graph_capture_plan(queries, queries, splits)
    assert capture is not None and capture.capacity == ctas
    builder.set_cudagraph_plan(ctas, splits, force_flash=False)
    workspace = builder._get_fork_cudagraph_workspace(torch.device("cpu"))
    assert builder._fork_prebuilt_result is not None
    packed = workspace.pack(
        builder._fork_prebuilt_result.plan,
        query_capacity=queries,
        cta_capacity=ctas,
        split_capacity=splits,
    )
    assert packed["fork_enabled"]

    def fail_rebuild(**kwargs):
        pytest.fail("long-prefix plans must reuse their physical-page snapshot")

    monkeypatch.setattr(fork_attn_backend, "_build_fork_plan", fail_rebuild)
    lengths += 1
    assert builder.prepare_cudagraph_plan(lengths, rows, indices, shared + batch)


@pytest.mark.parametrize(
    "prefix,batch,cohort,eligible",
    [
        (16368, 8, 8, False),
        (16384, 7, 7, False),
        (16384, 8, 4, False),
        (16384, 8, 8, True),
        (16384, 16, 8, True),
    ],
)
def test_fork_admission_requires_a_long_physical_prefix_within_one_cohort(
    prefix, batch, cohort, eligible, monkeypatch
):
    builder = _make_builder(block_size=16, num_heads_q=32, num_heads_kv=8, head_dim=128)
    builder.vllm_config.attention_config = AttentionConfig()
    builder.vllm_config.model_config.max_model_len = prefix + 512
    blocks = prefix // 16
    rows = np.tile(np.arange(blocks + 1, dtype=np.int32), (batch, 1))
    rows[:, -1] += np.arange(batch)
    # Same endpoint page does not establish a shared prefix: compare every page.
    rows[:, 0] += (np.arange(batch) // cohort) * (blocks + batch)
    lengths = torch.full((batch,), prefix + 1, dtype=torch.int32)
    indices = np.arange(batch, dtype=np.int32)

    if not eligible:

        def reject_forest(*args, **kwargs):
            pytest.fail("ineligible batches must not construct a forest")

        monkeypatch.setattr(fork_attn_backend, "_emit_forest_segments", reject_forest)
    requirement = builder.prepare_cudagraph_plan(
        lengths, rows, indices, int(rows.max()) + 1
    )
    assert (requirement is not None) == eligible
    metadata = _make_base_metadata(torch.from_numpy(rows), lengths)
    eager = builder._build_fork_kwargs(
        metadata, torch.arange(batch + 1), batch, (lengths, rows, indices)
    )
    assert eager.get("fork_enabled", False) == eligible
    if eligible:
        assert (
            builder.prepare_cudagraph_plan(
                lengths[:7], rows, indices[:7], int(rows.max()) + 1
            )
            is None
        )
        assert builder._fork_prebuilt_result is None
