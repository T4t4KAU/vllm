# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.fork_attn import (
    ForkAttentionBackend,
    ForkAttentionImpl,
    ForkAttentionMetadata,
    ForkAttentionMetadataBuilder,
    _flash_metadata_kwargs,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_fork_attention_backend_registered() -> None:
    backend_cls = AttentionBackendEnum.FORK_ATTN.get_class()
    assert backend_cls.get_name() == "FORK_ATTN"


@pytest.mark.parametrize(
    ("capability", "expected_error"),
    [
        (DeviceCapability(8, 0), None),
        (DeviceCapability(8, 9), None),
        (DeviceCapability(7, 5), "compute capability >= 8.0"),
    ],
)
def test_fork_attention_backend_device_capability(
    capability: DeviceCapability,
    expected_error: str | None,
) -> None:
    assert ForkAttentionBackend.supports_compute_capability(capability) is (
        expected_error is None
    )
    error = ForkAttentionBackend.supports_combination(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype=None,
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=capability,
    )

    if expected_error is None:
        assert error is None
    else:
        assert error is not None
        assert expected_error in error


def _make_builder(
    *,
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> ForkAttentionMetadataBuilder:
    builder = ForkAttentionMetadataBuilder.__new__(ForkAttentionMetadataBuilder)
    builder.block_size = block_size
    builder.num_heads_q = num_heads
    builder.num_heads_kv = num_kv_heads
    builder.headdim = head_dim
    builder.kv_cache_dtype = torch.float16
    return builder


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "expected"),
    [(14, 2, True), (32, 8, True), (14, 4, False)],
    ids=["qwen2_5_gqa7", "llama3_2_gqa4", "non_divisible"],
)
def test_fork_decode_supports_model_gqa_geometry(
    monkeypatch: pytest.MonkeyPatch,
    num_heads: int,
    num_kv_heads: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    builder = _make_builder(
        block_size=16,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=64,
    )
    metadata = SimpleNamespace(
        max_query_len=1,
        num_actual_tokens=2,
        seq_lens=torch.empty(2),
        causal=True,
        mm_prefix_range_tensor=None,
        rswa_prefix_lens=None,
    )

    assert builder._can_use_fork_decode(metadata) is expected


def _run_flash_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    batch, num_heads, head_dim = q.shape
    out = torch.empty_like(q)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    flash_attn_varlen_func(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seq_lens,
        max_seqlen_k=int(seq_lens.max().item()),
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=True,
        block_table=block_table,
        num_splits=0,
    )
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fork_cudagraph_workspace_uses_scheduler_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 2048)
    builder = _make_builder(
        block_size=16,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    )
    builder.device = torch.device("cuda")
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16)
    )
    builder.compilation_config = SimpleNamespace(max_cudagraph_capture_size=32)
    builder.model_config = SimpleNamespace(max_model_len=32768)

    workspace = builder._get_cudagraph_workspace()

    assert workspace.max_reqs == 32
    assert workspace.max_active_reqs == 16
    assert workspace.max_prefix_chunks == 8
    assert workspace.prefix_cohorts == 1
    assert workspace.query_tables[0].shape == (8, 16)
    assert workspace.query_tables[1].shape == (16, 1)
    assert workspace.num_split_per_seq.shape == (32,)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=1,
        query_start_loc=torch.arange(3, dtype=torch.int32, device="cuda"),
        max_seq_len=64,
        seq_lens=torch.full((2,), 64, dtype=torch.int32, device="cuda"),
        block_table=torch.arange(8, dtype=torch.int32, device="cuda").view(2, 4),
        slot_mapping=torch.arange(2, dtype=torch.int64, device="cuda"),
        use_cascade=False,
        common_prefix_len=32,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )
    builder._update_cudagraph_workspace(metadata, workspace)
    torch.accelerator.synchronize()

    assert workspace.kv_in_ctas[1][:2].tolist() == [64, 64]
    assert workspace.num_split_per_seq[:2].tolist() == [1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fork_cudagraph_workspace_uses_prefix_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 2048)
    builder = _make_builder(
        block_size=16,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    )
    builder.device = torch.device("cuda")
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16)
    )
    builder.compilation_config = SimpleNamespace(max_cudagraph_capture_size=32)
    builder.model_config = SimpleNamespace(max_model_len=32768)
    builder._fork_cudagraph_plan = SimpleNamespace(kind="common", capacity=2)

    workspace = builder._get_cudagraph_workspace(
        builder._get_cudagraph_prefix_chunk_bucket()
    )

    assert workspace.max_prefix_chunks == 2
    assert workspace.prefix_chunk_capacity_blocks == 256
    assert workspace.query_tables[0].shape == (2, 16)
    assert workspace.block_tables[0].shape == (2, 256)
    assert workspace.split_out.shape[2] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fork_cudagraph_workspace_matches_aligned_block_table() -> None:
    builder = _make_builder(
        block_size=16,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    )
    builder.device = torch.device("cuda")
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16)
    )
    builder.compilation_config = SimpleNamespace(max_cudagraph_capture_size=32)
    builder.model_config = SimpleNamespace(max_model_len=11424)

    workspace = builder._get_cudagraph_workspace(4)
    metadata = FlashAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=1,
        query_start_loc=torch.arange(3, dtype=torch.int32, device="cuda"),
        max_seq_len=8193,
        seq_lens=torch.full((2,), 8193, dtype=torch.int32, device="cuda"),
        block_table=torch.zeros((2, 720), dtype=torch.int32, device="cuda"),
        slot_mapping=torch.arange(2, dtype=torch.int64, device="cuda"),
        use_cascade=True,
        common_prefix_len=499 * 16,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )
    kwargs = builder._update_cudagraph_workspace(
        metadata,
        workspace,
        num_active_reqs=2,
    )
    torch.accelerator.synchronize()

    assert workspace.max_blocks == 720
    assert workspace.block_tables[1].shape == (16, 720)
    assert kwargs["fork_enabled"] is True
    assert workspace.num_split_per_seq[:2].tolist() == [5, 5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fork_cudagraph_workspace_excludes_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 2048)
    builder = _make_builder(
        block_size=16,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    )
    builder.device = torch.device("cuda")
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16)
    )
    builder.compilation_config = SimpleNamespace(max_cudagraph_capture_size=32)
    builder.model_config = SimpleNamespace(max_model_len=32768)
    workspace = builder._get_cudagraph_workspace(2)

    active_reqs = 13
    padded_reqs = 16
    prefix_blocks = 455
    suffix_blocks = 64
    prefix_len = prefix_blocks * builder.block_size
    seq_len = (prefix_blocks + suffix_blocks) * builder.block_size
    block_table = torch.arange(
        padded_reqs * (prefix_blocks + suffix_blocks),
        dtype=torch.int32,
        device="cuda",
    ).view(padded_reqs, prefix_blocks + suffix_blocks)
    seq_lens = torch.cat(
        (
            torch.full(
                (active_reqs,),
                seq_len,
                dtype=torch.int32,
                device="cuda",
            ),
            torch.zeros(
                padded_reqs - active_reqs,
                dtype=torch.int32,
                device="cuda",
            ),
        )
    )
    metadata = FlashAttentionMetadata(
        num_actual_tokens=padded_reqs,
        max_query_len=1,
        query_start_loc=torch.cat(
            (
                torch.arange(active_reqs + 1, dtype=torch.int32, device="cuda"),
                torch.full(
                    (padded_reqs - active_reqs,),
                    active_reqs,
                    dtype=torch.int32,
                    device="cuda",
                ),
            )
        ),
        max_seq_len=seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(padded_reqs, dtype=torch.int64, device="cuda"),
        use_cascade=True,
        common_prefix_len=prefix_len,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )

    kwargs = builder._update_cudagraph_workspace(
        metadata,
        workspace,
        active_reqs,
    )
    torch.accelerator.synchronize()

    assert kwargs["fork_enabled"]
    assert workspace.prefix_chunk_capacity_blocks == 256
    assert workspace.num_seqs_per_ctas[0][:2].tolist() == [
        active_reqs,
        active_reqs,
    ]
    assert workspace.kv_in_ctas[0][:2].tolist() == [4096, prefix_len - 4096]
    assert workspace.num_seqs_per_ctas[1][:active_reqs].tolist() == [1] * active_reqs
    assert workspace.num_seqs_per_ctas[1][active_reqs:].tolist() == [0] * (
        padded_reqs - active_reqs
    )
    assert workspace.kv_in_ctas[1][:active_reqs].tolist() == [1024] * active_reqs
    assert workspace.num_split_per_seq[:active_reqs].tolist() == [3] * active_reqs
    assert workspace.num_split_per_seq[active_reqs:padded_reqs].tolist() == [0] * (
        padded_reqs - active_reqs
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fork_workspace_does_not_force_suffix_only_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 2048)
    builder = _make_builder(
        block_size=16,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    )
    builder.device = torch.device("cuda")
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16)
    )
    builder.compilation_config = SimpleNamespace(max_cudagraph_capture_size=32)
    builder.model_config = SimpleNamespace(max_model_len=32768)
    builder._get_cudagraph_workspace()

    metadata = FlashAttentionMetadata(
        num_actual_tokens=2,
        max_query_len=1,
        query_start_loc=torch.arange(3, dtype=torch.int32, device="cuda"),
        max_seq_len=64,
        seq_lens=torch.full((2,), 64, dtype=torch.int32, device="cuda"),
        block_table=torch.arange(8, dtype=torch.int32, device="cuda").view(2, 4),
        slot_mapping=torch.arange(2, dtype=torch.int64, device="cuda"),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )

    assert builder._build_fork_kwargs(metadata) == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fork_forest_metadata_without_global_common_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 64)
    device = torch.device("cuda")
    builder = _make_builder(
        block_size=16,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    )
    block_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [10, 11, 12, 14],
            [20, 21, 22, 23],
            [20, 21, 24, 25],
        ],
        dtype=torch.int32,
        device=device,
    )
    metadata = FlashAttentionMetadata(
        num_actual_tokens=4,
        max_query_len=1,
        query_start_loc=torch.arange(5, dtype=torch.int32, device=device),
        max_seq_len=64,
        seq_lens=torch.full((4,), 64, dtype=torch.int32, device=device),
        block_table=block_table,
        slot_mapping=torch.arange(4, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )
    builder._fork_block_table_cpu = block_table.cpu().numpy()
    builder._fork_seq_lens_cpu = torch.full((4,), 64, dtype=torch.int32)

    kwargs = builder._build_fork_kwargs(metadata)
    torch.accelerator.synchronize()

    assert kwargs["fork_enabled"]
    assert kwargs["fork_num_split_per_seq"].tolist() == [2, 2, 2, 2]
    assert kwargs["fork_max_split_per_seq"] == 2
    num_shared_ctas = 0
    for q_table, num_seqs_per_cta in zip(
        kwargs["fork_query_tables"],
        kwargs["fork_num_seqs_per_ctas"],
        strict=True,
    ):
        assert q_table.shape[0] == num_seqs_per_cta.shape[0]
        num_shared_ctas += int((num_seqs_per_cta > 1).sum().item())
    assert num_shared_ctas >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fork_forest_metadata_emits_hierarchical_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 64)
    device = torch.device("cuda")
    builder = _make_builder(
        block_size=16,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    )
    block_table = torch.tensor(
        [
            [0, 1, 2, 4],
            [0, 1, 2, 5],
            [0, 1, 3, 6],
        ],
        dtype=torch.int32,
        device=device,
    )
    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=1,
        query_start_loc=torch.arange(4, dtype=torch.int32, device=device),
        max_seq_len=48,
        seq_lens=torch.full((3,), 48, dtype=torch.int32, device=device),
        block_table=block_table,
        slot_mapping=torch.arange(3, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )

    kwargs = builder._build_fork_kwargs(metadata)
    torch.accelerator.synchronize()

    assert kwargs["fork_enabled"]
    assert kwargs["fork_num_split_per_seq"].tolist() == [2, 2, 2]
    all_ctas = []
    for group_id, q_table in enumerate(kwargs["fork_query_tables"]):
        q_table_cpu = q_table.cpu().tolist()
        block_table_cpu = kwargs["fork_block_tables"][group_id].cpu().tolist()
        num_seqs_cpu = kwargs["fork_num_seqs_per_ctas"][group_id].cpu().tolist()
        rank_cpu = kwargs["fork_cta_ranks"][group_id].cpu().tolist()
        kv_cpu = kwargs["fork_kv_in_ctas"][group_id].cpu().tolist()
        for cta_id, num_seqs in enumerate(num_seqs_cpu):
            all_ctas.append(
                (
                    q_table_cpu[cta_id][:num_seqs],
                    block_table_cpu[cta_id],
                    rank_cpu[cta_id],
                    kv_cpu[cta_id],
                )
            )

    assert ([0, 1, 2], [0, 1], 0, 32) in all_ctas
    assert ([0, 1], [2], 1, 16) in all_ctas
    assert ([2], [3], 1, 16) in all_ctas


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="FORK requires SM80+",
)
def test_fork_attention_backend_forward_uses_fork(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 64)
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.float16
    batch = 8
    block_size = 32
    prefix_blocks = 4
    suffix_blocks = 2
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    prefix_len = prefix_blocks * block_size
    seq_len = (prefix_blocks + suffix_blocks) * block_size
    total_blocks = prefix_blocks + batch * suffix_blocks

    q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)
    kv_cache = torch.stack((k_cache, v_cache), dim=1)

    block_table_rows = []
    for seq_id in range(batch):
        start = prefix_blocks + seq_id * suffix_blocks
        suffix = list(range(start, start + suffix_blocks))
        block_table_rows.append(list(range(prefix_blocks)) + suffix)
    block_table = torch.tensor(block_table_rows, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32, device=device)

    base_metadata = FlashAttentionMetadata(
        num_actual_tokens=batch,
        max_query_len=1,
        query_start_loc=query_start_loc,
        max_seq_len=seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(batch, dtype=torch.int64, device=device),
        use_cascade=True,
        common_prefix_len=prefix_len,
        cu_prefix_query_lens=torch.tensor([0, batch], dtype=torch.int32, device=device),
        prefix_kv_lens=torch.tensor([prefix_len], dtype=torch.int32, device=device),
        suffix_kv_lens=seq_lens - prefix_len,
        causal=True,
    )
    builder = _make_builder(
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    metadata = ForkAttentionMetadata(
        **_flash_metadata_kwargs(base_metadata),
        **builder._build_fork_kwargs(base_metadata),
    )
    assert metadata.fork_enabled
    assert metadata.fork_max_split_per_seq == 3
    assert metadata.fork_mnw == [32, 64, 2, 16, 64, 1]
    assert metadata.fork_query_tables is not None
    assert metadata.fork_query_tables[0].shape == (2, 8)
    assert metadata.fork_block_tables is not None
    assert metadata.fork_block_tables[0].shape == (2, 2)

    impl = ForkAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / math.sqrt(head_dim),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
    )
    layer = SimpleNamespace(
        _q_scale=torch.ones(1, device=device),
        _k_scale=torch.ones(1, device=device),
        _v_scale=torch.ones(1, device=device),
    )
    output = torch.empty_like(q)

    called = False
    orig_fork_attention = ops.fork_attention

    def wrapped_fork_attention(*args, **kwargs):
        nonlocal called
        called = True
        return orig_fork_attention(*args, **kwargs)

    monkeypatch.setattr(ops, "fork_attention", wrapped_fork_attention)
    impl.forward(layer, q, q, q, kv_cache, metadata, output)
    assert called

    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)
    torch.testing.assert_close(output, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="FORK requires SM80+",
)
def test_fork_attention_backend_forward_uses_prefix_forest(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 64)
    torch.manual_seed(4)
    device = torch.device("cuda")
    dtype = torch.float16
    batch = 4
    block_size = 16
    seq_len = 64
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128

    q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        26, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)
    kv_cache = torch.stack((k_cache, v_cache), dim=1)
    block_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [10, 11, 12, 14],
            [20, 21, 22, 23],
            [20, 21, 24, 25],
        ],
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    base_metadata = FlashAttentionMetadata(
        num_actual_tokens=batch,
        max_query_len=1,
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32, device=device),
        max_seq_len=seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(batch, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )
    builder = _make_builder(
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    metadata = ForkAttentionMetadata(
        **_flash_metadata_kwargs(base_metadata),
        **builder._build_fork_kwargs(base_metadata),
    )
    assert metadata.fork_enabled
    assert metadata.fork_num_split_per_seq is not None
    assert metadata.fork_num_split_per_seq.tolist() == [2, 2, 2, 2]

    impl = ForkAttentionImpl(
        num_heads=num_heads,
        head_size=head_dim,
        scale=1.0 / math.sqrt(head_dim),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
    )
    layer = SimpleNamespace(
        _q_scale=torch.ones(1, device=device),
        _k_scale=torch.ones(1, device=device),
        _v_scale=torch.ones(1, device=device),
    )
    output = torch.empty_like(q)
    impl.forward(layer, q, q, q, kv_cache, metadata, output)

    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)
    torch.testing.assert_close(output, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="FORK requires SM80+",
)
def test_fork_forest_cudagraph_replay_handles_noncontiguous_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE", 128)
    torch.manual_seed(17)
    device = torch.device("cuda")
    dtype = torch.float16
    batch = 4
    block_size = 16
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128

    builder = _make_builder(
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    builder.device = device
    builder.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8)
    )
    builder.compilation_config = SimpleNamespace(max_cudagraph_capture_size=8)
    builder.model_config = SimpleNamespace(max_model_len=512)
    workspace = builder._get_cudagraph_forest_workspace(64)

    # A one-warp CTA may only copy 64 rows without crossing a 16-token page.
    assert workspace.mnw == [32, 128, 2, 16, 64, 1]

    shared_blocks = [10, 3, 20, 7, 30, 5, 40, 9]
    block_table = torch.tensor(
        [shared_blocks + [41 + 2 * req_id, 42 + 2 * req_id] for req_id in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full(
        (batch,),
        block_table.shape[1] * block_size,
        dtype=torch.int32,
        device=device,
    )
    metadata = FlashAttentionMetadata(
        num_actual_tokens=batch,
        max_query_len=1,
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32, device=device),
        max_seq_len=int(seq_lens[0]),
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=torch.arange(batch, dtype=torch.int64, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )
    assert builder._update_cudagraph_forest_workspace(metadata, workspace, batch)

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        64,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    output = torch.empty_like(q)

    def run_fork() -> None:
        ops.fork_attention(
            output,
            workspace.softmax_lse,
            workspace.split_out,
            workspace.split_lse,
            q,
            k_cache,
            v_cache,
            workspace.num_split_per_seq,
            workspace.query_tables,
            workspace.block_tables,
            workspace.num_seqs_per_ctas,
            workspace.cta_ranks,
            workspace.kv_in_ctas,
            workspace.mnw,
            workspace.max_split_per_seq,
            1.0 / math.sqrt(head_dim),
        )

    run_fork()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fork()

    replay_q = torch.randn_like(q)
    q.copy_(replay_q)
    graph.replay()
    torch.accelerator.synchronize()

    ref = _run_flash_ref(
        replay_q[:, 0],
        k_cache,
        v_cache,
        seq_lens,
        block_table,
    )
    torch.testing.assert_close(output[:, 0], ref, atol=2e-2, rtol=2e-2)

    invalid_mnw = workspace.mnw.copy()
    invalid_mnw[4] = 128
    with pytest.raises(RuntimeError, match="crosses a paged KV cache boundary"):
        ops.fork_attention(
            output,
            workspace.softmax_lse,
            workspace.split_out,
            workspace.split_lse,
            q,
            k_cache,
            v_cache,
            workspace.num_split_per_seq,
            workspace.query_tables,
            workspace.block_tables,
            workspace.num_seqs_per_ctas,
            workspace.cta_ranks,
            workspace.kv_in_ctas,
            invalid_mnw,
            workspace.max_split_per_seq,
            1.0 / math.sqrt(head_dim),
        )
