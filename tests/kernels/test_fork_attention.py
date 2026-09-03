# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections import defaultdict

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    reason="ForkAttention requires CUDA SM80+",
)


def _get_mnw(
    num_seqs: int,
    hratio: int,
    kv_len: int,
    head_dim: int,
    page_block_size: int | None = None,
) -> tuple[int, int, int]:
    m_val = num_seqs * hratio
    if m_val > 32:
        tile_m, warps = 64, 4
    elif m_val > 16:
        tile_m, warps = 32, 2
    else:
        tile_m, warps = 16, 1

    if kv_len < 32:
        tile_n = 16
    elif kv_len < 64:
        tile_n = 32
    elif kv_len < 128:
        tile_n = 64
    else:
        tile_n = 128
    if tile_m == 64:
        tile_n = max(32, tile_n)
    # A 256-wide head with N=128 requires more dynamic shared memory than
    # Ampere/Ada/Blackwell consumer GPUs expose per block.
    if head_dim == 256:
        tile_n = min(tile_n, 64)
    if page_block_size is not None:
        rows_per_thread = tile_n // (warps * 4)
        while tile_n > 16 and (
            rows_per_thread > page_block_size or page_block_size % rows_per_thread != 0
        ):
            tile_n //= 2
            rows_per_thread = tile_n // (warps * 4)
    return tile_m, tile_n, warps


def _pack_boxes(
    boxes: list[tuple[list[int], list[int], int, int]],
    *,
    num_seqs: int,
    hratio: int,
    device: torch.device,
    head_dim: int,
    page_block_size: int | None = None,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[int],
    int,
]:
    num_split_per_seq = [0] * num_seqs
    for q_ids, _, rank, _ in boxes:
        for q_id in q_ids:
            num_split_per_seq[q_id] = max(num_split_per_seq[q_id], rank + 1)

    largest_kernel_hratio = 4 if hratio >= 4 else 2 if hratio >= 2 else 1
    max_queries_per_cta = 64 // largest_kernel_hratio
    chunked_boxes: list[tuple[list[int], list[int], int, int]] = []
    for q_ids, blocks, rank, kv_len in boxes:
        for start in range(0, len(q_ids), max_queries_per_cta):
            chunked_boxes.append(
                (q_ids[start : start + max_queries_per_cta], blocks, rank, kv_len)
            )
    boxes = chunked_boxes

    grouped: dict[tuple[int, int, int], list[tuple[list[int], list[int], int, int]]]
    grouped = defaultdict(list)
    for box in boxes:
        q_ids, _, _, kv_len = box
        grouped[_get_mnw(len(q_ids), hratio, kv_len, head_dim, page_block_size)].append(
            box
        )

    q_tables: list[torch.Tensor] = []
    block_tables: list[torch.Tensor] = []
    num_seqs_per_ctas: list[torch.Tensor] = []
    cta_ranks: list[torch.Tensor] = []
    kv_in_ctas: list[torch.Tensor] = []
    mnw: list[int] = []

    for tile, group in sorted(grouped.items(), reverse=True):
        max_q = max(len(q_ids) for q_ids, _, _, _ in group)
        max_blocks = max(len(blocks) for _, blocks, _, _ in group)
        q_table: list[list[int]] = []
        block_table: list[list[int]] = []
        cta_num_seqs: list[int] = []
        ranks: list[int] = []
        kv_lens: list[int] = []
        for q_ids, blocks, rank, kv_len in group:
            q_table.append(q_ids + [0] * (max_q - len(q_ids)))
            block_table.append(blocks + [0] * (max_blocks - len(blocks)))
            cta_num_seqs.append(len(q_ids))
            ranks.append(rank)
            kv_lens.append(kv_len)

        q_tables.append(torch.tensor(q_table, dtype=torch.int32, device=device))
        block_tables.append(torch.tensor(block_table, dtype=torch.int32, device=device))
        num_seqs_per_ctas.append(
            torch.tensor(cta_num_seqs, dtype=torch.int32, device=device)
        )
        cta_ranks.append(torch.tensor(ranks, dtype=torch.int32, device=device))
        kv_in_ctas.append(torch.tensor(kv_lens, dtype=torch.int32, device=device))
        mnw.extend(tile)

    return (
        torch.tensor(num_split_per_seq, dtype=torch.int32, device=device),
        q_tables,
        block_tables,
        num_seqs_per_ctas,
        cta_ranks,
        kv_in_ctas,
        mnw,
        max(num_split_per_seq),
    )


def _run_fork(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    boxes: list[tuple[list[int], list[int], int, int]],
    page_block_size: int | None = None,
    *,
    out: torch.Tensor | None = None,
    softmax_lse: torch.Tensor | None = None,
    split_out: torch.Tensor | None = None,
    split_lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = q.device
    batch, _, num_heads, head_dim = q.shape
    hratio = num_heads // k_cache.shape[2]
    (
        num_split_per_seq,
        q_tables,
        block_tables,
        num_seqs_per_ctas,
        cta_ranks,
        kv_in_ctas,
        mnw,
        max_split,
    ) = _pack_boxes(
        boxes,
        num_seqs=batch,
        hratio=hratio,
        device=device,
        head_dim=head_dim,
        page_block_size=page_block_size,
    )

    if out is None:
        out = torch.empty_like(q)
    if softmax_lse is None:
        softmax_lse = torch.empty(
            (batch, num_heads, 1), dtype=torch.float32, device=device
        )
    if split_out is None:
        split_out = torch.empty(
            (batch, num_heads, max(max_split, 1), head_dim),
            dtype=torch.float32,
            device=device,
        )
    if split_lse is None:
        split_lse = torch.empty(
            (batch, num_heads, max(max_split, 1)),
            dtype=torch.float32,
            device=device,
        )

    ops.fork_attention(
        out,
        softmax_lse,
        split_out,
        split_lse,
        q,
        k_cache,
        v_cache,
        num_split_per_seq,
        q_tables,
        block_tables,
        num_seqs_per_ctas,
        cta_ranks,
        kv_in_ctas,
        mnw,
        max_split,
        1.0 / math.sqrt(head_dim),
    )
    return out, softmax_lse


def _run_flash_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, num_heads, head_dim = q.shape
    out = torch.empty((batch, num_heads, head_dim), dtype=q.dtype, device=q.device)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    out, softmax_lse = flash_attn_varlen_func(
        q=q.view(batch, num_heads, head_dim),
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
        return_softmax_lse=True,
    )
    return out.view_as(q), softmax_lse.transpose(0, 1).unsqueeze(-1)


def _assert_attention_close(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    actual_out, actual_lse = actual
    expected_out, expected_lse = expected
    torch.testing.assert_close(actual_out, expected_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_lse, expected_lse, atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads"),
    [(16, 4), (14, 2), (16, 2)],
    ids=["gqa4", "gqa7", "gqa8"],
)
def test_fork_attention_shared_prefix(
    dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch = 8
    block_size = 32
    seq_len = 128
    num_blocks = seq_len // block_size

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).repeat(
        batch, 1
    )
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    boxes = [(list(range(batch)), list(range(num_blocks)), 0, seq_len)]
    actual = _run_fork(q, k_cache, v_cache, boxes)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    _assert_attention_close(actual, ref)


@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads"),
    [(16, 4), (14, 2), (16, 2)],
    ids=["gqa4", "gqa7", "gqa8"],
)
def test_fork_attention_split_prefix_suffix(
    num_heads: int,
    num_kv_heads: int,
) -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.float16
    batch = 6
    block_size = 32
    head_dim = 128
    prefix_blocks = 4
    suffix_blocks = 2
    seq_len = (prefix_blocks + suffix_blocks) * block_size
    total_blocks = prefix_blocks + batch * suffix_blocks

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)

    table_rows = []
    boxes: list[tuple[list[int], list[int], int, int]] = [
        (list(range(batch)), list(range(prefix_blocks)), 0, prefix_blocks * block_size)
    ]
    for seq_id in range(batch):
        start = prefix_blocks + seq_id * suffix_blocks
        suffix = list(range(start, start + suffix_blocks))
        table_rows.append(list(range(prefix_blocks)) + suffix)
        boxes.append(([seq_id], suffix, 1, suffix_blocks * block_size))

    block_table = torch.tensor(table_rows, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    actual = _run_fork(q, k_cache, v_cache, boxes)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    _assert_attention_close(actual, ref)


def test_fork_attention_masks_partial_suffix_block() -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.float16
    batch = 8
    block_size = 32
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128
    prefix_blocks = 4
    suffix_blocks = 2
    suffix_tokens = block_size + 7
    seq_len = prefix_blocks * block_size + suffix_tokens
    total_blocks = prefix_blocks + batch * suffix_blocks

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)

    table_rows = []
    boxes: list[tuple[list[int], list[int], int, int]] = [
        (list(range(batch)), list(range(prefix_blocks)), 0, prefix_blocks * block_size)
    ]
    for seq_id in range(batch):
        start = prefix_blocks + seq_id * suffix_blocks
        suffix = list(range(start, start + suffix_blocks))
        table_rows.append(list(range(prefix_blocks)) + suffix)
        boxes.append(([seq_id], suffix, 1, suffix_tokens))

    block_table = torch.tensor(table_rows, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    actual = _run_fork(q, k_cache, v_cache, boxes)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    _assert_attention_close(actual, ref)


def test_fork_attention_interleaved_kv_cache_suffix_page_stride() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    dtype = torch.float16
    batch = 8
    block_size = 16
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128
    prefix_blocks = 16
    suffix_tokens = 257
    suffix_blocks = (suffix_tokens + block_size - 1) // block_size
    prefix_len = prefix_blocks * block_size
    seq_len = prefix_len + suffix_tokens
    total_blocks = prefix_blocks + batch * suffix_blocks

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)
    kv_cache = torch.stack((k_cache, v_cache), dim=1)
    k_cache, v_cache = kv_cache.unbind(1)

    table_rows = []
    boxes: list[tuple[list[int], list[int], int, int]] = [
        (list(range(batch)), list(range(prefix_blocks)), 0, prefix_len)
    ]
    next_block = prefix_blocks
    for seq_id in range(batch):
        suffix = list(range(next_block, next_block + suffix_blocks))
        next_block += suffix_blocks
        table_rows.append(list(range(prefix_blocks)) + suffix)
        boxes.append(([seq_id], suffix, 1, suffix_tokens))

    block_table = torch.tensor(table_rows, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    actual = _run_fork(q, k_cache, v_cache, boxes, page_block_size=block_size)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    _assert_attention_close(actual, ref)


def test_fork_attention_multiple_single_split_kernel_groups() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    batch, block_size, num_heads, num_kv_heads, head_dim = 2, 32, 4, 1, 64

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        5,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    boxes = [([0], [0, 1, 2, 3], 0, 128), ([1], [4], 0, 16)]
    block_table = torch.tensor(
        [[0, 1, 2, 3], [4, 0, 0, 0]], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([128, 16], dtype=torch.int32, device=device)
    out = torch.full_like(q, torch.nan)

    actual = _run_fork(q, k_cache, v_cache, boxes, out=out)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    assert torch.isfinite(out).all()
    _assert_attention_close(actual, ref)


def test_fork_attention_chunks_cta_to_tile_capacity() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    batch, block_size, num_heads, num_kv_heads, head_dim = 17, 32, 4, 1, 64
    num_blocks = 4

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    boxes = [(list(range(batch)), list(range(num_blocks)), 0, 128)]
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).repeat(
        batch, 1
    )
    seq_lens = torch.full((batch,), 128, dtype=torch.int32, device=device)
    out = torch.full_like(q, torch.nan)

    actual = _run_fork(q, k_cache, v_cache, boxes, out=out)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    assert torch.isfinite(out).all()
    _assert_attention_close(actual, ref)


def test_fork_attention_supports_padded_tensor_layouts() -> None:
    torch.manual_seed(6)
    device = torch.device("cuda")
    batch, block_size, num_heads, num_kv_heads, head_dim = 2, 32, 8, 2, 64
    prefix_blocks, suffix_blocks = 2, 1
    num_blocks = prefix_blocks + batch * suffix_blocks

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    v_values = torch.randn_like(k_cache)
    v_storage = torch.empty(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim + 8,
        dtype=torch.float16,
        device=device,
    )
    v_cache = v_storage[..., :head_dim]
    v_cache.copy_(v_values)

    boxes: list[tuple[list[int], list[int], int, int]] = [
        (list(range(batch)), list(range(prefix_blocks)), 0, 64)
    ]
    table_rows = []
    for seq_id in range(batch):
        suffix = [prefix_blocks + seq_id]
        boxes.append(([seq_id], suffix, 1, 32))
        table_rows.append(list(range(prefix_blocks)) + suffix)
    block_table = torch.tensor(table_rows, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), 96, dtype=torch.int32, device=device)

    sentinel = 12345.0
    lse_storage = torch.full(
        (batch, num_heads, 2), sentinel, dtype=torch.float32, device=device
    )
    softmax_lse = lse_storage[..., :1]
    split_out_storage = torch.full(
        (batch, num_heads, 2, head_dim + 4),
        sentinel,
        dtype=torch.float32,
        device=device,
    )
    split_out = split_out_storage[..., :head_dim]
    split_lse_storage = torch.full(
        (batch, num_heads, 4), sentinel, dtype=torch.float32, device=device
    )
    split_lse = split_lse_storage[..., ::2]

    actual = _run_fork(
        q,
        k_cache,
        v_cache,
        boxes,
        softmax_lse=softmax_lse,
        split_out=split_out,
        split_lse=split_lse,
    )
    ref = _run_flash_ref(q, k_cache, v_values, seq_lens, block_table)

    _assert_attention_close(actual, ref)
    assert torch.all(lse_storage[..., 1] == sentinel)
    assert torch.all(split_out_storage[..., head_dim:] == sentinel)
    assert torch.all(split_lse_storage[..., 1::2] == sentinel)


def test_fork_attention_permuted_pages_with_non_power_of_two_page_size() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch, block_size, num_heads, num_kv_heads, head_dim = 2, 48, 4, 1, 64
    logical_blocks = [0, 3, 6, 9, 12, 15, 18, 2, 5, 8]
    seq_len = len(logical_blocks) * block_size

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        20,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    boxes = [(list(range(batch)), logical_blocks, 0, seq_len)]
    block_table = torch.tensor(
        [logical_blocks, logical_blocks], dtype=torch.int32, device=device
    )
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    assert _get_mnw(batch, 4, seq_len, head_dim, block_size) == (16, 64, 1)
    actual = _run_fork(q, k_cache, v_cache, boxes, page_block_size=block_size)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    _assert_attention_close(actual, ref)


@pytest.mark.parametrize(
    ("batch", "kv_len", "expected_mnw"),
    [
        (1, 15, (16, 16, 1)),
        (1, 129, (16, 128, 1)),
        (17, 15, (32, 16, 2)),
        (17, 33, (32, 32, 2)),
        (33, 33, (64, 32, 4)),
    ],
)
def test_fork_attention_mnw_dispatches(
    batch: int, kv_len: int, expected_mnw: tuple[int, int, int]
) -> None:
    torch.manual_seed(batch + kv_len)
    device = torch.device("cuda")
    block_size, num_heads, num_kv_heads, head_dim = 32, 2, 2, 64
    num_blocks = (kv_len + block_size - 1) // block_size

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    boxes = [(list(range(batch)), list(range(num_blocks)), 0, kv_len)]
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).repeat(
        batch, 1
    )
    seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)

    assert _get_mnw(batch, 1, kv_len, head_dim) == expected_mnw
    actual = _run_fork(q, k_cache, v_cache, boxes)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    _assert_attention_close(actual, ref)


@pytest.mark.parametrize(
    ("dtype", "head_dim", "num_splits"),
    [
        (torch.bfloat16, 64, 5),
        (torch.float16, 128, 9),
        (torch.bfloat16, 256, 32),
    ],
)
def test_fork_attention_gather_buckets(
    dtype: torch.dtype, head_dim: int, num_splits: int
) -> None:
    torch.manual_seed(8 + num_splits)
    device = torch.device("cuda")
    batch, block_size, num_heads, num_kv_heads = 1, 32, 2, 1

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        num_splits,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    boxes = [([0], [rank], rank, block_size) for rank in range(num_splits)]
    block_table = torch.arange(num_splits, dtype=torch.int32, device=device).view(1, -1)
    seq_lens = torch.tensor([num_splits * block_size], dtype=torch.int32, device=device)

    actual = _run_fork(q, k_cache, v_cache, boxes)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    _assert_attention_close(actual, ref)


def test_fork_attention_gather_mixed_split_counts() -> None:
    torch.manual_seed(9)
    device = torch.device("cuda")
    batch, block_size, num_heads, num_kv_heads, head_dim = 2, 32, 2, 1, 64
    num_splits = 5

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(
        num_splits + 1,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    boxes = [([0], [rank], rank, block_size) for rank in range(num_splits)]
    boxes.append(([1], [num_splits], 0, block_size))
    block_table = torch.tensor(
        [list(range(num_splits)), [num_splits] + [0] * (num_splits - 1)],
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.tensor(
        [num_splits * block_size, block_size], dtype=torch.int32, device=device
    )

    actual = _run_fork(q, k_cache, v_cache, boxes)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    _assert_attention_close(actual, ref)


def test_fork_attention_cuda_graph_replay() -> None:
    torch.manual_seed(10)
    device = torch.device("cuda")
    batch, block_size, num_heads, num_kv_heads, head_dim = 2, 32, 8, 2, 64
    prefix_blocks = 2
    num_blocks = prefix_blocks + batch

    q = torch.randn(batch, 1, num_heads, head_dim, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    boxes: list[tuple[list[int], list[int], int, int]] = [
        (list(range(batch)), list(range(prefix_blocks)), 0, 64)
    ]
    table_rows = []
    for seq_id in range(batch):
        suffix = [prefix_blocks + seq_id]
        boxes.append(([seq_id], suffix, 1, 32))
        table_rows.append(list(range(prefix_blocks)) + suffix)

    (
        num_split_per_seq,
        query_tables,
        block_tables,
        num_seqs_per_ctas,
        cta_ranks,
        kv_in_ctas,
        mnw,
        max_split,
    ) = _pack_boxes(
        boxes,
        num_seqs=batch,
        hratio=num_heads // num_kv_heads,
        device=device,
        head_dim=head_dim,
    )
    out = torch.empty_like(q)
    softmax_lse = torch.empty(batch, num_heads, 1, dtype=torch.float32, device=device)
    split_out = torch.empty(
        batch,
        num_heads,
        max_split,
        head_dim,
        dtype=torch.float32,
        device=device,
    )
    split_lse = torch.empty(
        batch, num_heads, max_split, dtype=torch.float32, device=device
    )

    def launch() -> None:
        ops.fork_attention(
            out,
            softmax_lse,
            split_out,
            split_lse,
            q,
            k_cache,
            v_cache,
            num_split_per_seq,
            query_tables,
            block_tables,
            num_seqs_per_ctas,
            cta_ranks,
            kv_in_ctas,
            mnw,
            max_split,
            1.0 / math.sqrt(head_dim),
        )

    launch()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    q.normal_()
    k_cache.normal_()
    v_cache.normal_()
    graph.replay()
    torch.accelerator.synchronize()

    block_table = torch.tensor(table_rows, dtype=torch.int32, device=device)
    seq_lens = torch.full((batch,), 96, dtype=torch.int32, device=device)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)
    _assert_attention_close((out, softmax_lse), ref)
