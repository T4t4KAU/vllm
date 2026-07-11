# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections import defaultdict

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


def _get_mnw(
    num_seqs: int,
    hratio: int,
    kv_len: int,
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
    if page_block_size is not None:
        while tile_n > 16 and tile_n // (warps * 32 // 8) > page_block_size:
            tile_n //= 2
    return tile_m, tile_n, warps


def _pack_boxes(
    boxes: list[tuple[list[int], list[int], int, int]],
    *,
    num_seqs: int,
    hratio: int,
    device: torch.device,
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

    grouped: dict[tuple[int, int, int], list[tuple[list[int], list[int], int, int]]]
    grouped = defaultdict(list)
    for box in boxes:
        q_ids, _, _, kv_len = box
        grouped[_get_mnw(len(q_ids), hratio, kv_len, page_block_size)].append(box)

    q_tables: list[torch.Tensor] = []
    block_tables: list[torch.Tensor] = []
    num_seqs_per_ctas: list[torch.Tensor] = []
    cta_ranks: list[torch.Tensor] = []
    kv_in_ctas: list[torch.Tensor] = []
    mnw: list[int] = []

    for tile, group in sorted(grouped.items(), reverse=True):
        max_q = max(len(q_ids) for q_ids, _, _, _ in group)
        max_blocks = max(len(blocks) for _, blocks, _, _ in group)
        q_table = []
        block_table = []
        num_seqs = []
        ranks = []
        kv_lens = []
        for q_ids, blocks, rank, kv_len in group:
            q_table.append(q_ids + [0] * (max_q - len(q_ids)))
            block_table.append(blocks + [0] * (max_blocks - len(blocks)))
            num_seqs.append(len(q_ids))
            ranks.append(rank)
            kv_lens.append(kv_len)

        q_tables.append(torch.tensor(q_table, dtype=torch.int32, device=device))
        block_tables.append(torch.tensor(block_table, dtype=torch.int32, device=device))
        num_seqs_per_ctas.append(
            torch.tensor(num_seqs, dtype=torch.int32, device=device)
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
) -> torch.Tensor:
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
        page_block_size=page_block_size,
    )

    out = torch.empty_like(q)
    softmax_lse = torch.empty((batch, num_heads, 1), dtype=torch.float32, device=device)
    split_out = torch.empty(
        (batch, num_heads, max(max_split, 1), head_dim),
        dtype=torch.float32,
        device=device,
    )
    split_lse = torch.empty(
        (batch, num_heads, max(max_split, 1)), dtype=torch.float32, device=device
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
    return out


def _run_flash_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    batch, _, num_heads, head_dim = q.shape
    out = torch.empty((batch, num_heads, head_dim), dtype=q.dtype, device=q.device)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    flash_attn_varlen_func(
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
    )
    return out.view_as(q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="FORK requires SM80+",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fork_attention_shared_prefix(dtype: torch.dtype, head_dim: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch = 8
    block_size = 32
    num_heads = 16
    num_kv_heads = 4
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
    out = _run_fork(q, k_cache, v_cache, boxes)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="FORK requires SM80+",
)
def test_fork_attention_split_prefix_suffix() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.float16
    batch = 6
    block_size = 32
    num_heads = 16
    num_kv_heads = 4
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

    out = _run_fork(q, k_cache, v_cache, boxes)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="FORK requires SM80+",
)
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

    out = _run_fork(q, k_cache, v_cache, boxes)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
    reason="FORK requires SM80+",
)
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

    out = _run_fork(q, k_cache, v_cache, boxes, page_block_size=block_size)
    ref = _run_flash_ref(q, k_cache, v_cache, seq_lens, block_table)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
