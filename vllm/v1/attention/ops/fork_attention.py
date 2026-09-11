# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged decode attention over shared physical KV segments."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fork_segment(
    Q,
    K,
    V,
    QueryTable,
    BlockTable,
    NumQueries,
    Ranks,
    Lengths,
    SplitOut,
    SplitLSE,
    q_batch: tl.constexpr,
    q_head: tl.constexpr,
    k_page: tl.constexpr,
    k_row: tl.constexpr,
    k_head: tl.constexpr,
    v_page: tl.constexpr,
    v_row: tl.constexpr,
    v_head: tl.constexpr,
    table_width: tl.constexpr,
    query_width: tl.constexpr,
    out_batch: tl.constexpr,
    out_head: tl.constexpr,
    out_split: tl.constexpr,
    lse_batch: tl.constexpr,
    lse_head: tl.constexpr,
    scale: tl.constexpr,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    segment, head = tl.program_id(0), tl.program_id(1)
    count = tl.load(NumQueries + segment)
    length = tl.load(Lengths + segment)
    if count > 0 and length > 0:
        rank = tl.load(Ranks + segment)
        m = tl.arange(0, M)
        d = tl.arange(0, D)
        query = tl.load(
            QueryTable + segment * query_width + m // RATIO,
            mask=m // RATIO < count,
            other=0,
        )
        qh = head * RATIO + m % RATIO
        q = tl.load(
            Q + query[:, None] * q_batch + qh[:, None] * q_head + d[None, :],
            mask=(m // RATIO < count)[:, None],
            other=0,
        )
        maximum = tl.full((M,), -float("inf"), tl.float32)
        denominator = tl.full((M,), 0, tl.float32)
        acc = tl.full((M, D), 0, tl.float32)
        for start in range(tl.cdiv(length, N)):
            n = start * N + tl.arange(0, N)
            page = tl.load(
                BlockTable + segment * table_width + n // PAGE, mask=n < length, other=0
            ).to(tl.int64)
            k = tl.load(
                K
                + page[None, :] * k_page
                + (n % PAGE)[None, :] * k_row
                + head * k_head
                + d[:, None],
                mask=(n < length)[None, :],
                other=0,
            )
            score = tl.dot(q, k) * (scale * 1.4426950408889634)
            score = tl.where((n < length)[None, :], score, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(score, 1))
            correction = tl.exp2(maximum - next_maximum)
            probability = tl.exp2(score - next_maximum[:, None])
            denominator = denominator * correction + tl.sum(probability, 1)
            v = tl.load(
                V
                + page[:, None] * v_page
                + (n % PAGE)[:, None] * v_row
                + head * v_head
                + d[None, :],
                mask=(n < length)[:, None],
                other=0,
            )
            acc = acc * correction[:, None]
            acc = tl.dot(probability.to(v.dtype), v, acc)
            maximum = next_maximum
        result = acc / denominator[:, None]
        lse = maximum + tl.log2(denominator)
        tl.store(
            SplitOut
            + query[:, None] * out_batch
            + qh[:, None] * out_head
            + rank * out_split
            + d[None, :],
            result,
            mask=(m // RATIO < count)[:, None],
        )
        tl.store(
            SplitLSE + query * lse_batch + qh * lse_head + rank,
            lse,
            mask=m // RATIO < count,
        )


@triton.jit
def _fork_reduce(
    SplitOut,
    SplitLSE,
    Counts,
    Out,
    LSE,
    split_batch: tl.constexpr,
    split_head: tl.constexpr,
    split_rank: tl.constexpr,
    lse_batch: tl.constexpr,
    lse_head: tl.constexpr,
    out_batch: tl.constexpr,
    out_head: tl.constexpr,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
):
    query, head = tl.program_id(0), tl.program_id(1)
    count = tl.load(Counts + query)
    if count > 0:
        s = tl.arange(0, S)
        d = tl.arange(0, D)
        lse = tl.load(
            SplitLSE + query * lse_batch + head * lse_head + s,
            mask=s < count,
            other=-float("inf"),
        )
        maximum = tl.max(lse, 0)
        weights = tl.exp2(lse - maximum)
        denominator = tl.sum(weights, 0)
        values = tl.load(
            SplitOut
            + query * split_batch
            + head * split_head
            + s[:, None] * split_rank
            + d[None, :],
            mask=(s < count)[:, None],
            other=0,
        )
        result = tl.sum(values * weights[:, None], 0) / denominator
        tl.store(Out + query * out_batch + head * out_head + d, result)
        tl.store(
            LSE + query * HEADS + head,
            (maximum + tl.log2(denominator)) * 0.6931471805599453,
        )


def fork_attention(
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    split_out: torch.Tensor,
    split_lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    counts: torch.Tensor,
    query_tables: list[torch.Tensor],
    block_tables: list[torch.Tensor],
    num_queries: list[torch.Tensor],
    ranks: list[torch.Tensor],
    lengths: list[torch.Tensor],
    mnw: list[int],
    max_splits: int,
    scale: float,
    *,
    block_n: int = 128,
    num_warps: int = 4,
) -> None:
    """Execute a validated GQA4, head-dimension-128 forest plan."""
    heads, dim = q.shape[-2:]
    ratio = heads // k.shape[2]
    assert ratio == 4 and dim == 128
    for i, query_table in enumerate(query_tables):
        m = mnw[3 * i]
        _fork_segment[(query_table.shape[0], k.shape[2])](
            q,
            k,
            v,
            query_table,
            block_tables[i],
            num_queries[i],
            ranks[i],
            lengths[i],
            split_out,
            split_lse,
            q.stride(0),
            q.stride(-2),
            *k.stride()[:3],
            *v.stride()[:3],
            block_tables[i].shape[1],
            query_table.shape[1],
            *split_out.stride()[:3],
            *split_lse.stride()[:2],
            scale,
            k.shape[1],
            ratio,
            dim,
            m,
            block_n,
            num_warps=num_warps,
        )
    _fork_reduce[(q.shape[0], heads)](
        split_out,
        split_lse,
        counts,
        out,
        softmax_lse,
        *split_out.stride()[:3],
        *split_lse.stride()[:2],
        out.stride(0),
        out.stride(-2),
        heads,
        dim,
        triton.next_power_of_2(max_splits),
        num_warps=4,
    )
