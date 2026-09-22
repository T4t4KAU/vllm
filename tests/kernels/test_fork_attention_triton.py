# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.fork_attn import (
    _build_fork_plan,
    _ForkCUDAGraphWorkspace,
)
from vllm.v1.attention.ops.fork_attention import fork_attention, fork_flatten_attention

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    reason="Triton ForkAttention requires CUDA SM80+",
)


@pytest.mark.parametrize("heads,kvheads,dim", [(8, 8, 64), (16, 8, 128), (32, 4, 256)])
def test_flatten_preserves_gqa_head_mapping(heads, kvheads, dim):
    torch.manual_seed(11)
    batch, page = 5, 16
    rows = [[0, 1, 2 + i] for i in range(batch)]
    lengths = [33 + i for i in range(batch)]
    q = torch.randn(batch, 1, heads, dim, dtype=torch.float16, device="cuda")
    cache = torch.randn(
        batch + 2, 2, page, kvheads, dim, dtype=q.dtype, device=q.device
    )
    k, v = cache.unbind(1)
    out = torch.empty_like(q)
    plan = _build_fork_plan(
        query_start_locs=list(range(batch + 1)),
        seq_lens=lengths,
        block_rows=rows,
        num_actual_tokens=batch,
        block_size=page,
        head_ratio=heads // kvheads,
        require_shared=True,
    )
    workspace = _ForkCUDAGraphWorkspace(
        num_heads_q=heads,
        num_heads_kv=kvheads,
        head_dim=dim,
        block_size=page,
        max_model_len=64,
        max_queries=batch,
        max_ctas=16,
        max_splits=4,
        device=q.device,
        flatten_chunk_tokens=32,
    )
    meta = workspace.pack(plan, query_capacity=batch, cta_capacity=16, split_capacity=4)
    fork_flatten_attention(
        out,
        meta["fork_softmax_lse"],
        meta["fork_split_out"],
        meta["fork_split_lse"],
        q,
        k,
        v,
        meta["fork_num_split_per_seq"],
        meta["fork_flat_metadata"],
        meta["fork_flat_chunk_tokens"],
        4,
        dim**-0.5,
    )
    for query, (row, length) in enumerate(zip(rows, lengths)):
        keys = (
            k[row]
            .reshape(-1, kvheads, dim)[:length]
            .repeat_interleave(heads // kvheads, dim=1)
            .float()
        )
        values = (
            v[row]
            .reshape(-1, kvheads, dim)[:length]
            .repeat_interleave(heads // kvheads, dim=1)
            .float()
        )
        scores = torch.einsum("hd,thd->ht", q[query, 0].float(), keys) * dim**-0.5
        expected = torch.einsum("ht,thd->hd", scores.softmax(-1), values)
        torch.testing.assert_close(
            out[query, 0].float(), expected, atol=2e-3, rtol=2e-2
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("flatten", [False, True])
@pytest.mark.parametrize(
    "batch,prefix,page,page_base",
    [
        (4, 8192, 16, 0),
        (16, 24576, 16, 0),
        (9, 128, 32, 0),
        (17, 512, 16, 0),
        (2, 0, 16, 0),
        pytest.param(2, 0, 16, 65536, id="kv-offset-above-int32"),
    ],
)
def test_fork_triton_replays_changed_forest(
    dtype, batch, prefix, page, page_base, flatten
):
    torch.manual_seed(7)
    device = torch.device("cuda")
    heads, kvheads, dim = 32, 8, 128
    shared = prefix // page
    # A second branch point per pair plus two private pages per request.
    pairs = (batch + 1) // 2
    total = shared + pairs + batch * 3
    permutation = (torch.randperm(total) + page_base).tolist()
    rows = [
        [permutation[j] for j in range(shared)]
        + [permutation[shared + i // 2]]
        + [permutation[shared + pairs + 3 * i + j] for j in range(3)]
        for i in range(batch)
    ]
    # The high-page case crosses 2**31 elements (4 GiB); initialize active pages only.
    cache = torch.empty(
        page_base + total, 2, page, kvheads, dim, device=device, dtype=dtype
    )
    cache[page_base:].normal_()
    k, v = cache.unbind(1)
    # Include padded query/output strides and inactive graph rows.
    capacity = 1 << batch.bit_length()
    q = torch.randn(capacity, 1, heads + 1, dim, device=device, dtype=dtype)[
        :, :, :heads
    ]
    out = torch.full_like(q, 42.0)
    workspace = _ForkCUDAGraphWorkspace(
        num_heads_q=heads,
        num_heads_kv=kvheads,
        head_dim=dim,
        block_size=page,
        max_model_len=prefix + 4 * page,
        max_queries=capacity,
        max_ctas=256,
        max_splits=32,
        device=device,
        flatten_chunk_tokens=1024 if flatten else None,
    )
    meta = workspace.pack(
        None, query_capacity=capacity, cta_capacity=256, split_capacity=32
    )

    def run():
        if flatten:
            fork_flatten_attention(
                out,
                meta["fork_softmax_lse"],
                meta["fork_split_out"],
                meta["fork_split_lse"],
                q,
                k,
                v,
                meta["fork_num_split_per_seq"],
                meta["fork_flat_metadata"],
                meta["fork_flat_chunk_tokens"],
                32,
                1 / math.sqrt(dim),
            )
            return
        fork_attention(
            out,
            meta["fork_softmax_lse"],
            meta["fork_split_out"],
            meta["fork_split_lse"],
            q,
            k,
            v,
            meta["fork_num_split_per_seq"],
            meta["fork_query_tables"],
            meta["fork_block_tables"],
            meta["fork_num_seqs_per_ctas"],
            meta["fork_cta_ranks"],
            meta["fork_kv_in_ctas"],
            meta["fork_mnw"],
            32,
            1 / math.sqrt(dim),
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for iteration in range(3):
        # Cross full-page boundaries, reorder requests, then shrink the batch.
        active = batch if iteration < 2 else max(1, batch // 2)
        active_rows = (rows if iteration == 0 else rows[::-1])[:active]
        lengths = [
            prefix + page + page - 1 + (i % 3) + iteration for i in range(active)
        ]
        plan = _build_fork_plan(
            query_start_locs=list(range(active + 1)),
            seq_lens=lengths,
            block_rows=active_rows,
            num_actual_tokens=active,
            block_size=page,
            head_ratio=4,
            require_shared=False,
        )
        assert plan is not None
        workspace.pack(
            plan, query_capacity=capacity, cta_capacity=256, split_capacity=32
        )
        out.fill_(42.0)
        q.normal_()
        graph.replay()
        torch.accelerator.synchronize()
        reference, lse = flash_attn_varlen_func(
            q=q[:active, 0],
            k=k,
            v=v,
            cu_seqlens_q=torch.arange(active + 1, device=device, dtype=torch.int32),
            max_seqlen_q=1,
            seqused_k=torch.tensor(lengths, device=device, dtype=torch.int32),
            max_seqlen_k=max(lengths),
            softmax_scale=1 / math.sqrt(dim),
            causal=True,
            block_table=torch.tensor(active_rows, device=device, dtype=torch.int32),
            num_splits=0,
            return_softmax_lse=True,
        )
        torch.testing.assert_close(out[:active, 0], reference, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(
            meta["fork_softmax_lse"][:active, :, 0], lse.T, atol=2e-3, rtol=2e-3
        )
        assert torch.all(out[active:] == 42.0)
