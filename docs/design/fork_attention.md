# ForkAttention

ForkAttention groups decode queries that share physical KV cache blocks so that
a CTA can reuse those blocks across queries. The backend builds a forest from
the current block table, computes attention over its segments, and merges the
partial outputs using their softmax log-sum-exp values.

This port starts from upstream v0.28.0 (`2cf0a6915`) and adapts the ForkAttention
implementation from the Agentrix branch (`c3004078c`).

## Build and enable

Build the CUDA extensions from this checkout using the PyTorch and CUDA versions
specified by v0.28.0:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . --torch-backend=auto
```

Enable the backend with synchronous scheduling and automatic prefix caching:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve <model> \
    --attention-config.backend FORK_ATTN \
    --enable-prefix-caching \
    --no-async-scheduling
```

The Fork kernel handles causal, single-token decode queries with shared physical
prefix blocks. It accepts FP16/BF16, head sizes 64/128/256, and block sizes that
are multiples of 16 on NVIDIA GPUs with compute capability 8.0 or newer. The
SM120 path uses Triton for head size 128 and a query/KV head ratio of four; other
supported shapes use the CUDA operator. In a mixed batch, qualifying decode
groups use Fork while prefills and the other requests use the inherited
FlashAttention implementation. Batches without a qualifying group use Flash.

The default admission policy requires at least eight simultaneous decode queries
sharing at least 16,384 tokens of an identical physical prefix. Both conditions
must hold within one group; total batch size or equal sequence lengths alone do
not qualify. The prefix is checked across complete KV blocks before constructing
the forest. As branches finish, batches below the threshold return to Flash.

`AttentionConfig.fork_min_shared_tokens` and `fork_min_queries` control these
thresholds. The defaults start from the measured long-prefix fanout workloads;
they are not a hardware-independent performance crossover. Operator experiments
can set them to 0 and 2 respectively. Admission uses the actual scheduled queries
and physical KV at each decode step.

## v0.28.0 integration

- **KV cache:** inherit the upstream allocation and write path. Split the logical
  `[num_blocks, num_kv_heads, block_size, 2 * head_size]` tensor into K/V views,
  preserving NHD/HND strides without copying the cache. Validate physical page
  IDs against each group's kernel page count after manager-block splitting.
- **Metadata:** use the renamed `mm_prefix_query_range_tensor` field. The forest
  requires exact CPU sequence lengths, so the backend rejects adaptive
  verification modes that permit CPU/device query-length mismatches and rebuilds
  metadata for draft steps instead of inheriting FlashAttention's draft updater.
  Graph workspaces retain packed static rows and update changed tail lengths or
  rows. Shrinking groups clear stale rows; capacity changes and failed packs
  invalidate the snapshot before reuse. The fixed metadata transfer is retained.
  Each immutable plan shares its graph segment mapping between capacity planning
  and packing; replacing the plan starts a fresh mapping cache.
- **Mixed batches:** select single-token queries by their scheduled query lengths
  and admit each cohort by its complete physical prefix. Plan only the admitted
  queries; remap CPU block-table rows and output token positions independently.
  Contiguous token ranges use tensor views, while interleaved requests gather
  queries and scatter outputs back to their original positions. Flash receives
  metadata for the remaining requests, with its own query offsets and sequence
  lengths. The full batch's cascade and AOT schedule are excluded from that
  subset. The original metadata remains available for layers that use Flash.
  Both paths read the same KV cache after the upstream layer's single KV update.
- **Planning:** reuse CPU block tables, forest snapshots, metadata buffers, and
  split-output workspaces. Prepare the exact plan before graph dispatch. Reuse
  complete shared edges across private-tail block boundaries once each query
  has a complete private edge. Check all previously active physical pages and
  validate newly appended pages, then extend only the last private chunk and
  partial block. Completed private chunks also become reusable. Changes
  to row mapping, physical pages, admission thresholds, or decreasing sequence
  lengths rebuild the forest. Rebuild with larger chunks when growing tails
  exceed the split budget. Admission rejections remain cached across tail growth
  once every query has enough complete blocks to evaluate the required prefix;
  split-limit rejections still reconsider block boundaries. Graph and eager
  execution use the same admission policy. Small batches and short
  contexts exit before scanning their block tables. Each group's completed
  decision, including a rejection, is consumed once when building metadata;
  a rejected group prevents a Fork graph without discarding other groups' work.
- **CUDA Graphs:** the V2 runner captures Fork variants for single-token decode
  with fixed metadata addresses and capacity buckets derived from one shared
  sizing function, within a 256 MiB workspace budget per builder. Graph dispatch
  checks both Fork capacity and upstream query-length constraints. DP ranks
  synchronize the required capacities before dispatch; mixed Fork/Flash ranks
  select a common Flash graph when the batch shapes permit it. Capture sizes
  below `fork_min_queries` are excluded. Configurations whose maximum context
  cannot contain the required complete prefix allocate no Fork graph workspace.
  Mixed batches use the upstream piecewise graphs with dynamic attention
  execution. Pure decode continues to use the captured Fork or Flash variants;
  full Flash graph dispatch never consumes mixed Fork metadata.
- **Operator addressing:** Triton widens physical page IDs before calculating KV
  offsets. The CUDA operator requires contiguous split counts, matching its
  device-side indexing.

The operator is registered as `torch.ops._C.fork_attention`, exposed through
`vllm._custom_ops.fork_attention`, and compiled into `_C_stable_libtorch`.

## Correctness checks

Run on a CUDA host after building this checkout:

```bash
.venv/bin/python -m pytest -q \
    tests/kernels/test_fork_attention.py \
    tests/kernels/test_fork_attention_triton.py \
    tests/v1/attention/test_fork_attention_backend.py \
    tests/v1/worker/test_fork_cudagraph.py \
    tests/v1/worker/test_attn_utils.py \
    tests/v1/worker/test_gpu_block_table.py
```

These cover numerical attention results, hierarchical and partial-block plans,
admission by physical prefix and query group, metadata/workspace reuse, graph
compatibility, and CPU block-table updates.

### Port validation

Validated on RTX 5090 (SM120), Python 3.12, PyTorch 2.13.0+cu130, CUDA 13.0.88,
and CUTLASS 4.4.2. The four Fork CUDA translation units and their stable operator
registration were built as a separate extension; the remaining native extensions
and generated dependencies came from the official v0.28.0 wheel.

- The suites above plus `tests/v1/cudagraph/test_cudagraph_manager.py` passed:
  **168 tests**. Backend numerical cases include both NHD/HND cache layouts,
  FP16/BF16, and head sizes 64/128/256. Regression cases cover Triton KV offsets
  beyond 4 GiB, strided split-count rejection, and long-prefix graph capacities.
  Incremental decode cases cover hierarchical cohorts, completed private chunks,
  cached rejections, physical-page changes, row remapping, and sequence rollback.
  Mixed-batch numerical cases cover contiguous and interleaved requests,
  unrelated decodes, physical CPU row permutations, and untouched output padding.
- A single RTX 5090 Qwen3-VL-8B FP16 comparison checked the default admission
  policy in graph and eager execution against official v0.28.0 FlashAttention.
  Synthetic cases with a 4K prefix and eight branches, or a 16K prefix and two
  branches, used Flash. A 16K prefix with eight branches used Fork; when four
  branches finished, the remaining four switched to Flash. Two distinct 16K
  prefixes with four branches each used Flash, with one initial planning call
  and no rebuilds over the following 14 decode steps in either execution mode.
  All **512 generated token IDs per mode** matched across these five cases.
- A single RTX 5090 Qwen3-VL-8B FP16 comparison used 16K/32K/64K shared prefixes,
  64 branches with 16 private prompt tokens, and 128 generated tokens per branch.
  With the repository's planning and graph dispatch, all **24,576 token IDs**
  matched the official v0.28.0 FlashAttention baseline. All **381 steady decode
  steps** replayed Fork graphs. This was one regression run per workload before
  the admission thresholds were added.
- Incremental planning was compared with the preceding Fork implementation on
  one RTX 5090 using three runs per synthetic workload. Median mean decode
  latency changed from 17.44 to 16.87 ms (16K/8), 32.83 to 31.23 ms (16K/64),
  and 53.90 to 50.80 ms (64K/64). The 64K/64 per-step P95 changed from 120.82
  to 53.13 ms. Two independent four-query groups remained on Flash, at 21.17
  versus 21.13 ms. Each 127-step decode needed only its initial full forest
  build. All 55,296 generated token positions matched official FlashAttention.
- Packed metadata reuse was compared with the incremental planner using the
  same model and three-run protocol. Median mean decode latency changed from
  16.72 to 16.47 ms (16K/8), 30.77 to 28.23 ms (16K/64), and 50.95 to 45.43 ms
  (64K/64). Two independent four-query groups used Flash at 20.98 versus
  21.04 ms. Separate profiling measured 64K/64 metadata packing at 2.95 ms/step,
  down from 6.37 ms/step. All 55,296 generated token positions matched official
  FlashAttention. Repacking checks cover shrinking rows, capacity changes,
  capture resets, and recovery after a partially written failed pack.
  These synthetic measurements are not AgentX dataset scores.
- Mixed prefill/decode was compared with the preceding Fork implementation on
  one RTX 5090 using Qwen3-VL-8B FP16 and three runs per synthetic workload.
  Every fourth decode step included either a 256-token tool continuation or
  an unrelated 2,048-token prefill. Median mean mixed-step latency changed from
  108.22 to 68.99 ms (16K/8, tool), 385.28 to 176.90 ms (16K/64, unrelated
  prefill), and 984.12 to 183.45 ms (64K/64, tool). Two independent four-query
  groups remained on Flash; mean latency over all measured steps changed from
  43.21 to 43.65 ms. All 42,360 generated token positions per implementation
  matched official FlashAttention. Separate eager and profiling runs matched
  another 14,520 and 13,920 positions, respectively. Nsight verified mixed
  Fork/Flash execution, one KV write per layer per step, and Flash-only execution
  for rejected groups. These synthetic measurements are not AgentX scores.
- Ruff, Python 3.12 mypy, C++/CUDA formatting, Markdown formatting, and the
  repository's import, SPDX, and CUDA API checks passed.
