#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "../../torch_utils.h"
#include "fork.h"

#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

#include <vector>

namespace vllm {
namespace fork_attention {

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

void check_cuda(const Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be on CUDA");
}

void check_dtype(const Tensor& tensor, ScalarType dtype, const char* name) {
  STD_TORCH_CHECK(tensor.scalar_type() == dtype, name, " has wrong dtype");
}

void check_last_dim_contiguous(const Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.stride(tensor.dim() - 1) == 1, name,
                  " must have contiguous last dimension");
}

void check_shape_1d(const Tensor& tensor, int64_t dim0, const char* name) {
  STD_TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == dim0, name,
                  " has wrong shape");
}

void check_shape_2d(const Tensor& tensor, int64_t dim0, int64_t dim1,
                    const char* name) {
  STD_TORCH_CHECK(
      tensor.dim() == 2 && tensor.size(0) == dim0 && tensor.size(1) == dim1,
      name, " has wrong shape");
}

void check_device_match(const Tensor& tensor, const Tensor& reference,
                        const char* name) {
  STD_TORCH_CHECK(tensor.get_device_index() == reference.get_device_index(),
                  name, " must be on the same CUDA device as query");
}

void set_params_base(FORK_NAMESPACE::base_params& params, const Tensor& q,
                     const Tensor& k, const Tensor& v, Tensor& out,
                     const Tensor& num_split_per_seq, Tensor& softmax_lse,
                     Tensor& split_out, Tensor& split_lse,
                     int64_t max_split_per_seq, double softmax_scale) {
  params = {};

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();

  params.kv_row_stride = k.stride(1);
  params.q_head_stride = q.stride(2);
  params.kv_head_stride = k.stride(2);
  params.q_batch_stride = q.stride(0);
  params.kv_batch_stride = k.stride(0);

  params.o_ptr = out.mutable_data_ptr();
  params.o_head_stride = out.stride(2);
  params.o_batch_stride = out.stride(0);

  params.softmax_lse_ptr = softmax_lse.mutable_data_ptr();
  params.softmax_lse_batch_stride = softmax_lse.stride(0);

  params.b = q.size(0);
  params.h = q.size(2);
  params.h_k = k.size(2);
  params.q_head_ratio = params.h / params.h_k;
  params.q_head_offset = 0;
  params.page_block_size = k.size(1);

  params.scale_softmax = static_cast<float>(softmax_scale);
  params.scale_softmax_log2 = static_cast<float>(softmax_scale * M_LOG2E);

  params.max_split_per_seq = max_split_per_seq;
  params.num_split_per_seq_ptr = num_split_per_seq.data_ptr();

  if (max_split_per_seq == 1) {
    params.oaccum_ptr = out.mutable_data_ptr();
    params.softmax_lseaccum_ptr = softmax_lse.mutable_data_ptr();
    params.oaccum_batch_stride = out.stride(0);
    params.oaccum_head_stride = out.stride(2);
    params.oaccum_rank_stride = 1;
    params.softmax_lseaccum_batch_stride = softmax_lse.stride(0);
    params.softmax_lseaccum_head_stride = softmax_lse.stride(1);
  } else {
    params.oaccum_ptr = split_out.mutable_data_ptr();
    params.softmax_lseaccum_ptr = split_lse.mutable_data_ptr();
    params.oaccum_batch_stride = split_out.stride(0);
    params.oaccum_head_stride = split_out.stride(1);
    params.oaccum_rank_stride = split_out.stride(2);
    params.softmax_lseaccum_batch_stride = split_lse.stride(0);
    params.softmax_lseaccum_head_stride = split_lse.stride(1);
  }
}

void set_params_kernel(FORK_NAMESPACE::fork_fwd_params& params,
                       const Tensor& query_table, const Tensor& block_table,
                       const Tensor& num_seqs_per_cta, const Tensor& cta_rank,
                       const Tensor& kv_in_cta, int64_t tile_q, int64_t tile_kv,
                       int64_t warps) {
  params.query_tables_ptr = query_table.data_ptr();
  params.block_tables_ptr = block_table.data_ptr();
  params.num_seqs_per_CTA_ptr = num_seqs_per_cta.data_ptr();
  params.CTA_rank_ptr = cta_rank.data_ptr();
  params.kv_in_CTA_ptr = kv_in_cta.data_ptr();
  params.block_tables_row_stride = block_table.stride(0);
  params.query_tables_row_stride = query_table.stride(0);
  params.CTAs = query_table.size(0);
  params.tile_q = tile_q;
  params.tile_kv = tile_kv;
  params.Warps = warps;
}

}  // namespace

void fork_attention(Tensor& out, Tensor& softmax_lse, Tensor& split_out,
                    Tensor& split_lse, const Tensor& q, const Tensor& k_cache,
                    const Tensor& v_cache, const Tensor& num_split_per_seq,
                    const std::vector<Tensor>& query_tables,
                    const std::vector<Tensor>& block_tables,
                    const std::vector<Tensor>& num_seqs_per_ctas,
                    const std::vector<Tensor>& cta_ranks,
                    const std::vector<Tensor>& kv_in_ctas,
                    const std::vector<int64_t>& mnw, int64_t max_split_per_seq,
                    double softmax_scale) {
  check_cuda(q, "q");
  check_cuda(k_cache, "k_cache");
  check_cuda(v_cache, "v_cache");
  check_cuda(out, "out");
  check_cuda(softmax_lse, "softmax_lse");
  check_cuda(num_split_per_seq, "num_split_per_seq");
  check_device_match(k_cache, q, "k_cache");
  check_device_match(v_cache, q, "v_cache");
  check_device_match(out, q, "out");
  check_device_match(softmax_lse, q, "softmax_lse");
  check_device_match(num_split_per_seq, q, "num_split_per_seq");

  STD_TORCH_CHECK(q.scalar_type() == ScalarType::Half ||
                      q.scalar_type() == ScalarType::BFloat16,
                  "ForkAttention supports only fp16 and bf16 query");
  check_dtype(k_cache, q.scalar_type(), "k_cache");
  check_dtype(v_cache, q.scalar_type(), "v_cache");
  check_dtype(out, q.scalar_type(), "out");
  check_dtype(softmax_lse, ScalarType::Float, "softmax_lse");
  check_dtype(num_split_per_seq, ScalarType::Int, "num_split_per_seq");

  STD_TORCH_CHECK(q.dim() == 4, "q must have shape [B, 1, H, D]");
  STD_TORCH_CHECK(k_cache.dim() == 4,
                  "k_cache must have shape [num_blocks, block_size, H_kv, D]");
  STD_TORCH_CHECK(v_cache.dim() == 4,
                  "v_cache must have shape [num_blocks, block_size, H_kv, D]");
  STD_TORCH_CHECK(out.dim() == 4, "out must have shape [B, 1, H, D]");
  STD_TORCH_CHECK(softmax_lse.dim() == 3,
                  "softmax_lse must have shape [B, H, 1]");
  STD_TORCH_CHECK(q.size(1) == 1,
                  "ForkAttention currently supports decode q_len == 1");
  STD_TORCH_CHECK(
      q.size(3) == 64 || q.size(3) == 128 || q.size(3) == 256,
      "ForkAttention supports head size 64, 128, or 256");
  STD_TORCH_CHECK(
      k_cache.size(1) % 16 == 0,
      "ForkAttention requires page block size to be divisible by 16");
  STD_TORCH_CHECK(k_cache.size(3) == q.size(3),
                  "k_cache and q head size mismatch");
  STD_TORCH_CHECK(v_cache.size(0) == k_cache.size(0) &&
                      v_cache.size(1) == k_cache.size(1) &&
                      v_cache.size(2) == k_cache.size(2) &&
                      v_cache.size(3) == k_cache.size(3),
                  "k_cache and v_cache shape mismatch");
  STD_TORCH_CHECK(out.size(0) == q.size(0) && out.size(1) == q.size(1) &&
                      out.size(2) == q.size(2) && out.size(3) == q.size(3),
                  "out and q shape mismatch");
  STD_TORCH_CHECK(softmax_lse.size(0) >= q.size(0) &&
                      softmax_lse.size(1) == q.size(2) &&
                      softmax_lse.size(2) == q.size(1),
                  "softmax_lse shape mismatch");
  STD_TORCH_CHECK(q.size(2) % k_cache.size(2) == 0,
                  "query heads must be divisible by kv heads");
  check_last_dim_contiguous(q, "q");
  check_last_dim_contiguous(k_cache, "k_cache");
  check_last_dim_contiguous(v_cache, "v_cache");
  check_last_dim_contiguous(out, "out");
  check_last_dim_contiguous(softmax_lse, "softmax_lse");
  STD_TORCH_CHECK(
      num_split_per_seq.dim() == 1 && num_split_per_seq.size(0) >= q.size(0),
      "num_split_per_seq shape mismatch");

  const size_t num_kernels = query_tables.size();
  STD_TORCH_CHECK(num_kernels > 0,
                  "ForkAttention metadata must contain at least one kernel");
  STD_TORCH_CHECK(
      block_tables.size() == num_kernels &&
          num_seqs_per_ctas.size() == num_kernels &&
          cta_ranks.size() == num_kernels && kv_in_ctas.size() == num_kernels,
      "ForkAttention metadata tensor lists have inconsistent lengths");
  STD_TORCH_CHECK(mnw.size() == num_kernels * 3,
                  "mnw must contain [M, N, W] triples for each kernel");

  if (max_split_per_seq > 1) {
    check_cuda(split_out, "split_out");
    check_cuda(split_lse, "split_lse");
    check_device_match(split_out, q, "split_out");
    check_device_match(split_lse, q, "split_lse");
    check_dtype(split_out, ScalarType::Float, "split_out");
    check_dtype(split_lse, ScalarType::Float, "split_lse");
    STD_TORCH_CHECK(split_out.dim() == 4, "split_out must be 4D");
    STD_TORCH_CHECK(split_lse.dim() == 3, "split_lse must be 3D");
    STD_TORCH_CHECK(split_out.size(0) >= q.size(0) &&
                        split_out.size(1) == q.size(2) &&
                        split_out.size(2) >= max_split_per_seq &&
                        split_out.size(3) == q.size(3),
                    "split_out shape mismatch");
    STD_TORCH_CHECK(split_lse.size(0) >= q.size(0) &&
                        split_lse.size(1) == q.size(2) &&
                        split_lse.size(2) >= max_split_per_seq,
                    "split_lse shape mismatch");
  }

  FORK_NAMESPACE::base_params base_param;
  set_params_base(base_param, q, k_cache, v_cache, out, num_split_per_seq,
                  softmax_lse, split_out, split_lse, max_split_per_seq,
                  softmax_scale);

  std::vector<FORK_NAMESPACE::fork_fwd_params> params;
  params.reserve(num_kernels);
  for (size_t i = 0; i < num_kernels; ++i) {
    const Tensor& query_table = query_tables[i];
    const Tensor& block_table = block_tables[i];
    const Tensor& num_seqs_per_cta = num_seqs_per_ctas[i];
    const Tensor& cta_rank = cta_ranks[i];
    const Tensor& kv_in_cta = kv_in_ctas[i];
    check_cuda(query_table, "query_table");
    check_cuda(block_table, "block_table");
    check_cuda(num_seqs_per_cta, "num_seqs_per_cta");
    check_cuda(cta_rank, "cta_rank");
    check_cuda(kv_in_cta, "kv_in_cta");
    check_device_match(query_table, q, "query_table");
    check_device_match(block_table, q, "block_table");
    check_device_match(num_seqs_per_cta, q, "num_seqs_per_cta");
    check_device_match(cta_rank, q, "cta_rank");
    check_device_match(kv_in_cta, q, "kv_in_cta");
    check_dtype(query_table, ScalarType::Int, "query_table");
    check_dtype(block_table, ScalarType::Int, "block_table");
    check_dtype(num_seqs_per_cta, ScalarType::Int, "num_seqs_per_cta");
    check_dtype(cta_rank, ScalarType::Int, "cta_rank");
    check_dtype(kv_in_cta, ScalarType::Int, "kv_in_cta");
    STD_TORCH_CHECK(query_table.dim() == 2, "query_table must be 2D");
    STD_TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
    const int64_t ctas = query_table.size(0);
    check_shape_2d(query_table, ctas, query_table.size(1), "query_table");
    check_shape_2d(block_table, ctas, block_table.size(1), "block_table");
    check_shape_1d(num_seqs_per_cta, ctas, "num_seqs_per_cta");
    check_shape_1d(cta_rank, ctas, "cta_rank");
    check_shape_1d(kv_in_cta, ctas, "kv_in_cta");
    check_last_dim_contiguous(query_table, "query_table");
    check_last_dim_contiguous(block_table, "block_table");
    check_last_dim_contiguous(num_seqs_per_cta, "num_seqs_per_cta");
    check_last_dim_contiguous(cta_rank, "cta_rank");
    check_last_dim_contiguous(kv_in_cta, "kv_in_cta");

    const int64_t tile_kv = mnw[3 * i + 1];
    const int64_t warps = mnw[3 * i + 2];
    const int64_t kv_row_groups = warps * 4;
    STD_TORCH_CHECK(
        warps > 0 && tile_kv % kv_row_groups == 0 &&
            tile_kv / kv_row_groups <= k_cache.size(1),
        "ForkAttention tile crosses a paged KV cache boundary: tile_kv=",
        tile_kv, ", warps=", warps, ", page_block_size=", k_cache.size(1));

    params.emplace_back(base_param);
    set_params_kernel(params.back(), query_table, block_table, num_seqs_per_cta,
                      cta_rank, kv_in_cta, mnw[3 * i], mnw[3 * i + 1],
                      mnw[3 * i + 2]);
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      q.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(q.get_device_index());
  if (q.scalar_type() == ScalarType::Half) {
    if (q.size(3) == 64) {
      FORK_NAMESPACE::fork_run_mha_fwd_splitkv_dispatch<cute::half_t, 64>(
          params, stream);
    } else if (q.size(3) == 128) {
      FORK_NAMESPACE::fork_run_mha_fwd_splitkv_dispatch<cute::half_t, 128>(
          params, stream);
    } else {
      FORK_NAMESPACE::fork_run_mha_fwd_splitkv_dispatch<cute::half_t, 256>(
          params, stream);
    }
  } else {
    if (q.size(3) == 64) {
      FORK_NAMESPACE::fork_run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t,
                                                        64>(params, stream);
    } else if (q.size(3) == 128) {
      FORK_NAMESPACE::fork_run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t,
                                                        128>(params, stream);
    } else {
      FORK_NAMESPACE::fork_run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t,
                                                        256>(params, stream);
    }
  }
}

}  // namespace fork_attention
}  // namespace vllm
