/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <stdint.h>

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace FLASH_NAMESPACE {

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) {
    return x > y ? x : y;
  }
};

template <>
struct MaxOp<float> {
  // This is slightly faster
  __device__ __forceinline__ float operator()(float const& x, float const& y) {
    return max(x, y);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) {
    return x + y;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int WARPS, typename T, typename Operator>
static __device__ __forceinline__ T WarpReduce(T x, Operator& op) {
  static_assert(WARPS == 32 || WARPS == 16 || WARPS == 8 || WARPS == 4 ||
                WARPS == 2 || WARPS == 1);
  const uint32_t mask = (1u << WARPS) - 1;
  for (int offset = WARPS / 2; offset > 0; offset >>= 1) {
    x = op(x, __shfl_down_sync(mask, x, offset));
  }
  return x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0,
          typename Tensor1, typename Tensor2, typename Tensor3,
          typename Tensor4, typename TiledMma, typename TiledCopyA,
          typename TiledCopyB, typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB,
                                     Tensor3 const& tCsA, Tensor4 const& tCsB,
                                     TiledMma tiled_mma,
                                     TiledCopyA smem_tiled_copy_A,
                                     TiledCopyB smem_tiled_copy_B,
                                     ThrCopyA smem_thr_copy_A,
                                     ThrCopyB smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));  // M
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));  // N
  if (!A_in_regs) {
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  }
  if (!B_in_regs) {
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  }
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      if (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1),
                   tCrA_copy_view(_, _, i + 1));
      }
      if (!B_in_regs) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1),
                   tCrB_copy_view(_, _, i + 1));
      }
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename TiledMma, typename TiledCopy,
          typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0& acc, Tensor1& tCrA,
                                        Tensor2& tCrB, Tensor3 const& tCsB,
                                        TiledMma tiled_mma,
                                        TiledCopy smem_tiled_copy_B,
                                        ThrCopy smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));  // N
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1),
                 tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2,
// MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                     make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
  static_assert(mma_shape_K == 8 || mma_shape_K == 16);
  if constexpr (mma_shape_K == 8) {
    return acc_layout;
  } else {
    auto l = logical_divide(acc_layout,
                            Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l),
                       get<2, 1>(l));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(
    Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag =
      convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(
          tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

// resolves offset of a slice of a paged kv copy from gmem.
// assumes that the tensor has already been positioned at the correct head.
template <typename Kernel_traits>
__forceinline__ __device__ int64_t resolve_thread_kv_page_slice_offset(
    const int tidx, const int n_block, const int page_block_size,
    const int* block_table, const int64_t page_stride, const int row_stride,
    const int max_rows) {
  constexpr int kGmemThreadsPerRow = Kernel_traits::KVGmemThreadsPerRow;  // 8
  constexpr int KVRowsPerThread = Kernel_traits::KVRowsPerThread;         // 16
  constexpr int kGmemElemsPerLoad = Kernel_traits::KVGmemElemsPerLoad;    // 8
  constexpr int kBlockN = Kernel_traits::kBlockN;

  const int col_offset = tidx % kGmemThreadsPerRow * kGmemElemsPerLoad;
  const int block_row_offset = tidx / kGmemThreadsPerRow * KVRowsPerThread;

  int global_row_offset = n_block * kBlockN + block_row_offset;
  if (max_rows > 0) {
    global_row_offset = min(global_row_offset, max_rows - 1);
  }
  const int page_offset = global_row_offset % page_block_size;
  const int virtual_page_idx = global_row_offset / page_block_size;

  return block_table[virtual_page_idx] * page_stride +
         (page_offset * row_stride + col_offset);
}

template <int HRatio, int GmemThreadPerRow, int RowsPerThread,
          int GmemElemsPerLoad>
__forceinline__ __device__ int resolve_thread_qo_slice_offset(
    const int tidx, const int batch_stride, const int head_stride,
    const int* query_table) {
  const int col_offset = tidx % GmemThreadPerRow * GmemElemsPerLoad;
  const int row_offset = tidx / GmemThreadPerRow * RowsPerThread;

  const int query_idx = row_offset / HRatio;
  return query_table[query_idx] * batch_stride +
         row_offset % HRatio * head_stride + col_offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Layout reshape function. Given a layout with modes ((v1, v2), m, k), returns
// (v1, v2, k), where v2 may be a tuple itself, in the case of swizzled
// smem-backed thread tiles. This ensures that paged and non-paged copies result
// in equivalently shaped, if not necessarily strided, tensors.
template <class Shape, class Stride>
__forceinline__ __device__ auto reshape_thread_tile(Layout<Shape, Stride> l) {
  return make_layout(append(get<0>(l.shape()), get<2>(l.shape())),
                     append(get<0>(l.stride()), get<2>(l.stride())));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Clear_OOB_MN = false, typename TiledCopy,
          typename Engine0, typename Layout0, typename Engine1,
          typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy(
    TiledCopy tiled_copy, Tensor<Engine0, Layout0> const& S,
    Tensor<Engine1, Layout1>& D, Tensor<Engine2, Layout2> const& identity_MN,
    const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K

#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
      }
    } else if (Clear_OOB_MN) {
      cute::clear(D(_, m, _));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true,
          bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2,
          typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(
    TiledCopy tiled_copy, Tensor<Engine0, Layout0> const& S,
    Tensor<Engine1, Layout1>& D, Tensor<Engine2, Layout2> const& identity_MN,
    Tensor<Engine3, Layout3> const& predicate_K, const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K
  // There's no case where !Clear_OOB_K && Clear_OOB_MN
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || predicate_K(k)) {
          cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        } else if (Clear_OOB_K) {
          cute::clear(D(_, m, k));
        }
      }
    } else if (Clear_OOB_MN) {
      cute::clear(D(_, m, _));
    }
  }
}

}  // namespace FLASH_NAMESPACE
