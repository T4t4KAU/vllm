#pragma once

#include "namespace_config.h"
#include "cuda_arch.h"
#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template <typename Element_, typename ElementAccum_, int kBlockM_, int kBlockN_,
          int kHeadDim_, int Warps, int HRatio_>
struct fwd_kernel_traits {
  using Element = Element_;
  using ElementAccum = ElementAccum_;
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kSwizzle = 3;
  static constexpr int kBlockKSmem = 64;
  static constexpr int kNWarps = Warps;
  static constexpr int HRatio = HRatio_;

  using SmemLayoutAtom = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
  using SmemLayoutVtransposed = decltype(composition(
      SmemLayoutKV{},
      make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
  static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
  static constexpr int SmemSize = kSmemQSize + kSmemKVSize;
  static constexpr int KVGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  static constexpr int KVGmemThreadsPerRow =
      kBlockKSmem / KVGmemElemsPerLoad;  // 8

  using GmemLayoutAtom = Layout<
      Shape<Int<kNWarps * 32 / KVGmemThreadsPerRow>, Int<KVGmemThreadsPerRow>>,
      Stride<Int<KVGmemThreadsPerRow>, _1>>; /*(16,8):(8,1)*/
  using Gmem_copy_struct = fork_cuda::CpAsync<cute::uint128_t>;
  // static constexpr int QRowsPerThread = kBlockM / (kNWarps * 32 /
  // QKVGmemThreadsPerRow); // (kBlockM / (128/8))
  static constexpr int QRowsPerThread = HRatio;
  static constexpr int QGmemThreadsPerRow =
      kNWarps * 32 / (kBlockM / QRowsPerThread);
  static constexpr int QGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(Element);
  using GmemLayoutAtomQ =
      Layout<Shape<Int<kBlockM / QRowsPerThread>, Int<QGmemThreadsPerRow>>,
             Stride<Int<QGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyQ = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{}, GmemLayoutAtomQ{},  // thr_layout
      Layout<Shape<Int<QRowsPerThread>, Int<QGmemElemsPerLoad>>,
             Stride<Int<QGmemElemsPerLoad>, _1>>{}));  // val_layout
  static constexpr int KVRowsPerThread =
      kBlockN / (kNWarps * 32 / KVGmemThreadsPerRow);  // (kBlockN / (128/8))
  using GmemTiledCopyKVPaged = decltype(make_tiled_copy(
      Copy_Atom<Gmem_copy_struct, Element>{}, GmemLayoutAtom{},
      Layout<Shape<Int<KVRowsPerThread>, Int<KVGmemThreadsPerRow>>,
             Stride<Int<KVGmemThreadsPerRow>, _1>>{}));
  using MMA_Atom_Arch = fork_cuda::Mma<Element>;
  using TiledMma =
      TiledMMA<MMA_Atom_Arch,
               Layout<Shape<Int<Warps>, _1, _1>>,  // 4x1x1 or 8x1x1 thread
                                                   // group
               Tile<Int<kBlockM>, _16, _16>>;
  using SmemCopyAtom = fork_cuda::LdMatrix<Element>;
  using SmemCopyAtomTransposed = fork_cuda::LdMatrixTransposed<Element>;
  using SmemLayoutAtomO = decltype(composition(
      Swizzle<kSwizzle, 3, 3>{},
      Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemCopyAtomO =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;
  static constexpr int ORowsPerThread = HRatio;
  static constexpr int OGmemThreadsPerRow =
      kNWarps * 32 / (kBlockM / ORowsPerThread);
  static constexpr int OGmemElemsPerLoad =
      sizeof(cute::uint128_t) / sizeof(ElementAccum);
  using GmemLayoutAtomO =
      Layout<Shape<Int<kBlockM / ORowsPerThread>, Int<OGmemThreadsPerRow>>,
             Stride<Int<OGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyO = decltype(make_tiled_copy(
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
      GmemLayoutAtomO{},
      Layout<Shape<Int<ORowsPerThread>, Int<OGmemElemsPerLoad>>,
             Stride<Int<OGmemElemsPerLoad>, _1>>{}));
};
