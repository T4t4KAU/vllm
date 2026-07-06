#pragma once

#include <cute/arch/copy.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/numeric_types.hpp>

#include <cutlass/numeric_types.h>
#include <type_traits>

namespace fork_sm120 {

template <class Source, class Destination = Source>
struct CpAsyncCacheGlobal {
  using SRegisters = Source[1];
  using DRegisters = Destination[1];

  static_assert(sizeof(Source) == sizeof(Destination));
  static_assert(sizeof(Source) == 16);

  CUTE_HOST_DEVICE static void copy(const Source& source,
                                    Destination& destination) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const Source* global_ptr = &source;
    const uint32_t shared_ptr = cute::cast_smem_ptr_to_uint(&destination);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
                     shared_ptr),
                 "l"(global_ptr), "n"(sizeof(Source)));
#else
    CUTE_INVALID_CONTROL_PATH("ForkAttention cp.async requires SM120");
#endif
  }
};

struct MmaF16Op {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(float& d0, float& d1, float& d2, float& d3,
                                   const uint32_t& a0, const uint32_t& a1,
                                   const uint32_t& a2, const uint32_t& a3,
                                   const uint32_t& b0, const uint32_t& b1,
                                   const float& c0, const float& c1,
                                   const float& c2, const float& c3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c2), "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("ForkAttention FP16 MMA requires SM120");
#endif
  }
};

struct MmaBf16Op {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void fma(float& d0, float& d1, float& d2, float& d3,
                                   const uint32_t& a0, const uint32_t& a1,
                                   const uint32_t& a2, const uint32_t& a3,
                                   const uint32_t& b0, const uint32_t& b1,
                                   const float& c0, const float& c1,
                                   const float& c2, const float& c3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0),
          "f"(c1), "f"(c2), "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("ForkAttention BF16 MMA requires SM120");
#endif
  }
};

struct LdMatrixOp {
  using SRegisters = cute::uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void copy(const cute::uint128_t& source, uint32_t& d0,
                                    uint32_t& d1, uint32_t& d2, uint32_t& d3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const uint32_t shared_ptr = cute::cast_smem_ptr_to_uint(&source);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(shared_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("ForkAttention ldmatrix requires SM120");
#endif
  }
};

struct LdMatrixTransposedOp {
  using SRegisters = cute::uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void copy(const cute::uint128_t& source, uint32_t& d0,
                                    uint32_t& d1, uint32_t& d2, uint32_t& d3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    const uint32_t shared_ptr = cute::cast_smem_ptr_to_uint(&source);
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(shared_ptr));
#else
    CUTE_INVALID_CONTROL_PATH(
        "ForkAttention transposed ldmatrix requires SM120");
#endif
  }
};

template <int N>
CUTE_DEVICE void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

}  // namespace fork_sm120

namespace cute {

template <class Source, class Destination>
struct Copy_Traits<fork_sm120::CpAsyncCacheGlobal<Source, Destination>> {
  using ThrID = Layout<_1>;
  using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<Source>::value>>>;
  using DstLayout = Layout<Shape<_1, Int<sizeof_bits<Destination>::value>>>;
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<fork_sm120::LdMatrixOp> {
  using ThrID = Layout<_32>;
  using SrcLayout = Layout<Shape<_32, _128>, Stride<_128, _1>>;
  using DstLayout =
      Layout<Shape<_32, Shape<_32, _4>>, Stride<_32, Stride<_1, _1024>>>;
  using RefLayout = DstLayout;
};

template <>
struct Copy_Traits<fork_sm120::LdMatrixTransposedOp> {
  using ThrID = Layout<_32>;
  using SrcLayout = Layout<Shape<_32, _128>, Stride<_128, _1>>;
  using DstLayout = Layout<Shape<Shape<_4, _8>, Shape<_16, _2, _4>>,
                           Stride<Stride<_256, _16>, Stride<_1, _128, _1024>>>;
  using RefLayout = DstLayout;
};

template <>
struct MMA_Traits<fork_sm120::MmaF16Op> {
  using ValTypeD = float;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using Shape_MNK = Shape<_16, _8, _16>;
  using ThrID = Layout<_32>;
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
                         Stride<Stride<_32, _1>, Stride<_16, _8, _128>>>;
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                         Stride<Stride<_16, _1>, Stride<_8, _64>>>;
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<_2, _2>>,
                         Stride<Stride<_32, _1>, Stride<_16, _8>>>;
};

template <>
struct MMA_Traits<fork_sm120::MmaBf16Op> : MMA_Traits<fork_sm120::MmaF16Op> {
  using ValTypeA = bfloat16_t;
  using ValTypeB = bfloat16_t;
};

}  // namespace cute

namespace fork_sm120 {

template <typename CopyType>
using CpAsync = CpAsyncCacheGlobal<CopyType>;

using MmaF16 = cute::MMA_Atom<MmaF16Op>;
using MmaBf16 = cute::MMA_Atom<MmaBf16Op>;

template <typename Element>
using Mma = std::conditional_t<std::is_same_v<Element, cutlass::half_t>, MmaF16,
                               MmaBf16>;

template <typename Element>
using LdMatrix = cute::Copy_Atom<LdMatrixOp, Element>;

template <typename Element>
using LdMatrixTransposed = cute::Copy_Atom<LdMatrixTransposedOp, Element>;

}  // namespace fork_sm120
