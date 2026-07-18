#pragma once

#include "namespace_config.h"
#include "../../torch_utils.h"

#include "static_switch.h"
#include "fork.h"
#include "fork_fwd_kernel.h"
#include <cstdlib>

namespace FORK_NAMESPACE {

// Ampere and newer provide the async copies and tensor-core instructions used
// by the kernel. Keep grid-constant parameters limited to architectures where
// they were already enabled.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  #define ARCH_SUPPORTS_FORK
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  #define KERNEL_PARAM_MODIFIER __grid_constant__
#else
  #define KERNEL_PARAM_MODIFIER
#endif

#define FORK_UNSUPPORTED_ARCH \
  printf("FATAL: ForkAttention requires building for sm80 or newer!");

inline void fork_cuda_check(cudaError_t error, const char* file, int line) {
  STD_TORCH_CHECK(error == cudaSuccess,
                  "ForkAttention CUDA error: ", cudaGetErrorString(error),
                  " at ", file, ":", line);
}

#define FORK_CUDA_CHECK(expr) \
  FORK_NAMESPACE::fork_cuda_check((expr), __FILE__, __LINE__)
#define FORK_CUDA_KERNEL_LAUNCH_CHECK() FORK_CUDA_CHECK(cudaGetLastError())

#define DEFINE_FORK_FORWARD_KERNEL(kernelName, ...)   \
  template <typename Kernel_traits, typename... Args> \
  __global__ void kernelName(                         \
      KERNEL_PARAM_MODIFIER const fork_fwd_params params)

DEFINE_FORK_FORWARD_KERNEL(fork_fwd_splitkv_kernel) {
#if defined(ARCH_SUPPORTS_FORK)
  FORK_NAMESPACE::forward<Kernel_traits>(params);
#else
  FORK_UNSUPPORTED_ARCH
#endif
}

template <typename fwd_kernel_traits>
void launch(fork_fwd_params& params, cudaStream_t& stream) {
  dim3 grid(params.CTAs, params.h_k);
  constexpr size_t smem_size =
      fwd_kernel_traits::SmemSize >= fwd_kernel_traits::kSmemQSize * 2
          ? fwd_kernel_traits::SmemSize
          : fwd_kernel_traits::kSmemQSize * 2;
  auto kernel = &fork_fwd_splitkv_kernel<fwd_kernel_traits>;

  if (smem_size >= 48 * 1024) {
    FORK_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  kernel<<<grid, fwd_kernel_traits::kNWarps * 32, smem_size, stream>>>(params);
  FORK_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename elem_type, int Headdim, int HRatio>
void launch_head_group(std::vector<fork_fwd_params>& params,
                       cudaStream_t stream, int q_head_offset) {
  for (fork_fwd_params& param : params) {
    param.q_head_offset = q_head_offset;
  }

  if (params[0].max_split_per_seq == 1) {
    MNW_SWITCH(params[0].tile_q, params[0].tile_kv, params[0].Warps, kBlockM,
               kBlockN, Warps, [&]() {
                 launch<fwd_kernel_traits<elem_type, elem_type, kBlockM,
                                          kBlockN, Headdim, Warps, HRatio>>(
                     params[0], stream);
               });
    return;
  }

  for (fork_fwd_params& param : params) {
    MNW_SWITCH(param.tile_q, param.tile_kv, param.Warps, kBlockM, kBlockN,
               Warps, [&]() {
                 launch<fwd_kernel_traits<elem_type, float, kBlockM, kBlockN,
                                          Headdim, Warps, HRatio>>(param,
                                                                   stream);
               });
  }
}

template <typename elem_type, int Headdim>
void fork_run_mha_fwd_splitkv_dispatch(std::vector<fork_fwd_params>& params,
                                       cudaStream_t stream) {
  const int q_head_ratio = params[0].q_head_ratio;
  int q_head_offset = 0;
  // Larger groups break vectorized Q/O layouts for some head and warp shapes.
  while (q_head_ratio - q_head_offset >= 4) {
    launch_head_group<elem_type, Headdim, 4>(params, stream, q_head_offset);
    q_head_offset += 4;
  }
  if (q_head_ratio - q_head_offset >= 2) {
    launch_head_group<elem_type, Headdim, 2>(params, stream, q_head_offset);
    q_head_offset += 2;
  }
  if (q_head_ratio - q_head_offset == 1) {
    launch_head_group<elem_type, Headdim, 1>(params, stream, q_head_offset);
  }

  if (params[0].max_split_per_seq == 1) {
    return;
  }

  dim3 grid_gather(params[0].b, params[0].h);
  // The gather store assigns one thread to each output head coordinate.
  // Qwen3.5/3.6 therefore need all 256 threads, not the 128-thread launch
  // used by the previous maximum head dimension.
  constexpr int WARPS = Headdim == 64 ? 2 : Headdim == 128 ? 4 : 8;
  GBLOCKM_SWITCH(params[0].max_split_per_seq, [&]() {
    gather_kernel<elem_type, Headdim, BLOCKM, WARPS>
        <<<grid_gather, WARPS * 32, 0, stream>>>(params[0]);
  });
  FORK_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace FORK_NAMESPACE
