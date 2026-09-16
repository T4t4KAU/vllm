#include "fork_fwd_launch_template.h"

namespace FORK_NAMESPACE {

// Compile head groups separately to keep each PTX module manageable.
extern template void launch_head_group<cute::half_t, 128, 1>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream,
    int q_head_offset);
extern template void launch_head_group<cute::half_t, 128, 2>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream,
    int q_head_offset);
extern template void launch_head_group<cute::half_t, 128, 4>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream,
    int q_head_offset);
extern template void launch_head_group<cutlass::bfloat16_t, 128, 1>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream,
    int q_head_offset);
extern template void launch_head_group<cutlass::bfloat16_t, 128, 2>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream,
    int q_head_offset);
extern template void launch_head_group<cutlass::bfloat16_t, 128, 4>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream,
    int q_head_offset);

template void fork_run_mha_fwd_splitkv_dispatch<cute::half_t, 128>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream);
template void fork_run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream);

}  // namespace FORK_NAMESPACE
