#include "fork_fwd_launch_template.h"

namespace FORK_NAMESPACE {

template void fork_run_mha_fwd_splitkv_dispatch<cute::half_t, 64>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream);
template void fork_run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream);

}  // namespace FORK_NAMESPACE
