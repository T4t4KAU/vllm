#include "fork_fwd_launch_template.h"

namespace FORK_NAMESPACE {

template void fork_run_mha_fwd_splitkv_dispatch<cute::half_t, 256>(
    std::vector<fork_fwd_params>& params, cudaStream_t stream);

}
