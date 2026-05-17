#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointJtjnjtrDirect(
    float* prev_principal_point_njtr,
    unsigned int prev_principal_point_njtr_num_alloc,
    SharedIndex* prev_principal_point_njtr_indices,
    float* prev_principal_point_jac,
    unsigned int prev_principal_point_jac_num_alloc,
    float* const out_prev_principal_point_njtr,
    unsigned int out_prev_principal_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar