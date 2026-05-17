#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointFixedNextFocalFixedNextPrincipalPointJtjnjtrDirect(
    double* prev_focal_njtr,
    unsigned int prev_focal_njtr_num_alloc,
    SharedIndex* prev_focal_njtr_indices,
    double* prev_focal_jac,
    unsigned int prev_focal_jac_num_alloc,
    double* const out_prev_focal_njtr,
    unsigned int out_prev_focal_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar