#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointFixedNextPrincipalPointJtjnjtrDirect(
    float* next_focal_njtr,
    unsigned int next_focal_njtr_num_alloc,
    SharedIndex* next_focal_njtr_indices,
    float* next_focal_jac,
    unsigned int next_focal_jac_num_alloc,
    float* const out_next_focal_njtr,
    unsigned int out_next_focal_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar