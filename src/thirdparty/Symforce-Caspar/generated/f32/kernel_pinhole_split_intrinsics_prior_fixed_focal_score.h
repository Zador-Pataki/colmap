#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsPriorFixedFocalScore(
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* prior,
    unsigned int prior_num_alloc,
    float* inv_std,
    unsigned int inv_std_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* const out_rTr,
    size_t problem_size);

}  // namespace caspar