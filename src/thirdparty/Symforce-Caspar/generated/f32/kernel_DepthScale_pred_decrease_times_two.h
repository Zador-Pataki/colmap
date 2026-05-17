#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScalePredDecreaseTimesTwo(
    float* DepthScale_step,
    unsigned int DepthScale_step_num_alloc,
    float* DepthScale_precond_diag,
    unsigned int DepthScale_precond_diag_num_alloc,
    const float* const diag,
    float* DepthScale_njtr,
    unsigned int DepthScale_njtr_num_alloc,
    float* const out_DepthScale_pred_dec,
    size_t problem_size);

}  // namespace caspar