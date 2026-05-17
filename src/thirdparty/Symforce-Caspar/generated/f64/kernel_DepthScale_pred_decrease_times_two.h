#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScalePredDecreaseTimesTwo(
    double* DepthScale_step,
    unsigned int DepthScale_step_num_alloc,
    double* DepthScale_precond_diag,
    unsigned int DepthScale_precond_diag_num_alloc,
    const double* const diag,
    double* DepthScale_njtr,
    unsigned int DepthScale_njtr_num_alloc,
    double* const out_DepthScale_pred_dec,
    size_t problem_size);

}  // namespace caspar