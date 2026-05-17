#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleAlphaDenominatorOrBetaNumerator(
    double* DepthScale_p_kp1,
    unsigned int DepthScale_p_kp1_num_alloc,
    double* DepthScale_w,
    unsigned int DepthScale_w_num_alloc,
    double* const DepthScale_out,
    size_t problem_size);

}  // namespace caspar