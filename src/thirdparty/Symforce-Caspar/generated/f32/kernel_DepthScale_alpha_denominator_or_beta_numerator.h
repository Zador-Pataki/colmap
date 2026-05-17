#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleAlphaDenominatorOrBetaNumerator(
    float* DepthScale_p_kp1,
    unsigned int DepthScale_p_kp1_num_alloc,
    float* DepthScale_w,
    unsigned int DepthScale_w_num_alloc,
    float* const DepthScale_out,
    size_t problem_size);

}  // namespace caspar