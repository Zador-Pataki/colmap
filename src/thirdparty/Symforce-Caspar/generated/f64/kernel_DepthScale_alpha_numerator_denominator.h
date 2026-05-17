#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleAlphaNumeratorDenominator(
    double* DepthScale_p_kp1,
    unsigned int DepthScale_p_kp1_num_alloc,
    double* DepthScale_r_k,
    unsigned int DepthScale_r_k_num_alloc,
    double* DepthScale_w,
    unsigned int DepthScale_w_num_alloc,
    double* const DepthScale_total_ag,
    double* const DepthScale_total_ac,
    size_t problem_size);

}  // namespace caspar