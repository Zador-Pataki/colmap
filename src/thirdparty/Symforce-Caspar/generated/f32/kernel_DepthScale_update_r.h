#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleUpdateR(float* DepthScale_r_k,
                       unsigned int DepthScale_r_k_num_alloc,
                       float* DepthScale_w,
                       unsigned int DepthScale_w_num_alloc,
                       const float* const negalpha,
                       float* out_DepthScale_r_kp1,
                       unsigned int out_DepthScale_r_kp1_num_alloc,
                       float* const out_DepthScale_r_kp1_norm2_tot,
                       size_t problem_size);

}  // namespace caspar