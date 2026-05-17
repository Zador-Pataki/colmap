#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleUpdateP(float* DepthScale_z,
                       unsigned int DepthScale_z_num_alloc,
                       float* DepthScale_p_k,
                       unsigned int DepthScale_p_k_num_alloc,
                       const float* const beta,
                       float* out_DepthScale_p_kp1,
                       unsigned int out_DepthScale_p_kp1_num_alloc,
                       size_t problem_size);

}  // namespace caspar