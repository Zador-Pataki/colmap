#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleUpdateP(double* DepthScale_z,
                       unsigned int DepthScale_z_num_alloc,
                       double* DepthScale_p_k,
                       unsigned int DepthScale_p_k_num_alloc,
                       const double* const beta,
                       double* out_DepthScale_p_kp1,
                       unsigned int out_DepthScale_p_kp1_num_alloc,
                       size_t problem_size);

}  // namespace caspar