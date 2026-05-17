#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleUpdateRFirst(double* DepthScale_r_k,
                            unsigned int DepthScale_r_k_num_alloc,
                            double* DepthScale_w,
                            unsigned int DepthScale_w_num_alloc,
                            const double* const negalpha,
                            double* out_DepthScale_r_kp1,
                            unsigned int out_DepthScale_r_kp1_num_alloc,
                            double* const out_DepthScale_r_0_norm2_tot,
                            double* const out_DepthScale_r_kp1_norm2_tot,
                            size_t problem_size);

}  // namespace caspar