#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleUpdateMp(double* DepthScale_r_k,
                        unsigned int DepthScale_r_k_num_alloc,
                        double* DepthScale_Mp,
                        unsigned int DepthScale_Mp_num_alloc,
                        const double* const beta,
                        double* out_DepthScale_Mp_kp1,
                        unsigned int out_DepthScale_Mp_kp1_num_alloc,
                        double* out_DepthScale_w,
                        unsigned int out_DepthScale_w_num_alloc,
                        size_t problem_size);

}  // namespace caspar