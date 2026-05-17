#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleStartWContribute(float* DepthScale_precond_diag,
                                unsigned int DepthScale_precond_diag_num_alloc,
                                const float* const diag,
                                float* DepthScale_p,
                                unsigned int DepthScale_p_num_alloc,
                                float* out_DepthScale_w,
                                unsigned int out_DepthScale_w_num_alloc,
                                size_t problem_size);

}  // namespace caspar