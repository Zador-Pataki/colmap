#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleStartWContribute(double* DepthScale_precond_diag,
                                unsigned int DepthScale_precond_diag_num_alloc,
                                const double* const diag,
                                double* DepthScale_p,
                                unsigned int DepthScale_p_num_alloc,
                                double* out_DepthScale_w,
                                unsigned int out_DepthScale_w_num_alloc,
                                size_t problem_size);

}  // namespace caspar