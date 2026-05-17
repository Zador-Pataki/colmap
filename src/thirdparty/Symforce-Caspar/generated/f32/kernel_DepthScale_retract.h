#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void DepthScaleRetract(float* DepthScale,
                       unsigned int DepthScale_num_alloc,
                       float* delta,
                       unsigned int delta_num_alloc,
                       float* out_DepthScale_retracted,
                       unsigned int out_DepthScale_retracted_num_alloc,
                       size_t problem_size);

}  // namespace caspar