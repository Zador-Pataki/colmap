#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeLogDepthFixedRotationScore(double* rotation,
                                       unsigned int rotation_num_alloc,
                                       double* translation,
                                       unsigned int translation_num_alloc,
                                       SharedIndex* translation_indices,
                                       double* scale,
                                       unsigned int scale_num_alloc,
                                       SharedIndex* scale_indices,
                                       double* point,
                                       unsigned int point_num_alloc,
                                       SharedIndex* point_indices,
                                       double* log_depth,
                                       unsigned int log_depth_num_alloc,
                                       double* loss,
                                       unsigned int loss_num_alloc,
                                       double* const out_rTr,
                                       size_t problem_size);

}  // namespace caspar