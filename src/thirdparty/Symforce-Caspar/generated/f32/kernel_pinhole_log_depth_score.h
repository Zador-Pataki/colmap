#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeLogDepthScore(float* pose,
                          unsigned int pose_num_alloc,
                          SharedIndex* pose_indices,
                          float* scale,
                          unsigned int scale_num_alloc,
                          SharedIndex* scale_indices,
                          float* point,
                          unsigned int point_num_alloc,
                          SharedIndex* point_indices,
                          float* log_depth,
                          unsigned int log_depth_num_alloc,
                          float* loss,
                          unsigned int loss_num_alloc,
                          float* const out_rTr,
                          size_t problem_size);

}  // namespace caspar