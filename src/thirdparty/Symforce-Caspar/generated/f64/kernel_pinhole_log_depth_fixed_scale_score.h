#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeLogDepthFixedScaleScore(double* pose,
                                    unsigned int pose_num_alloc,
                                    SharedIndex* pose_indices,
                                    double* point,
                                    unsigned int point_num_alloc,
                                    SharedIndex* point_indices,
                                    double* log_depth,
                                    unsigned int log_depth_num_alloc,
                                    double* loss,
                                    unsigned int loss_num_alloc,
                                    double* scale,
                                    unsigned int scale_num_alloc,
                                    double* const out_rTr,
                                    size_t problem_size);

}  // namespace caspar