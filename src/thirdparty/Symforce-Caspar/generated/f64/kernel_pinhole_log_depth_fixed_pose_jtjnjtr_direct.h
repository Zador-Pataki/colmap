#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeLogDepthFixedPoseJtjnjtrDirect(
    double* scale_njtr,
    unsigned int scale_njtr_num_alloc,
    SharedIndex* scale_njtr_indices,
    double* scale_jac,
    unsigned int scale_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_scale_njtr,
    unsigned int out_scale_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar