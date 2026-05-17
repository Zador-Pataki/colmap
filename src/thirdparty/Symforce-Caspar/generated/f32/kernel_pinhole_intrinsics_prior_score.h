#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeIntrinsicsPriorScore(float* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 float* prior,
                                 unsigned int prior_num_alloc,
                                 float* inv_std,
                                 unsigned int inv_std_num_alloc,
                                 float* const out_rTr,
                                 size_t problem_size);

}  // namespace caspar