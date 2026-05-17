#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeIntrinsicsPriorScore(double* calib,
                                 unsigned int calib_num_alloc,
                                 SharedIndex* calib_indices,
                                 double* prior,
                                 unsigned int prior_num_alloc,
                                 double* inv_std,
                                 unsigned int inv_std_num_alloc,
                                 double* const out_rTr,
                                 size_t problem_size);

}  // namespace caspar