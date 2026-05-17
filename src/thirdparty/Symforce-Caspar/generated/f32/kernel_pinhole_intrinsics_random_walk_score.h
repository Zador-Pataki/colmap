#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeIntrinsicsRandomWalkScore(float* prev_calib,
                                      unsigned int prev_calib_num_alloc,
                                      SharedIndex* prev_calib_indices,
                                      float* next_calib,
                                      unsigned int next_calib_num_alloc,
                                      SharedIndex* next_calib_indices,
                                      float* inv_std,
                                      unsigned int inv_std_num_alloc,
                                      float* const out_rTr,
                                      size_t problem_size);

}  // namespace caspar