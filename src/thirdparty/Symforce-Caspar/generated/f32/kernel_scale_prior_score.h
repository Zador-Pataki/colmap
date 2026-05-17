#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ScalePriorScore(float* scale,
                     unsigned int scale_num_alloc,
                     SharedIndex* scale_indices,
                     float* inv_std,
                     unsigned int inv_std_num_alloc,
                     float* loss,
                     unsigned int loss_num_alloc,
                     float* const out_rTr,
                     size_t problem_size);

}  // namespace caspar