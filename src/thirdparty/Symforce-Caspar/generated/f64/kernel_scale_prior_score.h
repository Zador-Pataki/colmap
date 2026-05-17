#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ScalePriorScore(double* scale,
                     unsigned int scale_num_alloc,
                     SharedIndex* scale_indices,
                     double* inv_std,
                     unsigned int inv_std_num_alloc,
                     double* loss,
                     unsigned int loss_num_alloc,
                     double* const out_rTr,
                     size_t problem_size);

}  // namespace caspar