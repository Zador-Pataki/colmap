#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ScalePriorResJac(float* scale,
                      unsigned int scale_num_alloc,
                      SharedIndex* scale_indices,
                      float* inv_std,
                      unsigned int inv_std_num_alloc,
                      float* loss,
                      unsigned int loss_num_alloc,
                      float* out_res,
                      unsigned int out_res_num_alloc,
                      float* const out_scale_njtr,
                      unsigned int out_scale_njtr_num_alloc,
                      float* const out_scale_precond_diag,
                      unsigned int out_scale_precond_diag_num_alloc,
                      float* const out_scale_precond_tril,
                      unsigned int out_scale_precond_tril_num_alloc,
                      size_t problem_size);

}  // namespace caspar