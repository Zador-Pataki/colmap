#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ScalePriorResJac(double* scale,
                      unsigned int scale_num_alloc,
                      SharedIndex* scale_indices,
                      double* inv_std,
                      unsigned int inv_std_num_alloc,
                      double* loss,
                      unsigned int loss_num_alloc,
                      double* out_res,
                      unsigned int out_res_num_alloc,
                      double* const out_scale_njtr,
                      unsigned int out_scale_njtr_num_alloc,
                      double* const out_scale_precond_diag,
                      unsigned int out_scale_precond_diag_num_alloc,
                      double* const out_scale_precond_tril,
                      unsigned int out_scale_precond_tril_num_alloc,
                      size_t problem_size);

}  // namespace caspar