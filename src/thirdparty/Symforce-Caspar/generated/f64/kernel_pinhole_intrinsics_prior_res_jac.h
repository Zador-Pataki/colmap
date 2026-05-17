#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeIntrinsicsPriorResJac(double* calib,
                                  unsigned int calib_num_alloc,
                                  SharedIndex* calib_indices,
                                  double* prior,
                                  unsigned int prior_num_alloc,
                                  double* inv_std,
                                  unsigned int inv_std_num_alloc,
                                  double* out_res,
                                  unsigned int out_res_num_alloc,
                                  double* const out_calib_njtr,
                                  unsigned int out_calib_njtr_num_alloc,
                                  double* const out_calib_precond_diag,
                                  unsigned int out_calib_precond_diag_num_alloc,
                                  double* const out_calib_precond_tril,
                                  unsigned int out_calib_precond_tril_num_alloc,
                                  size_t problem_size);

}  // namespace caspar