#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeIntrinsicsRandomWalkResJac(
    float* prev_calib,
    unsigned int prev_calib_num_alloc,
    SharedIndex* prev_calib_indices,
    float* next_calib,
    unsigned int next_calib_num_alloc,
    SharedIndex* next_calib_indices,
    float* inv_std,
    unsigned int inv_std_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_prev_calib_jac,
    unsigned int out_prev_calib_jac_num_alloc,
    float* const out_prev_calib_njtr,
    unsigned int out_prev_calib_njtr_num_alloc,
    float* const out_prev_calib_precond_diag,
    unsigned int out_prev_calib_precond_diag_num_alloc,
    float* const out_prev_calib_precond_tril,
    unsigned int out_prev_calib_precond_tril_num_alloc,
    float* out_next_calib_jac,
    unsigned int out_next_calib_jac_num_alloc,
    float* const out_next_calib_njtr,
    unsigned int out_next_calib_njtr_num_alloc,
    float* const out_next_calib_precond_diag,
    unsigned int out_next_calib_precond_diag_num_alloc,
    float* const out_next_calib_precond_tril,
    unsigned int out_next_calib_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar