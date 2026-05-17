#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeIntrinsicsRandomWalkJtjnjtrDirect(
    double* prev_calib_njtr,
    unsigned int prev_calib_njtr_num_alloc,
    SharedIndex* prev_calib_njtr_indices,
    double* prev_calib_jac,
    unsigned int prev_calib_jac_num_alloc,
    double* next_calib_njtr,
    unsigned int next_calib_njtr_num_alloc,
    SharedIndex* next_calib_njtr_indices,
    double* next_calib_jac,
    unsigned int next_calib_jac_num_alloc,
    double* const out_prev_calib_njtr,
    unsigned int out_prev_calib_njtr_num_alloc,
    double* const out_next_calib_njtr,
    unsigned int out_next_calib_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar