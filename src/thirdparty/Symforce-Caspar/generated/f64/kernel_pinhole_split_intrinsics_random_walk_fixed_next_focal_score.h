#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsRandomWalkFixedNextFocalScore(
    double* prev_focal,
    unsigned int prev_focal_num_alloc,
    SharedIndex* prev_focal_indices,
    double* prev_principal_point,
    unsigned int prev_principal_point_num_alloc,
    SharedIndex* prev_principal_point_indices,
    double* next_principal_point,
    unsigned int next_principal_point_num_alloc,
    SharedIndex* next_principal_point_indices,
    double* inv_std,
    unsigned int inv_std_num_alloc,
    double* next_focal,
    unsigned int next_focal_num_alloc,
    double* const out_rTr,
    size_t problem_size);

}  // namespace caspar