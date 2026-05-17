#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsRandomWalkFixedNextFocalJtjnjtrDirect(
    float* prev_focal_njtr,
    unsigned int prev_focal_njtr_num_alloc,
    SharedIndex* prev_focal_njtr_indices,
    float* prev_focal_jac,
    unsigned int prev_focal_jac_num_alloc,
    float* prev_principal_point_njtr,
    unsigned int prev_principal_point_njtr_num_alloc,
    SharedIndex* prev_principal_point_njtr_indices,
    float* prev_principal_point_jac,
    unsigned int prev_principal_point_jac_num_alloc,
    float* next_principal_point_njtr,
    unsigned int next_principal_point_njtr_num_alloc,
    SharedIndex* next_principal_point_njtr_indices,
    float* next_principal_point_jac,
    unsigned int next_principal_point_jac_num_alloc,
    float* const out_prev_focal_njtr,
    unsigned int out_prev_focal_njtr_num_alloc,
    float* const out_prev_principal_point_njtr,
    unsigned int out_prev_principal_point_njtr_num_alloc,
    float* const out_next_principal_point_njtr,
    unsigned int out_next_principal_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar