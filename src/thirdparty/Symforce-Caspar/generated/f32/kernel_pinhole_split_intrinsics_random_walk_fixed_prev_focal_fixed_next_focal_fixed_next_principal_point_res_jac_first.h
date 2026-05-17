#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedNextFocalFixedNextPrincipalPointResJacFirst(
    float* prev_principal_point,
    unsigned int prev_principal_point_num_alloc,
    SharedIndex* prev_principal_point_indices,
    float* inv_std,
    unsigned int inv_std_num_alloc,
    float* prev_focal,
    unsigned int prev_focal_num_alloc,
    float* next_focal,
    unsigned int next_focal_num_alloc,
    float* next_principal_point,
    unsigned int next_principal_point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_prev_principal_point_njtr,
    unsigned int out_prev_principal_point_njtr_num_alloc,
    float* const out_prev_principal_point_precond_diag,
    unsigned int out_prev_principal_point_precond_diag_num_alloc,
    float* const out_prev_principal_point_precond_tril,
    unsigned int out_prev_principal_point_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar