#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsPriorFixedPrincipalPointResJac(
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* prior,
    unsigned int prior_num_alloc,
    double* inv_std,
    unsigned int inv_std_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar