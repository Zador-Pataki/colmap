#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsRandomWalkFixedPrevPrincipalPointResJac(
    double* prev_focal,
    unsigned int prev_focal_num_alloc,
    SharedIndex* prev_focal_indices,
    double* next_focal,
    unsigned int next_focal_num_alloc,
    SharedIndex* next_focal_indices,
    double* next_principal_point,
    unsigned int next_principal_point_num_alloc,
    SharedIndex* next_principal_point_indices,
    double* inv_std,
    unsigned int inv_std_num_alloc,
    double* prev_principal_point,
    unsigned int prev_principal_point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* out_prev_focal_jac,
    unsigned int out_prev_focal_jac_num_alloc,
    double* const out_prev_focal_njtr,
    unsigned int out_prev_focal_njtr_num_alloc,
    double* const out_prev_focal_precond_diag,
    unsigned int out_prev_focal_precond_diag_num_alloc,
    double* const out_prev_focal_precond_tril,
    unsigned int out_prev_focal_precond_tril_num_alloc,
    double* out_next_focal_jac,
    unsigned int out_next_focal_jac_num_alloc,
    double* const out_next_focal_njtr,
    unsigned int out_next_focal_njtr_num_alloc,
    double* const out_next_focal_precond_diag,
    unsigned int out_next_focal_precond_diag_num_alloc,
    double* const out_next_focal_precond_tril,
    unsigned int out_next_focal_precond_tril_num_alloc,
    double* out_next_principal_point_jac,
    unsigned int out_next_principal_point_jac_num_alloc,
    double* const out_next_principal_point_njtr,
    unsigned int out_next_principal_point_njtr_num_alloc,
    double* const out_next_principal_point_precond_diag,
    unsigned int out_next_principal_point_precond_diag_num_alloc,
    double* const out_next_principal_point_precond_tril,
    unsigned int out_next_principal_point_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar