#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitIntrinsicsRandomWalkFixedPrevFocalFixedPrevPrincipalPointJtjnjtrDirect(
    double* next_focal_njtr,
    unsigned int next_focal_njtr_num_alloc,
    SharedIndex* next_focal_njtr_indices,
    double* next_focal_jac,
    unsigned int next_focal_jac_num_alloc,
    double* next_principal_point_njtr,
    unsigned int next_principal_point_njtr_num_alloc,
    SharedIndex* next_principal_point_njtr_indices,
    double* next_principal_point_jac,
    unsigned int next_principal_point_jac_num_alloc,
    double* const out_next_focal_njtr,
    unsigned int out_next_focal_njtr_num_alloc,
    double* const out_next_principal_point_njtr,
    unsigned int out_next_principal_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar