#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitFixedRotationFixedFocalFixedPointJtjnjtrDirect(
    float* translation_njtr,
    unsigned int translation_njtr_num_alloc,
    SharedIndex* translation_njtr_indices,
    float* translation_jac,
    unsigned int translation_jac_num_alloc,
    float* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    float* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    float* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar