#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationStartWContribute(
    float* PinholeTranslation_precond_diag,
    unsigned int PinholeTranslation_precond_diag_num_alloc,
    const float* const diag,
    float* PinholeTranslation_p,
    unsigned int PinholeTranslation_p_num_alloc,
    float* out_PinholeTranslation_w,
    unsigned int out_PinholeTranslation_w_num_alloc,
    size_t problem_size);

}  // namespace caspar