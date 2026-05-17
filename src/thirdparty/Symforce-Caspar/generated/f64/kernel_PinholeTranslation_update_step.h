#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationUpdateStep(
    double* PinholeTranslation_step_k,
    unsigned int PinholeTranslation_step_k_num_alloc,
    double* PinholeTranslation_p_kp1,
    unsigned int PinholeTranslation_p_kp1_num_alloc,
    const double* const alpha,
    double* out_PinholeTranslation_step_kp1,
    unsigned int out_PinholeTranslation_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar