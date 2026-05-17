#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationUpdateP(
    float* PinholeTranslation_z,
    unsigned int PinholeTranslation_z_num_alloc,
    float* PinholeTranslation_p_k,
    unsigned int PinholeTranslation_p_k_num_alloc,
    const float* const beta,
    float* out_PinholeTranslation_p_kp1,
    unsigned int out_PinholeTranslation_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar