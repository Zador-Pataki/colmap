#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationUpdateR(
    float* PinholeTranslation_r_k,
    unsigned int PinholeTranslation_r_k_num_alloc,
    float* PinholeTranslation_w,
    unsigned int PinholeTranslation_w_num_alloc,
    const float* const negalpha,
    float* out_PinholeTranslation_r_kp1,
    unsigned int out_PinholeTranslation_r_kp1_num_alloc,
    float* const out_PinholeTranslation_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar