#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationUpdateR(
    double* PinholeTranslation_r_k,
    unsigned int PinholeTranslation_r_k_num_alloc,
    double* PinholeTranslation_w,
    unsigned int PinholeTranslation_w_num_alloc,
    const double* const negalpha,
    double* out_PinholeTranslation_r_kp1,
    unsigned int out_PinholeTranslation_r_kp1_num_alloc,
    double* const out_PinholeTranslation_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar