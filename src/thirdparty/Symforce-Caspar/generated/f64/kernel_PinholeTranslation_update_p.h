#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationUpdateP(
    double* PinholeTranslation_z,
    unsigned int PinholeTranslation_z_num_alloc,
    double* PinholeTranslation_p_k,
    unsigned int PinholeTranslation_p_k_num_alloc,
    const double* const beta,
    double* out_PinholeTranslation_p_kp1,
    unsigned int out_PinholeTranslation_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar