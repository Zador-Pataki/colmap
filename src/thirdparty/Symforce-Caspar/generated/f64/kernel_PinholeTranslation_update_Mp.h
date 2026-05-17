#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationUpdateMp(
    double* PinholeTranslation_r_k,
    unsigned int PinholeTranslation_r_k_num_alloc,
    double* PinholeTranslation_Mp,
    unsigned int PinholeTranslation_Mp_num_alloc,
    const double* const beta,
    double* out_PinholeTranslation_Mp_kp1,
    unsigned int out_PinholeTranslation_Mp_kp1_num_alloc,
    double* out_PinholeTranslation_w,
    unsigned int out_PinholeTranslation_w_num_alloc,
    size_t problem_size);

}  // namespace caspar