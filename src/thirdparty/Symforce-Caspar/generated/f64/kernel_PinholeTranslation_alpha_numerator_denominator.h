#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationAlphaNumeratorDenominator(
    double* PinholeTranslation_p_kp1,
    unsigned int PinholeTranslation_p_kp1_num_alloc,
    double* PinholeTranslation_r_k,
    unsigned int PinholeTranslation_r_k_num_alloc,
    double* PinholeTranslation_w,
    unsigned int PinholeTranslation_w_num_alloc,
    double* const PinholeTranslation_total_ag,
    double* const PinholeTranslation_total_ac,
    size_t problem_size);

}  // namespace caspar