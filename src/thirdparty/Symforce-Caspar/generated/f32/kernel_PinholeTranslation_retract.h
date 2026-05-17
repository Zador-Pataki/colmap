#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationRetract(
    float* PinholeTranslation,
    unsigned int PinholeTranslation_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_PinholeTranslation_retracted,
    unsigned int out_PinholeTranslation_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar