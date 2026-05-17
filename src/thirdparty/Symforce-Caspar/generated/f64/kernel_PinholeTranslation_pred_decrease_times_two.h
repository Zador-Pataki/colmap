#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeTranslationPredDecreaseTimesTwo(
    double* PinholeTranslation_step,
    unsigned int PinholeTranslation_step_num_alloc,
    double* PinholeTranslation_precond_diag,
    unsigned int PinholeTranslation_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeTranslation_njtr,
    unsigned int PinholeTranslation_njtr_num_alloc,
    double* const out_PinholeTranslation_pred_dec,
    size_t problem_size);

}  // namespace caspar