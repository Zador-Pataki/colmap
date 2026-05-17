#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeLogDepthFixedRotationFixedPointResJacFirst(
    double* rotation,
    unsigned int rotation_num_alloc,
    double* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    double* scale,
    unsigned int scale_num_alloc,
    SharedIndex* scale_indices,
    double* log_depth,
    unsigned int log_depth_num_alloc,
    double* loss,
    unsigned int loss_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_rTr,
    double* out_translation_jac,
    unsigned int out_translation_jac_num_alloc,
    double* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    double* const out_translation_precond_diag,
    unsigned int out_translation_precond_diag_num_alloc,
    double* const out_translation_precond_tril,
    unsigned int out_translation_precond_tril_num_alloc,
    double* out_scale_jac,
    unsigned int out_scale_jac_num_alloc,
    double* const out_scale_njtr,
    unsigned int out_scale_njtr_num_alloc,
    double* const out_scale_precond_diag,
    unsigned int out_scale_precond_diag_num_alloc,
    double* const out_scale_precond_tril,
    unsigned int out_scale_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar