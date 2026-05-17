#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFixedRotationFixedCalibFixedPointScore(
    float* rotation,
    unsigned int rotation_num_alloc,
    float* translation,
    unsigned int translation_num_alloc,
    SharedIndex* translation_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* calib,
    unsigned int calib_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* const out_rTr,
    size_t problem_size);

}  // namespace caspar