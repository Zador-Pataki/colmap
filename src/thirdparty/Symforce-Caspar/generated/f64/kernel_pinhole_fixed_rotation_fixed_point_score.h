#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFixedRotationFixedPointScore(double* rotation,
                                         unsigned int rotation_num_alloc,
                                         double* translation,
                                         unsigned int translation_num_alloc,
                                         SharedIndex* translation_indices,
                                         double* calib,
                                         unsigned int calib_num_alloc,
                                         SharedIndex* calib_indices,
                                         double* pixel,
                                         unsigned int pixel_num_alloc,
                                         double* weight_loss,
                                         unsigned int weight_loss_num_alloc,
                                         double* point,
                                         unsigned int point_num_alloc,
                                         double* const out_rTr,
                                         size_t problem_size);

}  // namespace caspar